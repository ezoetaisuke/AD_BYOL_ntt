import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020

from .calc_mahalanobis import mahalanobis_distance
from .datasets import create_loader
from .utils import (
    compute_roc_pr,
    ensure_dir,
    extract_subclass_from_path,
    plot_confusion,
    plot_hist_all_subclasses,
    plot_roc_pr,
)


def _resolve_checkpoint_path(cfg):
    ckpt_full = cfg.get("filenames", {}).get("checkpoint_best_full_path")
    if ckpt_full:
        return ckpt_full
    return os.path.join(cfg["output_dir"], cfg["filenames"]["checkpoint_best"])


def _resolve_stats_path(cfg):
    stats_path = str(cfg.get("mahala", {}).get("stats_path", "")).strip()
    if stats_path:
        return stats_path
    return os.path.join(cfg["output_dir"], cfg["filenames"].get("mahala_stats_pt", "checkpoints/mahala_stats.pt"))


@torch.no_grad()
def _score_loader(loader, model, mu, precision, device):
    scores, y_true, paths = [], [], []
    for x, y, p in tqdm(loader, desc="[scoring]", leave=False):
        x = x.to(device)
        emb = model(x)
        if emb.dim() > 2:
            emb = emb.flatten(start_dim=1)

        md = mahalanobis_distance(emb, mu, precision, sqrt=True).detach().cpu().numpy()
        scores.extend(np.asarray(md, dtype=np.float64).ravel().tolist())
        y_true.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
        paths.extend(list(p))

    return np.asarray(scores, dtype=np.float64), np.asarray(y_true, dtype=np.int64), list(paths)


def run_eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("Only model.type='byol_a' is supported.")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ====== DataLoaders (AE eval.pyと同じ流れを維持) ======
    train_loader, train_ds = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False)
    val_loader, val_ds = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False)
    test_ok_loader, test_ok_ds = create_loader(cfg["data"]["test_ok_glob"], label=0, cfg=cfg, shuffle=False)
    test_ng_loader, test_ng_ds = create_loader(cfg["data"]["test_ng_glob"], label=1, cfg=cfg, shuffle=False)

    # ====== BYOL-Aモデル作成・重みロード ======
    byol_cfg = cfg.get("byol", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    model = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    # warm-up forward（入力shape整合）
    x0, _, _ = next(iter(train_loader))
    _ = model(x0.to(device))

    ckpt_path = _resolve_checkpoint_path(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # ====== Mahalanobis統計（main_get_mahala_stats.pyで保存したもの）をロード ======
    stats_path = _resolve_stats_path(cfg)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Mahalanobis stats not found: {stats_path}")

    mahala_pack = torch.load(stats_path, map_location="cpu")
    mu = mahala_pack["mean"].to(device=device, dtype=torch.float32)
    precision = mahala_pack["precision"].to(device=device, dtype=torch.float32)

    # ====== 埋め込み→Mahalanobis距離（ファイル単位） ======
    _train_scores, _train_y, _train_paths = _score_loader(train_loader, model, mu, precision, device)
    val_scores, _val_y, _val_paths = _score_loader(val_loader, model, mu, precision, device)
    test_ok_scores, test_ok_y, test_ok_paths = _score_loader(test_ok_loader, model, mu, precision, device)
    test_ng_scores, test_ng_y, test_ng_paths = _score_loader(test_ng_loader, model, mu, precision, device)

    # ====== 良否判定 ======
    thr_cfg = cfg.get("threshold", {})
    method = thr_cfg.get("method", "p99_val_ok")

    y_true = np.concatenate([test_ok_y, test_ng_y], axis=0)
    scores = np.concatenate([test_ok_scores, test_ng_scores], axis=0)

    if method == "p99_val_ok":
        threshold = float(np.percentile(val_scores, 99.0))
    elif method == "youden_test":
        from sklearn.metrics import roc_curve

        fpr, tpr, thr_list = roc_curve(y_true, scores)
        threshold = float(thr_list[np.argmax(tpr - fpr)])
    elif method == "f1max_test":
        thr_candidates = np.unique(scores)
        best_f1, best_thr = -1.0, None
        for t in thr_candidates:
            y_pred_tmp = (scores >= t).astype(np.int64)
            f1 = f1_score(y_true, y_pred_tmp)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        threshold = float(best_thr)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    y_pred = (scores >= threshold).astype(np.int64)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    print(metrics)

    rows = []
    all_paths = test_ok_paths + test_ng_paths
    for pth, y, s, yp in zip(all_paths, y_true.tolist(), scores.tolist(), y_pred.tolist()):
        rows.append({"path": str(pth), "y_true": int(y), "score": float(s), "y_pred": int(yp)})
    score_path = Path(out_dir) / cfg["filenames"].get("scores_csv", "scores.csv")
    ensure_dir(str(score_path))
    pd.DataFrame(rows).to_csv(score_path, index=False)

    metrics_path = Path(out_dir) / cfg["filenames"].get("eval_metrics_csv", "eval_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    hist_path = Path(out_dir) / cfg["filenames"].get("score_hist_all_subclasses_png", "score_hist_all_subclasses.png")
    subclasses = [extract_subclass_from_path(str(p)) for p in all_paths]
    plot_hist_all_subclasses(
        str(hist_path),
        scores,
        subclasses,
        title="All Subclasses (OK + NG): Score Histogram with Color by Subclass",
    )

    cm_path = Path(out_dir) / cfg["filenames"].get("confusion_matrix_png", "confusion_matrix.png")
    plot_confusion(str(cm_path), y_true, y_pred, labels=("OK", "NG"))

    roc_pack, pr_pack = compute_roc_pr(y_true, scores)
    roc_path = Path(out_dir) / cfg["filenames"].get("roc_png", "roc.png")
    pr_path = Path(out_dir) / cfg["filenames"].get("pr_png", "pr.png")
    plot_roc_pr(str(roc_path), str(pr_path), roc_pack, pr_pack)

    skip_log = Path(out_dir) / "skipped_wavs.log"
    with open(skip_log, "a", encoding="utf-8") as f:
        for p, reason in (train_ds.skipped_files + val_ds.skipped_files + test_ok_ds.skipped_files + test_ng_ds.skipped_files):
            f.write(f"{p}\t{reason}\n")

    return str(score_path)
