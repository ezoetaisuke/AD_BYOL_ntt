import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020

from .calc_mahalanobis import cov_to_precision, mahalanobis_distance
from .datasets import AudioDataset
from .features import FeatureExtractor
from .utils import (
    compute_roc_pr,
    ensure_dir,
    extract_subclass_from_path,
    plot_confusion,
    plot_hist_all_subclasses,
    plot_roc_pr,
)


def _load_wave(path, cfg):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1) if cfg["audio"].get("to_mono", True) else y[:, 0]

    mode = str(cfg.get("byol", {}).get("mode", "pretrained")).lower()
    prep = cfg.get("byol", {}).get("pretrained_input", {})
    target_sr = int(cfg["audio"]["target_sr"])

    if mode == "pretrained" and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if mode == "scratch" and sr != target_sr and cfg["audio"].get("resample_if_mismatch", False):
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    min_len = prep.get("min_len", None) if mode == "pretrained" else None
    if min_len is not None and len(y) < int(min_len):
        y = np.pad(y, (0, int(min_len) - len(y)), mode="constant")

    target_len = prep.get("target_len", None) if mode == "pretrained" else None
    trim_mode = str(prep.get("trim_mode", "center")).lower()
    if target_len is not None:
        target_len = int(target_len)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        elif len(y) > target_len:
            start = 0 if trim_mode == "left" else (len(y) - target_len) // 2
            y = y[start:start + target_len]

    return y


def _chunk_feature(feat: np.ndarray, cfg):
    byol_cfg = cfg.get("byol", {})
    emb_cfg = byol_cfg.get("time_embedding", {})
    sr = int(cfg["audio"]["target_sr"])
    hop = int(cfg["feature"]["logmel"]["hop_length"])

    chunk_sec = float(emb_cfg.get("chunk_sec", 0.5))
    hop_sec = float(emb_cfg.get("hop_sec", 0.25))
    chunk_frames = max(1, int(round(chunk_sec * sr / hop)))
    hop_frames = max(1, int(round(hop_sec * sr / hop)))

    t = feat.shape[-1]
    if t < chunk_frames:
        feat = np.pad(feat, ((0, 0), (0, 0), (0, chunk_frames - t)), mode="constant")
        t = feat.shape[-1]

    out = []
    start = 0
    while start + chunk_frames <= t:
        out.append(feat[:, :, start:start + chunk_frames])
        start += hop_frames
    if not out:
        out = [feat[:, :, :chunk_frames]]
    return out


def _aggregate(scores, cfg):
    agg = cfg.get("byol", {}).get("score_aggregate", {})
    method = str(agg.get("method", "topk_mean")).lower()
    arr = np.asarray(scores, dtype=np.float64)
    if method == "max":
        return float(np.max(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "percentile":
        return float(np.percentile(arr, float(agg.get("percentile", 95.0))))
    if method == "topk_mean":
        ratio = float(agg.get("topk_ratio", 0.1))
        k = max(1, int(round(arr.size * ratio)))
        return float(np.mean(np.partition(arr, -k)[-k:]))
    if method in {"mean+std", "mean_std"}:
        return float(np.mean(arr) + float(agg.get("std_ratio", 1.0)) * np.std(arr))
    raise ValueError(f"unsupported aggregate method: {method}")


def _extract_chunk_embeddings(files, cfg, model, device):
    feat_extractor = FeatureExtractor(cfg)
    all_embs = []
    per_file_embs = []
    valid_files = []
    skipped = []

    for path in tqdm(files, desc="embedding"):
        try:
            y = _load_wave(path, cfg)
            feat = feat_extractor(y)
            chunks = _chunk_feature(feat, cfg)
            x = torch.from_numpy(np.stack(chunks, axis=0)).to(device)
            with torch.no_grad():
                emb = model(x).detach().cpu()
            all_embs.append(emb)
            per_file_embs.append(emb)
            valid_files.append(path)
        except Exception as exc:
            skipped.append((path, str(exc)))

    if not all_embs:
        raise RuntimeError("No valid wavs to evaluate.")

    return torch.cat(all_embs, dim=0), per_file_embs, valid_files, skipped


def run_eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("Only model.type='byol_a' is supported.")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_ds = AudioDataset(cfg["data"]["train_ok_glob"], label=0, cfg=cfg)
    val_ds = AudioDataset(cfg["data"]["val_ok_glob"], label=0, cfg=cfg)
    test_ok_ds = AudioDataset(cfg["data"]["test_ok_glob"], label=0, cfg=cfg)
    test_ng_ds = AudioDataset(cfg["data"]["test_ng_glob"], label=1, cfg=cfg)

    byol_cfg = cfg.get("byol", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    model = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    ckpt_path = cfg.get("filenames", {}).get("checkpoint_best_full_path")
    if not ckpt_path:
        ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    train_emb, _, _, skipped_train = _extract_chunk_embeddings(train_ds.files, cfg, model, device)
    val_emb, val_emb_by_file, val_paths, skipped_val = _extract_chunk_embeddings(val_ds.files, cfg, model, device)
    test_ok_emb, test_ok_emb_by_file, test_ok_paths, skipped_ok = _extract_chunk_embeddings(test_ok_ds.files, cfg, model, device)
    test_ng_emb, test_ng_emb_by_file, test_ng_paths, skipped_ng = _extract_chunk_embeddings(test_ng_ds.files, cfg, model, device)

    # Mahalanobis計算ロジックは既存関数を流用
    mu = train_emb.mean(dim=0).to(device)
    xc = train_emb.to(device) - mu.unsqueeze(0)
    cov = (xc.T @ xc) / max(1, train_emb.shape[0] - 1)
    precision = cov_to_precision(
        cov,
        eps=float(cfg.get("mahala", {}).get("eps", 1e-6)),
        use_pinv=bool(cfg.get("mahala", {}).get("use_pinv", True)),
    ).to(device)

    def file_scores(emb_list):
        out = []
        for emb in emb_list:
            frame_scores = mahalanobis_distance(emb.to(device), mu, precision, sqrt=True).detach().cpu().numpy()
            out.append(_aggregate(frame_scores, cfg))
        return np.asarray(out, dtype=np.float64)

    val_scores = file_scores(val_emb_by_file)
    test_ok_scores = file_scores(test_ok_emb_by_file)
    test_ng_scores = file_scores(test_ng_emb_by_file)

    thr_cfg = cfg.get("threshold", {})
    p = float(thr_cfg.get("val_percentile", 99.0))
    threshold = float(np.percentile(val_scores, p))

    y_true = np.concatenate([
        np.zeros_like(test_ok_scores, dtype=np.int64),
        np.ones_like(test_ng_scores, dtype=np.int64),
    ])
    scores = np.concatenate([test_ok_scores, test_ng_scores], axis=0)
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
        for p, reason in (skipped_train + skipped_val + skipped_ok + skipped_ng):
            f.write(f"{p}\t{reason}\n")

    return str(score_path)
