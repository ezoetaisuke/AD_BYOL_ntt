import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020

from .datasets import create_loader


def _compute_mean_precision(embeddings: torch.Tensor, eps: float = 1e-6):
    mu = embeddings.mean(dim=0)
    xc = embeddings - mu
    cov = (xc.T @ xc) / max(1, embeddings.shape[0] - 1)
    cov = cov + float(eps) * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    precision = torch.linalg.pinv(cov)
    return mu, precision


def _mahalanobis(embeddings: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor):
    xc = embeddings - mean.unsqueeze(0)
    md2 = torch.sum((xc @ precision) * xc, dim=1)
    return torch.sqrt(torch.clamp(md2, min=0.0))


def _collect_embeddings(loader, model, device):
    scores, labels, paths = [], [], []
    with torch.no_grad():
        for x, y, p in tqdm(loader, desc="embedding"):
            x = x.to(device)
            emb = model(x)
            scores.append(emb.cpu())
            labels.extend([int(v) for v in y.numpy().tolist()])
            paths.extend(list(p))
    emb = torch.cat(scores, dim=0)
    return emb, np.asarray(labels, dtype=np.int64), paths


def run_eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("Only model.type='byol_a' is supported.")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False)
    val_loader, _ = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False)
    test_ok_loader, _ = create_loader(cfg["data"]["test_ok_glob"], label=0, cfg=cfg, shuffle=False)
    test_ng_loader, _ = create_loader(cfg["data"]["test_ng_glob"], label=1, cfg=cfg, shuffle=False)

    byol_cfg = cfg.get("model", {}).get("byol_a", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    model = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    ckpt_path = cfg.get("filenames", {}).get("checkpoint_best_full_path")
    if not ckpt_path:
        ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    train_emb, _, _ = _collect_embeddings(train_loader, model, device)
    val_emb, _, _ = _collect_embeddings(val_loader, model, device)
    test_ok_emb, test_ok_y, test_ok_paths = _collect_embeddings(test_ok_loader, model, device)
    test_ng_emb, test_ng_y, test_ng_paths = _collect_embeddings(test_ng_loader, model, device)

    mean, precision = _compute_mean_precision(train_emb.to(device), eps=float(cfg.get("mahala", {}).get("eps", 1e-6)))

    val_scores = _mahalanobis(val_emb.to(device), mean, precision).cpu().numpy()
    test_ok_scores = _mahalanobis(test_ok_emb.to(device), mean, precision).cpu().numpy()
    test_ng_scores = _mahalanobis(test_ng_emb.to(device), mean, precision).cpu().numpy()

    p = float(cfg.get("threshold", {}).get("val_percentile", 99.0))
    threshold = float(np.percentile(val_scores, p))

    y_true = np.concatenate([test_ok_y, test_ng_y], axis=0)
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
    for pth, y, s, yp in zip(test_ok_paths + test_ng_paths, y_true.tolist(), scores.tolist(), y_pred.tolist()):
        rows.append({"path": str(pth), "y_true": int(y), "score": float(s), "y_pred": int(yp)})
    score_path = Path(out_dir) / cfg["filenames"].get("scores_csv", "scores.csv")
    pd.DataFrame(rows).to_csv(score_path, index=False)

    metrics_path = Path(out_dir) / cfg["filenames"].get("eval_metrics_csv", "eval_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    return str(score_path)
