import os

import torch
import yaml
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020

from src.calc_mahalanobis import cov_to_precision
from src.datasets import create_loader
from src.utils import ensure_dir


@torch.no_grad()
def estimate_mahala_pack(
    model: torch.nn.Module,
    loaders: list,
    device: torch.device,
    eps: float = 1.0e-6,
    use_pinv: bool = True,
) -> dict:
    """Estimate Mahalanobis statistics from BYOL-A embeddings."""
    model.eval()

    sum_vec = None
    sum_outer = None
    n_total = 0

    for loader in loaders:
        for (x, _, _) in tqdm(loader, desc="[estimate mahala stats]", leave=False):
            x = x.to(device)
            emb = model(x)
            if emb.dim() > 2:
                emb = emb.flatten(start_dim=1)

            v = emb.detach().to(device=device, dtype=torch.float64)

            if sum_vec is None:
                d = v.size(1)
                sum_vec = torch.zeros(d, device=device, dtype=torch.float64)
                sum_outer = torch.zeros(d, d, device=device, dtype=torch.float64)

            sum_vec += v.sum(dim=0)
            sum_outer += v.transpose(0, 1) @ v
            n_total += int(v.size(0))

    if n_total <= 1:
        raise RuntimeError(f"Not enough embeddings to estimate covariance (n={n_total}).")

    mu = (sum_vec / float(n_total)).to(dtype=torch.float32)
    cov = (sum_outer - float(n_total) * (mu[:, None].double() @ mu[None, :].double())) / float(n_total - 1)
    cov = cov.to(dtype=torch.float32)

    precision = cov_to_precision(cov, eps=eps, use_pinv=use_pinv)

    return {
        "mean": mu.detach().cpu(),
        "cov": cov.detach().cpu(),
        "precision": precision.detach().cpu(),
        "n": int(n_total),
        "feature_dim": int(mu.numel()),
        "eps": float(eps),
        "use_pinv": bool(use_pinv),
        "feature_type": "byol_embedding",
    }


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


def get_mahala_stats(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("main_get_mahala_stats.py now supports model.type='byol_a' only.")

    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)

    byol_cfg = cfg.get("byol", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    model = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    x0, _, _ = next(iter(train_loader))
    with torch.no_grad():
        _ = model(x0.to(device))

    ckpt_path = _resolve_checkpoint_path(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    if not cfg.get("mahala", {}).get("enabled", False):
        raise RuntimeError("cfg['mahala']['enabled'] is False. Enable it to estimate statistics.")

    raw_globs = cfg["mahala"]["data"]["train_ok_glob"] + cfg["mahala"]["data"]["val_ok_glob"]
    ok_globs = [g for g in raw_globs if g and str(g).strip() != ""]
    if not ok_globs:
        raise RuntimeError("[mahala] enabled=True, but data paths are missing.")

    loaders_for_calc = []
    for g in ok_globs:
        ld, _ = create_loader(g, label=0, cfg=cfg, shuffle=False, drop_last=False)
        loaders_for_calc.append(ld)

    mahala_pack = estimate_mahala_pack(
        model=model,
        loaders=loaders_for_calc,
        device=device,
        eps=float(cfg["mahala"]["eps"]),
        use_pinv=bool(cfg["mahala"]["use_pinv"]),
    )

    stats_path = _resolve_stats_path(cfg)
    ensure_dir(stats_path)
    torch.save(mahala_pack, stats_path)
    return stats_path


if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mahala_stats_ckpt = get_mahala_stats(cfg)
    print("Statistical data for Mahalanobis distance calculation is stored in the file below.")
    print(mahala_stats_ckpt)
