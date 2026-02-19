import copy
import glob
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from byol_a.byol_a.augmentations import MixupBYOLA, RandomResizeCrop
from byol_a.byol_a.models import AudioNTT2020Task6

from .calc_mahalanobis import cov_to_precision, mahalanobis_distance
from .features import FeatureExtractor
from .utils import ensure_dir, set_seed


class RandomSpecAugment(nn.Module):
    """BYOL学習の2-view生成用の軽量Augment（スペクトログラムに適用）."""

    def __init__(
        self,
        noise_std: float,
        dropout_p: float,
        enable_freq_mask: bool = True,
        enable_time_mask: bool = True,
        freq_mask_ratio: float = 0.15,
        time_mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.noise_std = float(noise_std)
        self.dropout = nn.Dropout2d(p=float(dropout_p))
        self.enable_freq_mask = bool(enable_freq_mask)
        self.enable_time_mask = bool(enable_time_mask)
        self.freq_mask_ratio = max(0.0, min(1.0, float(freq_mask_ratio)))
        self.time_mask_ratio = max(0.0, min(1.0, float(time_mask_ratio)))

    @staticmethod
    def _mask_along_axis(x: torch.Tensor, axis: int, ratio: float) -> torch.Tensor:
        if ratio <= 0.0:
            return x

        axis_size = x.size(axis)
        if axis_size <= 1:
            return x

        max_width = max(1, int(round(axis_size * ratio)))
        batch_size = x.size(0)
        for b in range(batch_size):
            width = int(torch.randint(1, max_width + 1, (1,), device=x.device).item())
            start_max = axis_size - width
            start = int(torch.randint(0, start_max + 1, (1,), device=x.device).item())

            if axis == 2:
                x[b, :, start:start + width, :] = 0.0
            elif axis == 3:
                x[b, :, :, start:start + width] = 0.0

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        # 入力shape想定: [B, C, F, T]
        if self.enable_freq_mask:
            x = self._mask_along_axis(x, axis=2, ratio=self.freq_mask_ratio)
        if self.enable_time_mask:
            x = self._mask_along_axis(x, axis=3, ratio=self.time_mask_ratio)

        x = self.dropout(x)
        return x + torch.randn_like(x) * self.noise_std




class BYOLAAugmentation(nn.Module):
    """BYOL-A公式の拡張（Mixup + RandomResizeCrop）をバッチ入力に適用する。"""

    def __init__(
        self,
        enable_mixup: bool = True,
        mixup_ratio: float = 0.4,
        log_mixup_exp: bool = True,
        mixup_memory_size: int = 2048,
        enable_random_resize_crop: bool = True,
        virtual_crop_scale=(1.0, 1.5),
        freq_scale=(0.6, 1.5),
        time_scale=(0.6, 1.5),
    ):
        super().__init__()
        self.enable_mixup = bool(enable_mixup)
        self.enable_random_resize_crop = bool(enable_random_resize_crop)
        self.mixup = MixupBYOLA(
            ratio=float(mixup_ratio),
            n_memory=int(mixup_memory_size),
            log_mixup_exp=bool(log_mixup_exp),
        )
        self.random_resize_crop = RandomResizeCrop(
            virtual_crop_scale=tuple(virtual_crop_scale),
            freq_scale=tuple(freq_scale),
            time_scale=tuple(time_scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        augmented = []
        for i in range(x.size(0)):
            y = x[i]
            if self.enable_mixup:
                y = self.mixup(y)
            if self.enable_random_resize_crop:
                y = self.random_resize_crop(y)
            augmented.append(y)
        return torch.stack(augmented, dim=0)


def build_augmentation_module(aug_cfg: Dict, legacy_cfg: Dict) -> nn.Module:
    method = str(aug_cfg.get("method", "random_specaugment")).lower()
    if method == "byola":
        byola_cfg = aug_cfg.get("byola", {})
        return BYOLAAugmentation(
            enable_mixup=bool(byola_cfg.get("enable_mixup", True)),
            mixup_ratio=float(byola_cfg.get("mixup_ratio", 0.4)),
            log_mixup_exp=bool(byola_cfg.get("log_mixup_exp", True)),
            mixup_memory_size=int(byola_cfg.get("mixup_memory_size", 2048)),
            enable_random_resize_crop=bool(byola_cfg.get("enable_random_resize_crop", True)),
            virtual_crop_scale=tuple(byola_cfg.get("virtual_crop_scale", [1.0, 1.5])),
            freq_scale=tuple(byola_cfg.get("freq_scale", [0.6, 1.5])),
            time_scale=tuple(byola_cfg.get("time_scale", [0.6, 1.5])),
        )

    if method != "random_specaugment":
        raise ValueError(f"Unsupported byola.ssl_train.augmentation.method: {method}")

    spec_cfg = legacy_cfg.get("spec_augment", {})
    return RandomSpecAugment(
        noise_std=float(legacy_cfg.get("augment_noise_std", 0.05)),
        dropout_p=float(legacy_cfg.get("augment_dropout", 0.1)),
        enable_freq_mask=bool(spec_cfg.get("freq_mask_enable", False)),
        enable_time_mask=bool(spec_cfg.get("time_mask_enable", False)),
        freq_mask_ratio=float(spec_cfg.get("freq_mask_ratio", 0.15)),
        time_mask_ratio=float(spec_cfg.get("time_mask_ratio", 0.15)),
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLSequenceWrapper(nn.Module):
    """
    AudioNTT2020Task6（encoder）を BYOL の online/target で包む。

    重要:
    - スコア設計では「時間方向を潰さない」ため、推論側では frame/chunk の系列埋め込みを使う。
    - ただし BYOL 損失自体は clip-level 比較のため、学習時のみ mean pooling を使う。
    """

    def __init__(self, encoder: AudioNTT2020Task6, feature_dim: int, proj_dim: int, proj_hidden_dim: int, ema_decay: float):
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = MLP(feature_dim, proj_hidden_dim, proj_dim)
        self.online_predictor = MLP(proj_dim, proj_hidden_dim, proj_dim)

        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = MLP(feature_dim, proj_hidden_dim, proj_dim)
        self.target_projector.load_state_dict(self.online_projector.state_dict())

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_decay = float(ema_decay)

    @staticmethod
    def _pool_seq(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    @torch.no_grad()
    def update_target(self) -> None:
        for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * op.data
        for op, tp in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * op.data

    def loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.online_projector(self._pool_seq(self.online_encoder(x1)))
        z2 = self.online_projector(self._pool_seq(self.online_encoder(x2)))
        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        with torch.no_grad():
            t1 = self.target_projector(self._pool_seq(self.target_encoder(x1)))
            t2 = self.target_projector(self._pool_seq(self.target_encoder(x2)))

        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        t1 = F.normalize(t1, dim=-1)
        t2 = F.normalize(t2, dim=-1)
        return ((2.0 - 2.0 * (p1 * t2).sum(dim=-1)) + (2.0 - 2.0 * (p2 * t1).sum(dim=-1))).mean()


@dataclass
class PrepResult:
    waveform: Optional[np.ndarray]
    sr: Optional[int]
    skip_reason: Optional[str] = None


def _expand_globs(patterns) -> List[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files: List[str] = []
    for pat in patterns:
        if pat:
            files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No files found for pattern(s): {patterns}")
    return files


def _pad_wave(y: np.ndarray, target_len: int, pad_mode: str) -> np.ndarray:
    pad = target_len - len(y)
    if pad <= 0:
        return y
    if pad_mode == "repeat":
        reps = int(math.ceil(target_len / len(y))) if len(y) > 0 else 1
        return np.tile(y, reps)[:target_len]
    return np.pad(y, (0, pad), mode="constant")


def _load_wave(path: str, cfg: Dict) -> PrepResult:
    """
    波形読み込みと前処理。

    pretrained時のみ、BYOL-A事前学習の入力想定に厳密に合わせるため
    resample/pad/trim を実施する。
    scratch時は「強制整形しない」のが仕様:
      - 分布を人為的に変えないため（過度な0埋め/リサンプルの混入を避ける）
      - 既存収録条件の時系列特性をできるだけ保持するため
    """
    byola_cfg = cfg["byola"]
    prep_cfg = byola_cfg.get("pretrained_input", {})
    mode = byola_cfg["mode"]

    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        return PrepResult(None, None, f"read_error:{exc}")

    y = y.mean(axis=1) if cfg["audio"].get("to_mono", True) else y[:, 0]

    if mode == "pretrained":
        target_sr = int(prep_cfg.get("target_sr", cfg["audio"]["target_sr"]))
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        min_len = int(prep_cfg.get("min_len", 0))
        target_len = int(prep_cfg.get("target_len", 0))
        pad_mode = prep_cfg.get("pad_mode", "zero")

        if min_len > 0 and len(y) < min_len:
            y = _pad_wave(y, min_len, pad_mode)
        if target_len > 0:
            if len(y) < target_len:
                y = _pad_wave(y, target_len, pad_mode)
            elif len(y) > target_len:
                trim_mode = prep_cfg.get("trim_mode", "center")
                if trim_mode == "center":
                    start = (len(y) - target_len) // 2
                else:
                    start = 0
                y = y[start:start + target_len]
    else:
        # scratch は整形しない。log-melに必要な最小長だけ安全確認。
        min_required = int(cfg["feature"]["logmel"]["n_fft"])
        if len(y) < min_required:
            return PrepResult(None, None, f"too_short_for_logmel:{len(y)}<{min_required}")
        # FeatureExtractor は cfg.audio.target_sr 前提で計算するため、
        # scratchでは SR 不一致時はスキップしてログ化する。
        if int(sr) != int(cfg["audio"]["target_sr"]):
            return PrepResult(None, None, f"sr_mismatch_scratch:{sr}!={cfg['audio']['target_sr']}")

    return PrepResult(y.astype(np.float32), int(sr), None)


def _extract_feature(path: str, cfg: Dict, feat_cache: Dict[int, FeatureExtractor]) -> Tuple[Optional[np.ndarray], Optional[str], Optional[int]]:
    prep = _load_wave(path, cfg)
    if prep.waveform is None:
        return None, prep.skip_reason, prep.sr

    if prep.sr not in feat_cache:
        feat_cfg = copy.deepcopy(cfg)
        feat_cfg["audio"] = dict(cfg["audio"])
        feat_cfg["audio"]["target_sr"] = int(prep.sr)
        feat_cache[prep.sr] = FeatureExtractor(feat_cfg)

    try:
        feat = feat_cache[prep.sr](prep.waveform)  # [1,F,T]
    except Exception as exc:
        return None, f"feature_error:{exc}", prep.sr

    if not np.isfinite(feat).all():
        return None, "non_finite_feature", prep.sr
    return feat.astype(np.float32), None, prep.sr


def _chunk_feature(feat: np.ndarray, cfg: Dict) -> List[np.ndarray]:
    """[1,F,T] を chunk_sec/hop_sec で時間分割し、時間情報を保持する。"""
    hop_length = int(cfg["feature"]["logmel"]["hop_length"])
    sr = int(cfg["audio"]["target_sr"])
    time_cfg = cfg["byola"]["time_embedding"]

    chunk_frames = max(1, int(round(float(time_cfg["chunk_sec"]) * sr / hop_length)))
    hop_frames = max(1, int(round(float(time_cfg["hop_sec"]) * sr / hop_length)))

    t_total = feat.shape[-1]
    if t_total < chunk_frames:
        pad = chunk_frames - t_total
        feat = np.pad(feat, ((0, 0), (0, 0), (0, pad)), mode="constant")
        t_total = feat.shape[-1]

    chunks: List[np.ndarray] = []
    start = 0
    while start + chunk_frames <= t_total:
        chunks.append(feat[:, :, start:start + chunk_frames])
        start += hop_frames

    if not chunks:
        chunks = [feat[:, :, :chunk_frames]]
    return chunks


def _build_encoder(cfg: Dict, device: torch.device) -> AudioNTT2020Task6:
    model_cfg = cfg["byola"]["model"]
    n_mels = int(cfg["feature"]["logmel"]["n_mels"])
    d = int(model_cfg.get("feature_dim", 2048))
    encoder = AudioNTT2020Task6(n_mels=n_mels, d=d).to(device)

    if cfg["byola"]["mode"] == "pretrained":
        weight_path = model_cfg["pretrained_weight_path"]
        state = torch.load(weight_path, map_location=device)
        encoder.load_state_dict(state, strict=True)
        print(f"[INFO] Loaded pretrained BYOL-A weights: {weight_path}")
    else:
        print("[INFO] BYOL-A scratch mode: random initialization.")

    return encoder


def _aggregate_time_scores(scores: np.ndarray, agg_cfg: Dict) -> float:
    method = str(agg_cfg.get("method", "topk_mean")).lower()
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan")

    if method == "max":
        return float(np.max(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method in {"mean+std", "mean_std"}:
        std_ratio = float(agg_cfg.get("std_ratio", 1.0))
        return float(np.mean(arr) + std_ratio * np.std(arr))
    if method == "percentile":
        q = float(agg_cfg.get("percentile", 95.0))
        return float(np.percentile(arr, q))
    if method == "topk_mean":
        r = float(agg_cfg.get("topk_ratio", 0.1))
        k = max(1, int(round(arr.size * r)))
        topk = np.partition(arr, -k)[-k:]
        return float(np.mean(topk))

    raise ValueError(f"Unsupported score_aggregate.method: {method}")


def _extract_frame_embeddings(path: str, cfg: Dict, encoder: AudioNTT2020Task6, device: torch.device, feat_cache: Dict[int, FeatureExtractor]) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    feat, reason, _ = _extract_feature(path, cfg, feat_cache)
    if feat is None:
        return None, reason

    chunks = _chunk_feature(feat, cfg)
    batch = torch.from_numpy(np.stack(chunks, axis=0)).to(device)  # [N,1,F,T]

    with torch.no_grad():
        seq = encoder(batch)  # [N, T', D]
        emb = seq.reshape(-1, seq.size(-1)).detach().cpu()

    return emb, None


def _pack_batch(feats: List[np.ndarray], device: torch.device) -> torch.Tensor:
    max_t = max(v.shape[-1] for v in feats)
    packed = []
    for v in feats:
        pad = max_t - v.shape[-1]
        if pad > 0:
            v = np.pad(v, ((0, 0), (0, 0), (0, pad)), mode="constant")
        packed.append(v)
    return torch.from_numpy(np.stack(packed, axis=0)).to(device)


def _iterate_ssl_loss(files: List[str], cfg: Dict, wrapper: BYOLSequenceWrapper, aug: RandomSpecAugment, optimizer: Optional[optim.Optimizer], device: torch.device, feat_cache: Dict[int, FeatureExtractor], desc: str) -> Tuple[float, int, List[Tuple[str, str]]]:
    train_mode = optimizer is not None
    wrapper.train(mode=train_mode)

    skipped: List[Tuple[str, str]] = []
    losses: List[float] = []
    batch_size = int(cfg["byola"]["ssl_train"].get("batch_size", 8))
    batch_feats: List[np.ndarray] = []

    bar = tqdm(files, desc=desc)
    for path in bar:
        feat, reason, _ = _extract_feature(path, cfg, feat_cache)
        if feat is None:
            skipped.append((path, str(reason)))
            continue

        chunks = _chunk_feature(feat, cfg)
        batch_feats.extend(chunks)

        while len(batch_feats) >= batch_size:
            cur = batch_feats[:batch_size]
            batch_feats = batch_feats[batch_size:]
            x = _pack_batch(cur, device)
            x1, x2 = aug(x), aug(x)

            with torch.set_grad_enabled(train_mode):
                loss = wrapper.loss(x1, x2)
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    wrapper.update_target()
            losses.append(float(loss.item()))
            bar.set_postfix(loss=f"{np.mean(losses):.4f}")

    if len(batch_feats) >= 2:
        x = _pack_batch(batch_feats, device)
        x1, x2 = aug(x), aug(x)
        with torch.set_grad_enabled(train_mode):
            loss = wrapper.loss(x1, x2)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                wrapper.update_target()
        losses.append(float(loss.item()))

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    return mean_loss, len(losses), skipped


def _save_loss_artifacts(history: List[Dict], out_run_dir: Path) -> None:
    csv_path = out_run_dir / "loss_history.csv"
    png_path = out_run_dir / "loss_curve.png"
    ensure_dir(str(csv_path))

    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("BYOL loss")
    plt.title("BYOL-A SSL learning curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def _append_errors(errors_path: Path, rows: List[Tuple[str, str, str]]) -> None:
    if not rows:
        return
    ensure_dir(str(errors_path))
    df = pd.DataFrame(rows, columns=["phase", "path", "reason"])
    if errors_path.exists():
        old = pd.read_csv(errors_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(errors_path, index=False)


def _ssl_train_if_needed(cfg: Dict, encoder: AudioNTT2020Task6, device: torch.device, train_files: List[str], val_files: List[str], out_run_dir: Path) -> None:
    train_cfg = cfg["byola"]["ssl_train"]
    mode = cfg["byola"]["mode"]

    do_ssl = (mode == "scratch" and bool(train_cfg.get("enable_in_scratch", True))) or (
        mode == "pretrained" and bool(train_cfg.get("enable_in_pretrained", False))
    )
    if not do_ssl:
        print("[INFO] SSL training skipped by config.")
        return

    wrapper = BYOLSequenceWrapper(
        encoder,
        feature_dim=int(cfg["byola"]["model"].get("feature_dim", 2048)),
        proj_dim=int(train_cfg.get("proj_dim", 256)),
        proj_hidden_dim=int(train_cfg.get("proj_hidden_dim", 1024)),
        ema_decay=float(train_cfg.get("ema_decay", 0.99)),
    ).to(device)

    aug = build_augmentation_module(train_cfg.get("augmentation", {}), train_cfg).to(device)

    optimizer = optim.Adam(
        wrapper.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    sched_cfg = train_cfg.get("schedule", {})
    plateau_cfg = sched_cfg.get("plateau", {})
    use_plateau = bool(plateau_cfg.get("enable", True))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(plateau_cfg.get("factor", 0.5)),
        patience=int(plateau_cfg.get("patience", 5)),
        min_lr=float(plateau_cfg.get("min_lr", 1e-8)),
    ) if use_plateau else None

    early_cfg = sched_cfg.get("early_stopping", {})
    use_early = bool(early_cfg.get("enable", True))
    early_patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 0.0))

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history: List[Dict] = []
    feat_cache: Dict[int, FeatureExtractor] = {}
    error_rows: List[Tuple[str, str, str]] = []

    epochs = int(train_cfg.get("epochs", 10))
    best_path = out_run_dir / "best.pth"
    last_path = out_run_dir / "last.pth"

    for epoch in range(1, epochs + 1):
        train_loss, train_steps, train_skips = _iterate_ssl_loss(
            train_files, cfg, wrapper, aug, optimizer, device, feat_cache, f"train epoch={epoch}"
        )
        val_loss, val_steps, val_skips = _iterate_ssl_loss(
            val_files, cfg, wrapper, aug, None, device, feat_cache, f"val epoch={epoch}"
        )

        cur_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_steps": train_steps,
            "val_steps": val_steps,
            "lr": cur_lr,
        })

        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_steps={train_steps} val_steps={val_steps} lr={cur_lr:.3e}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({"model": wrapper.online_encoder.state_dict(), "epoch": epoch, "cfg": cfg}, best_path)
        else:
            no_improve += 1

        error_rows.extend([("train_ssl", p, r) for p, r in train_skips])
        error_rows.extend([("val_ssl", p, r) for p, r in val_skips])

        if use_early and no_improve >= early_patience:
            print(f"[INFO] Early stopping at epoch={epoch} (best_epoch={best_epoch}, best_val={best_val:.6f})")
            break

    torch.save({"model": wrapper.online_encoder.state_dict(), "epoch": history[-1]["epoch"], "cfg": cfg}, last_path)
    _save_loss_artifacts(history, out_run_dir)
    _append_errors(out_run_dir / "errors.log", error_rows)

    # 学習済みの online encoder を本体へ戻す
    encoder.load_state_dict(wrapper.online_encoder.state_dict(), strict=True)

    print(f"[SUMMARY] SSL done. epochs={len(history)} best_epoch={best_epoch} best_val={best_val:.6f}")
    print(f"[SUMMARY] best={best_path} last={last_path} history={out_run_dir / 'loss_history.csv'}")


def _resolve_run_dir(cfg: Dict) -> Path:
    base_dir = Path(cfg.get("output_root", "outputs"))
    run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    ensure_dir(str(run_dir / "dummy.txt"))
    return run_dir


def run_train_byola(cfg: Dict) -> str:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    run_dir = _resolve_run_dir(cfg)
    train_files = _expand_globs(cfg["data"]["train_ok_glob"])
    val_files = _expand_globs(cfg["data"]["val_ok_glob"])

    encoder = _build_encoder(cfg, device)
    _ssl_train_if_needed(cfg, encoder, device, train_files, val_files, run_dir)
    encoder.eval()

    feat_cache: Dict[int, FeatureExtractor] = {}
    embed_list: List[torch.Tensor] = []
    errors: List[Tuple[str, str, str]] = []

    for path in tqdm(train_files, desc="collect_train_embeddings"):
        emb, reason = _extract_frame_embeddings(path, cfg, encoder, device, feat_cache)
        if emb is None:
            errors.append(("maha_train", path, str(reason)))
            continue
        embed_list.append(emb)

    if not embed_list:
        raise RuntimeError("No valid training embeddings. Check wav paths and preprocessing config.")

    all_emb = torch.cat(embed_list, dim=0).to(device)
    mu = all_emb.mean(dim=0)
    xc = all_emb - mu
    cov = (xc.T @ xc) / max(1, all_emb.size(0) - 1)

    mahala_cfg = cfg["byola"].get("mahalanobis", {})
    if bool(mahala_cfg.get("diag_only", False)):
        cov = torch.diag(torch.diag(cov))
    precision = cov_to_precision(
        cov,
        eps=float(mahala_cfg.get("eps", 1e-6)),
        use_pinv=bool(mahala_cfg.get("use_pinv", True)),
    )

    npz_path = run_dir / "maha_stats.npz"
    np.savez(
        npz_path,
        mu=mu.detach().cpu().numpy(),
        cov=cov.detach().cpu().numpy(),
        precision=precision.detach().cpu().numpy(),
        n_embeddings=int(all_emb.size(0)),
        feature_dim=int(all_emb.size(1)),
        mode=cfg["byola"]["mode"],
    )

    # SSLを実行しない場合でも、互換のために best/last を保存しておく
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    if not best_path.exists():
        torch.save({"model": encoder.state_dict(), "epoch": 0, "cfg": cfg}, best_path)
    if not last_path.exists():
        torch.save({"model": encoder.state_dict(), "epoch": 0, "cfg": cfg}, last_path)

    _append_errors(run_dir / "errors.log", errors)

    print(f"[SUMMARY] run_dir={run_dir}")
    print(f"[SUMMARY] train_files={len(train_files)} used={len(embed_list)} skipped={len(errors)}")
    print(f"[SUMMARY] saved: best.pth last.pth maha_stats.npz")
    return str(run_dir)


def run_eval_byola(cfg: Dict) -> str:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    run_dir = _resolve_run_dir(cfg)
    encoder = _build_encoder(cfg, device)

    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"
    model_path = best_path if best_path.exists() else last_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {best_path} or {last_path}")
    state = torch.load(model_path, map_location=device)
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()

    stats_path = run_dir / "maha_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Mahalanobis stats not found: {stats_path}")
    pack = np.load(stats_path)
    mu = torch.from_numpy(pack["mu"]).to(device=device, dtype=torch.float32)
    precision = torch.from_numpy(pack["precision"]).to(device=device, dtype=torch.float32)

    files = _expand_globs(cfg["data"]["eval_glob"])
    feat_cache: Dict[int, FeatureExtractor] = {}
    rows = []
    errors: List[Tuple[str, str, str]] = []

    for path in tqdm(files, desc="eval"):
        emb, reason = _extract_frame_embeddings(path, cfg, encoder, device, feat_cache)
        if emb is None:
            errors.append(("eval", path, str(reason)))
            continue

        frame_scores = mahalanobis_distance(emb.to(device), mu, precision, sqrt=True).detach().cpu().numpy()
        score = _aggregate_time_scores(frame_scores, cfg["byola"]["score_aggregate"])
        rows.append({"path": path, "score": float(score), "n_frames": int(len(frame_scores))})

    if not rows:
        raise RuntimeError("No valid evaluation samples.")

    score_df = pd.DataFrame(rows)
    threshold = cfg["byola"].get("threshold", {}).get("value")
    if threshold is not None:
        threshold = float(threshold)
        score_df["threshold"] = threshold
        score_df["y_pred"] = (score_df["score"] >= threshold).astype(int)

    out_csv = run_dir / "scores.csv"
    ensure_dir(str(out_csv))
    score_df.to_csv(out_csv, index=False)
    _append_errors(run_dir / "errors.log", errors)

    print(f"[SUMMARY] run_dir={run_dir}")
    print(f"[SUMMARY] eval_files={len(files)} scored={len(rows)} skipped={len(errors)} csv={out_csv}")
    return str(out_csv)
