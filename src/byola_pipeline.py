import copy
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020Task6

from .calc_mahalanobis import cov_to_precision, mahalanobis_distance
from .features import FeatureExtractor
from .utils import ensure_dir, set_seed


class RandomSpecAugment(nn.Module):
    """BYOL学習で使う簡易Augment（軽量版）."""

    def __init__(self, noise_std: float, dropout_p: float):
        super().__init__()
        self.noise_std = float(noise_std)
        self.dropout = nn.Dropout2d(p=float(dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return x + torch.randn_like(x) * self.noise_std


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
    """AudioNTT2020Task6([B,T,D])をBYOL損失で自己教師あり学習するための薄いラッパ."""

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
        # 学習目的では系列をclip表現に集約（score設計では使わない）
        return x.mean(dim=1)

    @torch.no_grad()
    def update_target(self):
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


def _load_wave(path: str, cfg: Dict) -> PrepResult:
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
                y = y[start : start + target_len]
    else:
        # scratch時は「強制整形しない」。log-mel最小条件だけチェックしてskipする。
        min_required = int(cfg["feature"]["logmel"]["n_fft"])
        if len(y) < min_required:
            return PrepResult(None, None, f"too_short_for_logmel:{len(y)}<{min_required}")

    return PrepResult(y.astype(np.float32), int(sr), None)


def _pad_wave(y: np.ndarray, target_len: int, pad_mode: str) -> np.ndarray:
    pad = target_len - len(y)
    if pad <= 0:
        return y
    if pad_mode == "repeat":
        reps = int(np.ceil(target_len / len(y)))
        return np.tile(y, reps)[:target_len]
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right), mode="constant")


def _feature_extractor_for_sr(cfg: Dict, sr: int, cache: Dict[int, FeatureExtractor]) -> FeatureExtractor:
    # 既存features.pyの挙動を活かすため、srだけ差し替えたcfgでFeatureExtractorを再利用する。
    if sr not in cache:
        local_cfg = copy.deepcopy(cfg)
        local_cfg["audio"]["target_sr"] = int(sr)
        cache[sr] = FeatureExtractor(local_cfg)
    return cache[sr]


def _split_chunks(feat_1ft: np.ndarray, chunk_frames: int, hop_frames: int) -> List[np.ndarray]:
    _, f, t = feat_1ft.shape
    if chunk_frames <= 0 or chunk_frames >= t:
        return [feat_1ft]
    chunks: List[np.ndarray] = []
    for st in range(0, t - chunk_frames + 1, max(1, hop_frames)):
        chunks.append(feat_1ft[:, :, st : st + chunk_frames])
    if not chunks:
        chunks.append(feat_1ft)
    return chunks


def _extract_feature(path: str, cfg: Dict, feat_cache: Dict[int, FeatureExtractor]):
    prep = _load_wave(path, cfg)
    if prep.waveform is None:
        return None, prep.skip_reason, None
    feat = _feature_extractor_for_sr(cfg, prep.sr, feat_cache)(prep.waveform)
    return feat, None, prep.sr


def _extract_frame_embeddings(path: str, cfg: Dict, model: AudioNTT2020Task6, device: torch.device, feat_cache: Dict[int, FeatureExtractor]):
    feat, reason, sr = _extract_feature(path, cfg, feat_cache)
    if feat is None:
        return None, reason

    hop = int(cfg["feature"]["logmel"]["hop_length"])
    seconds_per_frame = hop / float(sr)
    time_cfg = cfg["byola"].get("time_embedding", {})
    chunk_sec = float(time_cfg.get("chunk_sec", 0.0))
    hop_sec = float(time_cfg.get("hop_sec", 0.0))
    chunk_frames = int(round(chunk_sec / seconds_per_frame)) if chunk_sec > 0 else 0
    hop_frames = int(round(hop_sec / seconds_per_frame)) if hop_sec > 0 else chunk_frames

    chunks = _split_chunks(feat, chunk_frames, hop_frames)

    seq_list: List[torch.Tensor] = []
    with torch.no_grad():
        for ch in chunks:
            x = torch.from_numpy(ch).unsqueeze(0).to(device)  # [1,1,F,T]
            seq = model(x)[0]  # [T',D]
            seq_list.append(seq.cpu())

    if not seq_list:
        return None, "empty_embedding"
    return torch.cat(seq_list, dim=0), None


def _aggregate_time_scores(frame_scores: np.ndarray, cfg: Dict) -> float:
    agg = cfg["byola"].get("score_aggregate", {})
    method = agg.get("method", "max")
    if method == "max":
        return float(np.max(frame_scores))
    if method == "mean":
        return float(np.mean(frame_scores))
    if method == "topk_mean":
        ratio = float(agg.get("topk_ratio", 0.1))
        k = max(1, int(np.ceil(len(frame_scores) * ratio)))
        return float(np.mean(np.sort(frame_scores)[-k:]))
    if method == "percentile":
        p = float(agg.get("percentile", 95.0))
        return float(np.percentile(frame_scores, p))
    raise ValueError(f"Unsupported score aggregation: {method}")


def _build_encoder(cfg: Dict, device: torch.device) -> AudioNTT2020Task6:
    byola_cfg = cfg["byola"]
    model_cfg = byola_cfg["model"]
    enc = AudioNTT2020Task6(n_mels=int(cfg["feature"]["logmel"]["n_mels"]), d=int(model_cfg.get("feature_dim", 2048))).to(device)
    if byola_cfg["mode"] == "pretrained":
        weight_path = model_cfg.get("pretrained_weight_path", "")
        if not weight_path:
            d = int(model_cfg.get("feature_dim", 2048))
            weight_path = f"byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d{d}.pth"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"pretrained weight not found: {weight_path}")
        enc.load_weight(weight_path, device=device, key_check=True)
    return enc


def _ssl_train_if_needed(cfg: Dict, encoder: AudioNTT2020Task6, device: torch.device, train_files: List[str]):
    byola_cfg = cfg["byola"]
    train_cfg = byola_cfg.get("ssl_train", {})
    if byola_cfg["mode"] == "pretrained" and not bool(train_cfg.get("enable_in_pretrained", False)):
        return
    if byola_cfg["mode"] == "scratch" and not bool(train_cfg.get("enable_in_scratch", True)):
        return

    wrapper = BYOLSequenceWrapper(
        encoder=encoder,
        feature_dim=int(byola_cfg["model"].get("feature_dim", 2048)),
        proj_dim=int(train_cfg.get("proj_dim", 256)),
        proj_hidden_dim=int(train_cfg.get("proj_hidden_dim", 1024)),
        ema_decay=float(train_cfg.get("ema_decay", 0.99)),
    ).to(device)
    aug = RandomSpecAugment(float(train_cfg.get("augment_noise_std", 0.05)), float(train_cfg.get("augment_dropout", 0.1))).to(device)
    optimizer = optim.Adam(wrapper.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-5)))

    feat_cache: Dict[int, FeatureExtractor] = {}
    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 10))

    def _pack_batch(feats: List[np.ndarray]) -> torch.Tensor:
        max_t = max(v.shape[-1] for v in feats)
        packed = []
        for v in feats:
            pad = max_t - v.shape[-1]
            if pad > 0:
                v = np.pad(v, ((0, 0), (0, 0), (0, pad)), mode="constant")
            packed.append(v)
        x = torch.from_numpy(np.stack(packed, axis=0)).to(device)  # [B,1,F,T]
        return x

    for epoch in range(1, epochs + 1):
        wrapper.train()
        epoch_loss = 0.0
        n_step = 0
        bar = tqdm(train_files, desc=f"ssl_train epoch={epoch}")
        batch_feats: List[np.ndarray] = []
        for path in bar:
            feat, reason, _ = _extract_feature(path, cfg, feat_cache)
            if feat is None:
                continue
            batch_feats.append(feat)
            if len(batch_feats) >= batch_size:
                x = _pack_batch(batch_feats)
                x1, x2 = aug(x), aug(x)
                optimizer.zero_grad(set_to_none=True)
                loss = wrapper.loss(x1, x2)
                loss.backward()
                optimizer.step()
                wrapper.update_target()
                epoch_loss += float(loss.item())
                n_step += 1
                bar.set_postfix(loss=f"{epoch_loss / max(1,n_step):.4f}")
                batch_feats = []

        if len(batch_feats) >= 2:
            x = _pack_batch(batch_feats)
            x1, x2 = aug(x), aug(x)
            optimizer.zero_grad(set_to_none=True)
            loss = wrapper.loss(x1, x2)
            loss.backward()
            optimizer.step()
            wrapper.update_target()
            epoch_loss += float(loss.item())
            n_step += 1

        if n_step > 0:
            print(f"[ssl] epoch={epoch} loss={epoch_loss / n_step:.6f}")

def run_train_byola(cfg: Dict) -> str:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_files = _expand_globs(cfg["data"]["train_ok_glob"])
    encoder = _build_encoder(cfg, device)
    _ssl_train_if_needed(cfg, encoder, device, train_files)
    encoder.eval()

    feat_cache: Dict[int, FeatureExtractor] = {}
    embed_list: List[torch.Tensor] = []
    skipped: List[Tuple[str, str]] = []
    for path in tqdm(train_files, desc="collect_train_embeddings"):
        emb, reason = _extract_frame_embeddings(path, cfg, encoder, device, feat_cache)
        if emb is None:
            skipped.append((path, str(reason)))
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
    precision = cov_to_precision(cov, eps=float(mahala_cfg.get("eps", 1e-6)), use_pinv=bool(mahala_cfg.get("use_pinv", True)))

    out_dir = Path(cfg["output_dir"])
    stats_path = out_dir / cfg["byola"]["save"]["stats_path"]
    model_path = out_dir / cfg["byola"]["save"]["encoder_path"]
    skip_log_path = out_dir / cfg["byola"]["save"].get("skip_log_path", "skip_train.csv")

    ensure_dir(str(stats_path))
    torch.save({
        "mean": mu.detach().cpu(),
        "precision": precision.detach().cpu(),
        "mode": cfg["byola"]["mode"],
        "n_embeddings": int(all_emb.size(0)),
        "feature_dim": int(all_emb.size(1)),
        "cfg": cfg,
    }, stats_path)
    torch.save({"model": encoder.state_dict(), "cfg": cfg}, model_path)

    pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(skip_log_path, index=False)
    print(f"[SUMMARY] train_files={len(train_files)} used={len(embed_list)} skipped={len(skipped)}")
    print(f"[SUMMARY] stats_saved={stats_path} model_saved={model_path}")
    return str(stats_path)


def run_eval_byola(cfg: Dict) -> str:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    encoder = _build_encoder(cfg, device)
    model_path = Path(cfg["output_dir"]) / cfg["byola"]["save"]["encoder_path"]
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        encoder.load_state_dict(state["model"])
    encoder.eval()

    stats_path = Path(cfg["output_dir"]) / cfg["byola"]["save"]["stats_path"]
    pack = torch.load(stats_path, map_location=device)
    mu = pack["mean"].to(device=device, dtype=torch.float32)
    precision = pack["precision"].to(device=device, dtype=torch.float32)

    files = _expand_globs(cfg["data"]["eval_glob"])
    feat_cache: Dict[int, FeatureExtractor] = {}
    rows = []
    skipped: List[Tuple[str, str]] = []

    for path in tqdm(files, desc="eval"):
        emb, reason = _extract_frame_embeddings(path, cfg, encoder, device, feat_cache)
        if emb is None:
            skipped.append((path, str(reason)))
            continue
        frame_scores = mahalanobis_distance(emb.to(device), mu, precision, sqrt=True).detach().cpu().numpy()
        score = _aggregate_time_scores(frame_scores, cfg)
        rows.append({"path": path, "score": float(score), "n_frames": int(len(frame_scores))})

    if not rows:
        raise RuntimeError("No valid evaluation samples.")

    score_df = pd.DataFrame(rows)
    thr_cfg = cfg["byola"].get("threshold", {})
    threshold = thr_cfg.get("value")
    if threshold is not None:
        threshold = float(threshold)
        score_df["threshold"] = threshold
        score_df["y_pred"] = (score_df["score"] >= threshold).astype(int)

    out_dir = Path(cfg["output_dir"])
    out_csv = out_dir / cfg["byola"]["save"]["eval_csv"]
    skip_log_path = out_dir / cfg["byola"]["save"].get("skip_log_eval_path", "skip_eval.csv")
    ensure_dir(str(out_csv))
    score_df.to_csv(out_csv, index=False)
    pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(skip_log_path, index=False)

    print(f"[SUMMARY] eval_files={len(files)} scored={len(rows)} skipped={len(skipped)} csv={out_csv}")
    return str(out_csv)
