import copy
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from byol_a.byol_a.augmentations import MixupBYOLA, RandomResizeCrop
from byol_a.byol_a.models import AudioNTT2020

from .datasets import create_loader
from .utils import ensure_dir, plot_learning_curve, save_metrics_csv, set_seed


class RandomSpecAugment(nn.Module):
    """Simple stochastic augment for log-mel spectrogram batches."""

    def __init__(
        self,
        noise_std: float = 0.05,
        dropout_p: float = 0.1,
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

    def forward(self, x):
        x = x.clone()
        # 入力shape想定: [B, C, F, T]
        if self.enable_freq_mask:
            x = self._mask_along_axis(x, axis=2, ratio=self.freq_mask_ratio)
        if self.enable_time_mask:
            x = self._mask_along_axis(x, axis=3, ratio=self.time_mask_ratio)

        x = self.dropout(x)
        noise = torch.randn_like(x) * self.noise_std
        return x + noise


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


def build_augmentation_module(aug_cfg: dict, legacy_cfg: dict) -> nn.Module:
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
        raise ValueError(f"Unsupported byol.augmentation.method: {method}")

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

    def forward(self, x):
        return self.net(x)


class BYOLWrapper(nn.Module):
    def __init__(self, encoder, feature_dim, proj_dim=256, proj_hidden_dim=1024, ema_decay=0.99):
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

    @torch.no_grad()
    def _update_target(self):
        for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * op.data
        for op, tp in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * op.data

    def loss(self, x1, x2):
        z1 = self.online_projector(self.online_encoder(x1))
        z2 = self.online_projector(self.online_encoder(x2))
        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        t1 = F.normalize(t1, dim=-1)
        t2 = F.normalize(t2, dim=-1)
        loss = (2.0 - 2.0 * (p1 * t2).sum(dim=-1)) + (2.0 - 2.0 * (p2 * t1).sum(dim=-1))
        return loss.mean()


def _resolve_pretrained_path(cfg):
    byol_cfg = cfg.get("byol", {})
    path = byol_cfg.get("pretrained_weight_path")
    if path:
        return path

    feat_d = int(byol_cfg.get("feature_d", 2048))
    return f"byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d{feat_d}.pth"


def run_train(cfg):
    """
    既存AE学習ループの入出力は維持しつつ、モデル本体のみBYOL-Aへ置換。
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("Only model.type='byol_a' is supported.")

    train_loader, train_ds = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=True, drop_last=True)
    val_loader, val_ds = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)

    byol_cfg = cfg.get("byol", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    encoder = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    if str(byol_cfg.get("mode", "pretrained")).lower() == "pretrained":
        weight_path = _resolve_pretrained_path(cfg)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"BYOL-A pretrained weight not found: {weight_path}")
        encoder.load_weight(weight_path, device=device, key_check=True)

    model = BYOLWrapper(
        encoder=encoder,
        feature_dim=feat_d,
        proj_dim=int(byol_cfg.get("proj_size", 256)),
        proj_hidden_dim=int(byol_cfg.get("proj_hidden_dim", 1024)),
        ema_decay=float(byol_cfg.get("ema_decay", 0.99)),
    ).to(device)

    aug = build_augmentation_module(byol_cfg.get("augmentation", {}), byol_cfg).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"].get("betas", (0.9, 0.999))),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    plateau_cfg = cfg.get("schedule", {}).get("plateau", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(plateau_cfg.get("factor", 0.5)),
        patience=int(plateau_cfg.get("patience", 5)),
        min_lr=float(plateau_cfg.get("min_lr", 1.0e-10)),
    )

    es_cfg = cfg.get("schedule", {}).get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 10))
    no_improve = 0

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    metrics_csv = os.path.join(out_dir, cfg["filenames"]["metrics_csv"])
    lc_png = os.path.join(out_dir, cfg["filenames"]["learning_curve_png"])
    # BYOL専用の履歴名も追加（既存metrics.csvは互換維持）
    loss_history_csv = os.path.join(out_dir, "loss_history.csv")
    ensure_dir(ckpt_path)

    skip_log = os.path.join(out_dir, "skipped_wavs.log")
    with open(skip_log, "w", encoding="utf-8") as f:
        for p, reason in (train_ds.skipped_files + val_ds.skipped_files):
            f.write(f"{p}\t{reason}\n")

    history = []
    best_val = float("inf")

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for x, _, _ in tqdm(train_loader, desc=f"train {epoch}"):
            x = x.to(device)
            x1, x2 = aug(x), aug(x)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x1, x2)
            loss.backward()
            optimizer.step()
            model._update_target()
            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            n_train += bs

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x, _, _ in tqdm(val_loader, desc=f"val {epoch}"):
                x = x.to(device)
                x1, x2 = aug(x), aug(x)
                loss = model.loss(x1, x2)
                bs = x.size(0)
                val_loss_sum += float(loss.item()) * bs
                n_val += bs

        train_loss = train_loss_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        scheduler.step(val_loss)

        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        save_metrics_csv(metrics_csv, history)
        pd.DataFrame(history).to_csv(loss_history_csv, index=False)
        plot_learning_curve(lc_png, history)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({"model": model.online_encoder.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            no_improve += 1

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"{'BEST ✔' if improved else ''}"
        )
        if no_improve >= es_patience:
            print(f"Early stopping: no val improvement for {es_patience} epochs")
            break

    print(f"Training done. best checkpoint: {ckpt_path}")
    return ckpt_path
