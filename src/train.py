import copy
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from byol_a.byol_a.models import AudioNTT2020

from .datasets import create_loader
from .utils import ensure_dir, plot_learning_curve, save_metrics_csv, set_seed


class RandomSpecAugment(nn.Module):
    """Simple stochastic augment for log-mel spectrogram batches."""

    def __init__(self, noise_std=0.05, dropout_p=0.1):
        super().__init__()
        self.noise_std = float(noise_std)
        self.dropout = nn.Dropout2d(p=float(dropout_p))

    def forward(self, x):
        # x: [B, 1, F, T]
        x = self.dropout(x)
        noise = torch.randn_like(x) * self.noise_std
        return x + noise


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
    byol_cfg = cfg.get("model", {}).get("byol_a", {})
    path = byol_cfg.get("pretrained_weight_path")
    if path:
        return path

    feat_d = int(byol_cfg.get("feature_d", 2048))
    return f"byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d{feat_d}.pth"


def run_train(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("model", {}).get("type") != "byol_a":
        raise ValueError("Only model.type='byol_a' is supported.")

    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=True, drop_last=True)
    val_loader, _ = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)

    byol_cfg = cfg.get("model", {}).get("byol_a", {})
    n_mels = int(cfg.get("feature", {}).get("logmel", {}).get("n_mels", byol_cfg.get("n_mels", 64)))
    feat_d = int(byol_cfg.get("feature_d", 2048))
    encoder = AudioNTT2020(n_mels=n_mels, d=feat_d).to(device)

    if bool(byol_cfg.get("use_pretrained", True)):
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

    aug = RandomSpecAugment(
        noise_std=float(byol_cfg.get("augment_noise_std", 0.05)),
        dropout_p=float(byol_cfg.get("augment_dropout", 0.1)),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"].get("betas", (0.9, 0.999))),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    metrics_csv = os.path.join(out_dir, cfg["filenames"]["metrics_csv"])
    lc_png = os.path.join(out_dir, cfg["filenames"]["learning_curve_png"])
    ensure_dir(ckpt_path)

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
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        save_metrics_csv(metrics_csv, history)
        plot_learning_curve(lc_png, history)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.online_encoder.state_dict(), "cfg": cfg}, ckpt_path)

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    print(f"Training done. best checkpoint: {ckpt_path}")
    return ckpt_path
