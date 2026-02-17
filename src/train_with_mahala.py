# =========================================================
# train.py（学習ルーチン本体）
#
# 主な責務:
# - AE（Conv2dAE）を OK（正常）データのみで学習し、最良epochのcheckpointを保存する
# - 学習過程を metrics_csv / learning_curve_png として保存する
# - 必要に応じて、再構成残差 diff=(x-x_hat) から Mahalanobis 用の統計量を推定し torch.save する
# - 追加で selective_mahala が enabled の場合、source/target 別に統計量を推定し保存する
#
# 入力:
# - cfg: YAMLからロードされたネスト辞書（main_train.py で読み込む想定）
# 出力:
# - checkpoint_best（torch.save）
# - metrics_csv（学習ログ）
# - learning_curve_png（学習曲線）
# - （任意）Mahalanobis stats（平均/共分散/precision 等）
# =========================================================

# --- 標準ライブラリ ---
import os
import math
import numpy as np
import torch
from torch import optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time

# --- 自作モジュール（データ/モデル/ユーティリティ） ---
from .datasets import create_loader
from .model_ae import Conv2dAE, recon_loss
from .utils import (
    set_seed, ensure_dir, save_metrics_csv, plot_learning_curve,
    vectorize_residual, update_running_mean_cov, finalize_mean_cov, cov_to_precision,
)

# ---------------------------------------------------------
# パス解決ユーティリティ
# - cfg 内のパスが「絶対パス」か「output_dir からの相対パス」かを吸収する
# - 空文字はそのまま返し、呼び出し側で必須チェックする想定
# ---------------------------------------------------------
def _resolve_relpath(out_dir: str, path: str) -> str:
    """Resolve path that can be absolute or relative to output_dir."""
    p = str(path).strip()
    if p == "":
        return ""
    return p if os.path.isabs(p) else os.path.join(out_dir, p)


@torch.no_grad()
def estimate_mahala_pack(
    model: torch.nn.Module,
    loaders: list,
    device: torch.device,
    vectorize: str = "freq",
    eps: float = 1.0e-6,
    use_pinv: bool = True,
) -> dict:
    """
    Estimate residual mean/covariance/precision for Mahalanobis scoring.
    residual diff = x - x_hat
    vectorize_residual(diff) -> vecs [N,D]
    """
    model.eval()
    sum_vec = None
    sum_outer = None
    n_total = 0
    last_meta = None

    for loader in loaders:
        for (x, _, _) in tqdm(loader, desc="[estimate mahala stats]", leave=False):
            x = x.to(device)
            x_hat, _ = model(x)
            diff = (x - x_hat)
            vecs, meta = vectorize_residual(diff, mode=vectorize)  # [N,D]
            last_meta = meta
            v = vecs.detach().to(device=device, dtype=torch.float64)

            if sum_vec is None:
                D = v.size(1)
                sum_vec = torch.zeros(D, device=device, dtype=torch.float64)
                sum_outer = torch.zeros(D, D, device=device, dtype=torch.float64)

            sum_vec += v.sum(dim=0)
            sum_outer += v.transpose(0, 1) @ v
            n_total += int(v.size(0))

    if n_total <= 1:
        raise RuntimeError(f"[mahala] Not enough residual vectors to estimate covariance (n={n_total}).")

    mu = (sum_vec / float(n_total)).to(dtype=torch.float32)
    cov = (sum_outer - float(n_total) * (mu[:, None].double() @ mu[None, :].double())) / float(n_total - 1)
    cov = cov.to(dtype=torch.float32)
    precision = cov_to_precision(cov, eps=eps, use_pinv=use_pinv)

    pack = {
        "mean": mu.detach().cpu(),
        "mu": mu.detach().cpu(),  # alias
        "cov": cov.detach().cpu(),
        "precision": precision.detach().cpu(),
        "n": int(n_total),
        "vectorize": str(vectorize).lower(),
        "eps": float(eps),
        "use_pinv": bool(use_pinv),
        "meta": last_meta,
    }
    return pack


# ---------------------------------------------------------
# 学習エントリポイント
# - AEの学習（train/val）
# - 最良checkpoint保存
# - （任意）Mahalanobis統計推定、Selective Mahalanobis統計推定
# ---------------------------------------------------------
def run_train(cfg):

    # cfg['seed']: 乱数シード（再現性担保の基本）
    set_seed(cfg["seed"])

    # device: 学習を実行するデバイス（CUDAがあればGPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cfg['data']['train_ok_glob']: 学習用 OK データのglob（例: 'data/train_ok/**/*.npy'）
    # create_loader は (DataLoader, dataset_info) のような2値を返す想定（ここでは後者は未使用）
    train_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=True,drop_last=True)

    # cfg['data']['val_ok_glob']: 検証用 OK データのglob
    val_loader, _ = create_loader(cfg["data"]["val_ok_glob"], label=0, cfg=cfg, shuffle=False,drop_last=False)

    # cfg['model']['type']: モデル種類（本実装はconv2d_ae固定）
    assert cfg["model"]["type"] == "conv2d_ae"

    # cfg['model']['bottleneck_dim']: 潜在次元（圧縮率に影響）
    model = Conv2dAE(in_ch=1, bottleneck_dim=cfg["model"]["bottleneck_dim"]).to(device)

    # warm-up forward: lazy形状確定のために1回だけforward（train_loaderが空だとStopIteration）
    x0, _, _ = next(iter(train_loader))
    x0 = x0.to(device)
    with torch.no_grad():
        _ = model(x0)
    model.to(device)

    # Optimizer（Adam）: cfg['train'] から lr/betas/weight_decay を参照
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Scheduler（ReduceLROnPlateau）: cfg['schedule']['plateau'] を参照（val_loss停滞でlr減衰）
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg["schedule"]["plateau"]["factor"],
        patience=cfg["schedule"]["plateau"]["patience"],
        min_lr=cfg["schedule"]["plateau"]["min_lr"]
    )

    # AMP（混合精度）: cfg['train']['amp']=True かつ CUDA の場合のみ有効
    use_amp = bool(cfg["train"]["amp"]) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # cfg['train']['grad_clip_norm']: 勾配クリッピング上限
    max_norm = cfg["train"]["grad_clip_norm"]

    # cfg['output_dir']: 成果物の出力ディレクトリ
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    ckpt_path = os.path.join(out_dir, cfg["filenames"]["checkpoint_best"])
    metrics_csv = os.path.join(out_dir, cfg["filenames"]["metrics_csv"])
    lc_png = os.path.join(out_dir, cfg["filenames"]["learning_curve_png"])
    ensure_dir(ckpt_path)
    ensure_dir(metrics_csv)
    ensure_dir(lc_png)

    # cfg['train']['epochs']: 最大epoch数
    # cfg['loss']['recon_type']/['gaussian_nll_sigma2']: 再構成損失の定義に関与
    epochs = cfg["train"]["epochs"]
    recon_type = cfg["loss"]["recon_type"]
    sigma2 = float(cfg["loss"].get("gaussian_nll_sigma2", 1.0))

    history = []

    # 早期終了
    monitor_best = float("inf")
    epochs_no_improve = 0
    patience = int(cfg["schedule"]["early_stopping"]["patience"])

    autocast_kwargs = {"device_type": "cuda", "dtype": torch.float16, "enabled": use_amp}

    for epoch in range(1, epochs + 1):
        
        # epoch ループ開始（train→val→scheduler→checkpoint→ログ）
        t0 = time.time()

        # ------------------------------
        # Train phase
        # ------------------------------
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for step, (x, _, _) in enumerate(tqdm(train_loader, desc=f"train epoch {epoch+1}/{epochs}")):
            
            # ミニバッチ学習（x: [B,1,H,W]想定 / 実際はDataset実装に依存）
            x = x.to(device)

            # 勾配初期化（set_to_none=Trueでメモリ効率向上）
            optimizer.zero_grad(set_to_none=True)

            # autocast: AMP有効時のみfloat16混在で計算
            with autocast(**autocast_kwargs):

                x_hat, _ = model(x)

                # recon_loss: 正常データをよく再構成できるように学習（異常スコアの基盤）
                loss = recon_loss(x, x_hat, recon_type, sigma2)

            # AMP対応のbackward（scale/unscale→clip→step→update）
            scaler.scale(loss).backward()   # 損失に拡大係数を掛けてbackward。
            scaler.unscale_(optimizer)  # Optimizerが持つparamの勾配を "実スケール" に戻す。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)  # 勾配にInf/NaNが混じっていればstepを自動スキップ。
            scaler.update() # スケール係数を自動調整。

            # ログ用
            bs = x.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs
        
        train_loss = train_loss_sum / max(1, train_n)

        # ------------------------------
        # Validation phase（βは当該エポックの代表値で評価）
        # ------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for (x, _, _) in tqdm(val_loader, desc=f"val epoch {epoch}/{epochs+1}"):
                x = x.to(device)
                with autocast(**autocast_kwargs):
                    x_hat, _ = model(x)
                    loss = recon_loss(x, x_hat, recon_type, sigma2)
                
                bs = x.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_n += bs
        
        val_loss = val_loss_sum / max(1, val_n)

        # scheduler更新（Plateauはval_lossを監視）
        scheduler.step(val_loss)

        # best更新時のみcheckpoint保存
        improved = val_loss < monitor_best
        if improved:
            monitor_best = val_loss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            epochs_no_improve += 1

        # ロギング
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(rec)
        save_metrics_csv(metrics_csv, history)
        plot_learning_curve(lc_png, history)

        print(f"[Epoch {epoch}]  train_loss={train_loss:.6f} val_loss={val_loss:.6f} " 
              f"{'BEST ✔' if improved else ''}")

        # Early stopping 条件を満たしたら終了
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improve {patience} epochs).")
            break

    print(f"Training done. Best checkpoint saved at: {ckpt_path}")

    # =========================================================
    # Post-training: estimate Mahalanobis stats on train residuals
    #  - No gradients / no optimizer step
    #  - Single-domain (source only)
    #  - residual diff = (x - x_hat)
    # =========================================================
    primary = str(cfg.get("scoring", {}).get("primary", "recon_mse")).lower()
    mahala_cfg = (cfg.get("scoring", {}) or {}).get("mahala", {}) or {}
    save_stats = bool(mahala_cfg.get("save_stats", False)) or (primary in ("mahala", "mahalanobis"))

    if save_stats:
        vectorize = str(mahala_cfg.get("vectorize", "freq")).lower()
        eps = float(mahala_cfg.get("eps", 1.0e-6))
        use_pinv = bool(mahala_cfg.get("use_pinv", True))

        # stats_path: explicit -> else output_dir + filenames.mahala_stats_pt
        stats_path = str(mahala_cfg.get("stats_path", "")).strip()
        if stats_path == "":
            stats_path = os.path.join(out_dir, cfg["filenames"].get("mahala_stats_pt", "checkpoints/mahala_stats.pt"))
        ensure_dir(stats_path)

        print(f"[mahala] Estimating covariance on train residuals... (vectorize={vectorize}, eps={eps}, pinv={use_pinv})")
        model.eval()

        # IMPORTANT: use a non-shuffled loader for deterministic stats
        train_stat_loader, _ = create_loader(cfg["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False, drop_last=False)

        pack = estimate_mahala_pack(
            model=model,
            loaders=[train_stat_loader],
            device=device,
            vectorize=vectorize,
            eps=eps,
            use_pinv=use_pinv,
        )

        torch.save(pack, stats_path)
        print(f"[mahala] Saved stats: {stats_path} (D={int(pack['mean'].numel())}, n={int(pack['n'])})")
 

    # =========================================================
    # Post-training: Selective Mahalanobis (source/target)
    # =========================================================
    sel_cfg = (cfg.get("selective_mahala", {}) or {})
    if bool(sel_cfg.get("enabled", False)):
        sel_train = (sel_cfg.get("train", {}) or {})
        sel_val = (sel_cfg.get("val", {}) or {})
        sel_save = (sel_cfg.get("save", {}) or {})

        mcfg = (cfg.get("scoring", {}) or {}).get("mahala", {}) or {}
        vectorize = str(mcfg.get("vectorize", "freq")).lower()
        eps = float(mcfg.get("eps", 1.0e-6))
        use_pinv = bool(mcfg.get("use_pinv", True))

        def _collect_nonempty(*vals):
            out = []
            for v in vals:
                if v is None:
                    continue
                s = str(v).strip()
                if s != "":
                    out.append(s)
            return out

        # trainとvalどちらもマハラノビス距離算出のために使用
        src_globs = _collect_nonempty(sel_train.get("source_path", ""), sel_val.get("source_path", ""))
        tgt_globs = _collect_nonempty(sel_train.get("target_path", ""), sel_val.get("target_path", ""))
        if len(src_globs) == 0 or len(tgt_globs) == 0:
            raise RuntimeError("[selective_mahala] enabled=True, but source/target paths are missing.")

        src_loaders = []
        for g in src_globs:
            ld, _ = create_loader(g, label=0, cfg=cfg, shuffle=False, drop_last=False)
            src_loaders.append(ld)

        tgt_loaders = []
        for g in tgt_globs:
            ld, _ = create_loader(g, label=0, cfg=cfg, shuffle=False, drop_last=False)
            tgt_loaders.append(ld)

        src_out = _resolve_relpath(out_dir, sel_save.get("source_precision_path", ""))
        tgt_out = _resolve_relpath(out_dir, sel_save.get("target_precision_path", ""))
        if src_out == "" or tgt_out == "":
            raise RuntimeError("[selective_mahala] enabled=True, but save paths are missing.")
        ensure_dir(src_out)
        ensure_dir(tgt_out)

        print(f"[selective_mahala] Estimating SOURCE stats... (vectorize={vectorize}, eps={eps}, pinv={use_pinv})")
        src_pack = estimate_mahala_pack(model=model, loaders=src_loaders, device=device, vectorize=vectorize, eps=eps, use_pinv=use_pinv)
        src_pack["domain"] = "source"
        src_pack["paths_used"] = src_globs

        print(f"[selective_mahala] Estimating TARGET stats... (vectorize={vectorize}, eps={eps}, pinv={use_pinv})")
        tgt_pack = estimate_mahala_pack(model=model, loaders=tgt_loaders, device=device, vectorize=vectorize, eps=eps, use_pinv=use_pinv)
        tgt_pack["domain"] = "target"
        tgt_pack["paths_used"] = tgt_globs

        torch.save(src_pack, src_out)
        torch.save(tgt_pack, tgt_out)
        print(f"[selective_mahala] Saved source stats: {src_out} (D={int(src_pack['mean'].numel())}, n={int(src_pack['n'])})")
        print(f"[selective_mahala] Saved target stats: {tgt_out} (D={int(tgt_pack['mean'].numel())}, n={int(tgt_pack['n'])})")

    return ckpt_path
