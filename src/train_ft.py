import os
import torch
import yaml
import glob
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from .datasets import AudioDataset, create_loader
from .model_ae import Conv2dAE, recon_loss
from .utils import set_seed, ensure_dir, save_metrics_csv, plot_learning_curve

def run_train_ft(cfg):
    # 1. FT用設定の取得と検証
    ft_cfg = cfg.get("finetune")
    if not ft_cfg:
        raise ValueError("Config missing 'finetune' section.")

    # 出力ディレクトリの設定
    base_out_dir = cfg.get("output_dir", "./result")
    ft_out_subdir = ft_cfg.get("out_subdir", "ft")
    output_dir = os.path.join(base_out_dir, ft_out_subdir)
    
    ckpt_names = ft_cfg.get("output", {}).get("filenames", {})
    ckpt_path = os.path.join(output_dir, ckpt_names.get("checkpoint_best", "checkpoints/best_ft.pt"))
    metrics_csv = os.path.join(output_dir, "metrics_ft.csv")
    lc_png = os.path.join(output_dir, "learning_curve_ft.png")

    ensure_dir(ckpt_path)

    # 乱数固定
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. データローダーの準備 (モデル初期化の前に必要)
    train_glob = ft_cfg["data"]["train_ok_glob"]
    val_glob = ft_cfg["data"].get("val_ok_glob")

    if val_glob:
        train_loader, _ = create_loader(train_glob, label=0, cfg=cfg, shuffle=True, drop_last=True)
        val_loader, _ = create_loader(val_glob, label=0, cfg=cfg, shuffle=False, drop_last=False)
    else:
        print("Validation glob not provided. Splitting training data (80/20).")
        full_dataset = AudioDataset(train_glob, label=0, cfg=cfg)
        n_total = len(full_dataset)
        if n_total < 2:
            raise ValueError(f"Too few samples ({n_total}) to split for validation.")
        
        indices = np.random.permutation(n_total)
        split = int(n_total * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=ft_cfg.get("train", {}).get("batch_size", 4), shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False)

    # 3. モデルのロードと初期化
    model = Conv2dAE(in_ch=1, bottleneck_dim=cfg["model"]["bottleneck_dim"]).to(device)
    
    # --- 重要：実際のデータを使って Lazy初期化を確定させる ---
    # 固定の 128x128 ではなく、Datasetから取得した実際の形状を使用する
    x0, _, _ = train_loader.dataset[0]
    x0 = x0.unsqueeze(0).to(device) # [1, 1, F, T]
    
    print(f"Initializing model with actual input shape: {x0.shape}")
    with torch.no_grad():
        model(x0) # ここで fc_enc などの Linear層のサイズが確定する
    
    # Base Checkpointのロード
    try:
        base_ckpt_path = cfg["finetune"]["base_ckpt_path"]
    except:
        base_ckpt_path = os.path.join(base_out_dir,cfg["filenames"]["checkpoint_best"])
    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt_path}")
    
    print(f"Loading base model from {base_ckpt_path}...")
    checkpoint = torch.load(base_ckpt_path, map_location=device)
    
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        print("\n" + "!"*60)
        print("次元不一致エラー (Dimension Mismatch Error):")
        print("Domain B の設定が Domain A の学習時と異なっている可能性があります。")
        print("以下を確認してください：")
        print(f"  - YAMLの feature.logmel.n_mels (現在は {cfg['feature']['logmel']['n_mels']})")
        print(f"  - 音声ファイルの長さ (Domain Aと同じ秒数である必要があります)")
        print(f"  - YAMLの audio.target_sr (現在は {cfg['audio']['target_sr']})")
        print("!"*60 + "\n")
        raise e

    # 4. 凍結戦略
    strategy = ft_cfg.get("strategy", {}).get("freeze", "encoder")
    if strategy == "encoder":
        print("Strategy: Freezing Encoder (Training Decoder only)")
        for param in model.enc.parameters(): param.requires_grad = False
        for param in model.fc_enc.parameters(): param.requires_grad = False
    elif strategy == "all_but_decoder":
        print("Strategy: Freezing all but Decoder layers")
        for name, param in model.named_parameters():
            if "dec" not in name: param.requires_grad = False
    else:
        print("Strategy: No layers frozen (Full Fine-tuning)")

    # 5. 学習準備
    lr = ft_cfg.get("train", {}).get("lr", cfg["train"]["lr"] * 0.1)
    epochs = ft_cfg.get("train", {}).get("epochs", 50)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    
    # 6. 学習ループ
    history = []
    monitor_best = float("inf")
    recon_type = cfg["loss"]["recon_type"]
    sigma2 = cfg["model"].get("sigma2", 1.0)

    print(f"Starting Fine-tuning... (Train samples: {len(train_loader.dataset)})")
    
    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        n_tr = 0
        for (x, _, _) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [FT-train]"):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = recon_loss(x, x_hat, recon_type, sigma2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            bs = x.size(0)
            tr_sum += loss.item() * bs
            n_tr += bs
        
        train_loss = tr_sum / max(1, n_tr)

        model.eval()
        va_sum = 0.0
        n_va = 0
        with torch.no_grad():
            for (x, _, _) in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                loss = recon_loss(x, x_hat, recon_type, sigma2)
                va_sum += loss.item() * x.size(0)
                n_va += x.size(0)
        val_loss = va_sum / max(1, n_va)

        improved = val_loss < monitor_best
        if improved:
            monitor_best = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg, "ft_cfg": ft_cfg}, ckpt_path)
            print(f"  --> Saved Best FT Model (val_loss: {val_loss:.6f})")

        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        history.append(rec)
        save_metrics_csv(metrics_csv, history)
        plot_learning_curve(lc_png, history)

    print(f"Fine-tuning completed. Results saved in: {output_dir}")