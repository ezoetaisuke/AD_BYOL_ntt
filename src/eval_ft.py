import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from pathlib import PurePath
import torch.nn.functional as F
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from .datasets import create_loader
from .model_ae import Conv2dAE, recon_loss
from .utils import (
    ensure_dir, compute_roc_pr, plot_roc_pr,
    plot_hist_by_class, plot_confusion,
    extract_subclass_from_path, plot_hist_by_subclass,
    plot_hist_all_subclasses,
    save_spec_triplet_png,
)

def _as_1d_float_list(x):
    if np.isscalar(x):
        return [float(x)]
    arr = np.asarray(x, dtype=float)
    return arr.ravel().tolist()

def _aggregate_time(err_map, method="topk_mean", topk_ratio=0.1):
    err_map = np.asarray(err_map)
    if err_map.ndim == 4:
        per_t = err_map.mean(axis=2).squeeze(1)
    elif err_map.ndim == 3:
        per_t = err_map.mean(axis=1)
    else:
        raise ValueError(f"err_map must be 3D or 4D, got {err_map.shape}")

    B, T = per_t.shape[0], per_t.shape[-1]
    if method == "mean":
        out = per_t.mean(axis=1)
    elif method == "max":
        out = per_t.max(axis=1)
    elif method == "topk_mean":
        k = max(1, int(round(T * float(topk_ratio))))
        idx = np.argpartition(per_t, -k, axis=1)[:, -k:]
        topk_vals = np.take_along_axis(per_t, idx, axis=1)
        out = topk_vals.mean(axis=1)
    else:
        raise ValueError(f"Unknown aggregate method: {method}")
    return out



def run_eval_ft(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg["output_dir"],cfg["finetune"]["out_subdir"])
    ft_output_cfg = cfg["finetune"]["output"]["filenames"]
    ckpt_path_ft = os.path.join(out_dir, ft_output_cfg["checkpoint_best"])

    print(f"ckpt_path_for_fine_tuning:{ckpt_path_ft}")

    if not os.path.exists(ckpt_path_ft):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_ft}")

    # ====== DataLoaders ======
    train_loader, _   = create_loader(cfg["finetune"]["data"]["train_ok_glob"], label=0, cfg=cfg, shuffle=False)
    val_loader, _     = create_loader(cfg["finetune"]["data"]["val_ok_glob"],   label=0, cfg=cfg, shuffle=False)
    test_ok_loader, _ = create_loader(cfg["finetune"]["data"]["test_ok_glob"],  label=0, cfg=cfg, shuffle=False)
    test_ng_loader, _ = create_loader(cfg["finetune"]["data"]["test_ng_glob"],  label=1, cfg=cfg, shuffle=False)

    primary    = cfg["scoring"]["primary"]
    agg_method = cfg["scoring"]["aggregate_method"]
    topk_ratio = cfg["scoring"]["topk_ratio"]
    recon_type = cfg["loss"]["recon_type"]
    sigma2     = cfg["loss"]["gaussian_nll_sigma2"]

    # ###########################################
    # ### 波形画像を出力（非使用時はコメントアウト）
    # ###########################################
    # # 先頭バッチからパスを取得
    # x, y, paths = next(iter(test_ok_loader))
    # path = paths[0]

    # # 波形読み込み（mono化・必要なら再サンプリング）
    # y_wav, sr = sf.read(path, dtype="float32", always_2d=True)
    # y_wav = y_wav.mean(axis=1)  # mono

    # # cfgの設定に合わせてリサンプル（例：target_srに合わせる）
    # target_sr = cfg["audio"]["target_sr"]
    # if sr != target_sr and cfg["audio"]["resample_if_mismatch"]:
    #     y_wav = librosa.resample(y_wav, orig_sr=sr, target_sr=target_sr)
    #     sr = target_sr

    # # 可視化（時間軸）
    # t = np.arange(len(y_wav)) / sr
    # plt.figure(figsize=(10,3))
    # plt.plot(t, y_wav)
    # plt.grid()
    # plt.title(f"Waveform: {path}")
    # plt.xlabel("Time [s]"); plt.ylabel("Amplitude")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir,'waveform_tmp.png'))

    # ###########################################
    # ### 波形画像を出力（非使用時はコメントアウト）終了
    # ###########################################



    # ====== モデル作成 → ウォームアップ forward ======
    model = Conv2dAE(in_ch=1, bottleneck_dim=cfg["model"]["bottleneck_dim"]).to(device)


    def _first_batch_or_none(loader):
        try:
            return next(iter(loader))
        except StopIteration:
            return None

    batch = _first_batch_or_none(val_loader) or _first_batch_or_none(train_loader) or _first_batch_or_none(test_ok_loader)
    if batch is None:
        raise RuntimeError("No data available to perform warm-up forward in eval(). "
                           "Please check your data.globs in the config.")
    x0, _, _ = batch
    x0 = x0.to(device)
    with torch.no_grad():
        _ = model(x0)
    model.to(device)

    # ====== チェックポイントロード ======
    state = torch.load(ckpt_path_ft, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # --- 画像保存の設定を読み込み ---
    vis_cfg = (cfg.get("eval", {}) or {}).get("save_recon_images", {})  # ← 1) のyaml
    vis_enable   = bool(vis_cfg.get("enable", False))
    vis_folder   = str(vis_cfg.get("folder_name", "img"))
    vis_dir      = os.path.join(out_dir, vis_folder)
    vis_max      = int(vis_cfg.get("max_items", 0))
    vis_cmap     = str(vis_cfg.get("cmap", "magma"))
    vis_diffmode = str(vis_cfg.get("diff_mode", "abs")).lower()
    vis_dpi      = int(vis_cfg.get("dpi", 150))
    if vis_enable:
        os.makedirs(vis_dir, exist_ok=True)


    # --- val_okスコア（P99用） ---
    v_scores = []
    if primary in ("recon_mse", "recon_l1"):
        with torch.no_grad():
            for x, _, _ in tqdm(val_loader, desc="[val scoring]"):
                x = x.to(device)
                x_hat, _ = model(x)
                if primary == "recon_mse":
                    err_map = (x - x_hat).pow(2).detach().cpu().numpy()
                else:
                    err_map = (x - x_hat).abs().detach().cpu().numpy()
                v_scores.extend(_as_1d_float_list(_aggregate_time(err_map, method=agg_method, topk_ratio=topk_ratio)))
    val_scores = np.asarray(v_scores, dtype=float)

    # --- テストスコア計算（OK/NG両方） ---
    scores, y_true, paths = [], [], []
    saved_count = 0 # 画像保存枚数カウント

    with torch.no_grad():
        for loader, true_label in [(test_ok_loader, 0), (test_ng_loader, 1)]:
            for (x, y, p) in tqdm(loader, desc="[test scoring]"):
                x = x.to(device)
                x_hat, _ = model(x)

                if primary == "recon_mse":
                    err_map = (x - x_hat).pow(2).detach().cpu().numpy()
                elif primary == "recon_l1":
                    err_map = (x - x_hat).abs().detach().cpu().numpy()
                else:
                    raise ValueError(f"Unsupported primary: {primary}")
                
                file_scores = _aggregate_time(err_map, method=agg_method, topk_ratio=topk_ratio)

                # ---- 画像保存：この「サンプルのスコア」をタイトルに入れる ----
                if vis_enable:
                    B = x.size(0)
                    for i in range(B):
                        if (vis_max > 0) and (saved_count >= vis_max):
                            break
                        X    = x[i, 0].detach().cpu().numpy()
                        Xhat = x_hat[i, 0].detach().cpu().numpy()
                        lbl_name  = PurePath(p[i]).parts[-2]
                        file_name = os.path.splitext(PurePath(p[i]).parts[-1])[0]
                        fig_store_dir_path = os.path.join(vis_dir, lbl_name)
                        os.makedirs(fig_store_dir_path, exist_ok=True)
                        out_png = os.path.join(fig_store_dir_path, f"{file_name}.png")
                        save_spec_triplet_png(
                            out_png, X, Xhat,
                            diff_mode=vis_diffmode, cmap=vis_cmap, dpi=vis_dpi,
                            score=float(file_scores[i]),  # ← 各サンプルの値！
                        )
                        saved_count += 1



                # ---- CSV/評価用に積む ----
                scores.extend(_as_1d_float_list(file_scores))
                y_true.extend(y.numpy().astype(int).tolist())
                paths.extend(p)
                    
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    paths  = np.asarray(paths, dtype=object)

    # --- 閾値決定 ---
    method = cfg["threshold"]["method"]
    if method == "p99_val_ok":
        thr = float(np.percentile(val_scores, 99.0))
    elif method == "youden_test":
        from sklearn.metrics import roc_curve
        fpr, tpr, thr_list = roc_curve(y_true, scores)
        youden = tpr - fpr
        thr = float(thr_list[np.argmax(youden)])
    elif method == "f1max_test":
        thr_candidates = np.unique(scores)
        best_f1, best_thr = -1.0, None
        for t in thr_candidates:
            y_pred_tmp = (scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred_tmp)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        thr = float(best_thr)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # 予測
    y_pred = (scores >= thr).astype(int)

    # --- CSV出力 ---
    import pandas as pd
    scores_csv = os.path.join(out_dir, cfg["filenames"]["scores_csv"])
    ensure_dir(scores_csv)
    df = pd.DataFrame({"path": paths, "y_true": y_true, "score": scores, "y_pred": y_pred})
    df.to_csv(scores_csv, index=False)

    # --- 図表（基本） ---
    cm_png = os.path.join(out_dir, cfg["filenames"]["confusion_matrix_png"])
    plot_confusion(cm_png, y_true, y_pred, labels=("OK","NG"))

    roc_pack, pr_pack = compute_roc_pr(y_true, scores)
    roc_png = os.path.join(out_dir, cfg["filenames"]["roc_png"])
    pr_png  = os.path.join(out_dir, cfg["filenames"]["pr_png"])
    plot_roc_pr(roc_png, pr_png, roc_pack, pr_pack)

    # --- サブクラス色分けヒスト（OK+NG） ---
    subclasses = np.array([extract_subclass_from_path(p) for p in paths])
    all_hist_png = os.path.join(out_dir, cfg["filenames"]["score_hist_all_subclasses_png"])
    plot_hist_all_subclasses(
        all_hist_png,
        scores,
        subclasses.tolist(),
        title="All Subclasses (OK + NG): Score Histogram with Color by Subclass"
    )

    print(f"Eval done. Threshold={thr:.6f} ")
    print(f"- scores.csv: {scores_csv}")
    print(f"- confusion_matrix: {cm_png}")
    print(f"- score_hist ALL subclasses: {all_hist_png}")
    print(f"- roc/pr: {roc_png} / {pr_png}")
