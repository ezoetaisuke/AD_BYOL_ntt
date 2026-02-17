import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # ★ 追加：Seabornで見た目改善
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# 乱数固定：再現性を高める（完全決定論は性能低下の場合がある）
def set_seed(seed: int):
    """
    乱数シードを一括固定し、実験の再現性を確保する。
    
    Args:
        seed (int): 固定するシード値
    
    Note:
        - 完全な決定論的挙動（deterministic）を強制すると、畳み込み演算などの
          アルゴリズムが制限され、処理速度が低下する場合がある。
        - GPUを使用する場合、cuDNNのベンチマーク機能をオフにすることで
          入力サイズが変わっても同じアルゴリズムが選ばれるようにする。
    """

    # Python標準の乱数固定
    random.seed(seed)

    # Numpyの乱数固定
    np.random.seed(seed)

    # PyTorch CPU/GPUの乱数固定
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)    # マルチGPUの場合

    # cuDNNの挙動を固定（再現性重視の設定）
    # 決定論的アルゴリズムのみを使用するように強制
    torch.backends.cudnn.deterministic = True

    # 最適なアルゴリズムを動的に探す機能をオフ（入力サイズ固定なら再現性に寄与）
    torch.backends.cudnn.benchmark = False

# path から親ディレクトリ部分だけを取り出す（例: "a/b/c.csv" -> "a/b"）
# 親ディレクトリが存在しない場合のみ作成し、存在する場合は何もしない
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 学習履歴をCSVに保存
def save_metrics_csv(out_csv: str, history: list):
    """
    学習履歴 `history` を Pandas DataFrame に変換し、CSV として保存する。

    Parameters
    ----------
    out_csv : str
        出力CSVファイルパス（例: "runs/exp01/metrics.csv"）。
        事前に `ensure_dir(out_csv)` を呼ぶことで、親フォルダが無ければ作成される。
    history : list
        学習履歴。一般に `dict` のリストを想定（例: [{"epoch":1,"train_loss":...,"val_loss":...}, ...]）。
        DataFrame化により、dict のキーが列名になる。

    処理の流れ（ロジックは現状のまま）
    --------------------------------
    1) `history` を DataFrame 化（行＝エポックなどの記録、列＝指標名）
    2) 出力先 `out_csv` の親ディレクトリを作成（必要なら）
    3) CSV へ保存（index=False で行番号列は出さない）
    """
    # list[dict] を DataFrame に変換（キーが列名になる）
    df = pd.DataFrame(history)

    # 保存先CSVの親ディレクトリを作成（無ければ作る / あれば何もしない）
    ensure_dir(out_csv)

    # CSV書き出し：index=False で DataFrame のインデックス列を保存しない
    df.to_csv(out_csv, index=False)

# =============================================================================
# OKスプリット（train_ok / val_ok / test_ok）の path と score をCSVに保存
# =============================================================================
def save_ok_split_scores_csv(out_csv: str, records):
    """
    OKデータ（複数split）のスコアログをCSVとして保存する。

    Parameters
    ----------
    out_csv : str
        出力CSVパス（例: "runs/exp01/scores_ok_splits.csv"）
    records : list[dict] or pandas.DataFrame
        1行=1ファイル相当のレコード。
        必須列: split, path, score

    Notes
    -----
    - score は float へ変換する（変換不能は NaN 扱い）
    - NaN / inf の score 行は除外する（安全性のため）
    - 本関数は「ログ保存」のみ。スコア計算や閾値決定ロジックには影響しない。
    """
    ensure_dir(out_csv)

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(list(records))

    required_cols = ["split", "path", "score"]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"save_ok_split_scores_csv: missing columns: {missing}")

    # score を float化（失敗は NaN）
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # NaN/inf 除外
    score_np = df["score"].to_numpy(dtype=float, copy=False)
    mask = np.isfinite(score_np)
    n_drop = int((~mask).sum())
    if n_drop > 0:
        print(f"[WARN] Dropped {n_drop} rows due to non-finite score (NaN/inf) in {out_csv}")
    df = df.loc[mask, required_cols]

    df.to_csv(out_csv, index=False)
    return out_csv



# 学習曲線（ELBO）を保存
def plot_learning_curve(out_png: str, history: list):
    """
    学習履歴 `history` から「epoch vs loss（train/val）」の学習曲線を作成し、PNGとして保存する。

    Parameters
    ----------
    out_png : str
        出力PNGファイルパス（例: "runs/exp01/learning_curve.png"）。
        先に `ensure_dir(out_png)` を呼び、親ディレクトリが無ければ作成してから保存する。
    history : list
        学習履歴。要素は dict を想定し、最低限以下のキーを持つ前提:
          - "epoch"      : エポック番号（x軸）
          - "train_loss" : 訓練損失（y軸）
          - "val_loss"   : 検証損失（y軸）
        例: [{"epoch":1,"train_loss":..., "val_loss":...}, ...]

    Notes
    -----
    - 可視化は `matplotlib.pyplot` のステートフルAPI（plt.*）で描画しているため、
      `fig` を明示的に close してメモリリーク/図の積み上がりを防いでいる。
    - y軸ラベルは "Loss (lower is better)" としているため、指標が損失である想定。
    """

    # 保存先PNGの親ディレクトリを作成（無ければ作る / あれば何もしない）
    ensure_dir(out_png)

    # history からプロット用の系列を抽出
    # - epochs : x軸（エポック番号）
    # - tr_elbo/va_elbo : y軸（train/val の損失系列）
    epochs = [h["epoch"] for h in history]
    tr_elbo = [h["train_loss"] for h in history]
    va_elbo = [h["val_loss"] for h in history]

    # Figure を作成（plt.* で描画するが、最後に close するため参照を保持する）
    fig = plt.figure(figsize=(6,4))
    
    # 学習曲線（train/val）を同一グラフ上に描画
    plt.plot(epochs, tr_elbo, label="train loss")
    plt.plot(epochs, va_elbo, label="val loss")

    # 軸ラベル・凡例・グリッド設定（可読性を確保）
    plt.xlabel("Epoch"); plt.ylabel("Loss (lower is better)")
    plt.legend(); plt.grid(True)

    # 余白を詰めて保存（ラベル切れ防止）
    plt.tight_layout()

    # PNGとして保存（dpi=150 で適度な解像度）
    plt.savefig(out_png, dpi=150)

    # Figure を閉じてリソース解放（ループで多枚数保存するケースを想定）
    plt.close(fig)

# ROC/PR 計算
def compute_roc_pr(y_true, scores):
    fpr, tpr, thr_roc = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, thr_pr = precision_recall_curve(y_true, scores)
    pr_auc = auc(rec, prec)
    return (fpr, tpr, thr_roc, roc_auc), (prec, rec, thr_pr, pr_auc)

# ROC/PR 図
def plot_roc_pr(out_roc, out_pr, roc_pack, pr_pack):
    ensure_dir(out_roc); ensure_dir(out_pr)
    fpr, tpr, _, roc_auc = roc_pack
    prec, rec, _, pr_auc = pr_pack
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_roc, dpi=150); plt.close(fig1)

    fig2 = plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_pr, dpi=150); plt.close(fig2)

# ===============================
# ヒストグラム（Seabornスタイルに刷新）
# ===============================
def _shared_bins_from_arrays(*arrays, nbins: int = 20):
    """
    配列群の全体最小/最大から共有bin（np.linspace）を作る。
    すべてNaN or 同値の場合は固定本数で返す。
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if len(a) > 0]) if len(arrays) > 0 else np.asarray([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 20
    vmin, vmax = np.min(vals), np.max(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 10
    return np.linspace(vmin, vmax, nbins + 1)

# OK/NG 2色ヒスト（Seaborn版）
def plot_hist_by_class(out_png, scores_ok, scores_ng, title: str = "Anomaly Score Histogram",
                       xlabel: str = "Anomaly Score", ylabel: str = "Count",
                       nbins: int = 20):
    """
    OK(青)/NG(赤)の2色ヒストを Seaborn histplot で描画。
    - kde=True（分布のなめらか曲線）
    - stat='count'（縦軸は件数）
    - element='step'（縁取りステップ表示）
    """
    ensure_dir(out_png)
    bins = _shared_bins_from_arrays(scores_ok, scores_ng, nbins=nbins)

    df = pd.DataFrame({
        "score": np.concatenate([np.asarray(scores_ok, dtype=float), np.asarray(scores_ng, dtype=float)]),
        "label": (["OK"] * len(scores_ok)) + (["NG"] * len(scores_ng))
    })
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(
        data=df, x="score", hue="label",
        bins=bins, kde=True, stat="count", element="step", alpha=0.5, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# 混同行列
def plot_confusion(out_png, y_true, y_pred, labels=("OK","NG")):
    ensure_dir(out_png)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black", fontsize=12)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# 親フォルダ名をサブクラス名として抽出
def extract_subclass_from_path(path: str) -> str:
    base = os.path.basename(os.path.dirname(path))
    return base

# サブクラス別（OKのみ or NGのみなど）ヒスト（Seaborn版）
def plot_hist_by_subclass(out_png: str, scores: np.ndarray, groups: list, title: str,
                          xlabel: str = "Anomaly Score", ylabel: str = "Count",
                          nbins: int = 20):
    """
    同一集合（OKのみ、NGのみなど）内でサブクラスごとに色分けヒスト。
    KDE付き、共有bin、countスケール、step表示。
    """
    ensure_dir(out_png)
    scores = np.asarray(list(scores), dtype=float)
    uniq = sorted(list(set(groups)))
    bins = _shared_bins_from_arrays(scores, nbins=nbins)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab20", n_colors=max(20, len(uniq)))

    for i, g in enumerate(uniq):
        s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
        if s.size == 0:
            continue
        sns.histplot(s,
                     alpha=0.6, label=g, ax=ax, stat='count', bins=bins, element='step')

    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _fd_bin_edges(values: np.ndarray, min_bins: int = 12, max_bins: int = 24):
    """
    Freedman–Diaconisでビン幅を決めて、ビン数を[min_bins, max_bins]にクランプする。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0, 1, min_bins + 1)

    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        return np.linspace(vmin - 0.5, vmax + 0.5, min_bins + 1)

    # IQR
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float(q75 - q25)

    # FDのビン幅 h
    if iqr > 0.0:
        h = 2.0 * iqr / np.cbrt(v.size)
    else:
        # IQR=0 の退避（Scottの幅 or 固定幅）
        sd = float(np.std(v))
        h = 3.5 * sd / np.cbrt(v.size) if sd > 0 else (vmax - vmin) / max(min_bins, 1)

    if h <= 0:
        nbins = max(min_bins, 1)
    else:
        nbins = int(np.ceil((vmax - vmin) / h))
        nbins = int(np.clip(nbins, min_bins, max_bins))

    # 等間隔の境界に整形
    return np.linspace(vmin, vmax, nbins + 1)


def _shared_bins_fd_clamped(arrays, min_bins: int = 12, max_bins: int = 24):
    """
    複数配列（OK/NG/サブクラスなど）を結合して
    FD＋クランプで “共有bin” 境界を返す。
    """
    concat = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays if len(a) > 0], axis=0)
    return _fd_bin_edges(concat, min_bins=min_bins, max_bins=max_bins)

def save_spec_triplet_png(
    out_png: str,
    X: np.ndarray,        # [F, T] エンコーダ入力（特徴量）
    Xhat: np.ndarray,     # [F, T] デコーダ出力（再構成特徴量）
    diff_mode: str = "abs",   # "abs" or "signed"
    cmap: str = "magma",
    dpi: int = 150,
    score:float = None,
    show_colorbar: bool = True
):
    """
    入力, 再構成, 差分 を横一列に並べて保存する。
    - 入力/再構成は同じ vmin/vmax を共有して比較をしやすく。
    - 差分は diff_mode に応じて |X-Xhat| か (X-Xhat) を描画。
    """
    ensure_dir(out_png)

    X = np.asarray(X, dtype=float)
    Xhat = np.asarray(Xhat, dtype=float)

    # 入力・再構成は同一カラースケールで比較
    vmin = float(np.nanmin([X.min(), Xhat.min()]))
    vmax = float(np.nanmax([X.max(), Xhat.max()]))

    if diff_mode == "signed":
        D = X - Xhat
        d_absmax = float(np.max(np.abs(D)))
        d_vmin, d_vmax = -d_absmax, d_absmax
        diff_cmap = "coolwarm"
    else:
        D = np.abs(X - Xhat)
        d_vmin, d_vmax = float(D.min()), float(D.max())
        diff_cmap = cmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(X, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Input")
    im1 = axes[1].imshow(Xhat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Recon")
    im2 = axes[2].imshow(D, aspect="auto", origin="lower", cmap=diff_cmap, vmin=d_vmin, vmax=d_vmax)
    axes[2].set_title("Diff")

    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("Freq")

    # ---------------------------------------------------------
    # カラーバー（色と値の対応表）を追加
    #  - Input / Recon は同一スケール（vmin/vmax共有）なので 1本を共有
    #  - Diff は別スケールなので専用のカラーバーを付ける
    # ---------------------------------------------------------
    if show_colorbar:
        # Input + Recon（axes[0], axes[1]）で共有カラーバー
        cbar_main = fig.colorbar(im0, ax=axes[0:2], fraction=0.046, pad=0.02)
        cbar_main.set_label("Amplitude")

        # Diff 専用カラーバー
        diff_label = "Signed Error" if diff_mode == "signed" else "Abs Error"
        cbar_diff = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
        cbar_diff.set_label(diff_label)

    if score is not None:
        fig.suptitle(f"score={score:.6f}")

    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)

# ★OK/NGすべてのサブクラスを1枚に統合して色分け（Seaborn版）
def plot_hist_all_subclasses(
        out_png: str, 
        scores: np.ndarray, 
        groups: list, 
        title: str,
        xlabel: str = "Anomaly Score", 
        ylabel: str = "Count",
        bins="fd_clamped",
        min_bins: int = 12,
        max_bins: int = 24,
    ):

    """
    OK/NGを含む全サブクラスを1枚で色分け表示。
    ラベルは '... (OK)' / '... (NG)' のようにタグ付け。
    KDE付き、共有bin、countスケール、step表示。
    """
    ensure_dir(out_png)
    scores = np.asarray(list(scores), dtype=float)
    uniq = sorted(list(set(groups)))

    def decorate(name: str) -> str:
        tag = "OK" if "ok" in name.lower() else ("NG" if "ng" in name.lower() else "")
        return f"{name} ({tag})" if tag else name

    if isinstance(bins, str) and bins == "fd_clamped":
        bins_ = _shared_bins_fd_clamped([scores], min_bins, max_bins)
    else:
        bins_ = bins

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, g in enumerate(uniq):
        s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
        mu = float(np.mean(s))
        sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
        s_max = max(s)
        s_min = min(s)
        if s.size == 0:
            continue
        ax.hist(
            s,
            bins=20,
            alpha=0.8,
            histtype="stepfilled",
            # edgecolor="k",
            # linewidth=0.5,
            label=f"{g} (μ={mu:.2f}, σ={sd:.2f}, max={s_max:.2f}, min={s_min:.2f}, n={s.size})",
        )
        # ax.vlines(s, ymin=0, ymax=max(1, int(0.05 * s.size)), colors="k", alpha=0.35, linewidth=0.8)

    # ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# def plot_hist_all_subclasses_with_confidence(
#         out_png: str, 
#         scores: np.ndarray, 
#         groups: list, 
#         cfg,
#         threshold,
#         xlabel: str = "Anomaly Score", 
#         ylabel: str = "Count",
#         bins="fd_clamped",
#         min_bins: int = 12,
#         max_bins: int = 24,
#     ):

#     """
#     OK/NGを含む全サブクラスを1枚で色分け表示。
#     ラベルは '... (OK)' / '... (NG)' のようにタグ付け。
#     KDE付き、共有bin、countスケール、step表示。
#     """
#     ensure_dir(out_png)
#     scores = np.asarray(list(scores), dtype=float)
#     uniq = sorted(list(set(groups)))

#     def decorate(name: str) -> str:
#         tag = "OK" if "ok" in name.lower() else ("NG" if "ng" in name.lower() else "")
#         return f"{name} ({tag})" if tag else name

#     if isinstance(bins, str) and bins == "fd_clamped":
#         bins_ = _shared_bins_fd_clamped([scores], min_bins, max_bins)
#     else:
#         bins_ = bins

#     fig, ax = plt.subplots(figsize=(10, 6))

#     for i, g in enumerate(uniq):
#         s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
#         mu = float(np.mean(s))
#         sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
#         s_max = max(s)
#         s_min = min(s)
#         if s.size == 0:
#             continue
#         ax.hist(
#             s,
#             bins=20,
#             alpha=0.8,
#             histtype="stepfilled",
#             # edgecolor="k",
#             # linewidth=0.5,
#             label=f"{g} (μ={mu:.2f}, σ={sd:.2f}, max={s_max:.2f}, min={s_min:.2f}, n={s.size})",
#         )
#         # ax.vlines(s, ymin=0, ymax=max(1, int(0.05 * s.size)), colors="k", alpha=0.35, linewidth=0.8)


#         # ---------------------------------------------------------
#         # 追加: 確信度カーブと閾値の描画 (Confidence Curve & Threshold)
#         # ---------------------------------------------------------
#         ax2 = ax.twinx()

#         # 設定値の取得
#         tau = cfg['confidence']['tau']
#         as_percent = cfg['confidence']['as_percent']
#         add_text = cfg['confidence']['add_text']
#         text_digits = cfg['confidence']['text_digits']
#         clip_max = cfg['confidence']['sigmoid_clip_max']
#         clip_min = cfg['confidence']['sigmoid_clip_min']

#         # 描画用のX軸データを作成 (グラフの表示範囲全体をカバー)
#         x_min, x_max = ax.get_xlim()
#         x_plot = np.linspace(x_min, x_max, 400)

#         # 確信度の計算 (z = (x - thr) / tau)
#         z_plot = (x_plot - threshold) / tau

#         # utils.py内の sigmoid_np を使用
#         p_anom = sigmoid_np(z_plot, sigmoid_clip_max=clip_max, sigmoid_clip_min=clip_min)
#         p_norm = 1.0 - p_anom

#         scale = 100.0 if as_percent else 1.0
#         p_anom_val = p_anom * scale
#         p_norm_val = p_norm * scale

#         # プロット (異常確信度: 赤点線, 正常確信度: 緑点線)
#         ln1 = ax2.plot(x_plot, p_norm_val, color="green", linestyle="--", linewidth=1.5, alpha=0.8, label="Conf(Normal)")
#         ln2 = ax2.plot(x_plot, p_anom_val, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Conf(Anomaly)")

#         # 閾値の縦線 (黒破線)
#         ln3 = ax.axvline(threshold, color="black", linestyle="-.", linewidth=1.5, alpha=0.8, label="Threshold")

#         # 右軸ラベルの設定
#         ylabel_right = "Confidence (%)" if as_percent else "Confidence (0-1)"
#         ax2.set_ylabel(ylabel_right)
#         ax2.set_ylim(0, 105 if as_percent else 1.05)       

#         # 凡例の統合 (左軸のヒストグラム凡例 + 右軸のカーブ凡例 + 閾値)
#         # ax.legend() だとヒストグラムしか出ないので、全部まとめて ax.legend に渡す
#         lines1, labels1 = ax.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         # 閾値線(ln3)は ax に属しているが get_legend_handles_labels で取得されるはず
#         # もし重複防止などが必要ならここで整理するが、基本は結合でOK
#         # ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8, frameon=True)


#         # グリッドは左軸基準でOKだが、右軸は見にくいのでOFFにするか調整
#         ax2.grid(False)




#     # ax.set_title(title)
#     ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
#     ax.legend(loc="best", fontsize=9, frameon=True)
#     ax.grid(alpha=0.3, linestyle=":")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=150, bbox_inches="tight")
#     plt.close(fig)

def plot_hist_all_subclasses_with_confidence(
        out_png: str, 
        scores: np.ndarray, 
        groups: list, 
        cfg,
        threshold,
        xlabel: str = "Anomaly Score", 
        ylabel: str = "Count",
        bins="fd_clamped",
        min_bins: int = 12,
        max_bins: int = 24,
    ):
    ensure_dir(out_png)
    scores = np.asarray(list(scores), dtype=float)
    uniq = sorted(list(set(groups)))

    # ビン幅の決定
    if isinstance(bins, str) and bins == "fd_clamped":
        bins_ = _shared_bins_fd_clamped([scores], min_bins, max_bins)
    else:
        bins_ = bins

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 1. ヒストグラムの描画 (左軸: ax) ---
    for i, g in enumerate(uniq):
        s = np.asarray([sc for sc, gg in zip(scores, groups) if gg == g], dtype=float)
        if s.size == 0: continue
        
        mu = float(np.mean(s))
        sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
        
        ax.hist(
            s,
            bins=20,  # 修正：固定値20ではなく計算されたbins_を使用
            alpha=0.8,
            histtype="stepfilled",
            label=f"{g} (μ={mu:.2f}, σ={sd:.2f}, n={s.size})",
        )

    # --- 2. 確信度曲線と閾値の描画 (右軸: ax2) ---
    # ※ ループの外に出すことで、1回だけ描画するようにします
    conf_cfg = cfg.get('confidence', {})
    if conf_cfg.get('enable', True):
        ax2 = ax.twinx()

        # 設定値の取得
        tau = float(conf_cfg.get('tau', 1.0))
        as_percent = conf_cfg.get('as_percent', True)
        clip_max = conf_cfg.get('sigmoid_clip_max', 100.0)
        clip_min = conf_cfg.get('sigmoid_clip_min', -100.0)

        # 描画用のデータ作成
        x_min, x_max = ax.get_xlim()
        x_plot = np.linspace(x_min, x_max, 400)
        z_plot = (x_plot - threshold) / tau
        p_anom = sigmoid_np(z_plot, sigmoid_clip_max=clip_max, sigmoid_clip_min=clip_min)
        p_norm = 1.0 - p_anom

        scale = 100.0 if as_percent else 1.0
        
        # プロット (後で凡例をまとめるために、戻り値を受け取っておく)
        ax2.plot(x_plot, p_norm * scale, color="green", linestyle="--", linewidth=1.5, alpha=0.8, label="Conf(Normal)")
        ax2.plot(x_plot, p_anom * scale, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Conf(Anomaly)")
        
        # 閾値線
        ax.axvline(threshold, color="black", linestyle="-.", linewidth=1.5, alpha=0.8, label="Threshold")

        # 右軸ラベル
        ax2.set_ylabel("Confidence (%)" if as_percent else "Confidence (0-1)")
        ax2.set_ylim(0, 105 if as_percent else 1.05)
        ax2.grid(False)

    # --- 3. 凡例の統合処理 ---
    # 左軸(ax)と右軸(ax2)のラベル情報をすべて回収する
    h1, l1 = ax.get_legend_handles_labels()
    try:
        h2, l2 = ax2.get_legend_handles_labels()
    except NameError: # confidenceが無効でax2が作られなかった場合
        h2, l2 = [], []

    # 全てを統合して1つの凡例ボックスに表示
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9, frameon=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle=":")
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_hist_train_val_test_ok(out_png, train_scores, val_scores, test_ok_scores, bins_mode="auto"):
    """
    Train / Val / Test の各OKデータの異常スコア分布を1枚に重ねて描画する。
    """
    ensure_dir(out_png)

    # NaN/Infの除外処理（安全策）
    def clean_score(s):
        s = np.asarray(s)
        return s[np.isfinite(s)]

    t_s = clean_score(train_scores)
    v_s = clean_score(val_scores)
    te_ok_s = clean_score(test_ok_scores)

    # 3系列を合わせた全体範囲から共有binを作成
    all_scores = np.concatenate([t_s, v_s, te_ok_s])
    if all_scores.size == 0:
        return

    # 共有ビン：全データの最小・最大をカバーするように設定
    mn, mx = all_scores.min(), all_scores.max()
    shared_bins = np.linspace(mn, mx, 50)  # 50分割

    fig, ax = plt.subplots(figsize=(10, 6))

    # 各系列の描画（alphaで透明度を指定して重ねる）
    # 理由：Count（度数）をデフォルトとし、サンプル数の違いを直感的に把握可能にする
    ax.hist(t_s, bins=shared_bins, alpha=0.7, 
            label=f"train_ok (μ={float(np.mean(t_s)):.2f}, σ={float(np.std(t_s, ddof=1)) if t_s.size > 1 else 0.0:.2f}, max={max(t_s):.2f}, min={min(t_s):.2f}, n={t_s.size})", 
            color="tab:blue", edgecolor="white")
    ax.hist(v_s, bins=shared_bins, alpha=0.7, 
            label=f"val_ok (μ={float(np.mean(v_s)):.2f}, σ={float(np.std(v_s, ddof=1)) if v_s.size > 1 else 0.0:.2f}, max={max(v_s):.2f}, min={min(v_s):.2f}, n={v_s.size})", 
            color="tab:green", edgecolor="white")
    ax.hist(te_ok_s, bins=shared_bins, alpha=0.7, 
            label=f"test_ok (μ={float(np.mean(te_ok_s)):.2f}, σ={float(np.std(te_ok_s, ddof=1)) if te_ok_s.size > 1 else 0.0:.2f}, max={max(te_ok_s):.2f}, min={min(te_ok_s):.2f}, n={te_ok_s.size})", 
            color="tab:orange", edgecolor="white")

    ax.set_title("Anomaly Score Histogram (train_ok / val_ok / test_ok)")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)




# ===============================
# Confidence helper (NOT calibrated probability)
# ===============================
def sigmoid_np(x: np.ndarray, sigmoid_clip_max=100.0, sigmoid_clip_min=-100.0) -> np.ndarray:
    """Numerically-stable sigmoid for numpy arrays.

    Note:
        This returns a monotonic score in (0, 1) but it is NOT a calibrated probability.
        In this project, we use it only to map the margin to a [0,1] "confidence-like" value.
    """
    x = np.asarray(x, dtype=float)
    # Clip to avoid overflow in exp for large magnitude values.
    x = np.clip(x, sigmoid_clip_min, sigmoid_clip_max)
    return 1.0 / (1.0 + np.exp(-x))