import os
import math
import numpy as np
import torch
from torch import optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml

from src.datasets import create_loader
from src.model_ae import Conv2dAE
from src.calc_mahalanobis import (vectorize_residual,cov_to_precision)



# ---------------------------------------------------------
# Mahalanobis 用の統計量（平均/共分散/精度行列）推定
#
# 用語:
# - residual（残差）: diff = x - x_hat（入力xと再構成x_hatの差）
# - vectorize（ベクトル化）: 残差テンソルを [N, D] の2次元行列に変換する処理
#   - N: サンプル数（バッチや時間窓など、vectorize_residual の実装に依存）
#   - D: 特徴次元（周波数方向の集約など、mode に依存）
# - covariance（共分散行列）: D×D 行列。特徴のばらつきと相関を表します。
# - precision（精度行列）: 共分散の（擬似）逆行列。Mahalanobis 距離で使用します。
#
# 返り値:
# - torch.save 可能な dict（mean/cov/precision などをCPUテンソルとして格納）
# 例外:
# - 残差ベクトルが 2本未満だと共分散が推定できないため RuntimeError を投げます。
# ---------------------------------------------------------

@torch.no_grad()
def estimate_mahala_pack(
    model: torch.nn.Module,
    loaders: list,
    device: torch.device,
    vectorize: str = "freq",
    eps: float = 1.0e-6,
    use_pinv: bool = True,
) -> dict:
    
    # 推定中は学習を行わないため eval モードに切り替えます。
    # - Dropout / BatchNorm などがある場合、推論時の挙動になります。
    model.eval()

    # 残差ベクトル [N, D] の平均・共分散を、全データに対して集計するための累積変数です。
    # sum_vec: 各次元の総和（D次元）
    # sum_outer: v^T v の総和（D×D）
    # n_total: 残差ベクトル本数（Nの総計）
    sum_vec = None
    sum_outer = None
    n_total = 0
    last_meta = None

    # loaders は DataLoader のリストを想定します。
    # - selective_mahala では複数ドメイン/複数globを渡すため、リストになっています。
    for loader in loaders:

        # DataLoader から 1バッチ取り出します。
        # このデータ構造 (x, _, _) は create_loader の実装に依存しますが、
        # 少なくとも x はモデル入力となるテンソルです。
        for (x, _, _) in tqdm(loader, desc="[estimate mahala stats]", leave=False):
            
            # x: 入力テンソル。Conv2dAE(in_ch=1) のため、一般には [B, 1, H, W] 形状を想定します。
            # - H/W の意味（周波数×時間など）は Dataset 実装に依存します。
            # device へ転送します（CPU/GPU）。
            x = x.to(device)
            
            # x_hat: オートエンコーダによる再構成結果（入力と同shapeを期待）。
            # 2つ目の戻り値（_）は潜在表現などの補助情報の可能性があります（このファイルでは未使用）。
            x_hat, _ = model(x)

            # diff: 再構成残差（residual）。異常検知ではこの残差が大きいほど異常らしい、という仮定を置きます。
            diff = (x - x_hat)

            # 残差テンソル diff を 2次元の特徴行列に変換します。
            # vecs: [N, D]（N=サンプル本数、D=特徴次元）
            # meta: ベクトル化時の軸情報など。後でスコアを元の軸へ戻す用途で使う想定です。
            vecs, meta = vectorize_residual(diff, mode=vectorize)  # [N,D]
            last_meta = meta

            # 統計推定は数値誤差が溜まりやすいので float64 へ昇格して集計します。
            # - ただし最終的に保存する mean/cov/precision は float32 に戻しています。
            v = vecs.detach().to(device=device, dtype=torch.float64)

            # 1回目だけ、特徴次元 D を確定し、累積用テンソルを確保します。
            # - D は vectorize_residual の出力次元に依存します。
            if sum_vec is None:
                D = v.size(1)
                sum_vec = torch.zeros(D, device=device, dtype=torch.float64)
                sum_outer = torch.zeros(D, D, device=device, dtype=torch.float64)

            # バッチ内のベクトルを集計します（平均との差や共分散を後で一括計算する方式）。
            sum_vec += v.sum(dim=0)

            # 外積の総和を加算します。
            # - v.transpose(0, 1) @ v は (D×N)@(N×D) = D×D になり、D^2 のメモリ/計算量が必要です。            
            sum_outer += v.transpose(0, 1) @ v
            n_total += int(v.size(0))

    # 共分散はサンプル数が 2 以上でないと定義できないため、最低本数をチェックします。
    if n_total <= 1:
        raise RuntimeError(f"Not enough residual vectors to estimate covariance (n={n_total}).")

    # 平均ベクトル μ を計算します。
    mu = (sum_vec / float(n_total)).to(dtype=torch.float32)

    # 不偏共分散（n-1 で割る）を計算します。
    # cov = (Σ v v^T - n μ μ^T) / (n - 1)    
    cov = (sum_outer - float(n_total) * (mu[:, None].double() @ mu[None, :].double())) / float(n_total - 1)
    cov = cov.to(dtype=torch.float32)

    # 共分散行列から precision（逆共分散）を作ります。
    # - eps: 対角に加える微小値など、数値安定化に使う想定（cov_to_precision の実装に依存）。
    # - use_pinv=True の場合、特異行列でも擬似逆行列で対応します。
    precision = cov_to_precision(cov, eps=eps, use_pinv=use_pinv)

    # 推定結果を 1つの dict にまとめます。
    # - CPUへ移して保存することで、環境差（GPU有無）に左右されにくくします。
    # - mean と mu は同じものの別名です（既存コード互換のため）。
    mahala_pack = {
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
    return mahala_pack



def get_mahala_stats(cfg):

    # 実行デバイスを決定します。
    # - CUDA が利用可能なら GPU、それ以外は CPU を選択します。
    # - 以降、モデル・入力テンソル・checkpoint 読み込み map_location にもこの device を使います。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 正常データ（OK）だけを読む DataLoader を作ります。
    # - cfg["data"]["train_ok_glob"]: 学習用OKデータの glob（例: "/path/to/train_ok/**/*.npy" など）
    # - label=0: OK（正常）ラベルとして扱う想定（create_loader 実装に依存）
    # - shuffle=True, drop_last=True:
    #   - ここで作った train_loader は、この関数内では主に warm-up forward のために 1バッチ取り出します。
    #   - shuffle=True なので、取り出す 1バッチは毎回変わる可能性があります（再現性が必要なら注意）。
    train_loader, _ = create_loader(
        cfg["data"]["train_ok_glob"], label=0, cfg=cfg, 
        shuffle=True, drop_last=True
        )

    # モデル種別のガード（設定ミスの早期検知）。
    # - cfg["model"]["type"] が "conv2d_ae" であることを前提に Conv2dAE を生成します。
    assert cfg["model"]["type"] == "conv2d_ae"

    # AE（Conv2dAE）を初期化します。
    # - in_ch=1: 入力チャネル数（例：スペクトログラムを 1ch として扱う設計）
    # - cfg["model"]["bottleneck_dim"]: ボトルネック次元（圧縮表現の次元）
    # - .to(device): GPU/CPU にモデルを配置
    model = Conv2dAE(in_ch=1, bottleneck_dim=cfg["model"]["bottleneck_dim"]).to(device)

    # warm-up forward 用に、train_loader から 1バッチ取り出します。
    # - create_loader の戻りが (x, _, _) 形式である前提（2つ目以降は未使用）。
    # - 例外リスク:
    #   - glob が空 / データが 0 件の場合、next(iter(train_loader)) が StopIteration になる可能性があります。
    x0, _, _ = next(iter(train_loader))
    
    # 入力も device に転送します。
    x0 = x0.to(device)

    # warm-up forward を行います（学習はしない）。
    # 目的（代表例）:
    # - Lazy 系レイヤ等で「最初の forward で shape が確定してパラメータが初期化される」場合の準備
    # - 以降の load_state_dict を安全に通すための下準備になり得ます（モデル実装に依存）。
    with torch.no_grad():
        _ = model(x0)

    # 念のため再度 device に載せています（通常は model(...) の時点で不要なことが多い）。
    model.to(device)

    # checkpoint パスを解決します。
    # 優先順位:
    # 1) cfg["filenames"]["checkpoint_best_full_path"]（フルパスが設定されている場合）
    # 2) os.path.join(cfg["output_dir"], cfg["filenames"]["checkpoint_best"])（出力ディレクトリ + ファイル名
    try:
        ckpt_path = cfg["filenames"]["checkpoint_best_full_path"]
    except:
        ckpt_path = os.path.join(cfg["output_dir"], cfg["filenames"]["checkpoint_best"])
    
    # checkpoint が存在しない場合は即エラー（以降の torch.load が失敗するため）。
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # checkpoint を読み込み、モデル重みを復元します。
    # - state の形式は「state["model"] に state_dict が入っている」前提です。
    # - map_location=device により、GPU/CPU 環境差があっても読み込めるようにしています。
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    
    # 推論モードへ（Dropout/BatchNorm 等がある場合に挙動が推論向けになります）。
    model.eval()


    # Mahalanobis 統計推定を実行するかどうか。
    # - cfg["mahala"]["enabled"] が True の場合のみ、統計（平均/共分散/precision）を推定して保存します。
    if cfg["mahala"]["enabled"]:

        # 統計推定に使う OK データの glob を集約します。
        # - cfg["mahala"]["data"]["train_ok_glob"]: 統計推定用の train OK glob（リスト想定）
        # - cfg["mahala"]["data"]["val_ok_glob"]: 統計推定用の val OK glob（リスト想定）
        # ここでは train + val をまとめています（「正常分布」を厚くする意図と考えられます）。
        raw_globs = cfg["mahala"]["data"]["train_ok_glob"] + cfg["mahala"]["data"]["val_ok_glob"]
        
        # 空文字・None・空白だけの要素を除去します（設定ミス耐性）。
        ok_globs = [g for g in raw_globs if g and str(g).strip() != ""]
        
        # glob が 0 件なら統計推定できないためエラー。
        if len(ok_globs) == 0:
            raise RuntimeError("[mahala] enabled=True, but data paths are missing.")
        
        # glob ごとに DataLoader を作り、estimate_mahala_pack に渡すためのリストにします。
        # - shuffle=False: 統計推定なのでシャッフル不要（再現性・デバッグ性も上がる）
        # - drop_last=False: 端数バッチも含めて全サンプルを使う意図
        loaders_for_calc_mahala_stats = []
        for g in ok_globs:
            ld, _ = create_loader(g, label=0, cfg=cfg, shuffle=False, drop_last=False)
            loaders_for_calc_mahala_stats.append(ld)
        
        # Mahalanobis 統計（mean/cov/precision 等）を推定します。
        # 重要: 推定対象のテンソルは estimate_mahala_pack 内で決まります。
        # - AE の再構成 x_hat を計算し、残差 diff = x - x_hat を作ります。
        # - diff を vectorize_residual(diff, mode=cfg["mahala"]["vectorize"]) で [N, D] に変換し、
        #   その [N, D] を全 loader 分まとめて平均/共分散/precision を推定します。
        # 参照する cfg キー:
        # - cfg["mahala"]["vectorize"]: 残差のベクトル化モード（例: "freq" 等、実装依存）
        # - cfg["mahala"]["eps"]: precision 推定時の数値安定化パラメータ（実装依存）
        # - cfg["mahala"]["use_pinv"]: 特異行列対策として擬似逆を使うかどうか（実装依存）
        mahala_pack = estimate_mahala_pack(
            model=model,
            loaders=loaders_for_calc_mahala_stats,
            device=device,
            vectorize=cfg["mahala"]["vectorize"],
            eps=float(cfg["mahala"]["eps"]),
            use_pinv=cfg["mahala"]["use_pinv"],
        )

        # 推定した統計量を保存します（torch.load で再利用する想定）。
        # - cfg["mahala"]["stats_path"]: 保存先パス
        # 例外リスク:
        # - 親ディレクトリが存在しない場合、torch.save が失敗します。
        torch.save(mahala_pack, cfg["mahala"]["stats_path"])

    return cfg["mahala"]["stats_path"]

if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mahala_stats_ckpt = get_mahala_stats(cfg)
    print('Statistical data for Mahalanobis distance calculation is stored in the file below.')
    print(mahala_stats_ckpt)