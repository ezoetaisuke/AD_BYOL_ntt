
import torch

# ============================================================
# 本モジュールの目的（ユースケースの全体像）
# ------------------------------------------------------------
# - 典型例: AutoEncoder(AE) の「入力特徴量」と「再構成特徴量」の差分(residual = diff)を取り、
#   その residual をベクトル化 → 共分散Σ/平均μを用意 → Mahalanobis distance(MD) を計算して
#   異常度スコアとして利用する。
# - Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ)
#   ここで Σ^{-1} を precision（精度行列）と呼ぶ。
# - 重要前提:
#   * shape が想定と違うと即エラーになる（特に [B,C,F,T] / [N,D]）。
#   * device(CPU/GPU) の不一致は PyTorch では実行時エラーになりやすい。
#   * 共分散は特異（rank不足）になりやすく、逆行列(inv)が壊れることがあるため
#     eps による対角加算や pinv（擬似逆行列）を使う設計になっている。
# ============================================================

def load_mahala_pack(path: str, device: torch.device) -> dict:

    # --------------------------------------------------------
    # 目的:
    #   事前に保存しておいた Mahalanobis 用パラメータ（平均 mean と precision）を読み込む。
    #
    # 入力:
    #   path   : torch.save / torch.load で保存されたファイルパス（中身はdict想定）
    #   device : 推論/計算に使いたい device（例: torch.device("cuda")）
    #
    # 処理:
    #   1) まず CPU にロード（map_location="cpu"）
    #      - GPUがない環境でも読める
    #      - その後、必要なテンソルだけ指定deviceへ移す
    #   2) pack から "precision" と "mean" を取り出す
    #      - mean は "mean" が無ければ "mu" を代替キーとして探す
    #   3) precision/mean を float32 へ統一し、device に転送
    #
    # 出力:
    #   pack : dict
    #     - pack["precision"]: torch.Tensor [D,D], dtype=float32, device=device
    #     - pack["mean"]     : torch.Tensor [D]  , dtype=float32, device=device
    #     - それ以外のキーが入っていた場合はそのまま残る（ただし転送されない点に注意）
    #
    # 例外になりやすい条件:
    #   - "precision" が無い / mean("mean" or "mu") が無い → KeyError
    #   - 保存データが想定と違う型（dict以外など） → torch.load後の利用で例外になる可能性
    # --------------------------------------------------------

    pack = torch.load(path, map_location="cpu")
    precision = pack.get("precision", None)
    mean = pack.get("mean", pack.get("mu", None))
    if precision is None or mean is None:
        raise KeyError(f"Mahalanobis pack missing keys. need precision+mean: {path}")

    # precision/mean は後続のMD計算で使うため、計算用deviceへ移し dtype を float32 に揃える
    # （float32は推論で標準的。数値安定性が必要ならfloat64運用も検討対象）
    pack["precision"] = precision.to(device=device, dtype=torch.float32)
    pack["mean"] = mean.to(device=device, dtype=torch.float32)
    return pack

def vectorize_residual(diff: torch.Tensor, mode: str = "freq"):

    # --------------------------------------------------------
    # 目的:
    #   AE residual（diff）を「共分散推定」や「Mahalanobis距離計算」に使える 2次元行列 [N,D]
    #   に変換する（= ベクトル化）。
    #
    # 背景:
    #   Mahalanobis距離はベクトル x∈R^D に対して定義されるため、
    #   画像/スペクトログラムのような4次元テンソル [B,C,F,T] を flatten してベクトルに落とす必要がある。
    #
    # 入力:
    #   diff : torch.Tensor
    #     想定shapeは [B, C, F, T]
    #       B: batch（サンプル数）
    #       C: channel（このプロジェクトでは 1 を想定することが多い）
    #       F: frequency bin（周波数方向）
    #       T: time frame（時間方向）
    #   mode : str
    #     現在は "freq" のみ対応（それ以外は ValueError）
    #
    # 出力:
    #   vecs : torch.Tensor [N,D]
    #     mode="freq" のとき:
    #       N = B*T（各時刻フレームを1サンプルとして扱う）
    #       D = C*F（各フレームの周波数方向（+ch）を1本の特徴ベクトルにする）
    #   meta : dict
    #     後段で [N,D] を元の構造に戻したり、解釈を補助するためのメタ情報
    #
    # 例外になりやすい条件:
    #   - diff が Tensor でない → TypeError
    #   - mode="freq" なのに diff.dim()!=4 → ValueError（shape不整合）
    # --------------------------------------------------------

    if not torch.is_tensor(diff):
        raise TypeError("diff must be a torch.Tensor")

    # mode は文字列化して小文字へ正規化（"FREQ" 等の入力揺れを吸収）
    mode = str(mode).lower()

    if mode == "freq":

        # expected: [B, C, F, T]
        # mode="freq" のベクトル化方針:
        #   1) [B,C,F,T] を [B,T,C,F] に並び替え（timeを2番目に）
        #   2) 1フレーム分 [C,F] を flatten して長さ D=C*F のベクトルにする
        #   3) B と T をまとめて N=B*T 本のベクトル行列 [N,D] を作る
        if diff.dim() != 4:
            raise ValueError(f"diff must be 4D [B,C,F,T] for mode='freq', got {tuple(diff.shape)}")
        B, C, F, T = diff.shape

        # permute した後はメモリ上の並びが変わるため contiguous() を入れて view 可能な形にしている
        # （contiguous無しで view するとエラーになることがある）
        vecs = diff.permute(0, 3, 1, 2).contiguous().view(B * T, C * F)  # [B*T, C*F]
        meta = {"B": B, "T": T, "D": C * F, "mode": mode}
        return vecs, meta

    # 現状は "freq" のみ。想定外モードは即エラーにして早期にバグを発見できるようにしている。
    raise ValueError(f"Unsupported vectorize mode: {mode}. Use 'freq'.")


# def update_running_mean_cov(
#         mean=None,
#         M2=None,
#         n: int = 0,
#         x: torch.Tensor = None,
# ):
#     """
#     Online update of mean and covariance accumulator (Welford) for vectors x.

#     We maintain:
#       - mean: running mean [D]
#       - M2  : sum of outer products for covariance [D,D]
#       - n   : number of samples seen

#     After processing all samples, covariance is:
#       cov = M2 / max(1, n-1)

#     Parameters
#     ----------
#     mean, M2, n : current state
#     x : torch.Tensor
#         [N, D] batch of vectors

#     Returns
#     -------
#     mean, M2, n
#     """
#     if x is None:
#         raise ValueError("x must be provided")

#     if x.dim() != 2:
#         raise ValueError(f"x must be 2D [N,D], got {tuple(x.shape)}")

#     # initialize
#     if mean is None:
#         D = x.size(1)
#         mean = torch.zeros(D, device=x.device, dtype=torch.float64)
#     if M2 is None:
#         D = x.size(1)
#         M2 = torch.zeros(D, D, device=x.device, dtype=torch.float64)

#     x64 = x.to(dtype=torch.float64)
#     for i in range(x64.size(0)):
#         n1 = n + 1
#         delta = x64[i] - mean
#         mean = mean + delta / n1
#         delta2 = x64[i] - mean
#         # rank-1 update
#         M2 = M2 + torch.outer(delta, delta2)
#         n = n1
#     return mean, M2, n


# def finalize_mean_cov(mean: torch.Tensor, M2: torch.Tensor, n: int):
#     """
#     Finalize covariance from Welford accumulator.
#     """
#     if n <= 1:
#         raise ValueError(f"Need at least 2 samples to estimate covariance, got n={n}")
#     cov = M2 / (n - 1)
#     return mean, cov


def cov_to_precision(cov: torch.Tensor, eps: float = 1.0e-6, use_pinv: bool = True):
    
    # --------------------------------------------------------
    # 目的:
    #   共分散行列 Σ（cov）から precision 行列 P = Σ^{-1} を作る。
    #
    # 入力:
    #   cov : torch.Tensor [D,D]
    #     - D次元ベクトルの共分散行列
    #     - 共分散推定が不十分だと rank が落ちて「特異行列」になりやすい
    #   eps : float
    #     - 数値安定化のための対角成分への加算量（Tikhonov正則化 / ridge 的な役割）
    #     - Σ_reg = Σ + eps * I としてから逆行列を取る
    #   use_pinv : bool
    #     - True: torch.linalg.pinv（擬似逆）を使う
    #       * Σが特異でも計算が成立しやすい（ただし計算コストは高め）
    #     - False: torch.linalg.inv（通常の逆）を使う
    #       * Σ_reg が非正則だと例外になる可能性が高い
    #
    # 出力:
    #   precision : torch.Tensor [D,D], dtype=float32
    #
    # 例外になりやすい条件:
    #   - cov が2次元正方行列でない → ValueError
    #   - use_pinv=False かつ Σ_reg が特異 → torch.linalg.inv が例外を投げる可能性
    # --------------------------------------------------------

    if cov.dim() != 2 or cov.size(0) != cov.size(1):
        raise ValueError(f"cov must be square 2D, got {tuple(cov.shape)}")

    D = cov.size(0)

    # eye は cov と同じ device / dtype で生成（混在すると演算でエラーになりうる）
    eye = torch.eye(D, device=cov.device, dtype=cov.dtype)

    # eps を対角に足して、逆行列計算が破綻しにくい形にする
    cov_reg = cov + float(eps) * eye

    if use_pinv:
        # pinv は特異行列でも計算しやすいが、inv より計算負荷が高いことが多い
        prec = torch.linalg.pinv(cov_reg)
    else:
        # inv は高速な場合があるが、非正則だと失敗しやすい
        prec = torch.linalg.inv(cov_reg)

    # 返り値は float32 へ統一（以降のMD計算を float32 前提で動かす想定）
    return prec.to(dtype=torch.float32)


def mahalanobis_distance(x: torch.Tensor, mu: torch.Tensor, precision: torch.Tensor, sqrt: bool = True):
    # --------------------------------------------------------
    # 目的:
    #   ベクトル集合 x（[N,D]）に対して Mahalanobis 距離（または二乗距離）を一括計算する。
    #
    # 入力:
    #   x         : torch.Tensor [N,D]
    #     - N: サンプル数（例: B*T）
    #     - D: 特徴次元（例: C*F）
    #   mu        : torch.Tensor [D]
    #     - 平均ベクトル（共分散推定時の平均との差分を取る基準）
    #   precision : torch.Tensor [D,D]
    #     - precision 行列 P = Σ^{-1}
    #   sqrt      : bool
    #     - True の場合: sqrt(d^2) を返す（一般的な「距離」）
    #     - Falseの場合: d^2 を返す（スコアとしてはこちらの方が扱いやすいこともある）
    #
    # 計算式:
    #   xc = x - mu
    #   d^2 = diag( xc @ P @ xc^T )
    #   実装上は (xc @ P) と xc の要素積をとって行方向に sum することで対角成分だけを効率計算している。
    #
    # dtypeについて:
    #   - ここでは float32 に揃えて計算している（速度・標準的運用のため）
    #   - ただし D が大きい/分布が尖る場合は float64 の方が安定することもある（改善提案参照）
    #
    # 例外になりやすい条件:
    #   - x が2次元でない / mu が1次元でない / precision が2次元でない → ValueError
    #   - x と mu / precision の device が異なる（CPU vs GPU 等）と演算で RuntimeError になりうる
    #   - shapeは dim チェックのみで、D の一致までは検証していない（D不一致は行列積でエラーになりうる）
    # --------------------------------------------------------
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [N,D], got {tuple(x.shape)}")
    if mu.dim() != 1:
        raise ValueError(f"mu must be 1D [D], got {tuple(mu.shape)}")
    if precision.dim() != 2:
        raise ValueError(f"precision must be 2D [D,D], got {tuple(precision.shape)}")

    # 中心化（平均との差分）
    xc = x.to(dtype=torch.float32) - mu.to(dtype=torch.float32)

    # precision は [D,D] を想定
    P = precision.to(dtype=torch.float32)

    # (x P) * x row-wise
    # left: [N,D] = [N,D] @ [D,D]
    left = xc @ P

    # d2: [N]
    # - left * xc は要素積で [N,D]
    # - sum(dim=1) で各サンプルの (x-μ)^T P (x-μ) を得る
    # - 数値誤差でごく僅かに負になる場合があるため clamp_min(0.0) で下駄を履かせる
    d2 = (left * xc).sum(dim=1).clamp_min(0.0)

    # sqrt=True なら距離、False なら二乗距離（スコア）を返す
    return torch.sqrt(d2) if sqrt else d2




