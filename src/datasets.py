# =============================================================================
# datasets.py
# -----------------------------------------------------------------------------
# このファイルの役割：
#   - 音響（wav）ファイルを列挙し、読み込み→前処理→特徴抽出を行い、PyTorch Dataset として提供する
#   - その Dataset を DataLoader に包んで、学習/評価ループからバッチとして取り出せるようにする
#
# 本プロジェクト文脈（AEによる異常検知）での想定：
#   - 入力は固定長の音声（同一サンプル数）であることを前提にしている（expect_same_length=true の場合は厳密チェック）
#   - 特徴量は FeatureExtractor が生成（例：スペクトログラム等）。shape は feat=[1, F, T] を想定（コード内コメントより）
#
# 注意：FeatureExtractor の中身は別ファイル定義（.features）。このファイル単体では詳細は断定しない。
# =============================================================================

import glob
import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 特徴抽出器（別ファイル定義）
#   - 生波形 y（1次元）を受け取り、学習に使う特徴量（numpy array）を返す想定
#   - 具体的な周波数ビン数Fやフレーム数Tは FeatureExtractor の実装と cfg に依存する
from .features import FeatureExtractor

# Dataset: 1ファイル=1サンプルとして扱う音声Dataset
#   - __len__ はファイル数
#   - __getitem__ は (x, label, path) を返す
#     - x: 特徴量テンソル（想定 [1, F, T]）
#     - label: 0=OK, 1=NG のクラスラベル（スカラーTensor）
#     - path: 元ファイルパス（デバッグ/集計用）
class AudioDataset(Dataset):

    # 初期化：ファイル列挙・設定読み込み・基準長(ref_len)の決定
    #   - file_globs: glob パターン（str or list[str]）
    #   - label: このDatasetに付与する固定ラベル（OK=0, NG=1）
    #   - cfg: 設定辞書（音声設定/ローダ設定/特徴設定など）
    def __init__(self, file_globs, label, cfg):

        # 対象ファイル一覧を格納するリスト（globで列挙して詰める）
        self.files = []

        # file_globs の型に応じて glob 展開
        #   - str: 単一パターン
        #   - list: 複数パターン（すべて展開して結合）
        #   - それ以外: 設定ミスなので例外
        if isinstance(file_globs, str):
            self.files.extend(glob.glob(file_globs))
        elif isinstance(file_globs, list):
            for g in file_globs:
                self.files.extend(glob.glob(g))
        else:
            raise ValueError("file_globs must be str or list of str")


        # ファイル順序を固定（同じ入力に対して順番がブレないようにする）
        # ※ DataLoader の shuffle=True を使う場合は、ここでの順序は初期順としてのみ意味がある
        self.files = sorted(self.files)
        self.skipped_files = []
        
        # このDatasetが返す label は全サンプルで固定（OK用Dataset/NG用Datasetを分けて作る設計）
        self.label = label
        self.cfg = cfg

        # cfg 依存の設定を読み込む（音声の前処理に関わる重要パラメータ）
        #   - target_sr: 目標サンプリング周波数
        #   - resample_if_mismatch: sr不一致時にリサンプルするか（Falseなら即エラー）
        #   - to_mono: Trueなら多ch→モノラル（平均）にする
        self.sr = cfg["audio"]["target_sr"]
        byol_mode = str(cfg.get("byol", {}).get("mode", "pretrained")).lower()
        # pretrained時のみ入力制約に合わせる（要件）。scratch時は既存挙動を優先する。
        self.resample = bool(cfg["audio"]["resample_if_mismatch"]) if byol_mode == "pretrained" else False
        self.to_mono = cfg["audio"]["to_mono"]

        # 特徴抽出器を生成（重い初期化があれば __getitem__ ごとに作らないためここで作る）
        #   - __getitem__ では self.feature(y) を呼んで numpy 特徴量を得る
        self.feature = FeatureExtractor(cfg)

        # glob の結果が空なら学習/評価が成立しないため即エラー
        if len(self.files) == 0:
            raise RuntimeError("No files found for pattern(s): " + str(file_globs))
        
        # 参照用の基準（長さ・sr）を最初のファイルで決める
        #   - soundfile.read(..., always_2d=True) なので y0 は [num_samples, channels] の2次元配列
        #   - ここで決めた ref_len を後続ファイルにも要求する（expect_same_length=true の場合）
        valid_files = []
        for p in self.files:
            try:
                sf.info(p)
                valid_files.append(p)
            except Exception as exc:
                self.skipped_files.append((p, f"read_error:{exc}"))
        self.files = valid_files
        if not self.files:
            raise RuntimeError("All wav files are unreadable for pattern(s): " + str(file_globs))

        y0, sr0 = sf.read(self.files[0], dtype="float32", always_2d=True)
        
        # モノラル化の方針：
        #   - to_mono=True: 全チャンネル平均で 1ch 化（shape: [num_samples]）
        #   - to_mono=False: 先頭チャンネルのみ使用（shape: [num_samples]）
        #     ※ 全チャンネルを保持する設計ではない点に注意（仕様）
        y0 = y0.mean(axis=1) if self.to_mono else y0[:,0]

        # サンプリング周波数が目標と違う場合の扱い：
        #   - resample_if_mismatch=False ならここで即エラー（混在srのデータセットを防ぐ）
        if sr0 != self.sr and not self.resample:
            raise RuntimeError(f"Sampling rate mismatch: {sr0} != {self.sr} and resample disabled.")
        
        # resample_if_mismatch=True の場合は librosa.resample で target_sr に変換して統一する
        #   - リサンプルは計算コストがあるため、データ数が多いとI/Oよりここがボトルネックになることがある
        if sr0 != self.sr and self.resample:
            y0 = librosa.resample(y0, orig_sr=sr0, target_sr=self.sr)
        
        # 参照長(ref_len)：基準ファイルのサンプル数
        #   - __getitem__ で len(y)==ref_len をチェックし、固定長前提を守る（設定により）
        self.ref_len = len(y0)
        self.fixed_seconds = self.cfg.get("audio", {}).get("fixed_seconds", None)
        byol_cfg = self.cfg.get("byol", {})
        prep_cfg = byol_cfg.get("pretrained_input", {})
        if self.fixed_seconds is None and byol_mode == "pretrained":
            # 既存実装への最小差分として sec 指定を length 指定へマッピング
            target_len = prep_cfg.get("target_len", None)
            if target_len is not None:
                self.fixed_length = int(target_len)
            else:
                self.fixed_length = None
        else:
            self.fixed_length = None

        self.min_length = None
        if byol_mode == "pretrained":
            self.min_length = int(prep_cfg.get("min_len", 0)) if prep_cfg.get("min_len", None) is not None else None

        self.pad_mode = str(prep_cfg.get("pad_mode", "zero")).lower()
        self.trim_mode = str(prep_cfg.get("trim_mode", "center")).lower()

        if self.fixed_seconds is not None:
            self.fixed_length = int(round(float(self.fixed_seconds) * self.sr))

    def __len__(self):
        return len(self.files)

    # idx番目のファイルを読み込み、前処理→特徴抽出して返す
    # 返り値：
    #   - x: torch.Tensor（特徴量。想定 shape=[1,F,T]）
    #   - label: torch.Tensor（スカラー。dtype=torch.long。全サンプル固定）
    #   - path: str（元ファイルパス）
    # 例外が起きる主な条件：
    #   - ファイル読み込み失敗（破損wav等）
    #   - sr不一致で resample 無効
    #   - 固定長前提(expect_same_length)なのに長さが違う
    def __getitem__(self, idx):

        # このサンプルに対応する音声ファイルパスを取得
        # ※ idx の範囲外アクセスは Dataset の一般的な契約外（呼び出し側のバグ）
        path = self.files[idx]

        # 音声読み込み
        #   - y: np.ndarray shape=[num_samples, channels]（always_2d=True）
        #   - sr: int（読み込んだ音声のサンプリング周波数）
        y, sr = sf.read(path, dtype="float32", always_2d=True)

        # モノラル化（__init__ と同じ方針）：to_mono=Trueなら平均、Falseなら先頭ch
        y = y.mean(axis=1) if self.to_mono else y[:,0]

        # sr 不一致時の処理：resample 設定に従って変換するか、エラーにするかを分岐
        if sr != self.sr:
            if self.resample:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            else:
                raise RuntimeError(f"Sampling rate mismatch in {path}")

        # pretrained時のみ、min_lenを満たすようにpadする
        if self.min_length is not None and len(y) < self.min_length:
            pad = self.min_length - len(y)
            if self.pad_mode == "repeat" and len(y) > 0:
                rep = int(np.ceil(self.min_length / len(y)))
                y = np.tile(y, rep)[:self.min_length]
            else:
                y = np.pad(y, (0, pad), mode="constant")

        # BYOL-A 事前学習モデルの制約に合わせるため、必要なら固定秒数へゼロ埋め/切り出し
        if self.fixed_length is not None:
            if len(y) < self.fixed_length:
                pad = self.fixed_length - len(y)
                y = np.pad(y, (0, pad), mode="constant")
            elif len(y) > self.fixed_length:
                start = 0 if self.trim_mode == "left" else (len(y) - self.fixed_length) // 2
                y = y[start:start + self.fixed_length]

        # 固定長チェック：
        #   - fixed_seconds を使う場合は上で長さを揃える
        if self.fixed_length is None and len(y) != self.ref_len and self.cfg["audio"]["expect_same_length"]:
            raise RuntimeError(f"Length mismatch in {path}: {len(y)} != {self.ref_len}")

        # 特徴抽出：生波形 y（1次元）→ 特徴量 feat（numpy）
        #   - ここでの shape はコメントより feat=[1, F, T] を想定
        #   - F（周波数方向）や T（時間フレーム）は FeatureExtractor と cfg に依存（このファイル単体では断定不可）
        feat = self.feature(y)  # [1, F, T] float32

        # numpy → torch.Tensor へ変換
        #   - dtype は feat 側に依存（通常 float32 を想定）
        #   - DataLoader でバッチ化されると x は [B, 1, F, T] になる想定        
        x = torch.from_numpy(feat)  # tensor [1, F, T]

        # ラベルはスカラーTensor（0 or 1）
        #   - DataLoader でまとめられると shape=[B] の long tensor になるのが一般的
        label = torch.tensor(self.label, dtype=torch.long)

        # 戻り値の意味：
        #   - x: 入力特徴量（モデルへ入力）
        #   - label: 正解ラベル（評価指標/混同行列で使用）
        #   - path: 後段でCSV出力やサブクラス集計（親フォルダ名抽出）に使える
        return x, label, path


# DataLoader 生成ヘルパー
#   - glob_patterns: AudioDataset に渡すファイルパターン（str or list[str]）
#   - label: Dataset 全体に付与する固定ラベル（OK=0, NG=1）
#   - cfg: 設定（batch_size や num_workers など Loader 設定もここから読む）
#   - shuffle: True にすると各epochでサンプル順をランダム化（学習用）
#   - drop_last: True にすると最後の端数バッチを捨てる（BN使用時の形状安定など）
def create_loader(glob_patterns, label, cfg, shuffle=False, drop_last=False):

    # Dataset を生成（ここでファイル列挙と ref_len 決定が走る）
    ds = AudioDataset(glob_patterns, label, cfg)

    # PyTorch DataLoader を生成
    # 主要引数と意図：
    #   - batch_size: 1バッチのサンプル数（cfg['train']['batch_size']）
    #   - shuffle: 学習時は True が一般的、評価時は False で順序固定
    #   - drop_last: 端数バッチを捨てるか（Trueでバッチshapeが常に一定）
    #   - num_workers: 前処理/特徴抽出を並列化するワーカ数（cfg['loader']['num_workers']）
    #   - pin_memory: GPU転送を速くするためCPU側メモリをピン留め（cfg['loader']['pin_memory']）
    #   - persistent_workers: worker を epoch 間で維持して初期化コストを抑える（cfg['loader']['persistent_workers']）
    #     ※ num_workers=0 の場合は効果なし。環境によってはメモリ消費が増える点に注意。
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],  # バッチサイズ（cfg['train']['batch_size']）
        shuffle=shuffle,    # サンプル順のシャッフル有無（学習: True / 評価: False が一般的）
        drop_last=drop_last,    # 端数バッチを捨てるか（True: 常に同じバッチサイズに揃う）
        num_workers=cfg["loader"]["num_workers"],   # DataLoader worker 数（大きいほど並列だが、環境によりI/O競合やメモリ増大に注意）
        pin_memory=cfg["loader"]["pin_memory"], # pin_memory（GPUを使う場合に転送が速くなることがある）
        persistent_workers=cfg["loader"]["persistent_workers"]  # persistent_workers（worker を使い回して初期化を削減。num_workers>0 で有効）
    )
    return loader, ds
