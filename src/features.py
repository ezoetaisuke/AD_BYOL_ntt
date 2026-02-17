import numpy as np
import librosa

class FeatureExtractor:
    """
    2D-Conv向けに [1, F, T] の2D特徴を返す（チャンネル=1でConv2dに入力）
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.type = cfg["feature"]["type"]
        self.sr = cfg["audio"]["target_sr"]
        self.logmel_params = cfg["feature"]["logmel"]
        self.mfcc_params = cfg["feature"]["mfcc"]
        self.target_frames = cfg.get("feature", {}).get("target_frames", None)

    def __call__(self, y_1d):
        """
        y_1d: np.ndarray [N] モノラル波形（srはself.sr）
        return: np.ndarray [1, F, T]  （Conv2dで扱うための 1ch 追加）
        """
        if self.type == "logmel":
            return self._logmel(y_1d)
        elif self.type == "mfcc":
            return self._mfcc(y_1d)
        else:
            raise ValueError(f"Unsupported feature type: {self.type}")
    
    def _apply_crop_logmel(self, X: np.ndarray) -> np.ndarray:
        """
        X: [F, T] (log-mel spec)
        cfg["feature"]["crop"] で指定された周波数・時間範囲に切り出す。

        設定例（YAML想定）:
        feature:
          crop:
            enable: true
            freq:
              enable: true
              f_min_bin: 10    # メルバンド index (0〜n_mels-1)
              f_max_bin: 80
            time:
              enable: true
              t_min_sec: 0.02  # 秒指定（インパルス開始）
              t_max_sec: 0.06  # 秒指定（インパルス終了）

        - freq は f_min_bin, f_max_bin で [min, max] のメルバンドを選択
        - time は t_min_frame / t_max_frame か t_min_sec / t_max_sec のどちらかで指定
        """
        feat_cfg = self.cfg.get("feature", {})
        crop_cfg = feat_cfg.get("crop", {})
        if not crop_cfg.get("enable", False):
            return X

        F, T = X.shape

        # ----- 周波数方向 (メルバンド index) -----
        freq_cfg = crop_cfg.get("freq", {})
        if freq_cfg.get("enable", False):
            f0 = freq_cfg.get("f_min_bin", 0)
            f1 = freq_cfg.get("f_max_bin", F - 1)
            # None 許容
            if f0 is None:
                f0 = 0
            if f1 is None:
                f1 = F - 1
            f0 = max(0, int(f0))
            f1 = min(F - 1, int(f1))
            if f0 > f1:
                raise ValueError(f"freq crop invalid: f_min_bin={f0}, f_max_bin={f1}, F={F}")
            X = X[f0:f1+1, :]  # [F', T]

        # ----- 時間方向 (フレーム or 秒) -----
        time_cfg = crop_cfg.get("time", {})
        if time_cfg.get("enable", False):
            hop = self.logmel_params["hop_length"]

            # 開始フレーム
            if "t_min_frame" in time_cfg:
                t0 = int(time_cfg["t_min_frame"])
            elif "t_min_sec" in time_cfg:
                t0 = int(round(float(time_cfg["t_min_sec"]) * self.sr / hop))
            else:
                t0 = 0

            # 終了フレーム
            if "t_max_frame" in time_cfg:
                t1 = int(time_cfg["t_max_frame"])
            elif "t_max_sec" in time_cfg:
                t1 = int(round(float(time_cfg["t_max_sec"]) * self.sr / hop))
            else:
                t1 = T - 1

            t0 = max(0, t0)
            t1 = min(T - 1, t1)
            if t0 > t1:
                raise ValueError(f"time crop invalid: t_min={t0}, t_max={t1}, T={T}")
            X = X[:, t0:t1+1]  # [F' or F'', T']

        if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Crop resulted in empty feature map: shape={X.shape}")

        return X
    

    def _maybe_match_target_frames(self, X):
        if self.target_frames is None:
            return X
        target = int(self.target_frames)
        t = X.shape[1]
        if t < target:
            pad = target - t
            left = pad // 2
            right = pad - left
            X = np.pad(X, ((0, 0), (left, right)), mode="constant")
        elif t > target:
            start = (t - target) // 2
            X = X[:, start:start + target]
        return X

    def _maybe_cmvn(self, X):
        norm = self.cfg["feature"].get("normalize", {})
        if not norm.get("enable", False):
            return X
        eps = float(norm.get("eps", 1e-8))
        # サンプル全体で CMVN（[F,T] 全体）
        m = X.mean()
        s = X.std()
        return (X - m) / max(s, eps)

    def _logmel(self, y):
        p = self.logmel_params
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=p["n_fft"], hop_length=p["hop_length"],
            n_mels=p["n_mels"], fmin=p["fmin"], fmax=p["fmax"], power=p["power"]
        )
        # dB化（安定化のための微小量加算）
        S_db = librosa.power_to_db(np.maximum(S, 1e-12), ref=p["ref"])
        X = S_db.astype(np.float32)

        # ★ ここでインパルス関連の時間・周波数だけを切り出す
        X = self._apply_crop_logmel(X)

        X = self._maybe_match_target_frames(X)
        X = self._maybe_cmvn(X)
        return X[np.newaxis, ...]

    def _mfcc(self, y):
        p = self.mfcc_params
        M = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=p["n_mfcc"])
        if p["deltas"]:
            delta = librosa.feature.delta(M)
            delta2 = librosa.feature.delta(M, order=2)
            M = np.concatenate([M, delta, delta2], axis=0)
        X = M.astype(np.float32)
        X = self._maybe_match_target_frames(X)
        X = self._maybe_cmvn(X)
        return X[np.newaxis, ...]
