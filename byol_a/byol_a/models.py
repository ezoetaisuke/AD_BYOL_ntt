"""Model definitions.

Reference:
    Y. Koizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “The NTT DCASE2020 challenge task 6 system:
    Automated audio captioning with keywords and sentence length estimation,” DCASE2020 Challenge, Tech. Rep., 2020.
    https://arxiv.org/abs/2007.00225
"""

# BYOL-A 系の音声埋め込みネットワーク群の定義ファイル

import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class NetworkCommonMixIn():
    """Common mixin for network definition.
    各ネットワーククラスで共通して使うユーティリティ（重みロード等）をまとめた Mixin。
    """

    def load_weight(self, weight_file, device, state_dict=None, key_check=True):
        """Utility to load a weight file to a device.
        BYOL-A 事前学習済み重みを現在のネットワークにロードするためのヘルパ。
        """

        # state_dict が明示的に渡されていなければ、ファイルからロード
        state_dict = state_dict or torch.load(weight_file, map_location=device)

        # PyTorch-Lightning 等で保存された形式の場合は 'state_dict' キーを取り出す
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 不要なプレフィックス（fc. / features. など）を key から削る
        if key_check:
            weights = {}
            for k in state_dict:
                # fc., .fc., features., .features. の位置を検索
                m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
                if m is None: continue
                # マッチした位置以降を新しい key として使う
                new_k = k[m.start():]
                # 先頭が '.' の場合は削っておく
                new_k = new_k[1:] if new_k[0] == '.' else new_k
                weights[new_k] = state_dict[k]
        else:
            # key をそのまま使う場合
            weights = state_dict
        
        # 加工した state_dict を現在のモデルにロードし eval モードに設定
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        """全パラメータの requires_grad を一括で ON/OFF するユーティリティ。"""
        for p in self.parameters():
            p.requires_grad = trainable


class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network.
    入力: log-mel スペクトログラム [B, 1, n_mels, T]
    出力: 時系列埋め込み [B, T', d]
    """

    def __init__(self, n_mels, d):
        """
        Args:
            n_mels: 入力 log-mel スペクトログラムのメルフィルタ数
            d: 各時間フレームの出力埋め込み次元
        """
        super().__init__()
        # 3 ブロック構成の 2D-CNN（Conv -> BN -> ReLU -> MaxPool）
        # 時間・周波数方向を MaxPool2d により 1/8 までダウンサンプリングする
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
        )
        # 各時間フレームごとに [64 * (n_mels/8)] -> d へ写像する全結合層
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        # 出力埋め込み次元をメンバに保持（後続クラスの assert 等で使用）
        self.d = d

    def forward(self, x):
        """
        Args:
            x: [B, 1, n_mels, T] の log-mel スペクトログラム
        Returns:
            [B, T', d] の時間方向系列特徴（T' は元の時間フレーム数の 1/8 程度）
        """

        # CNN による特徴抽出
        x = self.features(x)       # (batch, ch, mel, time)       

        # 時間方向を 2 次元目に持ってくることで「フレームごとの特徴ベクトル」扱いにする
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        
        # テンソル形状を展開し、時間ごとに平坦化された特徴ベクトルにする
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        
        # 各時間フレームごとに全結合層で埋め込み次元 d へ射影
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.

    AudioNTT2020Task6 が出力する [B, T, d] を時間方向に集約して
    クリップ単位の埋め込み [B, d] に変換するラッパ。
    """

    def __init__(self, n_mels=64, d=512):
        """
        Args:
            n_mels: 入力 log-mel のメルフィルタ数
            d: 出力埋め込み次元（クリップ単位の特徴次元）
        """
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        """
        Args:
            x: [B, 1, n_mels, T] の log-mel スペクトログラム
        Returns:
            [B, d]: クリップ全体を表す固定長ベクトル
        """
        # まずフレーム系列特徴 [B, T, d] を取得
        x = super().forward(x)
        # 時間方向の max pooling（各次元で最も強く応答したフレームを拾う）
        (x1, _) = torch.max(x, dim=1)
        # 時間方向の mean pooling（時間平均でクリップの代表値を取る）
        x2 = torch.mean(x, dim=1)
        # max と mean を足し合わせて最終的なクリップ埋め込みにする
        x = x1 + x2
        # 形状が期待どおりか一応チェック
        assert x.shape[1] == self.d and x.ndim == 2
        return x


class AudioNTT2020Task6X(nn.Module, NetworkCommonMixIn):
    """A variant of DCASE2020 Task6 NTT Solution Audio Embedding Network.
    Enabeld to return features by layers.

    各畳み込みブロックおよび全結合層の手前／後の特徴を取り出せる拡張版。
    Examples:
        model(x) -> returns sample-level features of [B, T, D].
        model(x, layered=True) -> returns sample-level layered features of [B, T, 5*D]
        model.by_layers(model.(x, layered=True)) -> returns sample-level features by layers as a list of [B, T, D] * 5
    """

    def __init__(self, n_mels, d):
        """
        Args:
            n_mels: 入力 log-mel のメルフィルタ数
            d: 各レイヤの系列埋め込み次元
        """
        super().__init__()
        # 特徴抽出を 3 つの conv ブロックに分割定義
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        # Conv 出力を系列ベクトル [B, T, d] へ変換する 1 段目の全結合
        self.fc1 = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
        )
        # さらなる写像（Dropout を挟んだ 2 段目の MLP）
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d
        # conv1, conv2, conv3, fc1, fc2 の 5 箇所で特徴を出すので 5
        self.n_feature_layer = 5

    def forward(self, x, layered=False):
        """
        Args:
            x: [B, 1, n_mels, T] の log-mel スペクトログラム
            layered: True の場合は各レイヤの特徴を結合したものを返す
        Returns:
            layered == False: [B, T', d]
            layered == True : [B, T', 5*d]  （5 レイヤ分を最後の次元で結合）
        """
        def reshape_conv_feature(v):
            """
            Conv ブロック出力 [B, CH, F, T] を [B, T, d] に揃えるための補助関数。
            - (B, CH, F, T) -> (B, T, CH*F) に整形
            - 必要であれば 0 埋めで最後の次元を d に拡張
            - 時間長が target_t より長い場合は平均プーリングで縮約
            """
            B, CH, F, T = v.shape
            # (B, CH, F, T) -> (B, T, CH*F)
            v = v.permute(0, 3, 1, 2).reshape(B, T, CH*F)

            # 特徴次元が d 未満の場合は末尾に 0 パディングして d に合わせる
            if v.shape[-1] < self.d:
                v = torch.nn.functional.pad(v, (0, self.d - v.shape[-1]), 'constant', 0.0)
            
            # 時間長が target_t より長い場合、平均を取りながら 1/2 ずつ縮める
            while v.shape[1] > target_t:
                # 奇数フレーム数のときは最後の 2 フレームを平均して 1 フレームにまとめる
                if v.shape[1] % 2 == 1:
                    v = torch.cat([v[:, :-2], v[:, -2:].mean(1, keepdim=True)], axis=1)
                # [B, T, D] -> [B, T/2, D] に隣接フレーム平均でダウンサンプリング
                T = v.shape[1]
                v = v.reshape(B, T//2, 2, v.shape[-1])
                v = v.mean(2) # average adjoining two time frame features.
            return v

        # conv3 まで MaxPool2d(2) を 3 回通るので、時間長 T はおおよそ T/8 になる想定
        target_t = x.shape[-1] // 8
        features = []

        # conv1 ブロック
        x = self.conv1(x)         # (batch, ch, mel, time)
        features.append(reshape_conv_feature(x))

        # conv2 ブロック
        x = self.conv2(x)
        features.append(reshape_conv_feature(x))

        # conv3 ブロック
        x = self.conv3(x)
        features.append(reshape_conv_feature(x))

        # conv 出力を時系列ベクトルに変換
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)

        # fc1 の出力を保存
        x = self.fc1(x)
        features.append(x)

        # fc2 の出力を保存（最終系列特徴）
        x = self.fc2(x)
        features.append(x)

        if layered:
            # 各レイヤの特徴 [B, T, d] を最後の次元で結合 -> [B, T, 5*d]
            return torch.cat(features, dim=-1) # [B, T, 5*D]
        # layered=False の場合は最後の fc2 の出力のみ返す
        return x # [B, T, D]

    def by_layers(self, layered_features):
        """Decompose layered features into the list of features for each layer.
        layered=True で得た [B, T, 5*D] を [ [B, T, D]] * 5 のリストに分解する。
        """


        *B, LD = layered_features.shape
        # 最後の次元が 5*D であることを確認
        assert LD == self.n_feature_layer * self.d

        # [..., 5*D] -> [..., 5, D] に変形
        layered_features = layered_features.reshape(*B, self.n_feature_layer, self.d)
        # 入力の次元数に応じて、[layer, B, T, D] または [layer, B, D] となるように permute
        layered_features = (
            layered_features.permute(2, 0, 1, 3) 
            if len(layered_features.shape) > 3 
            else layered_features.permute(1, 0, 2)
        )

        # 各レイヤ分をリストとして返却
        return [layered_features[l] for l in range(self.n_feature_layer)]

    def load_weight(self, weight_file, device):
        """Whapper function for loading BYOL-A pre-trained weights.
        AudioNTT2020Task6 用の学習済み重みを、この Task6X の名前付けに合わせてロードする。
        """
        # 元のモデル (AudioNTT2020Task6) の層名 -> 本クラスの層名 のマッピング表
        namemap = {
            'features.0': 'conv1.0', 'features.1': 'conv1.1',
            'features.4': 'conv2.0', 'features.5': 'conv2.1',
            'features.8': 'conv3.0', 'features.9': 'conv3.1',
            'fc.0': 'fc1.0',
            'fc.3': 'fc2.1',
        }
        # 重みファイルを読み込み
        state_dict = torch.load(weight_file, map_location=device)
        new_dict = {}
        # key 名の変換と 'num_batches_tracked' を持つ BatchNorm の統計情報を削除
        for key in state_dict:
            if 'num_batches_tracked' in key:
                continue
            new_key = key
            for map_key in namemap:
                if map_key in key:
                    # 元の key の一部を本クラス用の名前に置き換える
                    new_key = key.replace(map_key, namemap[map_key])
                    break
            new_dict[new_key] = state_dict[key]
        # NetworkCommonMixIn の load_weight に委譲（key_check=False でそのまま使う）
        return super().load_weight(weight_file, device, state_dict=new_dict, key_check=False)


class AudioNTT2020X(AudioNTT2020Task6X):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    Enabeld to return features by layers.

    AudioNTT2020Task6X が返す系列特徴 [B, T, D] を時間方向に集約し、
    クリップ単位の埋め込み（必要に応じてレイヤごと）を返すクラス。

    Examples:
        model(x) -> returns sample-level features of [B, D].
        model(x, layered=True) -> returns sample-level layered features of [B, 5*D]
        model(x, layered=True, by_layers=True) -> returns sample-level features by layers as a list of [B, D] * 5
    """

    def __init__(self, n_mels=64, d=2048):
        """
        Args:
            n_mels: 入力 log-mel のメルフィルタ数
            d: クリップ単位の出力埋め込み次元
        """
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x, layered=False, by_layers=False):
        """
        Args:
            x: [B, 1, n_mels, T] の log-mel スペクトログラム
            layered: True の場合、各レイヤの特徴を連結した [B, T, 5*D] を使って集約
            by_layers: True の場合、クリップベクトルをレイヤごとのリストに分解して返す
        Returns:
            by_layers == False: [B, D]
            by_layers == True : [ [B, D] ] * 5
        """
        # Task6X の forward で系列特徴を取得
        x = super().forward(x, layered=layered)
        # 時間方向 max pooling
        (x1, _) = torch.max(x, dim=1)
        # 時間方向 mean pooling
        x2 = torch.mean(x, dim=1)
        
        # max と mean を足し合わせてクリップ埋め込みを得る
        x = x1 + x2
        if by_layers:
            # layered=True の場合は、最後の次元が 5*D の想定
            # それをレイヤごと ([B, D]) のリストに変換して返す
            return self.by_layers(x)
        # by_layers=False の場合は結合された 1 ベクトルのみ返す
        return x
