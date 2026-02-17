# BYOL-A 異常検知パイプライン（Mahalanobis）

## 1. 何を置き換えたか
- `main_train.py` / `main_eval.py`（AEベース）とは別に、BYOL-A専用の実行入口を追加しました。
  - 学習: `main_train_byola.py`
  - 評価: `main_eval_byola.py`
- 既存の `src/features.py` をそのまま使い、`feature.logmel` と `feature.crop` の挙動は同一です。

## 2. 時間情報を潰さない設計
この実装は **例案1** を採用しています。

1. 各wavから log-mel(+crop) を作る。
2. `byola.time_embedding.chunk_sec / hop_sec` で時間チャンクに分割する。
3. 各チャンクを BYOL-A Encoder (`AudioNTT2020Task6`) に通し、**時系列埋め込み `[T', D]`** を得る。
4. 各時刻埋め込みで Mahalanobis 距離を計算し、距離系列を作る。
5. 距離系列を `score_aggregate.method`（max / topk_mean / percentile / mean）で1スコアへ集約する。

> ファイル全体を最初から1ベクトルmean poolingして終わり、はしていません。

## 3. pretrained / scratch の切替
`configs/byola_config.yaml` の `byola.mode` で指定します。

- `pretrained`
  - BYOL-A 事前学習重みをロードして開始
  - 必要なら `ssl_train.enable_in_pretrained: true` で自己教師あり継続学習
- `scratch`
  - BYOL-A をランダム初期化で開始
  - 通常は `ssl_train.enable_in_scratch: true` で自己教師あり学習

## 4. pretrained時の resample / pad / trim 仕様
`byola.pretrained_input` で制御します（**pretrained時のみ適用**）。

- `target_sr`: 入力fsが違うときのリサンプル先
- `min_len`: これより短い波形をpad
- `target_len`: 最終的に揃える長さ（短ければpad, 長ければtrim）
- `pad_mode`: `zero` or `repeat`
- `trim_mode`: `center` or `left`

`scratch` のときは、上記の強制整形は行いません。
（ただし log-mel計算に必要な最小長未満のwavはskipし、ログCSVへ記録します）

## 5. 実行方法
```bash
python main_train_byola.py
python main_eval_byola.py
```

## 6. 生成物
- 学習時
  - `byola.save.stats_path`: Mahalanobis統計（mean, precision）
  - `byola.save.encoder_path`: Encoder重み
  - `byola.save.skip_log_path`: skipファイル一覧
- 評価時
  - `byola.save.eval_csv`: `path, score` 必須列（+ `y_pred`, `threshold` は設定時）
  - `byola.save.skip_log_eval_path`: skipファイル一覧
