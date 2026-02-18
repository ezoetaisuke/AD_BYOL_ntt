# AE→BYOL-A 置換メモ（最小差分）

## 変更方針
- 既存の `main_train.py` / `main_eval.py` の入口はそのまま利用。
- 波形→log-mel/crop は **既存 `src/features.py` を流用**。
- AE部分のみ BYOL-A encoder に置換し、異常スコアは従来同様 Mahalanobis 距離。
- `scores.csv` / `score_hist.png` / `metrics.csv` / `learning_curve.png` の出力互換を維持。

## 追加した主な仕様
- `byol.mode`: `pretrained` / `scratch`
- 時間情報保持のため、評価時に `chunk_sec` / `hop_sec` でチャンク分割して埋め込み列化
- チャンクごとのMahalanobis距離系列を `score_aggregate` で1ファイルスコアへ集約
- pretrained時のみ resample/pad/trim を適用
- 破損wav等はスキップし、`skipped_wavs.log` に保存

## 実行方法
```bash
python main_train.py
python main_eval.py
```

## 初心者向け補足
- 学習時の損失は BYOL loss です（再構成誤差ではありません）。
- 判定ロジック（Mahalanobis閾値判定）はAE版と同じ流れです。
- `loss_history.csv` も追加保存しているため、既存 `metrics.csv` と併用できます。
