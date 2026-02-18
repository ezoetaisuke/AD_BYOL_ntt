# BYOL-A 異常検知パイプライン（初心者向けガイド）

## 1) 何を置き換えたの？
- 旧: AE（再構成誤差ベース）
- 新: BYOL-A（自己教師あり表現学習）
- 判定は引き続き **Mahalanobis 距離** を使います。

実行入口:
- 学習: `main_train_byola.py`
- 評価: `main_eval_byola.py`

---

## 2) 重要ポイント（時間情報を潰さない）
1. wav → 既存 `src/features.py` の log-mel + crop（既存挙動そのまま）
2. スペクトログラムを `chunk_sec` / `hop_sec` で時間チャンク化
3. 各チャンクの埋め込み系列から Mahalanobis 距離系列を作る
4. `score_aggregate`（`max`, `topk_mean`, `percentile`, `mean+std` など）で 1ファイル1スコア化

> 単純な全体 mean pooling をデフォルトにしない設計です。

---

## 3) pretrained / scratch の違い（とても重要）

### pretrained
- BYOL-A 事前学習重みから開始。
- **このモードのみ** 以下を実施:
  - 入力 SR が違えば `target_sr` へ resample
  - 短ければ pad（`zero` or `repeat`）
  - 長ければ trim（`center` or `left`）

### scratch
- ランダム初期化で開始。
- 原則、波形読み込み時の強制整形をしません。
- 理由:
  - 人為的な resample/pad による分布改変を避けるため
  - 非定常・インパルスの時系列特性を維持しやすくするため
- ただし安全策として、log-mel計算不可な短波形は skip して `errors.log` に記録します。

---

## 4) 学習で保存されるもの
`outputs/<run_id>/` に以下を保存:
- `best.pth`（val_loss最小）
- `last.pth`（最終epoch）
- `loss_history.csv`
- `loss_curve.png`
- `maha_stats.npz`（`mu`, `cov`, `precision`）
- `errors.log`

評価時:
- `scores.csv`（`path`, `score`, 任意で `threshold`, `y_pred`）

---

## 5) 実行手順
```bash
# 1) 設定編集
vim configs/byola_config.yaml

# 2) 学習（train_ok + val_ok）
python main_train_byola.py

# 3) 評価（eval_glob）
python main_eval_byola.py
```

> `main_eval_byola.py` は `outputs/<run_id>/best.pth` と `maha_stats.npz` を読みます。  
> 学習と同じ `run_id` を指定してください。

---

## 6) 置換後フロー（短く）
1. BYOL-A を pretrained or scratch で用意
2. train_ok / val_ok で自己教師あり学習（任意）
3. train_ok の埋め込み列から Mahalanobis 統計推定
4. eval wav をチャンク埋め込み化 → 距離系列 → 時間集約スコア
