# このスクリプトは「評価（eval）」のエントリポイントです
# - configs/default.yaml から評価設定を読み込み（cfg: Python の dict）
# - src/eval.py の run_eval(cfg) を呼び出して評価処理を開始します
# - 評価の本体（モデル読み込み/推論/指標計算/結果保存など）は run_eval 側の実装を参照してください

import yaml

# 評価処理の本体（エントリポイントから呼び出す関数）を import
# - src/eval.py 内に run_eval(cfg) が定義されている前提
# - src が Python の import パスに乗っていないと ImportError になります
from src.eval import run_eval

# このファイルが「スクリプトとして直接実行されたとき」だけ、以下の処理を動かします
# - 例: python main_eval.py
# - 逆に、別モジュールから import された場合は実行されません
if __name__ == "__main__":

    # 評価設定を YAML から読み込みます
    # - パスは相対パス "configs/default.yaml" 固定
    # - 実行カレントディレクトリがプロジェクトルートでないと FileNotFoundError になり得ます
    # - encoding="utf-8" は日本語コメント等が含まれる YAML を想定した設定です
    with open("configs/default.yaml", "r", encoding="utf-8") as f:

        # YAML を Python の dict に変換します（安全なローダー）
        # - cfg は以降 run_eval に渡され、評価に必要な全設定（データ/モデル/出力先/指標など）の入力になります
        # - YAML が壊れていると yaml.YAMLError 系の例外が出ます
        cfg = yaml.safe_load(f)

    # 評価処理を開始します（推論・評価指標の計算・結果の保存などは run_eval 側の実装に依存）
    run_eval(cfg)
