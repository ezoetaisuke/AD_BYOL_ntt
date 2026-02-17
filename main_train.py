
# YAML（設定ファイル）を読み込むためのライブラリ（PyYAML）を import
import yaml

# 学習処理の本体（エントリポイントから呼び出す関数）を import
# - src/train.py 内に run_train(cfg) が定義されている前提
# - src が Python の import パスに乗っていないと ImportError になります
from src.train import run_train

# このファイルが「スクリプトとして直接実行されたとき」だけ、以下の処理を動かします
# - 例: python main_train.py
# - 逆に、別モジュールから import された場合は実行されません
if __name__ == "__main__":

    # 学習設定を YAML から読み込みます
    # - パスは相対パス "configs/default.yaml" 固定
    # - 実行カレントディレクトリがプロジェクトルートでないと FileNotFoundError になり得ます
    # - encoding="utf-8" は日本語コメント等が含まれる YAML を想定した設定です
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        
        # YAML を Python の dict に変換します（安全なローダー）
        # - cfg は以降 run_train に渡され、学習の全設定（データ/モデル/最適化/保存など）の入力になります
        # - YAML が壊れていると yaml.YAMLError 系の例外が出ます
        cfg = yaml.safe_load(f)

        
    # 学習処理を開始します（学習ループ・ログ・チェックポイント保存などは run_train 側の実装に依存）
    run_train(cfg)
