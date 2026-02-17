import yaml

from src.byola_pipeline import run_eval_byola


if __name__ == "__main__":
    with open("configs/byola_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_eval_byola(cfg)
