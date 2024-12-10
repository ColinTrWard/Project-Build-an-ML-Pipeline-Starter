import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # Uncomment if testing is required
    # "test_regression_model",
]

@hydra.main(version_base="1.1", config_name="config")
def go(config: DictConfig):
    # Setup W&B experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Parse steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Work in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        if "basic_cleaning" in active_steps:
            mlflow.run(
                to_absolute_path("src/basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_name": "clean_sample.csv",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            mlflow.run(
                to_absolute_path("src/data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                },
            )

        if "data_split" in active_steps:
            mlflow.run(
                to_absolute_path("src/data_split"),
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"].get("stratify_by", "none"),
                },
            )

        if "train_random_forest" in active_steps:
            # Serialize Random Forest configuration
            rf_config = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config, "w") as fp:
                random_forest_config = OmegaConf.to_container(
                    config["modeling"]["random_forest"], resolve=True
                )
                json.dump(random_forest_config, fp)

            mlflow.run(
                to_absolute_path("src/train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export",
                    "target": config["modeling"]["target"],
                    "numerical_features": ",".join(config["modeling"]["numerical_features"]),
                    "categorical_features": ",".join(config["modeling"]["categorical_features"]),
                    "date_features": ",".join(config["modeling"]["date_features"]),
                    "stratify_by": config["modeling"].get("stratify_by", "none"),
                    "max_tfidf_features": config["modeling"].get("max_tfidf_features", 5),
                },
            )

        if "test_regression_model" in active_steps:
            mlflow.run(
                to_absolute_path("components/test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()

