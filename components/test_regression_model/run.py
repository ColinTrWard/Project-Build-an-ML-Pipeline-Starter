import argparse
import logging
import os
import json
import pickle
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)

def process_artifact(artifact_dir):
    """
    Process files in the artifact directory.
    """
    for root, _, files in os.walk(artifact_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.csv'):
                return file_path
            elif not os.path.splitext(file)[1]:  # No extension
                logging.warning(f"File with no extension detected: {file_path}. Checking content...")
                with open(file_path, 'r') as f:
                    try:
                        pd.read_csv(f)
                        return file_path  # If it can be read as CSV
                    except Exception:
                        continue
    raise FileNotFoundError(f"No compatible data file (CSV) found in {artifact_dir}.")

def go(args):
    """
    Main function for test regression model.
    """
    run = wandb.init(project="nyc_airbnb", job_type="test_regression_model")

    logging.info(f"Downloading model artifact: {args.mlflow_model}")
    model_artifact = run.use_artifact(args.mlflow_model)
    model_dir = model_artifact.download()

    # Attempt to load the model
    model_path = os.path.join(model_dir, "model.pkl")
    if os.path.exists(model_path):
        logging.info(f"Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    logging.info(f"Downloading test dataset artifact: {args.test_dataset}")
    test_data_artifact = run.use_artifact(args.test_dataset)
    test_data_dir = test_data_artifact.download()

    logging.info(f"Processing artifact directory: {test_data_dir}")
    test_data_path = process_artifact(test_data_dir)
    logging.info(f"Loading test dataset from: {test_data_path}")
    test_data = pd.read_csv(test_data_path)

    # Load configuration
    logging.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    target_column = config.get("target_column")
    if target_column not in test_data.columns:
        raise ValueError(f"The test dataset must contain the target column '{target_column}' for evaluation.")
    
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    logging.info("Performing predictions...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"R² score on test set: {r2}")
    logging.info(f"Mean Absolute Error (MAE) on test set: {mae}")

    # Log metrics to Weights & Biases
    wandb.log({"test_r2_score": r2, "test_mae": mae})

    # Log evaluation results as artifact
    results_path = os.path.join(test_data_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({"r2_score": r2, "mae": mae}, f)
    
    artifact = wandb.Artifact("test_evaluation_results", type="evaluation_results")
    artifact.add_file(results_path)
    run.log_artifact(artifact)

    logging.info("Evaluation results logged as artifact.")
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a regression model.")
    parser.add_argument("--mlflow_model", type=str, required=True, help="Model artifact name in MLflow")
    parser.add_argument("--test_dataset", type=str, required=True, help="Test dataset artifact name in W&B")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")

    args = parser.parse_args()
    go(args)
