import argparse
import logging
import os
import json
import pandas as pd
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO)

def go(args):
    """
    Main function to train a random forest model and upload results to Weights & Biases.
    """
    # Start a Weights & Biases run
    run = wandb.init(
        project="nyc_airbnb",  # Ensure this matches your W&B project name
        job_type="train_random_forest",
        config={
            "val_size": args.val_size,
            "random_seed": args.random_seed,
            "stratify_by": args.stratify_by,
            "target": args.target,
            "numerical_features": args.numerical_features,
            "categorical_features": args.categorical_features,
            "rf_config_path": args.rf_config,
            "max_tfidf_features": args.max_tfidf_features,
            "date_features": args.date_features,
        },
    )

    logging.info(f"Resolving artifact path for: {args.trainval_artifact}")
    artifact = wandb.use_artifact(args.trainval_artifact)
    artifact_dir = artifact.download()

    logging.info(f"Downloaded artifact to: {artifact_dir}")

    # Debug: Log the directory structure to identify the correct path
    logging.info(f"Contents of artifact directory {artifact_dir}:")
    for root, dirs, files in os.walk(artifact_dir):
        logging.info(f"Root: {root}, Directories: {dirs}, Files: {files}")

    # Dynamically locate the trainval_data.csv within the artifact directory
    trainval_data_path = None
    for root, _, files in os.walk(artifact_dir):
        if "trainval_data.csv" in files:
            trainval_data_path = os.path.join(root, "trainval_data.csv")
            break

    if trainval_data_path is None:
        logging.error(f"trainval_data.csv not found in artifact directory: {artifact_dir}")
        raise FileNotFoundError(f"trainval_data.csv not found in artifact directory: {artifact_dir}")

    logging.info(f"Loading trainval data from {trainval_data_path}")
    trainval_data = pd.read_csv(trainval_data_path)

    # Splitting the dataset
    logging.info("Splitting the dataset...")
    stratify_col = trainval_data[args.stratify_by] if args.stratify_by else None
    train_df, val_df = train_test_split(
        trainval_data,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify_col,
    )

    logging.info(f"Train size: {train_df.shape}, Validation size: {val_df.shape}")

    # Prepare the Random Forest configuration
    logging.info(f"Loading Random Forest configuration from: {args.rf_config}")
    with open(args.rf_config, "r") as rf_file:
        rf_config = json.load(rf_file)

    logging.info(f"Loaded Random Forest Configuration: {rf_config}")

    # Features and target
    X_train = train_df[args.numerical_features + args.categorical_features]
    y_train = train_df[args.target]

    X_val = val_df[args.numerical_features + args.categorical_features]
    y_val = val_df[args.target]

    # Train the model
    logging.info("Training the Random Forest model...")
    model = RandomForestRegressor(**rf_config)
    model.fit(X_train, y_train)

    logging.info("Model training complete.")

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    logging.info(f"Train R^2 score: {train_score}")
    logging.info(f"Validation R^2 score: {val_score}")

    # Log metrics to Weights & Biases
    wandb.log({"train_score": train_score, "val_score": val_score})

    # Log the model and validation performance to W&B
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "random_forest_model.pkl")
        pd.to_pickle(model, model_path)

        logging.info("Uploading the trained model to W&B...")
        artifact = wandb.Artifact(
            args.output_artifact,
            type="model_export",
            description="Random Forest Model Export",
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)

    logging.info("Training pipeline complete.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument("--trainval_artifact", type=str, required=True, help="Train/validation artifact name")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--stratify_by", type=str, default=None, help="Column to stratify by")
    parser.add_argument("--target", type=str, required=True, help="Target column for prediction")
    parser.add_argument("--numerical_features", type=str, nargs="+", required=True, help="Numerical feature columns")
    parser.add_argument("--categorical_features", type=str, nargs="+", required=True, help="Categorical feature columns")
    parser.add_argument("--rf_config", type=str, required=True, help="Random Forest configuration file")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output artifact name")
    parser.add_argument("--max_tfidf_features", type=int, help="Max TF-IDF features")
    parser.add_argument("--date_features", type=str, nargs="+", help="Date feature columns")

    args = parser.parse_args()

    go(args)
