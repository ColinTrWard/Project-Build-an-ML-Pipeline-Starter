import argparse
import logging
import os
import json
import pandas as pd
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import wandb
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)


def process_artifact(artifact_dir):
    """
    Process files in the artifact directory based on their type.
    Returns the path to the trainval_data.csv file after processing.
    """
    trainval_data_path = None
    for root, _, files in os.walk(artifact_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith('.csv'):
                logging.info(f"CSV file detected: {file_path}")
                trainval_data_path = file_path
            elif file.endswith('.zip'):
                logging.info(f"ZIP file detected: {file_path}. Extracting...")
                shutil.unpack_archive(file_path, artifact_dir)
            elif file.endswith('.json'):
                logging.info(f"JSON file detected: {file_path}. Converting to CSV...")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                trainval_data_path = os.path.join(artifact_dir, 'trainval_data.csv')
                df.to_csv(trainval_data_path, index=False)
                logging.info(f"Converted JSON to CSV: {trainval_data_path}")
            elif file.endswith('.txt'):
                logging.info(f"Text file detected: {file_path}. Processing as plain text...")
                # Attempting to read with utf-8 encoding to handle various characters
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Here, you can convert plain text to a pandas dataframe or save it to a CSV if needed
                    trainval_data_path = os.path.join(artifact_dir, 'trainval_data.csv')
                    with open(trainval_data_path, 'w', encoding='utf-8') as f:
                        f.write(content)  # Saving the plain text as CSV (if that's what you need)
                    logging.info(f"Text file content saved as CSV: {trainval_data_path}")
                except UnicodeDecodeError as e:
                    logging.error(f"Failed to read {file_path} due to encoding issues: {e}")
                    raise
    if not trainval_data_path:
        raise FileNotFoundError("No compatible data file (CSV, ZIP, JSON, or TXT) found in the artifact.")

    return trainval_data_path


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
    artifact = run.use_artifact(args.trainval_artifact)
    artifact_dir = artifact.download()

    logging.info(f"Downloaded artifact to: {artifact_dir}")
    trainval_data_path = process_artifact(artifact_dir)

    logging.info(f"Loading trainval data from {trainval_data_path}")
    trainval_data = pd.read_csv(trainval_data_path)
    logging.info(f"Loaded trainval data with shape: {trainval_data.shape}")

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
    numerical_features = args.numerical_features
    categorical_features = args.categorical_features

    X_train = train_df[numerical_features + categorical_features]
    y_train = train_df[args.target]

    X_val = val_df[numerical_features + categorical_features]
    y_val = val_df[args.target]

    # Preprocessing for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Build the pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**rf_config)),
        ]
    )

    # Train the model
    logging.info("Training the Random Forest model with preprocessing...")
    model.fit(X_train, y_train)

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
    parser.add_argument("--categorical_features", type=str, nargs="+", required=True,
                        help="Categorical feature columns")
    parser.add_argument("--rf_config", type=str, required=True, help="Random Forest configuration file")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output artifact name")
    parser.add_argument("--max_tfidf_features", type=int, help="Max TF-IDF features")
    parser.add_argument("--date_features", type=str, nargs="+", help="Date feature columns")

    args = parser.parse_args()

    go(args)