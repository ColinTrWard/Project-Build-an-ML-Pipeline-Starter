import os
import argparse
import pandas as pd
import wandb
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json


def go(args):
    # Initialize W&B
    wandb.init(project="nyc_airbnb", job_type="train_random_forest")

    # Log configuration parameters
    wandb.config.update(vars(args))

    # Retrieve the artifact from W&B
    print(f"Fetching artifact: {args.trainval_artifact}")
    artifact = wandb.use_artifact(args.trainval_artifact)

    # Download the artifact and locate the exact file
    artifact_dir = artifact.download()
    print(f"Artifact directory: {artifact_dir}")

    # Search for trainval_data.csv in artifact directory and subdirectories
    file_name = "trainval_data.csv"
    file_path = None
    for root, _, files in os.walk(artifact_dir):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            break

    if not file_path:
        raise FileNotFoundError(
            f"'{file_name}' not found in artifact directory or subdirectories: {artifact_dir}. "
            f"Contents: {os.listdir(artifact_dir)}"
        )

    print(f"Successfully located the file: {file_path}")

    # Load the data
    data = pd.read_csv(file_path)
    print(f"Loaded data with shape: {data.shape}")

    target = args.target
    features = [col for col in data.columns if col != target]

    X = data[features]
    y = data[target]

    # Define preprocessing pipeline
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), args.numerical_features.split(",")),
            ("cat", OneHotEncoder(handle_unknown="ignore"), args.categorical_features.split(",")),
            ("date", StandardScaler(), args.date_features.split(",")),
        ]
    )

    # Load Random Forest configuration from JSON file
    with open(args.rf_config, "r") as f:
        rf_config = json.load(f)

    # Log Random Forest configuration to WandB
    wandb.config.update(rf_config, allow_val_change=True)

    rf_model = RandomForestRegressor(
        max_depth=rf_config["max_depth"],
        n_estimators=rf_config["n_estimators"],
        random_state=args.random_seed,
    )

    # Log the max TF-IDF features parameter
    wandb.log({"max_tfidf_features": args.max_tfidf_features})

    # Build pipeline
    sk_pipe = Pipeline([("preprocess", preprocess), ("model", rf_model)])

    # Split the dataset
    print(f"Splitting data with validation size: {args.val_size}")
    stratify = data[args.stratify_by] if args.stratify_by != "none" else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_seed, stratify=stratify
    )

    # Train and evaluate the model
    print("Training the Random Forest model...")
    sk_pipe.fit(X_train, y_train)
    y_pred = sk_pipe.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Log metrics to WandB
    wandb.log({"mae": mae, "mse": mse, "r2": r2})

    # Save and log the model artifact
    model_path = f"{args.output_artifact}.joblib"
    print(f"Saving the model to {model_path}...")
    joblib.dump(sk_pipe, model_path)

    artifact = wandb.Artifact(
        args.output_artifact, type="model", description="Random Forest model pipeline"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model")
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--numerical_features", type=str, required=True)
    parser.add_argument("--categorical_features", type=str, required=True)
    parser.add_argument("--date_features", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True, help="Maximum number of TF-IDF features to use")
    parser.add_argument(
        "--stratify_by", type=str, default="none", help="Column name for stratification during train-test split"
    )
    args = parser.parse_args()
    go(args)
