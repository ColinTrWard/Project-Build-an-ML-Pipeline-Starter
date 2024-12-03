#!/usr/bin/env python
"""
This script splits the provided dataframe into test and remainder datasets
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(vars(args))  # Log arguments to W&B

    logger.info(f"Fetching artifact {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    logger.info("Splitting trainval and test datasets")
    stratify_column = df[args.stratify_by] if args.stratify_by != "none" else None

    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_column,
    )

    for dataset, split_name in zip([trainval, test], ["trainval", "test"]):
        logger.info(f"Uploading {split_name}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            dataset.to_csv(fp.name, index=False)

            log_artifact(
                artifact_name=f"{split_name}_data.csv",
                artifact_type=f"{split_name}_data",
                artifact_description=f"{split_name} split of the dataset",
                file_path=fp.name,
                run=run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder datasets")

    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact to split (CSV file)")
    parser.add_argument("--test_size", type=float, required=True, help="Fraction of data for test split")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--stratify_by", type=str, default="none", help="Column for stratification (optional)")

    args = parser.parse_args()
    go(args)
