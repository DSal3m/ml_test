"""Train LOTR text classifier on CSV dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.main import build_lotr_pipeline


def train(csv_path: Path, model_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    model = build_lotr_pipeline()
    model.fit(df["text"].tolist(), df["label"].tolist())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lotr_dataset.csv")
    parser.add_argument("--out", default="models/lotr_model.joblib")
    args = parser.parse_args()

    saved = train(Path(args.data), Path(args.out))
    print(f"Model saved to: {saved}")


if __name__ == "__main__":
    main()
