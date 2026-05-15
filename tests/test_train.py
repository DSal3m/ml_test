from pathlib import Path
import tempfile

from scripts.train import train


def test_train_saves_model_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.joblib"
        saved = train(Path("data/lotr_dataset.csv"), out)
        assert saved.exists()
        assert saved == out
