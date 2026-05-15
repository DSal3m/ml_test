"""LOTR-themed text classification helpers."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_lotr_pipeline() -> Pipeline:
    """Build a simple text classification pipeline for LOTR topics."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def get_default_lotr_dataset() -> tuple[list[str], list[str]]:
    """Return a tiny built-in LOTR dataset for demo training."""
    texts = [
        "Frodo and Sam travel from the Shire",
        "Bilbo Baggins is a famous hobbit",
        "Legolas is an elven prince from Mirkwood",
        "Elrond rules in Rivendell among elves",
        "Sauron forged the One Ring in Mount Doom",
        "The Ring corrupts anyone who carries it",
        "Orcs march from Mordor under the Dark Lord",
        "Barad-dur stands in the land of Mordor",
    ]
    labels = [
        "hobbits",
        "hobbits",
        "elves",
        "elves",
        "rings",
        "rings",
        "mordor",
        "mordor",
    ]
    return texts, labels


def train_default_lotr_model() -> Pipeline:
    """Train and return a LOTR topic classifier on built-in sample data."""
    texts, labels = get_default_lotr_dataset()
    model = build_lotr_pipeline()
    model.fit(texts, labels)
    return model


def predict_topic(model: Pipeline, text: str) -> str:
    """Predict LOTR topic label for input text."""
    return str(model.predict([text])[0])
