from src.main import get_default_lotr_dataset, predict_topic, train_default_lotr_model


def test_default_dataset_shapes() -> None:
    texts, labels = get_default_lotr_dataset()
    assert len(texts) == len(labels)
    assert len(texts) >= 8


def test_model_predicts_known_topics() -> None:
    model = train_default_lotr_model()

    pred_ring = predict_topic(model, "the one ring was forged by sauron")
    pred_hobbit = predict_topic(model, "sam and frodo are hobbits from the shire")

    assert pred_ring in {"rings", "mordor"}
    assert pred_hobbit == "hobbits"
