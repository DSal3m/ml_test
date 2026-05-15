# ml_test — LOTR NLP mini-project

Проект по тематике **«Властелин колец»**: классификация текстов по персонажам, расам и локациям.

## Что внутри

- `data/lotr_dataset.csv` — реальный CSV-датасет для обучения (колонки `text`, `label`)
- `src/main.py` — пайплайн классификации (TF-IDF + LogisticRegression)
- `scripts/train.py` — обучение модели из CSV и сохранение в `models/`
- `tests/` — автотесты

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Обучение на CSV

```bash
python scripts/train.py --data data/lotr_dataset.csv --out models/lotr_model.joblib
```

## Использование

```python
from src.main import train_default_lotr_model, predict_topic

model = train_default_lotr_model()
print(predict_topic(model, "One Ring to rule them all"))
```
