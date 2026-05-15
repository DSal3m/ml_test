# ml_test — LOTR NLP mini-project

Проект посвящён тематике **«Властелин колец»** и предназначен для простого NLP-классификатора, обученного на текстовых данных по миру LOTR.

## Цель

Сделать минимально рабочий пайплайн, который:
- принимает фразы по тематике LOTR;
- предсказывает категорию (`hobbits`, `elves`, `rings`, `mordor`);
- может быть расширен на полноценный датасет и более сильную модель.

## Структура

- `src/` — код проекта
- `data/` — данные (игнорируются git)
- `tests/` — автотесты

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Как использовать

```python
from src.main import train_default_lotr_model, predict_topic

model = train_default_lotr_model()
print(predict_topic(model, "One Ring to rule them all"))
```
