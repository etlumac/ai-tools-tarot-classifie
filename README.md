# tarot-classifier

**AI-Таролог** — классификатор тем вопросов по 5 классам: `love` | `self` | `social` | `health` | `career`

Проект реализует полный ML-пайплайн: загрузка данных из Google Sheets → предобработка → PostgreSQL → обучение 4 моделей → трекинг экспериментов в MLflow.

## Что это и зачем

Учебный MLOps-проект: конвейер полного цикла для задачи текстовой классификации — от сырых данных до сравнения моделей, с упором на **инженерную часть** (оркестрация, хранение, воспроизводимость экспериментов), а не только на саму модель.

Использует тот же домен (классификация тематики вопроса для гадания на картах таро), что и мой отдельный сервис [ai-tarot](https://github.com/etlumac/ai-tarot) — но это независимый проект с другой целью: там классификатор является частью production-микросервиса внутри более крупной системы, здесь же — самостоятельный пайплайн, где в фокусе весь путь данных: приём → оркестрация (Airflow) → хранение (PostgreSQL) → обучение 4 разных моделей → сравнение экспериментов (MLflow). Не оптимизировался под переиспользование кода между репозиториями — они решают разные задачи.

**Статус**: этапы данных (Airflow-пайплайн) и обучения (MLflow) реализованы и рабочие. Сервинг моделей через BentoML/ONNX и нагрузочное тестирование (Locust) — следующий шаг, в `pyproject.toml` уже заложены как опциональные зависимости, но ещё не реализованы.

---

## Структура проекта

```
tarot_classifier/
├── configs/
│   └── config.yaml           # все параметры (пути, гиперпараметры, БД)
├── src/
│   ├── data/
│   │   ├── loader.py         # загрузка из Google Sheets / CSV / инкремент
│   │   └── preprocessing.py  # очистка, лемматизация, PostgreSQL
│   ├── models/
│   │   └── predict.py        # инференс всех моделей
│   ├── training/
│       └── train.py          # обучение + MLflow логирование    
│       
├── airflow/
│   └── dags/
│       └── tarot_pipeline.py  # DAG: загрузка → предобработка → PostgreSQL
├── tests/
│   └── test_preprocessing.py
├── data/
│   ├── raw/                   # сырые CSV
│   └── processed/             # лемматизированные CSV
├── models/                    # артефакты моделей (.pkl, .bin, .cbm, onnx/)
└── pyproject.toml             # Poetry зависимости
```
---


## 0. Первоначальная настройка (делается один раз)

### 0.1 Клонировать репозиторий и установить зависимости

```bash
cd tarot_classifier
pip install poetry
poetry install
```

### 0.2 Запустить PostgreSQL в Docker

```bash
docker run --name tarot-postgres \
  -e POSTGRES_USER=tarot \
  -e POSTGRES_PASSWORD=tarot \
  -e POSTGRES_DB=tarot_db \
  -p 5432:5432 \
  -d postgres:15
```

Проверить что запустился:
```bash
docker exec -it tarot-postgres psql -U tarot -d tarot_db -c "SELECT 1;"
```

### 0.3 Установить Airflow (только в Linux / WSL2)


```bash
# Внутри WSL2 — в папке проекта
poetry install 

# Инициализировать БД Airflow
export AIRFLOW_HOME=~/airflow
poetry run airflow db migrate

# Создать папку для DAG-ов
mkdir -p ~/airflow/dags
cp airflow/dags/tarot_pipeline.py ~/airflow/dags/

# Создать пользователя для UI (логин/пароль: admin/admin)
poetry run airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname Admin \
  --role Admin --email admin@example.com
```

### 0.4 Создать папки для данных

```bash
mkdir -p data/raw data/processed models
```

---

## 1. Первичная загрузка данных (Лаб 1)

Загружает данные из Google Sheets, лемматизирует и сохраняет в CSV и PostgreSQL:

```bash
poetry run python -c "
from src.data.loader import load_from_sheets
from src.data.preprocessing import preprocess, save_to_csv, save_to_postgres

df = load_from_sheets('1HUUrXxlB9QQ4VpcLNYNOwZMJZsww5c-dmQnvTHLejGs')
df_proc = preprocess(df)
save_to_csv(df_proc, 'data/processed/questions_lemm.csv')
save_to_csv(df_proc, 'data/raw/questions.csv')
save_to_postgres(df_proc, 'postgresql://tarot:tarot@localhost:5432/tarot_db')
"
```

Проверить результат:
```bash
ls -lh data/processed/questions_lemm.csv

docker exec -it tarot-postgres psql -U tarot -d tarot_db \
  -c "SELECT label, COUNT(*) FROM questions GROUP BY label ORDER BY label;"
```

Ожидаемый результат: ~667 строк, 5 классов.

---

## 2. Оркестрация через Airflow (Лаб 1)

Airflow автоматизирует пайплайн по расписанию (каждый день в 02:00 UTC) и поддерживает инкрементальные обновления.

### Запуск (каждый раз при открытии проекта)

Запускать **в WSL2**, в папке проекта. Нужно **два терминала**:

```bash
# Терминал 1 — веб-интерфейс
export AIRFLOW_HOME=~/airflow
poetry run airflow webserver --port 8080

# Терминал 2 — планировщик (в той же папке проекта)
export AIRFLOW_HOME=~/airflow
poetry run airflow scheduler
```

Открыть UI: http://localhost:8080 (логин: admin, пароль: admin)

### Запуск пайплайна вручную

```bash
# Убедиться что DAG актуален
cp airflow/dags/tarot_pipeline.py ~/airflow/dags/

# Запустить
export AIRFLOW_HOME=~/airflow
poetry run airflow dags trigger tarot_data_pipeline

# Проверить статус (подождать ~15-20 сек)
poetry run airflow dags list-runs -d tarot_data_pipeline
```

После успешного запуска проверить БД:
```bash
docker exec -it tarot-postgres psql -U tarot -d tarot_db \
  -c "SELECT COUNT(*) FROM questions;"
```
---

## 3. Обучение моделей с MLflow (Лаб 2)

### Запуск (каждый раз при открытии проекта)

```bash
# Терминал 1 — MLflow UI (оставить запущенным)
poetry run mlflow ui --port 5000

# Терминал 2 — обучение всех 4 моделей (10–30 мин)
poetry run train
```

Открыть MLflow: http://localhost:5000

### Что обучается

| Модель | test Accuracy | test Macro F1 |
|---|---|---|
| TF-IDF + LogReg | ~0.70 | ~0.69 |
| fastText | ~0.62 | ~0.62 |
| CatBoost + TF-IDF | ~0.61 | ~0.60 |
| ruBERT-tiny2 | ~0.76 | ~0.76 |

Победитель: **ruBERT-tiny2** — лучший Macro F1 при разумном времени обучения.


### Предсказание через CLI

```bash
poetry run predict "Вернётся ли он ко мне?"  -m ensemble
```

---

## 4. Быстрый старт при повторном открытии проекта

Если вы уже всё настроили и хотите просто продолжить работу:

```bash
# 1. Проверить что Docker запущен и PostgreSQL работает
docker start tarot-postgres
docker exec -it tarot-postgres psql -U tarot -d tarot_db -c "SELECT COUNT(*) FROM questions;"

# 2. В WSL2, в папке проекта — запустить Airflow (2 терминала)
export AIRFLOW_HOME=~/airflow
poetry run airflow webserver --port 8080   # терминал 1
poetry run airflow scheduler               # терминал 2

# 3. MLflow UI (отдельный терминал)
poetry run mlflow ui --port 5000
```

---

## Обоснование выбора хранилища данных

**PostgreSQL** — основное хранилище обработанных данных:
- Данные структурированы: `(id, text, label, text_lemm, created_at)`
- Объём небольшой (< 10k строк) — реляционная БД достаточна
- `ON CONFLICT DO NOTHING` обеспечивает дедупликацию при инкрементах
- Hadoop HDFS и Spark избыточны при таком масштабе
- S3 — только для хранения артефактов MLflow (опционально)


## 5. Экспорт в onnx (Лаб 3)
```bash
cd onnx
python export.py
```

## 6. Запуск сервиса bentoml (Лаб 3)
```bash
cd bentoml
bentoml serve service.py
``` 
## 7. Нагрузочное тестирование с locust (Лаб 3)
```bash
cd locust
# 10 пользователей, скорость спауна 2/сек, длительность 2 минуты
locust -f locustfile.py --host http://localhost:3000 -u 10 -r 2 -t 120s --html report.html
```

## 8. Grafana (Лаб 3)
```bash
cd monitoring
docker compose up -d
```
Логин admin пароль admin
