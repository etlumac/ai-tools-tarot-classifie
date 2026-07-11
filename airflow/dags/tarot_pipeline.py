"""
airflow/dags/tarot_pipeline.py
DAG для оркестрации пайплайна: загрузка → предобработка → сохранение в PostgreSQL.
Запускается ежедневно; поддерживает инкрементальное обновление данных.

Деплой:
  1. Скопировать файл в ~/airflow/dags/
  2. airflow db init
  3. airflow webserver --port 8080 &
  4. airflow scheduler &
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# ---------------------------------------------------------------------------
# Конфигурация DAG
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

SPREADSHEET_ID = "1HUUrXxlB9QQ4VpcLNYNOwZMJZsww5c-dmQnvTHLejGs"
RAW_CSV        = "/opt/tarot/data/raw/questions.csv"
INCREMENT_CSV  = "/opt/tarot/data/raw/increment.csv"
PROCESSED_CSV  = "/opt/tarot/data/processed/questions_lemm.csv"
POSTGRES_URL   = "postgresql://tarot:tarot@localhost:5432/tarot_db"


# ---------------------------------------------------------------------------
# Задачи (Tasks)
# ---------------------------------------------------------------------------

def task_load_from_sheets(**context) -> None:
    """
    T1: Загрузка данных из Google Sheets.
    При первом запуске — полная загрузка.
    При последующих — только проверка на новые строки (инкремент).
    """
    import sys
    sys.path.insert(0, "/opt/tarot")
    from src.data.loader import load_from_sheets, load_from_csv, load_increment

    execution_date = context["execution_date"]
    is_first_run   = not Path(RAW_CSV).exists()

    if is_first_run:
        print("[DAG] Первый запуск — полная загрузка из Sheets")
        df = load_from_sheets(SPREADSHEET_ID)
        Path(RAW_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False, encoding="utf-8")
        print(f"[DAG] Сохранено: {len(df)} строк → {RAW_CSV}")
    else:
        # Инкрементальная загрузка: забираем актуальную версию из Sheets
        # и ищем новые строки относительно текущего RAW_CSV
        print(f"[DAG] Инкрементальная загрузка (дата: {execution_date.date()})")
        df_new = load_from_sheets(SPREADSHEET_ID)
        df_new.to_csv(INCREMENT_CSV, index=False, encoding="utf-8")

        df_combined = load_increment(INCREMENT_CSV, RAW_CSV)
        df_combined.to_csv(RAW_CSV, index=False, encoding="utf-8")
        print(f"[DAG] Обновлённый датасет: {len(df_combined)} строк → {RAW_CSV}")


def task_preprocess(**context) -> None:
    """
    T2: Очистка, лемматизация, сохранение в CSV и PostgreSQL.
    Читает RAW_CSV → пишет PROCESSED_CSV + PostgreSQL.
    """
    import sys
    sys.path.insert(0, "/opt/tarot")
    from src.data.loader import load_from_csv
    from src.data.preprocessing import preprocess, save_to_csv, save_to_postgres

    df_raw = load_from_csv(RAW_CSV)
    df_processed = preprocess(df_raw, min_len=5)

    # Сохраняем в CSV (быстрый доступ для обучения)
    save_to_csv(df_processed, PROCESSED_CSV)

    # Сохраняем в PostgreSQL (основное хранилище, инкрементальный upsert)
    try:
        save_to_postgres(df_processed, POSTGRES_URL, table_name="questions")
    except Exception as exc:
        print(f"[DAG] Предупреждение: PostgreSQL недоступен — {exc}")
        print("[DAG] Данные сохранены только в CSV (продолжаем)")


def task_validate_data(**context) -> None:
    """
    T3: Базовая валидация данных после предобработки.
    Проверяет: количество классов, минимальный размер выборки, отсутствие NaN.
    """
    import sys
    import pandas as pd
    sys.path.insert(0, "/opt/tarot")

    REQUIRED_LABELS = {"love", "self", "social", "health", "career"}
    MIN_SAMPLES_PER_CLASS = 10

    df = pd.read_csv(PROCESSED_CSV)

    # Проверка колонок
    assert {"text", "label", "text_lemm"}.issubset(df.columns), \
        f"Отсутствуют обязательные колонки. Есть: {list(df.columns)}"

    # Проверка NaN
    nan_count = df[["text", "label", "text_lemm"]].isna().sum().sum()
    assert nan_count == 0, f"Найдены NaN: {nan_count}"

    # Проверка классов
    found_labels = set(df["label"].unique())
    assert REQUIRED_LABELS == found_labels, \
        f"Ожидаемые классы: {REQUIRED_LABELS}, найдено: {found_labels}"

    # Минимальный размер класса
    counts = df["label"].value_counts()
    for label, count in counts.items():
        assert count >= MIN_SAMPLES_PER_CLASS, \
            f"Мало примеров для класса '{label}': {count} < {MIN_SAMPLES_PER_CLASS}"

    print(f"[DAG] Валидация пройдена")
    print(f"       Всего строк: {len(df)}")
    print(f"       Классы: {counts.to_dict()}")


def task_notify_success(**context) -> None:
    """T4: Уведомление об успешном завершении пайплайна."""
    execution_date = context["execution_date"]
    import pandas as pd
    df = pd.read_csv(PROCESSED_CSV)
    print(f"""
   Пайплайн завершён успешно
   Дата запуска:  {execution_date.date()}
   Строк в датасете: {len(df)}
   Обработанный файл: {PROCESSED_CSV}
   PostgreSQL: {POSTGRES_URL}
   Следующий запуск: завтра в 02:00 UTC
    """)


# ---------------------------------------------------------------------------
# Определение DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="tarot_data_pipeline",
    description="Пайплайн предобработки данных для AI-Таролога",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",  # ежедневно в 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["tarot", "ml", "preprocessing"],
    doc_md="""
## tarot_data_pipeline

Автоматический пайплайн предобработки данных.

### Шаги:
1. **load_from_sheets** — загрузка/обновление из Google Sheets
2. **preprocess** — очистка, лемматизация, сохранение в PostgreSQL
3. **validate_data** — проверка качества данных
4. **notify_success** — лог об успехе

### Запуск вручную:
```bash
airflow dags trigger tarot_data_pipeline
```
    """,
) as dag:

    t1_load = PythonOperator(
        task_id="load_from_sheets",
        python_callable=task_load_from_sheets,
        provide_context=True,
    )

    t2_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
        provide_context=True,
    )

    t3_validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
        provide_context=True,
    )

    t4_notify = PythonOperator(
        task_id="notify_success",
        python_callable=task_notify_success,
        provide_context=True,
    )

    # Граф зависимостей: T1 → T2 → T3 → T4
    t1_load >> t2_preprocess >> t3_validate >> t4_notify
