"""
src/data/preprocessing.py
Очистка текста, лемматизация, дедупликация, сохранение в PostgreSQL / CSV.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import pymorphy3
from sqlalchemy import create_engine, text


morph = pymorphy3.MorphAnalyzer()


# ---------------------------------------------------------------------------
# Очистка
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Базовая очистка: убирает HTML, ссылки, лишние символы."""
    text = str(text).strip()
    text = re.sub(r"<[^>]+>", "", text)               # HTML-теги
    text = re.sub(r"http\S+", "", text)                # ссылки
    text = re.sub(r"[^\w\s\?\!\.\,\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize(text: str) -> str:
    """Лемматизация через pymorphy3."""
    tokens = text.lower().split()
    lemmas = []
    for token in tokens:
        token_clean = re.sub(r"[^\w]", "", token)
        if not token_clean:
            continue
        parsed = morph.parse(token_clean)
        if parsed:
            lemmas.append(parsed[0].normal_form)
    return " ".join(lemmas)


# ---------------------------------------------------------------------------
# Полный пайплайн предобработки
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
    """
    Полный пайплайн:
      1. clean_text
      2. Дедупликация по оригинальному тексту
      3. Фильтрация по длине
      4. Лемматизация → колонка text_lemm

    Args:
        df:       DataFrame с колонками text, label
        min_len:  минимальная длина текста (символов)

    Returns:
        DataFrame с колонками: text, label, text_lemm
    """
    print("[preprocess] Шаг 1/4: очистка текста...")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)

    print("[preprocess] Шаг 2/4: дедупликация...")
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"             {before} → {len(df)} (удалено {before - len(df)})")

    print("[preprocess] Шаг 3/4: фильтрация по длине...")
    df = df[df["text"].str.len() >= min_len].reset_index(drop=True)
    print(f"             Осталось: {len(df)} строк")

    print("[preprocess] Шаг 4/4: лемматизация (может занять 1-2 мин)...")
    t0 = time.time()
    df["text_lemm"] = df["text"].apply(lemmatize)
    print(f"             Готово за {time.time() - t0:.1f} сек")

    print(f"\n[preprocess] Готово. Распределение классов:")
    print(df["label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def save_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Сохраняет DataFrame в CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[preprocess] Сохранено в CSV: {path} ({len(df)} строк)")


def save_to_postgres(df: pd.DataFrame, postgres_url: str, table_name: str = "questions") -> None:
    """
    Сохраняет данные в PostgreSQL.
    При конфликте по тексту (уникальный ключ) — игнорирует дубликаты (upsert-style).

    Обоснование выбора PostgreSQL:
      - Данные структурированы (text, label, text_lemm, created_at)
      - Объём небольшой (< 10k строк), реляционная БД достаточна
      - Нужна поддержка инкрементов с дедупликацией по ключу
      - Hadoop HDFS избыточен для такого масштаба
      - S3 используется только для хранения артефактов MLflow
    """
    engine = create_engine(postgres_url)

    # Создаём целевую таблицу если не существует
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id         SERIAL PRIMARY KEY,
                text       TEXT UNIQUE NOT NULL,
                label      VARCHAR(32) NOT NULL,
                text_lemm  TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

    # Временная таблица → INSERT ... ON CONFLICT DO NOTHING
    tmp_table = f"{table_name}_tmp"

    # 🔧 FIX: используем bulk_insert_mappings вместо to_sql для совместимости
    records = df[["text", "label", "text_lemm"]].to_dict(orient="records")

    with engine.begin() as conn:
        # Создаём временную таблицу
        conn.execute(text(f"""
            CREATE TEMP TABLE IF NOT EXISTS {tmp_table} (
                text       TEXT,
                label      VARCHAR(32),
                text_lemm  TEXT
            ) ON COMMIT DROP
        """))

        # Массовая вставка через bulk_insert_mappings
        conn.execute(
            text(f"INSERT INTO {tmp_table} (text, label, text_lemm) VALUES (:text, :label, :text_lemm)"),
            records
        )

        # UPSERT: копируем в целевую таблицу, игнорируя дубликаты
        conn.execute(text(f"""
            INSERT INTO {table_name} (text, label, text_lemm)
            SELECT text, label, text_lemm FROM {tmp_table}
            ON CONFLICT (text) DO NOTHING
        """))
        # Временная таблица автоматически удалится (ON COMMIT DROP)

    print(f"[preprocess] Данные сохранены в PostgreSQL: таблица '{table_name}'")


def load_from_postgres(postgres_url: str, table_name: str = "questions") -> pd.DataFrame:
    """Загружает все данные из PostgreSQL."""
    engine = create_engine(postgres_url)
    df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY id", engine)
    print(f"[preprocess] Загружено из PostgreSQL: {len(df)} строк")
    return df
