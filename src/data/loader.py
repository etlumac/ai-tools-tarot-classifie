"""
src/data/loader.py
Загрузка данных из Google Sheets и локальных CSV.
Поддерживает первичную загрузку и инкрементальное обновление.
"""
from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

import pandas as pd


SHEETS = {
    "love": "love",
    "self": "self",
    "social": "social",
    "health": "health",
    "career": "career",
}


def _load_sheet(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame | None:
    """Загружает один лист Google Sheets через публичный CSV-экспорт."""
    encoded = urllib.parse.quote(sheet_name)
    url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded}"
    )
    try:
        df = pd.read_csv(url, header=None)
        return df
    except Exception as exc:
        print(f"[loader] Ошибка загрузки листа '{sheet_name}': {exc}")
        return None


def load_from_sheets(spreadsheet_id: str, sheets: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Загружает все листы из Google Sheets и собирает в единый DataFrame.

    Returns:
        DataFrame с колонками: text, label
    """
    if sheets is None:
        sheets = SHEETS

    all_rows: list[dict] = []
    print("[loader] Загружаем датасет из Google Sheets...")

    for sheet_name, label in sheets.items():
        print(f"  → лист «{sheet_name}» (class={label})")
        df = _load_sheet(spreadsheet_id, sheet_name)
        if df is None:
            continue

        texts: list[str] = []
        for col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            vals = [v.strip() for v in vals if v.strip() and v.strip().lower() not in ("nan", "")]
            texts.extend(vals)

        # Убираем строки без кириллицы и слишком короткие
        texts = [t for t in texts if len(t) >= 5 and re.search(r"[а-яёА-ЯЁ]", t)]
        print(f"     {len(texts)} вопросов")

        for text in texts:
            all_rows.append({"text": text, "label": label})

    df_raw = pd.DataFrame(all_rows)
    print(f"[loader] Итого загружено: {len(df_raw)} вопросов")
    return df_raw


def load_from_csv(path: str | Path) -> pd.DataFrame:
    """Загружает данные из локального CSV файла."""
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, (
        "CSV должен содержать колонки 'text' и 'label'"
    )
    print(f"[loader] Загружено из {path}: {len(df)} строк")
    return df


def load_increment(increment_path: str | Path, base_path: str | Path) -> pd.DataFrame:
    """
    Загружает инкрементальные данные и объединяет с основным датасетом.
    Дедуплицирует по тексту.

    Args:
        increment_path: путь к CSV с новыми данными
        base_path: путь к текущему основному CSV

    Returns:
        Объединённый DataFrame без дубликатов
    """
    base = load_from_csv(base_path)
    increment = load_from_csv(increment_path)

    before = len(base)
    combined = pd.concat([base, increment], ignore_index=True)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    after = len(combined)

    print(f"[loader] Инкремент: {before} + {len(increment)} → {after} (новых: {after - before})")
    return combined
