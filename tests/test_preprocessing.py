"""Базовые тесты предобработки данных."""
import pandas as pd
import sys
sys.path.insert(0, ".")

from src.data.preprocessing import clean_text, lemmatize, preprocess


def test_clean_text_removes_html():
    assert "<b>" not in clean_text("<b>Привет</b>")

def test_clean_text_removes_urls():
    assert "http" not in clean_text("Привет http://example.com пока")

def test_lemmatize_returns_string():
    result = lemmatize("Вернётся ли он ко мне?")
    assert isinstance(result, str) and len(result) > 0

def test_preprocess_output_columns():
    df = pd.DataFrame({
        "text": ["Вернётся ли он ко мне?", "Найду ли я работу?"],
        "label": ["love", "career"]
    })
    result = preprocess(df)
    assert {"text", "label", "text_lemm"}.issubset(result.columns)

def test_preprocess_removes_short_texts():
    df = pd.DataFrame({"text": ["ok", "Вернётся ли он ко мне?"], "label": ["love", "love"]})
    result = preprocess(df, min_len=5)
    assert len(result) == 1
