"""
src/models/predict.py
Инференс: загрузка моделей и предсказание по тексту.
Поддерживает одиночные тексты и батчи. Используется в BentoML-сервисе (Лаб 3).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np


LABELS = ["love", "self", "social", "health", "career"]
ARTIFACTS_DIR = Path("models")


class TarotPredictor:
    """
    Унифицированный интерфейс предсказания для всех моделей.
    Поддерживает: logreg, fasttext, catboost, bert, ensemble.
    """

    def __init__(self, model_type: Literal["logreg", "fasttext", "catboost", "bert", "ensemble"] = "logreg"):
        self.model_type = model_type
        self._logreg = None
        self._fasttext = None
        self._catboost = None
        self._bert = None
        self._tokenizer = None
        self._le = None  # LabelEncoder для catboost
        self._ensemble_weights = (0.3, 0.2, 0.5)  # lr, cb, bert (из grid search)

        self._load_models(model_type)

    def _load_models(self, model_type: str):
        if model_type in ("logreg", "ensemble"):
            path = ARTIFACTS_DIR / "tfidf_logreg_v1.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self._logreg = pickle.load(f)
                print(f"[predict] LogReg загружен: {path}")

        if model_type in ("fasttext", "ensemble"):
            path = ARTIFACTS_DIR / "fasttext_v1.bin"
            if path.exists():
                try:
                    import fasttext
                    self._fasttext = fasttext.load_model(str(path))
                    print(f"[predict] fastText загружен: {path}")
                except ImportError:
                    print("[predict] fastText не установлен")

        if model_type in ("catboost", "ensemble"):
            path = ARTIFACTS_DIR / "catboost_v1.cbm"
            if path.exists():
                try:
                    from catboost import CatBoostClassifier
                    from sklearn.preprocessing import LabelEncoder
                    self._catboost = CatBoostClassifier()
                    self._catboost.load_model(str(path))
                    self._le = LabelEncoder().fit(LABELS)
                    print(f"[predict] CatBoost загружен: {path}")
                except ImportError:
                    print("[predict] CatBoost не установлен")

        if model_type in ("bert", "ensemble"):
            path = ARTIFACTS_DIR / "rubert_tiny2_tarot"
            if Path(path).exists():
                try:
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(str(path))
                    self._bert = AutoModelForSequenceClassification.from_pretrained(str(path))
                    self._bert.eval()
                    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self._bert.to(self._device)
                    print(f"[predict] BERT загружен: {path}")
                except ImportError:
                    print("[predict] PyTorch/Transformers не установлены")

    # ------------------------------------------------------------------
    # Предсказания отдельных моделей → probas shape (N, 5) в порядке LABELS
    # ------------------------------------------------------------------

    def _logreg_probas(self, texts: list[str]) -> np.ndarray:
        if self._logreg is None:
            raise RuntimeError("LogReg модель не загружена")
        import pymorphy3, re, time
        morph = pymorphy3.MorphAnalyzer()
        def lemmatize(t):
            tokens = t.lower().split()
            lemmas = []
            for tok in tokens:
                tok_clean = re.sub(r"[^\w]", "", tok)
                if not tok_clean: continue
                parsed = morph.parse(tok_clean)
                if parsed: lemmas.append(parsed[0].normal_form)
            return " ".join(lemmas)
        texts_lemm = [lemmatize(t) for t in texts]
        raw = self._logreg.predict_proba(texts_lemm)
        classes = list(self._logreg.classes_)
        return np.array([[row[classes.index(l)] for l in LABELS] for row in raw])

    def _fasttext_probas(self, texts: list[str]) -> np.ndarray:
        if self._fasttext is None:
            raise RuntimeError("fastText модель не загружена")
        result = []
        for t in texts:
            labels, probs = self._fasttext.predict(t.replace("\n", " "), k=len(LABELS))
            ltp = {l.replace("__label__", ""): p for l, p in zip(labels, probs)}
            result.append([ltp.get(l, 0.0) for l in LABELS])
        return np.array(result)

    def _catboost_probas(self, texts: list[str]) -> np.ndarray:
        if self._catboost is None:
            raise RuntimeError("CatBoost модель не загружена")
        # CatBoost ожидает TF-IDF признаки — для инференса нужен vectorizer из logreg
        if self._logreg is None:
            raise RuntimeError("Для CatBoost нужен загруженный LogReg (для TF-IDF)")
        import pymorphy3, re
        morph = pymorphy3.MorphAnalyzer()
        def lemmatize(t):
            tokens = t.lower().split()
            lemmas = []
            for tok in tokens:
                tok_clean = re.sub(r"[^\w]", "", tok)
                if not tok_clean: continue
                parsed = morph.parse(tok_clean)
                if parsed: lemmas.append(parsed[0].normal_form)
            return " ".join(lemmas)
        texts_lemm = [lemmatize(t) for t in texts]
        vectorizer = self._logreg.named_steps["tfidf"]
        tfidf_feats = vectorizer.transform(texts_lemm).toarray()
        extra = np.array([[
            len(t), len(t.split()), int("?" in t),
            int(any(w in t.split() for w in ["я", "мой", "моя", "меня", "мне"])),
            int(any(w in t.split() for w in ["он", "она", "партнёр", "муж", "жена"])),
            int(any(w in t.split() for w in ["работа", "деньги", "карьера"])),
            int(any(w in t.split() for w in ["здоровье", "тело", "болеть", "вес"])),
        ] for t in texts_lemm])
        X = np.hstack([tfidf_feats, extra])
        raw = self._catboost.predict_proba(X)
        cb_classes = list(self._le.classes_)
        return np.array([[row[cb_classes.index(l)] for l in LABELS] for row in raw])

    def _bert_probas(self, texts: list[str]) -> np.ndarray:
        if self._bert is None:
            raise RuntimeError("BERT модель не загружена")
        import torch
        import torch.nn.functional as F
        LABEL2ID = {l: i for i, l in enumerate(LABELS)}
        all_probas = []
        for text in texts:
            enc = self._tokenizer(text, max_length=64, padding="max_length",
                                   truncation=True, return_tensors="pt")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._bert(**enc).logits
            probas = F.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probas.append(probas)
        return np.array(all_probas)

    # ------------------------------------------------------------------
    # Основной интерфейс
    # ------------------------------------------------------------------

    def predict(self, texts: list[str] | str, confidence_threshold: float = 0.0) -> list[dict]:
        """
        Предсказывает класс и уверенность для каждого текста.

        Args:
            texts: строка или список строк
            confidence_threshold: если max_proba < threshold → label="uncertain"

        Returns:
            Список dict: {label, confidence, probas}
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "logreg":
            probas = self._logreg_probas(texts)
        elif self.model_type == "fasttext":
            probas = self._fasttext_probas(texts)
        elif self.model_type == "catboost":
            probas = self._catboost_probas(texts)
        elif self.model_type == "bert":
            probas = self._bert_probas(texts)
        elif self.model_type == "ensemble":
            w_lr, w_cb, w_bert = self._ensemble_weights
            probas = np.zeros((len(texts), len(LABELS)))
            if self._logreg:
                probas += w_lr * self._logreg_probas(texts)
            if self._catboost:
                probas += w_cb * self._catboost_probas(texts)
            if self._bert:
                probas += w_bert * self._bert_probas(texts)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        results = []
        for i, prob_row in enumerate(probas):
            max_conf = float(prob_row.max())
            label = LABELS[int(prob_row.argmax())]
            if confidence_threshold > 0 and max_conf < confidence_threshold:
                label = "uncertain"
            results.append({
                "label": label,
                "confidence": max_conf,
                "probas": {l: float(p) for l, p in zip(LABELS, prob_row)},
            })
        return results


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AI-Таролог: предсказание категории вопроса")
    parser.add_argument("text", nargs="*", default=["Вернётся ли он ко мне?"], help="Текст вопроса")
    parser.add_argument("-m", "--model",
                        choices=["logreg", "fasttext", "catboost", "bert", "ensemble"],
                        default="logreg", help="Модель для предсказания")
    parser.add_argument("-t", "--threshold", type=float, default=0.0, help="Порог уверенности")
    args = parser.parse_args()

    text = " ".join(args.text)
    predictor = TarotPredictor(model_type=args.model)
    result = predictor.predict(text, confidence_threshold=args.threshold)[0]

    print(f"Текст:       {text}")
    print(f"Модель:      {args.model}")
    print(f"Класс:       {result['label']}")
    print(f"Уверенность: {result['confidence']:.2%}")
    print("Вероятности:")
    for label, prob in sorted(result["probas"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {label:8s}: {prob:.3f} {bar}")


if __name__ == "__main__":
    main()
