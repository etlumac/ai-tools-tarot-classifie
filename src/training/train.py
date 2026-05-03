"""
src/training/train.py
Обучение всех моделей с логированием в MLflow.
Запуск: poetry run train  (или python -m src.training.train)
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Опциональные импорты
try:
    import fasttext
    HAS_FASTTEXT = True
except ImportError:
    HAS_FASTTEXT = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_BERT = True
except ImportError:
    HAS_BERT = False


LABELS = ["love", "self", "social", "health", "career"]
ARTIFACTS_DIR = Path("models")
DATA_PATH = Path("data/processed/questions_lemm.csv")
DATA_ORIG_PATH = Path("data/raw/questions.csv")  # оригинальные тексты для BERT


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_data() -> tuple:
    """Загружает данные и делает стратифицированный split."""
    df = pd.read_csv(DATA_PATH)
    df_orig = pd.read_csv(DATA_ORIG_PATH) if DATA_ORIG_PATH.exists() else df.copy()

    X = df["text_lemm"].values
    y = df["label"].values
    X_orig = df_orig["text"].values if "text" in df_orig.columns else X

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Оригинальные тексты для BERT (тот же split)
    X_orig_train, X_orig_temp, _, _ = train_test_split(
        X_orig, y, test_size=0.30, random_state=42, stratify=y
    )
    X_orig_val, X_orig_test, _, _ = train_test_split(
        X_orig_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_orig_train, X_orig_val, X_orig_test,
    )


def log_metrics(y_true, y_pred, prefix: str = "test") -> dict:
    """Считает и логирует метрики в MLflow."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)

    mlflow.log_metrics({
        f"{prefix}_accuracy": acc,
        f"{prefix}_f1_macro": f1,
        **{f"{prefix}_recall_{lbl}": report[lbl]["recall"] for lbl in LABELS if lbl in report},
    })

    print(f"  [{prefix}] Accuracy={acc:.3f}  Macro F1={f1:.3f}")
    return {"accuracy": acc, "f1_macro": f1, "report": report}


# ---------------------------------------------------------------------------
# Обучение моделей
# ---------------------------------------------------------------------------

def train_logreg(X_train, X_val, X_test, y_train, y_val, y_test) -> Pipeline:
    """TF-IDF + LogisticRegression с GridSearchCV."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="tfidf_logreg"):
        mlflow.set_tags({
            "model_type": "logreg",
            "vectorizer": "tfidf",
            "framework": "sklearn",
            "data_version": str(DATA_PATH),
        })

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
        ])
        param_grid = {
            "tfidf__max_features": [3000, 5000, 8000],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "clf__C": [0.1, 0.5, 1.0, 5.0],
            "clf__class_weight": [None, "balanced"],
        }

        mlflow.log_param("param_grid", str(param_grid))
        mlflow.log_param("cv_folds", 3)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("cv_f1_macro", gs.best_score_)

        log_metrics(y_val, best.predict(X_val), prefix="val")
        log_metrics(y_test, best.predict(X_test), prefix="test")

        # Сохраняем артефакт
        model_path = ARTIFACTS_DIR / "tfidf_logreg_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best, f)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(best, "model")

        print(f"  Лучшие параметры: {gs.best_params_}")
        return best


def train_fasttext(X_train, X_val, X_test, y_train, y_val, y_test) -> object:
    """fastText supervised."""
    if not HAS_FASTTEXT:
        print("[train] fastText не установлен, пропускаем.")
        return None

    import tempfile
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def write_ft_file(X, y, path):
        with open(path, "w", encoding="utf-8") as f:
            for text, label in zip(X, y):
                f.write(f"__label__{label} {text.replace(chr(10), ' ')}\n")

    with mlflow.start_run(run_name="fasttext"):
        params = dict(dim=100, epoch=50, lr=0.5, wordNgrams=2,
                      minn=2, maxn=6, loss="softmax", minCount=1)
        mlflow.set_tags({"model_type": "fasttext", "data_version": str(DATA_PATH)})
        mlflow.log_params(params)

        train_f = "/tmp/ft_train.txt"
        val_f   = "/tmp/ft_val.txt"
        test_f  = "/tmp/ft_test.txt"
        write_ft_file(X_train, y_train, train_f)
        write_ft_file(X_val,   y_val,   val_f)
        write_ft_file(X_test,  y_test,  test_f)

        model = fasttext.train_supervised(input=train_f, verbose=0, **params)

        def ft_predict(texts):
            return [model.predict(t.replace("\n", " "))[0][0].replace("__label__", "")
                    for t in texts]

        log_metrics(y_val,  ft_predict(X_val),  prefix="val")
        log_metrics(y_test, ft_predict(X_test), prefix="test")

        model_path = ARTIFACTS_DIR / "fasttext_v1.bin"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        return model


def train_catboost(X_train, X_val, X_test, y_train, y_val, y_test,
                   vectorizer: TfidfVectorizer) -> object:
    """CatBoost на TF-IDF признаках."""
    if not HAS_CATBOOST:
        print("[train] CatBoost не установлен, пропускаем.")
        return None

    from sklearn.preprocessing import LabelEncoder
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def extra_features(texts):
        feats = []
        for t in texts:
            feats.append([
                len(t), len(t.split()), int("?" in t),
                int(any(w in t.split() for w in ["я", "мой", "моя", "меня", "мне"])),
                int(any(w in t.split() for w in ["он", "она", "партнёр", "муж", "жена"])),
                int(any(w in t.split() for w in ["работа", "деньги", "карьера"])),
                int(any(w in t.split() for w in ["здоровье", "тело", "болеть", "вес"])),
            ])
        return np.array(feats)

    with mlflow.start_run(run_name="catboost_tfidf"):
        params = dict(iterations=500, learning_rate=0.05, depth=6,
                      loss_function="MultiClass", eval_metric="Accuracy",
                      early_stopping_rounds=50, random_seed=42)
        mlflow.set_tags({"model_type": "catboost", "data_version": str(DATA_PATH)})
        mlflow.log_params(params)
        mlflow.log_param("extra_features", True)

        X_tr = np.hstack([vectorizer.transform(X_train).toarray(), extra_features(X_train)])
        X_vl = np.hstack([vectorizer.transform(X_val).toarray(),   extra_features(X_val)])
        X_te = np.hstack([vectorizer.transform(X_test).toarray(),  extra_features(X_test)])

        le = LabelEncoder().fit(LABELS)
        model = CatBoostClassifier(**params, verbose=100)
        model.fit(X_tr, le.transform(y_train),
                  eval_set=(X_vl, le.transform(y_val)),
                  use_best_model=True)

        log_metrics(y_val,  le.inverse_transform(model.predict(X_vl).flatten()), prefix="val")
        log_metrics(y_test, le.inverse_transform(model.predict(X_te).flatten()), prefix="test")

        model_path = ARTIFACTS_DIR / "catboost_v1.cbm"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        return model


def train_bert(X_orig_train, X_orig_val, X_orig_test,
               y_train, y_val, y_test) -> object:
    """ruBERT-tiny2 fine-tuning."""
    if not HAS_BERT:
        print("[train] PyTorch/Transformers не установлены, пропускаем.")
        return None

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "cointegrated/rubert-tiny2"
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}
    BATCH_SIZE, MAX_LEN, EPOCHS, LR = 32, 64, 15, 3e-5

    class TarotDataset(Dataset):
        def __init__(self, texts, labels, tok):
            self.texts = texts
            self.labels = [LABEL2ID[l] for l in labels]
            self.tok = tok

        def __len__(self): return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(self.texts[idx], max_length=MAX_LEN,
                           padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="rubert_tiny2"):
        mlflow.set_tags({"model_type": "bert", "base_model": MODEL_NAME,
                         "data_version": str(DATA_ORIG_PATH)})
        mlflow.log_params({"batch_size": BATCH_SIZE, "max_len": MAX_LEN,
                           "epochs": EPOCHS, "lr": LR, "model": MODEL_NAME})

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=5, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        ).to(DEVICE)

        train_loader = DataLoader(TarotDataset(X_orig_train, y_train, tokenizer),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(TarotDataset(X_orig_val,   y_val,   tokenizer), batch_size=BATCH_SIZE)
        test_loader  = DataLoader(TarotDataset(X_orig_test,  y_test,  tokenizer), batch_size=BATCH_SIZE)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_f1, best_state = 0.0, None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in train_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                labs = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                out  = model(input_ids=ids, attention_mask=mask, labels=labs)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                total_loss += out.loss.item()

            # Валидация
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(input_ids=batch["input_ids"].to(DEVICE),
                                attention_mask=batch["attention_mask"].to(DEVICE))
                    val_preds.extend(out.logits.argmax(-1).cpu().numpy())
                    val_true.extend(batch["label"].numpy())

            val_f1  = f1_score([ID2LABEL[i] for i in val_true],
                                [ID2LABEL[i] for i in val_preds], average="macro")
            val_acc = accuracy_score(val_true, val_preds)
            avg_loss = total_loss / len(train_loader)

            # Логируем каждую эпоху → кривые обучения в MLflow
            mlflow.log_metrics({"train_loss": avg_loss, "val_f1_macro": val_f1,
                                 "val_accuracy": val_acc}, step=epoch)
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | loss={avg_loss:.4f} | val_f1={val_f1:.3f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

        # Тест
        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE))
                test_preds.extend(out.logits.argmax(-1).cpu().numpy())
                test_true.extend(batch["label"].numpy())

        log_metrics([ID2LABEL[i] for i in test_true],
                    [ID2LABEL[i] for i in test_preds], prefix="test")

        bert_path = ARTIFACTS_DIR / "rubert_tiny2_tarot"
        model.save_pretrained(bert_path)
        tokenizer.save_pretrained(bert_path)
        mlflow.log_artifacts(str(bert_path), artifact_path="bert_model")

        return model, tokenizer


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["storage"]["mlflow_tracking_uri"])
    mlflow.set_experiment("tarot-classifier")

    print("=" * 60)
    print("Загружаем данные...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_orig_train, X_orig_val, X_orig_test) = load_data()

    print("\n[1/4] TF-IDF + LogReg")
    logreg = train_logreg(X_train, X_val, X_test, y_train, y_val, y_test)
    vectorizer = logreg.named_steps["tfidf"]

    print("\n[2/4] fastText")
    train_fasttext(X_train, X_val, X_test, y_train, y_val, y_test)

    print("\n[3/4] CatBoost")
    train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)

    print("\n[4/4] ruBERT-tiny2")
    train_bert(X_orig_train, X_orig_val, X_orig_test, y_train, y_val, y_test)

    print("\nОбучение завершено. Все эксперименты залогированы в MLflow.")
    print(f"   mlflow ui --port 5000")


if __name__ == "__main__":
    main()
