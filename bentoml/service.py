import bentoml
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../models/rubert_tiny2_tarot")
session = ort.InferenceSession("../onnx/models/rubert_tiny2_tarot/model.onnx")


@bentoml.service(resources={"cpu": "10"})
class TarotClassifier:
    @bentoml.api
    def predict(self, text: str) -> dict:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        ort_inputs = {k: inputs[k] for k in inputs}

        logits = session.run(None, ort_inputs)[0][0]

        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        return {
            "class_id": int(np.argmax(probs)),
            "probabilities": probs.tolist()
        }