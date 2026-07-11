import os
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../models/rubert_tiny2_tarot"
ONNX_PATH = "./models/rubert_tiny2_tarot/model.onnx"
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

model.config._attn_implementation = "eager"
model.config.use_flash_attention_2 = False
model.config.use_cache = False

dummy_text = "dummy"
inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", max_length=512)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch"}
    },
    opset_version=16,
    do_constant_folding=True,
    export_params=True
)
print("Экспорт завершён")

with torch.no_grad():
    pt_logits = model(input_ids, attention_mask, token_type_ids).logits.numpy()

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
ort_inputs = {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy(),
    "token_type_ids": token_type_ids.numpy()
}
onnx_logits = sess.run(None, ort_inputs)[0]

print(f"PyTorch: {pt_logits[0][:3]}")
print(f"ONNX   : {onnx_logits[0][:3]}")