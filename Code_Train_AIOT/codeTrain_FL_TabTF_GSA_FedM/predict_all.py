import os
import torch
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tab_transformer_pytorch import TabTransformer
from data_processing import load_and_process_data

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../fl_40_outputs/models", "aggregated_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "../fl_40_outputs/models", "scaler_server.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../fl_40_outputs/models", "label_encoder_server.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../dataset", "data_All_1k.csv")

X_raw, y, categorical_cols, num_classes, _, _ = load_and_process_data(DATA_PATH)

scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

cont_cols = [c for c in X_raw.columns if c not in categorical_cols]
X_cont = scaler.transform(X_raw[cont_cols]) if cont_cols else X_raw.values
X_cat = X_raw[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_raw), 0))

X_cont = torch.tensor(X_cont, dtype=torch.float32)
X_cat = torch.tensor(X_cat, dtype=torch.long)
y_true_tensor = torch.tensor(y, dtype=torch.long)

hyperparams = {
    "categories": [X_raw[c].nunique() for c in categorical_cols],
    "num_continuous": X_cont.shape[1],
    "dim": 128,
    "dim_out": num_classes,
    "depth": 6,
    "heads": 8,
    "attn_dropout": 0.3,
    "ff_dropout": 0.3,
    "mlp_hidden_mults": (4, 2),
    "mlp_act": torch.nn.ReLU()
}
model = TabTransformer(**hyperparams)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    outputs = model(X_cat, X_cont)
    preds = torch.argmax(outputs, dim=1).numpy()

pred_labels = label_encoder.inverse_transform(preds)
true_labels = label_encoder.inverse_transform(y)

acc = accuracy_score(y, preds)
f1 = f1_score(y, preds, average="weighted", zero_division=0)
precision = precision_score(y, preds, average="weighted", zero_division=0)
recall = recall_score(y, preds, average="weighted", zero_division=0)

print(f"[RESULT] Accuracy: {acc:.4f}")
print(f"[RESULT] F1-score: {f1:.4f}")
print(f"[RESULT] Precision: {precision:.4f}")
print(f"[RESULT] Recall: {recall:.4f}")

correct = 0
wrong = 0

print("\n[DETAIL] Dự đoán vs Thực tế:")
for i in range(len(pred_labels)):
    if pred_labels[i] == true_labels[i]:
        print(f"Row {i+1}: Predict = {pred_labels[i]} ✅  | True = {true_labels[i]}")
        correct += 1
    else:
        print(f"Row {i+1}: Predict = {pred_labels[i]} ❌  | True = {true_labels[i]}")
        wrong += 1

print("\n[SUMMARY]")
print(f"✅ Đúng: {correct}")
print(f"❌ Sai: {wrong}")
print(f"Tỉ lệ đúng: {correct / len(pred_labels) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y, preds))

results_df = pd.DataFrame({
    'Sample': [f"Sample {i+1}" for i in range(len(pred_labels))],
    'Predicted': pred_labels,
    'True': true_labels
})

CSV_PATH = os.path.join(BASE_DIR, "../fl_40_outputs/logs", f"prediction_details_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

results_df.to_csv(CSV_PATH, index=False, encoding='utf-8')

print(f"Chi tiết dự đoán đã được lưu tại: {CSV_PATH}")