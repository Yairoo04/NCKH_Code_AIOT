import io
import os
import time
import logging
from typing import List, Tuple, Dict
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans, DBSCAN
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import joblib
import flwr as fl

from flwr.server import ServerApp, ServerConfig
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from tab_transformer_pytorch import TabTransformer
from data_processing import load_and_process_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

OUTPUT_DIR = "fl_1k_DDoS_outputs"
DATA_PATH = "dataset/data_DDoS_1k.csv"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
IMAGES_SERVER_DIR = os.path.join(OUTPUT_DIR, "images", "server")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "log_server.txt")
AGGREGATED_MODEL_PATH = os.path.join(MODEL_DIR, "aggregated_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_server.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_server.pkl")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.pkl")
NUM_CONTINUOUS_PATH = os.path.join(MODEL_DIR, "num_continuous.pkl")
CAT_COLS_PATH = os.path.join(MODEL_DIR, "cat_cols.pkl")  # NEW: tên cột categorical (thứ tự cố định)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13
})

SIMILARITY_THRESHOLD = 0.7
LAMBDA_PERF = 0.4
LAMBDA_CONV = 0.6

BATCH_SIZE = 256
EPOCHS = 50

NUM_CLIENTS = 2

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a",
)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

class FLServerTabTransformer:
    def __init__(self):
        self.IMAGES_SERVER_DIR = IMAGES_SERVER_DIR
        self.MODEL_DIR = MODEL_DIR
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.clients = [f"client_{i}" for i in range(NUM_CLIENTS)]
        print("[SERVER] Initializing server.")
        logging.info("[SERVER] Initializing server.")
        ensure_dir(MODEL_DIR)
        ensure_dir(IMAGES_SERVER_DIR)

        X, y, categorical_cols, num_classes, le, _ = load_and_process_data(DATA_PATH)
        self.le = le
        print(f"[SERVER] Data: {len(X)} samples, {len(categorical_cols)} categorical cols, {num_classes} classes")
        logging.info(f"[SERVER] Data: {len(X)} samples, {len(categorical_cols)} categorical cols, {num_classes} classes")

        client_datasets = self.split_data_for_clients(X, y, NUM_CLIENTS)
        for i, df_client in enumerate(client_datasets):
            client_data_path = os.path.join(OUTPUT_DIR, f"client_{i}_data.csv")
            df_client.to_csv(client_data_path, index=False)
            print(f"[SERVER] Client {i} assigned {len(df_client)} samples")
            logging.info(f"[SERVER] Client {i} assigned {len(df_client)} samples")

        cont_cols = [c for c in X.columns if c not in categorical_cols]
        self.scaler = RobustScaler()
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 / 3, stratify=y_temp, random_state=42
        )

        if cont_cols:
            X_train_cont = self.scaler.fit_transform(X_train[cont_cols])
            X_val_cont = self.scaler.transform(X_val[cont_cols])
            X_test_cont = self.scaler.transform(X_test[cont_cols])
        else:
            X_train_cont = X_train.values
            X_val_cont = X_val.values
            X_test_cont = X_test.values

        # categorical as indices (int)
        X_train_cat = X_train[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_train), 0))
        X_val_cat = X_val[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_val), 0))
        X_test_cat = X_test[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_test), 0))

        # === SAVE METADATA for Clients ===
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(le, ENCODER_PATH)
        categories_sizes = [X[c].nunique() for c in categorical_cols]
        joblib.dump(categories_sizes, CATEGORIES_PATH)
        joblib.dump(len(cont_cols), NUM_CONTINUOUS_PATH)
        joblib.dump(categorical_cols, CAT_COLS_PATH)  # NEW

        self.X_train_cont = torch.tensor(X_train_cont, dtype=torch.float32)
        self.X_val_cont = torch.tensor(X_val_cont, dtype=torch.float32)
        self.X_test_cont = torch.tensor(X_test_cont, dtype=torch.float32)
        self.X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        self.X_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
        self.X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(self.X_train_cat, self.X_train_cont, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = TensorDataset(self.X_val_cat, self.X_val_cont, self.y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_classes = num_classes
        self.categorical_cols = categorical_cols

        self.val_loss_history: List[float] = []
        self.val_accuracy_history: List[float] = []
        self.f1_score_history: List[float] = []
        self.roc_auc_history: List[float] = []
        self.round_times: List[float] = []

        cont_dim = len(cont_cols)
        self.hyperparams = {
            "categories": categories_sizes,
            "num_continuous": cont_dim,
            "dim": 128,
            "dim_out": num_classes,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.3,
            "ff_dropout": 0.3,
            "mlp_hidden_mults": (4, 2),
            "mlp_act": torch.nn.ReLU(),
        }
        self.model = TabTransformer(**self.hyperparams)

        # === LOAD PREVIOUS GLOBAL MODEL IF EXISTS (for continued fine-tuning) ===
        if os.path.exists(AGGREGATED_MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(AGGREGATED_MODEL_PATH, map_location="cpu"))
                print("[SERVER] Loaded existing aggregated model for fine-tuning.")
                logging.info("[SERVER] Loaded existing aggregated model for fine-tuning.")
            except Exception as e:
                print(f"[SERVER] WARNING: Failed to load previous model: {e}")
                logging.warning(f"[SERVER] Failed to load previous model: {e}")

        print(f"[SERVER] TabTransformer ready - {self.hyperparams}")
        logging.info(f"[SERVER] TabTransformer ready - {self.hyperparams}")

    def _plot_training_progress(self):
        try:
            if not self.val_accuracy_history:
                return
            fig, ax1 = plt.subplots(figsize=(10, 6))
            rounds = list(range(1, len(self.val_accuracy_history) + 1))
            for i, acc in enumerate(self.val_accuracy_history):
                ax1.annotate(f"{acc:.2f}", (rounds[i], acc), textcoords="offset points", xytext=(0, 5),
                             ha='center', fontsize=8, color="blue")
            l1, = ax1.plot(rounds, self.val_accuracy_history, label="Val Accuracy", color="blue", marker='o')
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Val Accuracy / Val Loss", color="blue")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(0, 1)
            l2 = None
            if self.val_loss_history:
                l2, = ax1.plot(rounds, self.val_loss_history, label="Val Loss", color="orange", marker='*')

            ax2 = ax1.twinx()
            l3, = ax2.plot(rounds, self.f1_score_history, label="F1-Score", color="red", marker='s')
            l4 = None
            if self.roc_auc_history:
                l4, = ax2.plot(rounds, self.roc_auc_history, label="ROC-AUC", color="green", marker='^')

            ax2.set_ylabel("F1-Score / ROC-AUC", color="red")
            ax2.set_ylim(0, 1)
            plt.title("Training Progress - Server", fontweight="bold")
            lines = [l for l in [l1, l2, l3, l4] if l is not None]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
            fig.tight_layout()
            plt.savefig(os.path.join(self.IMAGES_SERVER_DIR, f"server_training_progress_round_{len(self.val_accuracy_history)}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("[SERVER] Đã lưu tiến trình huấn luyện.")
            logging.info("[SERVER] Đã lưu tiến trình huấn luyện.")
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ biểu đồ tiến trình huấn luyện: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ biểu đồ tiến trình huấn luyện: {e}")

    def _plot_round_time(self):
        try:
            if not self.round_times:
                return
            plt.figure(figsize=(8, 5))
            rounds = list(range(1, len(self.round_times) + 1))
            plt.plot(rounds, self.round_times, label="Thời gian vòng", color="green", marker='o')
            plt.xlabel("Vòng")
            plt.ylabel("Thời gian (giây)")
            plt.title("Thời gian vòng - Server")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            for i, time_s in enumerate(self.round_times):
                plt.annotate(f'{time_s:.1f}s', (i+1, time_s), textcoords="offset points", xytext=(0,10), ha='center')
            plt.savefig(os.path.join(self.IMAGES_SERVER_DIR, f"server_round_time_round_{len(self.round_times)}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("[SERVER] Đã lưu biểu đồ thời gian vòng.")
            logging.info("[SERVER] Đã lưu biểu đồ thời gian vòng.")
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ biểu đồ thời gian vòng: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ biểu đồ thời gian vòng: {e}")

    def _plot_prf_curve(self, y_true, y_scores, num_thresholds=30):
        try:
            thresholds = np.linspace(0, 1, num_thresholds)
            precisions, recalls, f1s = [], [], []

            if y_scores.ndim == 2 and y_scores.shape[1] > 1:  
                y_prob = y_scores.max(axis=1)
            else:
                y_prob = y_scores if y_scores.ndim == 1 else y_scores[:, 1]

            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                precisions.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
                recalls.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))
                f1s.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precisions, marker='o', label="Precision", color="skyblue")
            plt.plot(thresholds, recalls, marker='o', label="Recall", color="darkblue")
            plt.plot(thresholds, f1s, marker='o', label="F1", color="lightgreen")

            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Precision–Recall–F1 Curve - Server (Round {len(self.val_accuracy_history)})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            save_path = os.path.join(self.IMAGES_SERVER_DIR, f"server_prf_curve_round_{len(self.val_accuracy_history)}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[SERVER] Đã lưu Precision-Recall-F1 curve: {save_path}")
            logging.info(f"[SERVER] Đã lưu Precision-Recall-F1 curve")
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ PRF curve: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ PRF curve: {e}")

    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            label_counts = np.bincount(y_true, minlength=self.num_classes)
            top_5_indices = np.argsort(label_counts)[::-1][:5]
            mask = np.isin(y_true, top_5_indices)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            if len(y_true_filtered) == 0 or len(y_pred_filtered) == 0:
                raise ValueError("No data available for top 5 labels")

            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_5_indices)
            row_sums = cm.sum(axis=1)
            row_sums[row_sums == 0] = 1
            cm_percent = cm.astype('float') / row_sums[:, np.newaxis]
            cm_percent = np.nan_to_num(cm_percent)

            # Sử dụng LabelEncoder để lấy tên nhãn gốc
            top_5_labels = self.le.inverse_transform(top_5_indices)

            annot = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    percent = cm_percent[i, j] * 100
                    annot[i, j] = f"{percent:.1f}%"

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 18
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_percent * 100, annot=annot, fmt="", cmap='Blues', cbar=True,
                        xticklabels=top_5_labels, yticklabels=top_5_labels,
                        linewidths=0.5, linecolor='gray')

            ax.set_title(f'Confusion Matrix (%) - Server (Round {len(self.val_accuracy_history)})', 
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=18)
            ax.set_ylabel('True Label', fontsize=18)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = os.path.join(self.IMAGES_SERVER_DIR, f"server_confusion_matrix_round_{len(self.val_accuracy_history)}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SERVER] Saved confusion matrix at {save_path}")
            logging.info(f"[SERVER] Saved confusion matrix at {save_path}")
            
        except Exception as e:
            print(f"[SERVER] Error plotting confusion matrix: {str(e)}")
            logging.error(f"[SERVER] Error plotting confusion matrix: {str(e)}")

    def _plot_roc_curve(self, y_true, y_scores):
        try:
            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            plt.figure(figsize=(10, 6))
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
            colors = cycle(["blue", "red", "green", "orange", "purple"])
            for i, color in zip(range(self.num_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC={roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - Server")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.IMAGES_SERVER_DIR, "server_roc_curve.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print("[SERVER] Đã lưu ROC curve")
            logging.info("[SERVER] Đã lưu ROC curve")
        except Exception as e:
            print(f"[SERVER] Lỗi vẽ ROC curve: {str(e)}")
            logging.error(f"[SERVER] Lỗi vẽ ROC curve: {str(e)}")

    def split_data_for_clients(self, X_df: pd.DataFrame, y: np.ndarray, num_clients: int):
        y = np.asarray(y)
        labels = np.unique(y)
        idx_per_client = [[] for _ in range(num_clients)]
        for label in labels:
            idx_label = np.where(y == label)[0]
            np.random.shuffle(idx_label)
            splits = np.array_split(idx_label, num_clients)
            for i, part in enumerate(splits):
                idx_per_client[i].extend(part.tolist())

        client_dfs: List[pd.DataFrame] = []
        for i in range(num_clients):
            inds = np.array(idx_per_client[i], dtype=int)
            X_client = X_df.iloc[inds].copy()
            y_client = y[inds]
            df_client = X_client.copy()
            df_client["Label"] = y_client
            client_dfs.append(df_client)
            dist = dict(Counter(y_client))
            print(f"[SERVER] Client {i}: {len(df_client)} samples, label dist: {dist}")
            logging.info(f"[SERVER] Client {i}: {len(df_client)} samples, label dist: {dist}")

        total_rows = sum(len(df) for df in client_dfs)
        assert total_rows == len(X_df), "Tổng số mẫu sau chia phải bằng toàn bộ dữ liệu"
        return client_dfs

    def evaluate_val(self, model: torch.nn.Module) -> Dict[str, float]:
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        all_preds, all_labels, all_scores = [], [], []
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for cat_data, cont_data, labels in self.val_loader:
                out = model(cat_data, cont_data)
                loss = loss_fn(out, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
                all_scores.extend(torch.softmax(out, dim=1).numpy())

        val_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, np.array(all_scores), multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = 0.0
        return {"val_loss": val_loss, "val_accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}

    def evaluate_test(self, model: torch.nn.Module) -> Dict[str, float]:
        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        all_preds, all_labels, all_scores = [], [], []
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for cat_data, cont_data, labels in DataLoader(
                TensorDataset(self.X_test_cat, self.X_test_cont, self.y_test),
                batch_size=256, shuffle=False
            ):
                out = model(cat_data, cont_data)
                loss = loss_fn(out, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
                all_scores.extend(torch.softmax(out, dim=1).numpy())

        test_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, np.array(all_scores), multi_class="ovr", average="weighted")
        except ValueError:
            roc_auc = 0.0
        return {"test_loss": test_loss, "test_accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}


class FedMADE_GSA_Strategy(FedAvg):
    def __init__(self, server_obj, clients, eps=0.3, min_samples=3, lr=0.001, max_iter=100, gsa_thresh=0.7, evaluate_fn=None):
        super().__init__(evaluate_fn=evaluate_fn)
        self.server_obj = server_obj
        self.clients = clients

    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        metrics = self.server_obj.evaluate_val(self.server_obj.model)
        self.server_obj.val_loss_history.append(metrics["val_loss"])
        self.server_obj.val_accuracy_history.append(metrics["val_accuracy"])
        self.server_obj.f1_score_history.append(metrics["f1_score"])
        self.server_obj.roc_auc_history.append(metrics["roc_auc"])

        print(f"[SERVER] Round {server_round} - "
              f"Val Loss: {metrics['val_loss']:.4f}, "
              f"Val Acc: {metrics['val_accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}"
              )

        aggregated = super().aggregate_fit(server_round, results, failures)
        round_time = time.time() - start_time
        self.server_obj.round_times.append(round_time)
        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

def main():
    server = FLServerTabTransformer()

    def get_eval_fn(server: FLServerTabTransformer):
        def evaluate(server_round, parameters, config):
            params_dict = zip(server.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            server.model.load_state_dict(state_dict, strict=True)
            metrics = server.evaluate_val(server.model)
            return metrics["val_loss"], metrics
        return evaluate

    strategy = FedMADE_GSA_Strategy(
        server_obj=server,
        clients=server.clients,
        evaluate_fn=get_eval_fn(server),
        eps=0.3, min_samples=3, lr=0.001, max_iter=100, gsa_thresh=0.7
    )

    print("[SERVER] Starting Flower server at 127.0.0.1:8080")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    print("\n[SERVER] ===== FINAL TEST EVALUATION =====")
    test_metrics = server.evaluate_test(server.model)
    print(f"Test Loss={test_metrics['test_loss']:.4f}, "
          f"Test Acc={test_metrics['test_accuracy']:.4f}, "
          f"F1={test_metrics['f1_score']:.4f}"
          )
    torch.save(server.model.state_dict(), os.path.join(server.MODEL_DIR, "aggregated_model.pt"))
    print(f"[SERVER] Final model saved to {os.path.join(server.MODEL_DIR, 'aggregated_model.pt')}")

    server._plot_training_progress()
    server._plot_round_time()
    
    test_loader = DataLoader(TensorDataset(server.X_test_cat, server.X_test_cont, server.y_test),
                             batch_size=256, shuffle=False)
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for cat_data, cont_data, labels in test_loader:
            out = server.model(cat_data, cont_data)
            pred = torch.argmax(out, dim=1)
            score = torch.softmax(out, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())
            y_scores.extend(score.numpy())

    server._plot_confusion_matrix(y_true, y_pred)
    server._plot_roc_curve(np.array(y_true), np.array(y_scores))

if __name__ == "__main__":
    main()
