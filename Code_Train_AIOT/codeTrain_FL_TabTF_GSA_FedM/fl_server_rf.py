import io
import os
import torch
import joblib
import numpy as np
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from data_processing import load_and_process_data
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_DIR = "fl_outputs"
DATA_PATH = "dataset/yourDataset_Training.csv"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
IMAGES_SERVER_DIR = os.path.join(OUTPUT_DIR, "images", "server")
LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "log_server.txt")
AGGREGATED_MODEL_PATH = os.path.join(MODEL_DIR, "aggregated_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_server.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_server.pkl")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.pkl")
NUM_CONTINUOUS_PATH = os.path.join(MODEL_DIR, "num_continuous.pkl")
SIMILARITY_THRESHOLD = 0.7
LAMBDA_PERF = 0.4
LAMBDA_CONV = 0.6
MAX_ROUNDS = 3
CONVERGENCE_THRESHOLD = 0.001
BATCH_SIZE = 256

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='a'
)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

class FLServerTabTransformer:
    def __init__(self):
        print("[SERVER] Khởi tạo server...")
        logging.info("[SERVER] Khởi tạo server...")
        ensure_dir(MODEL_DIR)
        ensure_dir(IMAGES_SERVER_DIR)
        
        X, y, categorical_cols, num_classes, le, _ = load_and_process_data(DATA_PATH)

        # Step 1: Split into train (70%) and temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Step 2: Split temp (30%) into validation (20%) and test (10%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42
        )

        cont_cols = [c for c in X.columns if c not in categorical_cols]
        self.scaler = RobustScaler()

        if cont_cols:
            X_train_cont = self.scaler.fit_transform(X_train[cont_cols])
            X_val_cont = self.scaler.transform(X_val[cont_cols])
            X_test_cont = self.scaler.transform(X_test[cont_cols])
        else:
            X_train_cont = X_train.values
            X_val_cont = X_val.values
            X_test_cont = X_test.values

        X_train_cat = X_train[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_train), 0))
        X_val_cat = X_val[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_val), 0))
        X_test_cat = X_test[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_test), 0))

        print("[SERVER] Lưu label encoder, categories, và num_continuous...")
        logging.info("[SERVER] Lưu label encoder, categories, và num_continuous...")
        try:
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(le, ENCODER_PATH)
            joblib.dump([X[c].nunique() for c in categorical_cols], CATEGORIES_PATH)
            joblib.dump(len(cont_cols), NUM_CONTINUOUS_PATH)
            print(f"[SERVER] Đã lưu thành công các file cần thiết")
            logging.info(f"[SERVER] Đã lưu thành công các file cần thiết")
        except Exception as e:
            print(f"[SERVER] Lỗi khi lưu: {e}")
            logging.error(f"[SERVER] Lỗi khi lưu: {e}")
            raise

        # Convert to tensors
        self.X_train_cont = torch.tensor(X_train_cont, dtype=torch.float32)
        self.X_val_cont = torch.tensor(X_val_cont, dtype=torch.float32)
        self.X_test_cont = torch.tensor(X_test_cont, dtype=torch.float32)
        self.X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        self.X_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
        self.X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(self.X_train_cat, self.X_train_cont, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        val_dataset = TensorDataset(self.X_val_cat, self.X_val_cont, self.y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = TensorDataset(self.X_test_cat, self.X_test_cont, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_classes = num_classes
        self.categorical_cols = categorical_cols

        print(f"[SERVER] Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print(f"[SERVER] Data shapes: train={self.X_train_cont.shape}, val={self.X_val_cont.shape}, test={self.X_test_cont.shape}")
        logging.info(f"[SERVER] Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        logging.info(f"[SERVER] Data shapes: train={self.X_train_cont.shape}, val={self.X_val_cont.shape}, test={self.X_test_cont.shape}")

        self.accuracy_history = []
        self.f1_score_history = []
        self.precision_history = []
        self.recall_history = []
        self.roc_auc_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.round_times = []

        self.hyperparams = {
            "categories": [X[c].nunique() for c in categorical_cols],
            "num_continuous": len(cont_cols),
            "dim": 128,
            "dim_out": num_classes,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.3,
            "ff_dropout": 0.3,
            "mlp_hidden_mults": (4, 2),
            "mlp_act": torch.nn.ReLU()
        }
        self.model = TabTransformer(**self.hyperparams)
        self.previous_metrics = None
        self.convergence_round = None
        print(f"[SERVER] TabTransformer sẵn sàng - {self.hyperparams}")
        logging.info(f"[SERVER] TabTransformer sẵn sàng - {self.hyperparams}")

    def load_model(self, round_number: int):
        model_path = os.path.join(MODEL_DIR, f"aggregated_model_round_{round_number}.pt")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"[SERVER] Đã load model từ {model_path}")
                logging.info(f"[SERVER] Đã load model từ {model_path}")
            except Exception as e:
                print(f"[SERVER] Lỗi load model: {e}")
                logging.error(f"[SERVER] Lỗi load model: {e}")

    # def _plot_confusion_matrix(self, y_true, y_pred):
    #     try:
    #         labels = list(range(self.num_classes))
    #         cm = confusion_matrix(y_true, y_pred, labels=labels)
    #         cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         cm_percent = np.nan_to_num(cm_percent)

    #         fig, ax = plt.subplots(figsize=(max(8, self.num_classes), max(6, self.num_classes * 0.5)))
    #         sns.heatmap(cm_percent * 100, annot=True, fmt=".2f", cmap='Blues', cbar=True,
    #                     xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='gray')

    #         ax.set_title(f'Confusion Matrix (%) - Server (Round {len(self.val_accuracy_history)})', fontsize=14, fontweight='bold')
    #         ax.set_xlabel('Predicted Label', fontsize=12)
    #         ax.set_ylabel('True Label', fontsize=12)
    #         plt.xticks(rotation=45)
    #         plt.yticks(rotation=0)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_confusion_matrix_round_{len(self.val_accuracy_history)}.png"),
    #                     dpi=150, bbox_inches='tight')
    #         plt.close()
    #         print("[SERVER] Đã lưu confusion matrix (%).")
    #         logging.info("[SERVER] Đã lưu confusion matrix (%).")
    #     except Exception as e:
    #         print(f"[SERVER] Lỗi vẽ confusion matrix: {e}")
    #         logging.error(f"[SERVER] Lỗi vẽ confusion matrix: {e}")
                
    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
            # Calculate frequency of each label
            label_counts = np.bincount(y_true, minlength=self.num_classes)
            # Get indices of top 5 most frequent labels
            top_5_labels = np.argsort(label_counts)[::-1][:5]
            
            # Filter y_true and y_pred to include only top 5 labels
            mask = np.isin(y_true, top_5_labels)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            
            # Compute confusion matrix for filtered data
            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_5_labels)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_percent = np.nan_to_num(cm_percent)

            fig, ax = plt.subplots(figsize=(max(8, len(top_5_labels)), max(6, len(top_5_labels) * 0.5)))
            sns.heatmap(cm_percent * 100, annot=True, fmt=".2f", cmap='Blues', cbar=True,
                        xticklabels=top_5_labels, yticklabels=top_5_labels, linewidths=0.5, linecolor='gray')

            ax.set_title(f'Confusion Matrix (%) - Server {self.client_id} (Round {self.round_count})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = os.path.join(self.IMAGES_SERVER_DIR, f"server_confusion_matrix_round_{len(self.val_accuracy_history)}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print("[SERVER] Đã lưu confusion matrix (%).")
            logging.info("[SERVER] Đã lưu confusion matrix (%).")
            
        except Exception as e:
            print(f"[SERVER] Lỗi vẽ confusion matrix: {e}")
            logging.error(f"[SERVER] Lỗi vẽ confusion matrix: {e}")

    def _plot_roc_curve(self, y_true, y_scores):
        try:
            if self.num_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1], pos_label=1)
                roc_auc = roc_auc_score(y_true, y_scores[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Server (Round {len(self.val_accuracy_history)})')
                plt.legend(loc="lower right")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_roc_curve_round_{len(self.val_accuracy_history)}.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[SERVER] Đã lưu ROC curve với AUC: {roc_auc:.4f}")
                return roc_auc

            y_bin = label_binarize(y_true, classes=range(self.num_classes))
            if y_bin.shape[1] == 1:
                return 0.5

            roc_auc = roc_auc_score(y_bin, y_scores, average='macro', multi_class='ovr')

            fpr = {}
            tpr = {}
            roc_auc_dict = {}

            for i in range(self.num_classes):
                if np.sum(y_bin[:, i]) > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
                    roc_auc_dict[i] = roc_auc_score(y_bin[:, i], y_scores[:, i])

            plt.figure(figsize=(12, 8))
            colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
            for i, color in zip(range(self.num_classes), colors):
                if i in roc_auc_dict:
                    plt.plot(fpr[i], tpr[i], color=color, lw=1,
                             label=f"Lớp {i} (AUC = {roc_auc_dict[i]:.3f})")

            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Per-Class ROC - Server (Round {len(self.val_accuracy_history)}) - Macro AUC: {roc_auc:.3f}')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize='small', borderaxespad=0.)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_roc_curve_round_{len(self.val_accuracy_history)}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[SERVER] Đã lưu ROC curve với macro AUC: {roc_auc:.4f}")
            return roc_auc 
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ ROC curve: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ ROC curve: {e}")
            return 0.5

    def _plot_label_distribution(self, y_true, y_pred):
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                print("[SERVER] Không có dữ liệu để vẽ phân phối nhãn.")
                logging.warning("[SERVER] Không có dữ liệu để vẽ phân phối nhãn.")
                return
            fig, axs = plt.subplots(1, 2, figsize=(16, 6)) 
            sns.countplot(x=y_true, ax=axs[0])
            axs[0].set_title("Phân phối nhãn thực")
            axs[0].tick_params(axis='x', rotation=45) 

            sns.countplot(x=y_pred, ax=axs[1])
            axs[1].set_title("Phân phối nhãn dự đoán")
            axs[1].tick_params(axis='x', rotation=45)  

            plt.suptitle(f"Phân phối nhãn - Server (Round {len(self.val_accuracy_history)})")
            plt.tight_layout() 
            plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_label_distribution_round_{len(self.val_accuracy_history)}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print("[SERVER] Đã lưu phân phối nhãn.")
            logging.info("[SERVER] Đã lưu phân phối nhãn.")
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ phân phối nhãn: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ phân phối nhãn: {e}")

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

            if self.val_loss_history:
                l2, = ax1.plot(rounds, self.val_loss_history, label="Val Loss", color="orange", marker='*')
            else:
                l2 = None

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
            plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_training_progress_round_{len(self.val_accuracy_history)}.png"),
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
            for i, time in enumerate(self.round_times):
                plt.annotate(f'{time:.1f}s', (i+1, time), textcoords="offset points", xytext=(0,10), ha='center')
            plt.savefig(os.path.join(IMAGES_SERVER_DIR, f"server_round_time_round_{len(self.round_times)}.png"),
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

            save_path = os.path.join(IMAGES_SERVER_DIR, f"server_prf_curve_round_{len(self.val_accuracy_history)}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[SERVER] Đã lưu Precision-Recall-F1 curve: {save_path}")
            logging.info(f"[SERVER] Đã lưu Precision-Recall-F1 curve")
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ PRF curve: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ PRF curve: {e}")

    def evaluate_val(self, model):
        print(f"[SERVER] Bắt đầu đánh giá trên validation set...")
        logging.info(f"[SERVER] Bắt đầu đánh giá trên validation set...")
        
        try:
            model.eval()
            y_pred_list = []
            y_true_list = []
            y_scores_list = []
            total_loss = 0.0
            num_batches = 0

            loss_fn = torch.nn.CrossEntropyLoss()

            with torch.no_grad():
                for cat_data, cont_data, labels in self.val_loader:
                    outputs = model(cat_data, cont_data)
                    loss = loss_fn(outputs, labels)

                    y_pred = torch.argmax(outputs, dim=1)
                    y_scores = torch.softmax(outputs, dim=1)
                    
                    y_pred_list.append(y_pred.numpy())
                    y_true_list.append(labels.numpy())
                    y_scores_list.append(y_scores.numpy())

                    total_loss += loss.item()
                    num_batches += 1

            if num_batches == 0:
                print(f"[SERVER] Không có batch nào trong val_loader")
                logging.warning(f"[SERVER] Val loader rỗng")
                return {
                    "val_loss": 0.0,
                    "val_accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "roc_auc": 0.0
                }

            y_pred = np.concatenate(y_pred_list)
            y_true = np.concatenate(y_true_list)
            y_scores = np.concatenate(y_scores_list)
            avg_loss = total_loss / num_batches

            num_examples = len(y_true)

            if num_examples > 0:
                self._plot_confusion_matrix(y_true, y_pred)
                self._plot_label_distribution(y_true, y_pred)
                roc_auc = self._plot_roc_curve(y_true, y_scores)
                self._plot_prf_curve(y_true, y_scores)
            else:
                roc_auc = 0.0

            val_accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"[SERVER] Validation - Loss: {avg_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
            logging.info(f"[SERVER] Validation - Loss: {avg_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

            metrics = {
                "val_loss": float(avg_loss),
                "val_accuracy": float(val_accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "roc_auc": float(roc_auc)
            }
            return metrics
        except Exception as e:
            print(f"[SERVER] Lỗi khi đánh giá trên validation set: {e}")
            logging.error(f"[SERVER] Lỗi khi đánh giá trên validation set: {e}")
            return {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": 0.0
            }

    def evaluate(self, model):
        print(f"[SERVER] Bắt đầu đánh giá trên test set...")
        logging.info(f"[SERVER] Bắt đầu đánh giá trên test set...")
        
        try:
            model.eval()
            y_pred_list = []
            y_true_list = []
            y_scores_list = []
            total_loss = 0.0
            num_batches = 0

            loss_fn = torch.nn.CrossEntropyLoss()

            with torch.no_grad():
                for cat_data, cont_data, labels in self.test_loader:
                    outputs = model(cat_data, cont_data)
                    loss = loss_fn(outputs, labels)

                    y_pred = torch.argmax(outputs, dim=1)
                    y_scores = torch.softmax(outputs, dim=1)
                    
                    y_pred_list.append(y_pred.numpy())
                    y_true_list.append(labels.numpy())
                    y_scores_list.append(y_scores.numpy())

                    total_loss += loss.item()
                    num_batches += 1

            if num_batches == 0:
                print(f"[SERVER] Không có batch nào trong test_loader")
                logging.warning(f"[SERVER] Test loader rỗng")
                return {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "roc_auc": 0.0
                }

            y_pred = np.concatenate(y_pred_list)
            y_true = np.concatenate(y_true_list)
            y_scores = np.concatenate(y_scores_list)
            avg_loss = total_loss / num_batches

            num_examples = len(y_true)

            if num_examples > 0:
                self._plot_confusion_matrix(y_true, y_pred)
                self._plot_label_distribution(y_true, y_pred)
                roc_auc = self._plot_roc_curve(y_true, y_scores)
                self._plot_prf_curve(y_true, y_scores)
            else:
                roc_auc = 0.0

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"[SERVER] Test - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
            logging.info(f"[SERVER] Test - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

            metrics = {
                "loss": float(avg_loss),
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "roc_auc": float(roc_auc)
            }
            return metrics
        except Exception as e:
            print(f"[SERVER] Lỗi khi đánh giá trên test set: {e}")
            logging.error(f"[SERVER] Lỗi khi đánh giá trên test set: {e}")
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": 0.0
            }

    def compute_gradient_similarity(self, grad1, grad2):
        try:
            flat_grad1 = torch.cat([g.flatten() for g in grad1 if g is not None])
            flat_grad2 = torch.cat([g.flatten() for g in grad2 if g is not None])
            if len(flat_grad1) == 0 or len(flat_grad2) == 0:
                return 0.0
            cos_sim = torch.nn.functional.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0)).item()
            return cos_sim
        except Exception as e:
            print(f"[SERVER] Lỗi tính gradient similarity: {e}")
            logging.error(f"[SERVER] Lỗi tính gradient similarity: {e}")
            return 0.0

class FedMADEStrategy(FedAvg):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > MAX_ROUNDS:
            print(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng, dừng fit")
            logging.info(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng, dừng fit")
            return []
            
        if server_round > 1:
            self.server.load_model(server_round - 1)
            
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > MAX_ROUNDS:
            print(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng, dừng evaluate")
            logging.info(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng, dừng evaluate")
            return []
            
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List) -> Tuple[Optional[Parameters], Dict]:
        print(f"\n[SERVER] === ROUND {server_round} với {len(results)} clients ===")
        logging.info(f"[SERVER] Vòng {server_round} với {len(results)} clients")
        start_time = time.time()

        if not results:
            print("[SERVER] Không nhận được kết quả từ client!")
            logging.error("[SERVER] Không nhận được kết quả từ client!")
            return None, {"error": "Không nhận được kết quả từ client"}

        filtered_results = []
        for client, fit_res in results:
            metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else fit_res.get('metrics', {})
            stop_after_this_round = metrics.get("stop_after_this_round", False) if isinstance(metrics, dict) else False
            if not stop_after_this_round:
                filtered_results.append((client, fit_res))
            else:
                client_id = getattr(client, 'cid', str(client))
                print(f"[SERVER] Client {client_id} yêu cầu dừng sau round này")
                logging.info(f"[SERVER] Client {client_id} yêu cầu dừng sau round này")

        if not filtered_results:
            print(f"[SERVER] Tất cả client đã dừng, kết thúc huấn luyện ở vòng {server_round}.")
            logging.info(f"[SERVER] Tất cả client đã dừng, kết thúc huấn luyện ở vòng {server_round}.")
            return None, {"error": "Tất cả client đã dừng"}

        models, gradients, accuracies, client_ids = [], [], [], []

        for client, fit_res in filtered_results:
            try:
                client_id = getattr(client, 'cid', str(client))
                print(f"[SERVER] Xử lý kết quả từ client {client_id}")
                logging.info(f"[SERVER] Xử lý kết quả từ client {client_id}")
                
                tensors = parameters_to_ndarrays(fit_res.parameters)
                if len(tensors) < 2:
                    print(f"[SERVER] Client {client_id} thiếu tensor (có {len(tensors)}, cần 2)")
                    logging.warning(f"[SERVER] Client {client_id} thiếu tensor")
                    continue

                try:
                    state_dict_buf = io.BytesIO(tensors[0].tobytes())
                    grad_buf = io.BytesIO(tensors[1].tobytes())
                    state_dict = torch.load(state_dict_buf, weights_only=True, map_location='cpu')
                    grad_list = torch.load(grad_buf, weights_only=True, map_location='cpu')
                except Exception as e:
                    print(f"[SERVER] Lỗi load parameters từ client {client_id}: {e}")
                    logging.error(f"[SERVER] Lỗi load parameters từ client {client_id}: {e}")
                    continue

                if grad_list:
                    grad_norm_val = sum(torch.norm(g).item() for g in grad_list if g is not None)
                    if grad_norm_val > 100 or grad_norm_val < 1e-8:
                        print(f"[SERVER] Client {client_id} có gradient bất thường: {grad_norm_val:.6f}")
                        logging.warning(f"[SERVER] Client {client_id} gradient bất thường: {grad_norm_val:.6f}")
                        continue
                else:
                    print(f"[SERVER] Client {client_id} không có gradient")
                    logging.warning(f"[SERVER] Client {client_id} không có gradient")
                    continue

                try:
                    model = TabTransformer(**self.server.hyperparams)
                    model.load_state_dict(state_dict, strict=True)
                    models.append(model)
                    gradients.append(grad_list)
                    client_ids.append(client_id)
                    
                    metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else {}
                    accuracy = float(metrics.get("val_accuracy", 0.0))
                    accuracies.append(accuracy)
                    
                    print(f"[SERVER] Client {client_id} - val_accuracy: {accuracy:.4f}, grad_norm: {grad_norm_val:.4f}")
                    logging.info(f"[SERVER] Client {client_id} - val_accuracy: {accuracy:.4f}, grad_norm: {grad_norm_val:.4f}")
                    
                except Exception as e:
                    print(f"[SERVER] Lỗi tạo model cho client {client_id}: {e}")
                    logging.error(f"[SERVER] Lỗi tạo model cho client {client_id}: {e}")
                    continue

            except Exception as e:
                print(f"[SERVER] Lỗi tổng quát khi xử lý client {client_id}: {e}")
                logging.error(f"[SERVER] Lỗi tổng quát khi xử lý client {client_id}: {e}")
                continue

        if not models:
            print("[SERVER] Không có mô hình hợp lệ để tổng hợp!")
            logging.error("[SERVER] Không có mô hình hợp lệ để tổng hợp!")
            return None, {"error": "Không có mô hình hợp lệ"}

        print(f"[SERVER] Đã thu thập {len(models)} mô hình hợp lệ từ clients")
        logging.info(f"[SERVER] Đã thu thập {len(models)} mô hình hợp lệ từ clients")

        if len(accuracies) > 0:
            acc_weights = np.array([np.exp(acc) for acc in accuracies])
            acc_weights = acc_weights / np.sum(acc_weights)
            
            grad_norms = [sum(torch.norm(g).item() for g in grad if g is not None) for grad in gradients]
            conv_weights = np.array([np.exp(-norm) for norm in grad_norms])
            conv_weights = conv_weights / np.sum(conv_weights)
            
            weights = LAMBDA_PERF * acc_weights + LAMBDA_CONV * conv_weights
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(models)) / len(models)
            
        print(f"[SERVER] Trọng số tổng hợp: {[f'{w:.4f}' for w in weights]}")
        logging.info(f"[SERVER] Trọng số tổng hợp: {weights.tolist()}")

        inliers = []
        if len(gradients) > 1:
            for i, grad in enumerate(gradients):
                similarities = []
                for j, grad_j in enumerate(gradients):
                    if i != j:
                        sim = self.server.compute_gradient_similarity(grad, grad_j)
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities) if similarities else 0.0
                is_inlier = avg_sim >= SIMILARITY_THRESHOLD or len(similarities) == 0
                
                if is_inlier:
                    inliers.append(i)
                    
                print(f"[SERVER] Client {client_ids[i]}: avg_similarity={avg_sim:.4f}, inlier={is_inlier}")
                logging.info(f"[SERVER] Client {client_ids[i]}: avg_similarity={avg_sim:.4f}, inlier={is_inlier}")
        else:
            inliers = list(range(len(models)))
            
        if not inliers:
            print("[SERVER] Không có client inlier, sử dụng tất cả")
            logging.warning("[SERVER] Không có client inlier, sử dụng tất cả")
            inliers = list(range(len(models)))

        print(f"[SERVER] Sử dụng {len(inliers)}/{len(models)} clients để tổng hợp")
        logging.info(f"[SERVER] Sử dụng {len(inliers)}/{len(models)} clients để tổng hợp")

        try:
            aggregated_state_dict = {}
            for key in models[0].state_dict().keys():
                aggregated_state_dict[key] = torch.zeros_like(models[0].state_dict()[key])
                
                total_weight = sum(weights[i] for i in inliers)
                if total_weight > 0:
                    for i in inliers:
                        weight_normalized = weights[i] / total_weight
                        aggregated_state_dict[key] += weight_normalized * models[i].state_dict()[key]
                else:
                    for i in inliers:
                        aggregated_state_dict[key] += models[i].state_dict()[key] / len(inliers)

            self.server.model.load_state_dict(aggregated_state_dict)
            print("[SERVER] Đã tổng hợp parameters thành công")
            logging.info("[SERVER] Đã tổng hợp parameters thành công")
            
        except Exception as e:
            print(f"[SERVER] Lỗi khi tổng hợp parameters: {e}")
            logging.error(f"[SERVER] Lỗi khi tổng hợp parameters: {e}")
            return None, {"error": f"Lỗi tổng hợp: {str(e)}"}

        try:
            server_metrics = self.server.evaluate_val(self.server.model)
            val_loss, val_accuracy, f1, roc_auc = (
                server_metrics["val_loss"],
                server_metrics["val_accuracy"],
                server_metrics["f1_score"],
                server_metrics["roc_auc"]
            )

            client_val_loss = 0.0
            client_val_accuracy = 0.0
            total_examples = 0
            for _, fit_res in filtered_results:
                metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else {}
                num_examples = fit_res.num_examples if hasattr(fit_res, 'num_examples') else 0
                if num_examples > 0:
                    client_val_loss += metrics.get("val_loss", 0.0) * num_examples
                    client_val_accuracy += metrics.get("val_accuracy", 0.0) * num_examples
                    total_examples += num_examples
            
            if total_examples > 0:
                client_val_loss /= total_examples
                client_val_accuracy /= total_examples

            self.server.val_loss_history.append(val_loss)
            self.server.val_accuracy_history.append(val_accuracy)
            self.server.f1_score_history.append(f1)
            self.server.roc_auc_history.append(roc_auc)
            
            print(f"[SERVER] Server validation - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            print(f"[SERVER] Aggregated client metrics - Val Loss: {client_val_loss:.4f}, Val Accuracy: {client_val_accuracy:.4f}")
            logging.info(f"[SERVER] Server validation - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            logging.info(f"[SERVER] Aggregated client metrics - Val Loss: {client_val_loss:.4f}, Val Accuracy: {client_val_accuracy:.4f}")
            
        except Exception as e:
            print(f"[SERVER] Lỗi khi đánh giá server model trên validation set: {e}")
            logging.error(f"[SERVER] Lỗi khi đánh giá server model trên validation set: {e}")
            server_metrics = {
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "roc_auc": 0.0
            }
            val_loss, val_accuracy, f1, roc_auc = 0.0, 0.0, 0.0, 0.0
            self.server.val_loss_history.append(0.0)
            self.server.val_accuracy_history.append(0.0)

        if self.server.previous_metrics is not None:
            acc_diff = abs(val_accuracy - self.server.previous_metrics["val_accuracy"])
            f1_diff = abs(f1 - self.server.previous_metrics["f1_score"])
            roc_auc_diff = abs(roc_auc - self.server.previous_metrics["roc_auc"])
            
            if (acc_diff < CONVERGENCE_THRESHOLD and 
                f1_diff < CONVERGENCE_THRESHOLD and 
                roc_auc_diff < CONVERGENCE_THRESHOLD and 
                self.server.convergence_round is None):
                self.server.convergence_round = server_round
                print(f"[SERVER] Mô hình hội tụ tại vòng {server_round}")
                logging.info(f"[SERVER] Mô hình hội tụ tại vòng {server_round}")
        
        self.server.previous_metrics = {
            "val_accuracy": val_accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        round_time = time.time() - start_time
        self.server.round_times.append(round_time)
        print(f"[SERVER] Hoàn thành vòng {server_round} trong {round_time:.2f} giây")
        logging.info(f"[SERVER] Hoàn thành vòng {server_round} trong {round_time:.2f} giây")
        
        try:
            self.server._plot_training_progress()
            self.server._plot_round_time()
        except Exception as e:
            print(f"[SERVER] Lỗi khi vẽ biểu đồ: {e}")
            logging.error(f"[SERVER] Lỗi khi vẽ biểu đồ: {e}")

        try:
            round_model_path = os.path.join(MODEL_DIR, f"aggregated_model_round_{server_round}.pt")
            torch.save(self.server.model.state_dict(), round_model_path)
            print(f"[SERVER] Đã lưu model vào {round_model_path}")
            logging.info(f"[SERVER] Đã lưu model vào {round_model_path}")
        except Exception as e:
            print(f"[SERVER] Lỗi khi lưu model: {e}")
            logging.error(f"[SERVER] Lỗi khi lưu model: {e}")

        try:
            buf = io.BytesIO()
            torch.save(self.server.model.state_dict(), buf)
            buf.seek(0)
            parameters = ndarrays_to_parameters([np.frombuffer(buf.getvalue(), dtype=np.uint8)])
            
            final_metrics = server_metrics.copy()
            final_metrics["round_time"] = round_time
            final_metrics["num_clients"] = len(inliers)
            final_metrics["convergence_round"] = self.server.convergence_round
            final_metrics["client_val_loss"] = client_val_loss
            final_metrics["client_val_accuracy"] = client_val_accuracy
            
            print(f"[SERVER] === KẾT THÚC ROUND {server_round} ===\n")
            return parameters, final_metrics
            
        except Exception as e:
            print(f"[SERVER] Lỗi khi chuẩn bị parameters trả về: {e}")
            logging.error(f"[SERVER] Lỗi khi chuẩn bị parameters trả về: {e}")
            return None, {"error": f"Lỗi chuẩn bị parameters: {str(e)}"}

def main():
    server = FLServerTabTransformer()
    strategy = FedMADEStrategy(server)
    
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
            strategy=strategy
        )
    except Exception as e:
        print(f"[SERVER] Lỗi khi khởi động server: {e}")
        logging.error(f"[SERVER] Lỗi khi khởi động server: {e}")
        raise

if __name__ == "__main__":
    main()