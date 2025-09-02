import os
import io
import joblib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer
from data_processing import load_and_process_data
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from itertools import cycle

OUTPUT_DIR = "fl_outputs"
DATA_PATH = "dataset/yourDataset_Training.csv"
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001
SPARSITY = 0.3
SIMILARITY_THRESHOLD = 0.7
LAMBDA_PERF = 0.6
LAMBDA_CONV = 0.4
MAX_ROUNDS = 3

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

class FLClientTabTransformer(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.round_count = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.grad_norm_history = []

        self.MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
        self.IMAGES_CLIENT_DIR = os.path.join(OUTPUT_DIR, "images", f"client_{client_id}")
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, f"model_client_{client_id}.pt")
        self.SCALER_PATH = os.path.join(self.MODEL_DIR, f"scaler_client_{client_id}.pkl")
        self.ENCODER_PATH = os.path.join(self.MODEL_DIR, f"label_encoder_client_{client_id}.pkl")
        self.LOG_PATH = os.path.join(OUTPUT_DIR, "logs", f"client_log_{client_id}.txt")
        
        ensure_dir(os.path.dirname(self.LOG_PATH))
        logging.basicConfig(
            filename=self.LOG_PATH,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filemode='a'
        )

        print(f"[CLIENT {client_id}] ===== KHỞI TẠO CLIENT =====")
        logging.info(f"[CLIENT {client_id}] ===== KHỞI TẠO CLIENT =====")
        
        ensure_dir(self.MODEL_DIR)
        ensure_dir(self.IMAGES_CLIENT_DIR)

        try:
            print(f"[CLIENT {client_id}] Đang load dữ liệu...")
            logging.info(f"[CLIENT {client_id}] Đang load dữ liệu...")
            
            self.X_train_raw, self.y_train, self.categorical_cols, self.num_classes, self.le, self.cluster_labels = load_and_process_data(DATA_PATH)
            
            print(f"[CLIENT {client_id}] Dữ liệu: {len(self.X_train_raw)} samples, {len(self.categorical_cols)} categorical cols, {self.num_classes} classes")
            logging.info(f"[CLIENT {client_id}] Dữ liệu: {len(self.X_train_raw)} samples, {len(self.categorical_cols)} categorical cols, {self.num_classes} classes")
            
        except Exception as e:
            print(f"[CLIENT {client_id}] Lỗi load dữ liệu: {str(e)}")
            logging.error(f"[CLIENT {client_id}] Lỗi load dữ liệu: {str(e)}")
            raise

        # Step 1: Split into train (70%) and temp (30%)
        X_train_raw, X_temp, y_train, y_temp = train_test_split(
            self.X_train_raw, self.y_train, test_size=0.3, stratify=self.y_train, random_state=42
        )

        # Step 2: Split temp (30%) into validation (20%) and test (10%)
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42
        )

        self.X_train_raw = X_train_raw
        self.X_val_raw = X_val_raw
        self.X_test_raw = X_test_raw
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"[CLIENT {client_id}] Chia dữ liệu: {len(self.X_train_raw)} train / {len(self.X_val_raw)} val / {len(self.X_test_raw)} test")
        logging.info(f"[CLIENT {client_id}] Chia dữ liệu: {len(self.X_train_raw)} train / {len(self.X_val_raw)} val / {len(self.X_test_raw)} test")

        # Process continuous features
        cont_cols = [c for c in self.X_train_raw.columns if c not in self.categorical_cols]
        self.scaler = RobustScaler()

        if cont_cols:
            self.X_train = self.scaler.fit_transform(self.X_train_raw[cont_cols])
            self.X_val = self.scaler.transform(self.X_val_raw[cont_cols])
            self.X_test = self.scaler.transform(self.X_test_raw[cont_cols])
        else:
            self.X_train = self.X_train_raw.values
            self.X_val = self.X_val_raw.values
            self.X_test = self.X_test_raw.values

        if self.categorical_cols:
            X_train_cat = self.X_train_raw[self.categorical_cols].values.astype(int)
            X_val_cat = self.X_val_raw[self.categorical_cols].values.astype(int)
            X_test_cat = self.X_test_raw[self.categorical_cols].values.astype(int)
        else:
            X_train_cat = np.zeros((len(self.X_train_raw), 0))
            X_val_cat = np.zeros((len(self.X_val_raw), 0))
            X_test_cat = np.zeros((len(self.X_test_raw), 0))

        self.X_train_cont = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_val_cont = torch.tensor(self.X_val, dtype=torch.float32)
        self.X_test_cont = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        self.X_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
        self.X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.y_val = torch.tensor(self.y_val, dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(self.X_train_cat, self.X_train_cont, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = TensorDataset(self.X_val_cat, self.X_val_cont, self.y_val)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dataset = TensorDataset(self.X_test_cat, self.X_test_cont, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_counts = np.bincount(self.y_train.numpy())
        class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"[CLIENT {client_id}] Class distribution: {class_counts}")
        print(f"[CLIENT {client_id}] Class weights: {class_weights.numpy()}")
        logging.info(f"[CLIENT {client_id}] Class distribution: {class_counts}")

        self.hyperparams = {
            "categories": [self.X_train_raw[c].nunique() for c in self.categorical_cols],
            "num_continuous": self.X_train.shape[1],
            "dim": 128,
            "dim_out": self.num_classes,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.3,
            "ff_dropout": 0.3,
            "mlp_hidden_mults": (4, 2),
            "mlp_act": nn.ReLU()
        }
        self.model = TabTransformer(**self.hyperparams)
        
        if self._model_files_exist():
            self._load_model_and_scaler()
        else:
            print(f"[CLIENT {client_id}] Tạo mô hình mới...")
            logging.info(f"[CLIENT {client_id}] Tạo mô hình mới...")
            self._save_state()

        print(f"[CLIENT {client_id}] Model sẵn sàng với {self.num_classes} classes")
        print(f"[CLIENT {client_id}] Hyperparams: {self.hyperparams}")
        logging.info(f"[CLIENT {client_id}] Model sẵn sàng với {self.num_classes} classes")

    def _model_files_exist(self):
        return (os.path.exists(self.MODEL_PATH) and 
                os.path.exists(self.SCALER_PATH) and 
                os.path.exists(self.ENCODER_PATH))

    def _load_model_and_scaler(self):
        print(f"[CLIENT {self.client_id}] Loading model/scaler/encoder...")
        logging.info(f"[CLIENT {self.client_id}] Loading model/scaler/encoder...")
        try:
            state_dict = torch.load(self.MODEL_PATH, weights_only=True, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.scaler = joblib.load(self.SCALER_PATH)
            self.le = joblib.load(self.ENCODER_PATH)
            print(f"[CLIENT {self.client_id}] Đã load thành công")
            logging.info(f"[CLIENT {self.client_id}] Đã load thành công")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi load: {e}. Tạo mới...")
            logging.error(f"[CLIENT {self.client_id}] Lỗi load: {e}")
            self._save_state()

    def _save_state(self):
        try:
            torch.save(self.model.state_dict(), self.MODEL_PATH)
            joblib.dump(self.scaler, self.SCALER_PATH)
            joblib.dump(self.le, self.ENCODER_PATH)
            print(f"[CLIENT {self.client_id}] Đã lưu model/scaler/encoder")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu model/scaler/encoder")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi lưu: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi lưu: {e}")

    def _sparsify_gradients(self, gradients, sparsity=SPARSITY):
        if not gradients:
            return gradients
        try:
            flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
            if len(flat_grads) == 0:
                return gradients
                
            k = int((1 - sparsity) * len(flat_grads))
            if k == 0:
                k = 1
            threshold = torch.kthvalue(torch.abs(flat_grads), len(flat_grads) - k).values
            
            sparsified_count = 0
            total_count = 0
            for g in gradients:
                if g is not None:
                    mask = torch.abs(g) < threshold
                    sparsified_count += mask.sum().item()
                    total_count += g.numel()
                    g.masked_fill_(mask, 0.0)
                    
            actual_sparsity = sparsified_count / total_count if total_count > 0 else 0
            print(f"[CLIENT {self.client_id}] Sparsified gradients: {actual_sparsity:.3f} sparsity, threshold: {threshold:.6f}")
            logging.info(f"[CLIENT {self.client_id}] Sparsified gradients: {actual_sparsity:.3f} sparsity")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi sparsify gradients: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi sparsify gradients: {e}")
        return gradients

    def _compute_validation_metrics(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for cat_data, cont_data, labels in self.val_loader:
                out = self.model(cat_data, cont_data)
                loss = self.loss_fn(out, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
                
        val_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        if len(all_preds) > 0 and len(all_labels) > 0:
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        else:
            f1 = precision = recall = 0.0
            
        # print(f"[CLIENT {self.client_id}] Validation - Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
        # logging.info(f"[CLIENT {self.client_id}] Validation metrics - Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return val_loss, accuracy, f1, precision, recall

    def _compute_gradient_norm(self, gradients):
        if not gradients:
            return 0.0
        try:
            flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
            if len(flat_grads) == 0:
                return 0.0
            norm = torch.norm(flat_grads, p=2).item()
            print(f"[CLIENT {self.client_id}] Gradient norm: {norm:.6f}")
            logging.info(f"[CLIENT {self.client_id}] Gradient norm: {norm:.6f}")
            return norm
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi tính gradient norm: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi tính gradient norm: {e}")
            return 0.0

    def _compute_fedmade_weights(self, accuracy, grad_norm):
        w_perf = np.exp(accuracy)
        w_conv = np.exp(-grad_norm) if grad_norm > 0 else 1.0
        w = LAMBDA_PERF * w_perf + LAMBDA_CONV * w_conv
        print(f"[CLIENT {self.client_id}] FedMADE weights - w_perf: {w_perf:.4f}, w_conv: {w_conv:.4f}, final: {w:.4f}")
        logging.info(f"[CLIENT {self.client_id}] FedMADE weights - w_perf: {w_perf:.4f}, w_conv: {w_conv:.4f}, final: {w:.4f}")
        return w

    # def _plot_confusion_matrix(self, y_true, y_pred):
    #     try:
    #         labels = list(range(self.num_classes))
    #         cm = confusion_matrix(y_true, y_pred, labels=labels)
    #         cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         cm_percent = np.nan_to_num(cm_percent)

    #         fig, ax = plt.subplots(figsize=(max(8, self.num_classes), max(6, self.num_classes * 0.5)))
    #         sns.heatmap(cm_percent * 100, annot=True, fmt=".2f", cmap='Blues', cbar=True,
    #                     xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='gray')

    #         ax.set_title(f'Confusion Matrix (%) - Client {self.client_id} ', 
    #                     fontsize=14, fontweight='bold')
    #         ax.set_xlabel('Predicted Label', fontsize=12)
    #         ax.set_ylabel('True Label', fontsize=12)
    #         plt.xticks(rotation=45)
    #         plt.yticks(rotation=0)
    #         plt.tight_layout()
            
    #         save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_confusion_matrix_round_{self.round_count}.png")
    #         plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #         plt.close()
            
    #         print(f"[CLIENT {self.client_id}] Đã lưu confusion matrix")
    #         logging.info(f"[CLIENT {self.client_id}] Đã lưu confusion matrix")
            
    #     except Exception as e:
    #         print(f"[CLIENT {self.client_id}] Lỗi vẽ confusion matrix: {e}")
    #         logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ confusion matrix: {e}")

    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
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

            ax.set_title(f'Confusion Matrix (%) - Client {self.client_id} ', 
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=18)
            ax.set_ylabel('True Label', fontsize=18)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_confusion_matrix_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CLIENT {self.client_id}] Đã lưu confusion matrix")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu confusion matrix")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi vẽ confusion matrix: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ confusion matrix: {str(e)}")

    def _plot_label_distribution(self, y_true, y_pred):
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                print(f"[CLIENT {self.client_id}] Không có dữ liệu để vẽ phân phối nhãn")
                return
                
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            axs[0].bar(unique_true, counts_true, alpha=0.7, color='skyblue')
            axs[0].set_title("Ground Truth Distribution")
            axs[0].set_xlabel("Label")
            axs[0].set_ylabel("Count")
            axs[0].tick_params(axis='x', rotation=45)
            
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            axs[1].bar(unique_pred, counts_pred, alpha=0.7, color='lightcoral')
            axs[1].set_title("Prediction Distribution")
            axs[1].set_xlabel("Label")
            axs[1].set_ylabel("Count")
            axs[1].tick_params(axis='x', rotation=45)
            
            plt.suptitle(f"Label Distribution - Client {self.client_id} ")
            plt.tight_layout()
            
            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_label_distribution_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[CLIENT {self.client_id}] Đã lưu phân phối nhãn")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu phân phối nhãn")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi vẽ phân phối nhãn: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ phân phối nhãn: {e}")

    def _plot_roc_curve(self, y_true, y_scores):
        try:
            label_counts = np.bincount(y_true, minlength=self.num_classes)
            top_5_indices = np.argsort(label_counts)[::-1][:5]
            
            mask = np.isin(y_true, top_5_indices)
            y_true_filtered = y_true[mask]
            y_scores_filtered = y_scores[mask][:, top_5_indices] if y_scores.ndim == 2 else y_scores[mask]
            
            if len(y_true_filtered) == 0 or len(y_scores_filtered) == 0:
                raise ValueError("No data available for top 5 labels")

            y_bin = label_binarize(y_true_filtered, classes=top_5_indices)
            if y_bin.shape[1] == 1:
                return 0.5

            top_5_labels = self.le.inverse_transform(top_5_indices)

            roc_auc = roc_auc_score(y_bin, y_scores_filtered, average='macro', multi_class='ovr')
            fpr = dict()
            tpr = dict()
            roc_auc_dict = dict()

            for i, label_idx in enumerate(top_5_indices):
                if np.sum(y_bin[:, i]) > 0:
                    fpr[label_idx], tpr[label_idx], _ = roc_curve(y_bin[:, i], y_scores_filtered[:, i])
                    roc_auc_dict[label_idx] = roc_auc_score(y_bin[:, i], y_scores_filtered[:, i])

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 30

            plt.figure(figsize=(12, 8))
            colors = plt.cm.Set1(np.linspace(0, 1, len(top_5_indices)))
            for idx, color in zip(top_5_indices, colors):
                if idx in roc_auc_dict:
                    plt.plot(fpr[idx], tpr[idx], color=color, lw=1,
                            label=f"{top_5_labels[np.where(top_5_indices == idx)[0][0]]} (AUC = {roc_auc_dict[idx]:.3f})")

            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Per-Class ROC - Client {self.client_id}  - Macro AUC: {roc_auc:.3f}')
            plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02), fontsize=22, frameon=True)
            plt.grid(True, linestyle='--', alpha=0.7)

            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_roc_curve_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[CLIENT {self.client_id}] Đã lưu ROC curve với macro AUC: {roc_auc:.4f}")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu ROC curve với macro AUC: {roc_auc:.4f}")
            return roc_auc

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi vẽ ROC curve: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi vẽ ROC curve: {e}")
            return 0.5

    def _plot_training_progress(self):
        try:
            if not self.train_loss_history or not self.val_accuracy_history or len(self.train_loss_history) != len(self.val_accuracy_history):
                print(f"[CLIENT {self.client_id}] Bỏ qua vẽ training progress vì dữ liệu không khớp")
                return

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 24

            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            epochs = list(range(1, len(self.train_loss_history) + 1))

            l1, = ax1.plot(epochs, self.train_loss_history, label="Train Loss", color="red")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="red")
            ax1.tick_params(axis='y', labelcolor="red")
            ax1.grid(True, linestyle='--', alpha=0.7)

            l2, = ax1.plot(epochs, self.val_loss_history, label="Val Loss", color="gold")

            ax2 = ax1.twinx()
            l3, = ax2.plot(epochs, self.val_accuracy_history, label="Val Accuracy", color="blue")
            ax2.set_ylabel("Accuracy", color="blue")
            ax2.tick_params(axis='y', labelcolor="blue")

            plt.title(f"Training Progress - Client {self.client_id}", fontweight="bold")

            lines = [l1, l2, l3]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

            fig.tight_layout()

            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_training_progress_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[CLIENT {self.client_id}] Đã lưu tiến trình training")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu tiến trình training")

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi vẽ tiến trình training: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ tiến trình training: {e}")

    def _plot_grad_norm(self):
        try:
            if not self.grad_norm_history:
                return
                
            plt.figure(figsize=(8, 5))
            rounds = list(range(1, len(self.grad_norm_history) + 1))
            plt.plot(rounds, self.grad_norm_history, label="Gradient Norm", marker='o', color='green')
            plt.xlabel("Round")
            plt.ylabel("Gradient Norm")
            plt.title(f"Gradient Norm - Client {self.client_id}")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            for i, norm in enumerate(self.grad_norm_history):
                plt.annotate(f'{norm:.2f}', (i+1, norm), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
            
            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_grad_norm_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[CLIENT {self.client_id}] Đã lưu biểu đồ gradient norm")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu biểu đồ gradient norm")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi vẽ gradient norm: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ gradient norm: {e}")

    def _plot_prf_curve(self, y_true, y_scores, num_thresholds=30):
        try:
            thresholds = np.linspace(0, 1, num_thresholds)
            precisions, recalls, f1s = [], [], []

            if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                y_pred_classes = np.argmax(y_scores, axis=1)
                y_prob = np.max(y_scores, axis=1)
            else:
                if y_scores.ndim == 1:
                    y_prob = y_scores
                    y_pred_classes = (y_scores >= 0.5).astype(int)
                else:
                    y_prob = y_scores[:, 1] 
                    y_pred_classes = (y_prob >= 0.5).astype(int)

            for t in thresholds:
                if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                    confident_mask = y_prob >= t
                    if np.sum(confident_mask) == 0:
                        y_pred_thresh = np.zeros_like(y_true)
                    else:
                        y_pred_thresh = np.zeros_like(y_true)
                        y_pred_thresh[confident_mask] = y_pred_classes[confident_mask]
                else:
                    y_pred_thresh = (y_prob >= t).astype(int)
                
                precision = precision_score(y_true, y_pred_thresh, average="weighted", zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, average="weighted", zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, average="weighted", zero_division=0)
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precisions, marker='o', label="Precision", color="skyblue", linewidth=2)
            plt.plot(thresholds, recalls, marker='s', label="Recall", color="darkblue", linewidth=2)
            plt.plot(thresholds, f1s, marker='^', label="F1", color="lightgreen", linewidth=2)

            plt.xlabel("Threshold", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Precision–Recall–F1 Curve - Client {self.client_id} ", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.tight_layout()

            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_prf_curve_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[CLIENT {self.client_id}] Đã lưu Precision-Recall-F1 curve: {save_path}")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu Precision-Recall-F1 curve")

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi vẽ PRF curve: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi vẽ PRF curve: {e}")   


    def _plot_prf_curve_multiclass(self, y_true, y_scores, num_thresholds=30):
        """Vẽ PRF curve cho multi-class với cách tiếp cận khác"""
        try:
            thresholds = np.linspace(0.1, 1.0, num_thresholds)
            precisions, recalls, f1s = [], [], []
            
            y_pred_default = np.argmax(y_scores, axis=1)
            max_probs = np.max(y_scores, axis=1)
            
            for t in thresholds:
                confident_mask = max_probs >= t
                
                if np.sum(confident_mask) == 0:
                    precisions.append(0.0)
                    recalls.append(0.0) 
                    f1s.append(0.0)
                else:
                    y_pred_masked = np.full_like(y_true, -1)
                    y_pred_masked[confident_mask] = y_pred_default[confident_mask]
                    
                    if np.sum(confident_mask) > 0:
                        y_true_confident = y_true[confident_mask]
                        y_pred_confident = y_pred_masked[confident_mask]
                        
                        precision = precision_score(y_true_confident, y_pred_confident, 
                                                average="weighted", zero_division=0)
                        recall = recall_score(y_true_confident, y_pred_confident, 
                                            average="weighted", zero_division=0)
                        f1 = f1_score(y_true_confident, y_pred_confident, 
                                    average="weighted", zero_division=0)
                        
                        recall = recall * (np.sum(confident_mask) / len(y_true))
                    else:
                        precision = recall = f1 = 0.0
                        
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)

            plt.figure(figsize=(12, 8))
            plt.plot(thresholds, precisions, marker='o', label="Precision", 
                    color="red", linewidth=2, markersize=6)
            plt.plot(thresholds, recalls, marker='s', label="Recall", 
                    color="blue", linewidth=2, markersize=6)
            plt.plot(thresholds, f1s, marker='^', label="F1-Score", 
                    color="green", linewidth=2, markersize=6)

            plt.xlabel("Confidence Threshold", fontsize=12)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Precision-Recall-F1 vs Confidence Threshold\nClient {self.client_id} ", 
                    fontsize=14)
            # Increase figure size for better appearance
            plt.legend(fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.xlim([0.1, 1.0])
            plt.ylim([0, 1.05])
            
            max_f1_idx = np.argmax(f1s)
            max_f1_threshold = thresholds[max_f1_idx]
            max_f1_value = f1s[max_f1_idx]
            plt.annotate(f'Max F1: {max_f1_value:.3f}\nat t={max_f1_threshold:.2f}',
                        xy=(max_f1_threshold, max_f1_value),
                        xytext=(max_f1_threshold-0.1, max_f1_value+0.1),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, ha='center')
            
            plt.tight_layout()

            save_path = os.path.join(self.IMAGES_CLIENT_DIR, 
                                    f"client_{self.client_id}_prf_curve_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[CLIENT {self.client_id}] Đã lưu Precision-Recall-F1 curve: {save_path}")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu Precision-Recall-F1 curve")

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi vẽ PRF curve: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi vẽ PRF curve: {e}")

    def get_parameters(self, config=None):
        if self.round_count >= MAX_ROUNDS:
            print(f"[CLIENT {self.client_id}] Đã đạt {MAX_ROUNDS} rounds, từ chối tham gia")
            logging.info(f"[CLIENT {self.client_id}] Đã đạt {MAX_ROUNDS} rounds, từ chối tham gia")
            raise RuntimeError(f"Client {self.client_id} đã hoàn thành {MAX_ROUNDS} rounds")
            
        try:
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            buf.seek(0)
            print(f"[CLIENT {self.client_id}] Đã chuẩn bị parameters")
            return [np.frombuffer(buf.getvalue(), dtype=np.uint8)]
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi get_parameters: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi get_parameters: {e}")
            raise

    def set_parameters(self, parameters, config=None):
        try:
            if isinstance(parameters, list):
                buf = io.BytesIO(parameters[0].tobytes())
            else:
                buf = io.BytesIO(parameters_to_ndarrays(parameters)[0].tobytes())
                
            state_dict = torch.load(buf, weights_only=True, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=True)
            
            print(f"[CLIENT {self.client_id}] Đã cập nhật parameters từ server")
            logging.info(f"[CLIENT {self.client_id}] Đã cập nhật parameters từ server")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi set_parameters: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi set_parameters: {str(e)}")
            raise

    def fit(self, parameters, config=None):
        self.round_count += 1
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

        print(f"\n[CLIENT {self.client_id}] ===== ROUND {self.round_count}/{MAX_ROUNDS} =====")
        logging.info(f"[CLIENT {self.client_id}] Bắt đầu round {self.round_count}/{MAX_ROUNDS}")

        if self.round_count > MAX_ROUNDS:
            print(f"[CLIENT {self.client_id}] Vượt quá {MAX_ROUNDS} rounds, từ chối training")
            logging.info(f"[CLIENT {self.client_id}] Vượt quá {MAX_ROUNDS} rounds, từ chối training")
            raise RuntimeError(f"Client {self.client_id} đã hoàn thành {MAX_ROUNDS} rounds")

        try:
            self.set_parameters(parameters)
            
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-5)
            
            print(f"[CLIENT {self.client_id}] Bắt đầu training {EPOCHS} epochs...")
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for cat_data, cont_data, labels in self.train_loader:
                    outputs = self.model(cat_data, cont_data)
                    loss = self.loss_fn(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                avg_train_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
                self.train_loss_history.append(avg_train_loss)
                
                val_loss, val_acc, val_f1, val_precision, val_recall = self._compute_validation_metrics()
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)

                print(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
                logging.info(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")

            print(f"[CLIENT {self.client_id}] Training hoàn thành")

            # Compute gradient norm and FedMADE weights
            gradients = [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
            grad_norm = self._compute_gradient_norm(gradients)
            self.grad_norm_history.append(grad_norm)
            fedmade_weight = self._compute_fedmade_weights(self.val_accuracy_history[-1], grad_norm)

            gradients = self._sparsify_gradients(gradients, SPARSITY)

            self._save_state()

            self._plot_training_progress()
            self._plot_grad_norm()

            # Evaluate on validation set
            val_loss, num_examples, metrics = self.evaluate(parameters, config)
            final_accuracy = metrics.get("accuracy", 0.0)

            y_true_list, y_pred_list, y_scores_list = [], [], []

            self.model.eval()
            with torch.no_grad():
                for cat_data, cont_data, labels in self.val_loader:
                    outputs = self.model(cat_data, cont_data)
                    y_pred = torch.argmax(outputs, dim=1)
                    y_scores = torch.softmax(outputs, dim=1)
                    y_true_list.extend(labels.numpy())
                    y_pred_list.extend(y_pred.numpy())
                    y_scores_list.extend(y_scores.numpy())

            y_true = np.array(y_true_list)
            y_scores = np.array(y_scores_list)

            if self.num_classes == 2:
                self._plot_prf_curve(y_true, y_scores)
            else:
                self._plot_prf_curve_multiclass(y_true, y_scores)

            print(f"[CLIENT {self.client_id}] Final evaluation - Val Loss: {val_loss:.4f}, Val Accuracy: {final_accuracy:.4f}")
            print(f"[CLIENT {self.client_id}] Metrics: {metrics}")
            logging.info(f"[CLIENT {self.client_id}] Final evaluation - Val Loss: {val_loss:.4f}, Val Accuracy: {final_accuracy:.4f}")

            params_buf = io.BytesIO()
            torch.save(self.model.state_dict(), params_buf)
            params_buf.seek(0)

            grad_buf = io.BytesIO()
            torch.save(gradients, grad_buf)
            grad_buf.seek(0)

            metrics = {
                "accuracy": float(final_accuracy),
                "f1_score": float(metrics.get("f1_score", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "loss": float(metrics.get("loss", val_loss)),
                "val_loss": float(self.val_loss_history[-1]),
                "val_accuracy": float(self.val_accuracy_history[-1]),
                "fedmade_weight": float(fedmade_weight),
                "grad_norm": float(grad_norm),
                "cluster_label": int(np.bincount(self.cluster_labels).argmax() if len(self.cluster_labels) > 0 else 0),
                "stop_after_this_round": self.round_count >= MAX_ROUNDS,
                "num_samples": len(self.y_train)
            }

            if self.round_count >= MAX_ROUNDS:
                print(f"[CLIENT {self.client_id}] Hoàn thành {MAX_ROUNDS} rounds, sẽ dừng sau round này")
                logging.info(f"[CLIENT {self.client_id}] Hoàn thành {MAX_ROUNDS} rounds, sẽ dừng")

            print(f"[CLIENT {self.client_id}] ===== KẾT THÚC ROUND {self.round_count} =====\n")

            return (
                [np.frombuffer(params_buf.getvalue(), dtype=np.uint8), 
                np.frombuffer(grad_buf.getvalue(), dtype=np.uint8)],
                len(self.y_train),
                metrics
            )

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi trong fit: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi trong fit: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, parameters, config=None):
        print(f"[CLIENT {self.client_id}] Bắt đầu evaluation...")
        logging.info(f"[CLIENT {self.client_id}] Bắt đầu evaluation")

        try:
            if self.y_val is None or len(self.y_val) == 0:
                print(f"[CLIENT {self.client_id}] Không có dữ liệu val để evaluate")
                logging.warning(f"[CLIENT {self.client_id}] Val set trống, bỏ qua evaluate")
                return 0.0, 0, {}

            self.set_parameters(parameters)

            self.model.eval()

            y_pred_list = []
            y_true_list = []
            y_scores_list = []
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for cat_data, cont_data, labels in self.val_loader:
                    outputs = self.model(cat_data, cont_data)
                    loss = self.loss_fn(outputs, labels)

                    y_pred = torch.argmax(outputs, dim=1)
                    y_scores = torch.softmax(outputs, dim=1)
                    
                    y_pred_list.append(y_pred.numpy())
                    y_true_list.append(labels.numpy())
                    y_scores_list.append(y_scores.numpy())

                    total_loss += loss.item()
                    num_batches += 1

            if num_batches == 0:
                print(f"[CLIENT {self.client_id}] Không có batch nào trong val_loader")
                logging.warning(f"[CLIENT {self.client_id}] Val loader rỗng")
                return 0.0, 0, {}

            y_pred = np.concatenate(y_pred_list)
            y_true = np.concatenate(y_true_list)
            y_scores = np.concatenate(y_scores_list)
            avg_loss = total_loss / num_batches

            num_examples = len(y_true)

            if num_examples > 0:
                self._plot_confusion_matrix(y_true, y_pred)
                self._plot_label_distribution(y_true, y_pred)
                self._plot_roc_curve(y_true, y_scores)

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"[CLIENT {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
            logging.info(f"[CLIENT {self.client_id}] Evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")

            metrics = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "loss": float(avg_loss),
                "val_loss": float(avg_loss),
                "val_accuracy": float(accuracy)
            }
            return float(avg_loss), int(num_examples), metrics
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi evaluate: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi evaluate: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, 0, {}

def start_client(client_id):
    try:
        print(f"\n===== KHỞI ĐỘNG CLIENT {client_id} =====")
        client = FLClientTabTransformer(client_id)
        
        print(f"[CLIENT {client_id}] Kết nối đến server tại 127.0.0.1:8080...")
        fl.client.start_client(
            server_address="127.0.0.1:8080", 
            client=client.to_client()
        )
        
        print(f"[CLIENT {client_id}] Đã hoàn thành kết nối với server")
        logging.info(f"[CLIENT {client_id}] Đã hoàn thành kết nối với server")
        
    except Exception as e:
        print(f"[CLIENT {client_id}] Lỗi khởi động client: {str(e)}")
        logging.error(f"[CLIENT {client_id}] Lỗi khởi động client: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print(f"[CLIENT {client_id}] ===== CLIENT DỪNG =====")


if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    start_client(client_id)