import os
import joblib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer
from collections import Counter
from imblearn.over_sampling import SMOTE

matplotlib.use('Agg')

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13
})

OUTPUT_DIR = "fl_1k_DDoS_outputs"
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001
MAX_ROUNDS = 10

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

class FLClientTabTransformer(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.round_count = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.val_f1_history = []

        self.MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
        self.IMAGES_CLIENT_DIR = os.path.join(OUTPUT_DIR, "images", f"client_{client_id}")
        self.LOG_PATH = os.path.join(OUTPUT_DIR, "logs", f"client_log_{client_id}.txt")
        self.OPTIMIZER_STATE_PATH = os.path.join(self.MODEL_DIR, f"client_{client_id}_optimizer.pt")

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

        data_path = os.path.join(OUTPUT_DIR, f"client_{client_id}_data.csv")
        if not os.path.exists(data_path):
            raise ValueError(f"[CLIENT {client_id}] Data file not found at {data_path}")

        df = pd.read_csv(data_path)
        label_col = "Label"
        if label_col not in df.columns:
            raise ValueError(f"[CLIENT {client_id}] Required column '{label_col}' not found in {data_path}")

        X = df.drop(columns=[label_col])
        y = df[label_col].values.astype(int)

        scaler_path = os.path.join(self.MODEL_DIR, "scaler_server.pkl")
        categories_path = os.path.join(self.MODEL_DIR, "categories.pkl")
        num_cont_path = os.path.join(self.MODEL_DIR, "num_continuous.pkl")
        cat_cols_path = os.path.join(self.MODEL_DIR, "cat_cols.pkl")

        if not (os.path.exists(scaler_path) and os.path.exists(categories_path) and
                os.path.exists(num_cont_path) and os.path.exists(cat_cols_path)):
            raise RuntimeError(f"[CLIENT {client_id}] Missing server metadata. Ensure server produced models/*.pkl")

        self.scaler = joblib.load(scaler_path)
        categories_sizes = joblib.load(categories_path)
        num_cont = joblib.load(num_cont_path)
        categorical_cols = joblib.load(cat_cols_path)  # enforce same order as server

        num_classes = int(y.max() + 1)

        label_counts = Counter(y)
        print(f"[CLIENT {client_id}] Label distribution: {label_counts}")
        logging.info(f"[CLIENT {client_id}] Label distribution: {label_counts}")

        # Make sure columns exist (if server dropped some, align gracefully)
        missing_cats = [c for c in categorical_cols if c not in X.columns]
        if missing_cats:
            raise RuntimeError(f"[CLIENT {client_id}] Missing categorical columns in client data: {missing_cats}")

        cont_cols = [c for c in X.columns if c not in categorical_cols]

        # === Split train/val ===
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # === SMOTE on numeric data (assumes numeric features) ===
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logging.info(f"[CLIENT {client_id}] Applied SMOTE: Original {len(X_train)}, after SMOTE {len(X_train_resampled)}")

        # === Transform using server scaler (DO NOT FIT HERE) ===
        if cont_cols:
            X_train_cont = self.scaler.transform(X_train_resampled[cont_cols])
            X_val_cont = self.scaler.transform(X_val[cont_cols])
        else:
            X_train_cont = X_train_resampled.values
            X_val_cont = X_val.values

        # categorical as indices (int) – keep exact order from server
        X_train_cat = X_train_resampled[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_train_resampled), 0))
        X_val_cat = X_val[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_val), 0))
        self.y_train = torch.tensor(y_train_resampled, dtype=torch.long)
        self.y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(
            torch.tensor(X_train_cat, dtype=torch.long),
            torch.tensor(X_train_cont, dtype=torch.float32),
            self.y_train
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_cat, dtype=torch.long),
            torch.tensor(X_val_cont, dtype=torch.float32),
            self.y_val
        )
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # === Initialize model strictly from server metadata ===
        self.hyperparams = {
            "categories": categories_sizes,
            "num_continuous": num_cont,
            "dim": 128,
            "dim_out": num_classes,
            "depth": 6,
            "heads": 8,
            "attn_dropout": 0.3,
            "ff_dropout": 0.3,
            "mlp_hidden_mults": (4, 2),
            "mlp_act": nn.ReLU()
        }
        self.model = TabTransformer(**self.hyperparams)

        # Optionally warm start from last global model if present locally
        agg_model_path = os.path.join(self.MODEL_DIR, "aggregated_model.pt")
        if os.path.exists(agg_model_path):
            try:
                self.model.load_state_dict(torch.load(agg_model_path, map_location="cpu"))
                print(f"[CLIENT {client_id}] Loaded existing aggregated model for warm-start fine-tune.")
                logging.info(f"[CLIENT {client_id}] Loaded existing aggregated model for warm-start fine-tune.")
            except Exception as e:
                print(f"[CLIENT {client_id}] WARNING: Failed to load local aggregated model: {e}")
                logging.warning(f"[CLIENT {self.client_id}] Failed to load local aggregated model: {e}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        print(f"[CLIENT {client_id}] Model initialized with {self.hyperparams}")
        logging.info(f"[CLIENT {client_id}] Model initialized with {self.hyperparams}")

    def fit(self, parameters, config=None):
        self.round_count += 1
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.val_f1_history = []

        print(f"\n[CLIENT {self.client_id}] ===== ROUND {self.round_count}/{MAX_ROUNDS} =====")
        logging.info(f"[CLIENT {self.client_id}] Bắt đầu round {self.round_count}/{MAX_ROUNDS}")

        if self.round_count > MAX_ROUNDS:
            print(f"[CLIENT {self.client_id}] Vượt quá {MAX_ROUNDS} rounds, từ chối training")
            logging.info(f"[CLIENT {self.client_id}] Vượt quá {MAX_ROUNDS} rounds, từ chối training")
            raise RuntimeError(f"Client {self.client_id} đã hoàn thành {MAX_ROUNDS} rounds")

        try:
            # Load parameters from server if provided
            if parameters:
                self.set_parameters(parameters)
                print(f"[CLIENT {self.client_id}] Loaded server parameters for round {self.round_count}")
                logging.info(f"[CLIENT {self.client_id}] Loaded server parameters for round {self.round_count}. Parameters length: {len(parameters)}")

            # Load previous optimizer state if exists (for continuity)
            if os.path.exists(self.OPTIMIZER_STATE_PATH):
                self.optimizer.load_state_dict(torch.load(self.OPTIMIZER_STATE_PATH))
                print(f"[CLIENT {self.client_id}] Loaded previous optimizer state")
                logging.info(f"[CLIENT {self.client_id}] Loaded previous optimizer state from {self.OPTIMIZER_STATE_PATH}")

            self.model.train()
            
            print(f"[CLIENT {self.client_id}] Bắt đầu training {EPOCHS} epochs...")
            logging.info(f"[CLIENT {self.client_id}] Starting training for {EPOCHS} epochs")
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                epoch_batches = 0
                
                for cat_data, cont_data, labels in self.train_loader:
                    outputs = self.model(cat_data, cont_data)
                    loss = self.loss_fn(outputs, labels)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                avg_train_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
                self.train_loss_history.append(avg_train_loss)
                
                val_loss, val_acc, val_f1, val_precision, val_recall = self._compute_validation_metrics()
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)
                self.val_f1_history.append(val_f1)

                print(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
                logging.info(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Compute CPM on local val for metrics (though server recomputes)
            cpm = self._compute_cpm()
            logging.info(f"[CLIENT {self.client_id}] Computed CPM with shape {np.array(cpm).shape}")

            torch.save(self.optimizer.state_dict(), self.OPTIMIZER_STATE_PATH)
            print(f"[CLIENT {self.client_id}] Saved optimizer state to {self.OPTIMIZER_STATE_PATH}")
            logging.info(f"[CLIENT {self.client_id}] Saved optimizer state to {self.OPTIMIZER_STATE_PATH}")

            print(f"[CLIENT {self.client_id}] Training hoàn thành")
            logging.info(f"[CLIENT {self.client_id}] Training completed. Final Val Acc: {val_acc:.4f}")

            self._plot_training_progress()

            print(f"[CLIENT {self.client_id}] Plotting validation results...")
            logging.info(f"[CLIENT {self.client_id}] Generating plots for validation results")
            y_true, y_pred, y_scores = [], [], []
            self.model.eval()
            with torch.no_grad():
                for cat_data, cont_data, labels in self.val_loader:
                    out = self.model(cat_data, cont_data)
                    pred = torch.argmax(out, dim=1)
                    score = torch.softmax(out, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
                    y_scores.extend(score.cpu().numpy())

            self._plot_confusion_matrix(y_true, y_pred)
            self._plot_roc_curve(y_true, np.array(y_scores))
            self._save_classification_report(y_true, y_pred)

            metrics = {
                "loss": avg_train_loss,
                "accuracy": val_acc,
                "cpm": json.dumps(cpm.tolist()) 
            }
            return self.get_parameters(), len(self.y_train), metrics

        except Exception as e:
            print(f"[CLIENT {self.client_id}] Error during training: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Error during training: {str(e)}")
            raise

    def _compute_cpm(self):
        """Compute Class Probability Matrix (CPM) for FedMADE on validation set."""
        self.model.eval()
        cpm = np.zeros((self.num_classes, self.num_classes))
        class_counts = np.zeros(self.num_classes)

        with torch.no_grad():
            for cat_data, cont_data, labels in self.val_loader:
                outputs = self.model(cat_data, cont_data)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                for i, label in enumerate(labels.cpu().numpy()):
                    cpm[label] += probs[i]
                    class_counts[label] += 1

        for c in range(self.num_classes):
            if class_counts[c] > 0:
                cpm[c] /= class_counts[c]
            else:
                cpm[c] = np.zeros(self.num_classes)

        return cpm

    def _compute_validation_metrics(self):
        self.model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for cat_data, cont_data, labels in self.val_loader:
                outputs = self.model(cat_data, cont_data)
                loss = self.loss_fn(outputs, labels)
                total_val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / val_total if val_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return avg_val_loss, val_accuracy, val_f1, val_precision, val_recall

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

    def _plot_confusion_matrix(self, y_true, y_pred):
        try:
            # Load LabelEncoder từ file được lưu bởi server
            encoder_path = os.path.join(self.MODEL_DIR, "label_encoder_server.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"[CLIENT {self.client_id}] LabelEncoder file not found at {encoder_path}")
            self.le = joblib.load(encoder_path)

            label_counts = np.bincount(y_true, minlength=self.num_classes)
            top_5_indices = np.argsort(label_counts)[::-1][:5]
            
            mask = np.isin(y_true, top_5_indices)
            y_true_filtered = np.array(y_true)[mask]
            y_pred_filtered = np.array(y_pred)[mask]
            
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

            ax.set_title(f'Confusion Matrix (%) - Client {self.client_id}', 
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=18)
            ax.set_ylabel('True Label', fontsize=18)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            save_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_confusion_matrix_round_{self.round_count}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CLIENT {self.client_id}] Saved confusion matrix at {save_path}")
            logging.info(f"[CLIENT {self.client_id}] Saved confusion matrix at {save_path}")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Error plotting confusion matrix: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Error plotting confusion matrix: {str(e)}")

    def _plot_roc_curve(self, y_true, y_scores):
        try:
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
            plt.title(f"ROC Curve - Client {self.client_id}")
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.IMAGES_CLIENT_DIR, f"roc_curve_round_{self.round_count}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[CLIENT {self.client_id}] Đã lưu ROC curve")
            logging.info(f"[CLIENT {self.client_id}] Đã lưu ROC curve")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi vẽ ROC curve: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi vẽ ROC curve: {str(e)}")

    def _save_classification_report(self, y_true, y_pred):
        try:
            # Load LabelEncoder từ file được lưu bởi server
            encoder_path = os.path.join(self.MODEL_DIR, "label_encoder_server.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"[CLIENT {self.client_id}] LabelEncoder file not found at {encoder_path}")
            self.le = joblib.load(encoder_path)

            # Lấy tên nhãn gốc
            labels = self.le.inverse_transform(range(self.num_classes))
            
            # Tạo classification report
            report = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
            
            # Lưu report vào file văn bản
            report_path = os.path.join(self.IMAGES_CLIENT_DIR, f"client_{self.client_id}_classification_report_round_{self.round_count}.txt")
            with open(report_path, 'w') as f:
                f.write(f"Classification Report - Client {self.client_id} (Round {self.round_count})\n\n")
                f.write(report)
            
            print(f"[CLIENT {self.client_id}] Saved classification report at {report_path}")
            logging.info(f"[CLIENT {self.client_id}] Saved classification report at {report_path}")
            
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Error generating classification report: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Error generating classification report: {str(e)}")

    def get_parameters(self, config=None):
        return [param.data.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters, config=None):
        params_dict = zip(self.model.parameters(), parameters)
        for param, data in params_dict:
            param.data = torch.tensor(data, dtype=torch.float32)

def start_client(client_id):
    try:
        print(f"\n===== KHỞI ĐỘNG CLIENT {client_id} =====")
        client = FLClientTabTransformer(client_id)
        print(f"[CLIENT {client_id}] Connecting to server at 127.0.0.1:8080...")
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=client.to_client()
        )
        print(f"[CLIENT {client_id}] Completed connection with server")
        logging.info(f"[CLIENT {client_id}] Completed connection with server")
    except Exception as e:
        print(f"[CLIENT {client_id}] Error starting client: {str(e)}")
        logging.error(f"[CLIENT {client_id}] Error starting client: {str(e)}")
        raise
    finally:
        print(f"[CLIENT {client_id}] ===== CLIENT STOPPED =====")

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    start_client(client_id)