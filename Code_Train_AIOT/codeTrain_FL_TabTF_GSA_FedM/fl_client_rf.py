import os
import io
import joblib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flwr as fl
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer
from data_processing import load_and_process_data
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

logging.basicConfig(filename="client_log1.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

DATA_PATH = "dataset/data_All_1k.csv"
MODEL_DIR = "models_FL_Tab_GSA_FedM_1k"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
EPOCHS = 30
BATCH_SIZE = 256 
LR = 0.001  
SPARSITY = 0.5
SIMILARITY_THRESHOLD = 0.3 
LAMBDA_PERF = 0.6 
LAMBDA_CONV = 0.4 
MAX_ROUNDS = 3 

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

class FLClientTabTransformer(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.round_count = 0 
        print(f"[CLIENT {client_id}] Khởi tạo client TabTransformer...")
        logging.info(f"[CLIENT {client_id}] Khởi tạo client TabTransformer...")
        ensure_dir(MODEL_DIR)

        try:
            self.X_train_raw, self.y_train, self.categorical_cols, self.num_classes, self.le, self.cluster_labels = load_and_process_data(DATA_PATH)
            print(f"[CLIENT {client_id}] Đã load dữ liệu: {len(self.X_train_raw)} mẫu, {len(self.categorical_cols)} cột phân loại, {self.num_classes} lớp")
            logging.info(f"[CLIENT {client_id}] Đã load dữ liệu: {len(self.X_train_raw)} mẫu, {len(self.categorical_cols)} cột phân loại, {self.num_classes} lớp")
        except Exception as e:
            print(f"[CLIENT {client_id}] Lỗi khi load dữ liệu: {str(e)}")
            logging.error(f"[CLIENT {client_id}] Lỗi khi load dữ liệu: {str(e)}")
            raise

        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            self.X_train_raw, self.y_train, test_size=0.2, stratify=self.y_train, random_state=42
        )
        print(f"[CLIENT {client_id}] Dữ liệu: {len(self.X_train_raw)} train / {len(self.X_test_raw)} test")
        logging.info(f"[CLIENT {client_id}] Dữ liệu: {len(self.X_train_raw)} train / {len(self.X_test_raw)} test")

        cont_cols = [c for c in self.X_train_raw.columns if c not in self.categorical_cols]
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train_raw[cont_cols]) if cont_cols else self.X_train_raw.values
        self.X_test = self.scaler.transform(self.X_test_raw[cont_cols]) if cont_cols else self.X_test_raw.values
        X_train_cat = self.X_train_raw[self.categorical_cols].values.astype(int) if self.categorical_cols else np.zeros((len(self.X_train_raw), 0))
        X_test_cat = self.X_test_raw[self.categorical_cols].values.astype(int) if self.categorical_cols else np.zeros((len(self.X_test_raw), 0))

        print(f"[CLIENT {client_id}] Giá trị min/max sau chuẩn hóa: {self.X_train.min():.4f}/{self.X_train.max():.4f}")
        logging.info(f"[CLIENT {client_id}] Giá trị min/max sau chuẩn hóa: {self.X_train.min():.4f}/{self.X_train.max():.4f}")

        self.X_train_cont = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test_cont = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        self.X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

        train_dataset = TensorDataset(self.X_train_cat, self.X_train_cont, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = TensorDataset(self.X_test_cat, self.X_test_cont, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_counts = np.bincount(self.y_train)
        class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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

        print(f"[CLIENT {client_id}] Mô hình sẵn sàng với {self.num_classes} lớp đầu ra.")
        logging.info(f"[CLIENT {client_id}] Mô hình sẵn sàng với {self.num_classes} lớp đầu ra.")

    def _model_files_exist(self):
        return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)

    def _load_model_and_scaler(self):
        print(f"[CLIENT {self.client_id}] Tải lại model/scaler/encoder...")
        logging.info(f"[CLIENT {self.client_id}] Tải lại model/scaler/encoder...")
        try:
            state_dict = torch.load(MODEL_PATH, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.scaler = joblib.load(SCALER_PATH)
            self.le = joblib.load(ENCODER_PATH)
            print(f"[CLIENT {self.client_id}] Đã load model/scaler/encoder.")
            logging.info(f"[CLIENT {self.client_id}] Đã load model/scaler/encoder.")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi load model/scaler: {e}. Tạo mới...")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi load model/scaler: {e}")
            self._save_state()

    def _save_state(self):
        torch.save(self.model.state_dict(), MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.le, ENCODER_PATH)
        print(f"[CLIENT {self.client_id}] Đã lưu model/scaler/encoder.")
        logging.info(f"[CLIENT {self.client_id}] Đã lưu model/scaler/encoder.")

    def _sparsify_gradients(self, gradients, sparsity=SPARSITY):
        if not gradients:
            return gradients
        try:
            flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
            if len(flat_grads) == 0:
                return gradients
            k = int((1 - sparsity) * len(flat_grads))
            threshold = torch.kthvalue(torch.abs(flat_grads), len(flat_grads) - k).values
            for g in gradients:
                if g is not None:
                    g.masked_fill_(torch.abs(g) < threshold, 0.0)
            print(f"[CLIENT {self.client_id}] Sparsified gradients, sparsity: {sparsity}, threshold: {threshold:.4f}")
            logging.info(f"[CLIENT {self.client_id}] Sparsified gradients, sparsity: {sparsity}, threshold: {threshold:.4f}")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi sparsify gradients: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi sparsify gradients: {e}")
        return gradients

    def _compute_validation_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for cat_data, cont_data, labels in self.test_loader:
                out = self.model(cat_data, cont_data)
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"[CLIENT {self.client_id}] Validation Accuracy: {accuracy:.4f}")
        logging.info(f"[CLIENT {self.client_id}] Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def _compute_gradient_norm(self, gradients):
        if not gradients:
            return 0.0
        try:
            flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
            norm = torch.norm(flat_grads, p=2)
            print(f"[CLIENT {self.client_id}] Gradient norm: {norm.item():.4f}")
            logging.info(f"[CLIENT {self.client_id}] Gradient norm: {norm.item():.4f}")
            return norm.item()
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi tính gradient norm: {e}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi tính gradient norm: {e}")
            return 0.0

    def _compute_fedmade_weights(self, accuracy, grad_norm):
        w_perf = np.exp(accuracy)
        w_conv = np.exp(-grad_norm)
        w = LAMBDA_PERF * w_perf + LAMBDA_CONV * w_conv
        print(f"[CLIENT {self.client_id}] FedMADE weights: w_perf={w_perf:.4f}, w_conv={w_conv:.4f}, w={w:.4f}")
        logging.info(f"[CLIENT {self.client_id}] FedMADE weights: w_perf={w_perf:.4f}, w_conv={w_conv:.4f}, w={w:.4f}")
        return w

    def get_parameters(self, config=None):
        if self.round_count >= MAX_ROUNDS:
            print(f"[CLIENT {self.client_id}] Đã đạt {MAX_ROUNDS} vòng liên kết, từ chối tham gia thêm.")
            logging.info(f"[CLIENT {self.client_id}] Đã đạt {MAX_ROUNDS} vòng liên kết, từ chối tham gia thêm.")
            raise RuntimeError(f"Client {self.client_id} đã hoàn thành {MAX_ROUNDS} vòng liên kết.")
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        buf.seek(0)
        return [np.frombuffer(buf.getvalue(), dtype=np.uint8)]

    def set_parameters(self, parameters, config=None):
        try:
            if isinstance(parameters, list):
                buf = io.BytesIO(parameters[0].tobytes())
            else:
                buf = io.BytesIO(parameters_to_ndarrays(parameters)[0].tobytes())
            state_dict = torch.load(buf, weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[CLIENT {self.client_id}] Đã cập nhật tham số mô hình từ server.")
            logging.info(f"[CLIENT {self.client_id}] Đã cập nhật tham số mô hình từ server.")
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi cập nhật tham số: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi cập nhật tham số: {str(e)}")
            raise

    def fit(self, parameters, config=None):
        self.round_count += 1 
        print(f"[CLIENT {self.client_id}] Bắt đầu vòng liên kết {self.round_count}/{MAX_ROUNDS}")
        logging.info(f"[CLIENT {self.client_id}] Bắt đầu vòng liên kết {self.round_count}/{MAX_ROUNDS}")

        if self.round_count > MAX_ROUNDS:
            print(f"[CLIENT {self.client_id}] Đã vượt quá {MAX_ROUNDS} vòng liên kết, từ chối huấn luyện.")
            logging.info(f"[CLIENT {self.client_id}] Đã vượt quá {MAX_ROUNDS} vòng liên kết, từ chối huấn luyện.")
            raise RuntimeError(f"Client {self.client_id} đã hoàn thành {MAX_ROUNDS} vòng liên kết.")

        try:
            # Always load the server's aggregated model parameters
            self.set_parameters(parameters)
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-5)

            for epoch in range(EPOCHS):
                for cat_data, cont_data, labels in self.train_loader:
                    out = self.model(cat_data, cont_data)
                    loss = self.loss_fn(out, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")
                logging.info(f"[CLIENT {self.client_id}] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

            accuracy = self._compute_validation_accuracy()
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            grad_norm = self._compute_gradient_norm(gradients)
            weight = self._compute_fedmade_weights(accuracy, grad_norm)

            gradients = self._sparsify_gradients(gradients, SPARSITY)

            grad_buf = io.BytesIO()
            torch.save(gradients, grad_buf)
            grad_buf.seek(0)

            self._save_state()  # Save model state after training
            eval_metrics = self.evaluate(parameters, config)
            acc = eval_metrics[0]
            print(f"[CLIENT {self.client_id}] Accuracy sau fit: {acc:.4f}")
            logging.info(f"[CLIENT {self.client_id}] Accuracy sau fit: {acc:.4f}")

            params_buf = io.BytesIO()
            torch.save(self.model.state_dict(), params_buf)
            params_buf.seek(0)

            metrics = {
                "accuracy": float(acc),
                "f1_score": float(eval_metrics[2]["f1_score"]),
                "precision": float(eval_metrics[2]["precision"]),
                "recall": float(eval_metrics[2]["recall"]),
                "fedmade_weight": float(weight),
                "grad_norm": float(grad_norm),
                "cluster_label": int(np.bincount(self.cluster_labels).argmax()),
                "stop_after_this_round": self.round_count >= MAX_ROUNDS
            }

            if self.round_count >= MAX_ROUNDS:
                print(f"[CLIENT {self.client_id}] Đã hoàn thành {MAX_ROUNDS} vòng liên kết, sẽ dừng sau vòng này.")
                logging.info(f"[CLIENT {self.client_id}] Đã hoàn thành {MAX_ROUNDS} vòng liên kết, sẽ dừng sau vòng này.")

            return (
                [np.frombuffer(params_buf.getvalue(), dtype=np.uint8), np.frombuffer(grad_buf.getvalue(), dtype=np.uint8)],
                len(self.y_train),
                metrics
            )
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi trong quá trình fit: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi trong quá trình fit: {str(e)}")
            raise

    def evaluate(self, parameters, config=None):
        print(f"[CLIENT {self.client_id}] Bắt đầu đánh giá với config: {config}")
        logging.info(f"[CLIENT {self.client_id}] Bắt đầu đánh giá với config: {config}")
        try:
            self.set_parameters(parameters)
            self.model.eval()
            y_pred_list = []
            y_true_list = []
            with torch.no_grad():
                for cat_data, cont_data, labels in self.test_loader:
                    out = self.model(cat_data, cont_data)
                    y_pred = torch.argmax(out, dim=1)
                    y_pred_list.append(y_pred.numpy())
                    y_true_list.append(labels.numpy())
            y_pred = np.concatenate(y_pred_list)
            y_true = np.concatenate(y_true_list)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            num_examples = len(y_true)
            print(f"[CLIENT {self.client_id}] Phân phối dự đoán: {np.bincount(y_pred, minlength=self.num_classes).tolist()}")
            print(f"[CLIENT {self.client_id}] Phân phối nhãn thực: {np.bincount(y_true, minlength=self.num_classes).tolist()}")
            print(f"[CLIENT {self.client_id}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            logging.info(f"[CLIENT {self.client_id}] Phân phối dự đoán: {np.bincount(y_pred, minlength=self.num_classes).tolist()}")
            logging.info(f"[CLIENT {self.client_id}] Phân phối nhãn thực: {np.bincount(y_true, minlength=self.num_classes).tolist()}")
            logging.info(f"[CLIENT {self.client_id}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            return float(acc), num_examples, {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}
        except Exception as e:
            print(f"[CLIENT {self.client_id}] Lỗi khi đánh giá: {str(e)}")
            logging.error(f"[CLIENT {self.client_id}] Lỗi khi đánh giá: {str(e)}")
            raise

def start_client(client_id):
    try:
        client = FLClientTabTransformer(client_id)
        fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
        print(f"[CLIENT {client_id}] Đã kết nối với server.")
        logging.info(f"[CLIENT {client_id}] Đã kết nối với server.")
    except Exception as e:
        print(f"[CLIENT {client_id}] Lỗi khi khởi động client: {str(e)}")
        logging.error(f"[CLIENT {client_id}] Lỗi khi khởi động client: {str(e)}")
        raise 

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    start_client(client_id)