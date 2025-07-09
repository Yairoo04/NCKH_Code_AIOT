import io
import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar, MetricsAggregationFn, ndarrays_to_parameters, parameters_to_ndarrays
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from data_processing import load_and_process_data
import time
import logging

logging.basicConfig(
    filename="log_server.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode='w'
)

DATA_PATH = "dataset/data_All_1k.csv"
MODEL_DIR = "models_FL_Tab_GSA_FedM_1k"
AGGREGATED_MODEL_PATH = os.path.join(MODEL_DIR, "aggregated_model.pt")
SIMILARITY_THRESHOLD = 0.2
LAMBDA_PERF = 0.6
LAMBDA_CONV = 0.4
MAX_ROUNDS = 3
CONVERGENCE_THRESHOLD = 0.001

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

class FLServerTabTransformer:
    def __init__(self):
        print("[SERVER] Khởi tạo server...")
        logging.info("[SERVER] Khởi tạo server...")
        ensure_dir(MODEL_DIR)
        X, y, categorical_cols, num_classes, _, _ = load_and_process_data(DATA_PATH)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        cont_cols = [c for c in X.columns if c not in categorical_cols]
        scaler = RobustScaler()
        X_test_cont = scaler.fit_transform(X_test[cont_cols]) if cont_cols else X_test.values
        X_test_cat = X_test[categorical_cols].values.astype(int) if categorical_cols else np.zeros((len(X_test), 0))

        self.X_test_cont = torch.tensor(X_test_cont, dtype=torch.float32)
        self.X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

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
        self.previous_accuracy = None
        self.convergence_round = None
        self.round_times = []
        print(f"[SERVER] TabTransformer sẵn sàng - {self.hyperparams}")
        logging.info(f"[SERVER] TabTransformer sẵn sàng - {self.hyperparams}")

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            out = model(self.X_test_cat, self.X_test_cont)
            y_pred = torch.argmax(out, dim=1).numpy()
            y_true = self.y_test.numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"[SERVER] Đánh giá - Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"[SERVER] Phân phối dự đoán: {np.bincount(y_pred, minlength=self.hyperparams['dim_out']).tolist()}")
            print(f"[SERVER] Phân phối nhãn thực: {np.bincount(y_true, minlength=self.hyperparams['dim_out']).tolist()}")
            logging.info(f"[SERVER] Đánh giá - Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            logging.info(f"[SERVER] Phân phối dự đoán: {np.bincount(y_pred, minlength=self.hyperparams['dim_out']).tolist()}")
            logging.info(f"[SERVER] Phân phối nhãn thực: {np.bincount(y_true, minlength=self.hyperparams['dim_out']).tolist()}")
            return {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}

    def load_model(self, round_num):
        model_path = os.path.join(MODEL_DIR, f"aggregated_model_round_{round_num}.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"[SERVER] Đã tải mô hình từ {model_path}")
            logging.info(f"[SERVER] Đã tải mô hình từ {model_path}")
        else:
            print(f"[SERVER] Không tìm thấy mô hình tại {model_path}, sử dụng mô hình hiện tại.")
            logging.info(f"[SERVER] Không tìm thấy mô hình tại {model_path}, sử dụng mô hình hiện tại.")

class FedMADEStrategy(FedAvg):
    def __init__(self, server: FLServerTabTransformer):
        super().__init__(
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics
        )
        self.server = server
        self.client_clusters = {}

    def aggregate_evaluate_metrics(self, results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]]) -> Dict[str, float]:
        if not results:
            return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}

        accuracies, f1_scores, precisions, recalls = [], [], [], []
        total_examples = 0

        for client, res in results:
            try:
                client_id = client.cid if hasattr(client, 'cid') else str(client)
                num_examples = getattr(res, "num_examples", res.get("num_examples", 0))
                metrics = getattr(res, "metrics", res.get("metrics", {}))

                if num_examples == 0 or not metrics:
                    continue

                accuracy = float(metrics.get("accuracy", 0.0))
                f1 = float(metrics.get("f1_score", 0.0))
                precision = float(metrics.get("precision", 0.0))
                recall = float(metrics.get("recall", 0.0))

                accuracies.append(accuracy * num_examples)
                f1_scores.append(f1 * num_examples)
                precisions.append(precision * num_examples)
                recalls.append(recall * num_examples)
                total_examples += num_examples

            except Exception as e:
                logging.error(f"[SERVER] Lỗi xử lý kết quả từ client {client_id}: {e}")
                continue

        if total_examples == 0:
            return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}

        return {
            "accuracy": sum(accuracies) / total_examples,
            "f1_score": sum(f1_scores) / total_examples,
            "precision": sum(precisions) / total_examples,
            "recall": sum(recalls) / total_examples
        }

    def compute_gradient_similarity(self, grad1, grad2):
        if not grad1 or not grad2:
            print("[SERVER] Gradient rỗng, trả về độ tương đồng 0.0")
            logging.info("[SERVER] Gradient rỗng, trả về độ tương đồng 0.0")
            return 0.0
        try:
            flat_grad1 = torch.cat([g.flatten() for g in grad1 if g is not None])
            flat_grad2 = torch.cat([g.flatten() for g in grad2 if g is not None])
            if flat_grad1.numel() == 0 or flat_grad2.numel() == 0:
                print("[SERVER] Gradient rỗng, trả về độ tương đồng 0.0")
                logging.info("[SERVER] Gradient rỗng, trả về độ tương đồng 0.0")
                return 0.0
            norm1 = torch.norm(flat_grad1)
            norm2 = torch.norm(flat_grad2)
            if norm1 == 0 or norm2 == 0:
                print("[SERVER] Norm gradient bằng 0, trả về độ tương đồng 0.0")
                logging.info("[SERVER] Norm gradient bằng 0, trả về độ tương đồng 0.0")
                return 0.0
            flat_grad1 = flat_grad1 / (norm1 + 1e-10)
            flat_grad2 = flat_grad2 / (norm2 + 1e-10)
            similarity = torch.cosine_similarity(flat_grad1, flat_grad2, dim=0).item()
            print(f"[SERVER] Độ tương đồng gradient: {similarity:.4f}")
            logging.info(f"[SERVER] Độ tương đồng gradient: {similarity:.4f}")
            return similarity
        except Exception as e:
            print(f"[SERVER] Lỗi tính độ tương đồng: {e}")
            logging.error(f"[SERVER] Lỗi tính độ tương đồng: {e}")
            return 0.0

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > MAX_ROUNDS:
            print(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng liên kết, dừng server.")
            logging.info(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng liên kết, dừng server.")
            return []
        if server_round > 1:
            self.server.load_model(server_round - 1)
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List) -> Tuple[Optional[Parameters], Dict]:
        print(f"\n[SERVER] ROUND {server_round} với {len(results)} clients")
        logging.info(f"[SERVER] ROUND {server_round} với {len(results)} clients")
        start_time = time.time()

        if not results:
            print("[SERVER] Không nhận được kết quả từ client!")
            logging.info("[SERVER] Không nhận được kết quả từ client!")
            return None, {"error": "No client results received"}

        filtered_results = []
        for client, fit_res in results:
            filtered_results.append((client, fit_res))

        if not filtered_results:
            print(f"[SERVER] Không có client nào hợp lệ để tổng hợp ở vòng {server_round}.")
            logging.info(f"[SERVER] Không có client nào hợp lệ để tổng hợp ở vòng {server_round}.")
            return None, {"error": f"No valid clients for round {server_round}"}

        models = []
        gradients = []
        accuracies = []
        client_weights = []
        client_ids = []

        for client, fit_res in filtered_results:
            try:
                client_id = client.cid if hasattr(client, 'cid') else str(client)
                print(f"[SERVER] Nhận từ client {client_id}: tensors={len(fit_res.parameters.tensors)}")
                logging.info(f"[SERVER] Nhận từ client {client_id}: tensors={len(fit_res.parameters.tensors)}")
                tensors = parameters_to_ndarrays(fit_res.parameters)
                if len(tensors) < 2:
                    print(f"[SERVER] Thiếu tensor từ client {client_id}")
                    logging.info(f"[SERVER] Thiếu tensor từ client {client_id}")
                    continue
                
                state_dict_buf = io.BytesIO(tensors[0].tobytes())
                grad_buf = io.BytesIO(tensors[1].tobytes())
                
                state_dict = torch.load(state_dict_buf, weights_only=True)
                grad_list = torch.load(grad_buf, weights_only=True)

                model = TabTransformer(**self.server.hyperparams)
                model.load_state_dict(state_dict, strict=True)

                models.append(model)
                gradients.append(grad_list)

                metrics = fit_res.metrics if hasattr(fit_res, 'metrics') else {}
                accuracies.append(float(metrics.get("accuracy", 0.0)))
                client_ids.append(client_id)
                self.client_clusters[client_id] = metrics.get("cluster_label", 0)
            except Exception as e:
                print(f"[SERVER] Lỗi nạp client {client_id}: {e}")
                logging.error(f"[SERVER] Lỗi nạp client {client_id}: {e}")
                continue

        if not models:
            print("[SERVER] Không có mô hình nào hợp lệ.")
            logging.info("[SERVER] Không có mô hình nào hợp lệ.")
            return None, {"error": "No valid models created"}

        weights = []
        for i, grad in enumerate(gradients):
            w_perf = np.exp(accuracies[i]) / sum(np.exp(acc) for acc in accuracies) if sum(np.exp(acc) for acc in accuracies) > 0 else 1.0 / len(accuracies)
            grad_norm = sum(torch.norm(g).item() for g in grad if g is not None) if grad else 0.0
            w_conv = np.exp(-grad_norm) / sum(np.exp(-sum(torch.norm(g).item() for g in grad_list if g is not None) if grad_list else 0.0) for grad_list in gradients) if sum(np.exp(-sum(torch.norm(g).item() for g in grad_list if g is not None) if grad_list else 0.0) for grad_list in gradients) > 0 else 1.0 / len(gradients)
            w = LAMBDA_PERF * w_perf + LAMBDA_CONV * w_conv
            weights.append(w)
        print(f"[SERVER] Trọng số FedMADE: {weights}")
        logging.info(f"[SERVER] Trọng số FedMADE: {weights}")

        inliers = []
        for i, grad in enumerate(gradients):
            similarities = [self.compute_gradient_similarity(grad, grad_j) for j, grad_j in enumerate(gradients) if i != j]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity >= SIMILARITY_THRESHOLD:
                inliers.append(i)
            print(f"[SERVER] Client {client_ids[i]}: Độ tương đồng trung bình = {avg_similarity:.4f}, Là inlier: {i in inliers}")
            logging.info(f"[SERVER] Client {client_ids[i]}: Độ tương đồng trung bình = {avg_similarity:.4f}, Là inlier: {i in inliers}")

        if not inliers:
            print("[SERVER] Không có gradient nào vượt ngưỡng tương đồng! Sử dụng tất cả client để tổng hợp.")
            logging.info("[SERVER] Không có gradient nào vượt ngưỡng tương đồng! Sử dụng tất cả client để tổng hợp.")
            inliers = list(range(len(models)))

        aggregated_state_dict = {}
        total_weight = sum(weights[i] for i in inliers)
        for key in models[0].state_dict().keys():
            aggregated_state_dict[key] = torch.zeros_like(models[0].state_dict()[key])
            for i in inliers:
                aggregated_state_dict[key] += weights[i] * models[i].state_dict()[key]
            aggregated_state_dict[key] /= total_weight if total_weight > 0 else 1.0
        self.server.model.load_state_dict(aggregated_state_dict)

        metrics = self.server.evaluate(self.server.model)
        acc = metrics["accuracy"]

        if self.server.previous_accuracy is not None:
            if abs(acc - self.server.previous_accuracy) < CONVERGENCE_THRESHOLD and self.server.convergence_round is None:
                self.server.convergence_round = server_round
                print(f"[SERVER] Đã hội tụ tại vòng {server_round} với chênh lệch accuracy: {abs(acc - self.server.previous_accuracy):.6f}")
                logging.info(f"[SERVER] Đã hội tụ tại vòng {server_round} với chênh lệch accuracy: {abs(acc - self.server.previous_accuracy):.6f}")
        self.server.previous_accuracy = acc

        round_time = time.time() - start_time
        self.server.round_times.append(round_time)
        print(f"[SERVER] Thời gian vòng {server_round}: {round_time:.2f} giây")
        logging.info(f"[SERVER] Thời gian vòng {server_round}: {round_time:.2f} giây")

        try:
            round_model_path = os.path.join(MODEL_DIR, f"aggregated_model_round_{server_round}.pt")
            torch.save(self.server.model.state_dict(), round_model_path)
            print(f"[SERVER] Đã lưu mô hình tổng hợp vào {round_model_path}")
            logging.info(f"[SERVER] Đã lưu mô hình tổng hợp vào {round_model_path}")
        except Exception as e:
            print(f"[SERVER] Lỗi khi lưu mô hình tổng hợp: {e}")
            logging.error(f"[SERVER] Lỗi khi lưu mô hình tổng hợp: {e}")

        buf = io.BytesIO()
        torch.save(self.server.model.state_dict(), buf)
        buf.seek(0)
        return ndarrays_to_parameters([np.frombuffer(buf.getvalue(), dtype=np.uint8)]), metrics

def start_server():
    try:
        server = FLServerTabTransformer()
        strategy = FedMADEStrategy(server)
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
            strategy=strategy,
        )
        try:
            torch.save(server.model.state_dict(), AGGREGATED_MODEL_PATH)
            print(f"[SERVER] Đã lưu mô hình tổng hợp cuối cùng vào {AGGREGATED_MODEL_PATH}")
            print(f"[SERVER] Thời gian hội tụ: {sum(server.round_times):.2f} giây")
            logging.info(f"[SERVER] Đã lưu mô hình tổng hợp cuối cùng vào {AGGREGATED_MODEL_PATH}")
            logging.info(f"[SERVER] Thời gian hội tụ: {sum(server.round_times):.2f} giây")
            if server.convergence_round:
                print(f"[SERVER] Mô hình hội tụ tại vòng {server.convergence_round}")
                logging.info(f"[SERVER] Mô hình hội tụ tại vòng {server.convergence_round}")
        except Exception as e:
            print(f"[SERVER] Lỗi khi lưu mô hình tổng hợp cuối cùng: {e}")
            logging.error(f"[SERVER] Lỗi khi lưu mô hình tổng hợp cuối cùng: {e}")
        print("[SERVER] Server đã dừng.")
        logging.info("[SERVER] Server đã dừng.")
    except Exception as e:
        print(f"[SERVER] Lỗi khởi động server: {e}")
        logging.error(f"[SERVER] Lỗi khởi động server: {e}")

if __name__ == "__main__":
    start_server()