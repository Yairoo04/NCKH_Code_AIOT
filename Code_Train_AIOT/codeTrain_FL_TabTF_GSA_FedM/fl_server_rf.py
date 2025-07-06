import io
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, ndarrays_to_parameters
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_processing import load_and_process_data

DATA_PATH = "dataset/data_DDoS_1k.csv"
SIMILARITY_THRESHOLD = 0.3 
LAMBDA_PERF = 0.6  
LAMBDA_CONV = 0.4 
MAX_ROUNDS = 3  

class FLServerTabTransformer:
    def __init__(self):
        print("[SERVER] Khởi tạo server...")
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
        print(f"[SERVER] TabTransformer sẵn sàng - {self.hyperparams}")

    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            out = model(self.X_test_cat, self.X_test_cont)
            y_pred = torch.argmax(out, dim=1).numpy()
            acc = accuracy_score(self.y_test.numpy(), y_pred)
            print(f"[SERVER] Đánh giá Accuracy: {acc:.4f}")
            print(f"[SERVER] Phân phối dự đoán: {np.bincount(y_pred, minlength=self.hyperparams['dim_out']).tolist()}")
            print(f"[SERVER] Phân phối nhãn thực: {np.bincount(self.y_test.numpy(), minlength=self.hyperparams['dim_out']).tolist()}")
            return acc

class FedMADEStrategy(FedAvg):
    def __init__(self, server: FLServerTabTransformer):
        super().__init__()
        self.server = server
        self.client_clusters = {}

    def compute_gradient_similarity(self, grad1, grad2):
        if not grad1 or not grad2:
            return 0.0
        try:
            flat_grad1 = torch.cat([g.flatten() for g in grad1 if g is not None])
            flat_grad2 = torch.cat([g.flatten() for g in grad2 if g is not None])
            return torch.cosine_similarity(flat_grad1, flat_grad2, dim=0).item()
        except Exception as e:
            print(f"[SERVER] Lỗi tính độ tương đồng: {e}")
            return 0.0

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > MAX_ROUNDS:
            print(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng liên kết, dừng server.")
            return []
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        if server_round > MAX_ROUNDS:
            print(f"[SERVER] Đã đạt {MAX_ROUNDS} vòng liên kết, dừng đánh giá.")
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List) -> Tuple[Optional[Parameters], Dict]:
        print(f"\n[SERVER] ROUND {server_round} với {len(results)} clients")

        if not results:
            print("[SERVER] Không nhận được kết quả từ client!")
            return None, {"error": "No client results received"}

        filtered_results = [(client, fit_res) for client, fit_res in results 
                           if not fit_res.metrics.get("stop_after_this_round", False)]
        if not filtered_results:
            print("[SERVER] Tất cả client đã dừng, kết thúc huấn luyện.")
            return None, {"error": "All clients have stopped"}

        models = []
        gradients = []
        accuracies = []
        client_weights = []
        client_ids = []

        for client, fit_res in filtered_results:
            try:
                state_dict = torch.load(io.BytesIO(fit_res.parameters.tensors[0]), weights_only=False)
                grad_buf = io.BytesIO(fit_res.parameters.tensors[1])
                grad_list = torch.load(grad_buf, weights_only=False)
                model = TabTransformer(**self.server.hyperparams)
                model.load_state_dict(state_dict, strict=False)
                models.append(model)
                gradients.append(grad_list)
                accuracies.append(fit_res.metrics["accuracy"])
                client_ids.append(client.cid)
                self.client_clusters[client.cid] = fit_res.metrics.get("cluster_label", 0)
            except Exception as e:
                print(f"[SERVER] Lỗi nạp client {client.cid}: {e}")
                continue

        if not models:
            print("[SERVER] Không có mô hình nào hợp lệ.")
            return None, {"error": "No valid models created"}

        weights = []
        for i, grad in enumerate(gradients):
            w_perf = np.exp(accuracies[i]) / sum(np.exp(acc) for acc in accuracies) 
            grad_norm = sum(torch.norm(g).item() for g in grad if g is not None) if grad else 0.0
            w_conv = np.exp(-grad_norm) / sum(np.exp(-sum(torch.norm(g).item() for g in grad_list if g is not None) if grad_list else 0.0) for grad_list in gradients) 
            w = LAMBDA_PERF * w_perf + LAMBDA_CONV * w_conv 
            weights.append(w)
        print(f"[SERVER] Trọng số FedMADE: {weights}")

        inliers = []
        for i, grad in enumerate(gradients):
            similarities = [self.compute_gradient_similarity(grad, grad_j) for j, grad_j in enumerate(gradients) if i != j]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity >= SIMILARITY_THRESHOLD:
                inliers.append(i)
            print(f"[SERVER] Client {client_ids[i]}: Độ tương đồng trung bình = {avg_similarity:.4f}, Là inlier: {i in inliers}")

        if not inliers:
            print("[SERVER] Không có gradient nào vượt ngưỡng tương đồng!")
            return None, {"error": "No inlier gradients"}

        aggregated_state_dict = {}
        total_weight = sum(weights[i] for i in inliers)
        for key in models[0].state_dict().keys():
            aggregated_state_dict[key] = torch.zeros_like(models[0].state_dict()[key])
            for i in inliers:
                aggregated_state_dict[key] += weights[i] * models[i].state_dict()[key]
            aggregated_state_dict[key] /= total_weight if total_weight > 0 else 1.0
        self.server.model.load_state_dict(aggregated_state_dict)

        acc = self.server.evaluate(self.server.model)
        print(f"[SERVER] Accuracy mô hình tổng hợp: {acc:.4f}")

        buf = io.BytesIO()
        torch.save(self.server.model.state_dict(), buf)
        buf.seek(0)
        return ndarrays_to_parameters([np.frombuffer(buf.getvalue(), dtype=np.uint8)]), {"accuracy": acc}

def start_server():
    try:
        server = FLServerTabTransformer()
        strategy = FedMADEStrategy(server)
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
            strategy=strategy,
        )
        print("[SERVER] Server đã dừng.")
    except Exception as e:
        print(f"[SERVER] Lỗi khởi động server: {e}")

if __name__ == "__main__":
    start_server()