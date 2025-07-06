import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.cluster import KMeans

def load_and_process_data(file_path: str):
    print(f"[DATA] Đọc dữ liệu: {file_path}")
    sample = pd.read_csv(file_path, nrows=500)
    dtypes = {col: "str" if sample[col].dtype == "object" else "float64" for col in sample.columns}

    df = dd.read_csv(file_path, dtype=dtypes).compute()
    print(f"[DATA] Tổng số dòng: {len(df)}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"[DATA] Sau khi loại NaN: {len(df)}")

    large_value_cols = ['Rate', 'Tot sum', 'Max', 'AVG', 'Std', 'Tot size', 'Variance']
    for col in large_value_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
            print(f"[DATA] Áp dụng log-transform cho cột {col}")

    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        print(f"[DATA] Cột {col}: min={min_val:.4f}, max={max_val:.4f}")

    df["Label"] = df["Label"].str.lower().str.strip()

    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    num_classes = len(le.classes_)
    print(f"[DATA] Số lớp: {num_classes}, Phân phối: {np.bincount(y)}")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if "Label" in categorical_cols:
        categorical_cols.remove("Label")
    print(f"[DATA] Cột phân loại: {categorical_cols}")

    feature_cols = [c for c in df.columns if c not in ["Label", "Label_enc"]]
    X = df[feature_cols]
    print(f"[DATA] Đặc trưng: {feature_cols}")
    print(f"[DATA] Dữ liệu: {len(X)} mẫu")

    if len(X) > 0:
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X[numeric_cols])
        print(f"[DATA] Phân cụm KMeans: {np.bincount(cluster_labels)}")
    else:
        cluster_labels = np.zeros(len(X), dtype=int)

    return X, y, categorical_cols, num_classes, le, cluster_labels