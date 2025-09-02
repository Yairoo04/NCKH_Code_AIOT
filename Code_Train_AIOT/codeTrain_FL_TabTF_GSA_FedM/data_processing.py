import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.cluster import KMeans
import logging
import os

def load_and_process_data(file_path: str):
    logging.info(f"Starting data processing for file: {file_path}")
    print(f"[DATA] Reading data: {file_path}")
    try:
        logging.info("Reading sample data to infer column types")
        sample = pd.read_csv(file_path, nrows=500)
        dtypes = {col: "str" if sample[col].dtype == "object" else "float64" for col in sample.columns}
        logging.info(f"Inferred data types: {dtypes}")
    
        logging.info("Loading full dataset with dask")
        df = dd.read_csv(file_path, dtype=dtypes).compute()
        if df.empty:
            logging.error("Dataframe is empty after reading")
            raise ValueError("Dataframe is empty after reading.")
        logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"[DATA] Total rows: {len(df)}")

        logging.info("Replacing infinite values with NaN and dropping NaN rows")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if df.empty:
            logging.error("Dataframe is empty after dropping NaN")
            raise ValueError("Dataframe is empty after dropping NaN.")
        logging.info(f"Rows after dropping NaN: {len(df)}")
        print(f"[DATA] After dropping NaN: {len(df)}")

        large_value_cols = ['Rate', 'Tot sum', 'Max', 'AVG', 'Std', 'Tot size', 'Variance']
        logging.info(f"Applying log transformation to columns: {large_value_cols}")
        for col in large_value_cols:
            if col in df.columns:
                logging.info(f"Starting log transformation for column: {col}")
                df[col] = np.log1p(df[col].clip(lower=0))
                logging.info(f"Completed log transformation for column: {col}")
                print(f"[DATA] Applied log-transform to column: {col}")

        numeric_cols = df.select_dtypes(include=['float64']).columns
        logging.info(f"Numeric columns: {numeric_cols.tolist()}")
        for col in numeric_cols:
            min_val, max_val = df[col].min(), df[col].max()
            logging.info(f"Column {col}: min={min_val:.4f}, max={max_val:.4f}")
            print(f"[DATA] Column {col}: min={min_val:.4f}, max={max_val:.4f}")

        logging.info("Processing labels: converting to lowercase and stripping whitespace")
        df["Label"] = df["Label"].str.lower().str.strip()
        le = LabelEncoder()
        y = le.fit_transform(df["Label"])
        num_classes = len(le.classes_)
        label_counts = np.bincount(y).tolist()
        logging.info(f"Encoded labels: {num_classes} classes, distribution: {label_counts}")
        logging.info(f"Label classes: {le.classes_.tolist()}")
        print(f"[DATA] Number of classes: {num_classes}, Distribution: {np.bincount(y)}")

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if "Label" in categorical_cols:
            categorical_cols.remove("Label")
        logging.info(f"Categorical columns: {categorical_cols}")
        print(f"[DATA] Categorical columns: {categorical_cols}")

        feature_cols = [c for c in df.columns if c not in ["Label", "Label_enc"]]
        X = df[feature_cols]
        logging.info(f"Feature columns: {feature_cols}")
        logging.info(f"Feature matrix: {len(X)} samples, {len(feature_cols)} features")
        print(f"[DATA] Features: {feature_cols}")
        print(f"[DATA] Data: {len(X)} samples")

        if len(X) > 0:
            logging.info("Starting KMeans clustering with 3 clusters")
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(X[numeric_cols])
            cluster_counts = np.bincount(cluster_labels).tolist()
            logging.info(f"KMeans clustering completed. Cluster distribution: {cluster_counts}")
            print(f"[DATA] KMeans clustering: {np.bincount(cluster_labels)}")
        else:
            logging.warning("No data for clustering, assigning zero cluster labels")
            cluster_labels = np.zeros(len(X), dtype=int)

        logging.info("Data processing completed successfully")
        return X, y, categorical_cols, num_classes, le, cluster_labels
    except Exception as e:
        logging.error(f"Error during data processing: {str(e)}", exc_info=True)
        print(f"[DATA] Data processing error: {str(e)}")
        raise