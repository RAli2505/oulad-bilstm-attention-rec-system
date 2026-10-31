"""Export the trained paper baseline with its preprocessing statistics.

Run after ``python main.py``.  It does not retrain and reproduces the exact
seeded 64/16/20 split to reconstruct the two fitted scalers.
"""
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from config import MODEL_PATH, SEED
from data import get_data

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "models", "paper_hybrid_oulad.pt")

X_s, X_d, y, _, _ = get_data()
X_s_tv, _, X_d_tv, _, y_tv, _ = train_test_split(X_s, X_d, y, test_size=.20, stratify=y, random_state=SEED)
X_s_tr, _, X_d_tr, _, _, _ = train_test_split(X_s_tv, X_d_tv, y_tv, test_size=.20, stratify=y_tv, random_state=SEED)
scaler_s, scaler_d = StandardScaler().fit(X_s_tr), RobustScaler().fit(X_d_tr)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"model_state": torch.load(MODEL_PATH, map_location="cpu", weights_only=True),
            "static_dim": X_s.shape[1], "seq_len": X_d.shape[1],
            "static_mean": scaler_s.mean_, "static_scale": scaler_s.scale_,
            "dynamic_center": scaler_d.center_, "dynamic_scale": scaler_d.scale_,
            "protocol": "article_compatible_random_split_nonwithdrawn"}, OUT)
print(f"Saved inference bundle: {OUT}")
