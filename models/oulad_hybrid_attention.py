"""Inference adapter for the Omarbekova et al. OULAD hybrid-attention model.

The adapter deliberately keeps the article-compatible protocol separate from
``models.pred_oulad``: the former predicts *success* after excluding withdrawn
enrolments, whereas the latter is the time-aware dropout-risk model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class PaperHybridAttentionNet(nn.Module):
    """BiLSTM + Bahdanau attention + static features, identical to the baseline."""
    def __init__(self, static_dim: int, seq_len: int = 40):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.W_a = nn.Linear(128, 64)
        self.u_a = nn.Parameter(torch.randn(64) * 0.1)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 128), nn.BatchNorm1d(128),
                                        nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU())
        self.fc_final = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        temporal, _ = self.lstm(dynamic.unsqueeze(-1))
        attention = torch.softmax((torch.tanh(self.W_a(temporal)) * self.u_a).sum(-1, keepdim=True), dim=1)
        fused = torch.cat((torch.sum(attention * temporal, dim=1), self.static_net(static)), dim=1)
        return self.fc_final(fused)

    def attention_weights(self, dynamic: torch.Tensor) -> torch.Tensor:
        temporal, _ = self.lstm(dynamic.unsqueeze(-1))
        return torch.softmax((torch.tanh(self.W_a(temporal)) * self.u_a).sum(-1), dim=1)


class HybridAttentionRiskPredictor:
    """Loads a reproducible inference bundle and returns success and risk scores."""
    protocol = "article_compatible_random_split_nonwithdrawn"

    def __init__(self, bundle_path: str | Path, device: str = "cpu"):
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.static_mean = np.asarray(bundle["static_mean"], dtype=np.float32)
        self.static_scale = np.asarray(bundle["static_scale"], dtype=np.float32)
        self.dynamic_center = np.asarray(bundle["dynamic_center"], dtype=np.float32)
        self.dynamic_scale = np.asarray(bundle["dynamic_scale"], dtype=np.float32)
        self.model = PaperHybridAttentionNet(int(bundle["static_dim"]), int(bundle["seq_len"])).to(self.device)
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()

    def predict(self, static_raw: np.ndarray, weekly_clicks_raw: np.ndarray) -> dict:
        """Return P(success), P(risk), and per-week attention for one or many rows."""
        static = (np.atleast_2d(static_raw).astype(np.float32) - self.static_mean) / self.static_scale
        weekly_clicks = np.atleast_2d(weekly_clicks_raw).astype(np.float32)
        if weekly_clicks.shape[1] != self.dynamic_center.shape[0]:
            raise ValueError(f"Expected {self.dynamic_center.shape[0]} weekly features, got {weekly_clicks.shape[1]}")
        dynamic = (weekly_clicks - self.dynamic_center) / self.dynamic_scale
        with torch.no_grad():
            s = torch.as_tensor(static, device=self.device)
            d = torch.as_tensor(dynamic, device=self.device)
            success = torch.softmax(self.model(s, d), dim=1)[:, 1].cpu().numpy()
            attention = self.model.attention_weights(d).cpu().numpy()
        return {"success_probability": success, "dropout_probability": 1.0 - success,
                "attention_by_week": attention, "risk_protocol": self.protocol}
