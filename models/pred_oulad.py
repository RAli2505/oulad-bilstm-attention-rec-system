"""OULAD dropout prediction: multi-task LSTM (BCE classifier + Cox survival head).

Reads prepared OULAD parquet artifacts and builds a
per-student daily VLE click sequence (date < first_deadline), and trains a shared
LSTM encoder with two heads: binary dropout (ROC-AUC) and Cox proportional-hazards
risk (C-index). Standalone trainable model; outcome-weighted ranking is out of scope.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss
from evaluation_metrics import risk_metrics, intervention_metrics, deployable_intervention_metrics

logger = logging.getLogger("oulad.model")

ENROLL_KEYS = ["id_student", "code_module", "code_presentation"]
DROPOUT_RESULTS = {"Fail", "Withdrawn"}
MAX_DAYS = 61
HIDDEN, LAYERS, MLP_OUT = 128, 2, 64
DROPOUT_P = 0.3
LR, EPOCHS, PATIENCE = 1e-3, 100, 20
LAMBDA_COX = 1.0
BATCH = 256
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAT_COLS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NUM_COLS = ["num_of_prev_attempts", "studied_credits", "date_registration"]


def activity_vocab(daily: pd.DataFrame) -> list[str]:
    """Stable sorted list of VLE activity_type categories (pinned at build time)."""
    return sorted(daily["activity_type"].astype(str).unique().tolist())


class PredNet(nn.Module):
    """Shared LSTM encoder + masked temporal attention + static fusion + two heads."""

    def __init__(self, n_feat: int, n_static: int = 0):
        super().__init__()
        self.n_static = n_static
        self.lstm = nn.LSTM(n_feat, HIDDEN, LAYERS, dropout=DROPOUT_P, batch_first=True)
        self.attn = nn.Linear(HIDDEN, 1)
        self.mlp = nn.Sequential(nn.Linear(HIDDEN + n_static, MLP_OUT),
                                 nn.ReLU(), nn.Dropout(DROPOUT_P))
        self.cls = nn.Linear(MLP_OUT, 1)
        self.risk = nn.Linear(MLP_OUT, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                static: torch.Tensor | None = None):
        out, _ = self.lstm(x)                                   # [B, T, H]
        T = x.size(1)
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]  # [B, T]
        score = self.attn(out).squeeze(-1)                      # [B, T]
        score = score.masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1)                         # [B, T]
        ctx = (w.unsqueeze(-1) * out).sum(1)                    # [B, H]
        if self.n_static:
            assert static is not None, "n_static>0 requires static features"
            ctx = torch.cat([ctx, static], dim=1)               # [B, H + S]
        h = self.mlp(ctx)
        return self.cls(h).squeeze(-1), self.risk(h).squeeze(-1)  # [B], [B]


class BiLSTMAttentionNet(nn.Module):
    """Article-style Bi-LSTM + attention + static-MLP hybrid fusion.

    Dynamic daily VLE sequence -> Bi-LSTM -> Bahdanau attention; demographic
    and early-assessment features -> independent static MLP; their embeddings
    are fused only before the final multi-task risk heads.
    """

    def __init__(self, n_feat: int, n_static: int = 0):
        super().__init__()
        self.n_static = n_static
        self.lstm = nn.LSTM(n_feat, HIDDEN, LAYERS, dropout=DROPOUT_P,
                            batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(HIDDEN * 2, HIDDEN)
        self.attn_query = nn.Parameter(torch.randn(HIDDEN) * 0.1)
        self.static_mlp = nn.Sequential(nn.Linear(n_static, 128), nn.BatchNorm1d(128),
                                        nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(128, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(HIDDEN * 2 + 64, MLP_OUT),
                                 nn.ReLU(), nn.Dropout(DROPOUT_P))
        self.cls, self.risk = nn.Linear(MLP_OUT, 1), nn.Linear(MLP_OUT, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                static: torch.Tensor | None = None):
        out, _ = self.lstm(x)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        score = (torch.tanh(self.attn_proj(out)) * self.attn_query).sum(-1)
        weights = torch.softmax(score.masked_fill(~mask, float("-inf")), dim=1)
        ctx = (weights.unsqueeze(-1) * out).sum(1)
        if self.n_static:
            assert static is not None, "n_static>0 requires static features"
            ctx = torch.cat([ctx, self.static_mlp(static)], dim=1)
        h = self.mlp(ctx)
        return self.cls(h).squeeze(-1), self.risk(h).squeeze(-1)


class TemporalConvBlock(nn.Module):
    """Causal dilated 1-D convolution with a residual connection."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=2 * dilation, dilation=dilation)
        self.dropout = nn.Dropout(DROPOUT_P)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def _causal(self, y: torch.Tensor) -> torch.Tensor:
        # Conv1d padding is symmetric; remove the right side to preserve causality.
        return y[:, :, :-2 * self.dilation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._causal(self.conv1(x))
        y = self.dropout(torch.relu(self.norm1(y)))
        y = self._causal(self.conv2(y))
        y = self.dropout(torch.relu(self.norm2(y)))
        return torch.relu(x + y)


class TCNBiLSTMAttentionNet(nn.Module):
    """Causal TCN/CNN -> Bi-LSTM -> attention + static-MLP hybrid fusion.

    The TCN extracts short and medium temporal click patterns before the Bi-LSTM
    models their bidirectional context.  Padded days are masked before packing so
    the reverse LSTM cannot learn from artificial zero-padding.
    """

    def __init__(self, n_feat: int, n_static: int = 0):
        super().__init__()
        self.n_static = n_static
        self.input_proj = nn.Conv1d(n_feat, HIDDEN, kernel_size=1)
        self.tcn = nn.Sequential(TemporalConvBlock(HIDDEN, dilation=1),
                                 TemporalConvBlock(HIDDEN, dilation=2),
                                 TemporalConvBlock(HIDDEN, dilation=4))
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, LAYERS, dropout=DROPOUT_P,
                            batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(HIDDEN * 2, HIDDEN)
        self.attn_query = nn.Parameter(torch.randn(HIDDEN) * 0.1)
        self.static_mlp = nn.Sequential(nn.Linear(n_static, 128), nn.BatchNorm1d(128),
                                        nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(128, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(HIDDEN * 2 + 64, MLP_OUT),
                                 nn.ReLU(), nn.Dropout(DROPOUT_P))
        self.cls, self.risk = nn.Linear(MLP_OUT, 1), nn.Linear(MLP_OUT, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                static: torch.Tensor | None = None):
        T = x.size(1)
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        # [B, T, F] -> [B, F, T] -> causal TCN -> [B, T, H]
        z = self.tcn(self.input_proj(x.transpose(1, 2))).transpose(1, 2)
        z = z * mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                   total_length=T)
        score = (torch.tanh(self.attn_proj(out)) * self.attn_query).sum(-1)
        weights = torch.softmax(score.masked_fill(~mask, float("-inf")), dim=1)
        ctx = (weights.unsqueeze(-1) * out).sum(1)
        if self.n_static:
            assert static is not None, "n_static>0 requires static features"
            ctx = torch.cat([ctx, self.static_mlp(static)], dim=1)
        h = self.mlp(ctx)
        return self.cls(h).squeeze(-1), self.risk(h).squeeze(-1)


def cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative log Cox partial likelihood (Breslow) over the in-batch risk set.

    For each event, the risk set is all samples with time >= the event time.
    # ponytail: in-batch risk set; switch to a global risk set only if C-index is poor.
    """
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, dim=0)             # sum over time >= current
    n_events = event.sum().clamp(min=1.0)
    return -((risk - log_risk_set) * event).sum() / n_events


def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell concordance: over comparable pairs (event_i=1, time_i<time_j),
    fraction with risk_i > risk_j (ties counted 0.5). Higher risk = earlier event."""
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    ev = np.where(event == 1)[0]
    if len(ev) == 0:
        return 0.5
    tr = time[ev][:, None]
    rr = risk[ev][:, None]
    comparable = tr < time[None, :]                           # [n_event, n]
    if not comparable.any():
        return 0.5
    concordant = comparable & (rr > risk[None, :])
    ties = comparable & (rr == risk[None, :])
    return float((concordant.sum() + 0.5 * ties.sum()) / comparable.sum())


class StaticEncoder:
    """Fit-on-train one-hot (categorical) + standardize (numeric) -> dense [N, S].

    Categorical: vocab pinned at fit; unseen/NaN -> all-zero one-hot block plus a
    per-field *_missing flag. Numeric: train mean/std; NaN imputed to the train mean
    (0 after standardization), no separate missing flag. Leak-free: stats come only
    from the frame passed to fit().
    """

    def __init__(self, cat_cols=CAT_COLS, num_cols=NUM_COLS):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.vocab: dict[str, list[str]] = {}
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.n_static = 0

    def fit(self, df: pd.DataFrame) -> "StaticEncoder":
        for c in self.cat_cols:
            vals = df[c].dropna().astype(str).unique().tolist()
            self.vocab[c] = sorted(vals)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            self.mean[c] = float(col.mean())
            s = float(col.std())
            self.std[c] = s if s > 1e-8 else 1.0
        self.n_static = (sum(len(self.vocab[c]) + 1 for c in self.cat_cols)
                         + len(self.num_cols))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols: list[np.ndarray] = []
        for c in self.cat_cols:
            vocab = self.vocab[c]
            idx = {v: i for i, v in enumerate(vocab)}
            block = np.zeros((n, len(vocab)), dtype=np.float32)
            sv = df[c].to_numpy(dtype=object)
            for r in range(n):
                v = sv[r]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                j = idx.get(str(v))
                if j is not None:
                    block[r, j] = 1.0
            missing = (block.sum(axis=1) == 0).astype(np.float32)[:, None]
            cols.append(block)
            cols.append(missing)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            z = (np.nan_to_num(col, nan=self.mean[c]) - self.mean[c]) / self.std[c]
            cols.append(z.astype(np.float32)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def _stratified_split(arrays, frac: float = 0.85, seed: int = SEED):
    """Split (X, lengths, static, y, t, e) into two tuples, stratified on y (binary)."""
    X, lengths, static, y, t, e = arrays
    rng = np.random.default_rng(seed)
    fit_mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fit_mask[idx[:int(len(idx) * frac)]] = True
    pick = lambda m: (X[m], lengths[m], static[m], y[m], t[m], e[m])
    return pick(fit_mask), pick(~fit_mask)


class OULADPred:
    def __init__(self, processed_dir: str = "data/oulad",
                 data_dir: str = 'data/oulad/raw',
                 model_dir: str = "models", results_dir: str = "results/oulad",
                 window_frac: float | None = None, architecture: str = "lstm",
                 observation_days: int | None = None, static_mode: str = "full",
                 sequence_mode: str = "typed", tabular_model: str = "lightgbm",
                 experiment_tag: str = ""):
        self.processed_dir = Path(processed_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.window_frac = window_frac
        self.observation_days = observation_days
        if observation_days is not None and observation_days < 1:
            raise ValueError("observation_days must be >= 1")
        if static_mode not in {"full", "demographics"}:
            raise ValueError("static_mode must be 'full' or 'demographics'")
        if sequence_mode not in {"typed", "total"}:
            raise ValueError("sequence_mode must be 'typed' or 'total'")
        if tabular_model not in {"lightgbm", "xgboost"}:
            raise ValueError("tabular_model must be 'lightgbm' or 'xgboost'")
        self.static_mode = static_mode
        self.sequence_mode = sequence_mode
        self.tabular_model = tabular_model
        self.experiment_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_tag).strip("_")
        if architecture not in {"lstm", "bilstm_attention", "tcn_bilstm_attention", "ensemble"}:
            raise ValueError("architecture must be 'lstm', 'bilstm_attention', "
                             "'tcn_bilstm_attention', or 'ensemble'")
        self.architecture = architecture
        self.model: nn.Module | None = None
        self._acts: list[str] = []
        self.enc: StaticEncoder | None = None
        self.n_static = 0
        self._last_submitted_any = None

    def _make_model(self, n_features: int, n_static: int) -> nn.Module:
        if self.architecture == "bilstm_attention":
            return BiLSTMAttentionNet(n_features, n_static)
        if self.architecture == "tcn_bilstm_attention":
            return TCNBiLSTMAttentionNet(n_features, n_static)
        return PredNet(n_features, n_static)

    def _observation_cut(self, target: pd.DataFrame) -> pd.Series:
        """Last observable course day, shared by clicks and assessment features."""
        raw = (target["first_deadline"].astype(float) if self.window_frac is None else
               self.window_frac * target["module_presentation_length"].astype(float))
        upper = min(MAX_DAYS, self.observation_days) if self.observation_days else MAX_DAYS
        return raw.clip(lower=1, upper=upper)

    def build_sequences(self, daily: pd.DataFrame, target: pd.DataFrame,
                        keys: pd.DataFrame, acts: list[str]):
        """Return (X, lengths, y_drop, t_event, e_event) for the given enrollments.

        X: float32 [N, MAX_DAYS, F] daily clicks per activity_type, padded with zeros.
        lengths: int [N] real window length = min(ceil(first_deadline), MAX_DAYS), >=1.
        y_drop: int [N] 1 if final_result in {Fail, Withdrawn}.
        t_event/e_event: survival time (unreg day, else module length) / event (Withdrawn).
        """
        act_idx = {a: i for i, a in enumerate(acts)}
        F = len(acts)
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        enr_idx = {tuple(k): i for i, k in enumerate(keys.itertuples(index=False, name=None))}
        N = len(keys)

        tgt = target.set_index(ENROLL_KEYS)
        X = np.zeros((N, MAX_DAYS, F), dtype=np.float32)
        lengths = np.ones(N, dtype=np.int64)
        y = np.zeros(N, dtype=np.int64)
        t = np.zeros(N, dtype=np.float32)
        e = np.zeros(N, dtype=np.int64)

        # observation cutoff per enrollment: first_deadline (leak-free default),
        # or window_frac * module length (fallback B, wider but still pre-outcome).
        cut_map = self._observation_cut(tgt)

        # labels + lengths per enrollment
        for k, i in enr_idx.items():
            row = tgt.loc[k]
            cut = float(cut_map.loc[k])
            # ``cut_map`` is already capped: no post-window event may enter X.
            lengths[i] = max(1, min(int(np.ceil(cut)), MAX_DAYS))
            res = str(row["final_result"])
            y[i] = 1 if res in DROPOUT_RESULTS else 0
            unreg = row["date_unregistration"]
            if res == "Withdrawn" and pd.notna(unreg):
                e[i] = 1
                t[i] = float(unreg)
            else:
                # non-withdrawn, or Withdrawn with missing unreg date -> censored at module end
                e[i] = 0
                t[i] = float(row["module_presentation_length"])

        # clicks: filter to keys + date < cutoff, scatter-add into X
        d = daily.merge(keys, on=ENROLL_KEYS, how="inner").copy()
        d["_cut"] = d.set_index(ENROLL_KEYS).index.map(cut_map)
        # OULAD can contain pre-course dates. They are not observable after
        # course start and must never use NumPy's negative-index wraparound.
        d = d[(d["date"] >= 0) & (d["date"] < d["_cut"])]
        d["_activity"] = ("all_clicks" if self.sequence_mode == "total"
                          else d["activity_type"].astype(str))
        d = d[d["_activity"].isin(act_idx)]
        if len(d):
            ei = np.array([enr_idx[(r.id_student, r.code_module, r.code_presentation)]
                           for r in d.itertuples(index=False)])
            ri = d["date"].to_numpy().astype(int)
            ai = d["_activity"].map(act_idx).to_numpy().astype(int)
            np.add.at(X, (ei, ri, ai), d["sum_click"].to_numpy().astype(np.float32))

        return X, lengths, y, t, e

    def train(self, Xtr, len_tr, str_tr, ytr, ttr, etr,
              Xval, len_val, str_val, yval, tval, eval_, epochs: int = EPOCHS):
        torch.manual_seed(SEED)
        F = Xtr.shape[2]
        S = str_tr.shape[1]
        net = self._make_model(F, S).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        bce = nn.BCEWithLogitsLoss()   # no pos_weight: better-calibrated probabilities (brier)

        def to_dev(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device=DEVICE)

        Xtr_t, len_tr_t = to_dev(Xtr, torch.float32), to_dev(len_tr, torch.long)
        str_tr_t = to_dev(str_tr, torch.float32)
        ytr_t, ttr_t, etr_t = (to_dev(ytr, torch.float32), to_dev(ttr, torch.float32),
                               to_dev(etr, torch.float32))
        Xv, lv = to_dev(Xval, torch.float32), to_dev(len_val, torch.long)
        str_v = to_dev(str_val, torch.float32)
        yv, tv, ev = (to_dev(yval, torch.float32), to_dev(tval, torch.float32),
                      to_dev(eval_, torch.float32))

        n = len(Xtr)
        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(epochs):
            net.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, BATCH):
                idx = perm[s:s + BATCH]
                logit, risk = net(Xtr_t[idx], len_tr_t[idx], str_tr_t[idx])
                loss = bce(logit, ytr_t[idx]) + LAMBDA_COX * cox_loss(
                    risk, ttr_t[idx], etr_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl, vr = net(Xv, lv, str_v)
                vloss = (bce(vl, yv) + LAMBDA_COX * cox_loss(vr, tv, ev)).item()
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = vloss, {k: v.detach().cpu().clone()
                                                    for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    logger.info("pred early stop epoch %d", epoch)
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net
        return net

    def evaluate(self, Xte, len_te, str_te, yte, tte, ete, intervention_eligible=None, probability=None) -> dict:
        net = self.model
        net.eval()
        with torch.no_grad():
            logit, risk = net(torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE),
                              torch.as_tensor(len_te, dtype=torch.long, device=DEVICE),
                              torch.as_tensor(str_te, dtype=torch.float32, device=DEVICE))
            prob = torch.sigmoid(logit).cpu().numpy() if probability is None else np.asarray(probability)
            risk = risk.cpu().numpy()
        try:
            auc = float(roc_auc_score(yte, prob))
        except ValueError:
            auc = 0.5
        brier = float(brier_score_loss(yte, prob)) if len(set(yte.tolist())) > 1 else float(
            np.mean((prob - yte) ** 2))
        metrics = {"dropout_auc": auc, "c_index": c_index(risk, tte, ete), "brier": brier}
        metrics.update(risk_metrics(yte, prob))
        metrics.update(intervention_metrics(yte, prob) if intervention_eligible is None
                       else deployable_intervention_metrics(yte, prob, intervention_eligible))
        return metrics

    def evaluate_probability(self, yte, prob, risk, tte, ete, intervention_eligible=None) -> dict:
        """Evaluate externally produced probabilities, e.g. an ensemble."""
        prob = np.asarray(prob, dtype=np.float64)
        risk = np.asarray(risk, dtype=np.float64)
        try:
            auc = float(roc_auc_score(yte, prob))
        except ValueError:
            auc = 0.5
        brier = float(brier_score_loss(yte, prob)) if len(set(yte.tolist())) > 1 else float(
            np.mean((prob - yte) ** 2))
        metrics = {"dropout_auc": auc, "c_index": c_index(risk, tte, ete), "brier": brier}
        metrics.update(risk_metrics(yte, prob))
        metrics.update(intervention_metrics(yte, prob) if intervention_eligible is None
                       else deployable_intervention_metrics(yte, prob, intervention_eligible))
        return metrics

    def predict_probability(self, X, lengths, static):
        self.model.eval()
        with torch.no_grad():
            logit, _ = self.model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE),
                                   torch.as_tensor(lengths, dtype=torch.long, device=DEVICE),
                                   torch.as_tensor(static, dtype=torch.float32, device=DEVICE))
        return torch.sigmoid(logit).cpu().numpy()

    def predict_probability_and_risk(self, X, lengths, static):
        self.model.eval()
        with torch.no_grad():
            logit, risk = self.model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE),
                                     torch.as_tensor(lengths, dtype=torch.long, device=DEVICE),
                                     torch.as_tensor(static, dtype=torch.float32, device=DEVICE))
        return torch.sigmoid(logit).cpu().numpy(), risk.cpu().numpy()

    @staticmethod
    def tabular_features(X: np.ndarray, lengths: np.ndarray, static: np.ndarray) -> np.ndarray:
        """Leak-free behavioral feature bank for GBDT/XGBoost.

        Every aggregate is restricted to each student's observed prefix.  Besides
        volume, this captures recency, inactivity, activity diversity, rolling
        trends and deviation from the student's own preceding activity level.
        """
        X = np.asarray(X, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        static = np.asarray(static, dtype=np.float32)
        n, _, f = X.shape
        totals = X.sum(axis=1)
        active = (X.sum(axis=2) > 0).astype(np.float32)
        active_days = active.sum(axis=1, keepdims=True)
        total_clicks = totals.sum(axis=1, keepdims=True)
        mean_clicks = total_clicks / np.maximum(lengths[:, None], 1)
        last7 = np.zeros((n, f), dtype=np.float32)
        prev7 = np.zeros((n, f), dtype=np.float32)
        last14 = np.zeros((n, f), dtype=np.float32)
        active_last7 = np.zeros((n, 1), dtype=np.float32)
        active_prev7 = np.zeros((n, 1), dtype=np.float32)
        days_since_last = np.zeros((n, 1), dtype=np.float32)
        max_gap = np.zeros((n, 1), dtype=np.float32)
        activity_diversity = (totals > 0).sum(axis=1, keepdims=True).astype(np.float32)
        weekly = np.zeros((n, 4), dtype=np.float32)
        for i, l in enumerate(lengths):
            l = int(max(1, min(l, X.shape[1])))
            seq = X[i, :l]
            day_clicks = seq.sum(axis=1)
            day_active = day_clicks > 0
            last7[i] = seq[max(0, l - 7):l].sum(axis=0)
            prev7[i] = seq[max(0, l - 14):max(0, l - 7)].sum(axis=0)
            last14[i] = seq[max(0, l - 14):l].sum(axis=0)
            active_last7[i, 0] = day_active[max(0, l - 7):l].sum()
            active_prev7[i, 0] = day_active[max(0, l - 14):max(0, l - 7)].sum()
            active_idx = np.flatnonzero(day_active)
            days_since_last[i, 0] = l if len(active_idx) == 0 else l - 1 - active_idx[-1]
            gaps = np.diff(np.r_[-1, active_idx, l]) - 1
            max_gap[i, 0] = gaps.max() if len(gaps) else l
            for w in range(4):
                start, end = int(w * l / 4), int((w + 1) * l / 4)
                weekly[i, w] = day_clicks[start:end].sum()
        trend7 = last7.sum(axis=1, keepdims=True) - prev7.sum(axis=1, keepdims=True)
        recent_share = last14.sum(axis=1, keepdims=True) / np.maximum(total_clicks, 1.0)
        recent_rate = last7.sum(axis=1, keepdims=True) / np.minimum(lengths[:, None], 7)
        prior_days = np.maximum(lengths[:, None] - 7, 1)
        prior_rate = (total_clicks - last7.sum(axis=1, keepdims=True)) / prior_days
        personal_activity_delta = recent_rate - prior_rate
        personal_activity_ratio = recent_rate / np.maximum(prior_rate, 1.0)
        return np.concatenate([
            static, totals, last7, prev7, last14,
            active_days, total_clicks, mean_clicks, trend7, recent_share,
            active_last7, active_prev7, days_since_last, max_gap, activity_diversity,
            weekly, personal_activity_delta, personal_activity_ratio,
        ], axis=1).astype(np.float32)

    @staticmethod
    def train_tabular(Xtr, len_tr, str_tr, ytr, model_kind: str = "lightgbm"):
        Xt = OULADPred.tabular_features(Xtr, len_tr, str_tr)
        if model_kind == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", n_estimators=700,
                max_depth=5, learning_rate=0.025, min_child_weight=4,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0,
                random_state=SEED, n_jobs=-1,
            )
            model.fit(Xt, ytr)
            return model
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.04,
                l2_regularization=0.2,
                random_state=SEED,
            )
        model.fit(Xt, ytr)
        return model

    @staticmethod
    def predict_tabular(model, X, lengths, static) -> np.ndarray:
        Xt = OULADPred.tabular_features(X, lengths, static)
        return model.predict_proba(Xt)[:, 1]

    @staticmethod
    def fit_weighted_ensemble(y, p_lstm, p_bilstm, p_gbdt) -> dict:
        preds = [np.asarray(p_lstm), np.asarray(p_bilstm), np.asarray(p_gbdt)]
        best = {"weights": [1.0, 0.0, 0.0], "auc": -1.0, "brier": float("inf")}
        grid = np.linspace(0.0, 1.0, 21)
        for w1 in grid:
            for w2 in grid:
                if w1 + w2 > 1.0:
                    continue
                w3 = 1.0 - w1 - w2
                p = w1 * preds[0] + w2 * preds[1] + w3 * preds[2]
                try:
                    auc = float(roc_auc_score(y, p))
                except ValueError:
                    auc = 0.5
                brier = float(brier_score_loss(y, p))
                if (auc > best["auc"] + 1e-8 or
                        (abs(auc - best["auc"]) <= 1e-8 and brier < best["brier"])):
                    best = {"weights": [float(w1), float(w2), float(w3)],
                            "auc": auc, "brier": brier}
        return best

    @staticmethod
    def apply_weights(weights: list[float], p_lstm, p_bilstm, p_gbdt) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        return w[0] * np.asarray(p_lstm) + w[1] * np.asarray(p_bilstm) + w[2] * np.asarray(p_gbdt)

    def _early_assessment_features(self, target: pd.DataFrame,
                                   assess: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        """Assessment aggregates available before the same observation cutoff as VLE.

        Returns a frame aligned to `keys` with columns:
        submitted_any, banked_any, n_early_submitted, mean_early_score.
        Enrollments with no early submission get 0 / 0 / 0 / NaN (score missing).
        """
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        fd = target[ENROLL_KEYS].copy()
        fd["_cut"] = self._observation_cut(target).to_numpy()
        a = assess.merge(fd, on=ENROLL_KEYS, how="inner")
        a = a[a["date_submitted"] < a["_cut"]]
        if len(a):
            g = a.groupby(ENROLL_KEYS)
            agg = pd.DataFrame({
                "submitted_any": g.size().clip(upper=1),
                "n_early_submitted": g.size(),
                "mean_early_score": g["score"].mean(),
                "banked_any": g["is_banked"].max().clip(upper=1),
            }).reset_index()
        else:
            agg = pd.DataFrame(columns=ENROLL_KEYS +
                               ["submitted_any", "n_early_submitted",
                                "mean_early_score", "banked_any"])
        out = keys.merge(agg, on=ENROLL_KEYS, how="left")
        out["submitted_any"] = out["submitted_any"].fillna(0).astype(np.float32)
        out["banked_any"] = out["banked_any"].fillna(0).astype(np.float32)
        out["n_early_submitted"] = out["n_early_submitted"].fillna(0).astype(np.float32)
        # mean_early_score left as NaN when no early submission -> encoder imputes
        return out

    def build_static(self, target: pd.DataFrame, assess: pd.DataFrame,
                     keys: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Dense [N, S] static matrix for enrollments `keys`, aligned to keys order.

        fit=True pins the StaticEncoder vocab/stats on these rows (train pool);
        fit=False reuses the already-fit encoder (test). Leak-free: assessment
        features use only date_submitted < first_deadline.
        """
        keys = keys[ENROLL_KEYS].drop_duplicates().reset_index(drop=True)
        demo = keys.merge(target[ENROLL_KEYS + CAT_COLS + NUM_COLS],
                          on=ENROLL_KEYS, how="left")
        if self.static_mode == "full":
            ea = self._early_assessment_features(target, assess, keys)
            df = demo.merge(ea, on=ENROLL_KEYS, how="left")
            self._last_submitted_any = df["submitted_any"].to_numpy(dtype=np.float32)
            # Demographics plus assessment aggregates and submission flags.
            num_cols = NUM_COLS + ["n_early_submitted", "mean_early_score"]
            flags = df[["submitted_any", "banked_any"]].to_numpy(dtype=np.float32)
        else:
            # Ablation: demographics/static registration context only.
            df = demo
            self._last_submitted_any = np.zeros(len(df), dtype=np.float32)
            num_cols = NUM_COLS
            flags = np.zeros((len(df), 0), dtype=np.float32)
        if fit or self.enc is None:
            self.enc = StaticEncoder(cat_cols=CAT_COLS, num_cols=num_cols)
            self.enc.fit(df)
        encoded = self.enc.transform(df)                      # [N, enc.n_static]
        out = np.concatenate([encoded, flags], axis=1).astype(np.float32)
        self.n_static = out.shape[1]
        return out

    def _load_assessments(self) -> pd.DataFrame:
        a = pd.read_csv(self.data_dir / "studentAssessment.csv")
        m = pd.read_csv(self.data_dir / "assessments.csv")
        return a.merge(m[["id_assessment", "code_module", "code_presentation"]],
                       on="id_assessment", how="left")

    def _load_keys(self, name: str) -> pd.DataFrame:
        df = pd.read_parquet(self.processed_dir / "splits" / f"{name}.parquet")
        return df[ENROLL_KEYS]

    def build(self, epochs: int = EPOCHS) -> dict:
        daily = pd.read_parquet(self.processed_dir / "processed" / "daily_vle.parquet")
        target = pd.read_parquet(self.processed_dir / "processed" / "student_target.parquet")
        assess = self._load_assessments()
        self._acts = ["all_clicks"] if self.sequence_mode == "total" else activity_vocab(daily)

        tr_keys = self._load_keys("train")
        va_keys = self._load_keys("val")
        te_keys = self._load_keys("test")

        # fold A train+val into one 2013 pool; carve a stratified 15% internal
        # holdout purely for early-stop monitoring (test stays untouched).
        pool_keys = pd.concat([tr_keys, va_keys], ignore_index=True)
        Xp, lp, yp, tp, ep = self.build_sequences(daily, target, pool_keys, self._acts)
        Sp = self.build_static(target, assess, pool_keys, fit=True)   # fit encoder on pool
        Xte, lte, yte, tte, ete = self.build_sequences(daily, target, te_keys, self._acts)
        Ste = self.build_static(target, assess, te_keys, fit=False)   # reuse encoder

        fit, mon = _stratified_split((Xp, lp, Sp, yp, tp, ep), frac=0.85, seed=SEED)
        test_meta = te_keys.merge(target[ENROLL_KEYS + ["final_result", "date_unregistration",
                                                         "module_presentation_length"]], on=ENROLL_KEYS, how="left")
        cutoff = (test_meta["module_presentation_length"] * self.window_frac
                  if self.window_frac is not None else test_meta["module_presentation_length"])
        # A withdrawal recorded before the observation cutoff cannot be acted on.
        pre_cutoff = ((test_meta["final_result"] == "Withdrawn") &
                      test_meta["date_unregistration"].notna() &
                      (test_meta["date_unregistration"] < cutoff)).to_numpy()

        if self.architecture == "ensemble":
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression

            original_arch = self.architecture

            self.architecture = "lstm"
            lstm = self.train(*fit, *mon, epochs=epochs)
            p_lstm_mon, r_lstm_mon = self.predict_probability_and_risk(mon[0], mon[1], mon[2])
            p_lstm_test, r_lstm_test = self.predict_probability_and_risk(Xte, lte, Ste)
            lstm_state = {k: v.detach().cpu().clone() for k, v in lstm.state_dict().items()}

            self.architecture = "bilstm_attention"
            bilstm = self.train(*fit, *mon, epochs=epochs)
            p_bilstm_mon, r_bilstm_mon = self.predict_probability_and_risk(mon[0], mon[1], mon[2])
            p_bilstm_test, r_bilstm_test = self.predict_probability_and_risk(Xte, lte, Ste)
            bilstm_state = {k: v.detach().cpu().clone() for k, v in bilstm.state_dict().items()}

            gbdt = self.train_tabular(fit[0], fit[1], fit[2], fit[3], self.tabular_model)
            p_gbdt_mon = self.predict_tabular(gbdt, mon[0], mon[1], mon[2])
            p_gbdt_test = self.predict_tabular(gbdt, Xte, lte, Ste)

            ens = self.fit_weighted_ensemble(mon[3], p_lstm_mon, p_bilstm_mon, p_gbdt_mon)
            p_mon_raw = self.apply_weights(ens["weights"], p_lstm_mon, p_bilstm_mon, p_gbdt_mon)
            p_test_raw = self.apply_weights(ens["weights"], p_lstm_test, p_bilstm_test, p_gbdt_test)

            platt = LogisticRegression(C=1e6).fit(p_mon_raw.reshape(-1, 1), mon[3])
            p_platt_mon = platt.predict_proba(p_mon_raw.reshape(-1, 1))[:, 1]
            p_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_mon_raw, mon[3])
            p_iso_mon = iso.predict(p_mon_raw)
            p_iso = iso.predict(p_test_raw)
            mon_candidates = {"raw": p_mon_raw, "platt": p_platt_mon, "isotonic": p_iso_mon}
            test_candidates = {"raw": p_test_raw, "platt": p_platt, "isotonic": p_iso}
            chosen = min(mon_candidates, key=lambda n: brier_score_loss(mon[3], mon_candidates[n]))

            w = np.asarray(ens["weights"], dtype=np.float64)
            deep_risk = w[0] * r_lstm_test + w[1] * r_bilstm_test
            risk_for_cindex = deep_risk + w[2] * p_gbdt_test
            metrics = self.evaluate_probability(yte, test_candidates[chosen], risk_for_cindex,
                                                tte, ete, intervention_eligible=~pre_cutoff)
            metrics["calibration_method"] = chosen
            metrics["calibration_brier_monitor"] = {
                n: float(brier_score_loss(mon[3], p)) for n, p in mon_candidates.items()
            }
            metrics["calibration_brier"] = {
                n: float(brier_score_loss(yte, p)) for n, p in test_candidates.items()
            }
            metrics.update({
                "architecture": original_arch,
                "observation_days": self.observation_days,
                "static_mode": self.static_mode,
                "sequence_mode": self.sequence_mode,
                "tabular_model": self.tabular_model,
                "seed": SEED,
                "ensemble_components": ["lstm", "bilstm_attention", type(gbdt).__name__],
                "ensemble_weights": {
                    "lstm": ens["weights"][0],
                    "bilstm_attention": ens["weights"][1],
                    "gbdt": ens["weights"][2],
                },
                "ensemble_monitor_auc": ens["auc"],
                "ensemble_monitor_brier": ens["brier"],
                "n_train": int(len(Xp)),
                "n_test": int(len(Xte)),
                "n_pos": int(yp.sum()),
                "n_static": int(self.n_static),
            })

            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{self.experiment_tag}" if self.experiment_tag else ""
            model_path = self.model_dir / f"pred_oulad_ensemble{tag}.pt"
            metrics_path = self.results_dir / f"pred_metrics_ensemble{tag}.json"
            torch.save({
                "lstm_state_dict": lstm_state,
                "bilstm_attention_state_dict": bilstm_state,
                "weights": ens["weights"],
                "calibration_method": chosen,
                "architecture": original_arch,
                "max_days": MAX_DAYS,
                "activity_types": self._acts,
            }, model_path)
            with (self.model_dir / "pred_oulad_ensemble_gbdt.pkl").open("wb") as f:
                pickle.dump(gbdt, f)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            self.architecture = original_arch
            logger.info("pred(ensemble) dropout_auc=%.4f c_index=%.4f brier=%.4f "
                        "weights=%s (Ntr=%d Nte=%d pos=%d S=%d)",
                        metrics["dropout_auc"], metrics["c_index"], metrics["brier"],
                        metrics["ensemble_weights"], metrics["n_train"], metrics["n_test"],
                        metrics["n_pos"], metrics["n_static"])
            return {"model_path": str(model_path), "metrics_path": str(metrics_path)}

        self.train(*fit, *mon, epochs=epochs)
        # Calibrators see only the internal 2013 monitor split; test-2014 stays untouched.
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        p_mon = self.predict_probability(mon[0], mon[1], mon[2])
        p_test_raw = self.predict_probability(Xte, lte, Ste)
        platt = LogisticRegression(C=1e6).fit(p_mon.reshape(-1, 1), mon[3])
        p_platt_mon = platt.predict_proba(p_mon.reshape(-1, 1))[:, 1]
        p_platt = platt.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_mon, mon[3])
        p_iso_mon = iso.predict(p_mon)
        p_iso = iso.predict(p_test_raw)
        mon_candidates = {"raw": p_mon, "platt": p_platt_mon, "isotonic": p_iso_mon}
        candidates = {"raw": p_test_raw, "platt": p_platt, "isotonic": p_iso}
        # Select strictly on monitor calibration loss; test-2014 stays untouched.
        chosen = min(mon_candidates, key=lambda n: brier_score_loss(mon[3], mon_candidates[n]))
        metrics = self.evaluate(Xte, lte, Ste, yte, tte, ete, intervention_eligible=~pre_cutoff,
                                probability=candidates[chosen])
        metrics["calibration_method"] = chosen
        metrics["calibration_brier_monitor"] = {
            n: float(brier_score_loss(mon[3], p)) for n, p in mon_candidates.items()
        }
        metrics["calibration_brier"] = {n: float(brier_score_loss(yte, p)) for n, p in candidates.items()}
        metrics.update({
            "architecture": self.architecture,
            "observation_days": self.observation_days,
            "static_mode": self.static_mode,
            "sequence_mode": self.sequence_mode,
            "seed": SEED,
            "n_train": int(len(Xp)),
            "n_test": int(len(Xte)),
            "n_pos": int(yp.sum()),
            "n_static": int(self.n_static),
        })

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.architecture + (f"_{self.experiment_tag}" if self.experiment_tag else "")
        model_path = self.model_dir / f"pred_oulad_{suffix}.pt"
        metrics_path = self.results_dir / f"pred_metrics_{suffix}.json"
        torch.save(self.model.state_dict(), model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info("pred(fusion) dropout_auc=%.4f c_index=%.4f brier=%.4f "
                    "(Ntr=%d Nte=%d pos=%d S=%d)",
                    metrics["dropout_auc"], metrics["c_index"], metrics["brier"],
                    metrics["n_train"], metrics["n_test"], metrics["n_pos"],
                    metrics["n_static"])
        return {"model_path": str(model_path), "metrics_path": str(metrics_path)}


def main() -> None:
    global MAX_DAYS, SEED
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train OULAD dropout predictor (static fusion)")
    ap.add_argument("--processed-dir", default="data/oulad")
    ap.add_argument('--data-dir', default='data/oulad/raw')
    ap.add_argument("--window-frac", type=float, default=None,
                    help="widen click window to date < frac*module_length (fallback B)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="raise sequence length cap (default 61); pairs with --window-frac")
    ap.add_argument("--observation-days", type=int, default=None,
                    help="fixed number of observable course days; caps all features leak-free")
    ap.add_argument("--static-mode", choices=["full", "demographics"], default="full",
                    help="full includes early assessments; demographics is an ablation")
    ap.add_argument("--sequence-mode", choices=["typed", "total"], default="typed",
                    help="typed VLE activities or total daily clicks only (ablation)")
    ap.add_argument("--tabular-model", choices=["lightgbm", "xgboost"], default="lightgbm",
                    help="tabular base learner used only by architecture=ensemble")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="", help="suffix for model and metrics filenames")
    ap.add_argument("--architecture",
                    choices=["lstm", "bilstm_attention", "tcn_bilstm_attention", "ensemble"],
                    default="lstm")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    SEED = args.seed
    if args.max_days:
        MAX_DAYS = args.max_days  # ponytail: module-global override, fine for a CLI knob
    if args.build:
        OULADPred(processed_dir=args.processed_dir, data_dir=args.data_dir,
                  window_frac=args.window_frac, architecture=args.architecture,
                  observation_days=args.observation_days, static_mode=args.static_mode,
                  sequence_mode=args.sequence_mode, tabular_model=args.tabular_model,
                  experiment_tag=args.tag).build()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
