"""Pipeline orchestration for Frontiers AI 9:1811886.

Reproduces:
  Tables 1, 2, 3, 4, 5
  Figures 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from config import (device, SEED,
                    BATCH_TRAIN, BATCH_TEST, MODEL_PATH,
                    RUN_BASELINE_ML, RUN_KFOLD, RUN_ABLATION, RUN_LATENT_VIZ)
from data import download_oulad, get_data
from models import StaticBaseline, DynamicBaseline, HybridModel
from training import train_and_evaluate
from tables import (table1_comparative, table2_baselines,
                    table3_kfold, table4_balancing, table5_ablation)
from figures import (figure_class_distribution,
                     figure_engagement_daily, figure_engagement_gap,
                     figure_engagement_gap_smoothed,
                     figure_learning_curves, figure_val_f1,
                     figure_cm_roc, figure_tsne, figure_pca, figure_umap,
                     figure_attention)

print(f"Device: {device}"
      f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")


# 1. Data
download_oulad()
X_s_raw, X_d_raw, y, uid_target, vle_daily = get_data()


# Figure 1
figure_class_distribution(y)

# Figures 3, 5, 6
figure_engagement_daily(uid_target, vle_daily)
figure_engagement_gap(uid_target, vle_daily)
figure_engagement_gap_smoothed(uid_target, vle_daily)


# 2. Stratified 64/16/20 split (train/val/test)
X_s_tv, X_s_te, X_d_tv, X_d_te, y_tv, y_te = train_test_split(
    X_s_raw, X_d_raw, y, test_size=0.20, stratify=y, random_state=SEED)
X_s_tr, X_s_va, X_d_tr, X_d_va, y_tr, y_va = train_test_split(
    X_s_tv, X_d_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=SEED)
print(f"Split sizes: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

scaler_s = StandardScaler()
X_s_tr = scaler_s.fit_transform(X_s_tr)
X_s_va = scaler_s.transform(X_s_va)
X_s_te = scaler_s.transform(X_s_te)

scaler_d = RobustScaler()
X_d_tr = scaler_d.fit_transform(X_d_tr)
X_d_va = scaler_d.transform(X_d_va)
X_d_te = scaler_d.transform(X_d_te)

splits = {
    'X_s_tr': X_s_tr, 'X_s_va': X_s_va, 'X_s_te': X_s_te,
    'X_d_tr': X_d_tr, 'X_d_va': X_d_va, 'X_d_te': X_d_te,
    'y_tr':   y_tr,   'y_va':   y_va,   'y_te':   y_te,
}


def _loader(X_s, X_d, y_, batch, shuffle=False):
    ds = TensorDataset(torch.FloatTensor(X_s),
                       torch.FloatTensor(X_d),
                       torch.LongTensor(y_))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


train_loader = _loader(X_s_tr, X_d_tr, y_tr, BATCH_TRAIN, shuffle=True)
val_loader   = _loader(X_s_va, X_d_va, y_va, BATCH_TEST)
test_loader  = _loader(X_s_te, X_d_te, y_te, BATCH_TEST)

static_dim = X_s_tr.shape[1]
seq_len    = X_d_tr.shape[1]


# 3. Train 4 deep models — up to 100 epochs, early stopping on val_loss
# with patience=7 (article setup).
metrics_static = train_and_evaluate(
    'Static (Demo)',     StaticBaseline(static_dim),
    train_loader, val_loader, test_loader, y_tr)

metrics_dynamic = train_and_evaluate(
    'Dynamic (VLE)',     DynamicBaseline(seq_len),
    train_loader, val_loader, test_loader, y_tr)

metrics_imbal = train_and_evaluate(
    'Hybrid (Imbalanced)', HybridModel(static_dim, seq_len),
    train_loader, val_loader, test_loader, y_tr, use_weights=False)

metrics_hybrid = train_and_evaluate(
    'Hybrid (Balanced)',   HybridModel(static_dim, seq_len),
    train_loader, val_loader, test_loader, y_tr, use_weights=True)


# 4. Tables 1 and 4
table1_comparative(metrics_static, metrics_dynamic, metrics_hybrid)
table4_balancing(metrics_imbal, metrics_hybrid)


# Figures 4 + 7
results_dict = {
    'Static':            metrics_static,
    'Dynamic':           metrics_dynamic,
    'Hybrid (Imbal)':    metrics_imbal,
    'Hybrid (Balanced)': metrics_hybrid,
}
figure_learning_curves(results_dict)
figure_val_f1(results_dict)


# 5. Table 2 - classical ML baselines
if RUN_BASELINE_ML:
    table2_baselines(splits, metrics_hybrid)


# 6. Table 3 - k-fold CV
if RUN_KFOLD:
    table3_kfold(X_s_raw, X_d_raw, y, static_dim, seq_len)


# 7. Table 5 - ablation
if RUN_ABLATION:
    table5_ablation(splits, metrics_hybrid, static_dim, seq_len)


# 8. Visualization (Figures 8/9, 10/11/12, 13)
final_model = HybridModel(static_dim, seq_len).to(device)
final_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
final_model.eval()

probs_final, preds_final, targets_final, embeddings = [], [], [], []
with torch.no_grad():
    for s_b, d_b, y_b in test_loader:
        s_b, d_b = s_b.to(device), d_b.to(device)
        out = final_model(s_b, d_b, return_embedding=False)
        emb = final_model(s_b, d_b, return_embedding=True)
        probs_final.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
        preds_final.extend(torch.argmax(out, dim=1).cpu().numpy())
        targets_final.extend(y_b.numpy())
        embeddings.extend(emb.cpu().numpy())

probs_final   = np.array(probs_final)
preds_final   = np.array(preds_final)
targets_final = np.array(targets_final)
embeddings    = np.array(embeddings)

figure_cm_roc(targets_final, preds_final, probs_final)

if RUN_LATENT_VIZ:
    figure_tsne(embeddings, targets_final)
    figure_pca(embeddings, targets_final)
    figure_umap(embeddings, targets_final)

figure_attention(final_model, test_loader)

print("\nDone.")
