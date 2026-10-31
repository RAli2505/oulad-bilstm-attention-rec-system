# Hybrid Deep Learning for Academic Risk Prediction (OULAD)

Reproduction of: **Omarbekova et al. (2026)** — *A temporal attention-based hybrid deep learning model for student performance and academic risk prediction.* Frontiers in AI 9:1811886. DOI: 10.3389/frai.2026.1811886

A hybrid architecture: **Bi-LSTM + temporal attention** (weekly behavioral signals) + **MLP** (demographic / assessment static features) → binary *Pass / Fail* classification on the **OULAD** dataset.

---

## Project layout

```
.
├── config.py        # paths, seed, device, hyperparameters, section toggles
├── data.py          # OULAD download, feature engineering, build_flat_features
├── models.py        # StaticBaseline, DynamicBaseline, HybridModel
├── training.py      # train_and_evaluate with early stopping (val F1, patience=7)
├── tables.py        # Tables 1–5 generation
├── figures.py       # Figures 1, 3–13 generation
├── main.py          # full pipeline orchestration
├── oulad/           # dataset is auto-downloaded here
├── tables/          # Tables 1–5 saved as CSV
├── figures/         # Figures 1, 3–13 saved as PNG
├── best_model.pth   # best Hybrid (Balanced) weights after training
└── frai-9-1811886.pdf
```

---

## Quick start

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn tqdm \
            xgboost lightgbm umap-learn

python main.py
```

OULAD is auto-downloaded into `oulad/` on the first run.

---

## What `main.py` does

| Step | Artifacts |
|---|---|
| 1. Load OULAD, feature engineering | — |
| 2. Figure 1 — class distribution | `fig01_class_distribution.png` |
| 3. Figures 3, 5, 6 — daily engagement dynamics | `fig03/05/06_*.png` |
| 4. Stratified 64/16/20 split + scaling | — |
| 5. Train 4 models (Static / Dynamic / Hybrid Imbal / Hybrid Bal) | `best_model.pth` |
| 6. Tables 1, 4 — comparative analysis & balancing impact | `table1/4_*.csv` |
| 7. Figures 4 + 7 — training loss & validation F1 curves | `fig04_07/07_*.png` |
| 8. Table 2 — classical ML baselines (LR / RF / SVM / XGB / LGBM / MLP) | `table2_baselines.csv` |
| 9. Table 3 — 5-fold CV, mean ± std | `table3_kfold.csv` |
| 10. Table 5 — ablation by data window (weeks 0–15 / 0–20 / full) | `table5_ablation.csv` |
| 11. Figures 8, 9 — confusion matrix + ROC | `fig08_09_cm_roc.png` |
| 12. Figures 10, 11, 12 — t-SNE / PCA / UMAP of latent space | `fig10/11/12_*.png` |
| 13. Figure 13 — temporal attention weights across course weeks | `fig13_attention.png` |

---

## Alignment with the article

| Parameter | Value |
|---|---|
| Sequence length | 36 weeks (T=36) |
| Temporal stream | Bi-LSTM 2×64 + Bahdanau attention `α = softmax(uᵀ tanh(Wh+b))` |
| Static stream | Linear(128) → BatchNorm → ReLU → Dropout(0.4) → Linear(64) → ReLU |
| Fusion | concat(128 + 64) → Linear(64) → ReLU → Dropout(0.3) → Linear(2) |
| Optimizer | AdamW, lr=1e-3 |
| Loss | Weighted CrossEntropy with class weights `w_c = N / (k·n_c)` |
| Split | 64% train / 16% val / 20% test, stratified |
| Early stopping | patience=7 epochs on validation weighted F1, up to 100 epochs |
| Batch size | train=64, eval=256 |

---

## Toggles (`config.py`)

Expensive sections can be disabled:

```python
RUN_BASELINE_ML = True   # Table 2
RUN_KFOLD       = True   # Table 3 (slow!)
RUN_ABLATION    = True   # Table 5
RUN_LATENT_VIZ  = True   # Figures 10/11/12
KFOLD_SPLITS    = 5
KFOLD_EPOCHS    = 30     # epoch cap inside k-fold
```

---

## Expected metrics (article)

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|
| Static Vector | 0.8175 | 0.8167 | 0.8767 |
| Behavioral Vector | 0.8879 | 0.8855 | 0.9149 |
| **Hybrid (Balanced)** | **0.9024** | **0.9008** | **0.9497** |

Recall for the Fail class with Hybrid (Balanced): ~0.79.

---

## Data source

[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset)

License: CC BY 4.0.
