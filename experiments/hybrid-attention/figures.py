"""All figures from Frontiers AI 9:1811886 (Fig 1, 3-13)."""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from config import FIG_DIR, device

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

COLOR_FAIL = '#3a6c8c'
COLOR_PASS = '#4ca97a'


def _save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   saved {path}")


# Figure 1
def figure_class_distribution(y):
    counts = pd.Series(y).value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    plt.bar(['At-risk / Fail (0)', 'Successful / Pass (1)'],
            counts.values, color=[COLOR_FAIL, COLOR_PASS], edgecolor='black')
    for i, v in enumerate(counts.values):
        pct = 100 * v / counts.sum()
        plt.text(i, v + max(counts.values) * 0.01,
                 f"{v}\n({pct:.1f}%)", ha='center', fontsize=11)
    plt.title('Distribution of student outcomes', fontsize=14)
    plt.ylabel('Number of students')
    _save('fig01_class_distribution.png')


def _daily_means(uid_target, vle_daily):
    df = vle_daily.merge(uid_target.rename('target'),
                         left_on='uid', right_index=True, how='inner')
    g = df.groupby(['date', 'target'])['sum_click'].mean().unstack('target', fill_value=0.0)
    g = g.reindex(columns=[0, 1], fill_value=0.0)
    return g


# Figure 3
def figure_engagement_daily(uid_target, vle_daily):
    g = _daily_means(uid_target, vle_daily)
    plt.figure(figsize=(11, 5))
    plt.plot(g.index, g[1], color=COLOR_PASS, lw=2, label='Successful (Pass)')
    plt.plot(g.index, g[0], color=COLOR_FAIL, lw=2, label='At-risk (Fail)')
    plt.title('Daily engagement dynamics (mean clicks)', fontsize=14)
    plt.xlabel('Day relative to course start')
    plt.ylabel('Mean clicks per student')
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save('fig03_engagement_daily.png')


# Figure 5
def figure_engagement_gap(uid_target, vle_daily):
    g = _daily_means(uid_target, vle_daily)
    diff = g[1] - g[0]
    plt.figure(figsize=(11, 5))
    plt.bar(diff.index, diff.values,
            color=np.where(diff.values >= 0, COLOR_PASS, COLOR_FAIL),
            edgecolor='none', width=1.0)
    plt.axhline(0, color='black', lw=0.7)
    plt.title('Daily engagement gap (Successful - At-risk)', fontsize=14)
    plt.xlabel('Day relative to course start')
    plt.ylabel('Mean clicks (difference)')
    plt.grid(True, alpha=0.3)
    _save('fig05_engagement_gap.png')


# Figure 6
def figure_engagement_gap_smoothed(uid_target, vle_daily, window=7):
    g = _daily_means(uid_target, vle_daily)
    diff = (g[1] - g[0]).rolling(window=window, min_periods=1, center=True).mean()
    plt.figure(figsize=(11, 5))
    plt.plot(diff.index, diff.values, color='#1f77b4', lw=2)
    plt.fill_between(diff.index, 0, diff.values,
                     where=diff.values >= 0, color=COLOR_PASS, alpha=0.3)
    plt.fill_between(diff.index, 0, diff.values,
                     where=diff.values < 0, color=COLOR_FAIL, alpha=0.3)
    plt.axhline(0, color='black', lw=0.7)
    plt.title(f'Smoothed daily engagement gap ({window}-day moving average)',
              fontsize=14)
    plt.xlabel('Day relative to course start')
    plt.ylabel('Mean clicks (smoothed difference)')
    plt.grid(True, alpha=0.3)
    _save('fig06_engagement_gap_smoothed.png')


# Figure 4 + 7 combined (training loss + val F1)
def figure_learning_curves(results_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for (name, metrics), color, marker in zip(results_dict.items(), colors, markers):
        hist = metrics['History']
        epochs = range(1, len(hist['loss']) + 1)
        ax1.plot(epochs, hist['loss'], label=name, color=color,
                 marker=marker, markersize=5, alpha=0.8)
        ax2.plot(epochs, hist['val_f1'], label=name, color=color,
                 marker=marker, markersize=5, linestyle='--', alpha=0.8)

    ax1.set_title('Training loss dynamics', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss (CrossEntropy)')
    ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()

    ax2.set_title('Validation weighted F1-score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('F1-score')
    ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend()

    plt.tight_layout()
    _save('fig04_07_learning_curves.png')


# Figure 7 standalone
def figure_val_f1(results_dict):
    plt.figure(figsize=(11, 5))
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    for (name, metrics), color, marker in zip(results_dict.items(), colors, markers):
        hist = metrics['History']
        epochs = range(1, len(hist['val_f1']) + 1)
        plt.plot(epochs, hist['val_f1'], label=name, color=color,
                 marker=marker, markersize=5, lw=2)
    plt.title('Validation weighted F1-score', fontsize=14)
    plt.xlabel('Epochs'); plt.ylabel('Weighted F1')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    _save('fig07_val_f1.png')


# Figure 8 + 9
def figure_cm_roc(targets, preds, probs):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    cm = confusion_matrix(targets, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax[0], cbar=False, annot_kws={"size": 14})
    ax[0].set_title('Confusion Matrix (Hybrid Balanced)', fontsize=14)
    ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('True')
    ax[0].set_xticklabels(['Fail (0)', 'Pass (1)'])
    ax[0].set_yticklabels(['Fail (0)', 'Pass (1)'])

    fpr, tpr, _ = roc_curve(targets, probs)
    auc_val = roc_auc_score(targets, probs)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (area = {auc_val:.3f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_title('ROC Curve', fontsize=14)
    ax[1].set_xlabel('False Positive Rate'); ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")
    plt.tight_layout()
    _save('fig08_09_cm_roc.png')


def _project(name, projector, embeddings, targets, sample=2000):
    n = len(embeddings)
    idx = np.random.choice(n, min(sample, n), replace=False)
    emb_2d = projector.fit_transform(embeddings[idx])
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=targets[idx],
                          cmap='coolwarm', alpha=0.6, edgecolors='w', s=40)
    plt.colorbar(scatter, label='Outcome (0=Fail, 1=Pass)')
    plt.title(f'{name} of student latent vectors', fontsize=14)
    plt.xlabel('Dimension 1'); plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)


# Figure 10
def figure_tsne(embeddings, targets):
    print("Computing t-SNE ...")
    _project('t-SNE', TSNE(n_components=2, perplexity=30, random_state=42),
             embeddings, targets)
    _save('fig10_tsne.png')


# Figure 11
def figure_pca(embeddings, targets):
    print("Computing PCA ...")
    _project('PCA', PCA(n_components=2, random_state=42),
             embeddings, targets)
    _save('fig11_pca.png')


# Figure 12
def figure_umap(embeddings, targets):
    try:
        import umap
    except ImportError:
        print("[skip] UMAP not installed.")
        return
    print("Computing UMAP ...")
    _project('UMAP', umap.UMAP(n_components=2, random_state=42),
             embeddings, targets)
    _save('fig12_umap.png')


# Figure 13
def figure_attention(model, test_loader):
    model.eval()
    all_alpha = []
    with torch.no_grad():
        for _, d_b, _ in test_loader:
            d_b = d_b.to(device)
            all_alpha.append(model.attention_weights(d_b).cpu().numpy())
    avg_attention = np.concatenate(all_alpha, axis=0).mean(axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(avg_attention)), avg_attention,
             marker='o', color='purple', lw=2)
    plt.title('Temporal feature importance (attention mechanism analysis)',
              fontsize=14)
    plt.xlabel('Course Week (-3 to 32)')
    plt.ylabel('Average attention weight')
    plt.grid(True, alpha=0.3)
    _save('fig13_attention.png')
