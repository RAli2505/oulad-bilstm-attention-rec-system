"""Training loop aligned with article: AdamW, weighted CE, early stopping
on validation weighted F1 (patience=7), up to 100 epochs."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             recall_score, classification_report)
from sklearn.utils import class_weight
from tqdm.auto import tqdm

from config import device, LR, EPOCHS, PATIENCE, MODEL_PATH, SEQ_LEN


def truncate_dynamic(matrix, n_weeks_kept):
    """Zero-out weeks beyond `n_weeks_kept` (LSTM still expects SEQ_LEN steps)."""
    out = matrix.copy()
    if n_weeks_kept < SEQ_LEN:
        out[:, n_weeks_kept:] = 0.0
    return out


def _inverse_freq_weights(y, num_classes=2):
    """Article: w_c = N / (k * n_c)."""
    counts = np.bincount(y, minlength=num_classes).astype(float)
    n = float(len(y))
    return n / (num_classes * counts)


def _evaluate(model, loader, criterion):
    model.eval()
    loss_sum, all_p, all_pr, all_t = 0.0, [], [], []
    with torch.no_grad():
        for s_b, d_b, y_b in loader:
            s_b, d_b, y_b = s_b.to(device), d_b.to(device), y_b.to(device)
            out = model(s_b, d_b)
            loss_sum += criterion(out, y_b).item()
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)
            all_pr.extend(probs.cpu().numpy())
            all_p.extend(preds.cpu().numpy())
            all_t.extend(y_b.cpu().numpy())
    return (loss_sum / max(len(loader), 1),
            np.array(all_t), np.array(all_p), np.array(all_pr))


def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader,
                       y_tr, epochs=EPOCHS, patience=PATIENCE, use_weights=True):
    print(f"\nRun: {model_name}")
    t0 = time.time()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    if use_weights:
        cw = torch.FloatTensor(_inverse_freq_weights(y_tr)).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    history = {'loss': [], 'val_loss': [], 'f1': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    counter = 0
    stopped_at = epochs

    pbar = tqdm(range(epochs), desc=model_name, leave=True)
    for epoch in pbar:
        model.train()
        ep_loss = 0.0
        for s_b, d_b, y_b in train_loader:
            s_b, d_b, y_b = s_b.to(device), d_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(s_b, d_b), y_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(train_loader)

        v_loss, v_t, v_p, _ = _evaluate(model, val_loader, criterion)
        v_f1 = f1_score(v_t, v_p, average='weighted')

        history['loss'].append(avg_loss)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(v_f1)
        history['f1'].append(v_f1)

        pbar.set_postfix({'loss': f"{avg_loss:.4f}",
                          'val_loss': f"{v_loss:.4f}",
                          'val_f1': f"{v_f1:.4f}"})

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stopped_at = epoch + 1
                print(f"  [early stop] {model_name} at epoch {stopped_at} "
                      f"(best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)

    _, t_targets, t_preds, t_probs = _evaluate(model, test_loader, criterion)
    metrics = {
        'Accuracy': accuracy_score(t_targets, t_preds),
        'F1-Score': f1_score(t_targets, t_preds, average='weighted'),
        'ROC-AUC': roc_auc_score(t_targets, t_probs),
        'Recall_Risk_0': recall_score(t_targets, t_preds, pos_label=0),
        'History': history,
        'Full_Report': classification_report(t_targets, t_preds, output_dict=True),
        'Time': time.time() - t0,
        'Stopped_At': stopped_at,
        'Best_Val_F1': best_val_f1,
    }
    if "Hybrid (Balanced)" in model_name:
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"   Done. Test F1={metrics['F1-Score']:.4f}, "
          f"AUC={metrics['ROC-AUC']:.4f}, Recall_0={metrics['Recall_Risk_0']:.4f}")
    return metrics
