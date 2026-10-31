"""Paths, seeds, device. Mirrors Untitled6.ipynb section 1."""
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Reuse the repository's locally verified OULAD source files.
DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'data', 'oulad', 'raw'))
OUTPUT_ROOT = os.environ.get('OULAD_RUN_DIR', PROJECT_ROOT)
FIG_DIR      = os.path.join(OUTPUT_ROOT, "figures")
TABLES_DIR   = os.path.join(OUTPUT_ROOT, "tables")
MODEL_PATH   = os.path.join(OUTPUT_ROOT, "best_model.pth")

os.makedirs(DATA_PATH,  exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN     = 36
BATCH_TRAIN = 64
BATCH_TEST  = 256
LR          = 1e-3
EPOCHS      = 100
PATIENCE    = 7

# Section toggles (set False to skip expensive blocks)
RUN_BASELINE_ML = False
RUN_KFOLD       = False
RUN_ABLATION    = False
RUN_LATENT_VIZ  = False
KFOLD_SPLITS    = 5
KFOLD_EPOCHS    = 30  # cap for k-fold to limit runtime
