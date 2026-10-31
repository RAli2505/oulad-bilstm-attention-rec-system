"""OULAD loader and feature engineering. Mirrors Untitled6.ipynb section 2."""
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, SEQ_LEN


OULAD_URLS = [
    "https://analyse.kmi.open.ac.uk/open-dataset/download",
    "https://analyse.kmi.open.ac.uk/open_dataset/download",
]
REQUIRED = ["studentInfo.csv", "studentVle.csv",
            "studentAssessment.csv", "assessments.csv"]


def download_oulad(target_dir: str = DATA_PATH) -> None:
    if all(os.path.exists(os.path.join(target_dir, f)) for f in REQUIRED):
        print("[info] OULAD already present.")
        return
    zip_path = os.path.join(target_dir, "oulad.zip")
    last_err = None
    for url in OULAD_URLS:
        try:
            print(f"[info] downloading OULAD from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as fout:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fout.write(chunk)
            break
        except Exception as e:
            last_err = e
            print(f"[warn] failed: {e}")
    else:
        raise RuntimeError(f"Could not download OULAD. Last error: {last_err}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    os.remove(zip_path)


def get_data(path: str = DATA_PATH):
    """Returns X_static, X_dynamic, y, uid_target, vle_daily.

    vle_daily is the raw (uid, date, sum_click) within [0, 240] for the
    daily engagement figures (Fig 3, 5, 6).
    """
    print("Loading data ... (OULAD)")
    info        = pd.read_csv(f'{path}/studentInfo.csv')
    vle         = pd.read_csv(f'{path}/studentVle.csv')
    assess      = pd.read_csv(f'{path}/studentAssessment.csv')
    assess_info = pd.read_csv(f'{path}/assessments.csv')

    def create_uid(df):
        return (df['id_student'].astype(str) + "_"
                + df['code_module'] + "_" + df['code_presentation'])

    info['uid']   = create_uid(info)
    vle['uid']    = create_uid(vle)
    assess        = assess.merge(assess_info, on='id_assessment', how='left')
    assess['uid'] = create_uid(assess)

    print("Files loaded. Feature engineering ...")

    assess_features = assess.groupby('uid').agg({
        'score': ['mean', 'max', 'min'],
        'date_submitted': ['count'],
    })
    assess_features.columns = ['mean_score', 'max_score', 'min_score', 'submit_count']
    assess_features = assess_features.fillna(0)

    vle = vle[(vle['date'] >= -20) & (vle['date'] < 260)]
    vle['week'] = pd.cut(vle['date'], bins=np.arange(-21, 273, 7), labels=False)

    vle_pivot = vle.pivot_table(index='uid', columns='week',
                                values='sum_click', aggfunc='sum',
                                fill_value=0)

    full_weeks = pd.DataFrame(index=vle_pivot.index,
                              columns=range(SEQ_LEN)).fillna(0)
    vle_final = full_weeks.add(vle_pivot, fill_value=0).fillna(0)

    info = info[info['final_result'] != 'Withdrawn']
    info['target'] = info['final_result'].apply(
        lambda x: 1 if x in ['Pass', 'Distinction'] else 0)

    cat_cols = ['gender', 'region', 'highest_education',
                'imd_band', 'age_band', 'disability']
    for col in cat_cols:
        info[col] = info[col].astype(str)
        info[col] = LabelEncoder().fit_transform(info[col])

    common_uids = info['uid'][info['uid'].isin(vle_final.index)].values
    info = info.set_index('uid').loc[common_uids]
    vle_final = vle_final.loc[common_uids]
    info = info.join(assess_features, how='left').fillna(0)

    static_cols = cat_cols + ['num_of_prev_attempts', 'studied_credits',
                              'mean_score', 'max_score', 'submit_count']
    X_static  = info[static_cols].to_numpy(dtype=np.float32)
    X_dynamic = vle_final.to_numpy(dtype=np.float32)
    y         = info['target'].to_numpy(dtype=np.int64)

    uid_target = pd.Series(y, index=common_uids, name='target')
    vle_daily  = vle[(vle['date'] >= 0) & (vle['date'] <= 240)][[
        'uid', 'date', 'sum_click']].copy()

    print(f"Dataset ready. Students: {len(y)}. Class distribution: {np.bincount(y)}")
    return X_static, X_dynamic, y, uid_target, vle_daily


def build_flat_features(X_static_arr, X_dynamic_arr):
    """Static + clickstream summary stats for classical ML baselines (Table 2)."""
    Xs = np.asarray(X_static_arr,  dtype=np.float32)
    Xd = np.asarray(X_dynamic_arr, dtype=np.float32)
    agg_mean = Xd.mean(axis=1, keepdims=True)
    agg_std  = Xd.std(axis=1, keepdims=True)
    agg_max  = Xd.max(axis=1, keepdims=True)
    agg_sum  = Xd.sum(axis=1, keepdims=True)
    return np.concatenate([Xs, agg_mean, agg_std, agg_max, agg_sum], axis=1)
