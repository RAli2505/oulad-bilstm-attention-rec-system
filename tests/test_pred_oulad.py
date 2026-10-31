# tests/test_pred_oulad.py
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from models.pred_oulad import OULADPred, activity_vocab, MAX_DAYS, StaticEncoder

ENROLL_KEYS = ["id_student", "code_module", "code_presentation"]


def test_activity_vocab_sorted_unique():
    daily = pd.DataFrame({
        "activity_type": ["forumng", "oucontent", "forumng", "homepage"],
    })
    assert activity_vocab(daily) == ["forumng", "homepage", "oucontent"]


def _daily(rows):
    return pd.DataFrame(rows, columns=[
        "id_student", "code_module", "code_presentation",
        "date", "activity_type", "sum_click"])


def _target(rows):
    return pd.DataFrame(rows, columns=ENROLL_KEYS + [
        "final_result", "first_deadline", "date_unregistration",
        "module_presentation_length"])


def test_build_sequences_shapes_labels_padding():
    # 2 enrollments. Student 1 (AAA): clicks on days 0,2 before fd=5; one row at
    # day 7 (>= fd) MUST be ignored. final_result Withdrawn, unreg day 30.
    # Student 2 (AAA): clicks day 1 before fd=5. final_result Pass.
    daily = _daily([
        (1, "AAA", "2013J", 0, "forumng", 3),
        (1, "AAA", "2013J", 2, "oucontent", 5),
        (1, "AAA", "2013J", 7, "forumng", 9),     # >= fd -> excluded
        (2, "AAA", "2013J", 1, "oucontent", 4),
    ])
    target = _target([
        (1, "AAA", "2013J", "Withdrawn", 5.0, 30.0, 260),
        (2, "AAA", "2013J", "Pass", 5.0, np.nan, 260),
    ])
    keys = pd.DataFrame([(1, "AAA", "2013J"), (2, "AAA", "2013J")], columns=ENROLL_KEYS)
    acts = ["forumng", "oucontent"]

    p = OULADPred()
    X, lengths, y, t, e = p.build_sequences(daily, target, keys, acts)

    assert X.shape == (2, MAX_DAYS, 2)                 # [N, MAX_DAYS, F]
    # student 1: day0 forumng=3, day2 oucontent=5; day7 excluded
    assert X[0, 0, 0] == 3 and X[0, 2, 1] == 5 and X[0, 7, 0] == 0
    # student 2: day1 oucontent=4
    assert X[1, 1, 1] == 4
    assert lengths.tolist() == [5, 5]                  # ceil(fd)=5, capped at MAX_DAYS
    assert y.tolist() == [1, 0]                        # Withdrawn=dropout, Pass=no
    assert e.tolist() == [1, 0]                        # event only for Withdrawn
    assert t[0] == 30 and t[1] == 260                  # unreg day / censored at module_len


def test_static_encoder_fit_transform_and_unseen():
    train = pd.DataFrame({
        "gender": ["M", "F", "M"],
        "imd_band": ["0-10%", None, "90-100%"],   # NaN -> missing flag
        "num_of_prev_attempts": [0, 1, 2],
    })
    enc = StaticEncoder(cat_cols=["gender", "imd_band"],
                        num_cols=["num_of_prev_attempts"])
    Xtr = enc.fit_transform(train)
    assert Xtr.dtype == np.float32
    assert Xtr.shape[0] == 3
    # one-hot(gender: M,F = 2) + missing flag(gender)=1
    #  + one-hot(imd_band: 0-10%,90-100% = 2) + missing flag(imd_band)=1
    #  + num(num_of_prev_attempts)=1  => S = 2+1+2+1+1 = 7
    assert Xtr.shape[1] == 7
    assert enc.n_static == 7
    # numeric standardized: mean ~0 over train
    assert abs(Xtr[:, -1].mean()) < 1e-5
    # column layout (gender vocab sorted: F=0, M=1):
    #   [gender_F, gender_M, gender_missing, imd_0-10%, imd_90-100%, imd_missing, num]
    # row 1 has imd_band=None -> imd_missing flag set, imd one-hot zeros
    assert Xtr[1, 5] == 1.0                          # imd_missing flag
    assert Xtr[1, 3] == 0.0 and Xtr[1, 4] == 0.0   # imd one-hot all zero
    test = pd.DataFrame({
        "gender": ["X"],                 # unseen category -> all-zero one-hot
        "imd_band": ["0-10%"],
        "num_of_prev_attempts": [0],
    })
    Xte = enc.transform(test)
    assert Xte.shape == (1, 7)
    # gender one-hot block (first 2 cols: gender_F, gender_M) all zero for unseen "X"
    assert Xte[0, 0] == 0.0 and Xte[0, 1] == 0.0
    # gender_missing flag (col 2) must be 1 for unseen category "X"
    assert Xte[0, 2] == 1.0


import torch
from models.pred_oulad import PredNet, cox_loss, c_index


def test_prednet_output_shapes():
    net = PredNet(n_feat=4)
    x = torch.zeros(3, MAX_DAYS, 4)
    lengths = torch.tensor([5, 10, 1])
    logit, risk = net(x, lengths)
    assert logit.shape == (3,) and risk.shape == (3,)
    assert torch.isfinite(logit).all() and torch.isfinite(risk).all()


def test_cox_loss_finite():
    risk = torch.tensor([0.5, -0.2, 1.0, 0.0])
    time = torch.tensor([10.0, 20.0, 5.0, 30.0])
    event = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = cox_loss(risk, time, event)
    assert torch.isfinite(loss)


def test_c_index_perfect_and_reversed():
    # higher risk should mean earlier event. Perfect ordering -> 1.0, reversed -> 0.0.
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk_perfect = np.array([4.0, 3.0, 2.0, 1.0])
    risk_reversed = np.array([1.0, 2.0, 3.0, 4.0])
    assert c_index(risk_perfect, time, event) == 1.0
    assert c_index(risk_reversed, time, event) == 0.0


def _toy_split(n, F, seed, S=5):
    rng = np.random.default_rng(seed)
    X = rng.random((n, MAX_DAYS, F)).astype(np.float32)
    lengths = rng.integers(5, MAX_DAYS, size=n).astype(np.int64)
    static = rng.random((n, S)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    e = y.copy()                                  # event == dropout for the toy
    t = rng.integers(10, 260, size=n).astype(np.float32)
    return X, lengths, static, y, t, e


def test_train_and_evaluate_returns_metrics():
    F = 4
    tr = _toy_split(60, F, 0)
    va = _toy_split(20, F, 1)
    te = _toy_split(20, F, 2)
    p = OULADPred()
    p.train(*tr, *va, epochs=3)                   # short run
    m = p.evaluate(*te)
    for key in ("dropout_auc", "c_index", "brier"):
        assert key in m and 0.0 <= m[key] <= 1.0


def _write_pred_assets(processed, tmp):
    proc = Path(processed) / "processed"
    splt = Path(processed) / "splits"
    proc.mkdir(parents=True, exist_ok=True)
    splt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    acts = ["forumng", "oucontent", "homepage"]

    target_rows, daily_rows, assess_rows = [], [], []
    split_assign = {}     # enrollment -> split
    for sid in range(1, 61):
        res = ["Pass", "Fail", "Withdrawn", "Distinction"][sid % 4]
        unreg = float(rng.integers(20, 200)) if res == "Withdrawn" else np.nan
        target_rows.append((sid, "AAA", "2013J", res, 10.0, unreg, 260,
                            "M" if sid % 2 else "F", "R1", "HE",
                            "0-10%" if sid % 3 else None, "0-35", "N",
                            int(sid % 3), 60, -30.0))
        for day in range(0, 9):                       # all < fd=10
            a = acts[rng.integers(0, 3)]
            daily_rows.append((sid, "AAA", "2013J", day, a, int(rng.integers(1, 10))))
        # half submit early (day 5 < fd 10), half late (day 15)
        assess_rows.append((100, sid, 5 if sid % 2 == 0 else 15, 0, 70.0))
        split_assign[sid] = "train" if sid <= 40 else ("val" if sid <= 50 else "test")

    cols_t = ENROLL_KEYS + ["final_result", "first_deadline",
                            "date_unregistration", "module_presentation_length",
                            "gender", "region", "highest_education", "imd_band",
                            "age_band", "disability", "num_of_prev_attempts",
                            "studied_credits", "date_registration"]
    pd.DataFrame(target_rows, columns=cols_t).to_parquet(proc / "student_target.parquet")
    pd.DataFrame(daily_rows, columns=[
        "id_student", "code_module", "code_presentation", "date",
        "activity_type", "sum_click"]).to_parquet(proc / "daily_vle.parquet")

    cols_s = ENROLL_KEYS + ["final_result"]
    for name in ("train", "val", "test"):
        rows = [(s, "AAA", "2013J", "Pass") for s, sp in split_assign.items() if sp == name]
        pd.DataFrame(rows, columns=cols_s).to_parquet(splt / f"{name}.parquet")

    raw = Path(tmp) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(assess_rows, columns=[
        "id_assessment", "id_student", "date_submitted", "is_banked", "score"
    ]).to_csv(raw / "studentAssessment.csv", index=False)
    pd.DataFrame({"id_assessment": [100], "code_module": ["AAA"],
                  "code_presentation": ["2013J"]}).to_csv(raw / "assessments.csv", index=False)
    return str(raw)


def _assess_fixture():
    # two assessments; student 1 submits early (day 5 < fd 20), student 2 submits late (day 25 >= fd 20)
    assess = pd.DataFrame({
        "id_student": [1, 2],
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "date_submitted": [5, 25],
        "is_banked": [0, 0],
        "score": [80.0, 60.0],
    })
    target = pd.DataFrame({
        "id_student": [1, 2],
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "first_deadline": [20.0, 20.0],
        "gender": ["M", "F"],
        "region": ["R1", "R2"],
        "highest_education": ["HE", "A"],
        "imd_band": ["0-10%", None],
        "age_band": ["0-35", "35-55"],
        "disability": ["N", "N"],
        "num_of_prev_attempts": [0, 1],
        "studied_credits": [60, 120],
        "date_registration": [-30.0, -20.0],
    })
    keys = target[["id_student", "code_module", "code_presentation"]].copy()
    return assess, target, keys


def test_build_static_shapes_and_leakfree():
    assess, target, keys = _assess_fixture()
    pred = OULADPred()
    S = pred.build_static(target, assess, keys, fit=True)
    assert S.dtype == np.float32
    assert S.shape[0] == 2
    assert pred.n_static == S.shape[1] > 0
    # student 1 submitted early -> submitted_any == 1; student 2 late -> 0.
    sa = pred._last_submitted_any
    assert sa.tolist() == [1.0, 0.0]
    # transform-only path (fit=False) reuses encoder, same width
    S2 = pred.build_static(target, assess, keys, fit=False)
    assert S2.shape == S.shape


def test_fusion_forward_shapes():
    torch.manual_seed(0)
    B, T, F, S = 4, 61, 5, 7
    net = PredNet(F, S)
    x = torch.randn(B, T, F)
    lengths = torch.tensor([10, 20, 61, 5])
    static = torch.randn(B, S)
    cls, risk = net(x, lengths, static)
    assert cls.shape == (B,)
    assert risk.shape == (B,)
    # static actually influences the output
    static2 = static + 5.0
    cls2, _ = net(x, lengths, static2)
    assert not torch.allclose(cls, cls2)


def test_build_integration(tmp_path):
    proc = tmp_path / "data"
    raw = _write_pred_assets(proc, tmp_path)
    p = OULADPred(processed_dir=str(proc), data_dir=raw,
                  model_dir=str(tmp_path / "models"),
                  results_dir=str(tmp_path / "results"))
    out = p.build(epochs=3)
    assert Path(out["metrics_path"]).exists()
    m = json.loads(Path(out["metrics_path"]).read_text())
    for key in ("dropout_auc", "c_index", "brier", "n_train", "n_test", "n_pos", "n_static"):
        assert key in m
    assert 0.0 <= m["dropout_auc"] <= 1.0
    assert Path(out["model_path"]).exists()
