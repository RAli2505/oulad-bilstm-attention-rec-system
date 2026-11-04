"""The optimized feature engine must reproduce the reference experiment."""
import numpy as np
import pandas as pd
import pytest


def feature_fixture():
    rng = np.random.default_rng(19)
    rows = [dict(id_student=i, code_module=module, code_presentation='2013J')
            for i in range(16) for module in ['AAA', 'BBB']]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    indices = {role: np.flatnonzero(frame.id_student % 4 == i)
               for i, role in enumerate(['fit', 'selection', 'calibration', 'held'])}
    catalog = []
    for module in ['AAA', 'BBB']:
        for site, kind in [(1, 'content'), (2, 'forum'), (3, 'quiz'), (4, 'url'), (5, 'content')]:
            catalog.append(dict(id_site=site, code_module=module, code_presentation='2013J',
                                activity_type=kind, week_from=8 if site == 3 else (np.nan if site == 4 else 0),
                                week_to=np.nan if site == 2 else 8,
                                available_from_day=0 if site == 2 else np.nan,
                                available_until_day=20 if site == 1 else np.nan))
    events = []
    for row in rows:
        if row['id_student'] in [0, 15]:
            continue
        for _ in range(30):
            events.append({**row, 'id_site': int(rng.choice([1, 2, 3, 4, 5, 999])),
                           'date': int(rng.integers(-2, 65)), 'sum_click': int(rng.integers(0, 50))})
    events += [events[0], events[0]]  # daily aggregation precedes click saturation
    courses = pd.DataFrame([dict(code_module=m, code_presentation='2013J',
                                 module_presentation_length=90 if m == 'AAA' else 30) for m in ['AAA', 'BBB']])
    return pd.DataFrame(catalog), pd.DataFrame(events), frame, indices, courses


@pytest.mark.parametrize('day', [7, 14, 56])
@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('allow_review', [False, True])
def test_batch_engine_reproduces_reference_candidates_labels_and_features(day, static, allow_review):
    from recommendations.experiment import make_candidates, rank_indices
    from recommendations.batch_features import make_candidates_batched
    resources, events, frame, indices, courses = feature_fixture()
    reference, queries, splits, _ = make_candidates(resources, events, frame, indices, day, 7, courses,
                                                    include_no_decay=static, feature_engine='reference', allow_review=allow_review)
    for batch_size in [1, 5, 256]:
        actual, actual_queries, actual_splits, latency = make_candidates_batched(
            resources, events, frame, indices, day, 7, courses, include_no_decay=static, batch_size=batch_size, allow_review=allow_review)
        pd.testing.assert_frame_equal(reference, actual, check_dtype=False, atol=1e-12, rtol=1e-12)
        pd.testing.assert_frame_equal(queries, actual_queries, check_dtype=False, atol=1e-12, rtol=1e-12)
        pd.testing.assert_frame_equal(splits, actual_splits, check_dtype=False)
        assert latency == []  # do not invent per-query percentile timings for a batch
        for query in queries.itertuples():
            a, b = reference.iloc[query.start:query.stop], actual.iloc[query.start:query.stop]
            risk = (query.query % 9) / 10
            def ranked(part):
                score = (.5 * part.preference + (.2 + .2 * risk) * part.calendar +
                         (.3 - .2 * risk) * part.engagement).to_numpy()
                return part.item.to_numpy()[rank_indices(score, part.item.to_numpy(), part.kind.to_numpy(), 20, .15)].tolist()
            assert ranked(a) == ranked(b)


def test_batches_do_not_mutate_inputs_or_leak_future_clicks():
    from recommendations.batch_features import make_candidates_batched
    resources, events, frame, indices, courses = feature_fixture()
    original = events.copy(deep=True)
    before, _, _, _ = make_candidates_batched(resources, events, frame, indices, 14, 7, courses)
    events.loc[events.date >= 14, 'sum_click'] *= 100
    after, _, _, _ = make_candidates_batched(resources, events, frame, indices, 14, 7, courses)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(resources, feature_fixture()[0])
    events.loc[:, 'sum_click'] = original.sum_click
    pd.testing.assert_frame_equal(events, original)


def test_batch_engine_preserves_empty_history_and_empty_candidate_queries():
    from recommendations.experiment import make_candidates
    from recommendations.batch_features import make_candidates_batched
    resources, events, frame, indices, courses = feature_fixture()
    resources['available_from_day'] = 100
    resources['available_until_day'] = np.nan
    events = events.iloc[:0]
    expected = make_candidates(resources, events, frame, indices, 7, 7, courses, feature_engine='reference')
    actual = make_candidates_batched(resources, events, frame, indices, 7, 7, courses)
    for old, new in zip(expected[:3], actual[:3]):
        pd.testing.assert_frame_equal(old, new, check_dtype=False)
