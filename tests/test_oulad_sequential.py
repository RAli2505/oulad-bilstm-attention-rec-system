import numpy as np
import pandas as pd
import torch
import pytest

from recommendations.sequential import DailySASRecService


def fixture():
    resources = pd.DataFrame([dict(code_module='AAA', code_presentation='2013J', id_site=i)
                              for i in range(1, 5)])
    events = pd.DataFrame([dict(id_student=u, code_module='AAA', code_presentation='2013J',
                                id_site=item, date=day, sum_click=click)
                           for u in (1, 2) for day, item, click in
                           [(0, 1, 2), (0, 2, 1), (1, 2, 4), (2, 3, 1)]])
    return resources, events


def fitted(events=None):
    resources, original = fixture()
    service = DailySASRecService(dimension=8, heads=2, blocks=1, dropout=0)
    return service.fit(resources, original if events is None else events, [(1, 'AAA', '2013J')],
                       3, epochs=1, device='cpu', batch_size=4)


def test_fit_only_cutoff_and_global_rng():
    resources, events = fixture()
    before = torch.random.get_rng_state().clone()
    model = fitted()
    assert torch.equal(before, torch.random.get_rng_state())
    changed = events.copy()
    changed.loc[changed.id_student == 2, 'sum_click'] = 999
    future = events.iloc[:1].assign(date=99, sum_click=999)
    other = fitted(pd.concat([changed, future], ignore_index=True))
    for name, value in model.model.state_dict().items():
        assert torch.equal(value, other.model.state_dict()[name])
    assert model.metadata['role_counts']['fit_prefixes'] == 2


def test_baskets_order_frozen_and_cold():
    resources, events = fixture()
    model = fitted()
    key = (1, 'AAA', '2013J')
    state = {k: v.clone() for k, v in model.model.state_dict().items()}
    first = model.affinity(key, events, resources, 3)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, model.affinity(key, events.sample(frac=1, random_state=1), resources, 3))
    reordered = events.copy()
    reordered['date'] = 2 - reordered.date
    assert not np.allclose(first, model.affinity(key, reordered, resources, 3))
    assert first[-1] == 0
    assert not model.affinity((90, 'AAA', '2013J'), events, resources, 3).any()
    unknown = resources.iloc[:1].assign(id_site=1000)
    assert not model.affinity(key, events, unknown, 3).any()
    cold_history = events.iloc[:1].assign(id_site=4)
    assert not model.affinity(key, cold_history, resources, 3).any()
    try:
        model.affinity(key, events, resources, 2)
    except ValueError:
        pass
    else:
        raise AssertionError('Inference before fit cutoff must be rejected')
    for name, value in state.items():
        assert torch.equal(value, model.model.state_dict()[name])
    assert all(not p.requires_grad for p in model.model.parameters())


def test_roundtrip_and_causal_encoder(tmp_path):
    resources, events = fixture()
    model = fitted()
    path = tmp_path / 'sequence.pt'
    model.save(path)
    loaded = DailySASRecService.load(path)
    np.testing.assert_array_equal(model.affinity((1, 'AAA', '2013J'), events, resources, 3),
                                  loaded.affinity((1, 'AAA', '2013J'), events, resources, 3))
    assert model.item_to_index == loaded.item_to_index
    days = model._baskets(events, {(1, 'AAA', '2013J')}, 3)[(1, 'AAA', '2013J')]
    args = model._pack([days])
    with torch.inference_mode():
        first = model.model.encode(*args)
        args[0][0, -1, 0] = 1
        second = model.model.encode(*args)
    torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0, atol=0)


def test_selection_metadata_and_empty_training():
    resources, events = fixture()
    model = DailySASRecService(dimension=8, heads=2, blocks=1).fit(
        resources, events, [(1, 'AAA', '2013J')], 3,
        selection_keys=[(2, 'AAA', '2013J')], epochs=1, device='cpu')
    assert model.metadata['selection_used_for_stopping']
    assert model.metadata['loss_curve'][0]['selection_loss'] is not None
    empty = DailySASRecService(dimension=8, heads=2, blocks=1).fit(
        resources, events.iloc[:0], [], 3, epochs=1)
    assert empty.metadata['epochs'] == 0
    assert not empty.affinity((1, 'AAA', '2013J'), events, resources, 3).any()


def test_batched_packing_loss_and_gradient_match_reference():
    _, events = fixture()
    model = fitted()
    days = model._baskets(events, {(1, 'AAA', '2013J')}, 3)[(1, 'AAA', '2013J')]
    pairs = [(days[:1], days[1]), (days[:2], days[0])]
    packed = model._pack([x[0] for x in pairs])
    expected_ids = torch.tensor([[[1, 2], [0, 0]], [[1, 2], [2, 0]]])
    torch.testing.assert_close(packed[0], expected_ids)
    torch.testing.assert_close(packed[2], torch.tensor([1, 2]))
    model.model.requires_grad_(True)
    vectors = model.model(*packed)
    log_probs = (vectors @ model.model.items.weight[1:].T).log_softmax(-1)
    reference_losses = []
    for i, (_, target) in enumerate(pairs):
        idx = torch.tensor([x[0] - 1 for x in target])
        weights = torch.tensor([x[1] for x in target])
        reference_losses.append(-(log_probs[i, idx] * weights).sum() / weights.sum())
    reference = torch.stack(reference_losses).mean()
    actual = model._loss_prepared(model._prepare_pairs(pairs))
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    params = tuple(model.model.parameters())
    expected_gradients = torch.autograd.grad(reference, params)
    actual_gradients = torch.autograd.grad(actual, params)
    for expected, observed in zip(expected_gradients, actual_gradients):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-7)
    assert model.metadata['execution']['device'] == 'cpu'
    assert model.metadata['execution']['fit_seconds'] >= 0
    assert model.metadata['loss_curve'][0]['seconds'] >= 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_affinity_preserves_cutoffs_candidates_and_scores(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    resources, events = fixture()
    model = fitted()
    model.device = torch.device(device)
    model.model.to(model.device)
    histories = [events, events.loc[events.date < 1], events.iloc[:0]]
    requests = [((1, 'AAA', '2013J'), h, resources) for h in histories]
    expected = [model.affinity(*r, 3) for r in requests]
    for size in [1, 2, 128]:
        actual = model.affinity_batch(requests, 3, batch_size=size)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=2e-6)
        assert not actual[2].any()
    with pytest.raises(ValueError):
        model.affinity_batch(requests, 2)
