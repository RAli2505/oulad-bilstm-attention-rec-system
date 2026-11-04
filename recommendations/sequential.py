"""Standalone SASRec-inspired daily-basket adaptation for OULAD.

Daily click aggregates are implicit strengths, not correctness. There is no
invented within-day order, pretrained XES transfer, or RecBole dependency.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

ENROLLMENT = ['id_student', 'code_module', 'code_presentation']
ITEM = ['code_module', 'code_presentation', 'id_site']


def _keys(values):
    if isinstance(values, pd.DataFrame):
        values = values[ENROLLMENT].itertuples(index=False, name=None)
    result = set()
    for value in values:
        key = tuple(value)
        if len(key) != 3 or any(pd.isna(x) for x in key):
            raise ValueError('Expected complete enrollment keys')
        result.add((int(key[0]), str(key[1]), str(key[2])))
    return result


class _BasketTransformer(nn.Module):
    def __init__(self, n_items, max_days, dimension, heads, blocks, dropout):
        super().__init__()
        self.items = nn.Embedding(n_items + 1, dimension, padding_idx=0)
        self.positions = nn.Embedding(max_days, dimension)
        layer = nn.TransformerEncoderLayer(dimension, heads, dimension * 4,
                                           dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)

    def encode(self, ids, strengths, lengths):
        """Return causal per-day states; ids/strengths have shape B x days x items."""
        embedded = (self.items(ids) * strengths.unsqueeze(-1)).sum(2)
        embedded = embedded / strengths.sum(2, keepdim=True).clamp_min(1e-8)
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.norm(embedded + self.positions(positions)[None])
        padding = positions[None] >= lengths[:, None]
        causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], device=ids.device,
                                     dtype=torch.bool), diagonal=1)
        return self.encoder(embedded, mask=causal, src_key_padding_mask=padding)

    def forward(self, ids, strengths, lengths):
        states = self.encode(ids, strengths, lengths)
        return states[torch.arange(len(lengths), device=ids.device), lengths - 1]


class DailySASRecService:
    """Causal daily basket model with frozen, raw dot-product utilities."""
    def __init__(self, seed=42, max_days=30, dimension=64, heads=4, blocks=2,
                 dropout=0.1, learning_rate=0.001, click_cap=100):
        if max_days < 1 or dimension < 1 or heads < 1 or dimension % heads or blocks < 1:
            raise ValueError('Invalid transformer dimensions')
        if click_cap <= 0 or learning_rate <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid training settings')
        self.config = dict(seed=int(seed), max_days=int(max_days), dimension=int(dimension),
                           heads=int(heads), blocks=int(blocks), dropout=float(dropout),
                           learning_rate=float(learning_rate), click_cap=float(click_cap))
        self.item_to_index = {}
        self.metadata = {}
        self.model = None

    @staticmethod
    def _device(device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

    def _baskets(self, interactions, keys, day):
        if interactions.empty:
            return {}
        required = ENROLLMENT + ['id_site', 'date', 'sum_click']
        missing = set(required) - set(interactions.columns)
        if missing:
            raise ValueError(f'Missing interaction columns: {sorted(missing)}')
        frame = interactions.loc[pd.to_numeric(interactions.date, errors='coerce') < day, required].copy()
        frame = frame.loc[[tuple((int(r[0]), str(r[1]), str(r[2]))) in keys
                           for r in frame[ENROLLMENT].itertuples(index=False, name=None)]]
        frame['sum_click'] = pd.to_numeric(frame.sum_click, errors='coerce').fillna(0).clip(lower=0)
        frame = frame.groupby(ENROLLMENT + ['date', 'id_site'], sort=True, as_index=False).sum(numeric_only=True)
        result = {}
        for position, (enrollment, events) in enumerate(frame.groupby(ENROLLMENT, sort=True), 1):
            days = []
            for _, events_day in events.groupby('date', sort=True):
                basket = []
                for row in events_day.itertuples(index=False):
                    idx = self.item_to_index.get((str(row.code_module), str(row.code_presentation), int(row.id_site)))
                    if idx is not None and row.sum_click > 0:
                        strength = float(np.log1p(min(row.sum_click, self.config['click_cap'])))
                        basket.append((idx, strength))
                if basket:
                    days.append(sorted(basket))
            result[tuple(enrollment)] = days
            if len(keys) > 1 and (position % 512 == 0 or position == len(keys)):
                print(f'  SASRec baskets: {position}/{len(keys)} enrollments', flush=True)
        return result

    def _prefixes(self, sequences, limit):
        pairs = []
        for key in sorted(sequences):
            days = sequences[key]
            positions = np.unique(np.linspace(1, len(days) - 1, min(limit, len(days) - 1), dtype=int)) if len(days) > 1 else []
            pairs.extend((days[max(0, int(i) - self.config['max_days']):int(i)], days[int(i)]) for i in positions)
        return pairs

    @staticmethod
    def _prepare_history(history):
        """Cache CPU arrays once; never create a GPU tensor per daily basket."""
        width = max(map(len, history))
        ids = np.zeros((len(history), width), dtype=np.int64)
        strengths = np.zeros_like(ids, dtype=np.float32)
        for j, basket in enumerate(history):
            ids[j, :len(basket)] = [x[0] for x in basket]
            strengths[j, :len(basket)] = [x[1] for x in basket]
        return ids, strengths

    def _pack_prepared(self, histories):
        lengths = np.asarray([len(x[0]) for x in histories], dtype=np.int64)
        width = max(x[0].shape[1] for x in histories)
        ids = np.zeros((len(histories), int(lengths.max()), width), dtype=np.int64)
        strengths = np.zeros_like(ids, dtype=np.float32)
        for i, (item_ids, weights) in enumerate(histories):
            ids[i, :len(item_ids), :item_ids.shape[1]] = item_ids
            strengths[i, :len(weights), :weights.shape[1]] = weights
        return tuple(torch.from_numpy(x).to(self.device) for x in (ids, strengths, lengths))

    def _pack(self, histories):
        return self._pack_prepared([self._prepare_history(history) for history in histories])

    def _prepare_pairs(self, pairs):
        return [(self._prepare_history(history),
                 np.asarray([x[0] - 1 for x in target], dtype=np.int64),
                 np.asarray([x[1] for x in target], dtype=np.float32))
                for history, target in pairs]

    def _loss(self, pairs):
        return self._loss_prepared(self._prepare_pairs(pairs))

    def _loss_prepared(self, pairs):
        vectors = self.model(*self._pack_prepared([x[0] for x in pairs]))
        logits = vectors @ self.model.items.weight[1:].T
        log_probs = logits.log_softmax(-1)
        width = max(len(x[1]) for x in pairs)
        indices = np.zeros((len(pairs), width), dtype=np.int64)
        weights = np.zeros((len(pairs), width), dtype=np.float32)
        for i, (_, target_ids, target_weights) in enumerate(pairs):
            indices[i, :len(target_ids)] = target_ids
            weights[i, :len(target_weights)] = target_weights
        indices = torch.from_numpy(indices).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)
        # Padding has zero weight, so each row retains its own basket normalization.
        return (-(log_probs.gather(1, indices) * weights).sum(1) / weights.sum(1)).mean()

    def fit(self, resources, interactions, fit_keys, as_of_day, selection_keys=None,
            epochs=20, patience=4, device='auto', batch_size=128, max_prefixes_per_user=8):
        if epochs < 0 or patience < 1 or batch_size < 1 or max_prefixes_per_user < 1:
            raise ValueError('Invalid fit limits')
        started = perf_counter()
        fit_keys = _keys(fit_keys)
        selection_keys = _keys(selection_keys) if selection_keys is not None else set()
        if fit_keys & selection_keys:
            raise ValueError('Fit and selection enrollments must be disjoint')
        self.device = self._device(device)
        catalog = sorted({(str(a), str(b), int(c)) for a, b, c in resources[ITEM].itertuples(index=False, name=None)})
        self.item_to_index = {key: i + 1 for i, key in enumerate(catalog)}
        print(f'  SASRec preparing fit baskets: {len(fit_keys)} enrollments, cutoff={as_of_day}, device={self.device}', flush=True)
        train = self._prefixes(self._baskets(interactions, fit_keys, as_of_day), max_prefixes_per_user)
        print(f'  SASRec preparing selection baskets: {len(selection_keys)} enrollments', flush=True)
        selection = self._prefixes(self._baskets(interactions, selection_keys, as_of_day), max_prefixes_per_user)
        self.metadata = dict(adaptation='OULAD daily-basket SASRec-inspired; not canonical SASRec',
                             as_of_day=float(as_of_day), training_scope='exact fit enrollment keys; date < as_of_day',
                             role_counts={'fit_enrollments': len(fit_keys), 'selection_enrollments': len(selection_keys),
                                          'fit_prefixes': len(train), 'selection_prefixes': len(selection)},
                             max_prefixes_per_user=int(max_prefixes_per_user), prefix_policy='uniformly spaced per enrollment',
                             objective='next-day weighted basket full-catalog cross entropy',
                             epochs=0, best_epoch=0, loss_curve=[], selection_used_for_stopping=bool(selection))
        self.metadata['observed_fit_items'] = sorted({idx for history, target in train
                                                    for basket in [*history, target] for idx, _ in basket})
        self.metadata['train_scope'] = self.metadata['training_scope']
        prepared_train, prepared_selection = self._prepare_pairs(train), self._prepare_pairs(selection)
        self.metadata['execution'] = dict(device=str(self.device), dtype='float32',
                                          packing='cached CPU prefix arrays; batched device transfers',
                                          preprocessing_seconds=perf_counter() - started)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config['seed'])
            self.model = _BasketTransformer(len(catalog), **{k: self.config[k] for k in
                ['max_days', 'dimension', 'heads', 'blocks', 'dropout']}).to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
            rng = np.random.default_rng(self.config['seed'])
            best, stale, best_state = float('inf'), 0, None
            for epoch in range(epochs if train else 0):
                print(f'  SASRec epoch {epoch + 1}/{epochs}: training {len(train)} prefixes', flush=True)
                epoch_started = perf_counter()
                self.model.train()
                total = torch.zeros((), device=self.device)
                order = rng.permutation(len(train))
                last_progress = perf_counter()
                for start in range(0, len(train), batch_size):
                    batch = [prepared_train[i] for i in order[start:start + batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._loss_prepared(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    optimizer.step()
                    total += loss.detach() * len(batch)
                    if perf_counter() - last_progress >= 30:
                        print(f'  SASRec epoch {epoch + 1}/{epochs}: {min(start + batch_size, len(train))}/{len(train)} prefixes', flush=True)
                        last_progress = perf_counter()
                self.model.eval()
                selected_loss = None
                if selection:
                    with torch.inference_mode():
                        selected_loss = float(sum(self._loss_prepared(prepared_selection[s:s + batch_size]) *
                                                  len(prepared_selection[s:s + batch_size])
                                                  for s in range(0, len(selection), batch_size)) / len(selection))
                train_loss = float(total) / len(train)
                monitored = selected_loss if selected_loss is not None else train_loss
                self.metadata['loss_curve'].append({'epoch': epoch + 1, 'train_loss': train_loss,
                                                    'selection_loss': selected_loss,
                                                    'seconds': perf_counter() - epoch_started})
                self.metadata['epochs'] = epoch + 1
                print(f'  SASRec epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, selection_loss={selected_loss}, device={self.device}', flush=True)
                if monitored < best:
                    best, stale, best_state = monitored, 0, deepcopy(self.model.state_dict())
                    self.metadata['best_epoch'] = epoch + 1
                else:
                    stale += 1
                if selection and stale >= patience:
                    break
            if best_state is not None:
                self.model.load_state_dict(best_state)
        self.model.eval().requires_grad_(False)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.metadata['execution']['fit_seconds'] = perf_counter() - started
        return self

    def affinity(self, enrollment, history, available, as_of_day):
        if self.model is None:
            raise RuntimeError('Fit or load the service first')
        if as_of_day < self.metadata['as_of_day']:
            raise ValueError('Inference cutoff precedes the model fit cutoff')
        output = np.zeros(len(available), dtype=float)
        if not len(available) or not self.metadata['epochs']:
            return output
        key = next(iter(_keys([enrollment])))
        days = self._baskets(history, {key}, as_of_day).get(key, [])[-self.config['max_days']:]
        observed = set(self.metadata['observed_fit_items'])
        days = [[entry for entry in basket if entry[0] in observed] for basket in days]
        days = [basket for basket in days if basket]
        if not days:
            return output
        known = [(i, self.item_to_index.get((str(a), str(b), int(c))))
                 for i, (a, b, c) in enumerate(available[ITEM].itertuples(index=False, name=None))]
        with torch.inference_mode():
            vector = self.model(*self._pack([days]))[0]
            warm = [(position, idx) for position, idx in known if idx in observed]
            if warm:
                indices = torch.tensor([idx for _, idx in warm], device=self.device)
                scores = (self.model.items(indices) @ vector).cpu().numpy()
                output[[position for position, _ in warm]] = scores
        return np.nan_to_num(output)

    def affinity_batch(self, requests, as_of_day, batch_size=128):
        """Score (enrollment, scoped history, available catalog) in GPU batches.

        Same frozen FP32 model, cutoffs and warm-item policy as affinity().
        Only the execution batch changes; no observations or candidates drop.
        """
        if self.model is None:
            raise RuntimeError('Fit or load the service first')
        if as_of_day < self.metadata['as_of_day']:
            raise ValueError('Inference cutoff precedes the model fit cutoff')
        if batch_size < 1:
            raise ValueError('Positive batch size required')
        output = [np.zeros(len(available), dtype=float) for _, _, available in requests]
        if not self.metadata['epochs']:
            return output
        observed = set(self.metadata['observed_fit_items'])
        for start in range(0, len(requests), batch_size):
            histories, mappings = [], []
            for offset, (enrollment, history, available) in enumerate(requests[start:start + batch_size], start):
                key = next(iter(_keys([enrollment])))
                days = self._baskets(history, {key}, as_of_day).get(key, [])[-self.config['max_days']:]
                days = [[entry for entry in basket if entry[0] in observed] for basket in days]
                days = [basket for basket in days if basket]
                if not days or available.empty:
                    continue
                indices = [self.item_to_index.get((str(a), str(b), int(c)), 0)
                           for a, b, c in available[ITEM].itertuples(index=False, name=None)]
                histories.append(days)
                mappings.append((offset, indices))
            if not histories:
                continue
            with torch.inference_mode():
                vectors = self.model(*self._pack(histories))
                scores = (vectors @ self.model.items.weight.T).cpu().numpy()
            for row, (offset, indices) in enumerate(mappings):
                warm = [(position, idx) for position, idx in enumerate(indices) if idx in observed]
                if warm:
                    positions, tokens = zip(*warm)
                    output[offset][list(positions)] = scores[row, list(tokens)]
                output[offset] = np.nan_to_num(output[offset])
        return output

    def save(self, path):
        if self.model is None:
            raise RuntimeError('Fit the service first')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format_version': 1, 'config': self.config, 'metadata': self.metadata,
                    'items': [list(k) for k in self.item_to_index],
                    'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['format_version'] != 1:
            raise ValueError('Unsupported sequence artifact version')
        service = cls(**saved['config'])
        service.device = service._device(device)
        service.item_to_index = {tuple(k): i + 1 for i, k in enumerate(saved['items'])}
        service.metadata = saved['metadata']
        with torch.random.fork_rng(devices=[]):
            service.model = _BasketTransformer(len(service.item_to_index), **{k: service.config[k] for k in
                ['max_days', 'dimension', 'heads', 'blocks', 'dropout']})
        service.model.load_state_dict(saved['state_dict'])
        service.model.to(service.device).eval().requires_grad_(False)
        return service
