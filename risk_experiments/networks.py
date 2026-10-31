from __future__ import annotations

import math
import torch
from torch import nn
from models.pred_oulad import BiLSTMAttentionNet
from models.oulad_hybrid_attention import PaperHybridAttentionNet

CLASSICAL = ('lr', 'rf', 'svm', 'xgboost', 'lightgbm', 'mlp')
NEURAL = ('proposed', 'lstm', 'bilstm', 'transformer', 'no_attention', 'no_static', 'no_cox', 'paper_hybrid')
MODELS = CLASSICAL + NEURAL
COX = {'proposed', 'no_attention', 'no_static'}


class MainModel(BiLSTMAttentionNet):
    """Same main architecture as historical 0.87497; isolated ablations."""
    def __init__(self, features, static, name):
        super().__init__(features, static)
        self.variant = name
        if name == 'no_static':
            self.static_mlp = nn.Identity()
            self.mlp = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.3))
        if name == 'no_attention':
            self.attn_proj = nn.Identity()
            self.register_parameter('attn_query', None)

    def forward(self, x, s, weekly=None):
        out, _ = self.lstm(x)
        if self.variant == 'no_attention':
            context = out.mean(dim=1)
        else:
            score = (torch.tanh(self.attn_proj(out)) * self.attn_query).sum(dim=-1)
            context = (torch.softmax(score, dim=1).unsqueeze(-1) * out).sum(dim=1)
        if self.variant != 'no_static':
            context = torch.cat([context, self.static_mlp(s)], dim=1)
        h = self.mlp(context)
        return self.cls(h).squeeze(-1), self.risk(h).squeeze(-1)


class SequenceBaseline(nn.Module):
    def __init__(self, features, static, name, days):
        super().__init__()
        self.name = name
        self.static = nn.Sequential(nn.Linear(static, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                    nn.Dropout(.4), nn.Linear(128, 64), nn.ReLU())
        if name == 'transformer':
            self.input = nn.Linear(features, 128)
            layer = nn.TransformerEncoderLayer(128, 4, 256, .3, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            position = torch.arange(days).unsqueeze(1)
            rates = torch.exp(torch.arange(0, 128, 2) * (-math.log(10000.) / 128))
            pe = torch.zeros(days, 128)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(position * rates), torch.cos(position * rates)
            self.register_buffer('position', pe)
            width = 128
        else:
            self.encoder = nn.LSTM(features, 128, 2, dropout=.3, batch_first=True,
                                   bidirectional=name == 'bilstm')
            width = 256 if name == 'bilstm' else 128
        self.head = nn.Sequential(nn.Linear(width + 64, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1))

    def forward(self, x, s, weekly=None):
        if self.name == 'transformer':
            h = self.encoder(self.input(x) + self.position[None]).mean(dim=1)
        else:
            _, (hidden, _) = self.encoder(x)
            h = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.name == 'bilstm' else hidden[-1]
        z = self.head(torch.cat([h, self.static(s)], dim=1)).squeeze(-1)
        return z, None


class PaperAdaptation(PaperHybridAttentionNet):
    """Original 2x64 weekly-total architecture; retrained on shared risk labels."""
    def forward(self, x, s, weekly):
        logits = super().forward(s, weekly)
        return logits[:, 1] - logits[:, 0], None


def make_model(name, features, static, days):
    if name in ('proposed', 'no_attention', 'no_static', 'no_cox'):
        return MainModel(features, static, name)
    if name == 'paper_hybrid':
        return PaperAdaptation(static, (days + 6) // 7)
    if name in ('lstm', 'bilstm', 'transformer'):
        return SequenceBaseline(features, static, name, days)
    raise ValueError(name)


def breslow_loss(risk, time, event):
    """Correct tied event denominators; risk sets are within each minibatch."""
    order = torch.argsort(time, descending=True, stable=True)
    r, t, e = risk[order], time[order], event[order]
    _, counts = torch.unique_consecutive(t, return_counts=True)
    ends = counts.cumsum(0) - 1
    logden = torch.logcumsumexp(r, 0)[ends].repeat_interleave(counts)
    return -((r - logden) * e).sum() / e.sum().clamp(min=1)
