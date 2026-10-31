"""Architectures aligned with Frontiers AI 9:1811886."""
import torch
import torch.nn as nn


class StaticBaseline(nn.Module):
    """Static-only MLP baseline (smaller than the static branch inside Hybrid).

    Compact 64->32 head matches the article's reported Static Vector
    performance (weighted F1 ~ 0.8167). The 128->64 stack from the Hybrid's
    static branch only makes sense fused with the temporal context.
    """

    def __init__(self, static_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x_s, x_d):
        return self.net(x_s)


class DynamicBaseline(nn.Module):
    """Single-layer LSTM over weekly clickstream (behavioral baseline)."""

    def __init__(self, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x_s, x_d):
        x_d = x_d.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x_d)
        return self.fc(h_n[-1])


class HybridModel(nn.Module):
    """Bi-LSTM (2x64) + Bahdanau-style attention + static MLP -> classifier.

    Article attention:
        e_t = tanh(W_a * h_t + b_a)
        alpha_t = softmax(e_t^T * u_a)
        c       = sum_t alpha_t * h_t
    """

    ATTN_DIM = 64

    def __init__(self, static_dim, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)

        self.W_a = nn.Linear(128, self.ATTN_DIM)
        self.u_a = nn.Parameter(torch.randn(self.ATTN_DIM) * 0.1)

        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_final = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def _attend(self, lstm_out):
        e = torch.tanh(self.W_a(lstm_out))                # [B,T,A]
        scores = (e * self.u_a).sum(dim=-1, keepdim=True) # [B,T,1]
        alpha = torch.softmax(scores, dim=1)              # [B,T,1]
        context = torch.sum(alpha * lstm_out, dim=1)      # [B,128]
        return context, alpha

    def forward(self, x_s, x_d, return_embedding=False):
        x_d = x_d.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_d)
        context, _ = self._attend(lstm_out)
        static_out = self.static_net(x_s)
        combined = torch.cat((context, static_out), dim=1)
        if return_embedding:
            return combined
        return self.fc_final(combined)

    def attention_weights(self, x_d):
        x_d = x_d.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_d)
        _, alpha = self._attend(lstm_out)
        return alpha.squeeze(-1)
