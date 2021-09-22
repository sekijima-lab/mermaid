import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


class RolloutNetwork(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dr_rate=0.2, fr_len=25):
        super(RolloutNetwork, self).__init__()
        self.emb_fr = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm_fr = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dr_rate,
                               batch_first=True)
        self.header = nn.Linear(hidden_dim, vocab_size)
        self.fr_len = fr_len

    def forward(self, fr: torch.tensor, fr_len):
        h_fr = self.emb_fr(fr)
        h_fr = torch.nn.utils.rnn.pack_padded_sequence(h_fr, fr_len, batch_first=True, enforce_sorted=False)
        h_fr, _ = self.lstm_fr(h_fr)
        h_fr, _ = torch.nn.utils.rnn.pad_packed_sequence(h_fr, batch_first=True, total_length=self.fr_len)

        h = h_fr.contiguous().view(-1, h_fr.shape[2])
        y = self.header(h)
        y = y.contiguous().view(h_fr.shape[0], h_fr.shape[1], y.shape[1])

        return y


