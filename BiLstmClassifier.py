from typing import Any
from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn as nn

DROPOUT = 0.7


class BiLstmFixedLength(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # Word-Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Bi-directional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Linear Layer: Fully connected layer, 3 output features (roles)
        self.linear = nn.Linear(hidden_dim, 3)
        # Dropout 0.2
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


class BiLstmVariableLength(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(DROPOUT)
        # Word-Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Bi-directional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Linear Layer: Fully connected layer, 3 output features (roles)
        self.linear = nn.Linear(hidden_dim, 3)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


class BiLstmGloveVector(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights):
        super().__init__()
        # Word-Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False
        # Bi-directional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Linear Layer: Fully connected layer, 3 output features (roles)
        self.linear = nn.Linear(hidden_dim, 3)
        # Dropout 0.2
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
