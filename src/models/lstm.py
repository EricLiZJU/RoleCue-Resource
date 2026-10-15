"""Unidirectional LSTM with the same final-query objective."""
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=320, layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.recurrent = nn.LSTM(embedding_dim, hidden_size, layers, batch_first=True,
                                 dropout=dropout if layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_size, 1)

    def encode(self, token_ids):
        return self.recurrent(self.embedding(token_ids))[0]

    def forward(self, token_ids, lengths=None):
        encoded = self.encode(token_ids)
        if lengths is None:
            final = encoded[:, -1]
        else:
            index = (lengths-1).view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
            final = encoded.gather(1, index).squeeze(1)
        return self.readout(self.dropout(final)).squeeze(-1)
