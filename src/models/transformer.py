"""Causal Transformer with learned or length-extrapolatable sinusoidal positions."""
import math
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, heads=4, layers=2,
                 feedforward=672, max_length=16, dropout=0.1,
                 position_encoding='learned'):
        super().__init__()
        self.max_length = max_length
        self.position_encoding = position_encoding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if position_encoding == 'learned':
            self.positions = nn.Embedding(max_length, embedding_dim)
        elif position_encoding == 'sinusoidal':
            self.positions = None
        else:
            raise ValueError('position_encoding must be learned or sinusoidal')
        # Construct each layer separately: no identical cloned initializations.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, heads, feedforward, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(embedding_dim, 1)

    def _positions(self, length, device, dtype):
        if self.positions is not None:
            return self.positions(torch.arange(length, device=device))
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.embedding.embedding_dim, 2, device=device, dtype=dtype)
                        * (-math.log(10000.0)/self.embedding.embedding_dim))
        values = torch.zeros(length, self.embedding.embedding_dim, device=device, dtype=dtype)
        values[:, 0::2] = torch.sin(position*div)
        values[:, 1::2] = torch.cos(position*div[:values[:, 1::2].shape[1]])
        return values

    def encode(self, token_ids, lengths=None):
        length = token_ids.shape[1]
        if length > self.max_length:
            raise ValueError('Sequence exceeds configured context; truncation is forbidden')
        embedded = self.embedding(token_ids)
        hidden = embedded + self._positions(length, token_ids.device, embedded.dtype)
        mask = torch.ones(length, length, dtype=torch.bool, device=token_ids.device).triu(1)
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(length, device=token_ids.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask, src_key_padding_mask=padding_mask)
        return self.norm(hidden)

    def forward(self, token_ids, lengths=None):
        encoded = self.encode(token_ids, lengths)
        if lengths is None:
            final = encoded[:, -1]
        else:
            index = (lengths-1).view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
            final = encoded.gather(1, index).squeeze(1)
        return self.readout(self.dropout(final)).squeeze(-1)
