"""Matched-budget causal sequence classifiers."""
from src.models.rnn import RNNClassifier
from src.models.lstm import LSTMClassifier
from src.models.transformer import TransformerClassifier


def build_model(architecture, vocab_size, config):
    classes = {'rnn': RNNClassifier, 'lstm': LSTMClassifier, 'transformer': TransformerClassifier}
    return classes[architecture](vocab_size=vocab_size, **config)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
