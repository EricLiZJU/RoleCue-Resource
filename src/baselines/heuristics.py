"""Training-only single-cue and linear bag-of-words baselines."""
import math
import torch
from torch.nn import functional as F


def _clip(p, epsilon=1e-7):
    return min(1-epsilon, max(epsilon, p))


class ChanceBaseline:
    def fit(self, rows):
        if not rows:
            raise ValueError('Cannot fit baseline on empty data')
        return self

    def predict_proba(self, rows):
        return [.5]*len(rows)

    def report(self):
        return {'name': 'chance', 'fit_split': 'train', 'features': [], 'p_first': .5}


class OrderOnlyBaseline:
    """Learn the environment's constant probability that the first NP is Agent."""
    def fit(self, rows):
        self.p_first = sum(int(r['label_first_is_agent']) for r in rows)/len(rows)
        return self

    def predict_proba(self, rows):
        return [_clip(self.p_first)]*len(rows)

    def report(self):
        return {'name': 'order_only', 'fit_split': 'train', 'features': ['environment_position_prior'],
                'p_first': self.p_first}


class MorphOnlyBaseline:
    """Learn a symmetric marker-correctness rate while ignoring position prior."""
    def fit(self, rows):
        self.p_marker_correct = sum(int(r['marker_direction'] == r['true_role_direction'])
                                    for r in rows)/len(rows)
        return self

    def predict_proba(self, rows):
        return [_clip(self.p_marker_correct if r['marker_direction'] == 1
                      else 1-self.p_marker_correct) for r in rows]

    def report(self):
        return {'name': 'morph_only', 'fit_split': 'train', 'features': ['marker_direction'],
                'p_marker_correct': self.p_marker_correct, 'symmetric': True}


class BagOfWordsBaseline:
    """L2-regularized linear logistic regression over unordered token counts."""
    def __init__(self, vocabulary, l2=1e-4, max_iter=50):
        self.vocabulary = dict(vocabulary)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

    def _matrix(self, rows):
        x = torch.zeros((len(rows), len(self.vocabulary)), dtype=torch.float32)
        for i, row in enumerate(rows):
            for token in row['tokens'][:-1]:
                try:
                    x[i, self.vocabulary[token]] += 1
                except KeyError as exc:
                    raise ValueError(f'Unknown token in bag-of-words baseline: {exc}') from exc
        return x

    def fit(self, rows):
        x = self._matrix(rows)
        y = torch.tensor([r['label_first_is_agent'] for r in rows], dtype=torch.float32)
        weight = torch.zeros(len(self.vocabulary), requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=self.max_iter,
                                      tolerance_grad=1e-7, tolerance_change=1e-9,
                                      line_search_fn='strong_wolfe')
        evaluations = 0
        def closure():
            nonlocal evaluations
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(x.mv(weight)+bias, y) + self.l2*weight.square().sum()/2
            loss.backward()
            evaluations += 1
            return loss
        optimizer.step(closure)
        self.weight = weight.detach()
        self.bias = float(bias.detach())
        self.optimizer_evaluations = evaluations
        self.train_nll = float(F.binary_cross_entropy_with_logits(x.mv(self.weight)+self.bias, y))
        return self

    def predict_proba(self, rows):
        logits = self._matrix(rows).mv(self.weight)+self.bias
        return logits.sigmoid().clamp(1e-7, 1-1e-7).tolist()

    def report(self):
        nonzero = {token: float(self.weight[index]) for token, index in self.vocabulary.items()
                   if abs(float(self.weight[index])) > 1e-8}
        return {'name': 'bag_of_words_logistic', 'fit_split': 'train',
                'features': 'unordered token counts excluding constant query token',
                'l2': self.l2, 'max_iter': self.max_iter, 'optimizer': 'torch.LBFGS_strong_wolfe',
                'optimizer_evaluations': self.optimizer_evaluations, 'train_nll': self.train_nll,
                'bias': self.bias, 'weights': nonzero}


def probabilities_to_logits(probabilities):
    return torch.tensor([math.log(_clip(p)/_clip(1-p)) for p in probabilities], dtype=torch.float64)
