"""Training-only logistic cue learner for the fixed language environment.

With an intercept and the observable marker direction in {-1,+1}, the model is
saturated: its two fitted probabilities are the two empirical label rates.
"""
import math


def _clip(p, epsilon=1e-7):
    return min(1-epsilon, max(epsilon, p))


class CueLogisticBaseline:
    def fit(self, rows):
        counts = {-1: [0, 0], 1: [0, 0]}
        for row in rows:
            sm = int(row['marker_direction'])
            counts[sm][0] += 1
            counts[sm][1] += int(row['label_first_is_agent'])
        if any(n == 0 for n, _ in counts.values()):
            raise ValueError('Both marker directions are required to fit the cue learner')
        self.probabilities = {sm: positives/n for sm, (n, positives) in counts.items()}
        logits = {sm: math.log(_clip(p)/_clip(1-p)) for sm, p in self.probabilities.items()}
        self.intercept = (logits[1]+logits[-1])/2
        self.marker_coefficient = (logits[1]-logits[-1])/2
        self.counts = {str(sm): {'n': n, 'positive': positives}
                       for sm, (n, positives) in counts.items()}
        return self

    def predict_proba(self, rows):
        if not hasattr(self, 'probabilities'):
            raise RuntimeError('Fit the baseline before prediction')
        return [_clip(self.probabilities[int(row['marker_direction'])]) for row in rows]

    def report(self):
        return {'name': 'cue_logistic', 'fit_split': 'train', 'features': ['intercept', 'marker_direction'],
                'probabilities': {str(k): v for k, v in self.probabilities.items()},
                'intercept': self.intercept, 'marker_coefficient': self.marker_coefficient,
                'counts': self.counts}
