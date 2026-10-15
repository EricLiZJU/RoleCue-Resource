"""Fit and evaluate all preregistered non-sequential baselines."""
from src.baselines.cue_logistic import CueLogisticBaseline
from src.baselines.heuristics import (BagOfWordsBaseline, ChanceBaseline, MorphOnlyBaseline,
                                      OrderOnlyBaseline, probabilities_to_logits)
from src.training.eval import paired_diagnostics, prediction_rows, score


def fit_baselines(train_rows, vocabulary, bow_config=None):
    bow_config = bow_config or {}
    return [ChanceBaseline().fit(train_rows), CueLogisticBaseline().fit(train_rows),
            OrderOnlyBaseline().fit(train_rows),
            MorphOnlyBaseline().fit(train_rows),
            BagOfWordsBaseline(vocabulary, **bow_config).fit(train_rows)]


def evaluate_baseline(model, rows, counterfactual_rows=None):
    logits = probabilities_to_logits(model.predict_proba(rows))
    result = {'fit': model.report(), 'test': score(logits, rows),
              'test_prediction_rows': list(prediction_rows(logits, rows))}
    if counterfactual_rows is not None:
        cf_logits = probabilities_to_logits(model.predict_proba(counterfactual_rows))
        result['counterfactual'] = score(cf_logits, counterfactual_rows)
        result['paired_diagnostics'] = paired_diagnostics(cf_logits, counterfactual_rows)
        result['counterfactual_prediction_rows'] = list(prediction_rows(cf_logits, counterfactual_rows))
    return result
