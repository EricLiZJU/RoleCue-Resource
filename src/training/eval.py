"""Evaluation of frozen probabilities, with oracle metadata confined to scoring."""
from collections import defaultdict
import math
import torch
from torch.nn import functional as F
from src.training.datamodule import RoleDataset, make_loader


@torch.inference_mode()
def predict(model, rows, vocabulary, device, batch_size=256):
    model.eval()
    loader = make_loader(RoleDataset(rows, vocabulary), batch_size)
    outputs = []
    for token_ids, _, lengths in loader:
        outputs.append(model(token_ids.to(device), lengths.to(device)).cpu())
    return torch.cat(outputs)


def score(logits, rows):
    logits = logits.detach().double().cpu()
    binding = rows[0].get('tokenizer') == 'rolecue_binding_v1'
    if any((r.get('tokenizer') == 'rolecue_binding_v1') != binding for r in rows):
        raise ValueError('Cannot mix Core and Binding scoring rows')
    y = torch.tensor([r['label_query_is_agent'] if binding else r['label_first_is_agent']
                      for r in rows], dtype=torch.double)
    p = logits.sigmoid()
    q = torch.tensor([r['oracle_p_query_agent'] if binding else r['oracle']['evaluation_p_first_agent']
                      for r in rows], dtype=torch.double)
    qtrue = torch.where(y.bool(), q, 1-q)
    if torch.any(qtrue <= 0):
        raise ValueError('Evaluation oracle cannot assign zero mass to an observed IID truth')
    oracle_scores = torch.where(q == .5, .5, ((q > .5) == y.bool()).double())
    nll = F.binary_cross_entropy_with_logits(logits, y).item()
    oracle_nll = -qtrue.log().mean().item()
    eps = torch.finfo(torch.double).eps
    safe_p = p.clamp(eps, 1-eps)
    posterior_kl = torch.where(q > 0, q*(q.clamp_min(eps).log()-safe_p.log()), 0.) + \
                   torch.where(q < 1, (1-q)*((1-q).clamp_min(eps).log()-(1-safe_p).log()), 0.)
    result = {'n': len(rows), 'accuracy': ((p >= .5) == y.bool()).double().mean().item(),
              'nll': nll, 'brier': ((p-y)**2).mean().item(),
              'oracle_accuracy': oracle_scores.mean().item(), 'oracle_nll': oracle_nll,
              'oracle_nll_regret': nll-oracle_nll, 'oracle_probability_mae': (p-q).abs().mean().item(),
              'oracle_probability_signed_bias': (p-q).mean().item(),
              'oracle_posterior_kl': posterior_kl.mean().item()}
    probability_name = 'mean_p_query_agent' if binding else 'mean_p_first_agent'
    result[probability_name] = p.mean().item()
    for name, mask in (
            ('conflict', torch.tensor([r['cue_conflict'] for r in rows], dtype=torch.bool)),
            ('aligned', torch.tensor([not r['cue_conflict'] for r in rows], dtype=torch.bool))):
        if mask.any():
            result[name] = {'n': mask.sum().item(),
                            'accuracy': ((p[mask] >= .5) == y[mask].bool()).double().mean().item(),
                            'oracle_probability_mae': (p[mask]-q[mask]).abs().mean().item(),
                            'oracle_probability_signed_bias': (p[mask]-q[mask]).mean().item(),
                            'oracle_posterior_kl': posterior_kl[mask].mean().item(),
                            probability_name: p[mask].mean().item()}
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError('Non-finite evaluation metric')
    return result


def paired_diagnostics(logits, rows):
    """Entity-aligned four-cell intervention effects from a complete pair set."""
    if len(logits) != len(rows):
        raise ValueError('Prediction and counterfactual row counts differ')
    groups, oracle_groups = defaultdict(dict), defaultdict(dict)
    deviations = []
    for logit, row in zip(logits.detach().double().cpu().tolist(), rows):
        p_first = torch.sigmoid(torch.tensor(logit, dtype=torch.float64)).item()
        aligned_first = row['candidate_order'][0] == row['alignment_entity']
        p_entity = p_first if aligned_first else 1-p_first
        q_first = row['oracle']['training_p_first_agent']
        q_entity = q_first if aligned_first else 1-q_first
        groups[row['pair_id']][row['variant']] = p_entity
        oracle_groups[row['pair_id']][row['variant']] = q_entity
        deviations.append(abs(p_entity-q_entity))
    variants = ('reference', 'order_swap', 'marker_flip', 'both')
    if not groups or any(set(v) != set(variants) for v in groups.values()):
        raise ValueError('Incomplete counterfactual group')
    def effects(values):
        return (values['order_swap']-values['reference'],
                values['marker_flip']-values['reference'],
                .5*((values['order_swap']-values['reference'])+(values['both']-values['marker_flip'])),
                .5*((values['marker_flip']-values['reference'])+(values['both']-values['order_swap'])),
                values['both']-values['order_swap']-values['marker_flip']+values['reference'])
    model_effects = [effects(v) for v in groups.values()]
    oracle_effects = [effects(v) for v in oracle_groups.values()]
    names = ('anchored_order_effect', 'anchored_marking_effect', 'factorial_order_effect',
             'factorial_marking_effect', 'interaction')
    return {'groups': len(groups),
            'variant_mean_p_alignment_entity': {v: sum(g[v] for g in groups.values())/len(groups) for v in variants},
            'training_oracle_probability_mae': sum(deviations)/len(deviations),
            **{name: sum(e[i] for e in model_effects)/len(model_effects) for i, name in enumerate(names)},
            **{'oracle_'+name: sum(e[i] for e in oracle_effects)/len(oracle_effects)
               for i, name in enumerate(names)}}


def prediction_rows(logits, rows):
    for logit, record in zip(logits.tolist(), rows):
        p = torch.sigmoid(torch.tensor(logit, dtype=torch.float64)).item()
        row = {'record_id': record['record_id'], 'event_id': record['event_id'],
               'pair_id': record['pair_id'], 'variant': record['variant'],
               'candidate_order': record['candidate_order'], 'p_first_agent': p,
               'p_alignment_entity_agent': p if record['candidate_order'][0] == record['alignment_entity'] else 1-p,
               'target_first_agent': record['label_first_is_agent'],
               'oracle': record['oracle'], 'cue_conflict': record['cue_conflict']}
        for key in ('structure_cell', 'distances', 'competitor_entities'):
            if key in record:
                row[key] = record[key]
        yield row
