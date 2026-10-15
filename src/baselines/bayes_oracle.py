"""Exact ideal observer; zero-support observations have no defined posterior."""
import math


def check_parameters(ro, rm, so, sm):
    if not all(math.isfinite(r) and 0 <= r <= 1 for r in (ro, rm)):
        raise ValueError('Reliabilities must be finite probabilities')
    if so not in (-1, 1) or sm not in (-1, 1):
        raise ValueError('Cue directions must be -1 or +1')


def posterior_first(ro, rm, so=1, sm=1):
    check_parameters(ro, rm, so, sm)
    masses = [(ro if y == so else 1-ro) * (rm if sm == y else 1-rm) for y in (1, -1)]
    return masses[0] / sum(masses) if sum(masses) else None


def log_odds_posterior(ro, rm, so=1, sm=1):
    check_parameters(ro, rm, so, sm)
    if not (0 < ro < 1 and 0 < rm < 1):
        raise ValueError('Log-odds formula requires interior probabilities')
    z = so * math.log(ro/(1-ro)) + sm * math.log(rm/(1-rm))
    return 1/(1+math.exp(-z))


def enumerate_surface(ro, rm, so, first, second, marked_agent):
    """Enumerate role assignments and noise states, independently of the formula."""
    check_parameters(ro, rm, so, 1)
    if first == second or marked_agent not in (first, second):
        raise ValueError('Invalid observable surface')
    numerator = denominator = 0.0
    for agent, patient in ((first, second), (second, first)):
        for order_correct, po in ((True, ro), (False, 1-ro)):
            y = so if order_correct else -so
            order = (agent, patient) if y == 1 else (patient, agent)
            for mark_correct, pm in ((True, rm), (False, 1-rm)):
                nom = agent if mark_correct else patient
                if order == (first, second) and nom == marked_agent:
                    mass = 0.5 * po * pm
                    denominator += mass
                    numerator += mass * (agent == first)
    return numerator/denominator if denominator else None


def entity_probability(p_first, candidate_order, entity):
    if len(candidate_order) != 2 or len(set(candidate_order)) != 2 or entity not in candidate_order:
        raise ValueError('Expected two distinct candidates containing entity')
    return None if p_first is None else (p_first if candidate_order[0] == entity else 1-p_first)
