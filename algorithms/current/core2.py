import random
import config
from algorithms.utils import (
    val,
    prob_neighborhood,
    best_admissible_soln,
)


def tabu_search(
    soln_init: list[int],
    tabu_tenure,
    iter_max: int = 100,
) -> tuple[int, list[int]]:
    """
    Modified Tabu Search procedure.
    The solution is maintained as a list of unique POI indices (non-cyclic).
    Cyclicity is enforced in cost calculations via modulo arithmetic.
    """
    n_size = int(len(soln_init) * 0.1)
    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init[:]  # make a copy
    soln_best: list[int] = soln_init[:]
    soln_best_tracker: list[int] = []

    for iter_ctr in range(iter_max):
        nbhd, moves = neighborhood(soln_curr, tabu_list[:tabu_tenure], n_size)
        nbhr_best, move_best = best_admissible_soln(
            nbhd, moves, tabu_list[:tabu_tenure], soln_best
        )

        if nbhr_best is None:
            # Regenerate neighborhood if no admissible move is found.
            nbhd, moves = prob_neighborhood(soln_curr, tabu_list[:tabu_tenure])
            nbhr_best, move_best = best_admissible_soln(
                nbhd, moves, tabu_list[:tabu_tenure], soln_best
            )

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            soln_best_tracker.append(val(soln_best))
        soln_curr = nbhr_best

        # Update Tabu List (FIFO)
        tabu_list.append(move_best)
        while len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def neighborhood(
    soln: list[int], tabu_list: list[tuple[int, int]], n_size: int
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """
    Modified neighborhood generator.
    Instead of generating all possible swaps (which may be large),
    this version samples N_SIZE candidate moves based on a probabilistic measure.

    The probabilities are computed using a "local cost" for each index:
    cost = distance(prev, current) + distance(current, next),
    where distances are looked up using config.dms.

    Note: The solution is assumed to be a list of unique POI indices. The cyclic tour cost
    is computed by using (i+1)%n in the evaluation.
    """
    nbhd = []
    moves = []
    n = len(soln) - 1  # number of unique points; should match dataset size (e.g., 20)

    # Compute local cost for each index using modulo arithmetic
    local_costs = []
    for i in range(n):
        prev = soln[i - 1] if i > 0 else soln[-1]
        curr = soln[i]
        nxt = soln[(i + 1) % n]
        cost = config.dms[str(n + 1)][prev][curr] + config.dms[str(n + 1)][curr][nxt]
        local_costs.append(cost)

    total_cost = sum(local_costs)
    if total_cost == 0:
        probs = [1 / n for _ in range(n)]
    else:
        probs = [cost / total_cost for cost in local_costs]

    # Generate up to N_SIZE candidate moves
    for _ in range(n_size):
        i = random.choices(range(n), weights=probs, k=1)[0]
        # Choose a different index j
        j_candidates = [x for x in range(n) if x != i]
        j = random.choice(j_candidates)
        move = (min(soln[i], soln[j]), max(soln[i], soln[j]))
        if move in tabu_list:
            continue
        soln_mod = soln.copy()
        soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
        # Do not append the duplicate first element; keep soln_mod of length n.
        nbhd.append(soln_mod)
        moves.append(move)
        if len(nbhd) >= n_size:
            break
    return nbhd, moves
