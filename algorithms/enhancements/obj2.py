import config
import concurrent.futures
from functools import lru_cache
from algorithms.utils import (
    neighborhood,
    prob_neighborhood,
    p2_neighborhood,
    new_neighborhood,
)


def tabu_search(
    soln_init: list[int],
    tabu_tenure: int,
    iter_max: int = 100,
) -> tuple[int, list[int]]:
    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init.copy()
    soln_best: list[int] = soln_init.copy()
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    # Create a persistent process pool to avoid repeated overhead.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for iter_ctr in range(iter_max):
            nbhd, moves = new_neighborhood(soln_curr, tabu_list[:tabu_tenure])
            nbhr_best, move_best = best_admissible_soln(
                nbhd, moves, tabu_list[:tabu_tenure], soln_best, executor
            )

            # Fallback if no candidate meets the aspiration criteria.
            if nbhr_best is None:
                nbhr_best = nbhd[0]
                move_best = moves[0]

            curr_val = val(nbhr_best)
            if curr_val < val(soln_best):
                soln_best = nbhr_best
                stagnant_ctr = 0
                soln_best_tracker.append(curr_val)
            else:
                stagnant_ctr += 1

            soln_curr = nbhr_best

            # Update Tabu List
            tabu_list.append(move_best)
            while len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def val(soln: list[int]) -> int:
    """
    Evaluates the route's total cost.
    Minor micro-optimizations: local variable lookup.
    """
    total = 0
    n = len(soln)
    dms = config.dms[str(n)]  # Assuming config.dms is a dict-of-dicts.
    for i in range(n):
        total += dms[soln[i]][soln[(i + 1) % n]]
    return total


def evaluate_candidate(
    args: tuple[list[int], tuple[int, int], list[tuple[int, int]], int],
) -> tuple[int, list[int], tuple[int, int]] | None:
    """
    Evaluates one candidate solution.
    Returns (candidate_value, candidate_solution, move) if it improves upon the current best,
    else returns None.
    """
    candidate, move, tabu_list, soln_best_value = args
    candidate_value = val(candidate)
    if candidate_value < soln_best_value and move not in tabu_list:
        return candidate_value, candidate, move
    return None


def best_admissible_soln(
    nbhd: list[list[int]],
    moves: list[tuple[int, int]],
    tabu_list: list[tuple[int, int]],
    soln_best: list[int],
    executor: concurrent.futures.ProcessPoolExecutor,
) -> tuple[list[int] | None, tuple[int, int] | None]:
    """
    Evaluates all candidate neighbors in parallel using the provided executor.
    Returns the best candidate that is admissible.
    """
    soln_best_value = val(soln_best)
    args_list = [
        (candidate, move, tabu_list, soln_best_value)
        for candidate, move in zip(nbhd, moves)
    ]
    best_candidate = None
    best_value = float("inf")
    best_move = None

    # Use a chunksize to reduce overhead when mapping a large list.
    results = executor.map(evaluate_candidate, args_list, chunksize=10)
    for result in results:
        if result is not None:
            candidate_value, candidate, move = result
            if candidate_value < best_value:
                best_value = candidate_value
                best_candidate = candidate
                best_move = move

    return best_candidate, best_move
