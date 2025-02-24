from .utils import adaptive_perturb
from algorithms.utils import neighborhood, val, best_admissible_soln


def tabu_search_with_perturbation(
    soln_init: list[int],
    tabu_tenure: int = 20,
    iter_max: int = 100,
) -> tuple[list[int], list[int]]:

    tabu_list = []
    soln_curr = soln_init
    soln_best = soln_init
    stagnant_ctr = 0
    soln_best_tracker = []

    for iter_ctr in range(iter_max):
        nbhd = neighborhood(soln_curr, tabu_list)
        nbhr_best = best_admissible_soln(nbhd, tabu_list)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))
        else:
            stagnant_ctr += 1

        # Trigger adaptive perturbation after 10 stagnant iterations
        if stagnant_ctr >= 10:
            soln_curr = adaptive_perturb(soln_best, stagnant_ctr)
            stagnant_ctr = 0  # reset counter after perturbation
            continue  # skip tabu update for this iteration

        soln_curr = nbhr_best
        tabu_list.append(nbhr_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
