from algorithms.utils import neighborhood, val, best_admissible_soln


def tabu_search(
    soln_init: list[int],
    tabu_tenure: int = 20,
    iter_max: int = 100,
) -> tuple[list[int], list[int], list[int]]:

    tabu_list: list = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    for iter_ctr in range(iter_max):
        nbhd: list[list[int]] = neighborhood(soln_curr, tabu_list)
        nbhr_best: list[int] = best_admissible_soln(nbhd, tabu_list)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))
        else:
            stagnant_ctr += 1

        soln_curr = nbhr_best

        # Update Tabu List
        tabu_list.append(nbhr_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(1)

    return val(soln_best), soln_best_tracker
