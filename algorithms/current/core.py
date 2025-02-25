from algorithms.utils import neighborhood, val, best_admissible_soln


def tabu_search(
    soln_init: list[int],
    tabu_tenure: int = 20,
    iter_max: int = 100,
) -> tuple[list[int], list[int], list[int]]:

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    for iter_ctr in range(iter_max):
        nbhd, moves = neighborhood(soln_curr, tabu_list)
        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))
        else:
            stagnant_ctr += 1

        soln_curr = nbhr_best

        # Update Tabu List
        tabu_list.append(move_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
