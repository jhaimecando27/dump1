from algorithms.utils import neighborhood, val, best_admissible_soln


def ts_adaptive_tenure(
    soln_init: list[int], iter_max: int = 100
) -> tuple[list[int], list[int]]:
    n = len(soln_init)
    # Initialize tenure relative to problem size (e.g., half the solution length, with a floor value)
    adaptive_tenure = max(10, int(n * 0.5))

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
            # On improvement, reduce tenure slightly to focus search (but maintain a minimum)
            adaptive_tenure = max(10, int(adaptive_tenure * 0.9))
        else:
            stagnant_ctr += 1
            # On stagnation, increase tenure to encourage exploring new areas
            if stagnant_ctr % 5 == 0:
                adaptive_tenure = int(adaptive_tenure * 1.1)

        soln_curr = nbhr_best
        tabu_list.append(nbhr_best)
        if len(tabu_list) > adaptive_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
