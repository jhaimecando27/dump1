from algorithms.utils import val, best_admissible_soln, neighborhood
from .utils import dynamic_tenure
import math


def ts_adaptive_tenure(
    soln_init: list[int], iter_max: int = 100
) -> tuple[list[int], list[int]]:
    n = len(soln_init)
    adaptive_tenure = math.floor(n * 0.10)

    tabu_list = []
    soln_curr = soln_init
    soln_best = soln_init

    prev_best_value = val(soln_best)
    conv_rate: int = 0

    stagnant_ctr = 0
    soln_best_tracker = []

    for iter_ctr in range(iter_max):
        nbhd = neighborhood(soln_curr, tabu_list)
        nbhr_best = best_admissible_soln(nbhd, tabu_list)

        current_value = val(nbhr_best)
        if current_value < prev_best_value:
            soln_best = nbhr_best

            improvement = prev_best_value - current_value
            conv_rate = improvement / prev_best_value if prev_best_value else 0
            prev_best_value = current_value

            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))
        else:
            stagnant_ctr += 1
            conv_rate = 0

        soln_curr = nbhr_best

        adaptive_tenure = dynamic_tenure(adaptive_tenure, conv_rate, stagnant_ctr, n)

        tabu_list.append(nbhr_best)
        if len(tabu_list) > adaptive_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
