from .utils import adaptive_perturb, neighborhood, wave_perturb
from algorithms.utils import val, best_admissible_soln


def tabu_search_with_perturbation(
    soln_init: list[int],
    tabu_tenure: int = 20,
    iter_max: int = 100,
) -> tuple[list[int], list[int]]:

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    wave_amplitude = 0.15
    wave_frequency = 1.0
    improvement_threshold = 0.05

    val_prev = val(soln_best)

    for iter_ctr in range(iter_max):
        nbhd, moves = neighborhood(soln_curr, tabu_list)
        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)
        val_curr = val(nbhr_best)

        rel_improvement = (val_prev - val_curr) / val_prev if val_prev > 0 else 0

        if val_curr < val(soln_best):
            soln_best = nbhr_best
            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))

            # Adjust wave parameters based on improvement
            if rel_improvement > improvement_threshold:
                wave_amplitude = max(0.05, wave_amplitude - 0.02)  # Reduce disruption
                wave_frequency = max(1.0, wave_frequency - 0.2)  # Slow down wave
            else:
                wave_amplitude = min(0.25, wave_amplitude + 0.01)  # Slight increase
                wave_frequency = min(2.0, wave_frequency + 0.1)  # Speed up wave
        else:
            stagnant_ctr += 1
            if stagnant_ctr > 5:
                wave_amplitude = min(0.3, wave_amplitude + 0.02)
                wave_frequency = min(3.0, wave_frequency + 0.2)

        # Adaptive perturbation threshold
        perturbation_threshold = 6 + int(len(soln_init) / 40)

        if stagnant_ctr >= perturbation_threshold:
            soln_curr = wave_perturb(
                soln_best,
                wave_amplitude,
                wave_frequency,
                stagnant_ctr
            )
            stagnant_ctr = 0
            continue

        soln_curr = nbhr_best
        val_prev = val_curr

        tabu_list.append(move_best)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
