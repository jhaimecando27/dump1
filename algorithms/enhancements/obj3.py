from algorithms.utils import val, best_admissible_soln, neighborhood
from .utils import (
    calculate_phase_shift,
    update_quantum_state,
    calculate_quantum_tenure,
)
import math
import numpy as np
import random


# Solution 3: Scalability with quantum-inspired tenure
def ts_adaptive_tenure(
    soln_init: list[int], iter_max: int = 100
) -> tuple[list[int], list[int]]:
    n = len(soln_init)
    quantum_state = 0.5
    base_tenure = math.floor(n * 0.50)
    max_stagnant = 5  # Lower threshold for faster response
    tenure_reset_factor = 0.5  # Reset tenure to a fraction of its current value

    tabu_list = []
    soln_curr = soln_init
    soln_best = soln_init
    prev_best_value = val(soln_best)
    soln_best_tracker = []
    phase_shift = 0.0
    stagnant_ctr = 0  # Track stagnation

    for iter_ctr in range(iter_max):
        nbhd, moves = neighborhood(soln_curr, tabu_list[:base_tenure])
        nbhr_best, move_best = best_admissible_soln(
            nbhd, moves, tabu_list[:base_tenure], soln_best
        )

        current_value = val(nbhr_best)
        if current_value < prev_best_value:
            soln_best = nbhr_best
            improvement = prev_best_value - current_value
            phase_shift = calculate_phase_shift(improvement, prev_best_value)
            prev_best_value = current_value
            soln_best_tracker.append(val(soln_best))
            stagnant_ctr = 0
        else:
            stagnant_ctr += 1
            # Aggressive tenure adjustment during stagnation
            if stagnant_ctr > max_stagnant:
                phase_shift = -1.0  # Force tenure increase
                adaptive_tenure = max(
                    min_tenure, int(adaptive_tenure * tenure_reset_factor)
                )

        soln_curr = nbhr_best

        # Update quantum state with progress factor
        progress = iter_ctr / iter_max
        quantum_state = update_quantum_state(quantum_state, phase_shift, progress)

        # Calculate tenure with wider bounds and randomness
        adaptive_tenure = calculate_quantum_tenure(base_tenure, quantum_state, n)

        # Enforce canonical move format
        move_best = tuple(sorted(move_best))
        tabu_list.append(move_best)
        while len(tabu_list) > adaptive_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker
