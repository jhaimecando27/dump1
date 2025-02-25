from algorithms.utils import val, best_admissible_soln, neighborhood
from .utils import calculate_phase_shift, update_quantum_state, calculate_quantum_tenure
import math
import numpy as np


# Solution 3: Scalability with quantum-inspired tenure
def ts_adaptive_tenure(
    soln_init: list[int], iter_max: int = 100
) -> tuple[int, list[int]]:
    """
    Enhanced TS with dynamic adaptive tenure that responds to search behavior,
    problem characteristics, and solution quality trends.
    """
    n = len(soln_init)

    # Initialize parameters based on problem size
    base_tenure = max(5, math.floor(n * 0.05))
    min_tenure = max(3, math.floor(n * 0.02))
    max_tenure = min(n // 2, math.floor(n * 0.3))

    # Search state tracking
    tabu_list = []
    soln_curr = soln_init.copy()
    soln_best = soln_init.copy()
    prev_best_value = val(soln_best)
    soln_best_tracker = []
    current_tenure = base_tenure

    # Performance metrics
    stagnation_counter = 0
    improvement_history = []
    improvement_threshold = 0.01
    exploration_mode = False

    # Initialize search intensity parameters
    search_intensity = 0.5  # 0 to 1 scale

    for iter_ctr in range(iter_max):
        # Generate neighborhood with adaptive intensity
        nbhd, moves = neighborhood(soln_curr, tabu_list)

        # Find best admissible solution in neighborhood
        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)
        current_value = val(nbhr_best)

        # Calculate relative improvement
        relative_improvement = 0
        if prev_best_value > 0:
            relative_improvement = (prev_best_value - current_value) / prev_best_value

        # Track significant improvements
        if relative_improvement > improvement_threshold:
            improvement_history.append(relative_improvement)

        # Update best solution if improved
        if current_value < val(soln_best):
            soln_best = nbhr_best.copy()
            soln_best_tracker.append(val(soln_best))
            stagnation_counter = 0

            # Reward success by reducing tenure to intensify search
            if exploration_mode:
                # Gradual reduction when coming from exploration
                current_tenure = max(
                    min_tenure, current_tenure - math.ceil(current_tenure * 0.2)
                )
                exploration_mode = False
            else:
                # Small reduction for continuous improvement
                current_tenure = max(
                    min_tenure,
                    current_tenure - max(1, math.floor(current_tenure * 0.1)),
                )
        else:
            stagnation_counter += 1

            # Manage stagnation with adaptive tenure increases
            if stagnation_counter > 5:
                # Calculate variance in improvement history if available
                variance = 0
                if len(improvement_history) > 3:
                    variance = np.var(improvement_history[-3:]) if np else 0

                # Adjust tenure based on stagnation and improvement variance
                if stagnation_counter > 10 and not exploration_mode:
                    # Transition to exploration mode
                    current_tenure = min(
                        max_tenure, current_tenure + math.ceil(current_tenure * 0.5)
                    )
                    exploration_mode = True
                    search_intensity = min(1.0, search_intensity + 0.2)
                elif variance < 0.001 and len(improvement_history) > 3:
                    # Low variance suggests need for more diverse solutions
                    current_tenure = min(
                        max_tenure, current_tenure + math.ceil(current_tenure * 0.3)
                    )
                else:
                    # Gradual increase
                    current_tenure = min(
                        max_tenure,
                        current_tenure + max(1, math.floor(current_tenure * 0.1)),
                    )

        # Update current solution
        soln_curr = nbhr_best.copy()
        prev_best_value = current_value

        # Probabilistic tabu list update based on search state
        if exploration_mode and stagnation_counter > 15:
            # Radical strategy: reset tabu list occasionally during long stagnation
            if iter_ctr % 20 == 0:
                tabu_list = []
                current_tenure = base_tenure

        # Update tabu list with current move
        tabu_list.append(move_best)

        # Apply current adaptive tenure
        while len(tabu_list) > current_tenure:
            tabu_list.pop(0)

        # Adjust search intensity based on problem phase
        phase = iter_ctr / iter_max
        if phase > 0.75 and stagnation_counter < 3:
            # Intensify search in final phase if making progress
            search_intensity = max(0.2, search_intensity - 0.05)
        elif phase > 0.5 and stagnation_counter > 8:
            # Diversify if stuck in middle phase
            search_intensity = min(1.0, search_intensity + 0.1)

        # Scale tenure with problem size in middle phase
        if 0.3 < phase < 0.7:
            size_factor = math.log(n) / math.log(50)  # Logarithmic scaling
            current_tenure = min(
                max_tenure,
                max(min_tenure, math.floor(current_tenure * (1 + size_factor * 0.1))),
            )

    return val(soln_best), soln_best_tracker
