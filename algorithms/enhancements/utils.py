import numpy as np
import math
import random
from algorithms.utils import val


def wave_perturb(
    solution: list[int], amplitude: float, frequency: float, stagnation: int
) -> list[int]:
    """
    Enhanced wave perturbation with adaptive intensity and selective disruption
    """
    import math

    perturbed = solution.copy()
    n = len(perturbed)

    # Dynamic parameters based on problem size and stagnation
    base_intensity = min(0.3, 0.1 + (n / 1000))  # Scales with problem size
    stagnation_factor = min(0.5, stagnation / 20)  # Caps at 0.5
    intensity = base_intensity + stagnation_factor

    # Modulated wave parameters
    phase_shift = (stagnation % 6) * math.pi / 3
    wave_complexity = 1 + (
        stagnation // 10
    )  # Increases wave complexity with stagnation

    # Apply selective perturbation
    for i in range(n - 1):
        # Complex wave pattern
        wave = 0
        for harmonic in range(wave_complexity):
            wave += (amplitude / (harmonic + 1)) * math.sin(
                frequency * (harmonic + 1) * i * 2 * math.pi / n + phase_shift
            )

        if abs(wave) > 0.2 * intensity:  # Threshold scales with intensity
            swap_distance = max(1, int(abs(wave * n * intensity / 3)))
            swap_idx = (i + swap_distance) % (n - 1)
            if i != swap_idx:
                perturbed[i], perturbed[swap_idx] = perturbed[swap_idx], perturbed[i]

    # Maintain cycle closure
    perturbed[-1] = perturbed[0]
    return perturbed


def adaptive_perturb(
    soln: list[int],
    stagnant_ctr: int,
    base_strength: float = 0.1,
    max_strength: float = 0.7,
) -> list[int]:
    """
    Enhanced adaptive perturbation function that applies different perturbation types
    based on the level of stagnation.

    Args:
        soln: The current solution (list of integers).
        stagnant_ctr: The current count of consecutive non-improving iterations.
        base_strength: The base fraction of the solution to perturb.
        max_strength: The maximum fraction for moderate perturbation.

    Returns:
        A new solution after perturbation.
    """
    soln_new = soln.copy()
    n = len(soln_new)

    # If stagnation is severe, use a ruin-and-recreate strategy.
    # Ruin-and-recreate: remove a proportion (e.g., 20%) of elements and reinsert them randomly.
    num_remove = max(1, int(n * 0.2))
    indices = np.random.choice(n, size=num_remove, replace=False)
    remaining = [soln_new[i] for i in range(n) if i not in indices]
    removed = [soln_new[i] for i in indices]
    np.random.shuffle(removed)
    for elem in removed:
        pos = np.random.randint(0, len(remaining) + 1)
        remaining.insert(pos, elem)
    return remaining


def neighborhood(
    soln: list[int], tabu_list: list[tuple], intensity: float = 1.0
) -> tuple[list[list[int]], list[tuple]]:
    """Generates neighborhood of new solution from selected solution by
    making small changes.
    Args:
        soln: The current solution passed
        tabu_list: List of recent moves (tuples specifying move type and indices)
        intensity: Parameter to control neighborhood size (higher = more moves)
    Returns:
        nbhd: list of new solutions
        moves: list of moves that generated each solution
    """
    nbhd: list = []
    moves: list = []  # Store the moves that generated each neighbor
    n = len(soln) - 1  # Exclude last element as it should match first

    # 2-opt moves
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Check if this 2-opt move is in the tabu list
            # 2-opt moves are stored as ('2opt', i, j)
            if ("2opt", i, j) not in tabu_list and ("2opt", j, i) not in tabu_list:
                soln_mod: list[int] = soln.copy()
                soln_mod[i : j + 1] = reversed(soln_mod[i : j + 1])  # Fixed the slicing
                soln_mod[-1] = soln_mod[0]  # Ensure last element matches first
                nbhd.append(soln_mod)
                moves.append(("2opt", i, j))  # Store the move type and indices

    # Insertion moves (if intensity is high enough)
    if intensity > 0.5:
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if this insertion move is in the tabu list
                    # Insertion moves are stored as ('ins', i, j)
                    if ("ins", i, j) not in tabu_list:
                        soln_mod: list[int] = soln.copy()
                        value = soln_mod.pop(i)
                        soln_mod.insert(j, value)
                        soln_mod[-1] = soln_mod[0]  # Ensure last element matches first
                        nbhd.append(soln_mod)
                        moves.append(("ins", i, j))  # Store the move type and indices

    return nbhd, moves


def update_quantum_state(state: float, phase: float, progress: float) -> float:
    """
    Updates quantum state using interference patterns.
    """
    # Quantum walk-inspired state update
    new_state = state + phase * (1.0 - progress)  # Amplify effect early
    return max(0.1, min(0.9, new_state))


def calculate_phase_shift(improvement: float, prev_value: float) -> float:
    """
    Calculates phase shift based on improvement magnitude.
    """
    relative_improvement = improvement / prev_value if prev_value else 0
    return math.tanh(relative_improvement)  # Bounded between -1 and 1


def calculate_quantum_tenure(base: int, state: float, n: int) -> int:
    min_tenure = max(5, math.floor(n * 0.15))  # Higher minimum
    max_tenure = math.floor(n * 0.6)  # Wider maximum
    # Add randomness to tenure calculation
    tenure = min_tenure + math.floor(
        (max_tenure - min_tenure) * (state + 0.1 * random.random())
    )
    return tenure


def estimate_improvement_potential(nbhd: list[list[int]], current_val: int) -> float:
    """
    Estimates potential for further improvement based on neighborhood structure.
    """
    if not nbhd:
        return 0

    values = [val(n) for n in nbhd]
    min_val = min(values)
    improvement = current_val - min_val if min_val < current_val else 0
    return improvement / current_val if current_val else 0
