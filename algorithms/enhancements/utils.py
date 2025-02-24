import numpy as np
import math


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


def dynamic_tenure(tabu_tenure, convergence_rate, stagnant_ctr, poi_len):

    baseline_tenure = math.floor(poi_len * 0.10)

    # stagnation_threshold = math.ceil(len(soln_init) * 0.10)  # Dynamic threshold
    stagnation_threshold = 5

    adjustment = calculate_adjustment(convergence_rate)

    # Reset
    if stagnant_ctr == 0:
        return baseline_tenure

    # No improvement
    elif stagnant_ctr > 0 and stagnant_ctr < stagnation_threshold:
        return min(
            math.floor(poi_len / 2),
            tabu_tenure + adjustment,
        )

    # Continues no improvement
    elif stagnant_ctr >= stagnation_threshold:
        return min(
            math.floor(poi_len / 2),
            tabu_tenure + adjustment + int(stagnant_ctr / 2),
        )


def calculate_adjustment(convergence_rate: float) -> int:
    if convergence_rate >= 0.9:
        return 2
    else:
        return 1


def neighborhood(soln: list[int], tabu_list: list[list[int]], intensity: float = 1.0) -> list[list[int]]:
    """Generates neighborhood of new solution from selected solution by
    making small changes.
    Args:
        soln: The current solution passed
        tabu_list: List of recent solutions
    Returns:
        nbhd: list of new solutions
    Raises:
    """
    nbhd: list = []
    n = len(soln) - 1  # Exclude last element as it should match first

    # 2-opt
    for i in range(n - 1):
        for j in range(i + 1, n):
            soln_mod: list[int] = soln.copy()
            soln_mod[i:j] = reversed(soln_mod[i:j])
            soln_mod[-1] = soln_mod[0]
            if soln_mod not in tabu_list:
                nbhd.append(soln_mod)

    if intensity > 0.5:
        for i in range(n):
            for j in range(n):
                if i != j:
                    soln_mod: list[int] = soln.copy()
                    value = soln_mod.pop(i)
                    soln_mod.insert(j, value)
                    soln_mod[-1] = soln_mod[0]
                    if soln_mod not in tabu_list:
                        nbhd.append(soln_mod)

    return nbhd
