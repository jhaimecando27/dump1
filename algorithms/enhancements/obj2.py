from algorithms.utils import new_neighborhood, val, best_admissible_soln
from .utils import estimate_improvement_potential
import config


def search(
    soln_init: list[int],
    tabu_tenure: int,
    iter_max: int = 100,
) -> tuple[list[int], list[int], list[int]]:

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    for iter_ctr in range(iter_max):
        nbhd, moves = neighborhood(soln_curr, tabu_list[:tabu_tenure])
        nbhr_best, move_best = best_admissible_soln(
            nbhd, moves, tabu_list[:tabu_tenure], soln_best
        )

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            stagnant_ctr = 0
            soln_best_tracker.append(val(soln_best))
        else:
            stagnant_ctr += 1

        soln_curr = nbhr_best

        # Update Tabu List
        tabu_list.append(move_best)
        while len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return val(soln_best), soln_best_tracker


def neighborhood(
    soln: list[int], tabu_list: list[list[int]]
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Generates neighborhood using dynamic focal point sampling.

    This approach identifies critical focal points in the solution and
    concentrates swap operations around these points, drastically reducing
    the neighborhood size while maintaining search effectiveness.

    Args:
        soln: The current solution
        tabu_list: List of tabu moves
    Returns:
        nbhd: list of new solutions
        moves: list of moves corresponding to each solution
    """
    nbhd = []
    moves = []
    n = len(soln) - 1  # Last element is same as first

    # Dynamic focal point selection
    # Calculate segment costs
    segment_costs = []
    for i in range(n):
        next_idx = (i + 1) % n
        cost = config.dms[str(len(soln))][soln[i]][soln[next_idx]]
        segment_costs.append((i, cost))

    # Sort segments by cost (highest first)
    segment_costs.sort(key=lambda x: x[1], reverse=True)

    # Select top k costly segments as focal points (where k scales with sqrt(n))
    k = max(2, int(n**0.5))  # At least 2 points, scales with sqrt(n)
    focal_indices = [p[0] for p in segment_costs[:k]]

    # Add a random element for exploration (prevents getting stuck in local optima)
    non_focal = [i for i in range(n) if i not in focal_indices]
    if non_focal:
        import random

        random_idx = random.choice(non_focal)
        if random_idx not in focal_indices:
            focal_indices.append(random_idx)

    # For each focal point, try swapping with a limited set of other positions
    for i in focal_indices:
        # Define a radius of influence that scales logarithmically with problem size
        radius = max(2, int(2 * (n**0.5)))

        # Generate candidate positions for swapping within radius
        j_candidates = set()

        # Add positions within immediate vicinity
        for offset in range(1, min(radius + 1, n)):
            j_candidates.add((i + offset) % n)
            j_candidates.add((i - offset + n) % n)

        # Add some other focal points for cross-influence
        for focal in focal_indices:
            if focal != i:
                j_candidates.add(focal)

        # Process only valid candidates (ensuring i < j to avoid duplicates)
        for j in [j for j in j_candidates if j > i and j < n]:
            soln_mod = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
            soln_mod[-1] = soln_mod[0]  # Ensure last element is same as first

            # Skip if move is in tabu list
            if (soln[i], soln[j]) in tabu_list:
                continue

            nbhd.append(soln_mod)
            moves.append((soln[i], soln[j]))

    return nbhd, moves
