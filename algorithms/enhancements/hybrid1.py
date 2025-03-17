from algorithms.utils import val, best_admissible_soln
import config
import random
import math


def search(
    soln_init: list[int],
    iter_max: int = 100,
) -> tuple[list[int], list[int], list[int]]:
    iter_max = 100

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    soln_best_tracker: list[int] = []

    stagnant_ctr: int = 0
    stagnant_best: int = 0

    tabu_tenure: int = math.floor(len(soln_init) * 0.1)
    improvement_rate: float = 0.0
    solution_diversity_tracker: list[list[int]] = []

    for iter_ctr in range(iter_max):

        # Solution diversity calculation
        solution_diversity_tracker.append(soln_curr)
        solution_diversity = len(set(map(tuple, solution_diversity_tracker))) / len(
            solution_diversity_tracker
        )

        # Compute improvement rate
        if len(soln_best_tracker) > 1:
            improvement_rate = abs(
                (soln_best_tracker[-1] - soln_best_tracker[-2])
                / (soln_best_tracker[-1] + 1e-10)
            )

        # Dynamically adjust tabu tenure
        tabu_tenure = quantum_tenure_adaptation(
            soln_init,
            tabu_tenure,
            iter_ctr,
            iter_max,
            solution_diversity,
            improvement_rate,
        )

        if stagnant_ctr > len(soln_init) * 0.2:
            soln_curr = wave_resonance_perturbation(
                soln_curr, iter_ctr, iter_max, soln_best, stagnant_ctr
            )

        nbhd, moves = neighborhood(soln_curr, tabu_list)

        nbhr_best, move_best = best_admissible_soln(nbhd, moves, tabu_list, soln_best)

        if val(nbhr_best) < val(soln_best):
            soln_best = nbhr_best
            soln_best_tracker.append(val(soln_best))

            if stagnant_ctr > stagnant_best:
                print(f"Stagnant iterations: {stagnant_ctr}")
                stagnant_best = stagnant_ctr

            stagnant_ctr = 0
        else:
            print(f"Stagnant iterations: {stagnant_ctr}")
            stagnant_ctr += 1

        soln_curr = nbhr_best.copy()

        # Update Tabu List
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        tabu_list.append(move_best)

    return val(soln_best), soln_best_tracker, stagnant_best


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


def wave_resonance_perturbation(
    soln_curr: list[int],
    iter_ctr: int,
    iter_max: int,
    soln_best: list[int],
    stagnant_ctr: int,
) -> tuple[list[int], int, int]:
    """
    Wave-Resonance Adaptive Perturbation (WRAP) method for balanced exploration and exploitation.

    Args:
        soln_curr: Current solution
        iter_ctr: Current iteration counter
        iter_max: Maximum iterations
        soln_best: Best solution found so far
        stagnant_ctr: Counter for stagnant iterations
        improve_ctr: Counter for consecutive improvements

    Returns:
        Perturbed solution
    """

    n = len(soln_curr) - 1

    # Enhanced progress ratio calculation
    # Incorporates both iteration progress and stagnation/improvement metrics
    progress_metrics = {
        "iter_progress": iter_ctr / iter_max,
        "stagnation_factor": min(1, stagnant_ctr / iter_max),
        "scale_factor": 0.5 + (n // 50) * 0.5,
    }

    # Composite progress ratio
    perturbation_intensity = (
        progress_metrics["iter_progress"] + progress_metrics["stagnation_factor"]
    ) * progress_metrics["scale_factor"]

    # Adaptive wave parameters
    wave_amplitude = max(
        1,
        int(
            n
            * (1 - perturbation_intensity)
            * (1 + stagnant_ctr / iter_max)
            * (1 + math.log(n) / 10)
        ),
    )

    # Resonance factor with stagnation influence
    resonance_factor = math.sin(
        perturbation_intensity * math.pi * 2 * (1 + math.log(n) / 10)
    )

    # Create a copy of the current solution
    perturbed_soln = soln_curr.copy()

    # Wave-based exploration
    for _ in range(wave_amplitude):
        # Wave centers with stagnation influence
        wave_centers = [
            int(
                n
                * abs(
                    math.sin(
                        i
                        * resonance_factor
                        * (1 + stagnant_ctr / iter_max)
                        * (1 + math.log(n) / 10)
                    )
                )
            )
            for i in range(wave_amplitude)
        ]

        for center in wave_centers:
            # Dynamic wave radius
            wave_radius = max(
                1,
                int(
                    wave_amplitude
                    * (1 - abs(resonance_factor))
                    * (1 + stagnant_ctr / iter_max)
                    * (1 + math.log(n) / 10)
                ),
            )

            # Swap candidates selection
            swap_candidates = set()
            for offset in range(-wave_radius, wave_radius + 1):
                candidate = (center + offset) % n
                swap_candidates.add(candidate)

            # Unique swap strategy
            if len(swap_candidates) > 1:
                swap_point1, swap_point2 = random.sample(list(swap_candidates), 2)

                # Swap with probability based on solution quality
                swap_probability = max(0.3, 1 - val(perturbed_soln) / val(soln_best))

                if random.random() < swap_probability:
                    if val(perturbed_soln) < val(soln_best):
                        perturbed_soln[swap_point1], perturbed_soln[swap_point2] = (
                            perturbed_soln[swap_point2],
                            perturbed_soln[swap_point1],
                        )

    # Ensure the last element matches the first (for cyclic solutions)
    perturbed_soln[-1] = perturbed_soln[0]

    return perturbed_soln


def quantum_tenure_adaptation(
    soln_init: list[int],
    base_tenure: int,
    iter_ctr: int,
    iter_max: int,
    solution_diversity: float,
    improvement_rate: float,
) -> int:
    """
    Dynamically adjusts tabu tenure using quantum-inspired principles.

    Args:
        base_tenure: Initial tabu tenure
        iter_ctr: Current iteration
        iter_max: Maximum iterations
        solution_diversity: Measure of solution exploration
        improvement_rate: Rate of solution improvement

    Returns:
        Dynamically adjusted tabu tenure
    """
    # Quantum probability wave function inspired adaptation
    quantum_wave = math.sin(2 * math.pi * iter_ctr / iter_max)

    # Entanglement factor considering solution diversity and improvement
    entanglement_factor = (
        solution_diversity * (1 + improvement_rate) * abs(quantum_wave)
    )

    # Probabilistic tenure adjustment
    dynamic_tenure = int(
        base_tenure * (1 + entanglement_factor * (1 - abs(quantum_wave)))
    )

    # Ensure tenure remains within reasonable bounds
    return max(
        min(dynamic_tenure, len(soln_init) * 2), max(3, int(len(soln_init) * 0.1))
    )
