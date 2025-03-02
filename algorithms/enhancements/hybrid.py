import config
import numpy as np
import concurrent.futures
from .utils import (
    adaptive_perturb,
    neighborhood,
    wave_perturb,
    update_quantum_state,
    calculate_quantum_tenure,
    calculate_phase_shift,
)
from algorithms.utils import val, new_neighborhood
import random
import math


def tabu_search(
    soln_init: list[int],
    iter_max: int = 100,
) -> tuple[list[int], list[int]]:

    n = len(soln_init)
    quantum_state = 0.5
    tabu_tenure = math.floor(n * 0.15)  # Increased initial tenure
    max_stagnant = 7  # Increased stagnation threshold
    tenure_reset_factor = 0.4  # Adjusted
    phase_shift = 0.0

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init
    soln_global_best: list[int] = soln_init  # Track overall best solution
    prev_best_value = val(soln_best)
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    # Strategic oscillation parameters
    intensification_counter = 0
    diversification_threshold = 10
    elite_solutions = []  # Track elite solutions for recombination

    # Enhanced wave parameters
    wave_amplitude = 0.22  # Increased initial amplitude
    wave_frequency = 1.2
    improvement_threshold = 0.02  # More sensitive threshold

    val_prev = val(soln_best)
    val_global_best = val_prev

    # Adaptive memory components
    frequency_matrix = np.zeros((n, n))
    recency_matrix = np.zeros((n, n))
    elite_pool_size = min(8, iter_max // 15)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for iter_ctr in range(iter_max):
            # Record current solution in frequency matrix
            for i in range(n - 1):
                j = (i + 1) % (n - 1)
                frequency_matrix[soln_curr[i], soln_curr[j]] += 1
                recency_matrix[soln_curr[i], soln_curr[j]] = iter_ctr

            # Get neighborhood using path-guided approach
            nbhd, moves = enhanced_neighborhood(
                soln_curr,
                tabu_list[:tabu_tenure],
                frequency_matrix,
                recency_matrix,
                iter_ctr,
            )

            nbhr_best, move_best = adaptive_best_admissible_soln(
                nbhd,
                moves,
                tabu_list[:tabu_tenure],
                soln_best,
                val_global_best,
                executor,
                iter_ctr,
                iter_max,
            )
            val_curr = val(nbhr_best)

            rel_improvement = (val_prev - val_curr) / val_prev if val_prev > 0 else 0

            # Update best solution if improved
            if val_curr < val(soln_best):
                soln_best = nbhr_best
                improvement = prev_best_value - val_curr
                phase_shift = calculate_phase_shift(improvement, prev_best_value)
                prev_best_value = val_curr
                stagnant_ctr = 0
                soln_best_tracker.append(val(soln_best))
                intensification_counter += 1

                # Add to elite pool if significantly better
                if elite_solutions:
                    worst_elite_value = max(val(sol) for sol in elite_solutions)
                    if (
                        len(elite_solutions) < elite_pool_size
                        or val_curr < worst_elite_value
                    ):
                        elite_solutions.append(nbhr_best.copy())
                        if len(elite_solutions) > elite_pool_size:
                            # Remove worst elite solution
                            worst_idx = max(
                                range(len(elite_solutions)),
                                key=lambda i: val(elite_solutions[i]),
                            )
                            elite_solutions.pop(worst_idx)
                else:
                    # First elite solution
                    elite_solutions.append(nbhr_best.copy())

                # Update global best
                if val_curr < val_global_best:
                    val_global_best = val_curr
                    soln_global_best = nbhr_best.copy()

                # Dynamically adjust wave parameters based on improvement quality
                if rel_improvement > improvement_threshold:
                    wave_amplitude = max(
                        0.08, wave_amplitude - 0.02
                    )  # Reduce disruption less aggressively
                    wave_frequency = max(1.0, wave_frequency - 0.15)
                else:
                    wave_amplitude = min(0.28, wave_amplitude + 0.015)
                    wave_frequency = min(2.0, wave_frequency + 0.12)
            else:
                stagnant_ctr += 1
                intensification_counter = 0

                # More aggressive parameter adjustments during stagnation
                if stagnant_ctr > 3:
                    wave_amplitude = min(0.35, wave_amplitude + 0.025)
                    wave_frequency = min(3.0, wave_frequency + 0.25)
                if stagnant_ctr > max_stagnant:
                    phase_shift = -1.0
                    tabu_tenure = max(
                        n * 0.15, int(tabu_tenure * (1 + tenure_reset_factor))
                    )

            # Strategic oscillation between intensification and diversification
            if intensification_counter > diversification_threshold:
                # Apply strong diversification by recombining elite solutions
                if len(elite_solutions) >= 2:
                    parent1, parent2 = random.sample(elite_solutions, 2)
                    soln_curr = recombine_solutions(parent1, parent2, n)
                    soln_curr[-1] = soln_curr[0]  # Maintain cycle closure
                else:
                    soln_curr = strong_perturb(soln_global_best, n)
                intensification_counter = 0
                stagnant_ctr = 0
                continue

            # Adaptive perturbation threshold based on problem size and search stage
            perturbation_threshold = 5 + int(len(soln_init) / 30)
            perturbation_threshold = max(4, min(perturbation_threshold, 10))

            # Pattern-based perturbation
            if stagnant_ctr >= perturbation_threshold:
                if random.random() < 0.7:  # 70% chance for wave perturbation
                    soln_curr = enhanced_wave_perturb(
                        soln_best,
                        wave_amplitude,
                        wave_frequency,
                        stagnant_ctr,
                        frequency_matrix,
                    )
                else:  # 30% chance for path-relinking with global best
                    soln_curr = path_relink(soln_curr, soln_global_best, n)
                stagnant_ctr = 0
                continue

            soln_curr = nbhr_best
            val_prev = val_curr

            # Update quantum state with advanced dynamics
            progress = iter_ctr / iter_max
            quantum_state = enhanced_quantum_state(
                quantum_state, phase_shift, progress, stagnant_ctr
            )

            # Adaptive tabu tenure with problem-specific knowledge
            tabu_tenure = calculate_adaptive_tenure(
                tabu_tenure, quantum_state, n, stagnant_ctr, progress
            )

            tabu_list.append(move_best)
            while len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

    return val(soln_global_best), soln_best_tracker


def enhanced_neighborhood(
    soln: list[int],
    tabu_list: list[tuple[int, int]],
    frequency_matrix: np.ndarray,
    recency_matrix: np.ndarray,
    current_iter: int,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """
    Creates an enhanced neighborhood with guided selection of candidates
    """
    nbhd = []
    moves = []

    tabu_set = set(tabu_list)
    n = len(soln) - 1

    soln_unique = np.array(soln[:n])
    dmat = config.dms[str(n + 1)]
    prev = np.roll(soln_unique, 1)
    nxt = np.roll(soln_unique, -1)
    local_costs = dmat[prev, soln_unique] + dmat[soln_unique, nxt]

    # Enhanced edge cost calculation
    edge_costs = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate cost difference if i and j were swapped
            i_cost = local_costs[i]
            j_cost = local_costs[j]

            # Compute new connections cost if swapped
            i_prev, i_next = (i - 1) % n, (i + 1) % n
            j_prev, j_next = (j - 1) % n, (j + 1) % n

            # Avoid duplicated indices in small problems
            if i_next == j or j_next == i or i_prev == j or j_prev == i:
                continue

            old_cost = (
                dmat[soln[i_prev], soln[i]]
                + dmat[soln[i], soln[i_next]]
                + dmat[soln[j_prev], soln[j]]
                + dmat[soln[j], soln[j_next]]
            )

            new_cost = (
                dmat[soln[i_prev], soln[j]]
                + dmat[soln[j], soln[i_next]]
                + dmat[soln[j_prev], soln[i]]
                + dmat[soln[i], soln[j_next]]
            )

            improvement = old_cost - new_cost
            edge_costs[i, j] = improvement

    # Find promising edges using cost improvement and frequency/recency
    promising_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if edge_costs[i, j] > 0:  # Prioritize improvement
                # Calculate frequency-based score (prefer less frequently used)
                freq_score = 1.0 / (1.0 + frequency_matrix[soln[i], soln[j]])
                # Calculate recency-based score (prefer longer unused)
                recency_score = 1.0 / (
                    1.0 + max(1, current_iter - recency_matrix[soln[i], soln[j]])
                )
                # Combined score with weights
                score = edge_costs[i, j] * 0.6 + freq_score * 0.3 + recency_score * 0.1
                promising_edges.append((i, j, score))

    # Sort by score
    promising_edges.sort(key=lambda x: x[2], reverse=True)

    # Select top candidates plus some random ones for diversification
    num_candidates = min(25, n * (n - 1) // 4)  # Increased neighborhood size
    top_candidates = int(num_candidates * 0.8)  # 80% top scored
    rand_candidates = num_candidates - top_candidates  # 20% random

    # Add top candidates
    top_moves = [(x[0], x[1]) for x in promising_edges[:top_candidates]]

    # Add some random moves for diversification
    random_indices = list(range(n))
    random_moves = []
    attempts = 0
    while len(random_moves) < rand_candidates and attempts < 100:
        attempts += 1
        i, j = random.sample(random_indices, 2)
        move = (min(i, j), max(i, j))
        if move not in top_moves and move not in random_moves:
            random_moves.append(move)

    candidate_moves = top_moves + random_moves

    # Generate neighborhood
    for move in candidate_moves:
        if (soln[move[0]], soln[move[1]]) in tabu_set:
            continue
        soln_mod = soln.copy()
        i, j = move
        soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
        soln_mod[-1] = soln_mod[0]  # Maintain cycle closure
        nbhd.append(soln_mod)
        moves.append(move)

    return nbhd, moves


def adaptive_best_admissible_soln(
    nbhd: list[list[int]],
    moves: list[tuple[int, int]],
    tabu_list: list[tuple[int, int]],
    soln_best: list[int],
    global_best_val: int,
    executor: concurrent.futures.ProcessPoolExecutor,
    current_iter: int,
    max_iter: int,
) -> tuple[list[int] | None, tuple[int, int] | None]:
    """
    Enhanced evaluation with adaptive aspiration criteria
    """
    soln_best_value = val(soln_best)
    args_list = [
        (
            candidate,
            move,
            tabu_list,
            soln_best_value,
            global_best_val,
            current_iter,
            max_iter,
        )
        for candidate, move in zip(nbhd, moves)
    ]
    best_candidate = None
    best_value = float("inf")
    best_move = None

    # Define aspiration level based on progress
    progress = current_iter / max_iter
    aspiration_factor = 0.99 if progress < 0.3 else (0.97 if progress < 0.7 else 0.95)
    aspiration_threshold = global_best_val * aspiration_factor

    # Use a chunksize to reduce overhead when mapping a large list
    results = executor.map(evaluate_candidate_enhanced, args_list, chunksize=10)
    for result in results:
        if result is not None:
            candidate_value, candidate, move, is_tabu = result
            if candidate_value < best_value:
                # Accept if better than current best or meets aspiration criteria
                if not is_tabu or candidate_value < aspiration_threshold:
                    best_value = candidate_value
                    best_candidate = candidate
                    best_move = move

    # If no admissible move found, use first candidate as fallback
    if best_candidate is None and nbhd:
        best_candidate = nbhd[0]
        best_move = moves[0]

    return best_candidate, best_move


def evaluate_candidate_enhanced(
    args: tuple[list[int], tuple[int, int], list[tuple[int, int]], int, int, int, int],
) -> tuple[int, list[int], tuple[int, int], bool] | None:
    """
    Enhanced candidate evaluation with tabu status information
    """
    (
        candidate,
        move,
        tabu_list,
        soln_best_value,
        global_best_value,
        current_iter,
        max_iter,
    ) = args
    candidate_value = val(candidate)
    is_tabu = move in tabu_list

    # Define aspiration criteria based on search progress
    progress = current_iter / max_iter
    aspiration_level = global_best_value * (
        0.98 if progress < 0.3 else (0.95 if progress < 0.7 else 0.90)
    )

    # Accept if non-tabu or meets aspiration criteria
    if (
        not is_tabu and candidate_value < soln_best_value
    ) or candidate_value < aspiration_level:
        return candidate_value, candidate, move, is_tabu
    return None


def calculate_adaptive_tenure(
    base: int, state: float, n: int, stagnation: int, progress: float
) -> int:
    """
    Advanced adaptive tabu tenure calculation
    """
    min_tenure = max(7, math.floor(n * 0.15))
    max_tenure = math.floor(n * 0.7)  # Higher maximum for larger problems

    # Dynamic factors
    stagnation_factor = min(0.4, stagnation / 10)
    progress_factor = 0.5 * math.sin(math.pi * progress)  # Oscillation pattern

    # Apply quantum state with adjustments
    tenure = min_tenure + math.floor(
        (max_tenure - min_tenure)
        * (state + stagnation_factor + progress_factor + 0.15 * random.random())
    )

    # Ensure reasonable bounds
    return max(min_tenure, min(max_tenure, tenure))


def enhanced_quantum_state(
    state: float, phase: float, progress: float, stagnation: int
) -> float:
    """
    Enhanced quantum state with improved interference patterns
    """
    # Create oscillatory pattern based on progress
    oscillation = 0.2 * math.sin(2 * math.pi * progress)

    # Incorporate stagnation to adjust phase impact
    stagnation_effect = min(0.3, stagnation / 20)

    # Quantum walk with phase and oscillation
    new_state = state + phase * (1.0 - progress) + oscillation + stagnation_effect

    # Ensure bounds
    return max(0.1, min(0.9, new_state))


def enhanced_wave_perturb(
    solution: list[int],
    amplitude: float,
    frequency: float,
    stagnation: int,
    frequency_matrix: np.ndarray,
) -> list[int]:
    """
    Enhanced wave perturbation with pattern recognition
    """
    perturbed = solution.copy()
    n = len(perturbed)

    # Dynamic parameters with problem-specific scaling
    base_intensity = min(0.35, 0.12 + (n / 800))
    stagnation_factor = min(0.55, stagnation / 15)
    intensity = base_intensity + stagnation_factor

    # Enhanced wave modulation
    phase_shift = (stagnation % 7) * math.pi / 3.5
    wave_complexity = 1 + (stagnation // 8)

    # Get frequency-based information for intelligent perturbation
    edge_frequencies = frequency_matrix.copy()

    # Find edges to target based on frequency (focus on frequently used edges)
    edge_targets = []
    for i in range(n - 1):
        j = (i + 1) % (n - 1)
        edge_targets.append((i, edge_frequencies[perturbed[i], perturbed[j]]))

    # Sort edges by frequency
    edge_targets.sort(key=lambda x: x[1], reverse=True)

    # Target top frequency edges for disruption with higher probability
    target_indices = [x[0] for x in edge_targets[: int(n / 3)]]

    # Apply selective perturbation focusing on high-frequency regions
    for i in range(n - 1):
        # Complex wave pattern with multiple harmonics
        wave = 0
        for harmonic in range(1, wave_complexity + 1):
            wave += (amplitude / harmonic) * math.sin(
                frequency * harmonic * i * 2 * math.pi / n + phase_shift
            )

        # Higher probability of perturbation for targeted indices
        perturbation_threshold = 0.15 * intensity
        if i in target_indices:
            perturbation_threshold *= 0.7  # Lower threshold for targeted indices

        if abs(wave) > perturbation_threshold:
            # Adaptive swap distance based on wave magnitude
            swap_distance = max(1, int(abs(wave * n * intensity / 2.5)))
            swap_idx = (i + swap_distance) % (n - 1)
            if i != swap_idx:
                perturbed[i], perturbed[swap_idx] = perturbed[swap_idx], perturbed[i]

    # Maintain cycle closure
    perturbed[-1] = perturbed[0]
    return perturbed


def strong_perturb(solution: list[int], n: int) -> list[int]:
    """
    Strong perturbation for significant diversification
    """
    perturbed = solution.copy()

    # Number of swaps proportional to problem size
    num_swaps = max(3, n // 15)

    # Apply multiple random swaps
    for _ in range(num_swaps):
        i, j = random.sample(range(n - 1), 2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

    # Apply a segment reversal with probability 0.5
    if random.random() < 0.5:
        segment_start = random.randint(0, n - 4)
        segment_length = random.randint(2, min(n // 3, n - segment_start - 1))
        segment_end = segment_start + segment_length
        # Reverse segment
        perturbed[segment_start:segment_end] = list(
            reversed(perturbed[segment_start:segment_end])
        )

    # Maintain cycle closure
    perturbed[-1] = perturbed[0]
    return perturbed


def recombine_solutions(parent1: list[int], parent2: list[int], n: int) -> list[int]:
    """
    Recombine two parent solutions to create a new solution
    """
    # Select a random crossover point
    crossover_point = random.randint(1, n - 3)

    # Take first segment from parent1
    child = parent1[:crossover_point]

    # Add remaining elements from parent2 in their order
    remaining = [x for x in parent2 if x not in child]
    child.extend(remaining)

    # Ensure child has correct length
    child = child[: n - 1]

    # Add closure
    child.append(child[0])

    return child


def path_relink(current: list[int], target: list[int], n: int) -> list[int]:
    """
    Perform path relinking between current solution and target (usually global best)
    """
    # Create a copy of current solution
    result = current.copy()

    # Calculate number of moves to make (partial path)
    moves_count = random.randint(2, max(2, (n // 8)))

    # Identify differences between solutions
    differences = []
    for i in range(n - 1):
        if current[i] != target[i]:
            differences.append(i)

    # If no differences or too few, return with small random perturbation
    if len(differences) <= 1:
        i, j = random.sample(range(n - 1), 2)
        result[i], result[j] = result[j], result[i]
        result[-1] = result[0]  # Maintain cycle
        return result

    # Make a subset of moves toward target
    moves = random.sample(differences, min(moves_count, len(differences)))
    for idx in moves:
        # Find where the target value is in current solution
        for j in range(n - 1):
            if result[j] == target[idx]:
                # Swap to match target at this position
                result[idx], result[j] = result[j], result[idx]
                break

    # Maintain cycle closure
    result[-1] = result[0]
    return result
