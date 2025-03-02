from .utils import adaptive_perturb, neighborhood, wave_perturb
from algorithms.utils import val, best_admissible_soln, new_neighborhood
import random
import math


def tabu_search_with_perturbation(
    soln_init: list[int],
    iter_max: int = 100,
) -> tuple[list[int], list[int]]:

    n = len(soln_init)
    tabu_tenure = math.floor(n * 0.10)

    tabu_list: list[tuple[int, int]] = []
    soln_curr: list[int] = soln_init
    soln_best: list[int] = soln_init.copy()
    stagnant_ctr: int = 0
    soln_best_tracker: list[int] = []

    # Enhanced parameters with adaptive control
    wave_amplitude = 0.12
    wave_frequency = 1.0
    improvement_threshold = 0.02
    intensity_factor = 0.8  # For neighborhood generation

    # Strategic oscillation parameters
    oscillation_factor = 0.5
    oscillation_period = 10

    # Long-term memory structures
    frequency_matrix = {}  # Track move frequencies
    promising_regions = []  # Store promising solutions for path relinking

    # Initialize frequency matrix
    for i in range(len(soln_init)):
        for j in range(len(soln_init)):
            if i != j:
                frequency_matrix[(i, j)] = 0

    val_prev = val(soln_best)
    val_best_overall = val_prev
    elite_pool = []  # Store elite solutions for path relinking

    # Temperature parameters for simulated annealing hybrid
    temperature = len(soln_init) * 0.5
    cooling_rate = 0.95

    for iter_ctr in range(iter_max):
        # Adjust intensity based on oscillation and iteration progress
        current_intensity = intensity_factor + oscillation_factor * math.sin(
            2 * math.pi * iter_ctr / oscillation_period
        )
        current_intensity = max(0.5, min(1.5, current_intensity))

        # Generate neighborhood with dynamic intensity
        #nbhd, moves = neighborhood(
        #    soln_curr, tabu_list[:tabu_tenure], intensity=current_intensity
        #)
        nbhd, moves = new_neighborhood(
            soln_curr, tabu_list[:tabu_tenure]
        )

        if not nbhd:  # Handle empty neighborhood
            # Apply stronger perturbation and continue
            soln_curr = strong_perturbation(soln_best, stagnant_ctr)
            stagnant_ctr += 1
            continue

        # Use aspiration criteria and simulated annealing hybridization
        nbhr_best, move_best = best_admissible_with_aspiration(
            nbhd, moves, tabu_list[:tabu_tenure], soln_best, temperature
        )

        # If no admissible solution found, apply perturbation
        if nbhr_best is None:
            soln_curr = wave_perturb(
                soln_best, wave_amplitude, wave_frequency, stagnant_ctr
            )
            stagnant_ctr += 1
            continue

        val_curr = val(nbhr_best)

        # Update frequency matrix for long-term memory
        if isinstance(move_best, tuple) and len(move_best) >= 3:
            move_type, i, j = move_best[:3]
            if (i, j) in frequency_matrix:
                frequency_matrix[(i, j)] += 1

        rel_improvement = (val_prev - val_curr) / val_prev if val_prev > 0 else 0

        # Store elite solutions for path relinking
        if len(elite_pool) < 5 or val_curr < max(val(e) for e in elite_pool):
            if nbhr_best not in elite_pool:
                elite_pool.append(nbhr_best.copy())
                if len(elite_pool) > 5:
                    # Remove worst solution
                    elite_pool.sort(key=val)
                    elite_pool.pop()

        if val_curr < val(soln_best):
            soln_best = nbhr_best.copy()
            soln_best_tracker.append(val(soln_best))
            stagnant_ctr = 0

            # Store promising region
            promising_regions.append(soln_best.copy())
            if len(promising_regions) > 3:
                promising_regions.pop(0)

            # Adjust wave parameters based on improvement magnitude
            if rel_improvement > improvement_threshold:
                wave_amplitude = max(0.05, wave_amplitude - 0.01)  # Reduce disruption
                wave_frequency = max(0.8, wave_frequency - 0.1)  # Slow down wave
                # Reward improvement with more search in this area
                intensity_factor = min(1.2, intensity_factor + 0.05)
            else:
                wave_amplitude = min(0.2, wave_amplitude + 0.005)  # Slight increase
                wave_frequency = min(1.5, wave_frequency + 0.05)  # Speed up wave

            # If we've found a new global best, update parameters
            if val_curr < val_best_overall:
                val_best_overall = val_curr
                # Reset temperature for simulated annealing component
                temperature = len(soln_init) * 0.5
        else:
            stagnant_ctr += 1

            # Apply strategic oscillation when stagnating
            if stagnant_ctr % 3 == 0:
                intensity_factor = max(
                    0.5, min(1.5, intensity_factor * (1 + 0.1 * random.uniform(-1, 1)))
                )

            # Increase perturbation when stagnating
            if stagnant_ctr > 3:
                wave_amplitude = min(0.25, wave_amplitude + 0.01)
                wave_frequency = min(2.5, wave_frequency + 0.1)

            # Cool temperature for simulated annealing component
            temperature *= cooling_rate

        # Adaptive perturbation threshold based on problem size and search progress
        base_perturbation_threshold = 4 + int(len(soln_init) / 50)
        progress_factor = iter_ctr / iter_max
        current_perturbation_threshold = max(
            3, int(base_perturbation_threshold * (1 - 0.5 * progress_factor))
        )

        if stagnant_ctr >= current_perturbation_threshold:
            # Use different perturbation strategies based on search state
            if stagnant_ctr >= current_perturbation_threshold * 2:
                # Strong perturbation for extended stagnation
                soln_curr = strong_perturbation(soln_best, stagnant_ctr)
            elif len(promising_regions) > 0 and random.random() < 0.7:
                # Path relinking with a promising region
                source = random.choice(promising_regions)
                soln_curr = path_relink(soln_best, source)
            elif len(elite_pool) > 1 and random.random() < 0.3:
                # Path relinking between elite solutions
                source = random.choice(elite_pool)
                target = random.choice([s for s in elite_pool if s != source])
                soln_curr = path_relink(source, target)
            else:
                # Regular wave perturbation
                soln_curr = wave_perturb(
                    soln_best, wave_amplitude, wave_frequency, stagnant_ctr
                )
            stagnant_ctr = 0
            tabu_tenure = n * 0.1
            continue

        soln_curr = nbhr_best
        val_prev = val_curr

        # Update tabu list with adjusted tenure based on search progress
        tabu_list.append(move_best)

        # Use dynamic tabu tenure
        if stagnant_ctr > 0:
            # Increase tenure when stagnating to force exploration
            tabu_tenure = min(n * 0.1, tabu_tenure * 0.1)

        while len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Periodically apply path relinking to combine elite solutions
        if iter_ctr > 0 and iter_ctr % 20 == 0 and len(elite_pool) >= 2:
            elite_pool.sort(key=val)
            guide_soln = elite_pool[0]  # Best solution
            trial_soln = elite_pool[-1]  # Most different solution
            path_soln = path_relink(guide_soln, trial_soln)
            if val(path_soln) < val(soln_curr):
                soln_curr = path_soln
                if val(path_soln) < val(soln_best):
                    soln_best = path_soln.copy()
                    soln_best_tracker.append(val(soln_best))

    return val(soln_best), soln_best_tracker


def best_admissible_with_aspiration(
    nbhd: list[list[int]],
    moves: list[tuple[int, int]],
    tabu_list: list[list[int]],
    soln_best: list[int],
    temperature: float = 0,
) -> tuple[list[int], tuple]:
    """
    Enhanced version with aspiration criteria and simulated annealing principles
    """
    import random
    import math

    val_best: int = float("inf")
    nbhr_best: list[int] = None
    move_best: tuple[int, int] = None

    # Track the best non-tabu solution
    val_best_non_tabu = float("inf")
    nbhr_best_non_tabu = None
    move_best_non_tabu = None

    # Track the best tabu solution (for aspiration)
    val_best_tabu = float("inf")
    nbhr_best_tabu = None
    move_best_tabu = None

    for idx, nbhr_curr in enumerate(nbhd):
        val_curr: int = val(nbhr_curr)
        move_curr = moves[idx]

        is_tabu = move_curr in tabu_list

        # Check if this is the best solution seen so far
        if val_curr < val_best:
            val_best = val_curr

        # Track the best non-tabu solution
        if not is_tabu and val_curr < val_best_non_tabu:
            val_best_non_tabu = val_curr
            nbhr_best_non_tabu = nbhr_curr
            move_best_non_tabu = move_curr

        # Track the best tabu solution for aspiration
        if is_tabu and val_curr < val_best_tabu:
            val_best_tabu = val_curr
            nbhr_best_tabu = nbhr_curr
            move_best_tabu = move_curr

        # Aspiration criterion: accept tabu move if it's better than the best known
        if is_tabu and val_curr < val(soln_best):
            val_best_non_tabu = val_curr
            nbhr_best_non_tabu = nbhr_curr
            move_best_non_tabu = move_curr

        # Simulated annealing principle: accept worse moves with probability
        if temperature > 0 and is_tabu and val_curr > val(soln_best):
            # Calculate acceptance probability
            delta = val_curr - val(soln_best)
            prob = math.exp(-delta / temperature)
            if (
                random.random() < prob * 0.1
            ):  # Reduced probability to avoid excessive worsening
                val_best_non_tabu = val_curr
                nbhr_best_non_tabu = nbhr_curr
                move_best_non_tabu = move_curr

    # If we found a non-tabu solution, return it
    if nbhr_best_non_tabu is not None:
        return nbhr_best_non_tabu, move_best_non_tabu

    # If we have a tabu solution that's better than the best known, return it (aspiration)
    if nbhr_best_tabu is not None and val_best_tabu < val(soln_best):
        return nbhr_best_tabu, move_best_tabu

    # If we're using simulated annealing and have a tabu solution, consider accepting it
    if temperature > 0 and nbhr_best_tabu is not None:
        delta = val_best_tabu - val(soln_best)
        prob = math.exp(-delta / temperature)
        if random.random() < prob * 0.05:  # Very low probability
            return nbhr_best_tabu, move_best_tabu

    # No admissible solution found
    return None, None


def strong_perturbation(solution: list[int], stagnation: int) -> list[int]:
    """
    Apply a stronger perturbation when regular wave perturbation isn't effective
    """
    import random

    perturbed = solution.copy()
    n = len(perturbed)

    # Calculate number of swaps based on problem size and stagnation
    num_swaps = max(3, min(n // 5, int(2 + stagnation // 2)))

    # Perform random swaps
    for _ in range(num_swaps):
        i = random.randint(0, n - 2)
        j = random.randint(0, n - 2)
        while i == j:
            j = random.randint(0, n - 2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

    # Create some reversed segments
    segment_size = max(3, min(n // 8, 10))
    if n > segment_size * 2:
        start = random.randint(0, n - segment_size - 1)
        perturbed[start : start + segment_size] = reversed(
            perturbed[start : start + segment_size]
        )

    # Ensure cycle closure
    perturbed[-1] = perturbed[0]
    return perturbed


def path_relink(source: list[int], target: list[int]) -> list[int]:
    """
    Path relinking between two solutions to generate high-quality intermediate solutions
    """
    # Start with a copy of the source solution
    current = source.copy()
    n = len(current)

    # Create a dictionary of positions in the target
    target_positions = {target[i]: i for i in range(n - 1)}  # Exclude last element

    # Find positions that differ between source and target
    different_positions = []
    for i in range(n - 1):  # Exclude last element
        if current[i] != target[i]:
            different_positions.append(i)

    # Perform a limited number of moves toward the target
    num_moves = min(len(different_positions) // 3 + 1, n // 5)
    best_intermediate = current
    best_val = val(current)

    for _ in range(num_moves):
        if not different_positions:
            break

        # Choose a random position to fix
        idx = random.choice(different_positions)
        different_positions.remove(idx)

        # Find where the target value is in the current solution
        target_value = target[idx]
        source_idx = current.index(target_value)

        # Swap to match the target
        current[idx], current[source_idx] = current[source_idx], current[idx]

        # Ensure the last element matches the first
        current[-1] = current[0]

        # Check if this intermediate solution is better
        current_val = val(current)
        if current_val < best_val:
            best_val = current_val
            best_intermediate = current.copy()

    # Return the best intermediate solution found
    return best_intermediate


def wave_perturb(
    solution: list[int], amplitude: float, frequency: float, stagnation: int
) -> list[int]:
    """
    Enhanced wave perturbation with more sophisticated disruption patterns
    """
    import math
    import random

    perturbed = solution.copy()
    n = len(perturbed)

    # Dynamic parameters based on problem size and stagnation
    base_intensity = min(0.25, 0.05 + (n / 800))  # Scales with problem size
    stagnation_factor = min(0.4, stagnation / 25)  # Caps at 0.4
    intensity = base_intensity + stagnation_factor

    # Modulated wave parameters with multiple harmonics
    phase_shift = (stagnation % 8) * math.pi / 4
    wave_complexity = 1 + (stagnation // 8)  # Increases wave complexity with stagnation

    # Random disruptions to prevent cycling
    disruption_seed = random.random() * 10

    # Create a disruption probability map
    disruption_map = []
    for i in range(n - 1):
        # Complex wave pattern with multiple harmonics
        wave = 0
        for harmonic in range(1, wave_complexity + 1):
            wave += (amplitude / harmonic) * math.sin(
                frequency * harmonic * i * 2 * math.pi / n
                + phase_shift
                + disruption_seed
            )

            # Add cosine components for more complex patterns
            if harmonic % 2 == 0:
                wave += (amplitude / (harmonic * 2)) * math.cos(
                    frequency * harmonic * i * math.pi / n
                    + phase_shift / 2
                    + disruption_seed
                )

        disruption_map.append(abs(wave))

    # Sort positions by disruption value
    positions = list(range(n - 1))
    positions.sort(key=lambda p: disruption_map[p], reverse=True)

    # Apply disruptions to the most promising positions
    disruption_count = max(2, int(n * intensity / 10))
    for i in range(min(disruption_count, len(positions))):
        pos = positions[i]
        if disruption_map[pos] > 0.15 * intensity:
            # Apply different disruption types based on position and stagnation
            disruption_type = (pos + stagnation) % 3

            if disruption_type == 0:
                # Swap with a position determined by the wave pattern
                swap_distance = max(1, int(disruption_map[pos] * n * intensity / 2.5))
                swap_idx = (pos + swap_distance) % (n - 1)
                if pos != swap_idx:
                    perturbed[pos], perturbed[swap_idx] = (
                        perturbed[swap_idx],
                        perturbed[pos],
                    )

            elif disruption_type == 1 and pos < n - 3:
                # Small segment reversal
                seg_length = min(3, n - pos - 1)
                perturbed[pos : pos + seg_length] = reversed(
                    perturbed[pos : pos + seg_length]
                )

            else:
                # Insertion move
                target_pos = (pos + int(n * disruption_map[pos] * 0.3)) % (n - 1)
                if pos != target_pos:
                    value = perturbed[pos]
                    perturbed.pop(pos)
                    perturbed.insert(target_pos, value)

    # Maintain cycle closure
    perturbed[-1] = perturbed[0]
    return perturbed
