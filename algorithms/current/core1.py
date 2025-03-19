import random
from algorithms.utils import val


def tabu_search_genetic(
    soln_init: list[int],
    tabu_tenure: int,
    iter_max: int = 100,
) -> tuple[int, list[int]]:

    tabu_list: list[tuple] = []
    soln_curr = soln_init.copy()
    soln_best = soln_init.copy()
    soln_best_tracker = [val(soln_curr)]

    for _ in range(iter_max):
        # Generate initial neighborhood from soln_best via swaps
        initial_nbhd = neighborhood(soln_best, tabu_list[:tabu_tenure])
        pop = initial_nbhd

        # Generate crossover and mutation populations
        pop1 = crossover_population(pop)
        pop2 = mutate_population(pop)

        # Combine to form expanded neighborhood
        expanded_nbhd = pop + pop1 + pop2

        # Find best admissible solution
        nbhr_best = best_admissible_soln(
            expanded_nbhd, tabu_list, soln_best
        )

        if nbhr_best is None:
            break  # No admissible solution found

        # Update best solution if improved
        current_val = val(nbhr_best)
        if current_val < val(soln_best):
            soln_best = nbhr_best.copy()
            soln_best_tracker.append(current_val)

        # Update current solution and tabu list
        soln_curr = nbhr_best.copy()
        tabu_list.append(tuple(soln_curr))

    return val(soln_best), soln_best_tracker


def neighborhood(soln: list[int], tabu_list) -> list[list[int]]:
    nbhd = []
    n = len(soln) - 1  # Exclude last element (same as first)
    for i in range(n):
        for j in range(i + 1, n):
            soln_mod = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
            soln_mod[-1] = soln_mod[0]  # Ensure last element matches first
            if (soln[i], soln[j]) in tabu_list:
                continue
            nbhd.append(soln_mod)
    return nbhd


def crossover_population(pop):
    pop1 = []
    for i in range(0, len(pop) - 1, 2):
        parent1 = pop[i]
        parent2 = pop[i + 1]
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        pop1.extend([child1, child2])
    return pop1


def crossover(parent1, parent2):
    route1 = parent1[:-1]
    route2 = parent2[:-1]
    n = len(route1)
    if n == 0:
        return parent1.copy()
    start = random.randint(0, n - 1)
    end = random.randint(start, n - 1)
    child_route = [-1] * n
    child_route[start : end + 1] = route1[start : end + 1]
    remaining = [city for city in route2 if city not in child_route[start : end + 1]]
    ptr = 0
    for i in range(n):
        if child_route[i] == -1 and ptr < len(remaining):
            child_route[i] = remaining[ptr]
            ptr += 1
    child = child_route + [child_route[0]]
    return child


def mutate_population(pop):
    return [mutate(sol) for sol in pop]


def mutate(solution):
    mutation_type = random.choice(["2opt", "shift", "exchange"])
    if mutation_type == "2opt":
        return mutate_2opt(solution)
    elif mutation_type == "shift":
        return mutate_shift(solution)
    else:
        return mutate_exchange(solution)


def mutate_2opt(solution):
    route = solution[:-1]
    n = len(route)
    if n <= 1:
        return solution.copy()
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
    new_route.append(new_route[0])
    return new_route


def mutate_shift(solution):
    route = solution[:-1]
    n = len(route)
    if n == 0:
        return solution.copy()
    start = random.randint(0, n - 1)
    length = random.randint(1, max(1, n // 2))
    end = (start + length) % n
    if start < end:
        segment = route[start:end]
        remaining = route[:start] + route[end:]
    else:
        segment = route[start:] + route[:end]
        remaining = route[end:start]
    insert_pos = random.randint(0, len(remaining))
    new_route = remaining[:insert_pos] + segment + remaining[insert_pos:]
    new_route.append(new_route[0])
    return new_route


def mutate_exchange(solution):
    route = solution[:-1]
    n = len(route)
    if n == 0:
        return solution.copy()
    split_pos = random.randint(1, n - 1)
    new_route = route[split_pos:] + route[:split_pos]
    new_route.append(new_route[0])
    return new_route


def best_admissible_soln(expanded_nbhd, tabu_list, soln_best):
    best_val = float("inf")
    best_nbhr = None
    soln_best_val = val(soln_best)
    # Look for improving solutions not in tabu
    for nbhr in expanded_nbhd:
        nbhr_tuple = tuple(nbhr)
        current_val = val(nbhr)
        if current_val < soln_best_val and nbhr_tuple not in tabu_list:
            if current_val < best_val:
                best_val = current_val
                best_nbhr = nbhr
    if best_nbhr is not None:
        return best_nbhr
    # Fallback to best non-tabu solution
    best_non_tabu_val = float("inf")
    best_non_tabu = None
    for nbhr in expanded_nbhd:
        nbhr_tuple = tuple(nbhr)
        if nbhr_tuple in tabu_list:
            continue
        current_val = val(nbhr)
        if current_val < best_non_tabu_val:
            best_non_tabu_val = current_val
            best_non_tabu = nbhr
    return best_non_tabu
