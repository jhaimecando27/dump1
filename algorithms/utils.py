import random
import pandas as pd
import time
import config

from sklearn.cluster import KMeans
import numpy as np

from itertools import combinations


def neighborhood(soln: list[int], tabu_list: list[list[int]]) -> list[list[int]]:
    """Generates neighborhood of new solution from selected solution by
    making small changes. For current tabu search
    Args:
        soln: The current solution passed
        tabu_list: List of recent solutions
    Returns:
        nbhd: list of new solutions
    Raises:
    """
    nbhd: list = []
    moves: list = []

    # Make sure the last element is the same as the first element
    n = len(soln) - 1
    for i in range(int(n/2)):
        for j in range(i + 1, int(n/2)):
            soln_mod: list[int] = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
            soln_mod[-1] = soln_mod[0]
            if (soln[i], soln[j]) in tabu_list:
                continue
            nbhd.append(soln_mod)
            moves.append((soln[i], soln[j]))
    return nbhd, moves


def val(soln: list[int]) -> int:
    """Calculate the value of the solution
    Args:
        soln: The solution
    Returns:
        value: The value of the solution
    Raises:
    """
    value: int = 0
    n = len(soln)
    for i in range(n):
        poi_first: int = soln[i]
        poi_second: int = soln[(i + 1) % n]
        value += config.dms[str(len(soln))][poi_first][poi_second]

    return value


def best_admissible_soln(
    nbhd: list[list[int]],
    moves: list[tuple[int, int]],
    tabu_list: list[list[int]],
    soln_best: list[int],
) -> list[int]:
    """Finds the best admissible solution. It must be better than current
    solution and doesn't exist in tabu list.
    Args:
        nbhd: Neighborhood of solutions
        tabu_list: List of recent solutions
    Returns:
        nbhr_best: Best admissible neighbor in the neighborhood
    Raises:
    """
    val_best: int = float("inf")  # Starts with large value to accept 1st neighbor
    nbhr_best: list[int] = None
    move_best: tuple[int, int] = None

    for idx, nbhr_curr in enumerate(nbhd):
        val_curr: int = val(nbhr_curr)

        if moves[idx] not in tabu_list or val_curr < val_best:
            if val_curr < val_best:
                val_best = val_curr
                nbhr_best = nbhr_curr
                move_best = moves[idx]

    return nbhr_best, move_best


def prob_neighborhood(soln: list[int], tabu_list: list[list[int]]) -> list[list[int]]:
    nbhd: list = []
    moves: list = []

    n = len(soln) - 1

    node_weights = [i + 1 for i in range(n)]

    num_candidates = min(20, n * (n - 1) // 2)

    for _ in range(num_candidates):
        i = random.choices(range(n), weights=node_weights, k=1)[0]
        j_candidates = [j for j in range(n) if j != i]
        j_weights = [node_weights[j] for j in range(n) if j != i]
        j = random.choices(j_candidates, weights=j_weights, k=2)[0]

        if i > j:
            i, j = j, i

        if (soln[i], soln[j]) not in tabu_list:
            soln_mod = soln.copy()

            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]

            soln_mod[-1] = soln_mod[0]

            nbhd.append(soln_mod)
            moves.append((i, j))
    return nbhd, moves


def p2_neighborhood(
    soln: list[int], tabu_list: list[tuple]
) -> tuple[list[list[int]], list[tuple]]:
    """
    Generates a neighborhood of new solutions using probabilistic move generation.
    Instead of enumerating every possible swap, this version randomly generates N_SIZE moves.

    For each move:
      - (For a multi-route solution, you would pick a route k1 and then a (possibly different)
        route k2 from the current solution. Here, because soln is a single route, we set k1 = k2 = 0.)
      - Randomly pick two distinct indices (p1, p2) in the route (excluding the last element, which
        is assumed to duplicate the first to maintain circularity).
      - Build a move tuple (0, 0, p1, p2).
      - If the move is in tabu_list, skip it.
      - Otherwise, swap the nodes at positions p1 and p2 to form a new candidate solution and fix
        the last element to match the first.

    Args:
        soln: The current solution route (a list of node indices).
        tabu_list: List of tabu moves (each move is a tuple, e.g. (k1, k2, p1, p2)).

    Returns:
        nbhd: A list of new candidate solutions.
        moves: A list of moves corresponding to each candidate solution.
    """
    import random  # ensure random is available

    nbhd = []
    moves = []
    N_SIZE = 50  # Number of moves to generate

    # For the current single-route solution, treat it as one route.
    # (For a multi-route scenario, soln would be a list of routes and you would randomly choose k1 and k2.)
    route = soln.copy()
    n = len(route) - 1  # Exclude last element (kept as duplicate for circular route)

    for _ in range(N_SIZE):
        if n < 2:
            break  # Not enough nodes to swap
        # Randomly select two distinct indices from 0 to n-1
        p1 = route[random.randint(0, n - 1)]
        p2 = route[random.randint(0, n - 1)]

        while p1 == p2:
            p2 = route[random.randint(0, n - 1)]
        move = (soln[p1], soln[p2])

        # Check if the move is tabu
        if move in tabu_list:
            continue  # Skip tabu moves

        # Create a new candidate solution by swapping the nodes at indices p1 and p2
        new_route = route.copy()
        new_route[p1], new_route[p2] = new_route[p2], new_route[p1]
        # Ensure the route remains circular
        new_route[-1] = new_route[0]

        nbhd.append(new_route)
        moves.append(move)

    return nbhd, moves


def new_neighborhood(
    soln: list[int], tabu_list: list[tuple[int, int]]
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    nbhd = []
    moves = []

    tabu_set = set(tabu_list)

    n = len(soln) - 1

    soln_unique = np.array(soln[:n])
    dmat = config.dms[str(n + 1)]
    prev = np.roll(soln_unique, 1)
    nxt = np.roll(soln_unique, -1)
    local_costs = dmat[prev, soln_unique] + dmat[soln_unique, nxt]

    median_cost = np.median(local_costs)
    high_indices = [i for i in range(n) if local_costs[i] >= median_cost]
    if not high_indices:
        high_indices = list(range(n))

    candidate_moves = set()
    num_candidates = min(20, len(high_indices) * (len(high_indices) - 1) // 2)
    while len(candidate_moves) < num_candidates:
        i, j = random.sample(high_indices, 2)
        move = (min(i, j), max(i, j))
        candidate_moves.add(move)
    candidate_moves = list(candidate_moves)

    for move in candidate_moves:
        if (soln[move[0]], soln[move[1]]) in tabu_set:
            continue
        soln_mod = soln.copy()
        i, j = move
        soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
        soln_mod[-1] = soln_mod[0]
        nbhd.append(soln_mod)
        moves.append(move)

    return nbhd, moves
