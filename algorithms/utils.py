import random
import pandas as pd
import time
import config


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
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in tabu_list and (j, i) not in tabu_list:
                soln_mod: list[int] = soln.copy()
                soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
                soln_mod[-1] = soln_mod[0]
                nbhd.append(soln_mod)
                moves.append((i, j))
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

        if val_curr < val(soln_best) or moves[idx] not in tabu_list:
            if val_curr < val_best:
                val_best = val_curr
                nbhr_best = nbhr_curr
                move_best = moves[idx]

    return nbhr_best, move_best
