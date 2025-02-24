import random
import pandas as pd
import time
import config


def neighborhood(soln: list[int], tabu_list: list[list[int]]) -> list[list[int]]:
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

    # Make sure the last element is the same as the first element
    for i in range(len(soln) - 1):
        for j in range(i + 1, len(soln) - 1):
            soln_mod: list[int] = soln.copy()
            soln_mod[i], soln_mod[j] = soln_mod[j], soln_mod[i]
            soln_mod[-1] = soln_mod[0]
            nbhd.append(soln_mod)
    return nbhd


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
    nbhd: list[list[int]], tabu_list: list[list[int]]
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

    for nbhr_curr in nbhd:
        if nbhr_curr not in tabu_list:
            val_curr: int = val(nbhr_curr)
            if val_curr < val_best:
                val_best = val_curr
                nbhr_best = nbhr_curr

    return nbhr_best
