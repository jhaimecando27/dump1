from algorithms.utils import neighborhood, val, best_admissible_soln, p2_neighborhood
import math
import random


def tabu_search(
    soln_init: list[int],
    tabu_tenure,
    iter_max=100,
) -> tuple[int, list[int]]:
    """
    Modified Tabu Search implementation following the provided pseudocode.

    Args:
        soln_init: Initial solution
        tabu_tenure: Maximum size of the tabu list
        iter_max: Maximum number of iterations

    Returns:
        tuple: Value of best solution and tracker of best solution values over iterations
    """
    tabu_list: list[list[int]] = []
    soln_curr: list[int] = soln_init.copy()
    soln_best: list[int] = soln_init.copy()
    best_value: int = val(soln_best)
    soln_best_tracker: list[int] = [best_value]

    for iter_ctr in range(iter_max):
        # Generate neighbor by adding an unpicked item (using the existing neighborhood generator)
        nbhr_soln = generate_neighbor(soln_curr, tabu_list)

        # Add the neighbor solution to the tabu list
        tabu_list.append(nbhr_soln)

        # Evaluate the neighbor solution
        nbhr_value = val(nbhr_soln)

        # If the neighbor solution is better than the best solution, update the best solution
        if nbhr_value < best_value:
            soln_best = nbhr_soln.copy()
            best_value = nbhr_value
            soln_best_tracker.append(best_value)

        # Move to the neighbor solution
        soln_curr = nbhr_soln.copy()

        # Maintain tabu list size
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_value, soln_best_tracker


def generate_neighbor(soln: list[int], tabu_list: list[list[int]]) -> list[int]:
    """
    Generate a neighboring solution that is not in the tabu list.
    In this context, we're adapting the function to work with the TSP problem
    instead of the knapsack problem mentioned in the pseudocode.

    Args:
        soln: Current solution
        tabu_list: List of recent solutions

    Returns:
        list: A new solution not in the tabu list
    """
    # Keep generating neighbors until one is found that's not in the tabu list
    while True:
        # Create a modified copy of the solution
        nbhr_soln = soln.copy()

        # Perform a swap operation between two random positions
        n = len(soln) - 1  # Exclude the last element which is a copy of the first
        i, j = random.sample(range(n), 2)
        nbhr_soln[i], nbhr_soln[j] = nbhr_soln[j], nbhr_soln[i]

        # Ensure the last element is the same as the first (to maintain the cycle)
        nbhr_soln[-1] = nbhr_soln[0]

        # Check if this solution is in the tabu list
        if nbhr_soln not in tabu_list:
            return nbhr_soln
