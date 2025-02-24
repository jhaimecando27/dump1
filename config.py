import pandas as pd
import numpy as np
import random

total_iter = 10
dms = {}
soln_inits = {}
pois = ["80"]
# pois = ["500"]
tenures = [10, 20, 40]
distance_matrix: list[list[int]] = []

for poi in pois:
    csv = f"data/input/poi_{poi}.csv"
    distance_matrix = pd.read_csv(csv).values[:, 1:]
    soln_init = list(range(distance_matrix.shape[0]))
    random.shuffle(soln_init)
    dms.update({poi: distance_matrix})
    soln_inits.update({poi: soln_init})


def create_random_priorities(n_points):
    """
    Create random priorities for each POI. Each priority is an integer between 1 and 5,
    where 5 is the highest.

    Returns:
        priorities (dict): Mapping from POI index (0-based) to its priority.
    """
    # Using np.random.randint to get values in [1, 5]
    priorities = {i: int(np.random.randint(1, 6)) for i in range(n_points)}
    return priorities


def create_random_extra_penalties(
    n_points, penalty_probability=0.05, max_extra_penalty=50
):
    """
    Create random extra penalties for specific paths (POI pairs).
    For each ordered pair (i, j), with probability penalty_probability, an extra penalty is assigned.

    Returns:
        extra_penalties (dict): Mapping from (i, j) tuples to an extra penalty cost (int).
    """
    extra_penalties = {}
    for i in range(n_points):
        for j in range(n_points):
            if i != j and random.random() < penalty_probability:
                # Assign a random extra penalty cost between 1 and max_extra_penalty
                extra_penalties[(i, j)] = random.randint(1, max_extra_penalty)
    return extra_penalties


priorities = create_random_priorities(int(pois[0]))
extra_penalties = create_random_extra_penalties(
    int(pois[0]), penalty_probability=0.05, max_extra_penalty=50
)
