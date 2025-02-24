import pandas as pd
import random

total_iter = 10
dms = {}
soln_inits = {}
pois = ["20", "40", "80"]
#pois = ["500"]
tenures = [10, 20, 40]

for poi in pois:
    csv = f"data/input/poi_{poi}.csv"
    distance_matrix: list[list[int]] = pd.read_csv(csv).values[:, 1:]
    soln_init = list(range(distance_matrix.shape[0]))
    random.shuffle(soln_init)
    dms.update({poi: distance_matrix})
    soln_inits.update({poi: soln_init})
