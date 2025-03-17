import numpy as np
import csv
import random

locs = [100, 500, 5000, 50000]


def calculate_distance(loc1, loc2):
    euclidean_distance = np.linalg.norm(loc1 - loc2)
    return euclidean_distance * random.uniform(0.8, 1.2)  # Add random factor


for num_locations in locs:
    # Distance calculation function with random noise

    # Set up random locations
    file_name = f"poi_{num_locations}.csv"
    locations = np.random.rand(num_locations, 2) * 10  # Scale to 10 km

    # Write CSV incrementally to avoid high memory usage
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(range(num_locations))  # Integer headers

        for i in range(num_locations):
            row = [i] + [
                calculate_distance(locations[i], locations[j])
                for j in range(num_locations)
            ]
            writer.writerow(row)  # Write each row one at a time to save memory
