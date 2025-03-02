import numpy as np
import csv


def create_random_distance_matrix(n_points, dimension):
    """
    Generate random points and compute their pairwise Euclidean distance matrix.

    Parameters:
        n_points (int): Number of random points.
        dimension (int): Dimensionality of each point.

    Returns:
        points (np.ndarray): Random points of shape (n_points, dimension).
        distance_matrix (np.ndarray): Pairwise Euclidean distance matrix (n_points x n_points).
    """
    # Generate random points in the range [0, 1]
    points = np.random.rand(n_points, dimension)

    # Compute the distance matrix using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return points, distance_matrix


def save_matrix_to_csv_with_headers(matrix, filename):
    """
    Save a NumPy matrix to a CSV file with both column and row headers.
    The headers are numbers starting from 1.
    """
    n_rows, n_cols = matrix.shape
    # Create header row (empty first cell then column numbers)
    header = [""] + [str(i + 1) for i in range(n_cols)]

    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for i in range(n_rows):
            # Row header is the row number (starting at 1)
            row = [str(i + 1)] + ["{:.4f}".format(matrix[i, j]) for j in range(n_cols)]
            writer.writerow(row)
    print(f"Distance matrix saved to {filename}")


if __name__ == "__main__":
    # Specify problem size and dimension
    n_points = 1000  # number of cities
    dimension = 2  # using 2D points

    # Create random distance matrix
    points, dist_matrix = create_random_distance_matrix(n_points, dimension)

    # Save the distance matrix to a CSV file with headers
    csv_filename = f"poi_{n_points}.csv"
    save_matrix_to_csv_with_headers(dist_matrix, csv_filename)
