import numpy as np
import random
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
    header = [""] + [str(i+1) for i in range(n_cols)]
    
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for i in range(n_rows):
            # Row header is the row number (starting at 1)
            row = [str(i+1)] + ["{:.4f}".format(matrix[i, j]) for j in range(n_cols)]
            writer.writerow(row)
    print(f"Distance matrix saved to {filename}")

def tour_cost(tour, distance_matrix):
    """
    Compute the total cost (distance) of a TSP tour.
    """
    cost = 0.0
    n = len(tour)
    for i in range(n):
        cost += distance_matrix[tour[i], tour[(i+1) % n]]  # wrap-around for return
    return cost

def generate_neighbors(tour):
    """
    Generate neighbor tours by swapping any two cities.
    
    Returns:
        List of tuples: (neighbor_tour, move)
        where move is a tuple (i, j) indicating the swapped indices.
    """
    neighbors = []
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append((neighbor, (i, j)))
    return neighbors

def tabu_search_tsp(distance_matrix, max_iter=1000, tabu_tenure=10):
    """
    A simple tabu search for the Traveling Salesman Problem (TSP).
    
    Parameters:
        distance_matrix (np.ndarray): Distance matrix.
        max_iter (int): Maximum number of iterations.
        tabu_tenure (int): Number of iterations a move remains tabu.
        
    Returns:
        best_tour (list): The best tour found.
        best_cost (float): The cost of the best tour.
    """
    n = distance_matrix.shape[0]
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = tour_cost(current_tour, distance_matrix)
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    # Tabu list as a dictionary: move (tuple) -> expiration iteration
    tabu_list = {}
    
    for iter in range(max_iter):
        neighbors = generate_neighbors(current_tour)
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        for neighbor, move in neighbors:
            # Use a sorted tuple for consistency
            move = tuple(sorted(move))
            cost = tour_cost(neighbor, distance_matrix)
            
            # If move is tabu and doesn't satisfy aspiration, skip it
            if move in tabu_list and cost >= best_cost:
                continue
            
            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = neighbor
                best_move = move
        
        # If no valid neighbor is found, break the loop
        if best_neighbor is None:
            break
        
        current_tour = best_neighbor
        current_cost = best_neighbor_cost
        
        # Update best solution if improved
        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost
        
        # Add the move to the tabu list with an expiration iteration
        tabu_list[best_move] = iter + tabu_tenure
        
        # Remove expired moves from the tabu list
        expired_moves = [move for move, expire in tabu_list.items() if expire <= iter]
        for move in expired_moves:
            del tabu_list[move]
        
        # Optional: print progress
        # print(f"Iteration {iter}: current cost = {current_cost}, best cost = {best_cost}")
    
    return best_tour, best_cost

if __name__ == "__main__":
    # Specify problem size and dimension
    n_points = 500    # number of cities
    dimension = 2    # using 2D points
    
    # Create random distance matrix
    points, dist_matrix = create_random_distance_matrix(n_points, dimension)

    # Save the distance matrix to a CSV file with headers
    csv_filename = f"poi_{n_points}.csv"
    save_matrix_to_csv_with_headers(dist_matrix, csv_filename)
