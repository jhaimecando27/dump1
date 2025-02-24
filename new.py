import numpy as np
import math
from typing import Tuple, List, Optional
import config


class EnhancedTabuSearch:
    def __init__(self, distance_matrix: dict, max_iterations: int = 100):
        self.distance_matrix = distance_matrix
        self.max_iterations = max_iterations

    def calculate_solution_value(self, solution: List[int]) -> int:
        """Calculate total distance of the route."""
        value = 0
        n = len(solution)
        for i in range(n):
            poi_first = solution[i]
            poi_second = solution[(i + 1) % n]
            value += self.distance_matrix[str(n)][poi_first][poi_second]
        return value

    def generate_neighborhood(
        self, solution: List[int], tabu_list: List[List[int]], intensity: float = 1.0
    ) -> List[List[int]]:
        """Enhanced neighborhood generation with variable intensity."""
        neighborhood = []
        n = len(solution) - 1  # Exclude last element as it should match first

        # 2-opt moves
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_solution = solution.copy()
                # Reverse the segment between i and j
                new_solution[i:j] = reversed(new_solution[i:j])
                new_solution[-1] = new_solution[0]  # Ensure cycle closes
                if new_solution not in tabu_list:
                    neighborhood.append(new_solution)

        # Insert moves based on intensity
        if intensity > 0.5:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        new_solution = solution.copy()
                        value = new_solution.pop(i)
                        new_solution.insert(j, value)
                        new_solution[-1] = new_solution[0]
                        if new_solution not in tabu_list:
                            neighborhood.append(new_solution)

        return neighborhood

    def adaptive_perturbation(self, solution: List[int], stagnation: int) -> List[int]:
        """Enhanced perturbation with multiple strategies."""
        n = len(solution)
        perturbed = solution.copy()

        # Determine perturbation intensity based on stagnation
        intensity = min(0.8, 0.2 + (stagnation * 0.1))
        num_elements = max(2, int(n * intensity))

        if stagnation > 10:
            # Major perturbation: Multiple segment reversals
            for _ in range(3):
                i, j = sorted(np.random.choice(n - 1, 2, replace=False))
                perturbed[i:j] = reversed(perturbed[i:j])
        else:
            # Minor perturbation: Shuffle random segments
            indices = np.random.choice(n - 1, num_elements, replace=False)
            segment = [perturbed[i] for i in indices]
            np.random.shuffle(segment)
            for idx, value in zip(indices, segment):
                perturbed[idx] = value

        perturbed[-1] = perturbed[0]  # Ensure cycle closes
        return perturbed

    def calculate_adaptive_tenure(
        self, n: int, improvement_rate: float, stagnation: int
    ) -> int:
        """Dynamic tabu tenure calculation."""
        base_tenure = math.floor(n * 0.1)

        if improvement_rate > 0:
            # Reduce tenure when improving to intensify search
            tenure = max(base_tenure - 2, math.floor(n * 0.05))
        else:
            # Increase tenure with stagnation to diversify search
            tenure = min(base_tenure + stagnation, math.floor(n * 0.3))

        return tenure

    def adaptive_stopping(
        self, current_best: int, historical_best: List[int], iteration: int
    ) -> bool:
        """Intelligent stopping criteria."""
        if len(historical_best) < 10:
            return False

        recent_improvements = [
            abs(historical_best[i] - historical_best[i - 1])
            for i in range(len(historical_best) - 9, len(historical_best))
        ]

        avg_improvement = sum(recent_improvements) / len(recent_improvements)

        # Stop if average improvement is less than 0.1% and we're past 25% of max iterations
        return avg_improvement < (current_best * 0.001) and iteration > (
            self.max_iterations * 0.25
        )

    def search(self, initial_solution: List[int]) -> Tuple[int, List[int], float]:
        """Enhanced Tabu Search with adaptive mechanisms."""
        import time

        start_time = time.time()

        n = len(initial_solution)
        current_solution = initial_solution.copy()
        best_solution = initial_solution.copy()
        best_value = self.calculate_solution_value(best_solution)

        tabu_list = []
        tabu_tenure = math.floor(n * 0.1)  # Initial tenure

        stagnation_counter = 0
        historical_best = [best_value]
        improvement_rate = 0

        for iteration in range(self.max_iterations):
            # Generate neighborhood with adaptive intensity
            intensity = 1.0 - (stagnation_counter * 0.1)
            neighborhood = self.generate_neighborhood(
                current_solution, tabu_list, intensity
            )

            if not neighborhood:
                current_solution = self.adaptive_perturbation(
                    best_solution, stagnation_counter
                )
                continue

            # Find best neighbor
            best_neighbor = min(
                neighborhood, key=lambda x: self.calculate_solution_value(x)
            )
            best_neighbor_value = self.calculate_solution_value(best_neighbor)

            # Update best solution if improved
            if best_neighbor_value < best_value:
                improvement = best_value - best_neighbor_value
                improvement_rate = improvement / best_value
                best_value = best_neighbor_value
                best_solution = best_neighbor.copy()
                historical_best.append(best_value)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                improvement_rate = 0

            # Update current solution
            current_solution = best_neighbor

            # Adaptive tabu tenure
            tabu_tenure = self.calculate_adaptive_tenure(
                n, improvement_rate, stagnation_counter
            )

            # Update tabu list
            tabu_list.append(best_neighbor)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            # Check stopping criteria
            if self.adaptive_stopping(best_value, historical_best, iteration):
                break

            # Apply perturbation if needed
            if stagnation_counter >= 5:
                current_solution = self.adaptive_perturbation(
                    best_solution, stagnation_counter
                )
                stagnation_counter = 0

        return best_value, historical_best, time.time() - start_time


# Run the algorithm
if __name__ == "__main__":
    # Load distance matrix
    distance_matrix = {
        "3": [[0, 1, 2], [1, 0, 3], [2, 3, 0]],
        "4": [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]],
        "5": [[0, 1, 2, 3, 4], [1, 0, 5, 6, 7], [2, 5, 0, 8, 9], [3, 6, 8, 0, 10], [4, 7, 9, 10, 0]],
    }

    # Define initial solution
    initial_solution = [0, 1, 2, 3, 4]

    # Initialize Enhanced Tabu Search
    ets = EnhancedTabuSearch(distance_matrix, max_iterations=1000)

    # Run Enhanced Tabu Search
    best_value, historical_best, execution_time = ets.search(initial_solution)

    print(f"Best value found: {best_value}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Historical best: {historical_best}")
