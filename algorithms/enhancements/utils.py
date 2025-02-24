import numpy as np

def adaptive_perturb(soln: list[int], stagnant_ctr: int, 
                                base_strength: float = 0.1, max_strength: float = 0.7) -> list[int]:
    """
    Enhanced adaptive perturbation function that applies different perturbation types
    based on the level of stagnation.
    
    Args:
        soln: The current solution (list of integers).
        stagnant_ctr: The current count of consecutive non-improving iterations.
        base_strength: The base fraction of the solution to perturb.
        max_strength: The maximum fraction for moderate perturbation.
    
    Returns:
        A new solution after perturbation.
    """
    soln_new = soln.copy()
    n = len(soln_new)
    
    # If stagnation is severe, use a ruin-and-recreate strategy.
    # Ruin-and-recreate: remove a proportion (e.g., 20%) of elements and reinsert them randomly.
    num_remove = max(1, int(n * 0.2))
    indices = np.random.choice(n, size=num_remove, replace=False)
    remaining = [soln_new[i] for i in range(n) if i not in indices]
    removed = [soln_new[i] for i in indices]
    np.random.shuffle(removed)
    for elem in removed:
        pos = np.random.randint(0, len(remaining)+1)
        remaining.insert(pos, elem)
    return remaining

