import numpy as np
from itertools import permutations
import random
import math


def generate_permutation_matrices(n, num):
    """
    Generate `num` unique random permutation matrices of size n×n.

    Args:
        n (int): The size of each permutation matrix (n x n).
        num (int): Number of unique permutation matrices to generate.

    Returns:
        list[np.ndarray]: A list of unique permutation matrices.
    """
    # Total possible permutation matrices = n!
    total_possible = math.factorial(n)

    if num > total_possible:
        raise ValueError(f"Cannot generate {num} unique permutation matrices "
                         f"— only {total_possible} possible for n={n}.")

    # Generate all possible permutation matrices
    all_perms = list(permutations(range(n)))

    # Randomly sample 'num' unique permutations
    chosen_perms = random.sample(all_perms, num)

    # Convert each permutation into a permutation matrix
    matrices = []
    for perm in chosen_perms:
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        matrices.append(P)

    return matrices
def print_matrices(matrices):

    for idx, mat in enumerate(matrices, start=1):
        print(f"\n{"Matrix"} {idx}:")
        print("-" * (mat.shape[1] * 4))
        for row in mat:
            print("  ".join(f"{val:2d}" for val in row))
        print("-" * (mat.shape[1] * 4))


mats = generate_permutation_matrices(3,6)
print_matrices(mats)
