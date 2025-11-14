import numpy as np
from itertools import permutations
import random
import math


def generate_random_permutation_matrices(n, num):
    seen = set()
    matrices = []

    while len(matrices) < num:
        perm = tuple(np.random.permutation(n))
        if perm in seen:
            continue

        seen.add(perm)

        P = np.zeros((n, n), dtype=int)
        P[np.arange(n), perm] = 1
        matrices.append(P)

    return matrices

def print_matrices(matrices):

    for idx, mat in enumerate(matrices, start=1):
        print(f"\n{"Matrix"} {idx}:")
        print("-" * (mat.shape[1] * 4))
        for row in mat:
            print("  ".join(f"{val:2d}" for val in row))
        print("-" * (mat.shape[1] * 4))


mats = generate_random_permutation_matrices(1000,1000)
print_matrices(mats)
