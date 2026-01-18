import time
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List
import torch.nn.functional as F

"""
Method 1: Dynamic Programming (Held-Karp Algorithm)
This is the standard algorithmic solution. It uses bitmasking to keep track of visited cities and recursion with memoization to avoid re-calculating the same sub-paths.
Time Complexity: O(n^2 * 2^n) Space Complexity: O(n * 2^n)
"""
import sys

def solve_tsp_dynamic(dist_matrix):
    """
    Solves TSP using Held-Karp algorithm (Dynamic Programming).
    Returns (min_cost, path).
    """
    n = len(dist_matrix)
    
    # memoization table: key=(mask, current_city), value=min_cost
    memo = {}
    # parent table to reconstruct the path: key=(mask, current_city), value=prev_city
    parent = {}

    # State: mask (bitmask of visited cities), pos (current city index)
    def visit(mask, pos):
        # Base case: All cities visited? Return distance to return to start (city 0)
        if mask == (1 << n) - 1:
            return dist_matrix[pos][0] or float('inf') # Return inf if no path back

        state = (mask, pos)
        if state in memo:
            return memo[state]

        ans = float('inf')
        best_next_city = -1

        # Try visiting every unvisited city
        for city in range(n):
            if (mask >> city) & 1 == 0:  # If city not in mask
                new_ans = dist_matrix[pos][city] + visit(mask | (1 << city), city)
                if new_ans < ans:
                    ans = new_ans
                    best_next_city = city
        
        memo[state] = ans
        parent[state] = best_next_city
        return ans

    # Start from city 0, with only city 0 in the mask (1 << 0)
    min_cost = visit(1, 0)
    
    # --- Path Reconstruction ---
    path = [0]
    curr_mask = 1
    curr_city = 0
    
    # We trace the path until we have visited all cities
    for _ in range(n - 1):
        next_city = parent.get((curr_mask, curr_city))
        path.append(next_city)
        curr_mask |= (1 << next_city)
        curr_city = next_city
        
    path.append(0) # Return to start
    
    return min_cost, path

"""
Method 2: Brute Force (Permutations)
If your dataset is very small (n <= 10), this code is much easier to read and debug. It simply generates every possible order of cities and calculates the total distance.
"""

from itertools import permutations

def solve_tsp_brute_force(dist_matrix):
    n = len(dist_matrix)
    cities = list(range(n))
    min_path = None
    min_dist = float('inf')

    # Try every permutation of cities (starting at 0)
    # We fix the start node (0) to reduce permutations by factor of n
    for perm in permutations(cities[1:]):
        current_path = [0] + list(perm) + [0]
        current_dist = 0
        
        # Calculate distance for this path
        for i in range(len(current_path) - 1):
            u, v = current_path[i], current_path[i+1]
            current_dist += dist_matrix[u][v]
        
        if current_dist < min_dist:
            min_dist = current_dist
            min_path = current_path

    return min_dist, min_path

"""
Method 3: RL-guided DIDP
"""

# 1. State and Model Definitions
@dataclass(frozen=True)
class State:
    current_city: int
    visited_mask: int
    g_cost: float

class TSPGuidanceGNN(nn.Module):
    def __init__(self, node_dim=3, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Global mean pooling for simplicity in this example
        graph_latent = torch.mean(x, dim=1) 
        return self.fc(graph_latent)

# 2. Solver Class
class RGDIDPSolver:
    def __init__(self, model: nn.Module, beam_width: int):
        self.model = model
        self.beam_width = beam_width

    def get_h_batch(self, states: List[State], coords: np.ndarray):
        num_cities = len(coords)
        batch_features = []
        for s in states:
            # Build feature: [x, y, visited_status]
            visited = np.array([(s.visited_mask >> i) & 1 for i in range(num_cities)])
            node_feat = np.column_stack((coords, visited))
            batch_features.append(node_feat)
        
        inputs = torch.tensor(np.array(batch_features), dtype=torch.float32)
        with torch.no_grad():
            return self.model(inputs).squeeze().numpy()

    def solve(self, dist_matrix: np.ndarray, coords: np.ndarray):
        num_cities = len(dist_matrix)
        # Initial State: Start at City 0
        beam = [State(0, 1 << 0, 0.0)]

        for _ in range(1, num_cities):
            next_candidates = []
            for state in beam:
                for next_city in range(num_cities):
                    if not (state.visited_mask & (1 << next_city)):
                        new_g = state.g_cost + dist_matrix[state.current_city][next_city]
                        new_mask = state.visited_mask | (1 << next_city)
                        next_candidates.append(State(next_city, new_mask, new_g))

            if not next_candidates: break
            
            # Batch Heuristic Calculation
            h_scores = self.get_h_batch(next_candidates, coords)
            
            # Sorting & Pruning
            scored = sorted(zip(next_candidates, h_scores), 
                            key=lambda x: x[0].g_cost + x[1])
            beam = [item[0] for item in scored[:self.beam_width]]

        # Final return to origin
        return min(s.g_cost + dist_matrix[s.current_city][0] for s in beam)

# Training Algorithm: Fitted Value Iteration
# This script demonstrates how to generate trajectories and update the GNN.

import torch.optim as optim



# MAIN FUNCTION: Running an Instance
if __name__ == "__main__":
    # Setup dummy data for 5 cities
    num_cities = 12 # >= 13: long runtime
    coords = np.random.rand(num_cities, 2)
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(-1))
    # dist_matrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]];

    # Initialize Model and Solver
    start_time = time.perf_counter()
    
    model = TSPGuidanceGNN()
    solver = RGDIDPSolver(model, beam_width=5)    
    final_cost = solver.solve(dist_matrix, coords)
    print(f"Final Optimized Tour Cost: {final_cost}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")

    # --- Example Usage --- Comparision
    # Distance matrix (Adjacency Matrix)
    # Example: 5 cities
    start_time = time.perf_counter()
    cost, path = solve_tsp_dynamic(dist_matrix)
    print(f"Minimum Cost: {cost}")
    print(f"Optimal Path: {path}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")

    start_time = time.perf_counter()
    cost, path = solve_tsp_brute_force(dist_matrix)
    print(f"Brute Force Cost: {cost}")
    print(f"Brute Force Path: {path}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time} seconds")