import os
import sys
import time
import math
import random
import argparse
import copy
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# ==============================================================================
# Project Environment Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from Baseline.Util.load_data import read_excel
from Baseline.Util.generate_init_solution import generate_solution_nearest
from Baseline.Util.Solution import Solution
from Baseline.Util.util import *
from Baseline.Util.Config import Config

# ==============================================================================
# DABC Algorithm Parameters
# ==============================================================================
POP_SIZE = 100  # Total number of bees
EMPLOYED_SIZE = 10  # Number of employed bees (and nectars)
ONLOOKER_SIZE = 50  # Number of onlooker bees
LIMIT = 5000  # Maximum trials before abandoning a nectar source
R_TRIALS = 40  # Number of neighbor searches for onlookers


# ==============================================================================
# Helper Classes: Bee and Nectar
# ==============================================================================

class Bee:
    """Represents a bee in the colony with specific roles."""

    def __init__(self, bee_id: int, bee_type: int):
        self.id = bee_id
        self.type = bee_type  # 1: Employed, 2: Onlooker, 3: Scout

    def set_type(self, bee_type: int):
        self.type = bee_type

    def get_type(self) -> int:
        return self.type

    def get_id(self) -> int:
        return self.id


class Nectar:
    """Represents a nectar source associated with a specific solution."""

    def __init__(self, solution: Solution, fitness: Optional[float] = None):
        self.solution = solution
        self.search_num = 0
        self.fitness = fitness if fitness is not None else solution.get_fitness()
        self.bee: Optional[Bee] = None

    def set_bee(self, bee: Bee):
        self.bee = bee

    def add_search_num(self, count: int = 1):
        self.search_num += count

    def get_bee(self) -> Optional[Bee]:
        return self.bee


# ==============================================================================
# Neighbor Operators
# ==============================================================================

def neighbor_insertion(solution: Solution) -> Solution:
    """Performs a random task re-insertion neighbor search."""
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
        feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
        position = random.sample(list(feasible_positions), 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, robot_type)

    return Solution(solution.instance, sequence_map, path_init_task_map)


def neighbor_swap(solution: Solution) -> Solution:
    """Performs a random task swap neighbor search between two tasks."""
    task_list = list(range(1, solution.task_num + 1))
    task_1, task_2 = random.sample(task_list, 2)

    path_map = solution.get_path_map()

    def get_task_pos(p_map, t_id):
        pos = []
        for p_idx, path in p_map.items():
            for i, task in enumerate(path):
                if task == t_id:
                    pos.append((p_idx, i))
        return pos

    t1_pos = get_task_pos(path_map, task_1)
    t2_pos = get_task_pos(path_map, task_2)

    for pos in t1_pos:
        path_map[pos[0]][pos[1]] = task_2
    for pos in t2_pos:
        path_map[pos[0]][pos[1]] = task_1

    seq_map, p_init_map = path_map2sequence_map(path_map)
    return Solution(solution.instance, seq_map, p_init_map)


def get_neighbor_solution(solution: Solution) -> Solution:
    """Randomly selects and applies a neighbor operator."""
    operators = [neighbor_insertion, neighbor_swap]
    return random.choice(operators)(solution)


# ==============================================================================
# Core DABC Algorithm
# ==============================================================================

def get_index_roulette(nectar_list: List[Nectar], num: int) -> np.ndarray:
    """Selects nectar sources using Roulette Wheel Selection based on fitness."""
    costs = np.array([nc.fitness for nc in nectar_list])
    # Invert for minimization problem
    fitness_vals = ((-costs) - np.min(-costs)) + 1e-3
    probabilities = fitness_vals / fitness_vals.sum()
    return np.random.choice(np.arange(len(nectar_list)), size=num, replace=True, p=probabilities)


def run_dabc(file_path: str, max_iter: int, time_limit: int) -> Tuple[Solution, float, float]:
    """
    Executes the Discrete Artificial Bee Colony (DABC) algorithm.
    """
    start_t = time.time()
    print(f"[*] Loading instance: {file_path}")
    instance = read_excel(file_path)

    # 1. Initialization
    nectar_list: List[Nectar] = []
    onlooker_list: List[Bee] = []
    scout_list: List[Bee] = []

    # Initialize Employed Bees and Nectar Sources
    for i in range(EMPLOYED_SIZE):
        solution = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
        bee = Bee(i, 1)
        nectar = Nectar(solution)
        nectar.set_bee(bee)
        nectar_list.append(nectar)

    # Initialize Onlooker Bees
    for i in range(EMPLOYED_SIZE, EMPLOYED_SIZE + ONLOOKER_SIZE):
        onlooker_list.append(Bee(i, 2))

    # Track Global Best
    best_nectar = min(nectar_list, key=lambda x: x.fitness)
    best_fitness = best_nectar.fitness
    best_solution = best_nectar.solution

    print(f"[*] Starting DABC Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Loop
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # --- Stage 1: Employed Bees Phase ---
        new_nectar_list = []
        for nc in nectar_list:
            if (time.time() - start_t) > time_limit: break

            new_sol = get_neighbor_solution(nc.solution)
            new_fit = new_sol.get_fitness()

            if new_fit < nc.fitness:
                # Replace nectar source
                updated_nc = Nectar(new_sol, new_fit)
                updated_nc.set_bee(nc.get_bee())
                new_nectar_list.append(updated_nc)
                if new_fit < best_fitness:
                    best_solution, best_fitness, best_nectar = new_sol, new_fit, updated_nc
            else:
                nc.add_search_num()
                if nc.search_num > LIMIT and nc != best_nectar:
                    # Abandon nectar source and transition bee to scout
                    scout_bee = nc.get_bee()
                    scout_bee.set_type(3)
                    scout_list.append(scout_bee)
                else:
                    new_nectar_list.append(nc)
        nectar_list = new_nectar_list

        # --- Stage 2: Onlooker Bees Phase ---
        if nectar_list:
            onlooker_indices = get_index_roulette(nectar_list, len(onlooker_list))
            new_onlooker_list = []

            for i, onlooker_bee in enumerate(onlooker_list):
                if (time.time() - start_t) > time_limit: break

                target_nc = nectar_list[onlooker_indices[i]]
                best_neighbor, min_neighbor_fit = None, 1e9

                # Multiple neighborhood searches per onlooker
                for _ in range(R_TRIALS):
                    neighbor = get_neighbor_solution(target_nc.solution)
                    n_fit = neighbor.get_fitness()
                    if n_fit < min_neighbor_fit:
                        min_neighbor_fit, best_neighbor = n_fit, neighbor

                if min_neighbor_fit < target_nc.fitness:
                    # Onlooker becomes Employed, original Employed becomes Onlooker
                    new_nc = Nectar(best_neighbor, min_neighbor_fit)
                    onlooker_bee.set_type(1)
                    new_nc.set_bee(onlooker_bee)

                    # Store original bee to return to onlooker pool
                    old_bee = target_nc.get_bee()
                    old_bee.set_type(2)
                    new_onlooker_list.append(old_bee)

                    nectar_list[onlooker_indices[i]] = new_nc
                    if min_neighbor_fit < best_fitness:
                        best_solution, best_fitness, best_nectar = best_neighbor, min_neighbor_fit, new_nc
                else:
                    target_nc.add_search_num(R_TRIALS)
                    new_onlooker_list.append(onlooker_bee)
            onlooker_list = new_onlooker_list

        # --- Stage 3: Scout Bees Phase ---
        for scout in scout_list:
            new_sol = generate_solution_nearest(instance, random.uniform(0.75, 0.95))
            new_nc = Nectar(new_sol)
            scout.set_type(1)  # Revert to employed
            new_nc.set_bee(scout)
            nectar_list.append(new_nc)
            if new_nc.fitness < best_fitness:
                best_solution, best_fitness, best_nectar = new_sol, new_nc.fitness, new_nc
        scout_list = []

        if iteration % 20 == 0:
            elapsed = time.time() - start_t
            print(f"Iter {iteration:4d} | Best: {best_fitness:8.2f} | T: {elapsed:6.2f}s")

    return best_solution, best_fitness, time.time() - start_t


# ==============================================================================
# Execution Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Artificial Bee Colony (DABC) Solver")

    parser.add_argument(
        '-i', '--instance',
        type=str,
        required=True,
        help="Name or path of the instance Excel file (without .xlsx)."
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=100,
        help="Maximum number of DABC generations (default: 100)."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=1800,
        help="Time limit in seconds (default: 1800)."
    )

    args = parser.parse_args()

    # Handle file extension
    file_path = args.instance if args.instance.endswith(".xlsx") else f"{args.instance}.xlsx"

    # Execute Algorithm
    best_sol, best_fit, total_time = run_dabc(file_path, args.iter, args.time)

    # Summary Output
    print("\n" + "=" * 60)
    print(" DABC OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Final Fitness    : {best_fit:.4f}")
    print(f"Total Distance   : {best_sol.distance:.2f}")
    print(f"Total Tardiness  : {best_sol.tardiness:.2f}")
    print(f"Execution Time   : {total_time:.4f}s")
    print("=" * 60)