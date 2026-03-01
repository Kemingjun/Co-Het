import os
import sys
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from gurobipy import Model, GRB, quicksum, Var

# ==============================================================================
# Project Environment Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# Local module imports
from Baseline.Util.load_data import read_excel
from Baseline.Util.util import get_distance, path_map2sequence_map
from Baseline.Util.Solution import Solution
from Baseline.Util.Config import Config

# Constants
BIG_M = 1e5


class HRSPGurobiSolver:
    """
    Mathematical Programming Solver for the Heterogeneous Robot Scheduling Problem (HRSP).
    """

    def __init__(self, instance_path: str):
        self.instance_name = os.path.basename(instance_path)
        self.instance = read_excel(instance_path)
        self.n_tasks = len(self.instance)

        # Configuration from Config
        self.robot_num = Config.ROBOT_NUM
        self.robot_types = Config.TYPE_LIST
        self.weight = Config.WEIGHT

        # Pre-compute Manhattan distance matrix
        self.dist_matrix = self._compute_distance_matrix()

        # Model
        self.model = Model(f"HRSP_{self.instance_name}")
        # Variables containers
        self.x = {}
        self.S = {}
        self.F = {}
        self.T = {}
        self.A_k = {}
        self.C_k = {}

    def _compute_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        dist_map = {}
        nodes = [[0.0, 0.0]] + [[item[1], item[2]] for item in self.instance]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                dist_map[i, j] = get_distance(nodes[i], nodes[j])
        return dist_map

    def build_model(self):
        N = self.n_tasks
        R = self.robot_num
        K = len(self.robot_types)

        # 1. Variables
        self.x = self.model.addVars(N + 1, N + 1, range(1, R + 1), vtype=GRB.BINARY, name="x")
        self.A_k = self.model.addVars(N + 1, range(1, K + 1), vtype=GRB.CONTINUOUS, name="A")
        self.C_k = self.model.addVars(N + 1, range(1, K + 1), vtype=GRB.CONTINUOUS, name="C")
        self.S = self.model.addVars(N + 1, vtype=GRB.CONTINUOUS, name="S")
        self.F = self.model.addVars(N + 1, vtype=GRB.CONTINUOUS, name="F")
        self.T = self.model.addVars(N + 1, vtype=GRB.CONTINUOUS, lb=0, name="T")

        # 2. Objective
        total_dist = quicksum(
            self.x[i, j, r] * self.dist_matrix[i, j]
            for i in range(N + 1) for j in range(1, N + 1) for r in range(1, R + 1)
        )
        total_tardiness = quicksum(self.T[i] for i in range(1, N + 1))

        self.model.setObjective(
            total_dist * self.weight + total_tardiness * (1 - self.weight),
            GRB.MINIMIZE
        )

        # 3. Constraints (Reduced for brevity, same logic as before)
        # C1: No self-loops
        for i in range(1, N + 1):
            self.model.addConstr(quicksum(self.x[i, i, r] for r in range(1, R + 1)) == 0)

        # C2: Coverage
        for k_idx, (r_start, r_end) in enumerate(self.robot_types):
            for j in range(1, N + 1):
                self.model.addConstr(
                    quicksum(self.x[i, j, r] for i in range(N + 1) for r in range(r_start, r_end)) == 1)
                self.model.addConstr(
                    quicksum(self.x[j, i, r] for i in range(N + 1) for r in range(r_start, r_end)) == 1)

        # C3: Flow conservation
        for j in range(1, N + 1):
            for r in range(1, R + 1):
                self.model.addConstr(
                    quicksum(self.x[i, j, r] for i in range(N + 1)) == quicksum(self.x[j, i, r] for i in range(N + 1))
                )

        # C4: Depot
        for r in range(1, R + 1):
            self.model.addConstr(quicksum(self.x[0, j, r] for j in range(N + 1)) == 1)
            self.model.addConstr(quicksum(self.x[i, 0, r] for i in range(N + 1)) == 1)

        # C5: Time
        for k in range(1, K + 1):
            self.model.addConstr(self.C_k[0, k] == 0)

        for k_idx, (r_start, r_end) in enumerate(self.robot_types):
            k = k_idx + 1
            for r in range(r_start, r_end):
                for j in range(1, N + 1):
                    for i in range(N + 1):
                        self.model.addConstr(
                            self.C_k[i, k] + (self.dist_matrix[i, j] / Config.VELOCITY)
                            - BIG_M * (1 - self.x[i, j, r]) <= self.A_k[j, k]
                        )

            for i in range(1, N + 1):
                self.model.addConstr(self.A_k[i, k] <= self.S[i])
                op_duration = self.instance[i - 1][4][k_idx]
                self.model.addConstr(self.S[i] + op_duration <= self.C_k[i, k])
                self.model.addConstr(self.C_k[i, k] <= self.F[i])

        # C6: Tardiness
        for i in range(1, N + 1):
            deadline = self.instance[i - 1][3]
            self.model.addConstr(self.F[i] - deadline <= self.T[i])

    def solve(self, time_limit: int = 3600) -> Optional[Solution]:
        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("LogFile", "gurobi_log.txt")
        self.model.optimize()

        if self.model.SolCount == 0:
            logging.warning(f"No feasible solution found for {self.instance_name}")
            return None

        path_map = {}
        for r in range(1, self.robot_num + 1):
            path = []
            active_arcs = {i: j for i in range(self.n_tasks + 1) for j in range(self.n_tasks + 1)
                           if self.x[i, j, r].X > 0.5}
            curr = active_arcs.get(0, 0)
            while curr != 0:
                path.append(curr)
                curr = active_arcs.get(curr, 0)
            path_map[r] = path

        sequence_map, path_init_task_map = path_map2sequence_map(path_map)
        return Solution(self.instance, sequence_map, path_init_task_map)


# ==============================================================================
# Interface for Unified Runner
# ==============================================================================
def buildModel(instance_path: str, time_limit: int = 3600) -> Optional[Dict[str, Any]]:
    """
    Standard interface called by baseline_runner.py.

    Args:
        instance_path: Path to the instance file.
        time_limit: Time limit in seconds.

    Returns:
        Dictionary containing optimization results or None if infeasible.
    """
    solver = HRSPGurobiSolver(instance_path)
    solver.build_model()
    solution = solver.solve(time_limit=time_limit)

    if solution:
        return {
            'objective': solver.model.ObjVal,
            'status': solver.model.Status,
            'distance': solution.distance,
            'tardiness': solution.tardiness,
            'runtime': solver.model.Runtime
        }
    else:
        return None


# ==============================================================================
# Standalone Execution Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gurobi Solver for HRSP")
    parser.add_argument("-i", "--instance", required=True, help="Instance file path")
    parser.add_argument("-t", "--time_limit", type=int, default=3600, help="Time limit")
    args = parser.parse_args()

    result = buildModel(args.instance, args.time_limit)

    if result:
        print("\n" + "=" * 30)
        print(f"Gurobi Result:")
        print(f"Objective: {result['objective']:.4f}")
        print(f"Distance : {result['distance']:.2f}")
        print(f"Tardiness: {result['tardiness']:.2f}")
        print(f"Time     : {result['runtime']:.2f}s")
        print("=" * 30)
    else:
        print("No feasible solution found.")