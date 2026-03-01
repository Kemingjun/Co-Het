import random
import bisect
from typing import Dict, List, Any, Optional
from Baseline.Util.util import *  # Assuming custom utility functions are defined here
from Baseline.Util.Config import Config


class Solution:
    """
    Represents a scheduling solution for the Heterogeneous Robot Scheduling Problem (HRSP).
    Includes methods for fitness evaluation, path reconstruction, and encoding.
    """

    def __init__(self, instance: List[List[Any]], sequence_map: Dict, path_init_task_map: Dict):
        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map

        self.distance: Optional[float] = None
        self.tardiness: Optional[float] = None
        self.fitness: Optional[float] = None
        self.feasible: Optional[bool] = None
        self.path_map: Optional[Dict[int, List[int]]] = None

        # Number of tasks in the sequence
        self.task_num = len(self.sequence_map.keys())
        self.info_map = {task: dict() for task in sequence_map.keys()}

        # Solution encoding/decoding
        self.code = self.get_code()
        self.hash_key = hash(tuple(self.code))

    def get_path_map(self) -> Dict[int, List[int]]:
        """Reconstructs the full path for each robot based on the sequence map."""
        if self.path_map is not None:
            return copy_dict_int_list(self.path_map)

        path_map = {}
        for path_index, init_task in self.path_init_task_map.items():
            if init_task == 0:
                path_map[path_index] = []
                continue

            path = [init_task]
            # Determine robot type based on index thresholds
            robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
            next_task = self.sequence_map[init_task][f'robot_{robot_type}_next_task']

            while next_task != 0:
                path.append(next_task)
                next_task = self.sequence_map[next_task][f'robot_{robot_type}_next_task']
            path_map[path_index] = path

        return path_map

    def get_code(self) -> List[int]:
        """Encodes the multi-robot schedule into a single list representation."""
        code = [0]
        for path_index in range(1, Config.ROBOT_NUM + 1):
            init_task = self.path_init_task_map[path_index]
            if init_task != 0:
                code.append(init_task)
                robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
                next_task = self.sequence_map[init_task][f'robot_{robot_type}_next_task']
                while next_task != 0:
                    code.append(next_task)
                    next_task = self.sequence_map[next_task][f'robot_{robot_type}_next_task']

            if path_index != Config.ROBOT_NUM:
                code.append(0)  # Separator between different robots
        return code

    def get_fitness(self) -> float:
        """
        Evaluates the objective function (Weighted Distance + Tardiness).
        Simulates the execution with robot synchronization for collaborative tasks.
        """
        if self.fitness is not None:
            return self.fitness

        total_distance = 0.0
        total_tardiness = 0.0

        enabled_task_list = []
        # Track readiness of each robot type for specific tasks
        # task_require_robot_state[task_idx][type_idx] = 1 means ready
        task_require_robot_state = [[0 for _ in range(Config.ROBOT_TYPE_NUM)] for _ in range(self.task_num)]

        # Initialize start tasks for each robot path
        for path_index, init_task in self.path_init_task_map.items():
            if init_task != 0:
                robot_type = bisect.bisect_left(Config.INDEX_LIST, path_index) + 1
                task_require_robot_state[init_task - 1][robot_type - 1] = 1
                # Initialize pre_complete_time as 0 for the first task
                self.info_map[init_task][f'robot_{robot_type}_pre_complete_time'] = 0

                # Check if all required types for this task are ready
                if self.instance[init_task - 1][5] == task_require_robot_state[init_task - 1]:
                    enabled_task_list.append(init_task)

        completed_task_num = 0
        while completed_task_num < self.task_num:
            if not enabled_task_list:
                print("[Error] No enabled transitions found. The solution is infeasible.")
                self.feasible = False
                self.fitness = 1e6
                return self.fitness

            # Select a random task from the ready queue for simulation
            idx = random.randrange(len(enabled_task_list))
            enabled_task = enabled_task_list.pop(idx)

            source_position = [self.instance[enabled_task - 1][1], self.instance[enabled_task - 1][2]]
            arrival_time_map = {}
            task_distance = 0.0

            # Calculate arrival times for all required robot types
            for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if self.instance[enabled_task - 1][5][robot_type - 1] == 0:
                    continue  # This type is not required for the current task

                pre_complete_time = self.info_map[enabled_task][f'robot_{robot_type}_pre_complete_time']

                # Determine origin position
                if self.sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] == 0:
                    pre_position = Config.DEPOT
                else:
                    prev_id = self.sequence_map[enabled_task][f'robot_{robot_type}_pre_task'] - 1
                    pre_position = [self.instance[prev_id][1], self.instance[prev_id][2]]

                # Travel cost calculation
                dist = get_distance(pre_position, source_position)
                travel_time = dist / Config.VELOCITY
                arrival_time = pre_complete_time + travel_time

                task_distance += dist
                arrival_time_map[robot_type] = arrival_time
                self.info_map[enabled_task][f"robot_{robot_type}_arrival_time"] = arrival_time

                # Update readiness for the next task in this robot's sequence
                next_task = self.sequence_map[enabled_task][f'robot_{robot_type}_next_task']
                if next_task != 0:
                    task_require_robot_state[next_task - 1][robot_type - 1] = 1
                    if task_require_robot_state[next_task - 1] == self.instance[next_task - 1][5]:
                        enabled_task_list.append(next_task)

            # Execution starts when the last required robot arrives (Synchronization)
            execute_time = max(arrival_time_map.values())
            complete_time_list = []

            for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
                if self.instance[enabled_task - 1][5][robot_type - 1] == 0:
                    continue

                # Complete time = Start + Heterogeneous operation time
                op_time = self.instance[enabled_task - 1][4][robot_type - 1]
                complete_time = execute_time + op_time

                next_task = self.sequence_map[enabled_task][f'robot_{robot_type}_next_task']
                if next_task != 0:
                    self.info_map[next_task][f'robot_{robot_type}_pre_complete_time'] = complete_time
                complete_time_list.append(complete_time)

            final_complete_time = max(complete_time_list)

            # Record task metrics
            self.info_map[enabled_task]["execute_time"] = execute_time
            self.info_map[enabled_task]["complete_time"] = final_complete_time

            # Tardiness = Max(0, Finish Time - Due Date)
            due_date = self.instance[enabled_task - 1][3]
            task_tardiness = max(final_complete_time - due_date, 0)

            total_distance += task_distance
            total_tardiness += task_tardiness
            completed_task_num += 1

            # Detailed cost breakdown for analysis
            self.info_map[enabled_task]['distance'] = task_distance
            self.info_map[enabled_task]['tardiness'] = task_tardiness
            self.info_map[enabled_task]['cost'] = (
                    task_distance * Config.WEIGHT + task_tardiness * (1 - Config.WEIGHT)
            )

        # Final Objective Assignment
        self.fitness = total_distance * Config.WEIGHT + total_tardiness * (1 - Config.WEIGHT)
        self.distance = total_distance
        self.tardiness = total_tardiness
        self.feasible = True

        return self.fitness

    def get_path_init_task_map(self) -> Dict:
        """Returns a deep copy of the initial task map."""
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self) -> Dict:
        """Returns a deep copy of the task sequence map."""
        return copy_dict_int_dict(self.sequence_map)