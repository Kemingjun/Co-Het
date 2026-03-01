from Baseline.Util.util import *
from Baseline.Util.Solution import Solution
from Baseline.Util.Config import Config


# ==============================================================================
# Helper Functions for Greedy Insertion
# ==============================================================================

def search_insertions(instance: List[List], required_robot: List[int], current_robot_type: int,
                      destroyed_sequence_map: Dict, path_init_task_map: Dict,
                      destroyed_task: int, fitness_min_list: List[float],
                      greedy_destroyed_sequence_map_list: List[Optional[Dict]],
                      greedy_path_init_task_map_list: List[Optional[Dict]]):
    """
    Recursively searches for the optimal insertion positions across multiple robot types
    for a specific task to minimize the fitness increment.
    """
    # Base Case: All required robot types have been processed
    if current_robot_type == len(required_robot) + 1:
        fitness, _ = cal_fitness(instance, destroyed_sequence_map, path_init_task_map)
        if fitness < fitness_min_list[0]:
            fitness_min_list[0] = fitness
            greedy_destroyed_sequence_map_list[0] = copy_dict_int_dict(destroyed_sequence_map)
            greedy_path_init_task_map_list[0] = copy_dict_int_int(path_init_task_map)
        return

    # If this robot type is not required for the current task, skip to the next type
    if required_robot[current_robot_type - 1] == 0:
        search_insertions(instance, required_robot, current_robot_type + 1,
                          destroyed_sequence_map, path_init_task_map,
                          destroyed_task, fitness_min_list,
                          greedy_destroyed_sequence_map_list, greedy_path_init_task_map_list)
    else:
        # Explore all feasible insertion positions for the current robot type
        feasible_positions = get_feasible_insert_position(destroyed_sequence_map, path_init_task_map,
                                                          destroyed_task, current_robot_type)

        for pos in feasible_positions:
            # Create temporary maps to preserve state during recursion
            seq_map_tmp = copy_dict_int_dict(destroyed_sequence_map)
            path_map_tmp = copy_dict_int_int(path_init_task_map)

            insert_(seq_map_tmp, path_map_tmp, destroyed_task, pos, current_robot_type)

            search_insertions(instance, required_robot, current_robot_type + 1,
                              seq_map_tmp, path_map_tmp, destroyed_task,
                              fitness_min_list, greedy_destroyed_sequence_map_list,
                              greedy_path_init_task_map_list)


# ==============================================================================
# Initial Solution Generation Strategies
# ==============================================================================

def generate_solution_random(instance: List[List]) -> Solution:
    """
    Constructs an initial solution by inserting tasks in sequential order into
    randomly selected feasible positions.
    """
    task_num = len(instance)
    sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}

    for task in range(1, task_num + 1):
        for robot_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            if instance[task - 1][5][robot_type - 1] == 1:
                feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, robot_type)
                position = random.sample(list(feasible_positions), 1)[0]
                insert_(sequence_map, path_init_task_map, task, position, robot_type)

    return Solution(instance, sequence_map, path_init_task_map)


def generate_solution_greedy(instance: List[List]) -> Solution:
    """
    Constructs a solution by processing tasks sorted by deadline (Earliest Deadline First)
    and performing a greedy search for optimal insertion.
    """
    task_num = len(instance)
    # Sort tasks based on deadline (index 3)
    task_time_list = sorted([[t, instance[t - 1][3]] for t in range(1, task_num + 1)], key=lambda x: x[1])

    destroyed_sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}

    for task_info in task_time_list:
        destroyed_task = task_info[0]
        greedy_seq_list = [None]
        greedy_path_list = [None]
        fitness_min = [1e9]

        required_robot = instance[destroyed_task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=destroyed_task,
            fitness_min_list=fitness_min,
            greedy_destroyed_sequence_map_list=greedy_seq_list,
            greedy_path_init_task_map_list=greedy_path_list,
        )

        if greedy_seq_list[0] is None:
            logging.error(f"Infeasible construction: Task {destroyed_task} has no valid insertion point.")

        destroyed_sequence_map = greedy_seq_list[0]
        path_init_task_map = greedy_path_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


def generate_solution_greedy_random_shuffle(instance: List[List]) -> Solution:
    """
    Similar to greedy generation but processes tasks in a randomized order.
    """
    task_num = len(instance)
    task_list = list(range(1, task_num + 1))
    random.shuffle(task_list)

    destroyed_sequence_map = {}
    path_init_task_map = {robot: 0 for robot in range(1, Config.ROBOT_NUM + 1)}

    for task in task_list:
        greedy_seq_list = [None]
        greedy_path_list = [None]
        fitness_min = [1e9]

        required_robot = instance[task - 1][5]
        search_insertions(
            instance=instance,
            required_robot=required_robot,
            current_robot_type=1,
            destroyed_sequence_map=destroyed_sequence_map,
            path_init_task_map=path_init_task_map,
            destroyed_task=task,
            fitness_min_list=fitness_min,
            greedy_destroyed_sequence_map_list=greedy_seq_list,
            greedy_path_init_task_map_list=greedy_path_list,
        )

        if greedy_seq_list[0] is None:
            logging.error(f"Infeasible construction: Task {task} has no valid insertion point.")

        destroyed_sequence_map = greedy_seq_list[0]
        path_init_task_map = greedy_path_list[0]

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


def generate_solution_nearest(instance: List[List], factor: float = 0.9) -> Solution:
    """
    Constructs a solution using a Nearest-Neighbor heuristic combined with deadline pressure.
    Cost = factor * distance + (1 - factor) * time_delta.
    """
    task_num = len(instance)
    task_list = list(range(1, task_num + 1))

    # robot_state_map: {robot_id: [current_time, [current_x, current_y]]}
    robot_state_map = {i: [0.0, [0.0, 0.0]] for i in range(1, Config.ROBOT_NUM + 1)}
    path_map = {i: [] for i in range(1, Config.ROBOT_NUM + 1)}

    while task_list:
        # For each robot type, select the robot that becomes available the earliest
        robot_candidates = []
        for r_type in range(1, Config.ROBOT_TYPE_NUM + 1):
            type_indices = Config.TYPE_LIST[r_type - 1]
            type_robots = {k: v for k, v in robot_state_map.items() if type_indices[0] <= k < type_indices[1]}
            min_robot = min(type_robots, key=lambda k: type_robots[k][0])
            robot_candidates.append(min_robot)

        task_cost_list = []
        for task in task_list:
            task_deadline = instance[task - 1][3]
            task_pos = [instance[task - 1][1], instance[task - 1][2]]

            combined_cost = 0.0
            for robot in robot_candidates:
                r_time, r_pos = robot_state_map[robot][0], robot_state_map[robot][1]
                distance = math.fabs(task_pos[0] - r_pos[0]) + math.fabs(task_pos[1] - r_pos[1])
                delta_time = task_deadline - r_time
                combined_cost += (factor * distance + (1 - factor) * delta_time)

            task_cost_list.append([task, combined_cost])

        # Select the task with the minimum combined heuristic cost
        task_cost_list.sort(key=lambda x: x[1])
        selected_task = task_cost_list[0][0]

        # Calculate synchronization time (all required robots must arrive)
        arrival_times = []
        for robot in robot_candidates:
            r_time, r_pos = robot_state_map[robot][0], robot_state_map[robot][1]
            dist = math.fabs(r_pos[0] - instance[selected_task - 1][1]) + \
                   math.fabs(r_pos[1] - instance[selected_task - 1][2])
            arrival_times.append(r_time + dist / Config.VELOCITY)

        start_time = max(arrival_times)

        # Update states for selected robots
        for r_type_idx, robot in enumerate(robot_candidates):
            path_map[robot].append(selected_task)
            # Update time: start_time + processing duration for this robot type
            # Note: instance[4] logic preserved from original code
            robot_state_map[robot][0] = start_time + instance[4][r_type_idx]
            robot_state_map[robot][1] = [instance[selected_task - 1][1], instance[selected_task - 1][2]]

        task_list.remove(selected_task)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    return Solution(instance, sequence_map, path_init_task_map)