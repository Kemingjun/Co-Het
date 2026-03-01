from typing import List, Tuple


class Config:
    """
    Global configuration for the Heterogeneous Robot Scheduling Problem (HRSP).
    Contains robot specifications, environmental parameters, and objective weights.
    """

    # ==========================================================================
    # 2-Type Robot Configuration (Currently Active)
    # ==========================================================================
    ROBOT_TYPE_NUM: int = 2  # Number of distinct robot types
    ROBOT_NUM_LIST: List[int] = [4, 8]  # Count of robots per type (e.g., 4 Type-A, 8 Type-B)
    INDEX_LIST: List[int] = [4, 12]  # Cumulative indices for robot mapping

    # Range of indices for each type (1-indexed for mathematical consistency)
    # Type 1: Robots 1 to 4 | Type 2: Robots 5 to 12
    TYPE_LIST: List[List[int]] = [[1, 5], [5, 13]]

    ROBOT_NUM: int = 12  # Total number of robots (sum of ROBOT_NUM_LIST)

    # ==========================================================================
    # 3-Type Robot Configuration (Interface Reserved)
    # ==========================================================================
    """
    # Template for 3rd robot type integration:
    ROBOT_TYPE_NUM = 3
    ROBOT_NUM_LIST = [4, 6, 8]
    INDEX_LIST = [4, 10, 18]
    TYPE_LIST = [[1, 5], [5, 11], [11, 19]]
    ROBOT_NUM = 18
    """

    # ==========================================================================
    # Environmental & Optimization Parameters
    # ==========================================================================
    WEIGHT: float = 0.5  # Objective weight for multi-objective optimization
    DEPOT: Tuple[float, float] = (0.0, 0.0)  # Starting and ending coordinates for all robots
    VELOCITY: float = 1.0  # Standard velocity coefficient