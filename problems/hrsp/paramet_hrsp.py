import torch
import torch.nn.functional as F


class paramet_hrsp:
    """
    Parameter configurations for the Heterogeneous Robot Scheduling Problem (HRSP).
    """

    # --- Universal Constants ---
    ROBOT_VELOCITY = 1.0
    DEPOT = torch.tensor([0, 0])
    WEIGHT = 0.5  # Weight for balancing distance and tardiness in cost calculation
    time_norm = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # OPTION 1: Configuration for 2 Robot Types (Current Default)
    # =========================================================================
    ROBOT_TYPE_NUM = 2
    ROBOT_NUM = 12
    ROBOT_NUM_LIST = torch.tensor([4, 8])

    # =========================================================================
    # OPTION 2: Configuration for 3 Robot Types (Supplemented Interface)
    # To use this, uncomment the lines below and comment out Option 1
    # =========================================================================
    # ROBOT_TYPE_NUM = 3
    # ROBOT_NUM = 18
    # ROBOT_NUM_LIST = torch.tensor([4, 6, 8])
    # =========================================================================

    # --- Derived Attributes ---
    # Automatically generates one-hot encoding for each robot based on its type
    robot_one_hot = F.one_hot(
        torch.cat([
            torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(ROBOT_NUM_LIST)
        ]),
        num_classes=ROBOT_TYPE_NUM
    ).float().to(device)

    # Pre-calculates type indices for each robot (e.g., [0,0,0,0,1,1,1,1,1,1,1,1])
    robot_type_indices_list = []
    for i in range(ROBOT_TYPE_NUM):
        robot_type_indices_list.append(
            torch.full((ROBOT_NUM_LIST[i].item(),), i, dtype=torch.long)
        )

    robot_type_indices = torch.cat(robot_type_indices_list).to(device)