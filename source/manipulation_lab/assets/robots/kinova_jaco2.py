"""
This is a modified version of the Rethink Robotics Sawyer configuration published by the Isaac Lab Project.

The original config has been modified to a configclass for compatibility with Hydra.
"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Rethink Robotics arms.

The following configuration parameters are available:

* :obj:`SAWYER_CFG`: The Sawyer arm without any tool attached.

Reference: https://github.com/RethinkRobotics/sawyer_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.utils import configclass

@configclass
class N7S300Cfg(ArticulationCfg):
    manipulation = {
        "arm_joint_names": [
            "j2n7s300_joint_1", "j2n7s300_joint_2", "j2n7s300_joint_3", "j2n7s300_joint_4",
            "j2n7s300_joint_5", "j2n7s300_joint_6", "j2n7s300_joint_7"
        ],
        "gripper_joint_names": [
            "j2n7s300_joint_finger_1", "j2n7s300_joint_finger_2", "j2n7s300_joint_finger_3",
            "j2n7s300_joint_finger_tip_1", "j2n7s300_joint_finger_tip_2", "j2n7s300_joint_finger_tip_3"
            ],
        "ee_body_name": "j2n7s300_end_effector"
    }
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Jaco2/J2N7S300/j2n7s300_instanceable.usd",
        scale=None,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "j2n7s300_joint_1": 0.0,
            "j2n7s300_joint_2": 2.76,
            "j2n7s300_joint_3": 0.0,
            "j2n7s300_joint_4": 2.0,
            "j2n7s300_joint_5": 2.0,
            "j2n7s300_joint_6": 0.0,
            "j2n7s300_joint_7": 0.0,
            "j2n7s300_joint_finger_[1-3]": 0.2,  # close: 1.2, open: 0.2
            "j2n7s300_joint_finger_tip_[1-3]": 0.2,  # close: 1.2, open: 0.2
        },
    )
    actuators: dict = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_[1-7]"],
            effort_limit_sim={
                ".*_joint_[1-2]": 80.0,
                ".*_joint_[3-4]": 40.0,
                ".*_joint_[5-7]": 20.0,
            },
            stiffness={
                ".*_joint_[1-4]": 400.0,
                ".*_joint_[5-7]": 350.0,
            },
            damping={
                ".*_joint_[1-4]": 80.0,
                ".*_joint_[5-7]": 70.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
            effort_limit_sim=2.0,
            stiffness=1.2,
            damping=0.01,
        ),
    }

    soft_joint_pos_limit_factor: float = 1.0

@configclass
class N6S300Cfg(ArticulationCfg):
    manipulation = {
        "arm_joint_names": [
            "j2n6s300_joint_1", "j2n6s300_joint_2", "j2n6s300_joint_3",
            "j2n6s300_joint_4", "j2n6s300_joint_5", "j2n6s300_joint_6",
        ],
        "gripper_joint_names": [
            "j2n6s300_joint_finger_1", "j2n6s300_joint_finger_2", "j2n6s300_joint_finger_3",
            "j2n6s300_joint_finger_tip_1", "j2n6s300_joint_finger_tip_2", "j2n6s300_joint_finger_tip_3"
            ],
        "ee_body_name": "j2n6s300_end_effector"
    }
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Jaco2/J2N6S300/j2n6s300_instanceable.usd",
        scale=None,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "j2n6s300_joint_1": 0.0,
            "j2n6s300_joint_2": 2.76,
            "j2n6s300_joint_3": 0.0,
            "j2n6s300_joint_4": 2.0,
            "j2n6s300_joint_5": 2.0,
            "j2n6s300_joint_6": 0.0,
            "j2n6s300_joint_finger_[1-3]": 0.2,  # close: 1.2, open: 0.2
            "j2n6s300_joint_finger_tip_[1-3]": 0.2,  # close: 1.2, open: 0.2
        },
    )
    actuators: dict = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_[1-6]"],
            effort_limit_sim={
                ".*_joint_[1-2]": 80.0,
                ".*_joint_3": 40.0,
                ".*_joint_[4-6]": 20.0,
            },
            stiffness={
                ".*_joint_[1-3]": 400.0,
                ".*_joint_[4-6]": 350.0,
            },
            damping={
                ".*_joint_[1-3]": 80.0,
                ".*_joint_[4-6]": 70.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_finger_[1-3]", ".*_finger_tip_[1-3]"],
            effort_limit_sim=2.0,
            stiffness=1.2,
            damping=0.01,
        ),
    }

    soft_joint_pos_limit_factor: float = 1.0
