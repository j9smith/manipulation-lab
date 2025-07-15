"""Applies actions to the robot."""

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from typing import Optional
import torch

class ActionHandler:
    def __init__(self, env, control_mode):
        self.scene = env.unwrapped.scene
        self.robot = env.unwrapped.scene.articulations["robot"]
        self.control_mode = control_mode

        self.diff_ik_controller: Optional[DifferentialIKController] = None
        self.ee_body_idx = None
        self.ee_jacobi_idx = None

    def _initialise_ik_controller(self, use_relative_mode=False):
        """
        Initialises the IK controller object
        """
        if self.ee_body_idx is None:
            self.ee_body_idx = self._get_end_effector_body_index()

        if self.ee_jacobi_idx is None:
            self.ee_jacobi_idx = self._get_end_effector_jacobi_index()

        diff_ik_config = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=use_relative_mode,
            ik_method="dls"
        )

        self.diff_ik_controller = DifferentialIKController(
            cfg=diff_ik_config,
            num_envs=1,
            device="cuda"
        )

    def _get_end_effector_body_index(self):
        """
        Returns the index of the end effector in the body schema.
        """
        return self.robot.num_bodies - 1

    def _get_end_effector_jacobi_index(self):
        """
        Returns the index of the end effector in the Jacobian matrix. 
        """
        # Root body of fixed based robots is excluded from the Jacobian
        if self.robot.is_fixed_base:
            return self.robot.num_bodies - 2
        else:
            return self.robot.num_bodies - 1

    def _calculate_desired_joint_positions(self, action):
        """
        Returns desired joint positions for the robot.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.robot.device)
        else:
            action = action.to(self.robot.device)

        # Extract only the Cartesian elements of the action tensor
        cartesian_action = action[:, :6]

        # Extract only the gripper element of the action tensor (dim=1)
        # Then broadcast the gripper action to the two finger joints (dim=2)
        gripper_action = action[:, 6:].repeat(1, 2)

        # Get the Jacobian matrix for the end effector
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, :7]

        # Get the end effector and the root pose in the world frame
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]
        root_pose_w = self.robot.data.root_pose_w
        current_joint_pos = self.robot.data.joint_pos[:, :7]

        # Calculate the end effector pose in the robot's frame
        ee_pose_r, ee_quat_r = subtract_frame_transforms(
            t01=root_pose_w[:, 0:3], # Root position (x, y, z)
            q01=root_pose_w[:, 3:7], # Root orientation (qw, qx, qy, qz)
            t02=ee_pose_w[:, 0:3], # EE position (x, y, z)
            q02=ee_pose_w[:, 3:7] # EE orientation (qw, qx, qy, qz)
        )

        # Set the target pose for the IK controller
        if self.diff_ik_controller.cfg.use_relative_mode:
            self.diff_ik_controller.set_command(
                command=cartesian_action,
                ee_pos=ee_pose_r,
                ee_quat=ee_quat_r
            )
        else: self.diff_ik_controller.set_command(command=cartesian_action)

        # Calculate joint positions to reach target pose
        desired_joint_pos = self.diff_ik_controller.compute(
            ee_pos=ee_pose_r,
            ee_quat=ee_quat_r,
            jacobian=jacobian,
            joint_pos=current_joint_pos
        )

        # Calculate the new gripper positions
        previous_gripper_pos = self.robot.data.joint_pos.clone()[:, 7:]
        new_gripper_pos = previous_gripper_pos + gripper_action

        # Concatenate the desired joint positions with the gripper action
        desired_joint_pos = torch.cat([desired_joint_pos, new_gripper_pos], dim=1)

        return desired_joint_pos

    def apply(self, action):
        """
        Routes the action to the appropriate method based on the control mode
        """
        if self.control_mode == "delta_cartesian":
            self._apply_delta_cartesian(action)
        elif self.control_mode == "absolute_cartesian":
            self._apply_absolute_cartesian(action)
        elif self.control_mode == "joint_pos_target":
            self._apply_joint_pos_target(action)
        elif self.control_mode == None:
            raise ValueError(
                "Control mode is not set. Please specify a control mode: "
                "delta_cartesian, absolute_cartesian, or joint_pos_target")
        else:
            raise ValueError(
                f"Invalid control mode: {self.control_mode}. "
                f"Valid control modes: delta_cartesian, absolute_cartesian, joint_pos_target"
            )

    def _apply_delta_cartesian(self, action):
        """
        Applies Cartesian deltas commands to the robot via diff. IK
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != 7:
            raise ValueError(
                "Delta Cartesian actions must be of length 6. "
                f"The provided action has shape {action.shape}."
            )
        self._apply_ik_action(action, use_relative_mode=True)

    def _apply_absolute_cartesian(self, action):
        """
        Applies absolute Cartesian commands to the robot via IK
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != 7:
            raise ValueError(
                "Absolute Cartesian actions must be of length 7. "
                f"The provided action has shape {action.shape}."
            )
        self._apply_ik_action(action, use_relative_mode=False)

    def _apply_ik_action(self, action, use_relative_mode=False):
        """
        Applies IK actions to the robot
        """
        if self.diff_ik_controller is None:
            self._initialise_ik_controller(use_relative_mode)

        desired_joint_pos = self._calculate_desired_joint_positions(action)
        self.robot.set_joint_position_target(desired_joint_pos)
        self.scene.write_data_to_sim()

    def _apply_joint_pos_target(self, action):
        """
        Applies joint position target commands to the robot
        """
        pass