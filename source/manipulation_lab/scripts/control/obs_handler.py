"""Handles observations from the environment."""
from isaaclab.utils.math import subtract_frame_transforms
import numpy as np
import torch

class ObservationHandler:
    def __init__(self, env, use_oracle):
        self.env = env
        self.scene = self.env.unwrapped.scene
        self.robot = self.scene.articulations["robot"]
        self.sensors = self.scene.sensors

        self.use_oracle = use_oracle
        self.robot_root_pose_w = self.robot.data.root_pose_w
        self.robot_xyz_w, self.robot_quat_w = self.robot_root_pose_w[:, 0:3], self.robot_root_pose_w[:, 3:7]

    def get_obs(self):
        return {
            "sensors": self._get_sensor_obs(),
            "robot": self._get_robot_obs(),
            "oracle": self._get_oracle_obs() if self.use_oracle else None
        }

    def _get_sensor_obs(self):
        sensor_obs = {}

        # Iterate over all sensors in scene and get their output
        for sensor_name, sensor_obj in self.sensors.items():
            # sensor_obj.data.output is a dictionary of type:tensor, e.g., {'rgb': tensor, 'depth': tensor}
            # Move tensors to CPU then convert to numpy for portability
            # IsaacSim captures observations as (N, ...) where N is number of environments
            # We squeeze out the environment dimension
            processed_output = {type: tensor.squeeze(0).cpu().numpy() for type, tensor in sensor_obj.data.output.items()}
            sensor_obs[sensor_name] = processed_output

        # sensor_obs is dict: {sensor_name: {type:ndarray}}}
        return sensor_obs

    def _get_robot_obs(self):
        joint_pos = self.robot.data.joint_pos.squeeze(0).cpu().numpy()
        joint_vel = self.robot.data.joint_vel.squeeze(0).cpu().numpy()

        ee_pose_w = self.robot.data.body_pose_w[:, -1]
        ee_xyz_r, ee_quat_r = self._in_robot_frame(ee_pose_w)
        ee_pose_r = np.concatenate((ee_xyz_r, ee_quat_r), axis=0)

        robot_obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "ee_pose_r": ee_pose_r,
            "ee_pose_w": ee_pose_w[0].cpu().numpy()
        }
        return robot_obs
    
    def _get_oracle_obs(self):
        oracle_obs = {}

        for object_name, obj in self.scene.rigid_objects.items():
            obj_xyz_r, obj_quat_r = self._in_robot_frame(obj.data.root_link_pose_w)
            pose_r = np.concatenate((obj_xyz_r, obj_quat_r), axis=0)
            # pose_w = obj.data.root_link_pose_w.squeeze(0).cpu().numpy()
            # vel_w = obj.data.root_link_vel_w.squeeze(0).cpu().numpy()

            oracle_obs[object_name] = {
                # "pose_w": pose_w,
                # "velocity_w": vel_w,
                "pose_r": pose_r,
            }

        return oracle_obs
    
    def _in_robot_frame(self, obj_root_pose_w: torch.Tensor):
        """
        Calculates the position of a given object in the robot's frame.
        """
        obj_xyz_w, obj_quat_w = obj_root_pose_w[:, 0:3], obj_root_pose_w[:, 3:7]

        obj_xyz_r, obj_quat_r = subtract_frame_transforms(
            t01=self.robot_xyz_w,
            q01=self.robot_quat_w,
            t02=obj_xyz_w,
            q02=obj_quat_w
        )
        
        return obj_xyz_r[0].cpu().numpy(), obj_quat_r[0].cpu().numpy()