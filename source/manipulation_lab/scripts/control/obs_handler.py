"""Handles observations from the environment."""

class ObservationHandler:
    def __init__(self, env):
        self.env = env
        self.scene = self.env.unwrapped.scene
        self.robot = self.scene.articulations["robot"]
        self.sensors = self.scene.sensors

    def get_obs(self):
        return {
            "sensors": self._get_sensor_obs(),
            "robot": self._get_robot_obs()
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

        robot_obs = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel
        }
        return robot_obs