"""Handles observations from the environment."""

class ObservationHandler:
    def get_obs(self, env):
        obs = {}

        for sensor_name, sensor_obj in env.unwrapped.scene.sensors.items():
            sensor_data = sensor_obj.data
