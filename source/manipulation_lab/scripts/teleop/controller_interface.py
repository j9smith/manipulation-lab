import logging
logger = logging.getLogger("ManipulationLab.ControllerInterface")

from multiprocessing import Array
import torch

class ControllerInterface:
    def __init__(self, remote_connection=False, port=8888, controller: str="xbox", 
                 expected_action_dims: int=7, expected_command_dims: int=4, **kwargs):

        self.controller = controller

        # Remote operation config
        self.remote_connection = remote_connection
        self.port = port
        self.controller_array: Array = None

        # Expected dimensions of the controller array
        self.expected_action_dims = expected_action_dims
        self.expected_command_dims = expected_command_dims

        if remote_connection:
            self._initialise_socket_connection()

        self._initialise_controller()

    def _initialise_controller(self):
        """
        Print control commands.
        """
        if self.controller == "xbox":
            logger.info("Xbox Controller: A = start, B = abort, X = finish, Y = pause")

    def _initialise_socket_connection(self):
        """
        Initialise a socket connection to accept teleoperation from remote server
        """
        from manipulation_lab.scripts.teleop.socket_listener import start_socket_listener
        logger.info(f"Initialising remote connection")
        expected_dims = self.expected_action_dims + self.expected_command_dims
        # Initialise a shared array between the main process and the socket listener (threaded)
        self.controller_array = Array('f', [0.0] * expected_dims)
        self.thread = start_socket_listener(
            self.controller_array, 
            port=self.port, 
            expected_dims=expected_dims
        )

    def get_action(self):
        if self.remote_connection and self.controller_array is not None:
            action = list(self.controller_array)[:self.expected_action_dims]
        else: 
            action = [0.0] * self.expected_action_dims

        return torch.tensor(action, dtype=torch.float32)

    def get_episode_commands(self):
        if self.remote_connection and self.controller_array is not None:
            episode_commands = list(self.controller_array)[self.expected_action_dims:]
        else:
            episode_commands = [0.0] * self.expected_command_dims

        if episode_commands[0] == 1: # A
            return "start"
        elif episode_commands[1] == 1: # B
            return "abort"
        elif episode_commands[2] == 1: # X
            return "finish"
        elif episode_commands[3] == 1: # Y
            return "pause"
        else:
            return None

    def _get_xbox_input(self):
        """
        Get input from an Xbox controller.
        """
        if self.remote_connection and self.controller_array is not None:
            pass
