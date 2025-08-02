"""
Handles compilation of datasets with TeleopHandler.

Writes datasets in HDF5 format.

Follows the RLDS schema:
https://github.com/google-research/rlds
"""
import logging
logger = logging.getLogger("ManipulationLab.DatasetWriter")

import h5py
from h5py import string_dtype
from pathlib import Path
import numpy as np

class DatasetWriter:
    def __init__(
        self, 
        env_name: str, 
        task_name: str,
        task_language_instruction: str,
        task_phases: list[str],
        sim_dt: float, 
        buffer_size: int = 10000, 
        save_dir: str = "./datasets", 
        compression: str = "gzip", 
        rgb_as_uint8: bool = True,
        dagger_mode: bool = False,
        **kwargs
        ):

        # Metadata
        suffix = "dagger" if dagger_mode else "clean"
        self.base_dir = Path(save_dir) / env_name / task_name / suffix
        self.env_name = env_name
        self.task_name = task_name
        self.task_language_instruction = task_language_instruction
        self.task_phases = task_phases
        self.sim_dt = sim_dt
        self.buffer_size = buffer_size
        self.dagger_mode = dagger_mode

        # Episode data
        self.episode_data_buffer = {}
        self.episode_started = False
        self.episode_paused = False
        self.sim_time = 0.0

        # Dataset data
        self.episode_on_disk = False
        self.episode_file_path = None
        self.compression = compression
        self.rgb_as_uint8 = rgb_as_uint8

        self._reset_buffer()

        logger.debug(f"Using compression: {self.compression}")
        logger.debug(f"Using RGB as uint8: {self.rgb_as_uint8}")
        logger.info(f"Saving to {self.base_dir}")

    def _reset_buffer(self):
        """
        Initialises a clean episode data buffer.
        """
        self.episode_data_buffer = {
            "observations": [],
            "actions": [],
            "task_phase": [],
            "is_first": [],
            "is_last": [],
            "sim_time": []
        }

    def _reset_episode(self):
        """
        Reset the episode variables.
        """
        self.episode_started = False
        self.episode_paused = False
        self.episode_on_disk = False
        self.episode_file_path = None
        self._reset_buffer()

    def _get_next_episode_id(self):
        """
        Retrieves the next available episode ID.
        """
        existing_episodes = sorted(self.base_dir.glob("episode_*.hdf5"))
        existing_ids = [int(episode.stem.split("_")[1]) for episode in existing_episodes]
        return f"{max(existing_ids, default=-1) + 1:04d}"

    def start_episode(self):
        """
        Starts a new episode.
        """
        if self.episode_started:
            return

        self.episode_started = True

        if self.episode_paused:
            self.episode_paused = False
            return

        # Only create a new episode file if the episode is not paused
        self.episode_id = self._get_next_episode_id()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.episode_file_path = self.base_dir / f"episode_{self.episode_id}.hdf5"
        logger.info(f"Starting episode. Saving file to {self.episode_file_path}.")

    def end_episode(self):
        """
        Ends the current episode.
        """
        if not self.episode_started:
            return

        logger.info(f"Ending episode. Saving buffer to disk.")
        self.episode_data_buffer["is_last"][-1] = True # FIXME: Index error (-1 out of bounds)
        self._flush_buffer_to_disk()
        self.h5file.attrs["frame_count"] = self.time_dataset.shape[0]
        self.h5file.close()

        logger.info(f"Episode saved to {self.episode_file_path}")
        self._reset_episode()

    def pause_episode(self):
        """
        Pauses the current episode.
        """
        self.episode_paused = True

    def abort_episode(self):
        """
        Aborts the current episode and erases the dataset from disk.
        """
        logger.info(f"Episode aborted. Deleting {self.episode_file_path}.")
        if(self.episode_file_path and self.episode_file_path.exists()):
            self.episode_file_path.unlink()
        self._reset_episode()

    def append_frame(
        self, 
        obs: dict, 
        actions: dict,
        task_phase: int,
        is_first: bool, 
        is_last: bool, 
        sim_steps: int
        ):
        """
        Appends one frame of data to the dataset.
        """
        if not self.episode_started:
            return

        # Append data to buffer
        self.episode_data_buffer["observations"].append(obs) # List[dict]
        self.episode_data_buffer["actions"].append(actions) # List[dict]
        self.episode_data_buffer["task_phase"].append(task_phase) # List[int]
        self.episode_data_buffer["is_first"].append(is_first) # List[bool]
        self.episode_data_buffer["is_last"].append(is_last) # List[bool]
        self.episode_data_buffer["sim_time"].append(sim_steps * self.sim_dt) # List[float]

        # Flush buffer to disk if it's full
        if len(self.episode_data_buffer["actions"]) >= self.buffer_size:
            logger.info(f"Flushing buffer to disk")
            self._flush_buffer_to_disk()

    def _flush_buffer_to_disk(self):
        """
        Flushes the episode data buffer to disk. 
        """
        # Initialise the HDF5 file if it doesn't exist
        if not self.episode_on_disk:
            self.h5file = h5py.File(str(self.episode_file_path), "w")

            self.h5file.attrs["episode_id"] = self.episode_id
            self.h5file.attrs["env_name"] = self.env_name
            self.h5file.attrs["collection_mode"] = "dagger" if self.dagger_mode else "expert"
            self.h5file.attrs["task_name"] = self.task_name
            self.h5file.attrs["task_language_instruction"] = self.task_language_instruction
            self.h5file.attrs["task_phases"] = np.array(
                self.task_phases,
                dtype=string_dtype(encoding="utf-8")
            )
            self.h5file.attrs["sim_dt"] = self.sim_dt

            self.obs_group = self.h5file.create_group("observations")
            self.action_group = self.h5file.create_group("actions")
            self.flags_group = self.h5file.create_group("flags")
            self.task_phase_dataset = self.h5file.create_dataset(
                name="task_phase",
                shape=(0,),
                dtype='i',
                maxshape=(None,)
            )
            self.time_dataset = self.h5file.create_dataset(
                name="sim_time",
                shape=(0,),
                dtype='f',
                maxshape=(None,)
            )
            self.episode_on_disk = True

        def write_nested_dict(obs_list: list[dict], parent_group: h5py.Group):
            """
            Writes a nested dictionary to the HDF5 file.

            Data is passed as a list of dictionaries (one per frame), so we need to traverse them
            and write their data to a HDF5 group.

            e.g. obs = [
                        {'sensor_data':
                            {
                                'cam_1': 
                                    {
                                        'rgb': nparray, 
                                        'depth': nparray
                                    }
                                },
                                ...
                            }, 
                         'robot_data':
                             {
                                 'joint_pos': nparray,
                                 'joint_vel': nparray,
                             }, 
                        ...}, 
                    ...
                    ]

            We create a list of equivalent nested dictionaries across all frames and recurse
            into them until we reach the observation data. Then, we write that data to the HDF5 file.

            e.g observations/
                    sensor_data/
                        cam/
                            rgb: Dataset (Time, ...)
                            depth: Dataset (Time, ..)
                    robot_data/
                        joint_pos: Dataset (Time, ...)
                        joint_vel: Dataset (Time, ...)
                    ...
            """
            # Bail on empty list
            if not obs_list:
                return
            
            # Get the structure of the observation dictionary
            structure = obs_list[0]

            for key, value in structure.items():
                if isinstance(value, dict):
                    # Create a new group under parent if it doesn't exist
                    sub_group = parent_group.require_group(key)
                    
                    # Collect all equivalent dicts across all frames
                    sub_list = [obs[key] for obs in obs_list]

                    # Recurse into the sub-dicts
                    write_nested_dict(sub_list, sub_group)
                else:
                    # If not a dict, then we have reached the observation data
                    # Collect the data across all frames and stack it
                    stacked_values = np.stack([obs[key] for obs in obs_list])

                    if key.lower() == "rgb" and self.rgb_as_uint8:
                            stacked_values = stacked_values.astype(np.uint8)

                    # Write the stacked data to the parent group
                    if key not in parent_group:
                        # If the dataset doesn't exist, create it
                        parent_group.create_dataset(
                            name=key,
                            data=stacked_values,
                            maxshape=(None, *stacked_values.shape[1:]), # Fix all dims but time
                            chunks=True,
                            compression=self.compression
                        )
                    else:
                        # If the dataset exists, resize it and append the new data
                        dataset = parent_group[key]
                        if not isinstance(dataset, h5py.Dataset):
                            raise TypeError(f"Dataset {key} is not a h5py.Dataset")
                        current_size = dataset.shape[0]
                        dataset.resize(
                            (current_size + stacked_values.shape[0]),
                            axis=0
                        )
                        # Append the new data to the existing dataset
                        dataset[current_size:] = stacked_values

        # Write observations
        write_nested_dict(self.episode_data_buffer["observations"], self.obs_group)

        # Write actions
        write_nested_dict(self.episode_data_buffer["actions"], self.action_group)

        # Write flags
        for flag_name in ["is_first", "is_last"]:
            flag_data = np.stack(self.episode_data_buffer[flag_name], dtype=bool)

            if flag_name not in self.flags_group:
                self.flags_group.create_dataset(
                    name=flag_name,
                    data=flag_data,
                    maxshape=(None,),
                    chunks=True,
                    compression=self.compression
                )
            else:
                flag_dataset = self.flags_group[flag_name]
                if isinstance(flag_dataset, h5py.Dataset):
                    current_size = flag_dataset.shape[0]
                    flag_dataset.resize(
                        (current_size + flag_data.shape[0]),
                        axis=0
                    )
                    flag_dataset[current_size:] = flag_data
                else: raise TypeError(f"Flag dataset is not a h5py.Dataset")

        task_phase_data = np.stack(self.episode_data_buffer["task_phase"])
        current_size = self.task_phase_dataset.shape[0]
        self.task_phase_dataset.resize(
            (current_size + task_phase_data.shape[0]),
            axis=0
        )
        self.task_phase_dataset[current_size:] = task_phase_data

        # Write sim time
        sim_time_data = np.stack(self.episode_data_buffer["sim_time"])
        current_size = self.time_dataset.shape[0]
        self.time_dataset.resize(
            (current_size + sim_time_data.shape[0]),
            axis=0
        )
        self.time_dataset[current_size:] = sim_time_data

        self._reset_buffer()



