import logging
logger = logging.getLogger("ManipulationLab.DatasetWrapper")

import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable
from collections import OrderedDict
from manipulation_lab.scripts.dataset.reader import DatasetReader

import time

class DatasetWrapper(Dataset):
    """
    Implements a custom dataset wrapper on top of torch.Dataset. 
    """
    def __init__(
        self,
        dataset_dir: str,
        episode_indices: List[int],
        camera_keys: List[str],
        action_keys: List[str],
        proprio_keys: Optional[List[str]] = None,
        sensor_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        image_encoder: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialises a torch dataset wrapper for the target dataset.

        Parameters:
        - dataset_dir: str - The directory containing the dataset.
        - episode_indices: List[int] - The episodes to load into the dataset (for train/test/val splits).
        - camera_keys: List[str] - The keys to the target camera data in the episode dictionary,
        e.g. ['sensors/wrist_camera/rgb', 'sensors/left_shoulder_camera/rgb']. Any camera data can
        have transforms applied.
        - action_keys: List[str] - The keys to the target action data in the episode dictionary,
        e.g. ['ee_deltas', 'gripper_deltas'].
        - proprio_keys: Optional[List[str]] - The keys to the target proprioceptive data in the episode 
        dictionary, e.g. ['robot/joint_positions', 'robot/joint_velocities'].
        - sensor_keys: Optional[List[str]] - The keys to the target sensor data in the episode 
        dictionary, e.g. ['sensors/wrist_camera/depth'].
        - transform: Optional[Callable] - A function to transform the observation data.
        - image_encoder: Optional[Callable] - A function to encode the image data.
        - structured_obs: bool - Return the observation data as a structured dictionary (True), or
        a tensor (False).

        Returns:
        - obs, actions: Tuple[Tensor, Tensor] - The requested observation and action data for the target frame,
        if image_encoder is not None or structured_obs is True.
        - {camera_obs, proprio_obs, sensor_obs, actions}: Dict[str, Tensor] - The requested observation and 
        action data for the target frame, if image_encoder is None and structured_obs is False.
        """
        self.reader = DatasetReader(dataset_dir=dataset_dir)
        self.episode_indices = episode_indices
        self.camera_keys = camera_keys
        self.action_keys = action_keys
        self.proprio_keys = proprio_keys
        self.sensor_keys = sensor_keys
        self.transform = transform
        self.image_encoder = image_encoder
        self.encoder_device = next(self.image_encoder.parameters()).device if self.image_encoder is not None else None

        self.max_cache_size = kwargs.get("max_cache_size", "32")
        self._episode_cache = OrderedDict()

        self.index = self._build_index()

    def _build_index(self):
        """
        Builds a list of tuples (episode_idx, frame_idx) for all frames across all 
        episodes in the dataset.
        """
        index = []

        for episode_idx in self.episode_indices:
            ep_frame_count = self.reader.get_frame_count(episode_idx)
            index.extend([(episode_idx, frame_idx) for frame_idx in range(ep_frame_count)])

        return index

    def __len__(self):
        """
        Returns the total number of frames across all episodes in the dataset.
        """
        return len(self.index)

    def __str__(self):
        """
        Returns the structure of an example episode.
        """
        return f"Example episode:\n{self.reader.describe_structure()}"

    def __getitem__(self, item_idx: int):
        """
        Returns the target frame data at the specified index.
        """
        ep_idx, frame_idx = self.index[item_idx]

        if ep_idx not in self._episode_cache:
            self._episode_cache[ep_idx]= self.reader.load_episode(ep_idx)
            self._episode_cache.move_to_end(ep_idx)

            if (len(self._episode_cache) > int(self.max_cache_size)):
                key, _ = self._episode_cache.popitem(last=False)
                logger.info(f"Removing key from cache: {key}")

        else:
            episode = self._episode_cache[ep_idx]
            self._episode_cache.move_to_end(ep_idx)

        data = self._get_frame_data(self, episode, ep_idx, frame_idx)
        return data
    
    def _get_frame_data(self, episode, ep_idx, frame_idx):
        camera_obs = []
        if self.camera_keys:
            for camera_key in self.camera_keys:
                img = self._get_nested_data(episode["observations"], camera_key)[frame_idx]
                assert img.ndim == 3, f"Expected image to be (C, H, W), got {img.shape}"

                # (H, W, C) -> (C, H, W) and uint8 -> float32
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

                camera_obs.append(img)

        proprio_obs = []
        if self.proprio_keys:
            for prop_key in self.proprio_keys:
                prop = self._get_nested_data(episode["observations"], prop_key)[frame_idx]
                assert prop.ndim == 1, f"Expected proprioceptive data to be (N,), got {prop.shape}"
                prop = torch.tensor(prop, dtype=torch.float32)
                proprio_obs.append(prop)
        
        sensor_obs = []
        if self.sensor_keys:
            for sensor_key in self.sensor_keys:
                sensor = self._get_nested_data(episode["observations"], sensor_key)[frame_idx]
                assert sensor.ndim == 1, f"Expected sensor data to be (N,), got {sensor.shape}"
                sensor = torch.tensor(sensor, dtype=torch.float32)
                sensor_obs.append(sensor)
                
        actions = []
        for action_key in self.action_keys:
            action = self._get_nested_data(episode["actions"], action_key)[frame_idx]
            action = torch.tensor(action, dtype=torch.float32)
            if action.ndim == 0:
                action = action.unsqueeze(0) # Unsqueeze scalars
            assert action.ndim == 1, f"Expected action data to be (N,), got {action.shape}"
            actions.append(action)

        data = {
            "metadata":{
                "episode_idx": ep_idx,
                "frame_idx": frame_idx
            },
            "actions": { action_key: actions[idx] for idx, action_key in enumerate(self.action_keys) },
        }
        if self.camera_keys:
            data["camera"] = {camera_key: camera_obs[idx] for idx, camera_key in enumerate(self.camera_keys)}
        if self.proprio_keys:
            data["proprio"] = { prop_key: proprio_obs[idx] for idx, prop_key in enumerate(self.proprio_keys) }
        if self.sensor_keys:
            data["sensor"] = { sensor_key: sensor_obs[idx] for idx, sensor_key in enumerate(self.sensor_keys) }

    def _get_nested_data(self, nested_dict: dict, key: str):
        """
        Resolves and returns the target data nested inside the episode dictionary.

        Parameters:
        - nested_dict: dict - The nested dictionary to resolve the target data from.
        - key: str - The path to the target data in the nested dictionary.
        e.g., key="observations/camera/rgb"

        Returns:
        - The target data nested inside the episode dictionary.
        """
        # Split the key path into its component parts, e.g. ['observations', 'camera', 'rgb']
        keys = key.split("/")

        # Iterate over the keys, e.g. 'observations' -> 'camera' -> 'rgb'
        for key in keys:
            # Reassign the value of nested_dict to the value of the current key
            nested_dict = nested_dict[key]

        # e.g., return the data present at 'rgb'
        return nested_dict

class SequentialDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset_dirs,
        episode_indices: List[int],
        action_keys: List[str],
        sequence_length: int = 1,
        stride: int = 1,
        camera_keys: Optional[List[str]] = None,
        proprio_keys: Optional[List[str]] = None,
        sensor_keys: Optional[List[str]] = None,
        oracle_keys: Optional[List[str]] = None,
        image_encoder: Optional[Callable] = None,
        **kwargs
    ):
        self.reader = DatasetReader(dataset_dirs=dataset_dirs)
        self.episode_indices = episode_indices
        self.camera_keys = camera_keys
        self.action_keys = action_keys
        self.proprio_keys = proprio_keys
        self.sensor_keys = sensor_keys
        self.oracle_keys = oracle_keys

        self.sequence_length = sequence_length
        self.stride = stride

        self.image_encoder = image_encoder
        self.encoder_device = next(self.image_encoder.parameters()).device if self.image_encoder is not None else None
        
        self.max_cache_size = kwargs.get("max_cache_size", 64)
        self._episode_cache = OrderedDict()

        self.index = self._build_index()

    def _build_index(self):
        """
        Builds a list of tuples (episode_idx, frame_idx) for all frames across all 
        episodes in the dataset.
        """
        index = []

        for episode_idx in self.episode_indices:
            ep_frame_count = self.reader.get_frame_count(episode_idx)
            
            last_valid_start = ep_frame_count - self.sequence_length

            for frame_idx in range(0, last_valid_start + 1, self.stride):
                index.append((episode_idx, frame_idx))

        return index

    def __len__(self):
        """
        Returns the total number of frames across all episodes in the dataset.
        """
        return len(self.index)

    def __str__(self):
        """
        Returns the structure of an example episode.
        """
        return f"Example episode:\n{self.reader.describe_structure()}"
    
    def __getitem__(self, item_idx: int):
        """
        Returns the target frame data at the specified index.
        """
        ep_idx, start_frame_idx = self.index[item_idx]

        if ep_idx not in self._episode_cache:
            episode = self.reader.load_episode(ep_idx)
            self._episode_cache[ep_idx] = episode

            if (len(self._episode_cache) > int(self.max_cache_size)):
                key, _ = self._episode_cache.popitem(last=False)
                logger.info(f"Removing key from cache: {key}")

        else:
            episode = self._episode_cache[ep_idx]
            self._episode_cache.move_to_end(ep_idx)

        frames = []
        for i in range(self.sequence_length):
            frame_idx = start_frame_idx + i
            frame = self._get_frame_data(episode, ep_idx, frame_idx)
            frames.append(frame)

        data = self._stack_frame_data(frames)

        return data

    def _stack_frame_data(self, frames: list[dict]):
        """
        Takes different frames and stacks them along a time dimension in a new tensor.
        """
        stacked_data = {}
        for modality in ["camera", "proprio", "sensor", "oracle", "actions"]:
            if modality in frames[0]:
                stacked_data[modality] = {}
                for key in frames[0][modality].keys():
                    data = [frame[modality][key] for frame in frames]
                    stacked_data[modality][key] = torch.stack(data, dim=0)

        stacked_data["metadata"] = {
            "episode_idx": frames[0]["metadata"]["episode_idx"],
            "start_frame_idx": frames[0]["metadata"]["frame_idx"]
        }
        
        return stacked_data
    
    def _get_frame_data(self, episode, ep_idx, frame_idx):
        camera_obs = []
        if self.camera_keys:
            for camera_key in self.camera_keys:
                img = self._get_nested_data(episode["observations"], camera_key)[frame_idx]
                assert img.ndim == 3, f"Expected image to be (C, H, W), got {img.shape}"

                # (H, W, C) -> (C, H, W) and uint8 -> float32
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

                camera_obs.append(img)

        proprio_obs = []
        if self.proprio_keys:
            for prop_key in self.proprio_keys:
                prop = self._get_nested_data(episode["observations"], prop_key)[frame_idx]
                assert prop.ndim == 1, f"Expected proprioceptive data to be (N,), got {prop.shape}"
                prop = torch.tensor(prop, dtype=torch.float32)
                proprio_obs.append(prop)
        
        sensor_obs = []
        if self.sensor_keys:
            for sensor_key in self.sensor_keys:
                sensor = self._get_nested_data(episode["observations"], sensor_key)[frame_idx]
                assert sensor.ndim == 1, f"Expected sensor data to be (N,), got {sensor.shape}"
                sensor = torch.tensor(sensor, dtype=torch.float32)
                sensor_obs.append(sensor)

        oracle_obs = []
        if self.oracle_keys:
            for oracle_key in self.oracle_keys:
                oracle = self._get_nested_data(episode["observations"], oracle_key)[frame_idx]
                assert oracle.ndim == 1, f"Expected oracle data to be (N,), got {sensor.shape}"
                oracle = torch.tensor(oracle, dtype=torch.float32)
                oracle_obs.append(oracle)
                
        actions = []
        for action_key in self.action_keys:
            action = self._get_nested_data(episode["actions"], action_key)[frame_idx]
            action = torch.tensor(action, dtype=torch.float32)
            if action.ndim == 0:
                action = action.unsqueeze(0) # Unsqueeze scalars
            assert action.ndim == 1, f"Expected action data to be (N,), got {action.shape}"
            actions.append(action)

        data = {
            "metadata":{
                "episode_idx": ep_idx,
                "frame_idx": frame_idx
            },
            "actions": { action_key: actions[idx] for idx, action_key in enumerate(self.action_keys) },
        }
        if self.camera_keys:
            data["camera"] = {camera_key: camera_obs[idx] for idx, camera_key in enumerate(self.camera_keys)}
        if self.proprio_keys:
            data["proprio"] = { prop_key: proprio_obs[idx] for idx, prop_key in enumerate(self.proprio_keys) }
        if self.sensor_keys:
            data["sensor"] = { sensor_key: sensor_obs[idx] for idx, sensor_key in enumerate(self.sensor_keys) }
        if self.oracle_keys:
            data["oracle"] = { oracle_key: oracle_obs[idx] for idx, oracle_key in enumerate(self.oracle_keys) }

        return data

    def _get_nested_data(self, nested_dict: dict, key: str):
        """
        Resolves and returns the target data nested inside the episode dictionary.

        Parameters:
        - nested_dict: dict - The nested dictionary to resolve the target data from.
        - key: str - The path to the target data in the nested dictionary.
        e.g., key="observations/camera/rgb"

        Returns:
        - The target data nested inside the episode dictionary.
        """
        # Split the key path into its component parts, e.g. ['observations', 'camera', 'rgb']
        keys = key.split("/")

        # Iterate over the keys, e.g. 'observations' -> 'camera' -> 'rgb'
        for key in keys:
            # Reassign the value of nested_dict to the value of the current key
            nested_dict = nested_dict[key]

        # e.g., return the data present at 'rgb'
        return nested_dict