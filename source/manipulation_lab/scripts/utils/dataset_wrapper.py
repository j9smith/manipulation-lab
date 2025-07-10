import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable
from manipulation_lab.scripts.utils.dataset_reader import DatasetReader

class DatasetWrapper(Dataset):
    """
    Implements a custom dataset wrapper on top of torch.Dataset. 
    """
    def __init__(
        self,
        dataset_dir: str,
        camera_keys: List[str],
        action_keys: List[str],
        proprio_keys: Optional[List[str]] = None,
        sensor_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        image_encoder: Optional[Callable] = None,
        structured_obs: bool = False,
        **kwargs
    ):
        """
        Initialises a torch dataset wrapper for the target dataset.

        Parameters:
        - dataset_dir: str - The directory containing the dataset.
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
        self.camera_keys = camera_keys
        self.action_keys = action_keys
        self.proprio_keys = proprio_keys
        self.sensor_keys = sensor_keys
        self.transform = transform
        self.image_encoder = image_encoder
        self.structured_obs = structured_obs

        self.index = self._build_index()

    def _build_index(self):
        """
        Builds a list of tuples (episode_idx, frame_idx) for all frames across all 
        episodes in the dataset.
        """
        index = []

        for episode_idx in range(len(self.reader)):
            ep_frame_count = self.reader.get_frame_count(episode_idx)
            index.extend([(episode_idx, frame_idx) for frame_idx in range(ep_frame_count)])

        return index

    def __len__(self):
        """
        Returns the total number of frames across all episodes in the dataset.
        """
        return len(self.index)

    def __getitem__(self, item_idx: int):
        """
        Returns the target frame data at the specified index.
        """
        ep_idx, frame_idx = self.index[item_idx]
        episode = self.reader.load_episode(ep_idx)

        camera_obs = []
        for camera_key in self.camera_keys:
            img = self._get_nested_data(episode["observations"], camera_key)[frame_idx]
            assert img.ndim == 3, f"Expected image to be (C, H, W), got {img.shape}"

            # (H, W, C) -> (C, H, W) and uint8 -> float32
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            if self.transform is not None:
                img = self.transform(img)

            if self.image_encoder is not None:
                # (C, H, W) -> (1, C, H, W)
                # Encoder expects batch dimension
                img = self.image_encoder(img.unsqueeze(0))

                # (1, D) -> (D,)
                img = img.squeeze(0)

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
            assert action.ndim == 1, f"Expected action data to be (N,), got {action.shape}"
            action = torch.tensor(action, dtype=torch.float32)
            actions.append(action)
            

        if self.structured_obs or (self.image_encoder is None and self.proprio_keys and self.sensor_keys):
            data = {
                "metadata":{
                    "episode_idx": ep_idx,
                    "frame_idx": frame_idx
                },
                "vision": { camera_key: camera_obs[idx] for idx, camera_key in enumerate(self.camera_keys) },
                "actions": { action_key: actions[idx] for idx, action_key in enumerate(self.action_keys) },
            }
            if self.proprio_keys:
                data["proprio"] = { prop_key: proprio_obs[idx] for idx, prop_key in enumerate(self.proprio_keys) }
            if self.sensor_keys:
                data["sensor"] = { sensor_key: sensor_obs[idx] for idx, sensor_key in enumerate(self.sensor_keys) }

            return data

        else:
            obs = camera_obs.copy()
            if proprio_obs: obs += proprio_obs
            if sensor_obs: obs += sensor_obs
            obs = torch.cat(obs, dim=-1)
            actions = torch.cat(actions, dim=-1)
            return obs, actions

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
