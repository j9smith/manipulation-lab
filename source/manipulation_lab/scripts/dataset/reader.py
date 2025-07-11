import logging
logger = logging.getLogger("ManipulationLab.DatasetReader")

import h5py
from pathlib import Path

class DatasetReader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.exists(), f"Dataset directory {self.dataset_dir} does not exist"

        self.episodes = sorted(self.dataset_dir.glob("*.hdf5"))
        assert len(self.episodes) > 0, f"No episodes found in {self.dataset_dir}"

        logger.info(f"Found {len(self.episodes)} episodes in {self.dataset_dir}")

    def __len__(self):
        """
        Returns the number of episodes present in the target directory.
        """
        return len(self.episodes)

    def load_episode(self, episode_idx: int):
        """
        Loads and returns the target episode in dictionary format.
        """
        episode_path = self.episodes[episode_idx]
        logger.debug(f"Loading episode {episode_idx}: {episode_path.name}")

        # FIXME: Throws an error if the episode is malformed, breaks training
        # Ideally deal with this without having to cancel training
        try:
            with h5py.File(episode_path, "r") as file:
                episode = {}

                def read_group(group: h5py.Group | h5py.Dataset | h5py.Datatype):
                    if isinstance(group, h5py.Dataset):
                        return group[:]
                    elif isinstance(group, h5py.Datatype):
                        logger.warning(f"Encountered unexpected type {type(group)} for key {group.name}.")
                        return

                    result = {}
                    # Iterate over all keys in group
                    for key in group.keys():
                        value = group[key]
                        # If the data is another group, recurse
                        if isinstance(value, h5py.Group):
                            result[key] = read_group(value)
                        # Otherwise, if it's a dataset, read it
                        elif isinstance(value, h5py.Dataset):
                            result[key] = value[:]
                        else:
                            raise ValueError(f"Unknown type {type(value)} for group {group.name} and key {key}.")
                    return result

                episode["observations"] = read_group(file["observations"])
                episode["actions"] = read_group(file["actions"])
                episode["flags"] = read_group(file["flags"])
                episode["sim_time"] = read_group(file["sim_time"])

                return episode
        except Exception as e:
            import sys
            logger.critical(f"Failed to load episode {episode_idx}: {e}")
            sys.exit(1)
    
    def get_frame_count(self, episode_idx: int):
        """
        Returns the number of frames in the target episode.
        """
        episode_path = self.episodes[episode_idx]
        try:
            with h5py.File(episode_path, "r") as file:
                frame_count = file.attrs["frame_count"]
                if isinstance(frame_count, h5py.Empty):
                    raise ValueError(f"Frame count not found for episode {episode_path.name}.")
                else: return int(frame_count)
        except Exception as e:
            import sys
            logger.critical(f"Failed to get frame count for episode {episode_path.name}: {e}")
            sys.exit(1)

    def describe_structure(self, episode_idx: int = 0):
        """
        Returns the structure of an example episode dataset.
        """
        lines = []
        episode_path = self.episodes[episode_idx]

        with h5py.File(episode_path, "r") as file:
            lines.append("Attributes:")
            for key, value in file.attrs.items():
                lines.append(f"    {key}: {value}")

            lines.append("\nStructure:")
            def _visit(name, obj):
                indent = "    " * name.count("/")
                if isinstance(obj, h5py.Group):
                    lines.append(f"{indent}{name}/")
                elif isinstance(obj, h5py.Dataset):
                    lines.append(f"{indent}{name}: shape={obj.shape} dtype={obj.dtype}")

            file.visititems(_visit)
        
        return "\n".join(lines)