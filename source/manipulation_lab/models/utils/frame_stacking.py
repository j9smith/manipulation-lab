from collections import deque
import torch

class FrameStacker():
    def __init__(
            self,
            sequence_length: int,
            obs_dim: int,
            device
    ):
        self.seq_length = sequence_length
        self.dims = obs_dim
        self.device = device
        self.buffer = deque(maxlen=self.seq_length)

    def store(self, obs: torch.Tensor):
        obs = obs.to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        self.buffer.append(obs)

    def retrieve(self):
        if len(self.buffer) < self.seq_length:
            frame = self.buffer[0]
            frames_to_pad = self.seq_length - len(self.buffer)
            padding = [frame.clone()] * frames_to_pad
            buffer = padding + list(self.buffer)
        else:
            buffer = list(self.buffer)

        stacked = torch.stack(buffer, dim=1)

        B, T, D = stacked.shape
        assert T == self.seq_length and D == self.dims
    
        return stacked.view(B, T * D)