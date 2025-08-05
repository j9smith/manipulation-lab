import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int=7):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(self._input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self._output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3: x= x.squeeze(1)
        return self.network(x)