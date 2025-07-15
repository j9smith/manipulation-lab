import torch
from torch import nn

class MLPFusion(nn.Module):
    def __init__(self, input_dim, img_input_dim, proprio_input_dim, output_dim):
        super().__init__()

        self.img_input_dim = img_input_dim
        self.proprio_input_dim = proprio_input_dim

        self.img_mlp = nn.Sequential(
            nn.Linear(img_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        if x.ndim == 1: x = x.unsqueeze(0)
        img = x[:, : self.img_input_dim]
        proprio = x[:, self.img_input_dim :]

        img = self.img_mlp(img)
        proprio = self.proprio_mlp(proprio)
        fusion = self.fusion_mlp(torch.cat([img, proprio], dim=-1))

        return fusion