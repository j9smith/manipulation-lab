import torch
from torch import nn

class MLPFusion(nn.Module):
    def __init__(self, input_dim, img_input_dim, proprio_input_dim, output_dim):
        super().__init__()

        assert input_dim == img_input_dim + proprio_input_dim, f"Dim mismatch: input_dim: {input_dim}, img_input_dim: {img_input_dim}, proprio_input_dim: {proprio_input_dim}"

        self.img_input_dim = img_input_dim
        self.proprio_input_dim = proprio_input_dim

        self.img_mlp = nn.Sequential(
            nn.Linear(img_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
        )

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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

        # Clamp dim 4 to zero
        fusion[:, 4] = 0.0
        return fusion
    
class MLPFusionFrameStacked(MLPFusion):
    def __init__(
            self, 
            input_dim, 
            img_input_dim, 
            proprio_input_dim, 
            output_dim,
            sequence_length,
    ):
        super().__init__(
            input_dim=(img_input_dim * sequence_length) + (proprio_input_dim * sequence_length),
            img_input_dim=img_input_dim * sequence_length,
            proprio_input_dim=proprio_input_dim * sequence_length,
            output_dim=output_dim
        )
        self.sequence_length = sequence_length

        self.stacker = None

    def forward(self, x):
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.view(B, T * D)
        else:
            assert x.ndim == 2, f"Expected (B, D), got {x.shape}"
            if self.stacker is None:
                from manipulation_lab.models.utils.frame_stacking import FrameStacker
                self.stacker = FrameStacker(
                    sequence_length=self.sequence_length,
                    obs_dim=x.shape[-1],
                    device=x.device
                )
            self.stacker.store(x)
            x = self.stacker.retrieve()

        img = x[:, : self.img_input_dim]
        proprio = x[:, self.img_input_dim:]

        img = self.img_mlp(img)
        proprio = self.proprio_mlp(proprio)
        fusion = self.fusion_mlp(torch.cat([img, proprio], dim=-1))

        fusion[:, 4] = 0.0
        
        return fusion