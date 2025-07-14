import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

class ResNet18Encoder(nn.Module):
    """
    Docs:
    https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    """
    def __init__(
        self, 
        pretrained: bool = True, 
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Load encoder with pretrained weights
        self.encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
            )

        # Remove classification head
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        self.encoder.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, x):
        x = x.to(self.device)
        if x.ndim == 3:
            assert x.shape[0] == 3, f"Expected channel dimension to be 3, got {x.shape[0]}"
            x = x.unsqueeze(0)
        assert x.ndim == 4, f"Expected (B, C, H, W), got {x.shape}"

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.transform(x)

        with torch.no_grad():
            x = self.encoder(x)
        
        # Removing classifier head means ResNet outputs (B, 512, 1, 1)
        # We need to flatten the output to (B, 512)
        x = x.squeeze(-1).squeeze(-1)
        assert x.ndim == 2, f"Expected (B, 512), got {x.shape}"

        return x



