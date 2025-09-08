from r3m import load_r3m
import torch.nn.functional as F
from torchvision.transforms import v2
import torch
from torch.nn import Module

class R3MEncoder(Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

        self.encoder = load_r3m("resnet18")
        self.encoder.eval()
        self.encoder.to(device)

    def forward(self, img):
        """
        R3M resizes and normalises internally
        """
        if img.ndim == 3:
            assert img.shape[0] == 3, f"Expected channel dimension to be 3, got {img.shape[0]}"
            img = img.unsqueeze(0)
        assert img.ndim == 4, f"Expected (B, C, H, W), got {img.shape}"

        resized_img = F.interpolate(
            img, 
            size=256, 
            mode='bilinear',
            align_corners=False,
        )

        crop = v2.CenterCrop(size=(224, 224))
        cropped_img = crop(resized_img)

        processed_img = cropped_img.float().to(self.device)

        with torch.no_grad():
            embedded = self.encoder(processed_img) # (B, 2048)

        return embedded