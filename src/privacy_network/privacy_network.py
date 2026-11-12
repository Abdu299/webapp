from pathlib import Path

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from src.privacy_network import osv5m
from src.privacy_network.osv5m import load_model


def load_or_save():
    """Loads the frozen baseline model."""

    model_path = Path("cached_models/privacy_network/osv5m.torch")
    if not model_path.exists():
        model = load_model()
        torch.save(model, "cached_models/privacy_network/osv5m.torch")
        return model
    else:
        return torch.load(model_path, weights_only=False)


class TokenConvDecoder(nn.Module):
    """CLIP ViT tokens (B, 1 + 16*16, 1024) -> mask (B, 1, 224, 224)."""

    def __init__(self, dim=1024, out_size=(224, 224), out_channels=1):
        super().__init__()
        self.out_size = out_size

        # add some learnable parameters
        self.stem = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )

        # 16 -> 32 -> 64 -> 128
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

    def forward(self, tokens):
        B, N, C = tokens.shape
        s = int((N - 1) ** 0.5)  # 16 (drops CLS)
        f = tokens[:, 1:].reshape(B, s, s, C)  # (B, 16, 16, 1024)
        f = f.permute(0, 3, 1, 2)  # (B, 1024, 16, 16)

        f = self.up(self.stem(f))  # (B, 32, 128, 128)
        f = F.interpolate(f, size=self.out_size, mode="bilinear", align_corners=False)
        raw_values = self.head(f)  # (B, 1, 224, 224)
        return torch.sigmoid(raw_values) # 0 = no blur, 1 = max blur

class PrivacyNetwork(nn.Module):
    def __init__(self, encoder, out_size=(224, 224), block=1) -> None:
        super().__init__()
        self.encoder = encoder
        with torch.no_grad():
            dim = encoder.backbone({"img": torch.zeros(1, 3, 224, 224)}).shape[-1]  # 1024 for OSV-5M, 384 for GeoViT
        self.decoder = TokenConvDecoder(dim=dim, out_size=out_size, out_channels=1)
        self.block = block 

    def transform(self, x):
        return self.encoder.transform(x)

    def forward(self, x):
        tokens = self.encoder.backbone({"img": x})  # (B, 257, dim)
        mask = self.decoder(tokens)  # (B, 1, 224, 224)
        
        if self.block > 1:
            small = F.avg_pool2d(mask, self.block)  # (B, 1, 224/block, 224/block) rounded down
            mask = F.interpolate(small, size=mask.shape[-2:], mode="bilinear", align_corners=False)  # (B, 1, 224, 224)
        return mask


if __name__ == "__main__":
    osv5m = load_or_save()
    model = PrivacyNetwork(encoder=osv5m)
    x = model.transform(
        Image.open("./privacyData/osv5m/raw/images/test/00/100112495651554.jpg").convert("RGB")
    ).unsqueeze(0)

    print(model(x).shape)
