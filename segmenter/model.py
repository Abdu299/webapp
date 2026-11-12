"""
One-layer (linear) decoder on top of a frozen DINOv2, following Segmenter.

DINOv2 turns every 14x14 patch of the image into a vector of 384 numbers.
A single Linear layer reads a clue class score out of each patch vector, and the
patch scores are upsampled to pixel scores so the loss can be computed per pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

# Same backbone as Abdu's geolocation model, so the whole group uses one ViT.
BACKBONE = "facebook/dinov2-small"
PATCH_SIZE = 14


class LinearSegmenter(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = Dinov2Model.from_pretrained(BACKBONE)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # One linear layer from the feature size to num_classes
        # Reading the feature size directly from the backbone
        self.decoder = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, images):
        """images: [B, 3, H, W] normalised. Returns scores [B, num_classes, H, W]."""
        B, _, H, W = images.shape

        # Run backbone and get the tokens
        with torch.no_grad():
            tokens = self.backbone(pixel_values=images).last_hidden_state

        # Keep only the N patch tokens (dropping the class token)
        patch_tokens = tokens[:, 1:, :] 

        # Applying the decoder to every patch token
        scores = self.decoder(patch_tokens)

        # Reshape the patches back into the patch grid
        h, w = H // PATCH_SIZE, W // PATCH_SIZE
        scores = scores.reshape(B, h, w, -1).permute(0, 3, 1, 2) # [B, num_classes, h, w]

        # Upsample to image size
        scores = F.interpolate(scores, size=(H, W), mode="bilinear", align_corners=False)

        return scores


class MaskTransformerDecoder(nn.Module):
    """Segmenter's mask transformer section 3.2"""
    def __init__(self, dim, num_classes, num_layers=2, num_heads=6):
        super().__init__()
        self.num_classes = num_classes

        # Project frozen DINOv2 features into decoder space
        self.proj = nn.Linear(dim, dim)

        # One learnable vector per class with random start
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)

        # 2 tranformer layers. 6 heads like DINOv2-small so each head works on 384/6 = 64 numbers
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=4*dim,
            dropout=0.1, batch_first=True, norm_first=True
        )

        self.blocks = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(dim)

        # Separate projections for patches and classes before comparison
        self.patch_proj = nn.Linear(dim, dim, bias=False)
        self.class_proj = nn.Linear(dim, dim, bias=False)

        # Normalise K scores of each patch
        self.mask_norm = nn.LayerNorm(num_classes)

    def forward(self, patch_tokens):
        # Shapes: B = images in batch (16), N = patches per image (768),
        #         D = numbers per patch (384), K = classes (10)
        # patch_tokens: [B, N, D], one row of D numbers per patch

        B = patch_tokens.shape[0]  # needed to copy the class tokens once per image
        K = self.num_classes       # needed to split patches and classes apart again

        # Translate DINOv2 features into the decoder's own space. [B, N, D] -> [B, N, D]
        x = self.proj(patch_tokens)

        # The K learned class prototypes exist once: [1, K, D].
        # Copy them for every image in the batch: [B, K, D]
        classes = self.class_tokens.expand(B, -1, -1)

        # Glue class tokens under the patches along dim 1 (the token dim).
        # [B, N, D] + [B, K, D] -> x: [B, N + K, D]  (768 + 10 = 778 tokens)
        x = torch.concat([x, classes], dim=1)

        # Every token looks at all tokens of image and get context
        x = self.norm(self.blocks(x))



        # Split again: all but the last K rows are patches, the last K are classes.
        patches, classes = x[:, :-K], x[:, -K:]

        patches = self.patch_proj(patches)
        classes = self.class_proj(classes)

        patches = F.normalize(patches, dim=-1)
        classes = F.normalize(classes, dim=-1)

        # Dot product for socre of every patch for every cøass
        masks = patches @ classes.transpose(1, 2) # Flip [B, K, D] to [B, D, K]

        return self.mask_norm(masks)

class MaskSegmenter(LinearSegmenter):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.decoder = MaskTransformerDecoder(self.backbone.config.hidden_size, num_classes)

if __name__ == "__main__":
    # Quick check with a fake batch. Runs on CPU, no data needed.
    model = LinearSegmenter(num_classes=10)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("trainable parameters:", trainable)  # expect 384 * 10 + 10 = 3850
    print("frozen parameters:   ", frozen)     # about 22 million (DINOv2-small)

    images = torch.randn(2, 3, 336, 448)
    scores = model(images)
    print("scores:", scores.shape)  # expect [2, 10, 336, 448]
