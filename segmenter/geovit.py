import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

# Abdu's checkpoint (from origin/w3) and his label list, copied to my folder on Fox
CHECKPOINT = "/fp/projects01/ec674/williaan/checkpoints/geovit_abdu.pth"
LABELS = "/fp/projects01/ec674/williaan/data/country_to_label.json"

# ImageNet colour normalisation, same as Abdu's vitTest.py (the model was trained on this scale)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# Abdu used an older transformers that names DINOv2's attention layers differently; same weights
RENAMES = {".attention.attention.query.": ".attention.q_proj.",
           ".attention.attention.key.": ".attention.k_proj.",
           ".attention.attention.value.": ".attention.v_proj.",
           ".attention.output.dense.": ".attention.o_proj."}


def rename(key):
    for old, new in RENAMES.items():
        key = key.replace(old, new)
    return key


class GeoVit(nn.Module):
    """Copy of Abdu's GeoVit, so the checkpoint's weights fit.
    No fine_tune argument: it only matters for training, and we only predict."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)   # 384 -> countries

    def forward(self, images):
        # CLS token (position 0) = summary of the whole photo, what Abdu's classifier uses
        features = self.backbone(pixel_values=images).last_hidden_state[:, 0, :]   # [N, 384]
        return self.classifier(features)    # [N, countries], scores before softmax


def load_geovit(device, checkpoint=CHECKPOINT, labels_json=LABELS):
    """Returns the frozen model and a dict label -> country code (e.g. 10 -> "AU")."""
    with open(labels_json) as f:
        country_to_label = json.load(f)     # {"AD": 0, "AE": 1, ...}
    label_to_country = {label: country for country, label in country_to_label.items()}

    model = GeoVit(len(country_to_label))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:         # Abdu sometimes saves weights + settings together
        state = state["model_state_dict"]
    # Only rename to the new names if this transformers version uses them
    if any(".q_proj." in k for k in model.state_dict()):
        state = {rename(k): v for k, v in state.items()}
    model.load_state_dict(state)            # crashes if the class above doesn't match his
    return model.to(device).eval().requires_grad_(False), label_to_country


def prepare_geovit(batch):
    """[N, 3, 448, 448], values 0-1 -> [N, 3, 224, 224], normalised.
    Uses our 448 centre square (not Abdu's squashed photo) so the masks line up."""
    batch = F.interpolate(batch, size=224, mode="bilinear", antialias=True)   # bilinear like Abdu's Resize
    return (batch - MEAN.to(batch.device)) / STD.to(batch.device)
