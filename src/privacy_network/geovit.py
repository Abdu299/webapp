import json
from pathlib import Path

import numpy as np
import reverse_geocoder as rg
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import Dinov2Model

from src.privacy_network.datasets import photo_loader

# model/vit_countries_last3_geographic_lr0.0001.pth on branch w3
weights_path = Path("cached_models/geovit/geovit.pth")
# country code -> label, like {"AD": 0, ...}
countries_path = Path("privacyData/country_to_label.json")

# it used an older transformers that names DINOv2's attention layers differently, same weights
renames = {
    ".attention.attention.query.": ".attention.q_proj.",
    ".attention.attention.key.": ".attention.k_proj.",
    ".attention.attention.value.": ".attention.v_proj.",
    ".attention.output.dense.": ".attention.o_proj.",
}



class Backbone(Dinov2Model): # takes {"img": x}
    def forward(self, batch):
        return super().forward(pixel_values=batch["img"]).last_hidden_state  # (B, 257, 384)


class GeoVit(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)  # 384 -> countries

    def forward(
        self,
        x,  # (B, 3, 224, 224)
    ):
        tokens = self.backbone({"img": x})  # (B, 257, 384)

        # scores from the class token
        return self.classifier(tokens[:, 0])  # (B, num_countries)



def load_model():
    country_to_label = json.loads(
        countries_path.read_text()
    )

    countries = [None] * len(country_to_label)

    for country, label in country_to_label.items():
        countries[label] = country

    model = GeoVit(len(countries))

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    # Different Transformers versions use different
    # names for DINOv2 attention layers.
    #
    # Convert checkpoint keys to match the model
    # installed in the current environment.

    names = {
        ".attention.attention.query.": ".attention.q_proj.",
        ".attention.attention.key.": ".attention.k_proj.",
        ".attention.attention.value.": ".attention.v_proj.",
        ".attention.output.dense.": ".attention.o_proj."
    }

    expected_keys = set(model.state_dict().keys())

    converted_state = {}

    for key, value in state.items():
        new_key = key

        if key not in expected_keys:
            for old_name, new_name in names.items():
                if old_name in new_key:
                    candidate = new_key.replace(
                        old_name,
                        new_name
                    )

                    if candidate in expected_keys:
                        new_key = candidate
                        break

                elif new_name in new_key:
                    candidate = new_key.replace(
                        new_name,
                        old_name
                    )

                    if candidate in expected_keys:
                        new_key = candidate
                        break

        converted_state[new_key] = value

    # Keep strict loading to detect any real
    # architecture mismatch.
    model.load_state_dict(
        converted_state,
        strict=True
    )

    model.countries = countries

    model.transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return model.eval().requires_grad_(False)


def predict(
    model,
    x,  # (B, 3, 224, 224)
):
    # top country label
    return model(x).argmax(dim=1)  # (B,)


def true_countries(
    lat,  # (N,)
    lon,  # (N,)
):
    # country code like "NO" per location in degrees
    places = rg.search(list(zip(lat, lon)), mode=1, verbose=False)
    codes = []
    for place in places:
        codes.append(place["cc"])
    return codes


def guess(model, df, device):
    true = true_countries(df["lat"], df["lon"])

    # lat, lon in degrees, NaN for unreadable photos
    guesses = np.full((len(df), 2), np.nan)  # (N, 2)
    for batch in tqdm(photo_loader(df, model.transform), desc="geovit"):
        rows = batch["row"].tolist()
        with torch.no_grad():
            labels = predict(model, batch["image"].to(device)).tolist()  # (B,)
        for row, label in zip(rows, labels):
            lat = df["lat"].iloc[row]
            lon = df["lon"].iloc[row]
            # GeoViT names a country: the right one gives the photo's own point (0 km), a wrong one the farthest point (20015 km)
            if model.countries[label] == true[row]:
                guesses[row] = [lat, lon]
            else:
                guesses[row] = [-lat, lon + 180]
    return guesses


def judge(
    model, 
    x,  # (B, 3, 224, 224)
    lat,  # (B,)
    lon,  # (B,)
):  
    logits = model(x).float()  # (B, num_countries)

    # True at the true country
    # all False if GeoViT does not know it
    near = torch.zeros_like(logits, dtype=torch.bool)  # (B, num_countries)
    codes = true_countries(lat.tolist(), lon.tolist())
    for i in range(len(codes)):
        if codes[i] in model.countries:
            near[i, model.countries.index(codes[i])] = True
    top = logits.argmax(dim=1, keepdim=True)  # (B, 1)
    
    # True where the top country is wrong
    hidden = ~near.gather(1, top).squeeze(1)  # (B,)
    return logits, near, hidden
