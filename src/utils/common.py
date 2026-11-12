"""Custom utility functions."""

import math
import torch
import pandas as pd
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image


# Photos saved as examples for every method.
# Six photos the baseline places within 25 km, spread over the world.
example_images = [
    "Canada_00106_513453328_03108e034b_227_7665941@N08.jpg", # Vancouver
    "Sydney_00073_1226915900_eea86783cd_1128_65768710@N00.jpg", # Sydney
    "530405817_0673660c96_1209_7885919@N07.jpg", # Buenos Aires
    "Lahore_00001_321085080_28863df64f_130_82503882@N00.jpg", # Lahore
    "Portugal_00001_37087644_e25c1c4be1_31_41894197861@N01.jpg", # Lisbon
    "PuertoRico_00013_1356370638_0426ac7dcb_1264_23437487@N00.jpg", # San Juan
]

# A guess farther than this from the true location counts as private (same 750 km as the optimised blur)
radius_km = 750.0


def print_accuracy(path):
    """Print accuracy to terminal."""
    summary = pd.read_csv(path)
    print("Accuracy:")
    for t, acc in zip(summary["km"], summary["accuracy"]):
        print(f"within {t} km: {round(acc*100, 2)}%")


def print_quality(path):
    """Print quality to terminal"""
    quality = pd.read_csv(path)
    print("Image quality (mean):")
    for measure, mean in zip(quality["measure"], quality["mean"]):
        print(f"{measure}: {round(mean, 3)}")


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between points given in degrees.

        Works for numbers and for tensors of any size.
        Return: float if all inputs are numbers, otherwise tensor of distances in km
    """
    radius = 6371  # of earth

    only_numbers = True
    coords = []
    for v in (lat1, lon1, lat2, lon2):
        if isinstance(v, torch.Tensor):
            only_numbers = False
        else:
            v = torch.tensor(v, dtype=torch.float64)
        # convert from degrees to radians
        coords.append(torch.deg2rad(v))
    lat1, lon1, lat2, lon2 = coords

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # hav(theta) = sin^2(theta / 2)
    hav_lat = torch.sin(dlat / 2) * torch.sin(dlat / 2)
    hav_lon = torch.sin(dlon / 2) * torch.sin(dlon / 2)

    # haversine of the central angle:
        # hav(d/r) = hav_lat + cos(lat1) * cos(lat2) * hav_lon
    hav_central = hav_lat + torch.cos(lat1) * torch.cos(lat2) * hav_lon

    # invert the haversine, d = 2r * asin(sqrt(hav))
    # clamp to [0, 1] against rounding errors
    d = 2 * radius * torch.asin(torch.sqrt(hav_central.clamp(0, 1)))

    # return a plain number if the input was plain numbers
    if only_numbers:
        return d.item()
    return d


def save_output(df, out_dir, out_name):
    """Calculating distance and accuracy of the predictions and saving them.
 
        df: Coordinates in degrees, predictions in radians, and image quality per photo
        out_dir: Output folder of the model, like output/osv5m
        out_name: Prefix of the result files
    """
    print("Calculating distances ... ")
    df = df.copy()
 
    # Convert from radians to degrees to match image coords are degrees
    df["pred_lat"] = df["pred_lat"].apply(math.degrees)
    df["pred_lon"] = df["pred_lon"].apply(math.degrees)
 
    distances = []
    for lat, lon, pred_lat, pred_lon in df[["lat", "lon", "pred_lat", "pred_lon"]].values:
        distances.append(haversine(lat, lon, pred_lat, pred_lon))
    df["km"] = distances
 
    thresholds = [1, 25, 200, 750, 2500]
    accuracy = []
    for t in thresholds:
        within = 0
        for km in distances:
            if km <= t:
                within += 1
        accuracy.append(within / len(distances))
 
    # Save predictions, real distance, and accuracy
    print("Saving predictions and distance from true location ... ")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{out_name}_predictions.csv", index=False)
 
    summary = pd.DataFrame({"km": thresholds, "accuracy": accuracy})
    summary.to_csv(out_dir / f"{out_name}_accuracy.csv", index=False)
 
    # Mean image quality for the measures this run has
    measures = []
    means = []
    for measure in ["blurred_area", "ssim", "lpips"]:
        if measure in df.columns:
            measures.append(measure)
            means.append(df[measure].mean())
 
    quality = pd.DataFrame({"measure": measures, "mean": means})
    quality.to_csv(out_dir / f"{out_name}_quality.csv", index=False)

 
def unnormalize(x, transform):
    """Undo the normalization of the transform.

        Input: (1,3,H,W) tensor from transform, model transform
        Return: (1,3,H,W) tensor with values 0-1
    """
    # Find the normalisation step of the transform
    for step in transform.transforms:
        if isinstance(step, Normalize):
            normalize = step
            break
        
    mean = torch.tensor(normalize.mean).view(3, 1, 1)
    std = torch.tensor(normalize.std).view(3, 1, 1)

    # x = (img - mean) / std
    # img = x * std + mean
    img = (x.detach() * std + mean).clamp(0, 1) 
        # stops trcking gradients in optimised blur
        # clamp for png values

    return img


def save_example(x, transform, path, out_dir, out_name):
    """Save only example images as png. 

        Input: (1,3,H,W) tensor from transform, model transform (to undo its normalization), 
            original photo path, output folder of the model, output file name
        Return: None
    """
    if path.name not in example_images:
        return

    img = unnormalize(x, transform)

    example_dir = out_dir / "examples" / out_name
    example_dir.mkdir(parents=True, exist_ok=True) # save in its own folder
    
    to_pil_image(img[0]).save(example_dir / f"{path.stem}.png") # change to png whatever the image is