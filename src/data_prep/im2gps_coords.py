import csv
from pathlib import Path
import pandas as pd
from PIL import Image

from src.data_prep.privacy_split import save_split

images = sorted(Path("privacyData/im2gps").rglob("*.jpg"))


def print_image_comments(image):
    """Prints all JPEG comments of images[image] (image is an index)."""
    path = images[image]
    img = Image.open(path)
    print(f"Image: {path.name}")
    for marker, data in img.applist:
        if marker == "COM":
            print(data)
            
            
def get_latitude(i):
    """Latitude of images[i] from its JPEG comment, or None."""
    img = Image.open(images[i])
    for marker, data in img.applist:
        if marker == "COM" and b"latitude: " in data:
            lat = float(data.partition(b"latitude:")[2].split(b"\x00")[0])
            # print(f"Latitude: {lat}")
            return lat
    print("Could not find latitude.")
    return None


def get_longitude(i):
    """Longitude of images[i] from its JPEG comment, or None."""
    img = Image.open(images[i])
    for marker, data in img.applist:
        if marker == "COM" and b"longitude: " in data:
            lon = float(data.partition(b"longitude:")[2].split(b"\x00")[0])
            # print(f"Longitude: {lon}")
            return lon
    print("Could not find longitude.")
    return None


def save_coordinates():
    """Save coordinates for each image."""
    print("Saving coordinates for images ... ")
    with open("data/im2gps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "lat", "lon"])
        for i, path in enumerate(images):
            w.writerow([path.name, get_latitude(i), get_longitude(i)])


def save_privacy_split():
    """Saves the photos with name, lat, lon, path"""
    df = pd.read_csv("data/im2gps.csv")
    paths = {path.name: str(path) for path in images}
    df["name"] = [Path(filename).stem for filename in df["filename"]]  # like the png names of the other methods
    df["path"] = df["filename"].map(paths)
    save_split(df[["name", "lat", "lon", "path"]], "im2gps")


if __name__ == "__main__":
    save_coordinates()
    save_privacy_split()