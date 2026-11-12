import sys

import numpy as np
import pandas as pd
import torch

from src.data_prep.privacy_split import data_dir
from src.privacy_network.model import judge_module
from src.utils.common import haversine, radius_km


if __name__ == "__main__":
    model_name, splits = sys.argv[1], sys.argv[2:]
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = (True)

    model_module = judge_module(model_name)
    model = model_module.load_model().to(device)
    for split in splits:
        df = pd.read_csv(data_dir / split / "photos.csv", dtype={"name": str})
        guesses = model_module.guess(model, df, device)

        # NaN guesses give NaN distances
        km = haversine(
            torch.tensor(df["lat"].values),
            torch.tensor(df["lon"].values),
            torch.tensor(guesses[:, 0]),
            torch.tensor(guesses[:, 1]),
        ).numpy()
        pd.DataFrame({"name": df["name"], "km": km}).to_csv(
            data_dir / split / f"unblurred_km_{model_name}.csv", index=False
        )

        print(
            f"{split}: {len(df)} photos, {np.isnan(km).sum()} unreadable, {(km <= radius_km).sum()} located within {radius_km} km"
        )
