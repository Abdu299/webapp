import numpy as np
from geoclip import GeoCLIP
from tqdm import tqdm


def load_model():
    return GeoCLIP().eval().requires_grad_(False)


def predict(
    model,
    path,
):
    top_gps, top_prob = model.predict(str(path), top_k=1)  # (1, 2), (1,)
    return top_gps[0].tolist()  # lat, lon in degrees


def guess(model, df, device):
    # lat, lon in degrees, NaN for unreadable photos
    guesses = np.full((len(df), 2), np.nan)  # (N, 2)
    for i, path in enumerate(tqdm(df["path"], desc="geoclip")):
        try:
            guesses[i] = predict(model, path)
        except OSError:
            continue
    return guesses
