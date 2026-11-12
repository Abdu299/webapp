import math

import lightning as pl
import torch

from src.blur.extremal import BlurPyramid
from src.data_prep.privacy_split import load_split
from src.privacy_network import geoclip, geovit, osv5m
from src.privacy_network.osv5m import cell_centres, cells_near_true_location, load_model
from src.privacy_network.privacy_network import PrivacyNetwork
from src.utils.common import radius_km, unnormalize


# the model file with load_model, guess and judge
def judge_module(name):
    if name == "osv5m":
        return osv5m
    elif name == "geovit":
        return geovit
    elif name == "geoclip":
        return geoclip


def log_prob_near(logits, near):
    """Log of the total probability the model puts on cells near the true location

        logits: (B, num_cells) cell scores from the model
        near: (B, num_cells) bool, from cells_near_true_location
        Returns: (B,) one log-probability per photo

    """
    # log probabilities of every cell
    log_probs = torch.log_softmax(logits, dim=1)

    # cells that are not near get probability 0, which is log(0) = -inf
    log_probs_near = log_probs.masked_fill(~near, float("-inf"))

    # log(p1 + p2 + ...) over the near cells
    log_p_per_photo = torch.logsumexp(log_probs_near, dim=1)
    return log_p_per_photo # (B,)


def deletion_term(log_p, hidden, log_floor):
    """Low when the model puts little probability near the true location"""

    floored = log_p.clamp(min=log_floor)

    # hidden photos get a floor, so they are not blurred more for no gain
    deletion_per_photo = torch.where(hidden, floored, log_p)  # (B,)

    return deletion_per_photo.mean()


def area_term(mask):
    return mask.mean() # 0 = nothing blurred, 1 = everything fully blurred

def smoothness_term(mask): # (B, 1, H, W)
    dx = (mask[..., :, 1:] - mask[..., :, :-1]).abs().mean() # (B, 1, H, W-1)
    dy = (mask[..., 1:, :] - mask[..., :-1, :]).abs().mean() # (B, 1, H-1, W)
    return dx + dy


class Model(pl.LightningModule):
    def __init__(self, config, epochs):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.area_weight = config["area_weight"]
        self.smoothness_weight = config["smoothness_weight"]
        self.log_floor = math.log(config["floor"])
        self.epochs = epochs
        self.strict_loading = False  # checkpoints hold only the decoder
        self.name = config["name"]  # the judge

        self.judge_module = judge_module(self.name)
        self.model = PrivacyNetwork(encoder=self.judge_module.load_model(), block=config.get("block", 1))

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        super().train(mode)
        self.model.encoder.eval()  # the judge stays frozen, also while training
        return self

    def shared_step(self, batch, stage):
        x, lat, lon = batch["image"], batch["lat"], batch["lon"]

        # Blur with the predicted mask
        mask = self.model(x).float()  # (B, 1, 224, 224), 1 = most blur
        x_blurred = BlurPyramid(x).apply(mask)

        # Judge's scores on the blurred photos, which classes are near the true location, and if the top guess is hidden
        logits, near, hidden = self.judge_module.judge(self.model.encoder, x_blurred, lat, lon)

        # a photo with no near cell or country gives log(0) and NaN gradients, so leave it out
        ok = near.any(dim=1)
        log_p = log_prob_near(logits[ok], near[ok])
        hidden = hidden[ok]

        # Loss
        deletion = deletion_term(log_p, hidden, self.log_floor)
        area = area_term(mask)
        smoothness = smoothness_term(mask)
        loss = deletion + self.area_weight * area + self.smoothness_weight * smoothness

        B = len(lat)

        self.log(f"{stage}/loss", loss, batch_size=B, prog_bar=True)
        self.log(f"{stage}/deletion", deletion, batch_size=B)
        self.log(f"{stage}/area", area, batch_size=B, prog_bar=True)
        self.log(f"{stage}/smoothness", smoothness, batch_size=B)
        self.log(f"{stage}/hidden", hidden.float().mean(), batch_size=B, prog_bar=True)
        self.log(f"{stage}/prob_near", log_p.exp().mean(), batch_size=B)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.log_images(batch["image"][:8])
        return self.shared_step(batch, "val")

    def log_images(self, x):
        """Masks and blurred photos of the same 8 validation photos each epoch, in TensorBoard."""
        mask = self.model(x).float()
        blurred = unnormalize(BlurPyramid(x).apply(mask).float().cpu(), self.model.encoder.transform)
        tb = self.loggers[0].experiment  # the TensorBoardLogger
        tb.add_images("val/mask", mask.cpu(), self.current_epoch)
        tb.add_images("val/blurred", blurred, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        """Saves only the decoder; the OSV-5M weights come from load_model."""
        state = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            k: v for k, v in state.items() if k.startswith("model.decoder.")
        }


if __name__ == "__main__":
    df = load_split("train")
    centres = cell_centres(load_model())
    lat = torch.tensor(df["lat"].values)
    lon = torch.tensor(df["lon"].values)

    no_cell = []
    for i in range(0, len(df), 1000):
        near = cells_near_true_location(centres, lat[i : i + 1000], lon[i : i + 1000])
        no_cell += list(df["name"].iloc[i : i + 1000][~near.any(dim=1).numpy()])
    print(f"{len(no_cell)} of {len(df)} train photos have no cell within {radius_km} km")
    print(no_cell[:20])