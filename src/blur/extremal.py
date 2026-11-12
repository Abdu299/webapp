"""Methods for perturbation based blur based on Fong et al."""

import math
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from src.utils.common import haversine


class BlurPyramid:
    """A perturbation pyramid that contains L+1 progressively more perturbed versions of the image.

    Has L+1 copies of the image, each blurred as a whole with
    sigma_l = max_sigma * l/L, for l = 0, ..., L
    Level 0 is the unblurred original image, L is blurred with max_sigma
    """

    def __init__(self, x, max_sigma=20, levels=8):
        self.L = levels
        self.pyramid = torch.stack(
            [apply_gaussian_blur(x, max_sigma * l / levels) for l in range(levels + 1)]
        )

    def apply(self, m):
        """
        m: mask. tensor of shape (B, 1, H, W)
            Each pixel has value in [0, 1] equiv to amount of blur (0 = no blur, 1 = max blur).

        return: B blurred images of shape (B,3,H,W)
            Each pixel blurred as much as the mask decides for it
        """

        # Position of each pixel between the levels, like 2.4 is between levels 2 and 3.
        pos = m * self.L

        # Level numbers 0, ..., L with shape (L+1,1,1,1,1)
        blur_levels = torch.arange(self.L + 1, device=m.device, dtype=m.dtype).view(
            -1, 1, 1, 1, 1
        )

        # ReLU sets all levels weights to 0 except the two closest to the pixel
        # Weight 1-distance per level and pixel
        weights = F.relu(1 - (pos.unsqueeze(0) - blur_levels).abs())

        # Mix each pixel from its two nearest levels using the weights
        # shape (B,3,H,W)
        return (weights * self.pyramid).sum(0)


def apply_gaussian_blur(x, sigma):
    """See 3.4"""
    if sigma <= 0:
        return x
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(x, kernel_size, sigma)  # blurred image


def smooth_max(values, dim, T):
    """Smooth max.

    Input: tensor of values, dim/axis, and temperature T
        (f is stored as tensors of values)
    Returns: Tensor same shape as values without dim dimension
    """
    return (torch.sum(values * torch.exp(values / T), dim)) / (
        torch.sum(torch.exp(values / T), dim)
    )


def area_loss(m, areas):
    """Rank-based area constraint. One loss per mask.

    m: masks of shape (B,1,H,W), with each value in [0,1] (the amount of blur at that pixel)
    areas: B target areas, where areas[i] is the fraction of the image mask i should blur
    """

    B = m.shape[0]  # number of masks

    sorted_m = (
        m.reshape(B, -1).sort(dim=1).values
    )  # flatten and sort mask values, incr order (shape (B, N))
    N = sorted_m.shape[1]  # number of pixels

    # target sorted mask: (1 - a) * N zeros followed by a * N ones
    target = torch.zeros_like(sorted_m)
    for i, a in enumerate(areas):
        target[i, int((1 - a) * N) :] = 1

    # mean sqrd diff between sorted mask and target (shape (B,))
    return ((sorted_m - target) ** 2).mean(dim=1)


def log_prob_near(logits, near_cells):
    """Log of the total probability the model puts on cells within the radius of the true location."""
    # log probability of every cell
    # then combine those of the near cells into just one log-probability
    return torch.log_softmax(logits, dim=1)[:, near_cells].logsumexp(dim=1)


def kernel(u):
    """
    k(z) = exp(-max(0, z - 1)^2 / 4)
    k(u) = k(||u||)
    """
    z = u.norm(dim=-1)  # z = ||u||
    return torch.exp(
        -(z - 1).clamp(min=0).pow(2) / 4
    )  # k(z) = exp(-max(z - 1, 0)^2 / 4)


class SmoothMask:
    """Smooth max-convolution upsampling of a low-resolution mask. See Fong et al  3.5, eq. 6-7."""

    def __init__(self, H, W, step=7, radius=4, T=0.05, device="cpu"):
        """
        H, W: size of the full-resolution mask (the image size)
        step: lattice step rho (the number of pixels between neighbouring lattice points)
        radius: number of lattice points in each direction that can affect a pixel (4, as in their implemnetation)
        T: temperature of the smooth max, with 1/T ≈ 20 in the paper.
        """
        self.H, self.W = H, W
        self.step, self.radius, self.T = step, radius, T

        # Lattice size, with one extra point so the bottom and right edge pixels have a lattice point on both sides
        self.h = math.ceil(H / step) + 1
        self.w = math.ceil(W / step) + 1

        # Position of every pixel inside its lattice cell, in lattice units from 0 up to 1.
        frac_y = (torch.arange(self.h * step) % step / step).view(-1, 1)
        frac_x = (torch.arange(self.w * step) % step / step).view(1, -1)

        # Offsets (dy, dx) from a pixel's cell to each lattice point that can affect it.
        self.offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]

        # For each offset, the kernel value at every pixel.
        self.kernels = []
        for dy, dx in self.offsets:
            # Offset from every pixel to this lattice point in lattice units, shape (h * step, w * step, 2).
            u = torch.stack(torch.broadcast_tensors(frac_y - dy, frac_x - dx), dim=-1)
            self.kernels.append(kernel(u).to(device))

    def __call__(self, m_bar):
        """
        m_bar:  low-resolution masks of shape (B, 1, h, w), values in [0, 1].
        return: smooth full-resolution masks of shape (B, 1, H, W), values in [0, 1].
        """
        r, h, w = self.radius, self.h, self.w

        # Pad with zeros so lattice points beyond the image edge count as unblurred.
        padded = F.pad(m_bar, (r, r, r, r))
        candidates = []
        for (dy, dx), k in zip(self.offsets, self.kernels):
            # For each cell, the value of the lattice point at offset (dy, dx) from it.
            neighbour = padded[..., r + dy : r + dy + h, r + dx : r + dx + w]

            # Repeat each value over the step x step pixels of its cell.
            neighbour = neighbour.repeat_interleave(self.step, dim=2).repeat_interleave(
                self.step, dim=3
            )

            # This lattice point's candidate value at every pixel, k(u - v) * m_bar(v)
            candidates.append(k * neighbour)

        f = torch.stack(candidates)  # (K, B, 1, h*step, w*step)

        # Smooth max over the candidates (eq 7), which is close to the largest of them
        m = smooth_max(f, dim=0, T=self.T)

        # Crop the extra lattice pixels to the image size
        return m[..., : self.H, : self.W]


def extremal_blur(
    model,
    x,
    true_coords,
    cell_centers,
    target_coords=None,
    areas=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8),
    radius_km=750.0,
    max_sigma=20.0,
    levels=8,
    step=7,
    iters=1600,
    lr=0.01,
    lam=300.0,
    record_every=100,
):
    """
        model: frozen geolocation classifier, (B, 3, H, W) -> (B, num_cells) logits
        x: image of shape (1, 3, H, W) in the space the model expects
        true_coords: true location (lat, lon) in degrees
        cell_centers: tensor of shape (num_cells, 2) with each cell's centre (lat, lon) in degrees
        target_coords: None to hide the true location (untargeted), or (lat, lon) in degrees to move the guess there (targeted)
        record_every: how often (in iterations) the blurred area and probabilities are recorded
        Returns:
            dict w areas as keys and values (blurred image, full-resolution mask, distance in km of the predicted cell to the true location)
            convergence: list of rows every record_every iterations, per area: blurred area and probability on the cells within radius_km of the true location (and of the target, if targeted)

    Settings follow Fong et al. (2019), Sec. 4 "Implementation details":
      - areas {0.05, ..., 0.8}, masks initialised to all ones, SGD with momentum 0.9,
      - 1600 iterations, lambda = 300 doubled at 1/3 and 2/3, 1/T = 20, kernel
      - k(z) = exp(-max(0, z - 1)^2 / 4).

    Things not given in the paper, taken from the authors implementation:
      lr = 0.01, SGD dampening = 0.9, max_sigma = 20, lattice step = 7, 8 levels, radius 4.

    Score is the log-probability of all cells
      within radius_km (deletion game), not a single class logit.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dev = x.device
    B, H, W = len(areas), x.shape[2], x.shape[3]
    # cell centres on the same device as the model for indexing w its predictions
    cell_centers = cell_centers.to(dev)

    # cells within radius_km of the true location, and of the target if targeted
    near_true = (
        haversine(
            cell_centers[:, 0], cell_centers[:, 1], true_coords[0], true_coords[1]
        )
        < radius_km
    )
    near_target = None
    if target_coords is not None:
        near_target = (
            haversine(
                cell_centers[:, 0],
                cell_centers[:, 1],
                target_coords[0],
                target_coords[1],
            )
            < radius_km
        )

    pyramid = BlurPyramid(x, max_sigma, levels)
    gen = SmoothMask(H, W, step=step, device=dev)

    # masks initialised to all ones, which in
    # deletion game starts from a fully blurred image
    m_bar = torch.ones((B, 1, gen.h, gen.w), device=dev, requires_grad=True)
    opt = torch.optim.SGD([m_bar], lr=lr, momentum=0.9, dampening=0.9)

    convergence = []

    for t in tqdm(range(iters), desc="Iterations", leave=False):
        lam_t = lam * (
            2 ** (3 * t // iters)
        )  # doubled at 1/3 and 2/3 of the iterations
        m = gen(m_bar)
        logits = model(pyramid.apply(m))  # (B, num_cells), one row for each target area

        log_prob_true = log_prob_near(logits, near_true)
        if near_target is None:
            objective = log_prob_true  # untargeted: lower the probability near the true location
        else:
            log_prob_target = log_prob_near(logits, near_target)
            objective = (
                -log_prob_target
            )  # targeted: raise the probability near the target

        loss = (objective + lam_t * area_loss(m, areas)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            m_bar.clamp_(0, 1)

        # blurred area and probabilities every record_every iterations + at the last iteration
        if t % record_every == 0 or t == iters - 1:
            for i, a in enumerate(areas):
                row = {
                    "iteration": t,
                    "area": a,
                    "blurred_area": m[i].mean().item(),
                    "prob_true": log_prob_true[i].exp().item(),
                }
                if near_target is not None:
                    row["prob_target"] = log_prob_target[i].exp().item()
                convergence.append(row)

    with torch.no_grad():
        m = gen(m_bar)
        x_blurred = pyramid.apply(m)
        pred = model(x_blurred).argmax(1)
        d = haversine(
            cell_centers[pred, 0], cell_centers[pred, 1], true_coords[0], true_coords[1]
        )

    result = {
        float(a): (x_blurred[i : i + 1], m[i : i + 1], float(d[i]))
        for i, a in enumerate(areas)
    }
    return result, convergence


def smallest_private_area(result, radius_km=750.0):
    """Extremal area a*: the smallest area whose blur moves the guess beyond radius_km"""
    for area in sorted(result.keys()):
        blurred_image, mask, distance_km = result[area]
        if distance_km > radius_km:
            return area
    # if no area moved the guess far enough
    return None

