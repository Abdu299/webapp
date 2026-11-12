"""Image Quality related functions. """

from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.utils.common import unnormalize


def load_lpips():
    """Loads the LPIPS network to compare images the way humans do."""
    return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True) # normalize=True: values 0-1


def image_quality(x, x_blurred, transform, lpips_network):
    """How much the blurred image differs from the original.

        Input: original and blurred (1,3,H,W) tensors from transform, model transform, LPIPS network 
        Return: [ssim, lpips]. ssim is 1 and lpips is 0 for identical images
    """
    
    original = unnormalize(x, transform)
    blurred = unnormalize(x_blurred, transform)

    ssim = structural_similarity_index_measure(blurred, original, data_range=1.0).item()
    lpips = lpips_network(blurred, original).item()
    return [ssim, lpips]
