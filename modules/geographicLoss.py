
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeographicLoss(nn.Module):
    """Country-aware geographic penalty using representative GPS coordinates.

    Coordinates are estimated from TRAIN images for each country. This is a
    rough geographic approximation, not an exact border-to-border distance.

    """

    def __init__(self, country_coordinates, alpha=0.3, temperature=1000):
        super().__init__()

        # Alpha controls how much the geographic penalty contributes to the loss.
        # Temperature controls how quickly the penalty increases with distance.
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if len(country_coordinates) == 0:
            raise ValueError("Need at least one country")

        self.alpha = alpha
        self.temperature = temperature

        # Convert country coordinates to a tensor with shape [num_countries, 2].
        # Each row contains the representative latitude and longitude of a country.
        coords = torch.as_tensor(country_coordinates, dtype=torch.float32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("country_coordinates must be [num_countries, 2]")

        # Convert degrees to radians because PyTorch trigonometric functions
        # expect radians instead of degrees.
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])

        # Calculate latitude and longitude differences between every country pair.
        # Broadcasting creates matrices of shape [num_countries, num_countries].
        dlat = lat[None, :] - lat[:, None]
        dlon = lon[None, :] - lon[:, None]

        # Apply the Haversine formula to calculate great-circle distances.
        # This accounts for the Earth's curvature rather than using flat distances.
        a = torch.sin(dlat / 2)**2 + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon / 2)**2

        # Convert the Haversine result into distances in kilometers.
        # 6371 is the approximate radius of the Earth in kilometers.
        # clamp(0, 1) prevents numerical errors in the arcsine calculation.
        distances = 2 * 6371 * torch.asin(torch.sqrt(a.clamp(0, 1)))

        # Convert distances into penalties between 0 and 1.
        # Nearby countries receive smaller penalties than distant countries.
        # With temperature=1000, the penalty increases over a scale of 1000 km.
        # The penalty approaches 1 as the distance becomes very large.
        costs = 1 - torch.exp(-distances / temperature)

        # Predicting the correct country should have zero geographic penalty.
        costs.fill_diagonal_(0)

        # Store the penalty matrix as a model buffer.
        # Unlike trainable parameters, buffers are not updated by the optimizer.
        # They are automatically moved to the same device as the loss module.
        self.register_buffer("distance_costs", costs)

    def forward(self, logits, labels):

        # The number of model output classes must match the number of countries
        # used when constructing the geographic penalty matrix.
        if logits.shape[1] != self.distance_costs.shape[0]:
            raise ValueError("Country count does not match logits")

        
        # Standard cross-entropy penalizes incorrect country predictions.
        # It does not consider how far the predicted country is from the real one.
        ce = F.cross_entropy(logits, labels)

        # Softmax converts logits into probabilities for each country.
        # Select the distance penalties from each true country to all countries.
        # Multiply each penalty by the predicted probability of that country.
        # Sum across countries and average across the batch.
        # This gives an expected geographic penalty for the model's predictions.
        geographic_penalty = (self.distance_costs[labels] * F.softmax(logits, dim=1)).sum(dim=1).mean()

        # Combine classification loss with the geographic penalty.
        # The model is encouraged to predict the correct country while also
        # assigning less probability to geographically distant countries.
        return ce + self.alpha * geographic_penalty
