
import torch
import torch.nn as nn
from transformers import Dinov2Model


# Vision Transformer model for classifying images into countries.
# Uses a pretrained DINOv2 backbone to extract image features,
# followed by a new classification layer for our country labels.
class GeoVit(nn.Module):

    def __init__(self, num_classes, fine_tune="last2"):
        super().__init__()

        # Load pretrained DINOv2
        # DINOv2-small is a Vision Transformer pretrained using
        # self-supervised learning, so it already knows useful visual patterns.
        self.backbone = Dinov2Model.from_pretrained(
            "facebook/dinov2-small"
        )

        # Get the size of the feature vectors produced by DINOv2.
        # For DINOv2-small, the hidden size is 384.
        hidden_size = self.backbone.config.hidden_size

        # Add a new classification head that maps the image features
        # to one score for each country in the training dataset.
        # This layer is randomly initialized and learned during training.
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze the pretrained backbone by disabling gradient updates.
        # This prevents its weights from changing unless specific layers
        # are unfrozen below.
        for param in self.backbone.parameters():
            param.requires_grad = False

        if fine_tune == "last2":

            # Unfreeze only the last two Transformer encoder layers.
            # Earlier layers keep their pretrained features, while the
            # final layers can adapt to the geographic classification task.
            for layer in self.backbone.encoder.layer[-2:]:

                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif fine_tune == "last3":

            # Unfreeze the last three encoder layers.
            # This gives the model more capacity to adapt to our dataset,
            # but requires more computation than fine-tuning two layers.
            for layer in self.backbone.encoder.layer[-3:]:

                for param in layer.parameters():
                    param.requires_grad = True
                    

        elif fine_tune == "all":

            # Fine-tune the entire backbone
            # Every DINOv2 layer can now update its pretrained weights.
            # This offers more flexibility but increases training cost.
            for param in self.backbone.parameters():
                param.requires_grad = True

        else:
            raise ValueError("fine_tune must be 'last2', 'last3', or 'all'")


    # Run images through DINOv2 and convert the extracted features
    # into classification scores for all countries.
    def forward(self, images):

        # DINOv2 processes images as patches using Transformer layers.
        # The output contains a feature vector for each token.
        outputs = self.backbone(pixel_values=images)

        # CLS token represents the image
        # Index 0 selects the CLS token from every image in the batch.
        # The CLS token is used as a global representation of the image.
        # Shape: [batch_size, hidden_size].
        features = outputs.last_hidden_state[:, 0, :]

        # Predict country classes
        # The linear layer produces one logit per country.
        # Logits are raw scores, not probabilities. Cross-Entropy
        # converts them internally when calculating the training loss.
        logits = self.classifier(features)

        return logits
