import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = efficientnet_b0(weights=None).features

        first_conv = nn.Conv2d(
            4,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        with torch.no_grad():
            first_conv.weight[:, :3] = self.features[0][0].weight
            first_conv.weight[:, 3] = self.features[0][0].weight.mean(dim=1)

        self.features[0][0] = first_conv

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x