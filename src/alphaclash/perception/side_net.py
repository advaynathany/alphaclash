"""Small CNN: unit crop (64x64, box + band above) -> logit that the unit is the enemy's."""

import torch.nn as nn


def make_net() -> nn.Module:
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1))
