import torch.nn as nn


class EEGEmbeddingNet(nn.Module):
    def __init__(self, num_subjects=100, embedding_size=128, num_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 31),
                padding=(0, 15),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(
                16,
                32,
                kernel_size=(num_channels, 1),
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(
                32,
                64,
                kernel_size=(1, 15),
                padding=(0, 7),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_size),
            nn.LayerNorm(embedding_size),
        )
        self.classifier = nn.Linear(embedding_size, num_subjects)

    def forward(self, x):
        features = self.features(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return embedding, logits
