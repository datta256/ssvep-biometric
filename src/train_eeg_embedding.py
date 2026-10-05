import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from paths import CACHE_ROOT, resolve_cache_file

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128
NUM_SUBJECTS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class EEGDataset(Dataset):

    def __init__(self, metadata):

        self.metadata = metadata

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        item = self.metadata[index]

        eeg = np.load(resolve_cache_file(item["file"])).astype(np.float32)

        # Shape:
        # channels x samples
        eeg = torch.from_numpy(eeg)

        # Add convolution channel dimension:
        # 1 x channels x samples
        eeg = eeg.unsqueeze(0)

        label = item["subject"] - 1

        return eeg, label


# ---------------------------------------------------------
# EEG embedding network
# ---------------------------------------------------------

class EEGEmbeddingNet(nn.Module):

    def __init__(self, num_subjects=100, embedding_size=128):

        super().__init__()

        self.features = nn.Sequential(

            # Temporal filtering
            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 31),
                padding=(0, 15),
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ELU(),

            # Spatial filtering across 64 EEG channels
            nn.Conv2d(
                16,
                32,
                kernel_size=(64, 1),
                groups=16,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(0.25),

            # More temporal processing
            nn.Conv2d(
                32,
                64,
                kernel_size=(1, 15),
                padding=(0, 7),
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            nn.Dropout(0.25)
        )

        self.embedding = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                64,
                embedding_size
            ),

            nn.LayerNorm(embedding_size)
        )

        self.classifier = nn.Linear(
            embedding_size,
            num_subjects
        )


    def forward(self, x):

        x = self.features(x)

        embedding = self.embedding(x)

        logits = self.classifier(embedding)

        return embedding, logits


# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

metadata_file = CACHE_ROOT / "metadata.json"

with open(metadata_file, "r") as f:

    metadata = json.load(f)


print()
print("Training trials:", len(metadata))
print("Subjects:", NUM_SUBJECTS)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Embedding:", EMBEDDING_SIZE)
print()


# ---------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------
metadata = [
    item
    for item in metadata
    if item["frequency"] in (8.0, 9.0, 10.0, 11.0)
]

print(
    "Training trials (9 Hz only):",
    len(metadata)
)

print("Training trials after removing 12 Hz:", len(metadata))
dataset = EEGDataset(metadata)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = EEGEmbeddingNet(
    num_subjects=NUM_SUBJECTS,
    embedding_size=EMBEDDING_SIZE
)

model = model.to(DEVICE)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (x, y) in enumerate(loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        embedding, logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

        if batch_index % 25 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch_index + 1}/{len(loader)} "
                f"Loss {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / len(loader)
    )

    epoch_accuracy = (
        correct / total * 100
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} "
        f"Loss {epoch_loss:.4f} "
        f"Accuracy {epoch_accuracy:.2f}%\n"
    )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

output_file = CACHE_ROOT / "eeg_embedding_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "embedding_size": EMBEDDING_SIZE,
        "num_subjects": NUM_SUBJECTS,
    },
    output_file
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Model:", output_file)