import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from eeg_embedding_model import EEGEmbeddingNet
from eeg_channels import CHANNEL_NAMES
from paths import CACHE_ROOT, resolve_cache_file

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128
NUM_SUBJECTS = 100
SUPPORTED_FREQUENCIES = (
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
)
DEFAULT_TRAINING_FREQUENCIES = (8.0, 9.0, 10.0, 11.0)
DEFAULT_TRAINING_SESSIONS = (0, 1, 2, 3, 4)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--frequencies",
    type=float,
    nargs="+",
    choices=SUPPORTED_FREQUENCIES,
    default=DEFAULT_TRAINING_FREQUENCIES,
    help="SSVEP frequencies to include in training.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for model initialization and trial shuffling.",
)
parser.add_argument(
    "--sessions",
    type=int,
    nargs="+",
    choices=range(7),
    default=DEFAULT_TRAINING_SESSIONS,
    help="Session numbers to use for training.",
)
parser.add_argument(
    "--channels",
    nargs="+",
    choices=CHANNEL_NAMES,
    default=None,
    help="EEG channels to use; defaults to all 64 channels.",
)
parser.add_argument(
    "--output",
    default=str(CACHE_ROOT / "eeg_embedding_model.pt"),
    help="Checkpoint output path.",
)
args = parser.parse_args()
training_frequencies = tuple(sorted(set(args.frequencies)))
training_sessions = tuple(sorted(set(args.sessions)))
if args.channels and len(set(args.channels)) != len(args.channels):
    parser.error("Channel names must not be repeated.")
training_channels = tuple(
    CHANNEL_NAMES.index(channel)
    for channel in (args.channels or CHANNEL_NAMES)
)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class EEGDataset(Dataset):

    def __init__(self, metadata, channel_indices):

        self.metadata = metadata
        self.channel_indices = channel_indices

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        item = self.metadata[index]

        eeg = np.load(resolve_cache_file(item["file"])).astype(np.float32)
        eeg = eeg[self.channel_indices, :]

        # Shape:
        # channels x samples
        eeg = torch.from_numpy(eeg)

        # Add convolution channel dimension:
        # 1 x channels x samples
        eeg = eeg.unsqueeze(0)

        label = item["subject"] - 1

        return eeg, label


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
print("Random seed:", args.seed)
print("Channels:", [CHANNEL_NAMES[index] for index in training_channels])
print()


# ---------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------
metadata = [
    item
    for item in metadata
    if item["frequency"] in training_frequencies
    and item["session"] in training_sessions
]

print(
    "Training sessions:",
    training_sessions
)
print(
    f"Training trials ({', '.join(map(str, training_frequencies))} Hz):",
    len(metadata)
)
dataset = EEGDataset(metadata, training_channels)

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
    embedding_size=EMBEDDING_SIZE,
    num_channels=len(training_channels),
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

output_file = args.output

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "embedding_size": EMBEDDING_SIZE,
        "num_subjects": NUM_SUBJECTS,
        "training_frequencies": list(training_frequencies),
        "training_sessions": list(training_sessions),
        "channel_indices": list(training_channels),
        "channel_names": [CHANNEL_NAMES[index] for index in training_channels],
        "seed": args.seed,
    },
    output_file
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Model:", output_file)