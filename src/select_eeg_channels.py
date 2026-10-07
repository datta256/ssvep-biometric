import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eeg_channels import CHANNEL_NAMES
from eeg_embedding_model import EEGEmbeddingNet
from paths import CACHE_ROOT, resolve_cache_file

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_SIZE = 128
NUM_SUBJECTS = 100
DEFAULT_FREQUENCIES = (8.0, 9.0, 10.0, 11.0)

parser = argparse.ArgumentParser(
    description=(
        "Rank EEG channels by held-out subject-classification accuracy "
        "after single-channel occlusion."
    )
)
parser.add_argument(
    "--train-sessions",
    type=int,
    nargs="+",
    choices=range(7),
    default=(0, 1, 2),
    help="Sessions used to train the ranking model.",
)
parser.add_argument(
    "--validation-session",
    type=int,
    choices=range(7),
    default=3,
    help="Held-out session used for channel ranking.",
)
parser.add_argument(
    "--frequencies",
    type=float,
    nargs="+",
    choices=(8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0),
    default=DEFAULT_FREQUENCIES,
    help="Stimulus frequencies used for ranking.",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

train_sessions = tuple(sorted(set(args.train_sessions)))
frequencies = tuple(sorted(set(args.frequencies)))
if args.validation_session in train_sessions:
    parser.error("Validation session must be excluded from training.")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

metadata_file = CACHE_ROOT / "metadata.json"
with open(metadata_file, "r") as file:
    metadata = json.load(file)

training_items = [
    item
    for item in metadata
    if item["session"] in train_sessions
    and float(item["frequency"]) in frequencies
]
validation_items = [
    item
    for item in metadata
    if item["session"] == args.validation_session
    and float(item["frequency"]) in frequencies
]
if not training_items or not validation_items:
    raise ValueError(
        "No training or validation trials matched the requested sessions "
        "and frequencies. Check the cache and arguments."
    )

for split_name, items in (
    ("training", training_items),
    ("validation", validation_items),
):
    subjects = {int(item["subject"]) for item in items}
    if subjects != set(range(1, NUM_SUBJECTS + 1)):
        raise ValueError(
            f"{split_name.capitalize()} data must contain all "
            f"{NUM_SUBJECTS} subjects; found {len(subjects)}."
        )

print("Training sessions:", train_sessions)
print("Validation session:", args.validation_session)
print("Frequencies:", frequencies)
print("Training trials:", len(training_items))
print("Validation trials:", len(validation_items))


class EEGDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        eeg = np.load(resolve_cache_file(item["file"])).astype(np.float32)
        if eeg.shape != (len(CHANNEL_NAMES), 1500):
            raise ValueError(
                f"Expected trial shape ({len(CHANNEL_NAMES)}, 1500), "
                f"got {eeg.shape}: {item['file']}"
            )
        return torch.from_numpy(eeg).unsqueeze(0), int(item["subject"]) - 1


training_loader = DataLoader(
    EEGDataset(training_items),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
)
validation_loader = DataLoader(
    EEGDataset(validation_items),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

model = EEGEmbeddingNet(
    num_subjects=NUM_SUBJECTS,
    embedding_size=EMBEDDING_SIZE,
    num_channels=len(CHANNEL_NAMES),
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training a 64-channel ranking model for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for eeg, labels in training_loader:
        eeg = eeg.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"loss={total_loss / len(training_loader):.4f} "
        f"accuracy={100 * total_correct / total:.2f}%"
    )


def validation_accuracy(masked_channel=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg, labels in validation_loader:
            if masked_channel is not None:
                eeg[:, :, masked_channel, :] = 0
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, logits = model(eeg)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


baseline_accuracy = validation_accuracy()
results = []
print(f"Unmasked validation accuracy: {baseline_accuracy * 100:.2f}%")
for channel_index, channel_name in enumerate(CHANNEL_NAMES):
    accuracy = validation_accuracy(masked_channel=channel_index)
    results.append(
        {
            "index": channel_index,
            "name": channel_name,
            "accuracy": accuracy,
            "accuracy_drop": baseline_accuracy - accuracy,
        }
    )

results.sort(key=lambda result: result["accuracy_drop"], reverse=True)
print()
print("Single-channel occlusion importance (larger drop = more important)")
for rank, result in enumerate(results, start=1):
    print(
        f"{rank:02d}. {result['name']:>4} "
        f"(index {result['index']:02d}) "
        f"masked_accuracy={result['accuracy'] * 100:.2f}% "
        f"drop={result['accuracy_drop'] * 100:+.2f} pp"
    )

selected = results[:8]
print()
print("Candidate 8 channels (screening result; retrain and verify separately):")
print(" ".join(result["name"] for result in selected))
print("Indices:", " ".join(str(result["index"]) for result in selected))
print(
    "Note: this ranks single-channel ablation in a 100-class model. "
    "It does not establish the best jointly trained 8-channel biometric "
    "subset; use the strict enrollment/calibration/test protocol to validate."
)
