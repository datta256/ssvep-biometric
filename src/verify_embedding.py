import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve


CACHE_ROOT = r"E:\ssvep-cache"

MODEL_FILE = os.path.join(
    CACHE_ROOT,
    "eeg_embedding_model.pt"
)

TEST_METADATA_FILE = os.path.join(
    CACHE_ROOT,
    "test_metadata.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMBEDDING_SIZE = 128
NUM_SUBJECTS = 100

# 12 Hz was completely excluded from training.
TRAINED_FREQUENCIES = {
    8.0, 8.5, 9.0, 9.5,
    10.0, 10.5, 11.0, 11.5
}

UNSEEN_FREQUENCY = 12.0


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class EEGEmbeddingNet(nn.Module):

    def __init__(
        self,
        num_subjects=100,
        embedding_size=128
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 31),
                padding=(0, 15),
                bias=False
            ),

            nn.BatchNorm2d(16),
            nn.ELU(),

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

            nn.LayerNorm(
                embedding_size
            )
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
# Load model
# ---------------------------------------------------------

print("Device:", DEVICE)

model = EEGEmbeddingNet(
    num_subjects=NUM_SUBJECTS,
    embedding_size=EMBEDDING_SIZE
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# Load held-out metadata
# ---------------------------------------------------------

with open(TEST_METADATA_FILE, "r") as f:

    metadata = json.load(f)


# Enrollment:
# Session 5, but ONLY frequencies the model saw during training.

enrollment_items = [
    item
    for item in metadata
    if item["session"] == 5
    and float(item["frequency"]) in (8.0, 9.0, 10.0, 11.0)
]


# Test:
# Session 6, ONLY the completely unseen 12 Hz frequency.

test_items = [
    item
    for item in metadata
    if item["session"] == 6
    and float(item["frequency"])
    == UNSEEN_FREQUENCY
]


print()
print("Protocol:")
print("Training frequencies:", sorted(TRAINED_FREQUENCIES))
print("Enrollment session: 5")
print("Enrollment frequencies: 8.0, 9.0, 10.0, 11.0 Hz")
print("Test session: 6")
print("Test frequency:", UNSEEN_FREQUENCY)

print()
print(
    "Enrollment trials:",
    len(enrollment_items)
)

print(
    "12 Hz test trials:",
    len(test_items)
)


# ---------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------

def extract_embeddings(items):

    embeddings = []
    subjects = []

    with torch.no_grad():

        for index, item in enumerate(items):

            eeg = np.load(
                item["file"]
            ).astype(np.float32)

            eeg = torch.from_numpy(eeg)

            # 1 x 1 x 64 x 1500
            eeg = eeg.unsqueeze(0).unsqueeze(0)

            eeg = eeg.to(
                DEVICE,
                non_blocking=True
            )

            embedding, _ = model(eeg)

            embedding = (
                embedding
                .squeeze(0)
                .cpu()
                .numpy()
            )

            embeddings.append(embedding)

            subjects.append(
                item["subject"]
            )

            if (index + 1) % 100 == 0:

                print(
                    f"Extracted "
                    f"{index + 1}/{len(items)}"
                )

    return (
        np.asarray(embeddings),
        np.asarray(subjects)
    )


print()
print("Extracting enrollment embeddings...")

enrollment_embeddings, enrollment_subjects = (
    extract_embeddings(enrollment_items)
)


print()
print("Extracting unseen 12 Hz embeddings...")

test_embeddings, test_subjects = (
    extract_embeddings(test_items)
)


# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

def normalize_embeddings(x):

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / (
        norms + 1e-12
    )


enrollment_embeddings = normalize_embeddings(
    enrollment_embeddings
)

test_embeddings = normalize_embeddings(
    test_embeddings
)


# ---------------------------------------------------------
# Build enrollment templates
# ---------------------------------------------------------

templates = {}

for subject in range(
    1,
    NUM_SUBJECTS + 1
):

    subject_embeddings = (
        enrollment_embeddings[
            enrollment_subjects == subject
        ]
    )

    if len(subject_embeddings) == 0:
        continue

    template = subject_embeddings.mean(
        axis=0
    )

    template = template / (
        np.linalg.norm(template)
        + 1e-12
    )

    templates[subject] = template


print()
print(
    "Enrollment templates:",
    len(templates)
)


# ---------------------------------------------------------
# Score unseen 12 Hz trials
# ---------------------------------------------------------

genuine_scores = []
impostor_scores = []


print()
print("Scoring unseen 12 Hz trials...")


for i in range(
    len(test_embeddings)
):

    embedding = test_embeddings[i]

    true_subject = int(
        test_subjects[i]
    )

    for subject, template in templates.items():

        score = float(
            np.dot(
                embedding,
                template
            )
        )

        if subject == true_subject:

            genuine_scores.append(score)

        else:

            impostor_scores.append(score)


genuine_scores = np.asarray(
    genuine_scores
)

impostor_scores = np.asarray(
    impostor_scores
)


# ---------------------------------------------------------
# ROC / EER
# ---------------------------------------------------------

scores = np.concatenate([
    genuine_scores,
    impostor_scores
])

labels = np.concatenate([
    np.ones(
        len(genuine_scores)
    ),

    np.zeros(
        len(impostor_scores)
    )
])


auc = roc_auc_score(
    labels,
    scores
)


fpr, tpr, thresholds = roc_curve(
    labels,
    scores
)


fnr = 1.0 - tpr


eer_index = np.argmin(
    np.abs(
        fpr - fnr
    )
)


eer = (
    fpr[eer_index]
    + fnr[eer_index]
) / 2


eer_threshold = thresholds[
    eer_index
]


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 60)
print("UNSEEN-FREQUENCY EEG VERIFICATION")
print("=" * 60)

print(
    "Training frequencies:",
    sorted(TRAINED_FREQUENCIES)
)

print(
    "Test frequency:",
    UNSEEN_FREQUENCY,
    "Hz"
)

print()

print(
    "Genuine attempts:",
    len(genuine_scores)
)

print(
    "Impostor attempts:",
    len(impostor_scores)
)

print()

print(
    f"Genuine mean:    "
    f"{genuine_scores.mean():.6f}"
)

print(
    f"Genuine median:  "
    f"{np.median(genuine_scores):.6f}"
)

print(
    f"Impostor mean:   "
    f"{impostor_scores.mean():.6f}"
)

print(
    f"Impostor median: "
    f"{np.median(impostor_scores):.6f}"
)

print()

print(
    f"ROC-AUC:         "
    f"{auc:.6f}"
)

print(
    f"EER:             "
    f"{eer:.6f}"
)

print(
    f"EER percentage:  "
    f"{eer * 100:.4f}%"
)

print(
    f"EER threshold:   "
    f"{eer_threshold:.6f}"
)

print("=" * 60)