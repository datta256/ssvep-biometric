import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from paths import CACHE_ROOT, resolve_cache_file

MODEL_FILE = CACHE_ROOT / "eeg_embedding_model.pt"
TEST_METADATA_FILE = CACHE_ROOT / "test_metadata.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMBEDDING_SIZE = 128
NUM_SUBJECTS = 100

# 12 Hz was completely excluded from training.
SUPPORTED_FREQUENCIES = (
    8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
)
DEFAULT_TRAINED_FREQUENCIES = (8.0, 9.0, 10.0, 11.0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--enrollment-frequencies",
    type=float,
    nargs="+",
    choices=SUPPORTED_FREQUENCIES,
    help="Enrollment frequencies; defaults to the model's training frequencies.",
)
parser.add_argument(
    "--test-frequency",
    type=float,
    choices=SUPPORTED_FREQUENCIES,
    default=12.0,
    help="Held-out authentication frequency.",
)
parser.add_argument(
    "--enrollment-session",
    type=int,
    default=5,
    choices=range(7),
    help="Session used to create identity templates.",
)
parser.add_argument(
    "--calibration-session",
    type=int,
    choices=range(7),
    help="Separate session used to select an operating threshold.",
)
parser.add_argument(
    "--test-session",
    type=int,
    default=6,
    choices=range(7),
    help="Held-out session used for final evaluation.",
)
args = parser.parse_args()
if args.enrollment_session == args.test_session:
    parser.error("Enrollment and test sessions must be distinct.")
if args.calibration_session in (
    args.enrollment_session,
    args.test_session,
):
    parser.error("Calibration, enrollment, and test sessions must be distinct.")


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
trained_frequencies = tuple(
    checkpoint.get(
        "training_frequencies",
        DEFAULT_TRAINED_FREQUENCIES,
    )
)
trained_sessions = tuple(
    checkpoint.get("training_sessions", range(5))
)
enrollment_frequencies = tuple(
    args.enrollment_frequencies or trained_frequencies
)
test_frequency = args.test_frequency
if test_frequency in trained_frequencies:
    raise ValueError(
        f"Test frequency {test_frequency:g} Hz was included in the model's "
        "training frequencies; choose a held-out frequency."
    )
held_out_sessions = {
    args.enrollment_session,
    args.test_session,
}
if args.calibration_session is not None:
    held_out_sessions.add(args.calibration_session)
session_overlap = held_out_sessions.intersection(trained_sessions)
if session_overlap:
    raise ValueError(
        "Enrollment, calibration, and test sessions must be excluded from "
        f"training; overlap: {sorted(session_overlap)}."
    )

model = model.to(DEVICE)
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# Load cached metadata
# ---------------------------------------------------------

metadata = []
for metadata_path in (
    CACHE_ROOT / "metadata.json",
    TEST_METADATA_FILE,
):
    if metadata_path.is_file():
        with open(metadata_path, "r") as f:
            metadata.extend(json.load(f))


# Enrollment uses only frequencies the model saw during training.

enrollment_items = [
    item
    for item in metadata
    if item["session"] == args.enrollment_session
    and float(item["frequency"]) in enrollment_frequencies
]


# Test uses only the requested held-out frequency.

test_items = [
    item
    for item in metadata
    if item["session"] == args.test_session
    and float(item["frequency"])
    == test_frequency
]

calibration_items = []
if args.calibration_session is not None:
    calibration_items = [
        item
        for item in metadata
        if item["session"] == args.calibration_session
        and float(item["frequency"]) in enrollment_frequencies
    ]

if not enrollment_items:
    raise ValueError("No enrollment trials match the requested session/frequencies.")
if not test_items:
    raise ValueError("No test trials match the requested session/frequency.")
if args.calibration_session is not None and not calibration_items:
    raise ValueError(
        "No calibration trials match the requested session/frequencies."
    )
for split_name, items in (
    ("enrollment", enrollment_items),
    ("test", test_items),
    ("calibration", calibration_items),
):
    if not items:
        continue
    split_subjects = {int(item["subject"]) for item in items}
    if split_subjects != set(range(1, NUM_SUBJECTS + 1)):
        raise ValueError(
            f"{split_name.capitalize()} data must include all "
            f"{NUM_SUBJECTS} subjects; found {len(split_subjects)}."
        )

print()
print("Protocol:")
print("Training frequencies:", sorted(trained_frequencies))
print("Enrollment session:", args.enrollment_session)
print("Enrollment frequencies:", sorted(enrollment_frequencies))
if args.calibration_session is not None:
    print("Calibration session:", args.calibration_session)
print("Test session:", args.test_session)
print("Test frequency:", test_frequency)

print()
print(
    "Enrollment trials:",
    len(enrollment_items)
)

print(
    f"{test_frequency:g} Hz test trials:",
    len(test_items)
)
if args.calibration_session is not None:
    print("Calibration trials:", len(calibration_items))


# ---------------------------------------------------------
# Extract embeddings
# ---------------------------------------------------------

def extract_embeddings(items):

    embeddings = []
    subjects = []

    with torch.no_grad():

        for index, item in enumerate(items):

            eeg = np.load(
                resolve_cache_file(item["file"])
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
print(f"Extracting unseen {test_frequency:g} Hz embeddings...")

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

calibration_embeddings = np.empty((0, EMBEDDING_SIZE))
calibration_subjects = np.empty(0, dtype=int)
if calibration_items:
    print()
    print("Extracting calibration embeddings...")
    calibration_embeddings, calibration_subjects = extract_embeddings(
        calibration_items
    )
    calibration_embeddings = normalize_embeddings(calibration_embeddings)


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
if len(templates) != NUM_SUBJECTS:
    raise ValueError(
        f"Expected templates for {NUM_SUBJECTS} subjects; found {len(templates)}."
    )


def score_embeddings(embeddings, subjects):
    scores = []
    labels = []
    for embedding, true_subject in zip(embeddings, subjects):
        for subject, template in templates.items():
            scores.append(float(np.dot(embedding, template)))
            labels.append(int(subject == int(true_subject)))
    return np.asarray(scores), np.asarray(labels)


print()
print(f"Scoring {test_frequency:g} Hz test trials...")
scores, labels = score_embeddings(test_embeddings, test_subjects)

calibration_threshold = None
if calibration_items:
    calibration_scores, calibration_labels = score_embeddings(
        calibration_embeddings,
        calibration_subjects,
    )
    calibration_fpr, calibration_tpr, calibration_thresholds = roc_curve(
        calibration_labels,
        calibration_scores,
    )
    calibration_fnr = 1.0 - calibration_tpr
    calibration_eer_index = np.argmin(
        np.abs(calibration_fpr - calibration_fnr)
    )
    calibration_eer = (
        calibration_fpr[calibration_eer_index]
        + calibration_fnr[calibration_eer_index]
    ) / 2
    calibration_threshold = calibration_thresholds[
        calibration_eer_index
    ]


# ---------------------------------------------------------
# ROC / EER
# ---------------------------------------------------------

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
genuine_scores = scores[labels == 1]
impostor_scores = scores[labels == 0]


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 60)
print("UNSEEN-FREQUENCY EEG VERIFICATION")
print("=" * 60)

print(
    "Training frequencies:",
    sorted(trained_frequencies)
)
print("Training sessions:", sorted(trained_sessions))

print(
    "Test frequency:",
    test_frequency,
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

if calibration_threshold is not None:
    test_predictions = scores >= calibration_threshold
    false_accepts = int(np.sum((labels == 0) & test_predictions))
    false_rejects = int(np.sum((labels == 1) & ~test_predictions))
    impostor_count = int(np.sum(labels == 0))
    genuine_count = int(np.sum(labels == 1))
    far = false_accepts / impostor_count
    frr = false_rejects / genuine_count
    print()
    print("CALIBRATED OPERATING POINT")
    print("Calibration session:", args.calibration_session)
    print(f"Calibration threshold: {calibration_threshold:.6f}")
    print(f"Calibration EER: {calibration_eer:.6f} ({calibration_eer * 100:.4f}%)")
    print(f"False accepts: {false_accepts}")
    print(f"False rejects: {false_rejects}")
    print(f"FAR: {far:.6f} ({far * 100:.4f}%)")
    print(f"FRR: {frr:.6f} ({frr * 100:.4f}%)")
    print(f"Balanced accuracy: {((1 - far) + (1 - frr)) / 2:.6f}")