import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from eeg_embedding_model import EEGEmbeddingNet
from eeg_channels import CHANNEL_NAMES
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
    "--model",
    type=Path,
    default=MODEL_FILE,
    help="Model checkpoint to evaluate.",
)
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
    help="Single held-out authentication frequency (default: 12 Hz).",
)
parser.add_argument(
    "--test-frequencies",
    type=float,
    nargs="+",
    choices=SUPPORTED_FREQUENCIES,
    help="One or more held-out authentication frequencies.",
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
    "--calibration-frequencies",
    type=float,
    nargs="+",
    choices=SUPPORTED_FREQUENCIES,
    help=(
        "Frequencies used for threshold calibration; defaults to enrollment "
        "frequencies."
    ),
)
parser.add_argument(
    "--test-session",
    type=int,
    default=6,
    choices=range(7),
    help="Held-out session used for final evaluation.",
)
args = parser.parse_args()
if args.test_frequency is not None and args.test_frequencies is not None:
    parser.error("Use either --test-frequency or --test-frequencies, not both.")
test_frequencies = tuple(
    sorted(
        set(
            args.test_frequencies
            if args.test_frequencies is not None
            else [args.test_frequency if args.test_frequency is not None else 12.0]
        )
    )
)
if args.enrollment_session == args.test_session:
    parser.error("Enrollment and test sessions must be distinct.")
if args.calibration_session in (
    args.enrollment_session,
    args.test_session,
):
    parser.error("Calibration, enrollment, and test sessions must be distinct.")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Device:", DEVICE)

checkpoint = torch.load(
    args.model,
    map_location=DEVICE
)

channel_indices = tuple(
    checkpoint.get("channel_indices", range(len(CHANNEL_NAMES)))
)
if not channel_indices or any(
    index < 0 or index >= len(CHANNEL_NAMES)
    for index in channel_indices
):
    raise ValueError("Model checkpoint has invalid EEG channel indices.")

model = EEGEmbeddingNet(
    num_subjects=NUM_SUBJECTS,
    embedding_size=EMBEDDING_SIZE,
    num_channels=len(channel_indices),
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
calibration_frequencies = tuple(
    sorted(
        set(args.calibration_frequencies or enrollment_frequencies)
    )
)
trained_test_frequencies = set(test_frequencies).intersection(trained_frequencies)
if trained_test_frequencies:
    raise ValueError(
        "Test frequencies must be excluded from model training; overlap: "
        f"{sorted(trained_test_frequencies)}."
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

test_items_by_frequency = {
    frequency: [
        item
        for item in metadata
        if item["session"] == args.test_session
        and float(item["frequency"]) == frequency
    ]
    for frequency in test_frequencies
}
test_items = [
    item
    for frequency in test_frequencies
    for item in test_items_by_frequency[frequency]
]

calibration_items = []
if args.calibration_session is not None:
    calibration_items = [
        item
        for item in metadata
        if item["session"] == args.calibration_session
        and float(item["frequency"]) in calibration_frequencies
    ]

if not enrollment_items:
    raise ValueError("No enrollment trials match the requested session/frequencies.")
if not test_items:
    raise ValueError("No test trials match the requested session/frequencies.")
for frequency, items in test_items_by_frequency.items():
    if not items:
        raise ValueError(
            f"No test trials found for {frequency:g} Hz in session "
            f"{args.test_session}."
        )
    frequency_subjects = {int(item["subject"]) for item in items}
    if frequency_subjects != set(range(1, NUM_SUBJECTS + 1)):
        raise ValueError(
            f"Test data at {frequency:g} Hz must include all "
            f"{NUM_SUBJECTS} subjects; found {len(frequency_subjects)}."
        )
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
print(
    "Model channels:",
    checkpoint.get(
        "channel_names",
        [CHANNEL_NAMES[index] for index in channel_indices],
    ),
)
print("Enrollment frequencies:", sorted(enrollment_frequencies))
if args.calibration_session is not None:
    print("Calibration session:", args.calibration_session)
    print("Calibration frequencies:", list(calibration_frequencies))
print("Test session:", args.test_session)
print("Test frequencies:", list(test_frequencies))

print()
print(
    "Enrollment trials:",
    len(enrollment_items)
)

for frequency, items in test_items_by_frequency.items():
    print(f"{frequency:g} Hz test trials:", len(items))
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
            eeg = eeg[channel_indices, :]

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


test_data_by_frequency = {}
for frequency, items in test_items_by_frequency.items():
    print()
    print(f"Extracting unseen {frequency:g} Hz embeddings...")
    embeddings, subjects = extract_embeddings(items)
    test_data_by_frequency[frequency] = (embeddings, subjects)


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

for frequency, (embeddings, subjects) in test_data_by_frequency.items():
    test_data_by_frequency[frequency] = (
        normalize_embeddings(embeddings),
        subjects,
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


scores_by_frequency = {}
labels_by_frequency = {}
for frequency, (embeddings, subjects) in test_data_by_frequency.items():
    scores_by_frequency[frequency], labels_by_frequency[frequency] = (
        score_embeddings(embeddings, subjects)
    )
scores = np.concatenate(list(scores_by_frequency.values()))
labels = np.concatenate(list(labels_by_frequency.values()))

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


def calculate_metrics(metric_scores, metric_labels):
    metric_auc = roc_auc_score(metric_labels, metric_scores)
    fpr, tpr, thresholds = roc_curve(metric_labels, metric_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    return {
        "auc": metric_auc,
        "eer": (fpr[eer_index] + fnr[eer_index]) / 2,
        "eer_threshold": thresholds[eer_index],
        "genuine": metric_scores[metric_labels == 1],
        "impostor": metric_scores[metric_labels == 0],
    }


def print_metrics(label, metric_scores, metric_labels):
    metrics = calculate_metrics(metric_scores, metric_labels)
    genuine_scores = metrics["genuine"]
    impostor_scores = metrics["impostor"]
    print()
    print(f"--- {label} ---")
    print(f"Genuine attempts:  {len(genuine_scores)}")
    print(f"Impostor attempts: {len(impostor_scores)}")
    print(f"Genuine mean:      {genuine_scores.mean():.6f}")
    print(f"Genuine median:    {np.median(genuine_scores):.6f}")
    print(f"Impostor mean:     {impostor_scores.mean():.6f}")
    print(f"Impostor median:   {np.median(impostor_scores):.6f}")
    print(f"ROC-AUC:           {metrics['auc']:.6f}")
    print(f"Test EER:          {metrics['eer']:.6f} ({metrics['eer'] * 100:.4f}%)")
    print(f"Test EER threshold:{metrics['eer_threshold']:.6f}")
    if calibration_threshold is not None:
        predictions = metric_scores >= calibration_threshold
        false_accepts = int(np.sum((metric_labels == 0) & predictions))
        false_rejects = int(np.sum((metric_labels == 1) & ~predictions))
        impostor_count = int(np.sum(metric_labels == 0))
        genuine_count = int(np.sum(metric_labels == 1))
        far = false_accepts / impostor_count
        frr = false_rejects / genuine_count
        print(f"Calibrated threshold:{calibration_threshold:.6f}")
        print(f"False accepts:      {false_accepts}")
        print(f"False rejects:      {false_rejects}")
        print(f"Calibrated FAR:     {far:.6f} ({far * 100:.4f}%)")
        print(f"Calibrated FRR:     {frr:.6f} ({frr * 100:.4f}%)")
        print(
            "Balanced accuracy: "
            f"{((1 - far) + (1 - frr)) / 2:.6f}"
        )


print()
print("=" * 60)
print("UNSEEN-FREQUENCY EEG VERIFICATION")
print("=" * 60)
print("Training frequencies:", sorted(trained_frequencies))
print("Training sessions:", sorted(trained_sessions))
print("Test frequencies:", list(test_frequencies))
print("Test session:", args.test_session)
if calibration_threshold is not None:
    print("Calibration session:", args.calibration_session)
    print(f"Calibration threshold: {calibration_threshold:.6f}")
    print(f"Calibration EER: {calibration_eer:.6f} ({calibration_eer * 100:.4f}%)")

for frequency in test_frequencies:
    print_metrics(
        f"{frequency:g} Hz",
        scores_by_frequency[frequency],
        labels_by_frequency[frequency],
    )
print_metrics("POOLED HELD-OUT FREQUENCIES", scores, labels)
print("=" * 60)