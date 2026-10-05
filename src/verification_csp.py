import mne
import numpy as np
from pathlib import Path

from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score


DATASET = Path(r"E:\ssvep-data")

TRAIN_SESSIONS = {
    "ses-0",
    "ses-1",
    "ses-2",
    "ses-3",
    "ses-4",
}

ENROLL_SESSION = "ses-5"
TEST_SESSION = "ses-6"

# All 9 SSVEP frequencies in eldBETA
FREQUENCIES = {
    "8",
    "8.5",
    "9",
    "9.5",
    "10",
    "10.5",
    "11",
    "11.5",
    "12",
}


# ============================================================
# LOAD ALL SSVEP TRIALS
# ============================================================

records = []

files = sorted(
    DATASET.glob("sub-*/ses-*/eeg/*_eeg.set")
)

print(f"Found {len(files)} recordings")


for i, filename in enumerate(files, 1):

    subject = next(
        p for p in filename.parts
        if p.startswith("sub-")
    )

    session = next(
        p for p in filename.parts
        if p.startswith("ses-")
    )

    print(
        f"[{i}/{len(files)}] "
        f"{subject} {session}"
    )

    raw = mne.io.read_raw_eeglab(
        str(filename),
        preload=True,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    # --------------------------------------------------------
    # Find EVERY SSVEP trial
    # --------------------------------------------------------

    for onset, description in zip(
        raw.annotations.onset,
        raw.annotations.description,
    ):

        if description not in FREQUENCIES:
            continue

        frequency = float(description)

        # ----------------------------------------------------
        # Extract 6-second trial
        # ----------------------------------------------------

        start = int(onset * sfreq)
        stop = int((onset + 6) * sfreq)

        data = raw.get_data(
            start=start,
            stop=stop,
        )

        # Make sure the trial is complete
        if data.shape[1] < int(6 * sfreq):
            continue

        # ----------------------------------------------------
        # Band-pass 8-12 Hz
        # ----------------------------------------------------

        data = mne.filter.filter_data(
            data,
            sfreq=sfreq,
            l_freq=8,
            h_freq=12,
            verbose=False,
        )

        # ----------------------------------------------------
        # Downsample 1000 -> 250 Hz
        # ----------------------------------------------------

        data = mne.filter.resample(
            data,
            down=4,
            verbose=False,
        )

        records.append(
            (
                subject,
                session,
                frequency,
                data,
            )
        )


print()
print("==============================")
print(
    f"Usable trials: {len(records)}"
)
print("==============================")


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("==============================")
print("TRIAL COUNTS BY FREQUENCY")
print("==============================")


for frequency in sorted(
    set(record[2] for record in records)
):

    count = sum(
        1
        for record in records
        if record[2] == frequency
    )

    print(
        f"{frequency:>4.1f} Hz : {count}"
    )


# ============================================================
# TRAIN ONE VERIFIER PER SUBJECT
#
# Sessions 0-4:
#     training only
#
# Session 5:
#     enrollment only
#
# Session 6:
#     completely untouched test
# ============================================================

verifiers = {}


for target_subject_number in range(1, 101):

    target_subject = (
        f"sub-{target_subject_number}"
    )

    print()
    print(
        f"Training verifier for "
        f"{target_subject}"
    )

    X_train = []
    y_train = []

    for (
        subject,
        session,
        frequency,
        data
    ) in records:

        if session not in TRAIN_SESSIONS:
            continue

        X_train.append(data)

        if subject == target_subject:

            # Genuine
            y_train.append(1)

        else:

            # Impostor
            y_train.append(0)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    model = Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=16,
                    reg="oas",
                    log=True,
                    norm_trace=False,
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    verifiers[target_subject] = model


# ============================================================
# ENROLLMENT
#
# Session 5 is NEVER used to train CSP/SVM.
#
# Every available trial in session 5 is used to determine
# the global acceptance threshold.
# ============================================================

print()
print("==============================")
print("ENROLLMENT")
print("==============================")


enrollment_scores = []
enrollment_labels = []


for target_subject_number in range(1, 101):

    target_subject = (
        f"sub-{target_subject_number}"
    )

    model = verifiers[target_subject]

    for (
        subject,
        session,
        frequency,
        data
    ) in records:

        if session != ENROLL_SESSION:
            continue

        score = model.decision_function(
            np.asarray([data])
        )[0]

        if subject == target_subject:
            enrollment_labels.append(1)
        else:
            enrollment_labels.append(0)

        enrollment_scores.append(score)


enrollment_scores = np.asarray(
    enrollment_scores
)

enrollment_labels = np.asarray(
    enrollment_labels
)


print(
    f"Enrollment genuine: "
    f"{np.sum(enrollment_labels == 1)}"
)

print(
    f"Enrollment impostor: "
    f"{np.sum(enrollment_labels == 0)}"
)


# ============================================================
# ENROLLMENT THRESHOLD
# ============================================================

fpr, tpr, thresholds = roc_curve(
    enrollment_labels,
    enrollment_scores,
)

fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fpr - fnr)
)

enrollment_eer = (
    fpr[eer_index] +
    fnr[eer_index]
) / 2

enrollment_threshold = (
    thresholds[eer_index]
)

enrollment_auc = roc_auc_score(
    enrollment_labels,
    enrollment_scores,
)


print()
print(
    f"Enrollment AUC: "
    f"{enrollment_auc:.6f}"
)

print(
    f"Enrollment EER: "
    f"{enrollment_eer:.6f}"
)

print(
    f"Enrollment threshold: "
    f"{enrollment_threshold:.6f}"
)


# ============================================================
# FINAL AUTHENTICATION TEST
#
# Session 6 has never been used before this point.
# ============================================================

print()
print("==============================")
print("AUTHENTICATION TEST")
print("==============================")


test_scores = []
test_labels = []

test_frequencies = []


for target_subject_number in range(1, 101):

    target_subject = (
        f"sub-{target_subject_number}"
    )

    model = verifiers[target_subject]

    for (
        subject,
        session,
        frequency,
        data
    ) in records:

        if session != TEST_SESSION:
            continue

        score = model.decision_function(
            np.asarray([data])
        )[0]

        if subject == target_subject:
            test_labels.append(1)
        else:
            test_labels.append(0)

        test_scores.append(score)
        test_frequencies.append(frequency)


test_scores = np.asarray(test_scores)
test_labels = np.asarray(test_labels)
test_frequencies = np.asarray(test_frequencies)


# ============================================================
# OVERALL TEST ROC / EER
# ============================================================

fpr, tpr, thresholds = roc_curve(
    test_labels,
    test_scores,
)

fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fpr - fnr)
)

test_eer = (
    fpr[eer_index] +
    fnr[eer_index]
) / 2

test_eer_threshold = (
    thresholds[eer_index]
)

test_auc = roc_auc_score(
    test_labels,
    test_scores,
)


# ============================================================
# FAR / FRR USING ENROLLMENT THRESHOLD
# ============================================================

predictions = (
    test_scores >= enrollment_threshold
)


false_accepts = np.sum(
    (test_labels == 0) &
    predictions
)

false_rejects = np.sum(
    (test_labels == 1) &
    (~predictions)
)

impostor_count = np.sum(
    test_labels == 0
)

genuine_count = np.sum(
    test_labels == 1
)

far = (
    false_accepts /
    impostor_count
)

frr = (
    false_rejects /
    genuine_count
)


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

genuine_scores = test_scores[
    test_labels == 1
]

impostor_scores = test_scores[
    test_labels == 0
]


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print("========================================")
print("ALL-TRIAL EEG AUTHENTICATION RESULTS")
print("========================================")

print(
    f"Training sessions: "
    f"{sorted(TRAIN_SESSIONS)}"
)

print(
    f"Enrollment session: "
    f"{ENROLL_SESSION}"
)

print(
    f"Test session: "
    f"{TEST_SESSION}"
)

print()

print(
    f"Genuine attempts: "
    f"{genuine_count}"
)

print(
    f"Impostor attempts: "
    f"{impostor_count}"
)

print()

print(
    f"Genuine mean: "
    f"{np.mean(genuine_scores):.6f}"
)

print(
    f"Genuine median: "
    f"{np.median(genuine_scores):.6f}"
)

print()

print(
    f"Impostor mean: "
    f"{np.mean(impostor_scores):.6f}"
)

print(
    f"Impostor median: "
    f"{np.median(impostor_scores):.6f}"
)

print()

print(
    f"TEST ROC-AUC: "
    f"{test_auc:.6f}"
)

print(
    f"TEST EER: "
    f"{test_eer:.6f}"
)

print(
    f"TEST EER threshold: "
    f"{test_eer_threshold:.6f}"
)

print()

print(
    f"Enrollment threshold: "
    f"{enrollment_threshold:.6f}"
)

print(
    f"FAR @ enrollment threshold: "
    f"{far:.6f}"
)

print(
    f"FRR @ enrollment threshold: "
    f"{frr:.6f}"
)

print()

print(
    f"False accepts: "
    f"{false_accepts}"
)

print(
    f"False rejects: "
    f"{false_rejects}"
)


# ============================================================
# PERFORMANCE BY FREQUENCY
# ============================================================

print()
print("========================================")
print("PER-FREQUENCY RESULTS")
print("========================================")


for frequency in sorted(
    np.unique(test_frequencies)
):

    mask = (
        test_frequencies == frequency
    )

    scores = test_scores[mask]
    labels = test_labels[mask]

    if len(np.unique(labels)) < 2:
        continue

    frequency_auc = roc_auc_score(
        labels,
        scores,
    )

    frequency_fpr, frequency_tpr, frequency_thresholds = (
        roc_curve(
            labels,
            scores,
        )
    )

    frequency_fnr = 1 - frequency_tpr

    frequency_eer_index = np.nanargmin(
        np.abs(
            frequency_fpr -
            frequency_fnr
        )
    )

    frequency_eer = (
        frequency_fpr[
            frequency_eer_index
        ]
        +
        frequency_fnr[
            frequency_eer_index
        ]
    ) / 2

    frequency_predictions = (
        scores >= enrollment_threshold
    )

    frequency_false_accepts = np.sum(
        (labels == 0) &
        frequency_predictions
    )

    frequency_false_rejects = np.sum(
        (labels == 1) &
        (~frequency_predictions)
    )

    frequency_impostors = np.sum(
        labels == 0
    )

    frequency_genuine = np.sum(
        labels == 1
    )

    frequency_far = (
        frequency_false_accepts /
        frequency_impostors
    )

    frequency_frr = (
        frequency_false_rejects /
        frequency_genuine
    )

    print()
    print(
        f"{frequency:.1f} Hz"
    )

    print(
        f"  AUC: "
        f"{frequency_auc:.6f}"
    )

    print(
        f"  EER: "
        f"{frequency_eer:.6f}"
    )

    print(
        f"  FAR: "
        f"{frequency_far:.6f}"
    )

    print(
        f"  FRR: "
        f"{frequency_frr:.6f}"
    )


print()
print("========================================")
print("DONE")
print("========================================")