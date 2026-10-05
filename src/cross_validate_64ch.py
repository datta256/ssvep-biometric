import mne
import numpy as np
from paths import find_eeg_files

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

CHANNELS = [
    "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "O1", "Oz", "O2"
]

# --------------------------------------------------
# Load all recordings once
# --------------------------------------------------

records = []

files = find_eeg_files()

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

    print(f"[{i}/{len(files)}] {subject} {session}")

    raw = mne.io.read_raw_eeglab(
        str(filename),
        preload=True,
        verbose=False,
    )

    # Find 9 Hz trial
    onset = None

    for ann_onset, description in zip(
        raw.annotations.onset,
        raw.annotations.description,
    ):
        if description == "9":
            onset = ann_onset
            break

    if onset is None:
        continue

    picks = np.arange(len(raw.ch_names))

    data = raw.get_data(
        picks=picks,
        start=int(onset * raw.info["sfreq"]),
        stop=int((onset + 6) * raw.info["sfreq"]),
    )

    # FFT
    spectrum = np.abs(
        np.fft.rfft(data, axis=1)
    )

    freqs = np.fft.rfftfreq(
        data.shape[1],
        1 / raw.info["sfreq"],
    )

    band = (
        (freqs >= 8) &
        (freqs <= 12)
    )

    features = spectrum[:, band]

    # Keep feature size identical
    target_bins = 25

    if features.shape[1] < target_bins:
        continue

    features = features[:, :target_bins]

    # Log transform
    features = np.log1p(features)

    # Flatten:
    # 64 channels × 25 frequencies = 1600 features
    features = features.flatten()

    records.append(
        (subject, session, features)
    )


print()
print("==============================")
print(f"Usable recordings: {len(records)}")
print("==============================")


# --------------------------------------------------
# Leave-one-session-out validation
# --------------------------------------------------

results = []

for test_session_number in range(7):

    test_session = f"ses-{test_session_number}"

    print()
    print("==============================")
    print(f"TEST SESSION: {test_session}")
    print("==============================")

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for subject, session, features in records:

        if session == test_session:
            X_test.append(features)
            y_test.append(subject)

        else:
            X_train.append(features)
            y_train.append(subject)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(f"Training: {X_train.shape}")
    print(f"Testing:  {X_test.shape}")

    if len(X_test) == 0:
        print("No test data. Skipping.")
        continue

    model = SVC(
        kernel="rbf",
        C=10,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    results.append(accuracy)

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {np.sum(predictions == np.array(y_test))}/{len(y_test)}")


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print("========================================")
print("7-FOLD CROSS-SESSION RESULTS")
print("========================================")

for i, accuracy in enumerate(results):
    print(
        f"Session {i}: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

print()
print("----------------------------------------")

if results:
    print(
        f"Mean accuracy: "
        f"{np.mean(results):.4f} "
        f"({np.mean(results) * 100:.2f}%)"
    )

    print(
        f"Std: "
        f"{np.std(results):.4f}"
    )

    print(
        f"Random baseline: "
        f"{1 / 100:.4f} "
        f"(1.00%)"
    )