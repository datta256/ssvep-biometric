import mne
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET = Path(r"E:\ssvep-data")

X_train = []
y_train = []

X_test = []
y_test = []

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

    # Use ALL EEG channels
    data = raw.get_data(
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

    # 8-12 Hz
    band = (
        (freqs >= 8) &
        (freqs <= 12)
    )

    features = spectrum[:, band]

    # Make feature size identical
    target_bins = 25

    if features.shape[1] < target_bins:
        continue

    features = features[:, :target_bins]

    # Log transform
    features = np.log1p(features)

    # 64 channels × 25 frequencies
    features = features.flatten()

    if session == "ses-6":
        X_test.append(features)
        y_test.append(subject)
    else:
        X_train.append(features)
        y_train.append(subject)


X_train = np.array(X_train)
X_test = np.array(X_test)

print()
print("==============================")
print("DATA")
print("==============================")

print("Training:", X_train.shape)
print("Testing: ", X_test.shape)

print()
print("==============================")
print("TRAINING SVM")
print("==============================")

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

print()
print("==============================")
print("RESULT")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Random baseline: {1 / 100:.4f}")

print(
    f"Correct: "
    f"{np.sum(predictions == np.array(y_test))}"
    f"/{len(y_test)}"
)