import mne
import numpy as np
from paths import find_eeg_files

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

CHANNELS = [
    "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "O1", "Oz", "O2"
]

X_train = []
y_train = []

X_test = []
y_test = []

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

    # Find the 9 Hz trial
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

    picks = mne.pick_channels(
        raw.ch_names,
        include=CHANNELS,
    )

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

    # Keep exactly 8-12 Hz
    band = (
        (freqs >= 8) &
        (freqs <= 12)
    )

    features = spectrum[:, band]

    # Force exactly the same number of frequency bins
    target_bins = 25

    if features.shape[1] < target_bins:
        continue

    features = features[:, :target_bins]

    # Log transform
    features = np.log1p(features)

    # Flatten channels × frequencies
    features = features.flatten()

    # Sessions 0-5 = training
    # Session 6 = testing

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

print()
print(
    f"Correct: "
    f"{np.sum(predictions == np.array(y_test))}"
    f"/{len(y_test)}"
)