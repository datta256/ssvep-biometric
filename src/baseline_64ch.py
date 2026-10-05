import mne
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATASET = Path(r"E:\ssvep-data")

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

    print(f"[{i}/{len(files)}] {subject} {session}")

    raw = mne.io.read_raw_eeglab(
        str(filename),
        preload=True,
        verbose=False,
    )

    # ------------------------------------------------
    # Use a NON-STIMULUS baseline period
    #
    # First trial starts around 10 seconds.
    # We use 3 seconds before it.
    # ------------------------------------------------

    sfreq = raw.info["sfreq"]

    start = int(5 * sfreq)
    stop = int(8 * sfreq)

    data = raw.get_data(
        start=start,
        stop=stop,
    )

    # FFT
    spectrum = np.abs(
        np.fft.rfft(data, axis=1)
    )

    freqs = np.fft.rfftfreq(
        data.shape[1],
        1 / sfreq,
    )

    # Same 8-12 Hz band
    band = (
        (freqs >= 8) &
        (freqs <= 12)
    )

    features = spectrum[:, band]

    # Force exactly 25 bins
    target_bins = 25

    if features.shape[1] < target_bins:
        continue

    features = features[:, :target_bins]

    # Same transform as SSVEP experiment
    features = np.log1p(features)

    # 64 × 25 = 1600
    features = features.flatten()

    if features.shape[0] != 1600:
        continue

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
    print(
        f"Correct: "
        f"{np.sum(predictions == np.array(y_test))}"
        f"/{len(y_test)}"
    )


print()
print("========================================")
print("BASELINE EEG RESULTS")
print("========================================")

for i, accuracy in enumerate(results):
    print(
        f"Session {i}: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

print()

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

print("Random baseline: 0.0100 (1.00%)")