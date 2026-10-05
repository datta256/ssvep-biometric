import mne
import numpy as np
from pathlib import Path

from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


DATASET = Path(r"E:\ssvep-data")


records = []

files = sorted(
    DATASET.glob("sub-*/ses-*/eeg/*_eeg.set")
)

print(f"Found {len(files)} recordings")


# ==================================================
# LOAD DATA
# ==================================================

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

    sfreq = raw.info["sfreq"]

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

    # ------------------------------------------------
    # Extract the 6-second 9 Hz trial
    # ------------------------------------------------

    data = raw.get_data(
        start=int(onset * sfreq),
        stop=int((onset + 6) * sfreq),
    )

    # ------------------------------------------------
    # Band-pass 8-12 Hz
    # ------------------------------------------------

    data = mne.filter.filter_data(
        data,
        sfreq=sfreq,
        l_freq=8,
        h_freq=12,
        verbose=False,
    )

    # ------------------------------------------------
    # Downsample
    #
    # 1000 Hz -> 250 Hz
    # Makes CSP much faster.
    # ------------------------------------------------

    data = mne.filter.resample(
        data,
        down=4,
        verbose=False,
    )

    # Shape:
    # channels × samples
    #
    # CSP expects:
    # trials × channels × samples

    records.append(
        (subject, session, data)
    )


print()
print("==============================")
print(f"Usable recordings: {len(records)}")
print("==============================")


# ==================================================
# LEAVE-ONE-SESSION-OUT
# ==================================================

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

    for subject, session, data in records:

        if session == test_session:

            X_test.append(data)
            y_test.append(subject)

        else:

            X_train.append(data)
            y_train.append(subject)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(f"Training: {X_train.shape}")
    print(f"Testing:  {X_test.shape}")

    if len(X_test) == 0:
        continue

    # ==================================================
    # CSP + SVM
    # ==================================================

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

    print("Training CSP + SVM...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    results.append(accuracy)

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Correct: "
        f"{np.sum(predictions == np.array(y_test))}"
        f"/{len(y_test)}"
    )


# ==================================================
# RESULTS
# ==================================================

print()
print("========================================")
print("CSP + SVM RESULTS")
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


print(
    "Random baseline: "
    "0.0100 (1.00%)"
)