import mne
import numpy as np
from pathlib import Path

DATASET = Path(r"E:\ssvep-data")

CHANNELS = [
    "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "O1", "Oz", "O2"
]

fingerprints = {}

print("Finding EEG files...")

files = sorted(
    DATASET.glob("sub-*/ses-0/eeg/*_eeg.set")
)

print(f"Found {len(files)} subjects")

for i, filename in enumerate(files, 1):

    subject = filename.parts[-4]

    print(f"[{i}/{len(files)}] {subject}")

    raw = mne.io.read_raw_eeglab(
        str(filename),
        preload=True,
        verbose=False,
    )

    # Find the 9 Hz trial
    annotations = raw.annotations

    onset = None

    for ann_onset, description in zip(
        annotations.onset,
        annotations.description,
    ):
        if description == "9":
            onset = ann_onset
            break

    if onset is None:
        print("  No 9 Hz trial found")
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

    index = np.argmin(
        np.abs(freqs - 9)
    )

    fingerprint = spectrum[:, index]

    # Normalize so absolute signal amplitude matters less
    fingerprint = fingerprint / np.linalg.norm(fingerprint)

    fingerprints[subject] = fingerprint


subjects = sorted(fingerprints.keys())

print()
print("==============================")
print(f"Usable subjects: {len(subjects)}")
print("==============================")

# Compare every subject against every other subject

similarities = []

for i in range(len(subjects)):

    for j in range(i + 1, len(subjects)):

        a = fingerprints[subjects[i]]
        b = fingerprints[subjects[j]]

        similarity = float(np.dot(a, b))

        similarities.append(similarity)


similarities = np.array(similarities)

print()
print("==============================")
print("PAIRWISE SIMILARITY")
print("==============================")

print(f"Pairs:  {len(similarities)}")
print(f"Mean:   {similarities.mean():.4f}")
print(f"Median: {np.median(similarities):.4f}")
print(f"Min:    {similarities.min():.4f}")
print(f"Max:    {similarities.max():.4f}")
print(f"Std:    {similarities.std():.4f}")

print()
print("Lowest similarities:")

for value in np.sort(similarities)[:10]:
    print(f"  {value:.4f}")

print()
print("Highest similarities:")

for value in np.sort(similarities)[-10:][::-1]:
    print(f"  {value:.4f}")