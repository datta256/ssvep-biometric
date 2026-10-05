import mne
import numpy as np
from pathlib import Path

DATASET = Path(r"E:\ssvep-data")

CHANNELS = [
    "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "O1", "Oz", "O2"
]

fingerprints = {}

files = sorted(
    DATASET.glob("sub-*/ses-*/eeg/*_eeg.set")
)

print(f"Found {len(files)} EEG recordings")

for i, filename in enumerate(files, 1):

    parts = filename.parts

    subject = next(
        p for p in parts
        if p.startswith("sub-")
    )

    session = next(
        p for p in parts
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

    index = np.argmin(
        np.abs(freqs - 9)
    )

    fingerprint = spectrum[:, index]

    # Normalize spatial pattern
    fingerprint = fingerprint / np.linalg.norm(fingerprint)

    fingerprints[(subject, session)] = fingerprint


print()
print("==============================")
print("CALCULATING PAIRS")
print("==============================")

same_person = []
different_person = []

keys = list(fingerprints.keys())

for i in range(len(keys)):

    subject_a, session_a = keys[i]
    a = fingerprints[keys[i]]

    for j in range(i + 1, len(keys)):

        subject_b, session_b = keys[j]
        b = fingerprints[keys[j]]

        similarity = float(np.dot(a, b))

        if subject_a == subject_b:
            same_person.append(similarity)
        else:
            different_person.append(similarity)


same_person = np.array(same_person)
different_person = np.array(different_person)

print()
print("==============================")
print("SAME PERSON")
print("==============================")

print(f"Pairs:   {len(same_person)}")
print(f"Mean:    {same_person.mean():.4f}")
print(f"Median:  {np.median(same_person):.4f}")
print(f"Min:     {same_person.min():.4f}")
print(f"Max:     {same_person.max():.4f}")
print(f"Std:     {same_person.std():.4f}")


print()
print("==============================")
print("DIFFERENT PEOPLE")
print("==============================")

print(f"Pairs:   {len(different_person)}")
print(f"Mean:    {different_person.mean():.4f}")
print(f"Median:  {np.median(different_person):.4f}")
print(f"Min:     {different_person.min():.4f}")
print(f"Max:     {different_person.max():.4f}")
print(f"Std:     {different_person.std():.4f}")


print()
print("==============================")
print("SEPARATION")
print("==============================")

print(
    f"Mean difference: "
    f"{same_person.mean() - different_person.mean():.4f}"
)

print()
print("Same-person lowest:")
for x in np.sort(same_person)[:10]:
    print(f"  {x:.4f}")

print()
print("Different-person highest:")
for x in np.sort(different_person)[-10:][::-1]:
    print(f"  {x:.4f}")