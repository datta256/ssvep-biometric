import mne
import numpy as np
from paths import get_data_root

DATA_ROOT = get_data_root()

CHANNELS = [
    "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "O1", "Oz", "O2"
]

FILES = [
    (
        1,
        DATA_ROOT / "sub-1" / "ses-0" / "eeg" / "sub-1_ses-0_task-ssvep_run-0_eeg.set",
        10.304,
    ),
    (
        2,
        DATA_ROOT / "sub-2" / "ses-0" / "eeg" / "sub-2_ses-0_task-ssvep_run-0_eeg.set",
        70.691,
    ),
]

fingerprints = []

for subject, filename, onset in FILES:
    print(f"\nLoading subject {subject}...")

    raw = mne.io.read_raw_eeglab(
        str(filename),
        preload=True,
        verbose=False,
    )

    picks = mne.pick_channels(
        raw.ch_names,
        include=CHANNELS,
    )

    data = raw.get_data(
        picks=picks,
        start=int(onset * 1000),
        stop=int((onset + 6) * 1000),
    )

    freqs = np.fft.rfftfreq(
        data.shape[1],
        1 / raw.info["sfreq"],
    )

    spectrum = np.abs(
        np.fft.rfft(data, axis=1)
    )

    index = np.argmin(
        np.abs(freqs - 9)
    )

    fingerprint = spectrum[:, index]
    fingerprints.append(fingerprint)

    print("9 Hz spatial fingerprint:")

    for channel, value in zip(CHANNELS, fingerprint):
        print(f"  {channel:4s}: {value:.6f}")

# Normalize both fingerprints
a = fingerprints[0] / np.linalg.norm(fingerprints[0])
b = fingerprints[1] / np.linalg.norm(fingerprints[1])

similarity = float(np.dot(a, b))

print("\n==============================")
print(f"Cosine similarity: {similarity:.4f}")
print("==============================")