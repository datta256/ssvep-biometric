import json
import numpy as np
import mne
from paths import CACHE_ROOT, find_eeg_files

FREQUENCIES = {
    "8": 8.0,
    "8.5": 8.5,
    "9": 9.0,
    "9.5": 9.5,
    "10": 10.0,
    "10.5": 10.5,
    "11": 11.0,
    "11.5": 11.5,
    "12": 12.0,
}

TARGET_SFREQ = 250
TRIAL_SECONDS = 6

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
files = find_eeg_files("**/*_eeg.set")

print(f"Found {len(files)} EEG recordings")
print(f"Cache: {CACHE_ROOT}")
print()

metadata = []
total_trials = 0

for file_index, filepath in enumerate(files, 1):

    # Extract subject/session from path
    parts = filepath.parts

    subject = next(
        (p.removeprefix("sub-") for p in parts if p.startswith("sub-")),
        None
    )

    session = next(
        (p.removeprefix("ses-") for p in parts if p.startswith("ses-")),
        None
    )

    if subject is None or session is None:
        print(f"Skipping: {filepath}")
        continue

    subject = int(subject)
    session = int(session)

    # We only need sessions 0-4 for the embedding training set.
    if session > 4:
        continue

    print(
        f"[{file_index}/{len(files)}] "
        f"Subject {subject:03d} | Session {session}"
    )

    raw = mne.io.read_raw_eeglab(
        str(filepath),
        preload=True,
        verbose=False
    )

    # Keep EEG channels only
    raw.pick("eeg")

    # Bandpass once for the entire recording.
    # This is MUCH faster than filtering every trial separately.
    raw.filter(
        8.0,
        12.0,
        method="fir",
        verbose=False
    )

    # Downsample from 1000 Hz -> 250 Hz
    raw.resample(
        TARGET_SFREQ,
        npad="auto",
        verbose=False
    )

    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    # Create output directory for this subject/session
    output_dir = CACHE_ROOT / f"sub-{subject}" / f"ses-{session}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find every SSVEP trial
    for annotation_index, annotation in enumerate(raw.annotations):

        description = str(annotation["description"]).strip()

        if description not in FREQUENCIES:
            continue

        frequency = FREQUENCIES[description]

        onset = annotation["onset"]

        start_sample = int(round(onset * sfreq))
        end_sample = start_sample + int(TRIAL_SECONDS * sfreq)

        if end_sample > data.shape[1]:
            continue

        trial = data[:, start_sample:end_sample].astype(
            np.float32,
            copy=True
        )

        # Per-channel normalization
        mean = trial.mean(axis=1, keepdims=True)
        std = trial.std(axis=1, keepdims=True)

        trial = (trial - mean) / (std + 1e-6)

        trial_id = (
            f"sub-{subject:03d}_"
            f"ses-{session}_"
            f"trial-{annotation_index:02d}_"
            f"{frequency:g}hz"
        )

        output_file = output_dir / (trial_id + ".npy")

        np.save(output_file, trial)

        metadata.append({
            "file": output_file.relative_to(CACHE_ROOT).as_posix(),
            "subject": subject,
            "session": session,
            "frequency": frequency,
            "shape": list(trial.shape),
        })

        total_trials += 1

    del raw
    del data

    print(f"    Cached trials: {total_trials}")

# Save metadata
metadata_file = CACHE_ROOT / "metadata.json"

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("=" * 60)
print("CACHE COMPLETE")
print("=" * 60)
print(f"Recordings processed: {len(files)}")
print(f"Trials cached:        {total_trials}")
print(f"Metadata:             {metadata_file}")
print(f"Cache directory:      {CACHE_ROOT}")