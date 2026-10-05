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

# Only build the held-out sessions
TEST_SESSIONS = {5, 6}


CACHE_ROOT.mkdir(parents=True, exist_ok=True)
files = find_eeg_files("**/*_eeg.set")

print(f"Found {len(files)} EEG recordings")
print("Building sessions 5 and 6 only")
print()

metadata = []
total_trials = 0


for file_index, filepath in enumerate(files, 1):

    parts = filepath.parts

    subject = next(
        (
            p.removeprefix("sub-")
            for p in parts
            if p.startswith("sub-")
        ),
        None
    )

    session = next(
        (
            p.removeprefix("ses-")
            for p in parts
            if p.startswith("ses-")
        ),
        None
    )

    if subject is None or session is None:
        continue

    subject = int(subject)
    session = int(session)

    if session not in TEST_SESSIONS:
        continue

    print(
        f"Subject {subject:03d} | Session {session}"
    )

    raw = mne.io.read_raw_eeglab(
        str(filepath),
        preload=True,
        verbose=False
    )

    raw.pick("eeg")

    raw.filter(
        8.0,
        12.0,
        method="fir",
        verbose=False
    )

    raw.resample(
        TARGET_SFREQ,
        npad="auto",
        verbose=False
    )

    data = raw.get_data()

    sfreq = raw.info["sfreq"]

    output_dir = CACHE_ROOT / f"sub-{subject}" / f"ses-{session}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for annotation_index, annotation in enumerate(
        raw.annotations
    ):

        description = str(
            annotation["description"]
        ).strip()

        if description not in FREQUENCIES:
            continue

        frequency = FREQUENCIES[
            description
        ]

        onset = annotation["onset"]

        start_sample = int(
            round(onset * sfreq)
        )

        end_sample = (
            start_sample
            + int(TRIAL_SECONDS * sfreq)
        )

        if end_sample > data.shape[1]:
            continue

        trial = data[
            :,
            start_sample:end_sample
        ].astype(
            np.float32,
            copy=True
        )

        mean = trial.mean(
            axis=1,
            keepdims=True
        )

        std = trial.std(
            axis=1,
            keepdims=True
        )

        trial = (
            trial - mean
        ) / (
            std + 1e-6
        )

        trial_id = (
            f"sub-{subject:03d}_"
            f"ses-{session}_"
            f"trial-{annotation_index:02d}_"
            f"{frequency:g}hz"
        )

        output_file = output_dir / (trial_id + ".npy")

        np.save(
            output_file,
            trial
        )

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


metadata_file = CACHE_ROOT / "test_metadata.json"

with open(metadata_file, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print()
print("=" * 60)
print("TEST CACHE COMPLETE")
print("=" * 60)
print(
    f"Trials cached: {total_trials}"
)
print(
    f"Metadata: {metadata_file}"
)