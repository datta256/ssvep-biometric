import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    os.environ.get("SSVEP_DATA_DIR", PROJECT_ROOT / "data")
).expanduser().resolve()
CACHE_ROOT = Path(
    os.environ.get("SSVEP_CACHE_DIR", PROJECT_ROOT / "cache")
).expanduser().resolve()


def get_data_root():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_ROOT}\n"
            "Set SSVEP_DATA_DIR to the eldBETA dataset directory, or place "
            "the dataset in the project's data directory."
        )
    return DATA_ROOT


def find_eeg_files(pattern="sub-*/ses-*/eeg/*_eeg.set"):
    data_root = get_data_root()
    files = sorted(data_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No EEG .set files matching '{pattern}' found in {data_root}. "
            "Check that SSVEP_DATA_DIR points to the dataset root."
        )
    return files


def resolve_cache_file(filename):
    path = Path(filename)
    if path.is_file():
        return path

    candidate = CACHE_ROOT / path
    if candidate.is_file():
        return candidate

    parts = str(filename).replace("\\", "/").split("/")
    subject_index = next(
        (index for index, part in enumerate(parts) if part.startswith("sub-")),
        None,
    )
    if subject_index is not None:
        candidate = CACHE_ROOT.joinpath(*parts[subject_index:])
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Cached trial not found: {filename}. "
        f"Check that SSVEP_CACHE_DIR points to the cache directory: {CACHE_ROOT}"
    )
