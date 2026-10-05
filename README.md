# EEG / SSVEP Biometric Authentication

Research prototype for learning subject-specific representations from
SSVEP EEG and evaluating them for biometric identification and
verification.

The current work uses the **eldBETA NEMAR** dataset and a PyTorch EEG
embedding model. The main research result is cross-session biometric
verification, including a test where **12 Hz was completely excluded
from training** and used only as an unseen authentication frequency.

> **Status:** Research prototype. Rerun results below were obtained on
> the local dataset with the stated commands. They are not claims of
> production-grade biometric security or independent scientific
> replication; GPU, library, and seed differences can change results.

------------------------------------------------------------------------

## 1. What this project does

The pipeline is:

``` text
SSVEP EEG
   |
   v
EEG preprocessing
   |
   +--> Classical baselines
   |      +--> FFT + SVM
   |      +--> CSP + SVM
   |
   +--> PyTorch EEG embedding
          |
          v
       128-D embedding
          |
          v
     subject template
          |
          v
    cosine verification
          |
          v
      genuine/impostor
```

The longer-term goal is to use the local EEG inference result as an
authentication factor and eventually prove the inference using ZKML
(Bionetta) before authorizing a wallet/smart-contract action.

------------------------------------------------------------------------

## 2. Dataset

Dataset:

-   **eldBETA**
-   NEMAR dataset: `nm000130`
-   Version: `v1.0.3`
-   100 subjects
-   7 sessions per subject (`ses-0` ... `ses-6`)
-   64 EEG channels
-   1000 Hz sampling rate
-   9 SSVEP target frequencies:
    -   8
    -   8.5
    -   9
    -   9.5
    -   10
    -   10.5
    -   11
    -   11.5
    -   12 Hz
-   Dataset documentation describes each target presentation as a
    10-second trial (4-second cue, 5-second stimulus, 1-second rest).
    Event annotations mark a 6-second target epoch; the cache scripts
    extract six seconds from each frequency annotation onset.
-   Dataset DOI: [10.82901/nemar.nm000130](https://doi.org/10.82901/nemar.nm000130).
    The dataset metadata declares CC BY 4.0.
-   The local copy used for the reruns below contains 700 EEG recordings
    (100 subjects × 7 sessions) and occupies about **17.36 GiB**,
    including accompanying files. Download size depends on the included
    files and dataset version.

The dataset is not included in this repository.

------------------------------------------------------------------------

## 3. Portable data and cache paths

The scripts do not depend on a particular drive or operating system. By
default, they look for the dataset in `data/` and write preprocessed
trials, metadata, and model files to `cache/`, both inside the project
directory. These folders are ignored by Git.

To keep the large dataset and cache on another drive, configure their
locations with environment variables before running scripts. For example,
in PowerShell:

``` powershell
$env:SSVEP_DATA_DIR = "D:\datasets\eldBETA"
$env:SSVEP_CACHE_DIR = "D:\ssvep-cache"
```

On macOS/Linux, use:

``` bash
export SSVEP_DATA_DIR="$HOME/datasets/eldBETA"
export SSVEP_CACHE_DIR="$HOME/ssvep-cache"
```

Both variables are optional. If omitted, the project-local `data/` and
`cache/` directories are used. The variables must point to the dataset
root containing the `sub-*/ses-*/eeg/` directories and to the cache
directory, respectively. Set them again in each new terminal unless you
configure them permanently in your operating system.

For example, to keep using the existing E: drive folders on Windows:

``` powershell
$env:SSVEP_DATA_DIR = "E:\ssvep-data"
$env:SSVEP_CACHE_DIR = "E:\ssvep-cache"
```

------------------------------------------------------------------------

## 4. Hardware used

Current development machine:

-   NVIDIA GeForce RTX 3050 Laptop GPU
-   6 GB VRAM
-   CUDA 12.6
-   PyTorch CUDA build: `2.14.1+cu126`
-   Python 3.12.10
-   NumPy 2.5.2, MNE 1.13.2, scikit-learn 1.9.1

These are the versions used for the reruns documented here. GPU
acceleration is used for the embedding model; the classical experiments
do not require a GPU.

------------------------------------------------------------------------

# 5. Reproducing the working pipeline

## Step 1 --- Clone/open the project

Open the project in VS Code and run commands from the project directory:

``` powershell
cd "C:\path\to\ssvep-biometric"
```

Or open the folder directly in VS Code.

------------------------------------------------------------------------

## Step 2 --- Create/use the Python environment

Create a virtual environment inside the project:

``` powershell
py -3.12 -m venv .venv
```

Activate it and check Python:

``` powershell
.\.venv\Scripts\Activate.ps1
python --version
```

On macOS/Linux, activate it with:

``` bash
source .venv/bin/activate
```

------------------------------------------------------------------------

## Step 3 --- Install Python dependencies

Install the core packages:

``` powershell
python -m pip install --upgrade pip
```

``` powershell
pip install numpy scipy scikit-learn mne
```

Install the CUDA-enabled PyTorch build appropriate for the machine.

The working environment used:

``` text
torch 2.14.1+cu126
torchvision 0.29.1+cu126
torchaudio 2.11.0+cu126
```

Verify CUDA:

``` powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected result should indicate:

``` text
True
NVIDIA GeForce RTX 3050 Laptop GPU
```

------------------------------------------------------------------------

# 6. Download the dataset

The working dataset was downloaded using `nemar-py`.

Install it if necessary:

``` powershell
pip install nemar-py
```

Choose a dataset location. To use the project default:

``` powershell
New-Item -ItemType Directory -Force .\data
nemar-py download nm000130 -t v1.0.3 -o .\data --datatype eeg --downloader python -j 4 --trust-existing --verbose
```

To store it elsewhere, set the variable from the previous section and use
that path instead:

``` powershell
$env:SSVEP_DATA_DIR = "D:\datasets\eldBETA"
nemar-py download nm000130 -t v1.0.3 -o $env:SSVEP_DATA_DIR --datatype eeg --downloader python -j 4 --trust-existing --verbose
```

This is a large download. The local copy used for the reruns below
occupied approximately **17.36 GiB**; allow additional space for
download and preprocessing.

Make sure the selected drive has sufficient free space before starting.

------------------------------------------------------------------------

# 7. Expected dataset structure

A typical recording under the dataset root looks like:

``` text
data\
└── sub-1\
    └── ses-0\
        └── eeg\
            └── sub-1_ses-0_task-ssvep_run-0_eeg.set
```

The project searches for recordings using the subject/session directory
structure.

------------------------------------------------------------------------

# 8. Build the training cache

The raw EEGLAB files are relatively slow to repeatedly load and
preprocess.

The project therefore creates a NumPy cache.

Run:

``` powershell
python src\build_cache.py
```

The cache builder:

1.  Finds the recordings.
2.  Loads EEG.
3.  Picks EEG channels.
4.  Band-passes 8--12 Hz.
5.  Resamples 1000 Hz → 250 Hz.
6.  Extracts the 6-second SSVEP trials.
7.  Performs per-channel z-normalization.
8.  Saves each trial as `.npy`.
9.  Writes metadata.

The current training cache contains:

``` text
Sessions: 0–4
Recordings: 500
Usable trials: 4,500
```

------------------------------------------------------------------------

# 9. Build the held-out test cache

Run:

``` powershell
python src\build_test_cache.py
```

This creates the held-out cache for sessions 5 and 6.

Current result:

``` text
Sessions: 5–6
Usable trials: 1,800
```

Metadata is stored in the configured cache directory (the default is
`cache/`):

``` text
cache\test_metadata.json
```

------------------------------------------------------------------------

# 10. Classical baseline experiments

These experiments were used to establish how much identity information
exists before using the neural embedding.

## 10.1 Simple spectral fingerprint

`src/cross_session.py` extracts the 9 Hz FFT amplitude at the stimulus
frequency from ten posterior channels for each subject/session. It
normalizes each fingerprint and compares all pairs across seven sessions.

Posterior channels:

``` text
PO7 PO5 PO3 POz PO4 PO6 PO8 O1 Oz O2
```

The result showed that simple cosine similarity was not sufficient for
reliable identity separation.

Important result:

``` text
Same-subject session pairs: 2,100
Different-subject pairs:    242,550
Same-person cross-session mean similarity:
0.9486

Different-person cross-session mean similarity:
0.9130
```

The distributions overlap substantially. Pair comparisons share
recordings and are not independent samples.

------------------------------------------------------------------------

## 10.2 10-channel FFT + SVM

`src/cross_validate.py` uses 9 Hz trials and leave-one-session-out
evaluation. The rerun produced:

``` text
Held-out session   Accuracy   Correct / evaluated
ses-0              27.37%     26 / 95
ses-1              36.08%     35 / 97
ses-2              27.00%     27 / 100
ses-3              34.38%     33 / 96
ses-4              34.38%     33 / 96
ses-5              32.65%     32 / 98
ses-6              32.32%     32 / 99

Mean: 32.03%
Std (population): 3.27 percentage points
```

Some folds have fewer than 100 usable trials because not every
recording yielded a usable 9 Hz epoch.

Uniform random 100-class baseline:

``` text
1%
```

------------------------------------------------------------------------

## 10.3 64-channel FFT + SVM

`src/cross_validate_64ch.py` uses all EEG channels under the same
leave-one-session-out protocol. The rerun produced:

``` text
Held-out session   Accuracy   Correct / evaluated
ses-0              40.00%     38 / 95
ses-1              51.55%     50 / 97
ses-2              58.00%     58 / 100
ses-3              52.08%     50 / 96
ses-4              61.46%     59 / 96
ses-5              48.98%     48 / 98
ses-6              48.48%     48 / 99

Mean: 51.51%
Std (population): 6.43 percentage points
```

This showed that spatial information from all 64 channels was
substantially more useful than the small posterior-channel fingerprint.

------------------------------------------------------------------------

# 11. CSP + SVM identification

Run:

``` powershell
python src\csp_classifier.py
```

Configuration:

``` text
Frequency: 9 Hz
Channels: all 64
Band: 8–12 Hz
Resampling: 1000 → 250 Hz
CSP components: 16
CSP regularization: OAS
Classifier: RBF SVM
C: 10
Evaluation: leave-one-session-out
```

Result:

``` text
S0  92.00%
S1  98.00%
S2  96.00%
S3  96.00%
S4  95.00%
S5  96.00%
S6  94.00%

Mean: 95.29%
Std:   1.75%
```

Random baseline:

``` text
1%
```

This was a major milestone, but it should not be interpreted as proof
that the model is free of spatial/session/device artifacts.
MNE emitted a montage warning during this rerun: the inferred head
radius was 11.6 cm (above its expected range). Confirm the EEGLAB channel
coordinate units before relying on analyses that use sensor positions.

------------------------------------------------------------------------

# 12. CSP verification

Run:

``` powershell
python src\verification_csp.py
```

`src/verification_csp.py` uses:

``` text
Training:   sessions 0–4
Enrollment: session 5
Test:       session 6
```

It fits one binary verifier per subject. Session 5 scores determine one
global threshold; session 6 is the held-out test. All nine frequencies
are included by the script. A rerun with the current environment did
not produce verification metrics: after loading and preprocessing the
trials, it failed while fitting the first verifier due to a memory
allocation error (399 MiB requested on top of the large all-frequency
trial matrix). The run reported the following usable-trial counts across
all seven sessions before failing:

``` text
 8.0 Hz: 666    8.5 Hz: 685    9.0 Hz: 683
 9.5 Hz: 687   10.0 Hz: 666   10.5 Hz: 671
11.0 Hz: 688   11.5 Hz: 692   12.0 Hz: 671
```

Consequently, there are no verified CSP authentication AUC, EER, FAR,
or FRR numbers for this rerun. The earlier CSP verification metrics
have been withdrawn pending a successful run. The previous 100 genuine
/ 9,900 impostor count describes one frequency with 100 usable test
trials, not the nine-frequency aggregate.

------------------------------------------------------------------------

# 13. Train the EEG embedding model

The main neural experiment is:

``` powershell
python src\train_eeg_embedding.py
```

The model receives:

``` text
1 × 1 × 64 × 1500
```

which corresponds to:

``` text
64 EEG channels
1500 samples
250 Hz
6 seconds
```

Architecture:

``` text
Conv2D 1 → 16
BatchNorm
ELU

Spatial/depthwise Conv2D
16 → 32
kernel 64 × 1
groups = 16

BatchNorm
ELU
Average Pool
Dropout

Conv2D 32 → 64
kernel 1 × 15

BatchNorm
ELU
Average Pool
Dropout

Adaptive Average Pool

Linear 64 → 128

LayerNorm

Linear 128 → 100 subjects
```

Training:

``` text
Optimizer: AdamW
Learning rate: 1e-3
Batch size: 32
Epochs: 20
Device: CUDA
Default frequencies: 8, 9, 10, 11 Hz
Default random seed: 42
```

Choose a different frequency subset with `--frequencies`; the selected
frequencies and seed are saved in the model checkpoint and used by the
verifier to define enrollment.

The trained model is saved in the configured cache directory (the
default is `cache/`):

``` text
cache\eeg_embedding_model.pt
```

------------------------------------------------------------------------

# 14. All-frequency training option

The current supported biometric evaluation is the unseen-frequency
protocol in section 15; it does not evaluate verification across all
frequencies. To train the classifier on all nine frequencies (a
training diagnostic, not an all-frequency biometric verification
experiment), run:

``` powershell
python src\train_eeg_embedding.py --frequencies 8 8.5 9 9.5 10 10.5 11 11.5 12
```

The all-nine-frequency option has not been rerun for this audit.
Training accuracy is in-sample and should not be reported as biometric
verification performance.

------------------------------------------------------------------------

# 15. Critical experiment --- unseen 12 Hz

The strongest research question tested so far was:

> Can the model authenticate a person using a frequency that it never
> saw during training?

12 Hz is excluded from training and enrollment. The measurements below
come from fresh runs using seed 42, sessions 0–4 for training, session 5
for enrollment, and session 6 at 12 Hz for testing. Results can vary
across hardware and library versions.

For each experiment, run the matching training command in the subsection,
then run:

``` powershell
python src\verify_embedding.py
```

The verifier enrolls with the model's recorded training frequencies,
unless `--enrollment-frequencies` is explicitly supplied. It reports
ROC-AUC and EER from the session-6 scores; because EER is derived from
the test scores, it is a summary metric, not a threshold selected in
advance for deployment. The script estimates EER as the mean of FPR and
FNR at the observed ROC point where their absolute difference is
smallest; it does not interpolate the crossing.

## Experiment 15.1 --- 9 Hz → 12 Hz

Train on 9 Hz only (500 trials):

``` powershell
python src\train_eeg_embedding.py --frequencies 9 --seed 42
```

Final training accuracy: 100.00% (loss 0.1572).

Enrollment:

``` text
Session 5
9 Hz
100 trials
```

Test:

``` text
Session 6
12 Hz
100 trials
```

Result:

``` text
Genuine mean:    0.657853
Genuine median:  0.670921

Impostor mean:   0.024705
Impostor median: 0.021329

ROC-AUC: 0.984883
EER:     4.9040%
```

------------------------------------------------------------------------

## Experiment 15.2 --- 9 + 10 Hz → 12 Hz

Train on 9 and 10 Hz (1,000 trials):

``` powershell
python src\train_eeg_embedding.py --frequencies 9 10 --seed 42
```

Final training accuracy: 100.00% (loss 0.0616).

Enrollment:

``` text
Session 5
9 + 10 Hz
200 trials
```

Test:

``` text
Session 6
12 Hz
100 trials
```

Result (seeded rerun):

``` text
Genuine mean:    0.801344
Genuine median:  0.811724

Impostor mean:   0.026663
Impostor median: 0.022012

ROC-AUC: 0.999901
EER:     0.1061%
```

------------------------------------------------------------------------

## Experiment 15.3 --- 8 + 9 + 10 + 11 Hz → 12 Hz

Train on 8, 9, 10, and 11 Hz (2,000 trials):

``` powershell
python src\train_eeg_embedding.py --frequencies 8 9 10 11 --seed 42
```

Training result (seed 42):

``` text
Final accuracy: 99.95%
Final loss:     0.0167
```

Enrollment:

``` text
Session 5
8 + 9 + 10 + 11 Hz
400 trials
```

Test:

``` text
Session 6
12 Hz
100 trials
```

Result (seeded rerun):

``` text
Genuine mean:    0.853387
Genuine median:  0.868403

Impostor mean:   0.018408
Impostor median: 0.012677

ROC-AUC: 0.999978
EER:     0.0152%
```

------------------------------------------------------------------------

## Experiment 15.4 --- 8--11.5 Hz → 12 Hz

Train on 8 through 11.5 Hz (4,000 trials):

``` text
8, 8.5, 9, 9.5,
10, 10.5, 11, 11.5 Hz
```

``` powershell
python src\train_eeg_embedding.py --frequencies 8 8.5 9 9.5 10 10.5 11 11.5 --seed 42
```

12 Hz is excluded.

Training result (seed 42):

``` text
Final accuracy: 99.85%
Final loss:     0.0102
```

Enrollment:

``` text
Session 5
8–11.5 Hz
800 trials
```

Test:

``` text
Session 6
12 Hz
100 trials
```

Result (seeded rerun):

``` text
Genuine mean:    0.877862
Genuine median:  0.896381

Impostor mean:   0.052883
Impostor median: 0.051808

ROC-AUC: 0.999990
EER:     0.0303%
```

------------------------------------------------------------------------

# 16. Frequency-diversity result

These are four single-seed experiments on one subject population and
one held-out frequency, with no repeated-seed uncertainty estimates.
They are exploratory results, not evidence that more training
frequencies cause better generalization. Results are not monotonically
improving as training frequencies are added. The impostor scores also
share enrolled templates and test trials, so the pair counts are not
independent observations.

| Training frequencies | Training trials | Final train accuracy | Genuine mean | Impostor mean | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|
| 9 Hz | 500 | 100.00% | 0.657853 | 0.024705 | 0.984883 | 4.9040% |
| 9, 10 Hz | 1,000 | 100.00% | 0.801344 | 0.026663 | 0.999901 | 0.1061% |
| 8, 9, 10, 11 Hz | 2,000 | 99.95% | 0.853387 | 0.018408 | 0.999978 | 0.0152% |
| 8–11.5 Hz, 0.5 Hz steps | 4,000 | 99.85% | 0.877862 | 0.052883 | 0.999990 | 0.0303% |

Each verification has 100 genuine and 9,900 impostor scores. Values are
from one seeded run (seed 42); the EER is computed on session 6 and is
not a preselected deployment threshold. This observed sequence is not
monotonic, and the small differences should not be treated as a causal
frequency-diversity effect.

------------------------------------------------------------------------

# 17. Session- and frequency-held-out biometric benchmark

This stricter benchmark separates model fitting, enrollment, threshold
selection, and final evaluation by session:

``` text
Train:       sessions 0–3, frequencies 8–11.5 Hz (3,200 trials)
Enroll:      session 4, frequencies 8–11.5 Hz (800 trials)
Calibrate:   session 5, frequencies 8–11.5 Hz (800 trials)
Final test:  session 6, 12 Hz (100 trials)
```

All 100 subjects were present in each split. The model used seed 42 and
the same 20-epoch training configuration described above. Session 6 and
12 Hz were excluded from both model training and threshold calibration.
The session-5 threshold was selected at the observed ROC point where
calibration FPR and FNR were closest, then applied unchanged to session 6.

### Run the benchmark from a fresh clone

Run these commands in PowerShell from the project root. If the repository
is not on the computer yet:

``` powershell
git clone https://github.com/datta256/ssvep-biometric.git
Set-Location ssvep-biometric
```

Create and activate a Python environment, then install the packages:

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn mne
```

Install PyTorch for your hardware using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
The training script uses CUDA when available and otherwise runs on CPU.

Download the eldBETA `nm000130` v1.0.3 dataset as described in section 6.
Point the environment variables at the dataset root and a writable cache
directory. The cache directory should be new or contain the cache
metadata generated for this dataset:

``` powershell
$env:SSVEP_DATA_DIR = "D:\datasets\eldBETA"
$env:SSVEP_CACHE_DIR = "D:\ssvep-benchmark-cache"
```

Build the training and held-out-session caches:

``` powershell
python src\build_cache.py
python src\build_test_cache.py
```

Train and then evaluate using the separate enrollment, calibration, and
test sessions:

``` powershell
python src\train_eeg_embedding.py `
  --frequencies 8 8.5 9 9.5 10 10.5 11 11.5 `
  --sessions 0 1 2 3 `
  --seed 42

python src\verify_embedding.py `
  --enrollment-session 4 `
  --calibration-session 5 `
  --test-session 6 `
  --test-frequency 12
```

The training command writes `eeg_embedding_model.pt` into the configured
cache, and the verifier reads that model and the two metadata files from
the same cache. Training overwrites a model already at that location; use
a separate cache directory if you want to preserve an existing model.
Generated dataset files, caches, and model checkpoints are not needed to
push the source code.

To run the already-prepared benchmark without rebuilding its caches,
activate the environment, set both environment variables, and run only
the training and evaluation commands above.

Measured result on the local dataset and hardware:

``` text
Calibration threshold: 0.551420
Calibration EER:       0.8725%

Final test attempts:   100 genuine, 9,900 impostor
False accepts:         57 / 9,900
False rejects:         1 / 100
Test FAR:              0.5758%
Test FRR:              1.0000%
Balanced accuracy:     99.2121%

Test ROC-AUC:          0.998454
Test EER:              0.5202% (descriptive; threshold is test-derived)
```

The decision operating point is the **calibration threshold**, not the
threshold that minimizes the test-set EER. The test EER and ROC-AUC are
included as secondary, test-derived summaries and should not be used to
choose a deployment threshold. FAR and FRR use different denominators
(impostor and genuine attempts respectively); balanced accuracy is the
mean of the corresponding rejection and acceptance rates.

This is a single-seed, within-dataset benchmark on one subject
population, not evidence of cross-device or real-world performance.
The 9,900 impostor comparisons share test trials and enrollment
templates and therefore are not 9,900 independent observations. The
high observed rates are not a security guarantee; repeated seeds,
independent cohorts, and live EEG evaluation remain necessary.

------------------------------------------------------------------------

# 18. Current conclusion

The experiments are strong enough to justify moving from:

``` text
"Can this work at all?"
```

to:

``` text
"Can this work on real people with a real EEG headset?"
```

The current results are **not** sufficient to claim:

-   a production biometric error rate,
-   cross-device robustness,
-   replay resistance,
-   adversarial robustness,
-   wallet security,
-   or that the current neural network is directly suitable for ZK
    proving.

The next milestone is therefore a **10-person live EEG experiment**.

------------------------------------------------------------------------

# 19. Recommended live 10-person demo

Initial target:

``` text
10 participants
10 genuine attempts/person
```

Measure:

``` text
Genuine acceptance rate
Impostor rejection rate
Cross-session performance
Authentication latency
```

Do not train a fresh model on every test recording.

A basic protocol should be:

``` text
DAY / SESSION 1

Person A → enrollment
Person B → enrollment
...
Person J → enrollment

Then stop.
```

Later:

``` text
SESSION 2

Person A → authenticate
Person B → authenticate
...
Person J → authenticate
```

Also perform wrong-identity attempts:

``` text
Person A → claim B
Person B → claim A
...
```

------------------------------------------------------------------------

------------------------------------------------------------------------

# 19. Bionetta / ZKML next step

Once the live EEG model works, the intended architecture is:

``` text
EEG headset
     |
     v
Local raw EEG
     |
     v
Preprocessing
     |
     v
EEG model
     |
     v
Authentication decision
     |
     v
Bionetta ZK proof
     |
     v
Verifier
     |
     v
Smart contract / wallet
```

Raw EEG should remain local.

The blockchain should receive a proof/verification result, not the raw
EEG.

The current PyTorch architecture should **not** be assumed to be
directly ZK-friendly. A smaller/quantized ZK-friendly model may be
needed.

For the first ZK prototype, benchmark:

``` text
Proof generation time
Proof size
Proof verification time
On-chain verification gas
```

------------------------------------------------------------------------

# 20. Important reproducibility warnings

### Do not accidentally change the train/test protocol

For the earlier unseen-frequency experiments in section 15:

``` text
Training:
sessions 0–4
NO 12 Hz

Enrollment:
session 5
same frequencies as training

Test:
session 6
12 Hz ONLY
```

If 12 Hz enters training, it is no longer an unseen-frequency
experiment.

### Do not report training accuracy as biometric performance

A model reaching 100% training accuracy only means it can fit the
training subjects/trials.

The important metrics are held-out genuine/impostor verification
results.

For the stricter benchmark in section 17, use:

``` text
Training → sessions 0–3, 8–11.5 Hz
Enrollment → session 4, 8–11.5 Hz
Calibration → session 5, 8–11.5 Hz
Final test → session 6, 12 Hz
```

Do not use session 6 scores to select the operating threshold.

------------------------------------------------------------------------

# 21. Useful commands --- quick reproduction

Activate environment:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Build training cache:

``` powershell
python src\build_cache.py
```

Build test cache:

``` powershell
python src\build_test_cache.py
```

Run CSP identification:

``` powershell
python src\csp_classifier.py
```

Run CSP verification:

``` powershell
python src\verification_csp.py
```

Train embedding:

``` powershell
python src\train_eeg_embedding.py
```

Verify embedding:

``` powershell
python src\verify_embedding.py
```

------------------------------------------------------------------------

# 22. Current repository scripts

The project currently contains scripts including:

``` text
src/
├── all_subjects.py
├── baseline_64ch.py
├── build_cache.py
├── build_test_cache.py
├── classifier_64ch.py
├── classifier.py
├── compare.py
├── cross_session.py
├── cross_validate.py
├── cross_validate_64ch.py
├── csp_classifier.py
├── paths.py
├── train_eeg_embedding.py
├── verification_csp.py
└── verify_embedding.py
```

Not every script above is part of the final benchmark table in this
document. Some were exploratory/iteration scripts. When reproducing a
reported result, use the exact protocol and script described in the
corresponding experiment section.

------------------------------------------------------------------------

# 23. Bottom line

The current work has established a strong enough preliminary result to
justify a real-world prototype:

``` text
100 subjects
7 sessions
64-channel EEG
9 SSVEP frequencies

        ↓

Learned EEG embedding

        ↓

Cross-session verification

        ↓

Session-6 verification at an unseen 12-Hz frequency

        ↓

Report the measured rerun result above; do not interpret one run
as a production biometric error rate.
```

The next scientific question is no longer whether the public dataset
contains enough signal.

It is:

> **Does the effect survive when we put the headset on real people?**

If the answer is yes, the next engineering milestone is a small
Bionetta/ZKML proof-of-concept.

------------------------------------------------------------------------

# 24. Commit and push source changes to GitHub

After making and checking source/documentation changes, review what will
be pushed:

``` powershell
git status --short
git diff --check
git diff
```

Stage only the intended source and documentation files; do not stage
raw EEG recordings, generated cache files, model checkpoints, credentials,
or other private data. For example, to stage the benchmark changes:

``` powershell
git add README.md src\train_eeg_embedding.py src\verify_embedding.py
git diff --cached
git commit -m "Document and add strict biometric benchmark"
git push origin main
```

If Git reports that there is no `origin` remote, add the repository
remote once and then push:

``` powershell
git remote add origin https://github.com/datta256/ssvep-biometric.git
git push -u origin main
```

GitHub must authorize the account performing the push. Use GitHub CLI
(`gh auth login`) or Git Credential Manager when prompted; never place
an access token or password in the README, a command, or a source file.
If the current branch is not `main`, push its actual branch name instead.
