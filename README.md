# EEG / SSVEP Biometric Authentication

Research prototype for learning subject-specific representations from
SSVEP EEG and evaluating them for biometric identification and
verification.

The current work uses the **eldBETA NEMAR** dataset and a PyTorch EEG
embedding model. The main research result is cross-session biometric
verification, including a test where **12 Hz was completely excluded
from training** and used only as an unseen authentication frequency.

> **Status:** Research/prototype. The results below are reproducible on
> the current dataset/cache setup, but they are not a claim of
> production-grade biometric security.

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
-   Each SSVEP trial is 6 seconds
-   Raw dataset storage is approximately **37 GB** in the current setup.

The dataset is not included in this repository.

------------------------------------------------------------------------

## 3. Example local paths

The scripts currently use these example paths:

``` text
Project:
C:\path\to\ssvep-biometric

Raw dataset:
E:\ssvep-data

Preprocessed cache:
E:\ssvep-cache

Python virtual environment:
E:\ssvep-venv
```

If your paths are different, change the constants in the Python scripts
before running them.

------------------------------------------------------------------------

## 4. Hardware used

Current development machine:

-   NVIDIA GeForce RTX 3050 Laptop GPU
-   6 GB VRAM
-   CUDA 12.6
-   PyTorch CUDA build: `2.14.1+cu126`

GPU is used for the embedding model. The classical scikit-learn
experiments do not require a GPU.

------------------------------------------------------------------------

# 5. Reproducing the working pipeline

## Step 1 --- Clone/open the project

Open the project in VS Code:

``` powershell
cd C:\path\to\ssvep-biometric
```

Or open the folder directly in VS Code.

------------------------------------------------------------------------

## Step 2 --- Create/use the Python environment

The current environment was created on the E: drive to avoid filling the
C: drive.

Activate it:

``` powershell
E:\ssvep-venv\Scripts\Activate.ps1
```

Check Python:

``` powershell
python --version
```

The working environment used Python 3.12.

If you need to create the environment again:

``` powershell
py -3.12 -m venv E:\ssvep-venv
```

Then:

``` powershell
E:\ssvep-venv\Scripts\Activate.ps1
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

Download the dataset:

``` powershell
nemar-py download nm000130 -t v1.0.3 -o E:\ssvep-data --datatype eeg --downloader python -j 4 --trust-existing --verbose
```

This is a large download. The current dataset storage is approximately
**37 GB**.

Make sure the E: drive has sufficient free space before starting.

------------------------------------------------------------------------

# 7. Expected dataset structure

A typical recording looks like:

``` text
E:\ssvep-data\
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

Metadata is stored at:

``` text
E:\ssvep-cache\test_metadata.json
```

------------------------------------------------------------------------

# 10. Classical baseline experiments

These experiments were used to establish how much identity information
exists before using the neural embedding.

## 10.1 Simple spectral fingerprint

The early experiment compared FFT-based posterior-channel fingerprints.

Posterior channels:

``` text
PO7 PO5 PO3 POz PO4 PO6 PO8 O1 Oz O2
```

The result showed that simple cosine similarity was not sufficient for
reliable identity separation.

Important result:

``` text
Same-person cross-session mean similarity:
0.9486

Different-person cross-session mean similarity:
0.9130
```

The distributions overlapped substantially.

------------------------------------------------------------------------

## 10.2 10-channel FFT + SVM

The 10-channel SSVEP FFT classifier produced:

``` text
S0  27.37%
S1  36.08%
S2  27.00%
S3  34.38%
S4  34.38%
S5  32.65%
S6  32.32%

Mean: 32.03%
Std:   3.27%
```

Random 100-class baseline:

``` text
1%
```

------------------------------------------------------------------------

## 10.3 64-channel FFT + SVM

The all-channel version produced:

``` text
S0  40.00%
S1  51.55%
S2  58.00%
S3  52.08%
S4  61.46%
S5  48.98%
S6  48.48%

Mean: 51.51%
Std:   6.43%
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

------------------------------------------------------------------------

# 12. CSP verification

Run:

``` powershell
python src\verification_csp.py
```

The strict version uses:

``` text
Training:   sessions 0–4
Enrollment: session 5
Test:       session 6
```

There is one binary verifier per subject.

For the final strict test:

``` text
Genuine attempts: 100
Impostor attempts: 9,900

ROC-AUC: 0.999997
EER:     0.0101%
```

At the threshold selected from the enrollment session:

``` text
FAR: 0.0202%
FRR: 1.0%

False accepts: 2
False rejects: 1
```

Important:

The 1% FRR corresponds to only one genuine failure out of 100 genuine
test attempts. The estimate is therefore statistically fragile.

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
```

The trained model is saved as:

``` text
E:\ssvep-cache\eeg_embedding_model.pt
```

------------------------------------------------------------------------

# 14. Original all-frequency embedding experiment

Training:

``` text
Sessions 0–4
All 9 frequencies
4,500 trials
```

Training accuracy progression included:

``` text
Epoch 1:  41.58%
Epoch 2:  92.33%
Epoch 3:  98.29%
Epoch 5:  99.56%
Epoch 10: 99.91%
Epoch 13: 100.00%
Epoch 20: 99.89%
```

Verification:

``` text
Enrollment: session 5, all frequencies
Test:       session 6, all frequencies

Genuine:  900
Impostor: 89,100
```

Result:

``` text
Genuine mean:    0.898924
Genuine median:  0.910341

Impostor mean:   0.018983
Impostor median: 0.016805

ROC-AUC: 0.999996
EER:     0.0971%
```

------------------------------------------------------------------------

# 15. Critical experiment --- unseen 12 Hz

The strongest research question tested so far was:

> Can the model authenticate a person using a frequency that it never
> saw during training?

12 Hz was completely excluded from training.

## Experiment 15.1 --- 9 Hz → 12 Hz

Training:

``` text
9 Hz only
500 trials
```

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
Genuine mean:    0.686335
Genuine median:  0.709720

Impostor mean:   0.037242
Impostor median: 0.035757

ROC-AUC: 0.993822
EER:     3.1162%
```

------------------------------------------------------------------------

## Experiment 15.2 --- 9 + 10 Hz → 12 Hz

Training:

``` text
9 + 10 Hz
1,000 trials
```

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

Result:

``` text
Genuine mean:    0.807423
Genuine median:  0.822234

Impostor mean:   0.017368
Impostor median: 0.011303

ROC-AUC: 0.999833
EER:     0.2374%
```

------------------------------------------------------------------------

## Experiment 15.3 --- 8 + 9 + 10 + 11 Hz → 12 Hz

Training:

``` text
8 + 9 + 10 + 11 Hz
2,000 trials
```

Training result:

``` text
Final accuracy: 100.00%
Final loss:     0.0154
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

Result:

``` text
Genuine mean:    0.857635
Genuine median:  0.876018

Impostor mean:   0.010674
Impostor median: 0.006608

ROC-AUC: 0.999977
EER:     0.0758%
```

------------------------------------------------------------------------

## Experiment 15.4 --- 8--11.5 Hz → 12 Hz

Training:

``` text
8, 8.5, 9, 9.5,
10, 10.5, 11, 11.5 Hz
```

12 Hz was completely excluded.

Training:

``` text
4,000 trials
Final accuracy: 99.60%
Final loss:     0.0235
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

Result:

``` text
Genuine mean:    0.878883
Genuine median:  0.896234

Impostor mean:   0.053280
Impostor median: 0.053048

ROC-AUC: 0.999998
EER:     0.0101%
```

------------------------------------------------------------------------

# 16. Frequency-diversity result

The experiments give this progression:

  Training frequencies     Unseen test       EER
  ---------------------- ------------- ---------
  9 Hz                           12 Hz   3.1162%
  9 + 10 Hz                      12 Hz   0.2374%
  8 + 9 + 10 + 11 Hz             12 Hz   0.0758%
  8--11.5 Hz                     12 Hz   0.0101%

This is currently one of the most interesting observations in the
project.

A reasonable research interpretation is:

> Increasing SSVEP frequency diversity during representation learning is
> associated with improved generalization to an unseen stimulus
> frequency.

Do not phrase this as a proven causal law yet. More held-out frequencies
and repeated protocols are needed.

------------------------------------------------------------------------

# 17. Current conclusion

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

# 18. Recommended live 10-person demo

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

The goal is to determine whether the public-dataset result survives real
hardware and real people.

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

For the unseen-frequency experiment:

``` text
Training:
sessions 0–4
NO 12 Hz

Enrollment:
session 5
8–11.5 Hz (or the specified training-frequency subset)

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

### Do not reuse session 6 to select a threshold

For a strict evaluation:

``` text
Training → sessions 0–4
Threshold/enrollment → session 5
Final test → session 6
```

------------------------------------------------------------------------

# 21. Useful commands --- quick reproduction

Activate environment:

``` powershell
E:\ssvep-venv\Scripts\Activate.ps1
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

Unseen 12-Hz frequency

        ↓

EER down to 0.0101%
with 8–11.5 Hz training
```

The next scientific question is no longer whether the public dataset
contains enough signal.

It is:

> **Does the effect survive when we put the headset on real people?**

If the answer is yes, the next engineering milestone is a small
Bionetta/ZKML proof-of-concept.
