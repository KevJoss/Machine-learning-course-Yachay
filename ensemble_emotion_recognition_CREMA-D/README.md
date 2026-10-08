# Speech Emotion Recognition on CREMA-D with Ensemble Learning

Group project for the **Machine Learning** course at **Yachay Tech** (Ecuador). Five models with different audio representations classify six emotions from speech, and their probabilities are combined with **Uniform** and **Weighted Soft Voting**. The project analyzes when and why combining models helps: Leave-One-Model-Out, all 26 model combinations, probability correlation and error diversity. The selected ensemble is frozen on Validation and evaluated **once** on Test.

## Team

| Member             | Contribution                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Steve Tene         | Data loading, exploration, speaker-disjoint split, audio representations; **M1** (MFCC + SVM) and **M2** (eGeMAPS + Random Forest)       |
| Felipe Quilumbango | **M3** (WavLM + logistic regression); Soft Voting, weight search, Leave-One-Model-Out, 26 combinations                                   |
| Jhony Peñaherrera  | **M4** (CNN on Log-Mel spectrograms); probability correlation and error diversity                                                        |
| Kevin Sánchez      | **M5** (Whisper + MLP); notebook integration, model comparison, ensemble freezing, Test evaluation, inference on new audio of any length |

## Dataset

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D): 7,442 clips from 91 actors reading 12 sentences with 6 emotions: **ANG** (anger), **DIS** (disgust), **FEA** (fear), **HAP** (happy), **NEU** (neutral), **SAD** (sad).

The split is **speaker-disjoint** (no actor appears in two partitions), stratified by sex, about 70/15/15:

| Partition  | Clips | Actors |
| ---------- | ----- | ------ |
| Train      | 5,152 | 63     |
| Validation | 1,148 | 14     |
| Test       | 1,142 | 14     |

## Methodology

```
TRAIN       → each of the 5 models is trained once
VALIDATION  → hyperparameters, individual comparison, Uniform/Weighted Soft Voting,
              Leave-One-Model-Out, 26 combinations, correlation, error diversity
            → the final ensemble is FROZEN (results/frozen_config.json)
TEST        → single final evaluation
```

- Every scaler and normalization statistic is fitted on Train only.
- All ensemble analyses reuse the stored Validation probabilities: **no model is retrained** per combination.
- Weights are found by grid search on Validation (step 0.05, weights summing to 1; 10,626 options for 5 models), maximizing Macro F1.

## Models (Validation)

| Model | Representation                                    | Classifier           | Macro F1  |
| ----- | ------------------------------------------------- | -------------------- | --------- |
| M1    | MFCC statistics (80 dims)                         | SVM (RBF)            | 0.558     |
| M2    | eGeMAPSv02 functionals (88 dims)                  | Random Forest        | 0.540     |
| M3    | WavLM-base-plus, layer 6 (frozen)                 | Logistic regression  | 0.742     |
| M4    | Log-Mel spectrogram (64 mels, 3 s)                | 2-D CNN from scratch | 0.700     |
| M5    | Whisper-small encoder, mean of 13 layers (frozen) | MLP (256)            | **0.783** |

The representation mattered more than the classifier: pretrained embeddings (Whisper, WavLM) > CNN learned from scratch > hand-crafted acoustic descriptors.

## Ensemble (Validation, Macro F1)

| System                           | Macro F1 |
| -------------------------------- | -------- |
| Uniform Soft Voting (5 models)   | 0.789    |
| Weighted Soft Voting (5 models)  | 0.813    |
| Best 2 models: M3 + M5           | 0.803    |
| Best 3 models: M3 + M4 + M5      | 0.812    |
| Best 4 models: M1 + M3 + M4 + M5 | 0.813    |

**Selected ensemble:** Weighted Soft Voting with **M3 + M5** (weights 0.55 / 0.45). Larger ensembles gained about 1 point on Validation, within the noise of 14 validation actors; the group chose two models for production efficiency (fewer pipelines to maintain, lower inference time).

## Final result (Test, evaluated once)

| System           | Validation | Test      |
| ---------------- | ---------- | --------- |
| M3               | 0.742      | 0.713     |
| M5               | 0.783      | **0.763** |
| Ensemble M3 + M5 | 0.803      | 0.758     |

On Test, the ensemble performs on par with its best member (0.5-point difference, within noise): most of the Validation gain came from **selection optimism**, because the same 14 Validation actors were used for many decisions. The result is reported as is; nothing was changed after looking at Test. Speaker-grouped cross-validation would give more stable selection.

## Inference on new audio of any length

Section 25 of the main notebook classifies any audio file, or every file in `new_audio/`:

| Duration | Strategy                                                                                                                                                                     |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| < 0.25 s | Rejected                                                                                                                                                                     |
| 0.25–5 s | Single pass (flagged as unreliable below 1 s)                                                                                                                                |
| > 5 s    | 3-second windows with 50% overlap; silent windows (30 dB below the loudest) are skipped; the frozen ensemble classifies each window and probabilities are averaged over time |

Long recordings also get a plot of the emotion over time, since a long audio rarely contains a single emotion.

## Repository structure

```
├── notebooks/
│   ├── main_ensemble_activity.ipynb   # final notebook: sections 1–27, fully executed
│   
├── common/utils.py                    # shared contract: class order, metrics, validated I/O
├── splits/splits.csv                  # official speaker-disjoint split (MD5-checked)
├── data/                              # metadata; audio files are NOT included (see data/README.md)
├── features/                          # cached features/embeddings, regenerated by the notebook (not versioned)
├── models/M1 … M5/                    # trained artifacts used for inference
├── probs/val, probs/test/             # per-model probabilities
├── results/                           # all analysis CSVs and figures, frozen_config.json
└── new_audio/                         # put your own audio files here (section 25.5)
```

## How to run

**Locally**

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Download the CREMA-D `AudioWAV` folder into `data/AudioWAV/` (see `data/README.md`).
2. Open `notebooks/main_ensemble_activity.ipynb` and run all cells. The first run extracts and caches the features (Whisper and WavLM embeddings need a GPU to be fast); later runs reuse the cache. Trained models are loaded from `models/`, not retrained.

**Google Colab**: copy the repository to Google Drive, put `AudioWAV.zip` in `data/`, and set `PROJECT` in section 2 to the Drive path. The notebook mounts Drive, unzips the audio to the local disk and installs openSMILE.

## Notes

- `utils.load_splits()` checks the MD5 hash of `splits/splits.csv`. `.gitattributes` prevents line-ending conversion so the hash is preserved.
- Inference with M3 and M5 downloads `microsoft/wavlm-base-plus` and `openai/whisper-small` from Hugging Face.
- CREMA-D: H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova and R. Verma, "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", *IEEE Transactions on Affective Computing*, 2014.
