# Data

The audio files are **not included** in this repository (about 600 MB).

1. Download the `AudioWAV` folder from the official CREMA-D repository: https://github.com/CheyneyComputerScience/CREMA-D
2. Place it here, so that the files are at `data/AudioWAV/1001_DFA_ANG_XX.wav`, etc. (7,442 files).
   For Google Colab, place `AudioWAV.zip` here instead; the notebook unzips it to the local disk.

Files included:

| File | Content |
|---|---|
| `metadata.csv` | One row per clip: actor, sentence, emotion, intensity, sex, sample rate, channels, duration, split |
| `VideoDemographics.csv` | Actor demographics from CREMA-D (used to stratify the split by sex) |
| `excluded_files.csv` | Defective files excluded from the project (none) |
