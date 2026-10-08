# Cached features

Not versioned (several hundred MB). The main notebook regenerates them on the first run and reuses them afterwards:

| Folder | Content | Section |
|---|---|---|
| `M1/` | 80 MFCC statistics per clip | 9 |
| `M2/` | 88 eGeMAPSv02 functionals per clip | 10 |
| `M3/` | WavLM-base-plus embeddings, 13 layers × 768 | 11 |
| `M4/` | Log-Mel spectrograms, 64 × 301 | 12 |
| `M5/` | Whisper-small encoder embeddings, 13 layers × 768 | 13 |
