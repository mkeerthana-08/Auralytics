# Dataset Guide — Engine Whisperer

## Recommended Dataset: CWRU Bearing Fault Dataset

**Source:** https://engineering.case.edu/bearingdatacenter/download-data-file

### Download Instructions

1. Go to the CWRU Bearing Data Center link above.
2. Download recordings from three categories:
   - **Normal baseline** → place WAV/MAT files in `data/raw/healthy/`
   - **Inner race fault (0.007" or 0.014")** → place in `data/raw/worn/`
   - **Inner race fault (0.021" or 0.028")** → place in `data/raw/critical/`
3. Aim for **30+ clips per class**, each **3–5 seconds long**.

### File Format

- Accepted: `.wav`, `.mp3`, `.flac`, `.ogg`
- Sample rate: any (will be resampled to 22 050 Hz automatically)
- Channels: any (will be converted to mono automatically)

## Self-Recorded Clips (Supplemental)

Recording your own clips (e.g., a small fan motor) adds recording-condition variety
and helps the model generalise beyond a single academic dataset.

### Tips for Recording

- Record in a quiet room to minimise background noise.
- Keep the microphone 5–15 cm from the motor casing.
- Record at least 5 seconds per clip.
- Label honestly: only place clips in `worn/` or `critical/` if you are confident
  the motor is in that state.

## Using Synthetic Data (No Recording Needed)

If you have no real recordings, the app will automatically generate and train on
synthetic engine signals at startup. Run the training script explicitly with:

```bash
python -m src.training.train_runner --synthetic
```

This produces a model trained on mathematically-generated signals that mimic
the acoustic signatures described in the implementation blueprint.
