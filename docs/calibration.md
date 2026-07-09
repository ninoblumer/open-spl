# Calibration

Calibration derives the **controller sensitivity** (V/Pa) — the factor that converts raw WAV
float samples into Pascal — from a recording of a known-level calibrator tone.

## From a WAV recording

```bash
python -m slm --calibrate --file cal.wav --cal-level 94.0 --cal-freq 1000.0
```

A 1/3-octave bandpass filter is applied around `--cal-freq` (default 1000.0 Hz) before the
RMS is computed, so harmonics and background noise do not corrupt the estimate. `--cal-level`
defaults to 94.0 dB SPL.

## From a live microphone (requires sounddevice)

Hold the calibrator to the microphone and run calibration in real time. The command listens
until the bandpass-filtered level has converged (rolling std-dev < 0.1 dB over 10 half-second
readings) and then stops automatically:

```bash
python -m slm --calibrate --device 0 --cal-level 94.0 --cal-freq 1000.0
```

## Controller sensitivity vs microphone sensitivity

The value returned by `--calibrate` is the **controller sensitivity** (V/Pa) — the factor
that converts raw WAV float samples into Pascal. It is **not** the same as the physical
microphone sensitivity on a datasheet.

For WAV files recorded by a calibrated hardware SLM (e.g. NTi XL2), the fullscale level (dBFS)
encodes the entire recording chain. The controller sensitivity collapses it to a single
number:

```
controller_sensitivity = 1 / (P_ref × 10^(FS_dB / 20))
```

Pass this via `--fs-db 128.1` or `--sensitivity-dbv`/`--sensitivity-mv` on the CLI, or
`sensitivity_from_fs_db()` in the [Python API](python-api.md). To derive sensitivity in
Python from a recording, use `calibrate_from_file()` / `calibrate_from_device()`.
