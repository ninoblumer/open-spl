# soundlevelmeter

An IEC 61672-1 and IEC 61260-1 compliant sound level meter in Python. Measures LAeq,
LCeq, LZeq, LASmax, LAFmax, octave-band levels (1/1, 1/3, 1/6, …), sound exposure levels (LE),
and more — from WAV files, a numpy array or a live microphone.

## Features

- **Broadband metrics** — A/C/Z frequency weighting; Fast/Slow/Impulse time weighting; Leq,
  max, min, sound exposure level, and moving-window variants.
- **Octave-band analysis (RTA)** — arbitrary N/M-octave filter banks (1/1, 1/3, 1/6, …) per
  IEC 61260-1.
- **Any source** — WAV files, an in-memory NumPy array or a live audio device.
- **Standards-based** — filters and weightings verified against the IEC 61672-1 / 61260-1
  conformance limits (`scripts/conformance_report.py`).
- **Three ways to drive it** — a command-line tool, an interactive REPL, and a Python API.

## Installation

```bash
git clone https://github.com/ninoblumer/python-soundlevelmeter
cd python-soundlevelmeter
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

To use `slm` as a shell command instead, you can install this tool directly. For example with
`pip install --user git+https://github.com/ninoblumer/python-soundlevelmeter.git@main`

## Quickstart

Measure a recorded WAV file:

```bash
# calibrate from a recording of a calibration tone
$ python -m slm --file calibration.wav -- calibrate
Sensitivity: 17.68 mV  |  -35.05 dBV
$ python -m slm --file recording.wav --sensitivity-dbv -35.05 --measure LAeq LAFmax LCeq LZeq:bands:63-8000
```

Broadband results land in `output/measurement_report.csv` (and a per-`dt` `_log.csv`); per-band
results in `output/measurement_rta_report.csv`. See [Getting started](docs/getting-started.md)
for live input, the REPL, sensitivity flags, and every option.

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

| Guide | What it covers |
|-------|----------------|
| [Getting started](docs/getting-started.md) | CLI usage, live input, the generator, the interactive REPL |
| [Metric name syntax](docs/metric-syntax.md) | How to name quantities (`LAeq`, `LAFmax_dt`, `LZeq:bands:63-8000`, …) |
| [Calibration](docs/calibration.md) | Deriving controller sensitivity from a calibrator tone |
| [Signal conditioning](docs/signal-conditioning.md) | The XL2-style input filter (`--signal-conditioning`) |
| [Configuration](docs/configuration.md) | TOML config schema for reproducible runs |
| [Python API](docs/python-api.md) | High-, mid-, and low-level library entry points |
| [Architecture](docs/architecture.md) | Controller → Engine → Bus → plugins → meters → Reporter |
| [Running the tests](docs/testing.md) | pytest, slow tests, the conformance margin report |
| [Real-time load benchmark](docs/benchmarks.md) | Measuring CPU headroom against the real-time budget |

## License

This project is licensed under the **GNU General Public License v3.0**.
See `LICENSE` for full details and `NOTICE` for third-party attributions.
