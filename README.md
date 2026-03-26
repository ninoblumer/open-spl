# open-spl

An IEC 61672-1 compliant Sound Level Meter (SLM) in Python. Measures LAeq, LCeq, LZeq,
LASmax, LAFmax, octave-band levels (1/1, 1/3, 1/6, …), sound exposure levels (LE), and more —
from WAV files or a live microphone.

---

## Installation

```bash
git clone https://github.com/ninoblumer/open-spl
cd open-spl
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

To use `slm` as a shell command instead, you can install this tool directly. For example with `pip install --user git+https://github.com/ninoblumer/open-spl.git@main`


---

## Usage

### One-shot measurement from a WAV file

```bash
python -m slm --file recording.wav --fs-db 128.1 --measure LAeq LAFmax LCeq LZeq:bands:63-8000
```

Results are written to `output/measurement_report.csv` (broadband) and
`output/measurement_rta_report.csv` (per-band). Use `--output` to change the path prefix and
`--dt` to set the logging interval (default 1 s).

### Using a TOML config file

```bash
python -m slm --file recording.wav --config config.toml --fs-db 128.1
```

```toml
# config.toml
[measurement]
dt     = 1.0
output = "output/my_measurement"

[metrics]
require = [
    "LAeq",
    "LAFmax",
    "LCSmax",
    "LAeq_dt",
    "LAFmax_dt",
    "LZeq:bands:63-8000",
    "LAeq:bands:1/3:31-16000",
]
```

### Interactive REPL

```bash
python -m slm
```

The REPL lets you load files, set sensitivity, add metrics, and start/stop measurements
interactively. Type `help` for a list of commands.

### Real-time input (requires PortAudio)

```bash
python -m slm --list-devices
python -m slm --device 0 --sensitivity-mv 50 --measure LAeq LAFmax --dt 1.0
```

---

## Metric name syntax

```
L[ACZ][FSI?](eq|max|min|E)?[_(dt|Ns|Nm|Nh)][:bands:[N/M:]fmin-fmax]
```

### Frequency weighting

| Letter | Weighting |
|--------|-----------|
| `A`    | A-weighting (IEC 61672-1) |
| `C`    | C-weighting (IEC 61672-1) |
| `Z`    | Z-weighting — flat passthrough (IEC 61672-1 Annex E.5) |

### Time weighting (required for max/min and bare metrics; forbidden for eq/E)

| Letter | Filter |
|--------|--------|
| `F`    | Fast (τ = 0.125 s) |
| `S`    | Slow (τ = 1 s) |
| `I`    | Impulse |

### Measure

| Suffix | Description |
|--------|-------------|
| `eq`   | Energy-equivalent level (Leq) — no time-weighting letter |
| `max`  | Maximum — requires time-weighting letter |
| `min`  | Minimum — requires time-weighting letter |
| `E`    | Sound exposure level (LE) — no time-weighting letter |
| *(none)* | Most-recent time-weighted sample — requires time-weighting letter, no window |

### Window suffix (optional)

| Suffix | Description |
|--------|-------------|
| *(none)* | Accumulating over the whole file/stream |
| `_dt`  | Moving window equal to the engine's logging interval |
| `_Ns`  | Moving N-second window (e.g. `_5s`, `_30s`) |
| `_Nm`  | Moving N-minute window (e.g. `_1m`) |
| `_Nh`  | Moving N-hour window (e.g. `_1h`) |

### Band suffix (optional)

| Suffix | Description |
|--------|-------------|
| `:bands:63-8000` | 1/1-octave bands, 63 Hz to 8 kHz |
| `:bands:1/3:31-16000` | 1/3-octave bands, 31 Hz to 16 kHz |
| `:bands:1/6:63-8000` | 1/6-octave bands, 63 Hz to 8 kHz |
| `:bands:N/M:fmin-fmax` | Any N/M-octave filter bank (M/N bands per octave) |

Omitting the `N/M:` fraction defaults to 1/1-octave.

### Examples

```
LAeq                      # A-weighted Leq, accumulating
LAFmax                    # A-weighted fast-time max, accumulating
LAFmax_dt                 # A-weighted fast-time max, moving (dt window)
LZeq_30s                  # Z-weighted Leq, 30-second moving window
LAF                       # A-weighted fast-time instantaneous sample
LAE                       # A-weighted sound exposure level
LZeq:bands:63-8000        # Z-weighted 1/1-octave Leq, 63–8000 Hz
LAeq:bands:1/3:31-16000   # A-weighted 1/3-octave Leq, 31–16000 Hz
```

---

## Calibration

Derive the controller sensitivity from a recording of a known-level calibrator tone:

```bash
python -m slm --calibrate --file cal.wav --cal-level 94.0 --cal-freq 1000.0
```

A 1/3-octave bandpass filter is applied around `--cal-freq` before the RMS is computed, so
harmonics and background noise do not corrupt the estimate.

### Controller sensitivity vs microphone sensitivity

The value returned by `--calibrate` is the **controller sensitivity** (V/Pa) — the factor
that converts raw WAV float samples into Pascal. It is **not** the same as the physical
microphone sensitivity on a datasheet.

For WAV files recorded by a hardware SLM (e.g. NTi XL2), the FS annotation in the filename
(e.g. `FS128.1dB(PK)`) encodes the entire recording chain. The controller sensitivity
collapses it to a single number:

```
controller_sensitivity = 1 / (P_ref × 10^(FS_dB / 20))
```

Pass this via `--fs-db 128.1` or `--sensitivity-dbv`/`--sensitivity-mv` on the CLI, or
`sensitivity_from_fs_db()` in the Python API.

---

## Architecture

```
Controller (FileController | SounddeviceController)
    │  reads audio blocks
    ▼
Engine
    │  routes blocks to each Bus, samples meters every dt seconds
    ├─► Bus(A) ──► PluginAWeighting ──► [plugins…] ──► [meters…]
    ├─► Bus(C) ──► PluginCWeighting ──► [plugins…] ──► [meters…]
    └─► Bus(Z) ──► PluginZWeighting ──► [plugins…] ──► [meters…]
                                                           │
Reporter ◄─────────────────────────────────────────────────┘
    │  writes CSV output files
```

**Key components:**

- **`Engine`** — main processing loop; owns buses; calls `reporter.record()` every `dt` seconds
- **`Bus`** — one frequency weighting + a chain of downstream plugins and meters
- **`PluginAWeighting` / `PluginCWeighting` / `PluginZWeighting`** — IIR frequency-weighting filters
- **`PluginFastTimeWeighting` / `PluginSlowTimeWeighting`** — exponential time-weighting filters
- **`PluginOctaveBand`** — arbitrary N/M-octave filter bank; outputs N channels
- **`LeqAccumulator` / `MaxAccumulator`** — whole-file/stream integrating meters
- **`LeqMovingMeter` / `MaxMovingMeter`** — sliding-window meters
- **`Reporter`** — collects meter readings and writes CSV output

`build_chain()` in `slm/assembly.py` constructs and wires up the above components from a list
of metric name strings, reusing shared buses and plugins where possible.

---

## Python API

### High-level

```python
from slm.app import run_measurement, calibrate_from_file, SLMConfig, sensitivity_from_fs_db

sens = calibrate_from_file("cal.wav", cal_level=94.0, cal_freq=1000.0)
config = SLMConfig.from_toml("config.toml")
run_measurement("recording.wav", sens, config, print_to_console=True)
```

### Mid-level (declarative)

```python
from slm import Engine, build_chain, parse_metric
from slm.io import FileController

controller = FileController("recording.wav", blocksize=1024)
controller.set_sensitivity(sens, unit="V")

engine = Engine(controller, dt=1.0)

specs = [parse_metric(m) for m in ["LAeq", "LAFmax", "LZeq:bands:63-8000"]]
build_chain(specs, engine)

engine.run()
engine.reporter.write("output/measurement")
```

### Low-level (manual)

```python
from slm import Engine
from slm.io import FileController
from slm.frequency_weighting import PluginAWeighting
from slm.time_weighting import PluginFastTimeWeighting
from slm.meter import LeqAccumulator, MaxAccumulator

controller = FileController("recording.wav", blocksize=1024)
controller.set_sensitivity(sens, unit="V")
engine = Engine(controller, dt=1.0)

bus_a = engine.add_bus("A", PluginAWeighting)
la = bus_a.frequency_weighting
laf = bus_a.add_plugin(PluginFastTimeWeighting(input=la))

laf.create_meter(LeqAccumulator, name="LAeq")
laf.create_meter(MaxAccumulator, name="LAFmax")

engine.reporter.add_column("LAeq", laf, "LAeq")
engine.reporter.add_column("LAFmax", laf, "LAFmax")

engine.run()
engine.reporter.write("output/measurement")
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See `LICENSE` for full details and `NOTICE` for third-party attributions.