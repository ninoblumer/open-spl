# soundlevelmeter

An IEC 61672-1 compliant Sound Level Meter (SLM) in Python. Measures LAeq, LCeq, LZeq,
LASmax, LAFmax, octave-band levels (1/1, 1/3, 1/6, …), sound exposure levels (LE), and more —
from WAV files or a live microphone.

---

## Installation

```bash
git clone https://github.com/ninoblumer/python-soundlevelmeter
cd python-soundlevelmeter
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

To use `slm` as a shell command instead, you can install this tool directly. For example with `pip install --user git+https://github.com/ninoblumer/python-soundlevelmeter.git@main`


---

## Usage

### One-shot measurement from a WAV file

```bash
python -m slm --file recording.wav --fs-db 128.1 --measure LAeq LAFmax LCeq LZeq:bands:63-8000
```

Results are written to `output/measurement_report.csv` (broadband) and
`output/measurement_rta_report.csv` (per-band). Use `--output` to change the path prefix and
`--dt` to set the logging interval (default 1 s).

Add `--realtime` to pace the file playback at actual recording speed — useful when you want
per-`dt` results to arrive at wall-clock intervals rather than all at once:

```bash
python -m slm --file recording.wav --fs-db 128.1 --realtime --measure LAeq LAF --dt 1.0
```

**Sensitivity flags** (mutually exclusive):

| Flag | Argument | Use when… |
|------|----------|-----------|
| `--fs-db DB` | dB SPL at 0 dBFS | WAV recorded by a hardware SLM with an FS annotation |
| `--sensitivity-mv MV` | mV/Pa | You know the microphone sensitivity from its datasheet |
| `--sensitivity-dbv DBV` | dBV re 1 V/Pa | Manufacturer lists sensitivity as e.g. `−34 dBV` |

### Using a TOML config file

```bash
python -m slm --file recording.wav --config config.toml --fs-db 128.1
```

CLI flags (`--measure`, `--output`, `--dt`) override the values in the config file when both
are supplied. See the [TOML configuration](#toml-configuration) section for the full schema.

### Interactive REPL

```bash
python -m slm
```

The REPL lets you load files, set sensitivity, add metrics, and start/stop measurements
interactively. Type `help` for a list of commands.

Use `-i` to pre-populate the shell with a file, sensitivity, and/or config before the prompt
opens — handy when you have a fixed setup but still want to adjust before running:

```bash
python -m slm -i --file recording.wav --fs-db 128.1 --config config.toml
```

Key REPL commands: `file`, `device`, `sensitivity`, `calibrate`, `add`, `remove`, `dt`,
`output`, `realtime`, `display`, `show`, `tree`, `save`, `load`, `start`.

### Real-time input (requires sounddevice)

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

### From a WAV recording

Derive the controller sensitivity from a recording of a known-level calibrator tone:

```bash
python -m slm --calibrate --file cal.wav --cal-level 94.0 --cal-freq 1000.0
```

A 1/3-octave bandpass filter is applied around `--cal-freq` before the RMS is computed, so
harmonics and background noise do not corrupt the estimate.

### From a live microphone (requires sounddevice)

Hold the calibrator to the microphone and run calibration in real time. The command listens
until the bandpass-filtered level has converged (rolling std-dev < 0.1 dB over 10 readings)
and then stops automatically:

```bash
python -m slm --calibrate --device 0 --cal-level 94.0 --cal-freq 1000.0
```

### Controller sensitivity vs microphone sensitivity

The value returned by `--calibrate` is the **controller sensitivity** (V/Pa) — the factor
that converts raw WAV float samples into Pascal. It is **not** the same as the physical
microphone sensitivity on a datasheet.

For WAV files recorded by a calibrated hardware SLM (e.g. NTi XL2), the fullscale level (dBFS) encodes the entire recording chain. The controller sensitivity
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

- **`Engine`** — main processing loop; owns buses; emits an `on_record(timestamp, dt)` tick each block to a sink callback (it holds no reference to the reporter)
- **`Bus`** — one frequency weighting + a chain of downstream plugins and meters
- **`PluginAWeighting` / `PluginCWeighting` / `PluginZWeighting`** — IIR frequency-weighting filters
- **`PluginFastTimeWeighting` / `PluginSlowTimeWeighting`** — exponential time-weighting filters
- **`PluginOctaveBand`** — arbitrary N/M-octave filter bank; outputs N channels
- **`LeqAccumulator` / `MaxAccumulator`** — whole-file/stream integrating meters
- **`LeqMovingMeter` / `MaxMovingMeter`** — sliding-window meters
- **`Reporter`** — a sink: columns registered via `add_columns` are sampled when `engine.on_record` fires, then written as CSV

`assemble_engine()` in `slm/assembly.py` is the factory that builds an engine and wires the
above components from a list of metric specs, returning `(engine, bindings)`; the caller
attaches a sink with `reporter.add_columns(bindings)` and `engine.on_record = reporter.record`.
Internally each metric is lowered `parse_metric → MetricSpec → plan_chain → ChainPlan`, and
`build_chain` walks that IR, reusing shared buses and plugins.

---

## Python API

### High-level

```python
from slm.app import (
    run_measurement, run_realtime_measurement,
    calibrate_from_file, calibrate_from_device,
    SLMConfig, sensitivity_from_fs_db,
)

# Sensitivity from WAV full-scale annotation
sens = sensitivity_from_fs_db(128.1)

# — or — derive from a calibrator recording
sens = calibrate_from_file("cal.wav", cal_level=94.0, cal_freq=1000.0)

# — or — derive from a live microphone (blocks until level converges)
sens = calibrate_from_device(device=0, cal_level=94.0, cal_freq=1000.0)

config = SLMConfig.from_toml("config.toml")

# Process a WAV file
run_measurement("recording.wav", sens, config, print_to_console=True)

# Process a WAV file at real-time pace
run_measurement("recording.wav", sens, config, realtime=True, print_to_console=True)

# Live measurement from a microphone (runs until Ctrl+C)
run_realtime_measurement(sens, config, device=0, samplerate=48_000, print_to_console=True)
```

### Mid-level (declarative)

```python
from slm import assemble_engine, parse_metric
from slm.io import FileController, Reporter

controller = FileController("recording.wav", blocksize=1024)
controller.set_sensitivity(sens, unit="V")

specs = [parse_metric(m) for m in ["LAeq", "LAFmax", "LZeq:bands:63-8000"]]
engine, bindings = assemble_engine(specs, controller, dt=1.0)

reporter = Reporter()
reporter.add_columns(bindings)        # what to read
engine.on_record = reporter.record    # when to read

engine.run()
reporter.write("output/measurement")
```

A metric name is parsed in two passes: `parse_metric` produces a `MetricSpec`
(the parsed name), and `plan_chain` lowers that into a `ChainPlan` — a structural,
engine-free description of the plugin/meter chain (a tuple of `NodeReq` ending in a
`MeterReq`). `build_chain` and the REPL `tree`/`inspect` commands are both backends
over this shared representation, so you can inspect how a metric will be wired
without constructing an engine:

```python
from slm import parse_metric, plan_chain

plan = plan_chain(parse_metric("LZFmax:bands:63-8000"))
print([n.kind for n in plan.nodes])   # ['freq_weighting', 'band', 'time_weighting']
```

### Low-level (manual)

```python
from slm import Engine
from slm.io import FileController, Reporter
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

reporter = Reporter()
reporter.add_column("LAeq", laf, "LAeq")
reporter.add_column("LAFmax", laf, "LAFmax")
engine.on_record = reporter.record

engine.run()
reporter.write("output/measurement")
```

---

## TOML configuration

A TOML file lets you pin every measurement option so runs are reproducible.

```toml
[measurement]
dt     = 1.0                      # logging interval in seconds (default: 1.0)
output = "output/my_measurement"  # path prefix for output CSV files (default: output/measurement)

[metrics]
require = [
    "LAeq",
    "LAFmax",
    "LASmax",
    "LCeq",
    "LZeq",
    "LAeq_dt",
    "LAFmax_dt",
    "LZeq:bands:63-8000",
    "LAeq:bands:1/3:31-16000",
]
```

Pass the file with `--config`:

```bash
python -m slm --file recording.wav --config config.toml --fs-db 128.1
```

**`[measurement]` keys**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dt` | float | `1.0` | Logging interval in seconds. Each metric is sampled once per `dt` seconds and written as one row in the output CSV. |
| `output` | string | `"output/measurement"` | Output path prefix. Broadband results go to `{output}_report.csv`; per-band RTA results go to `{output}_rta_report.csv`. |

**`[metrics]` keys**

| Key | Type | Description |
|-----|------|-------------|
| `require` | list of strings | Metrics to compute. Each string uses the metric name syntax described below. Unknown or invalid names raise an error at startup. |

Unknown top-level sections or unknown keys within a section raise a `ValueError` (strict validation).

---

## Running the tests

The test suite lives under `tests/` (unit tests plus IEC 61260-1 / 61672-1 conformance and
XL2 hardware-comparison integration tests). Run it with pytest:

```bash
python -m pytest tests/ -q
```

### Slow tests

A few long-running conformance tests are marked `@pytest.mark.slow` and are **skipped by
default**. Pass `--slow` to include them:

```bash
python -m pytest tests/ -q --slow
```

---

## Real-time load benchmark

The benchmark characterises how much CPU time the SLM engine consumes relative to the
real-time budget (`t_budget = blocksize / samplerate`) across a matrix of loadout × block
size.  No audio hardware is required — a built-in white-noise generator drives the engine.

### Loadouts

| ID | Description |
|----|-------------|
| K0 | Normative minimum: LAF + LAeq |
| K1 | Typical Class-1 handheld: A/C/Z with F- and S-time-weighting, Leq and max |
| K2 | K1 + full IEC 61260-1 1/3-octave filter bank on each weighting, with all broadband metric types (Leq, Fmax, Smax, F, S) per band |

### Processing utilisation (free-running)

Runs each cell at full speed (no real-time pacing) and records ρ = t_proc / t_budget for
every block.  ρ < 1 means the engine is fast enough for real-time use; ρ ≥ 1 means it is
not.

```bash
# Quick run — 10 s of audio per cell (~8 min total)
python tests/benchmark/realtime_benchmark.py --duration 10 --out output/quick

# Full run — 300 s per cell (default)
python tests/benchmark/realtime_benchmark.py --out output/full
```

Options:

```
--samplerate 48000          sample rate in Hz (default: 48000)
--blocksizes 128 256 …      block sizes to test (default: 128 256 512 1024 4096)
--loadouts K0 K1 K2         loadouts to test (default: all three)
--duration 300              audio-time duration per Quantity A cell in seconds
--out results/              output directory (created if absent)
```

### Real-time deadline violations

Runs a specified cell at real-time pace and counts how many blocks missed their deadline
(overruns).  Only meaningful for cells where Quantity A shows ρ near 1.

```bash
python tests/benchmark/realtime_benchmark.py --loadouts K2 --blocksizes 256 512 \
    --rt-duration 60 --run-b K2:256 K2:512 --out output/rt
```

Options:

```
--rt-duration 3600          wall-clock duration per Quantity B cell in seconds
--run-b K2:256 K2:512 …     cells to measure (LOADOUT:BLOCKSIZE tokens)
```

### Generating figures

After running the benchmark, produce publication-quality figures with:

```bash
python scripts/thesis_figures_performance.py --data output/quick
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See `LICENSE` for full details and `NOTICE` for third-party attributions.