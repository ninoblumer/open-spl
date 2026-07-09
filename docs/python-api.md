# Python API

The library exposes three layers, from a one-call runner down to manual chain wiring. All
three are importable from the top-level `slm` package or its `slm.app` / `slm.io` subpackages.

## High-level

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

`run_measurement` also accepts an in-memory NumPy array instead of a path — pass the
sample rate explicitly (there is no WAV header) and choose how results come back with
`output=`: `"csv"` (default, writes the CSV files) or `"return"` (skip disk). The report/CSV
decimal places come from `SLMConfig.precision` (default `1`, i.e. 0.1 dB resolution, matching
a Class-1 SLM display). It always returns a [`MeasurementResults`](#measurementresults) object:

```python
import numpy as np
from slm import run_measurement          # also re-exported from slm.app
from slm.app import SLMConfig

fs = 48_000
t = np.arange(fs) / fs
samples = 0.1 * np.sin(2 * np.pi * 1000 * t)     # 1 s, 1 kHz tone (mono)

config = SLMConfig(metrics=["LZeq", "LAeq"], dt=0.1)
results = run_measurement(samples, sens, config, samplerate=fs, output="return")

print(results.report)      # {'LZeq': ..., 'LAeq': ...}  — final values
print(results.log[0])      # {'timestamp': 0.0, 'LZeq': ..., 'LAeq': ...}  — per-dt rows
```

### MeasurementResults

`MeasurementResults` holds:

| Field | Meaning |
| --- | --- |
| `report` | Final broadband value per metric label (last logged row). |
| `log` | Per-`dt` broadband rows; each a dict of `timestamp` (float seconds) + labels. |
| `rta_report` | Final band-split (octave/RTA) array per band-column label. |
| `rta_log` | Per-`dt` band-split rows (`timestamp` + one array per label). |
| `band_frequencies` | Band center-frequency labels (e.g. `"1k"`) per band-column label. |

## Mid-level (declarative)

```python
import numpy as np
from slm import assemble_engine, parse_metric
from slm.io import ArrayController, Reporter

samples = np.asarray(...)             # any mono signal (synthesized or loaded)
controller = ArrayController(samples, samplerate=48_000, blocksize=1024)
controller.set_sensitivity(sens, unit="V")

specs = [parse_metric(m) for m in ["LAeq", "LAFmax", "LZeq:bands:63-8000"]]
engine, bindings = assemble_engine(specs, controller, dt=1.0)

reporter = Reporter()
reporter.add_columns(bindings)        # what to read
engine.on_record = reporter.record    # when to read

engine.run()
results = reporter.results()          # in-memory MeasurementResults
reporter.write("output/measurement")  # — or — write CSV files
```

Swap `ArrayController` for `FileController("recording.wav", blocksize=1024)` to read from a
WAV file instead; every downstream step is identical.

## Low-level (manual)

```python
import numpy as np
from slm import Engine
from slm.io import ArrayController, Reporter
from slm.frequency_weighting import PluginAWeighting
from slm.time_weighting import PluginFastTimeWeighting
from slm.meter import LeqAccumulator, MaxAccumulator

samples = np.asarray(...)             # any mono signal (synthesized or loaded)
controller = ArrayController(samples, samplerate=48_000, blocksize=1024)
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

See [Architecture](architecture.md) for how these components fit together.
