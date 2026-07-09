# Configuration

A TOML file lets you pin every measurement option so runs are reproducible.

```toml
[measurement]
dt            = 1.0                    # logging interval in seconds (default: 1.0)
output        = "output/my_measurement" # path prefix for output CSV files (default: output/measurement)
warmup        = 0.0                    # settle time in seconds before measuring (default: 0.0)
queue_maxsize = 0                      # real-time block queue depth (default: 0 = unbounded)
precision     = 1                      # decimal places for reported levels (default: 1 = 0.1 dB)
# signal_conditioning = "xl2"         # optional input filter in front of the weighting (default: none)

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

CLI flags override the loaded values when both are supplied (see
[Getting started](getting-started.md#using-a-toml-config-file)).

## `[measurement]` keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dt` | float | `1.0` | Logging interval in seconds. Each metric is sampled once per `dt` seconds and written as one row in the output CSV. |
| `output` | string | `"output/measurement"` | Output path prefix. Broadband results go to `{output}_report.csv` / `{output}_log.csv`; per-band RTA results go to `{output}_rta_report.csv` / `{output}_rta_log.csv`. In the REPL the directory and name halves are set separately with `output` and `name`. |
| `warmup` | float | `0.0` | Settle time in seconds before measuring. The signal is processed but not logged during warm-up, then the meters are reset so the accumulators ignore the start-up filter transient. Logged timestamps restart at 0 afterwards. |
| `queue_maxsize` | int | `0` | Real-time block queue depth (blocks buffered between the audio driver and the engine). `0` means an unbounded queue (no latency bound, no dropped blocks); a finite value bounds latency but drops blocks — counted as overruns — if the engine falls behind. Only affects live (`--device`/`--generator`) sources. |
| `precision` | int | `1` | Decimal places for reported/displayed levels, in both the console output and the CSV result files. `1` gives 0.1 dB resolution (a Class-1 SLM display); `2` gives 0.01 dB, and so on. Overridden by `--precision`. |
| `signal_conditioning` | string | *(none)* | Band-limiting filter inserted at the head of every bus, in front of the frequency weighting. `"xl2"` applies the preset; a custom `"F_HPF N_HPF F_LPF N_LPF"` spec builds a filter with those cutoffs/orders; omit (or `"none"`) to leave the chain unconditioned. See [Signal conditioning](signal-conditioning.md) for details. |

> **Note on `dt` resolution.** Log rows are written on block edges, so the logging
> cadence is quantized to the block duration (`blocksize/fs`). A `dt` that is an exact
> integer multiple of the block duration logs at the requested interval; otherwise each
> interval is rounded up to the next block edge and the effective `dt` drifts slightly
> above the configured value. (The moving-window length of `_dt` metrics is sample-exact
> regardless; only the instant at which it is sampled and logged is block-snapped.)

## `[metrics]` keys

| Key | Type | Description |
|-----|------|-------------|
| `require` | list of strings | Metrics to compute. Each string uses the [metric name syntax](metric-syntax.md). Unknown or invalid names raise an error at startup. |

Unknown top-level sections or unknown keys within a section raise a `ValueError` (strict
validation).
