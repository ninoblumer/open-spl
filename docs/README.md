# Documentation

Topic guides for **soundlevelmeter**. Start with [Getting started](getting-started.md); the
project overview and installation live in the [top-level README](../README.md).

| Guide | What it covers |
|-------|----------------|
| [Getting started](getting-started.md) | CLI usage: WAV files, live input, the generator, the interactive REPL, and every flag |
| [Metric name syntax](metric-syntax.md) | How to name quantities (`LAeq`, `LAFmax_dt`, `LZeq:bands:63-8000`, …) |
| [Calibration](calibration.md) | Deriving controller sensitivity from a calibrator tone |
| [Signal conditioning](signal-conditioning.md) | The XL2-style input filter (`--signal-conditioning`) |
| [Configuration](configuration.md) | TOML config schema for reproducible runs |
| [Python API](python-api.md) | High-, mid-, and low-level library entry points |
| [Architecture](architecture.md) | Controller → Engine → Bus → plugins → meters → Reporter |
| [Running the tests](testing.md) | pytest, slow tests, the conformance margin report |
| [Real-time load benchmark](benchmarks.md) | Measuring CPU headroom against the real-time budget |
