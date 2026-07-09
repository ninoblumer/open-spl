# Getting started

`slm` runs as a command-line tool (`python -m slm`) with three input sources — a WAV file, a
live audio device, or a built-in white-noise generator — plus an interactive REPL. This page
covers the command-line usage; see the [Python API](python-api.md) to drive it as a library.

## One-shot measurement from a WAV file, recorded with a (different) calibrated sound level meter and the full scale dB is known

```bash
python -m slm --file recording.wav --fs-db 128.1 --measure LAeq LAFmax LCeq LZeq:bands:63-8000 --output output/
```

Broadband results are written to `output/measurement_report.csv` and `output/measurement_log.csv`
(the per-`dt` time series); per-band results to `output/measurement_rta_report.csv` and
`output/measurement_rta_log.csv`. Use `--output` to change the path prefix and `--dt` to set the
logging interval (default 1 s).

The per-`dt` log scrolls to the console as it is measured, and a final **Report** block
(the overall broadband levels, and any per-band metrics) is printed once the run finishes —
the same values written to the `_report.csv` files.

Levels are reported to one decimal place (0.1 dB) by default. Use `--precision N` to change
the number of decimal places in both the console output and the CSV files:

```bash
python -m slm --file recording.wav --fs-db 128.1 --measure LAeq --precision 2
```

Use `--duration` for a fixed-length measurement given as `hh:mm:ss`, `mm:ss`, or just `ss`
(fields may be fractional); the run stops automatically on the next block edge. Without it,
a file is read to its end and live sources (`--device`/`--generator`) run until `Ctrl+C`:

```bash
python -m slm --device 0 --sensitivity-mv 50 --measure LAeq --duration 00:05:00
```

Use `--warmup SECONDS` to settle the processing chain before measuring: the signal is
processed but not logged, then the meters are reset so the accumulators ignore the start-up
filter transient. Logged timestamps restart at 0 after warm-up, and `--duration` measures
the post-warm-up length:

```bash
python -m slm --device 0 --sensitivity-mv 50 --measure LAeq --warmup 5 --duration 00:05:00
```

Add `--realtime` (or `-r`) to pace file playback at actual recording speed — useful when you
want per-`dt` results to arrive at wall-clock intervals rather than all at once. It requires
`--file`:

```bash
python -m slm --file recording.wav --fs-db 128.1 --realtime --measure LAeq LAF --dt 1.0
```

Use `--blocksize SAMPLES` to set the processing block size (default 1024). The block duration
(`blocksize / samplerate`) is the quantum at which log rows are written, so `--dt` is rounded
up to the next block edge:

```bash
python -m slm --file recording.wav --fs-db 128.1 --blocksize 4800 --measure LAeq --dt 1.0
```

Use `--display bars` for a live-updating bar graph instead of the default scrolling
`plain` text (needs a TTY; falls back to plain otherwise):

```bash
python -m slm --device 0 --sensitivity-mv 50 --measure LAeq:bands:63-8000 --display bars
```

Use `--signal-conditioning SPEC` to insert a band-limiting filter at the head of every
metering chain, in front of the frequency weighting (e.g. `--signal-conditioning xl2`). See
[Signal conditioning](signal-conditioning.md) for the full spec.

### Sensitivity flags (mutually exclusive)

| Flag | Argument | Use when… |
|------|----------|-----------|
| `--fs-db DB` | dB SPL at 0 dBFS | WAV recorded by a hardware SLM with an FS annotation |
| `--sensitivity-mv MV` | mV/Pa | You know the microphone sensitivity from its datasheet |
| `--sensitivity-dbv DBV` | dBV re 1 V/Pa | Manufacturer lists sensitivity as e.g. `−34 dBV` |

See [Calibration](calibration.md) for deriving sensitivity from a calibrator recording.

## White-noise generator (no audio hardware)

`--generator` drives the engine from a built-in Gaussian white-noise source — handy for
testing the pipeline. It requires `--measure` or `--config`:

```bash
python -m slm --generator --sensitivity-mv 50 --measure LAeq LZeq --duration 30
```

## Using a TOML config file

```bash
python -m slm --file recording.wav --config config.toml --fs-db 128.1
```

CLI flags (`--measure`, `--output`, `--dt`, `--precision`, `--warmup`, `--queue-maxsize`,
`--signal-conditioning`) override the values in the config file when both are supplied. See
[Configuration](configuration.md) for the full schema.

## Real-time input (requires sounddevice)

```bash
python -m slm --list-devices
python -m slm --device 0 --sensitivity-mv 50 --measure LAeq LAFmax --dt 1.0
```

`--device` accepts an index or a name substring (`--list-devices` shows both). `--samplerate`
sets the requested rate (default 48000 Hz; ignored for files, where it is read from the WAV
header). `--queue-maxsize N` bounds the block queue between the audio driver and the engine
(default `0` = unbounded); a finite value bounds latency but drops blocks if the engine falls
behind.

## Interactive REPL

```bash
python -m slm
```

The REPL lets you load files, set sensitivity, add metrics, and start/stop measurements
interactively. Type `help` for a list of commands.

Use `-i` (`--interactive`) to pre-populate the shell with a file, sensitivity, and/or config
before the prompt opens — handy when you have a fixed setup but still want to adjust before
running:

```bash
python -m slm -i --file recording.wav --fs-db 128.1 --config config.toml
```

Key REPL commands: `file`, `device`, `generator`, `sensitivity`, `calibrate`, `add`,
`remove`, `dt`, `precision`, `output`, `name`, `warmup`, `conditioning`, `queue`,
`samplerate`, `blocksize`, `realtime`, `display`, `show`, `tree`, `inspect`, `save`, `load`,
`start`. The
`samplerate` and `blocksize` commands mirror the `--samplerate`/`--blocksize` CLI flags
(sample rate applies to device/generator input only; for a WAV file it is read from the
header).

`conditioning SPEC` mirrors the `--signal-conditioning` flag (see
[Signal conditioning](signal-conditioning.md)): `conditioning xl2` selects the preset input
filter, `conditioning F_HPF N_HPF F_LPF N_LPF` builds a custom band-limiting filter
(e.g. `conditioning 20 2 23000 4`), and `conditioning none` removes it. With no argument it
shows the current setting.

`output DIR` sets the directory results are written to, and `name NAME` sets the
measurement name (the output file stem); files are written to `DIR/NAME_report.csv`,
`DIR/NAME_rta_report.csv`, etc.:

```
output results          # set the output directory
name street_noise_01    # set the measurement name
                        # -> results/street_noise_01_report.csv
```

`warmup SECONDS` runs a settling phase before measuring: the signal is processed but not
logged, then the meters are reset so the accumulators (Leq, max/min) start from a settled
filter state rather than from the initial transient. The logged timestamps restart at 0
after warm-up, and a `start` duration measures the post-warm-up length.

`tree` previews the planned plugin chain for the current metrics, and `inspect METRIC` prints
a detailed breakdown of a single metric — both without running anything.

`start` takes an optional duration for a fixed-length measurement, given as `hh:mm:ss`,
`mm:ss`, or just `ss` (fields may be fractional). With no argument it runs until the end
of the file or `Ctrl+C`:

```
start            # run until end of file / Ctrl+C
start 30         # run for 30 seconds
start 1:30       # run for 1 minute 30 seconds
start 01:00:00   # run for 1 hour
```

The run stops on a block edge, so the measured length is rounded up to the next whole
block (`blocksize/fs`).
