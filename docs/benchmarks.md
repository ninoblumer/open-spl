# Real-time load benchmark

The benchmark characterises how much CPU time the SLM engine consumes relative to the
real-time budget (`t_budget = blocksize / samplerate`) across a matrix of loadout × block
size. No audio hardware is required — a built-in white-noise generator drives the engine.

## Loadouts

| ID | Description |
|----|-------------|
| K0 | Normative minimum: LAF + LAeq |
| K1 | Typical Class-1 handheld: A/C/Z with F- and S-time-weighting, Leq and max |
| K2 | K1 + full IEC 61260-1 1/3-octave filter bank on each weighting, with all broadband metric types (Leq, Fmax, Smax, F, S) per band |

## Processing utilisation (free-running)

Runs each cell at full speed (no real-time pacing) and records ρ = t_proc / t_budget for
every block. ρ < 1 means the engine is fast enough for real-time use; ρ ≥ 1 means it is
not.

```bash
# Quick run — 10 s of audio per cell (~8 min total)
python scripts/realtime_benchmark.py --duration 10 --out output/quick

# Full run — 300 s per cell (default)
python scripts/realtime_benchmark.py --out output/full
```

Options:

```
--samplerate 48000          sample rate in Hz (default: 48000)
--blocksizes 128 256 …      block sizes to test (default: 128 256 512 1024 4096)
--loadouts K0 K1 K2         loadouts to test (default: all three)
--duration 300              audio-time duration per Quantity A cell in seconds
--out results/              output directory (created if absent)
```

## Real-time deadline violations

Runs a specified cell at real-time pace and counts how many blocks missed their deadline
(overruns). Only meaningful for cells where Quantity A shows ρ near 1.

```bash
python scripts/realtime_benchmark.py --loadouts K2 --blocksizes 256 512 \
    --rt-duration 60 --run-b K2:256 K2:512 --out output/rt
```

Options:

```
--rt-duration 3600          wall-clock duration per Quantity B cell in seconds
--run-b K2:256 K2:512 …     cells to measure (LOADOUT:BLOCKSIZE tokens); bare
                            --run-b runs every loadout×blocksize combination
--skip-a                    skip Quantity A entirely (requires --run-b)
```

## Generating figures

After running the benchmark, produce publication-quality figures with:

```bash
python scripts/plots_performance.py --data output/quick
```
