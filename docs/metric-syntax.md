# Metric name syntax

Every quantity the SLM measures is named with a compact string parsed by
`slm.assembly.parse_metric`. The same syntax is accepted by `--measure`, the REPL `add`
command, and the `[metrics] require` list in a [config file](configuration.md).

```
L[ACZ][FSI?](eq|max|min|E)?[_(dt|Ns|Nm|Nh)][:bands:[N/M:]fmin-fmax]
```

## Frequency weighting

| Letter | Weighting |
|--------|-----------|
| `A`    | A-weighting (IEC 61672-1) |
| `C`    | C-weighting (IEC 61672-1) |
| `Z`    | Z-weighting — flat passthrough (IEC 61672-1 Annex E.5) |

## Time weighting (required for max/min and bare metrics; forbidden for eq/E)

| Letter | Filter |
|--------|--------|
| `F`    | Fast (τ = 0.125 s) |
| `S`    | Slow (τ = 1 s) |
| `I`    | Impulse |

## Measure

| Suffix | Description |
|--------|-------------|
| `eq`   | Energy-equivalent level (Leq) — no time-weighting letter |
| `max`  | Maximum — requires time-weighting letter |
| `min`  | Minimum — requires time-weighting letter |
| `E`    | Sound exposure level (LE) — no time-weighting letter |
| *(none)* | Most-recent time-weighted sample — requires time-weighting letter, no window |

## Window suffix (optional)

| Suffix | Description |
|--------|-------------|
| *(none)* | Accumulating over the whole file/stream |
| `_dt`  | Moving window equal to the engine's logging interval |
| `_Ns`  | Moving N-second window (e.g. `_5s`, `_30s`) |
| `_Nm`  | Moving N-minute window (e.g. `_1m`) |
| `_Nh`  | Moving N-hour window (e.g. `_1h`) |

A moving-window metric reads `nan` until a full window of data has accumulated
(e.g. `LZeq_30s` logs `nan` for its first 30 s) rather than ramping up from a
partially filled window; an accumulating metric reads `nan` until its first
sample. Once enough data is present, real values are reported.

## Band suffix (optional)

| Suffix | Description |
|--------|-------------|
| `:bands:63-8000` | 1/1-octave bands, 63 Hz to 8 kHz |
| `:bands:1/3:31-16000` | 1/3-octave bands, 31 Hz to 16 kHz |
| `:bands:1/6:63-8000` | 1/6-octave bands, 63 Hz to 8 kHz |
| `:bands:N/M:fmin-fmax` | Any N/M-octave filter bank (M/N bands per octave) |

Omitting the `N/M:` fraction defaults to 1/1-octave.

## Examples

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
