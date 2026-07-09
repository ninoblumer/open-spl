# Signal conditioning

Signal conditioning inserts a band-limiting filter at the **head of every metering chain**,
in front of the frequency weighting. 

The same spec is accepted three ways:

- the `--signal-conditioning SPEC` CLI flag (see [Getting started](getting-started.md)),
- the `conditioning SPEC` REPL command,
- the `signal_conditioning` key in a [config file](configuration.md).

## Spec forms

`SPEC` is either a preset name or a custom band-limiting spec:

- **`xl2`** — preset band-limiting analog input filter: a 4.4 Hz Butterworth high-pass and a
  23 kHz Butterworth low-pass (both 4th-order), cascaded. Use it when comparing broadband
  results against an XL2.
- **`none`** (the default) — leaves the chain unconditioned.
- **`F_HPF N_HPF F_LPF N_LPF`** — a custom filter given as four values: high-pass cutoff (Hz)
  and order, then low-pass cutoff (Hz) and order. An order of `0` disables that stage (both
  `0` = passthrough); negative orders are rejected.

## CLI

```bash
python -m slm --file recording.wav --fs-db 128.1 --measure LZeq --signal-conditioning xl2

# custom: 20 Hz 2nd-order HPF + 23 kHz 4th-order LPF
python -m slm --file recording.wav --fs-db 128.1 --measure LZeq --signal-conditioning 20 2 23000 4
```

## REPL

`conditioning xl2` selects the preset input filter, `conditioning F_HPF N_HPF F_LPF N_LPF`
builds a custom band-limiting filter (e.g. `conditioning 20 2 23000 4`), and
`conditioning none` removes it. With no argument it shows the current setting.

## Config file

Set it in the `[measurement]` table:

```toml
[measurement]
signal_conditioning = "xl2"          # or "20 2 23000 4", or omit for none
```

See [Configuration](configuration.md#measurement-keys) for the full schema.

## Notes

- Apply signal conditioning to **broadband** chains only, not the octave/RTA path: RTA band
  edges (≈6.3 Hz–20 kHz) sit inside the passband, so the input filter would bias the outer
  bands.
- The 4th-order high-pass rings on a cold start (~0.29 dB peak overshoot). Use `--warmup` to
  settle the chain and reset the peak/max/min meters before measuring.
- `PluginZWeighting` remains a correct flat Z-weighting per IEC 61672-1; reach for the `xl2`
  preset only when you specifically want to match XL2 hardware, which band-limits its input.
