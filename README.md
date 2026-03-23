# Open SPL

This project implements audio and acoustics processing components, with a focus on sound level measurement and standardized frequency weighting.

## Overview

The project builds on established acoustics standards (e.g. A- and C-weighting filters) and is intended to run on any platform.

## Dependencies

This project makes use of **acoustic-toolbox**:

- acoustic-toolbox
  https://github.com/Universite-Gustave-Eiffel/acoustic-toolbox
  © Université Gustave Eiffel
  Licensed under the BSD 3-Clause License

Parts of the signal processing logic (e.g. frequency weighting filters) are based on this library.

## License

This project is licensed under the **BSD 3-Clause License**.

It includes code from *acoustic-toolbox*, which remains © Université Gustave Eiffel and is distributed under the same license.
See the `LICENSE` file for full details.

## Getting Started

1. Clone the repository
2. Install required dependencies
3. Build and run according to your target platform

Further documentation will be added as the project evolves.

## Calibration

The calibration routine derives the **controller sensitivity** from a recording of a known-level tone:

```
python -m slm --calibrate --file cal.wav --cal-level 94.0 --cal-freq 1000.0
```

A 1/3-octave bandpass filter is applied around `--cal-freq` before the RMS is computed, so
harmonics and background noise do not corrupt the estimate.

### Controller sensitivity vs microphone sensitivity

The value returned by `--calibrate` is the **controller sensitivity** (V/Pa), which is the
factor that converts raw WAV float samples into acoustic pressure in Pascal.  It is **not** the
same as the physical microphone sensitivity you would read off a datasheet or a calibrator
display.

For a WAV file recorded by a hardware SLM (e.g. the NTi XL2), the samples are normalized to
the ADC full-scale voltage.  The FS annotation embedded in the filename (e.g.
`FS128.1dB(PK)`) encodes the entire recording chain — microphone sensitivity, preamplifier
gain, and ADC gain — as a single number.  The controller sensitivity collapses this chain:

```
controller_sensitivity = 1 / (P_ref × 10^(FS_dB / 20))
```

**Example:** if the physical microphone sensitivity is −26 dBV/Pa and the XL2 annotates the
recording as FS 128.1 dBSPL(PK), the controller sensitivity is approximately −34 dBV/Pa.
The difference (≈ 8 dB) is the ADC full-scale voltage (≈ 2.55 V).  Using the physical
microphone sensitivity (−26 dBV/Pa) directly as the controller sensitivity would produce SPL
readings that are ≈ 8 dB too high.

Use the calibrated controller sensitivity (from `--calibrate` or `--fs-db`) whenever
processing WAV files recorded by a hardware SLM.  The physical microphone sensitivity is only
relevant when interfacing with a live ADC whose full-scale voltage is independently known.