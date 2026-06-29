# slm-test-02 — recording plan

Second validation set. **Goal:** compare the full `soundlevelmeter` pipeline against the
NTi XL2 on signals close to real-world application — not standard conformance (the XL2 is a
type-approved Class-1 SLM and is assumed to meet the standard).

Each recording is meant to drive the **whole SLM** end-to-end
(`parse_metric → assemble_engine → engine.run → Reporter`) and diff the emitted
`_log.csv` / `_report.csv` / `_rta_*.csv` against the XL2's parsed log/report — a real
integration test, not helper math.

## Measurement method: offline comparison + live smoke test

**Primary method — offline (the validation).** XL2 stays the recorder. It logs its own
metrics *and* records the audio WAV; the WAV is fed to the SLM offline via
`run_measurement`. Both the reference and the SLM input derive from the *same* captured
audio, so the comparison is apples-to-apples and independent of any interface coloration.
This is what the integration tests assert against.

**Secondary — live functional check (not a precision comparison).** A RME Fireface UCX is
available. Run the SLM **live** off the UCX input (`run_realtime_measurement` /
`SounddeviceController`) to confirm the real-time path actually works — block streaming,
queue/overrun behaviour, dt-interval logging, the live display. This is a smoke test of the
realtime pipeline, *not* a tight numeric comparison.

A side-by-side live setup (split the source: one leg to the XL2, one to the UCX) is possible
but the UCX front-end is not guaranteed perfectly flat, so small deviations vs the XL2 are
expected — treat any live-vs-XL2 numbers as indicative only. Keep tight numeric assertions on
the offline WAV path. Do at least one live test run early to confirm the realtime path works
before relying on it.

## Scenarios

| # | Scene | Duration | Notes |
|---|-------|----------|-------|
| 1 | Road traffic noise | 5–10 min | outdoor, with vehicle pass-bys (transients) |
| 2 | Pink noise, lab room, **moving source** | ≥ 1 min | source moved during recording → time-varying spectrum |
| 3 | Music playback, lab room | 3 min | sustained broadband with bass content; repeatable |
| 4 | Tapping machine, receiving room | 3 min | building-acoustics impact sound, periodic impulsive |
| 5 | Manual random hammering on a wooden wall | 2–3 min | ~1 hit per second, random timing |
| 6 | Background noise | 3 min | near the noise floor |

## XL2 configuration (same for all six recordings)

- **Profile: Full mode** — logs the entire broadband set per `dt` (all weightings × S/F/I ×
  max/min/eq, peaks, etc.).
- **Broadband + RTA logged simultaneously**, both at **Log-Interval = 1 s** (matches the SLM's
  `LAeq_dt` default and the ±0.18 dB comparison tolerance).
- **RTA resolution: 1/3 octave**, frequency weighting **Z** (logs `LZeq_dt` per band; 1/3-oct
  is a superset of 1/1, and the SLM supports both).
- **Audio recording: 24-bit / 48 kHz**, with the `FSxxx.xdB(PK)` filename annotation — this is
  what the sensitivity-from-FS path reads; works for acoustic recordings just as for the
  electrical slm-test-01 ones.
- **Session calibration**: record a 94 dB / 1 kHz reference at the start (SLM_000 style) so
  sensitivity is pinned per session.
- **Range 20–120 dB** for scenes 1–5; consider the lower range for the quiet background
  (scene 6) for better SNR (the FS annotation tracks the range automatically).
- Watch the **Overload** column on the impulsive scenes (4, 5) — keep hammer/tapping peaks
  below full scale.

One config captures everything; no per-scenario metric selection is needed.

## Comparison targets (SLM vs XL2) per scene

The SLM reproduces the **eq / max-min / peak / SEL / 1-3-oct-spectrum** families. SEL is the
`E` measure (metric names `LAE` / `LCE` / `LZE`, no time-weighting letter; XL2 columns
`LAE_dt` per interval and `LAE`/`LCE`/`LZE` in the report). The primary columns each scene is
meant to exercise:

| Scene | Comparison targets |
|-------|--------------------|
| 1 Road traffic | `LAeq`, `LCeq`, `LZeq` overall **+ `LAeq_dt` series**; `LAE` (whole-record exposure); `LAFmax`, `LASmax`; `LCPKmax`; 1/3-oct `LZeq` spectrum |
| 2 Pink noise (moving) | 1/3-oct **`LZeq_dt` spectrum series**; `LZeq`/`LAeq` overall |
| 3 Music | `LAeq`, `LCeq` overall + `_dt`; `LAE`; 1/3-oct spectrum; `LCPKmax` |
| 4 Tapping machine | 1/3-oct `LZeq` spectrum; `LAFmax`, `LAImax`; `LCPKmax`; `LAE` |
| 5 Hammering ~1/s | `LCPKmax` (per-hit peak); `LAImax` (impulse weighting); `LASmax`/`LAFmax`; `LAE`; F/S/I divergence |
| 6 Background | `LAeq`, `LZeq` overall; 1/3-oct `LZeq` spectrum at low level |

## Not compared

- **Percentiles (`LAFn%`)** — **not implemented in the soundlevelmeter, and not planned.**
  The XL2 logs them in Full mode anyway, but they are not a comparison target.

## Caveats to handle in the integration test

- **Time alignment.** slm-test-01 showed a ~0.4 s offset between XL2-log start and WAV start,
  which only bit at moments of abrupt level change. Hold **overall report values** to a tight
  tolerance (~0.2 dB) but expect the per-second `_dt` series to diverge at sharp transients
  (pass-by onset, a hammer hit). Either cross-correlate the two series to align, or compare the
  series with a looser tolerance / excluding rapid-change samples. Long, stationary records
  make the overall metrics dominate.
- **Calibration.** Derive sensitivity from the session cal tone (`calibrate_from_file`) or the
  WAV `FSxxx.xdB(PK)` annotation — confirm the recording mode writes that annotation.


## Measurements:
Calibration 29.06.2026 14:39; 44.2 mV/Pa 

SLM_000: Calibration tone
SLM_001: Pink Noise from a  moving source in a room, moving the microphone aswell
SLM_002: Background Noise in a room
SLM_003: Tapping machine noise
SLM_004: Manual hammering
