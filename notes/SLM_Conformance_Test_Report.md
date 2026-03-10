# SLM Conformance Test Report
## Standards-Based Test Specification for a Python Sound Level Meter (SLM)

**Generated from:** IEC 61260-1:2014, IEC 61672-1:2013, IEC 61672-2:2013, IEC 61672-3:2013  
**Scope:** Testable numerical/algorithmic specifications extracted from the English sections of each standard. Hardware-only, EMC, and physical calibration requirements are noted but not expanded into software tests.

---

## 1. Overview

The four standards in the bundle cover:

| Standard | Title | Relevant to SLM |
|---|---|---|
| **IEC 61260-1:2014** | Octave-band and fractional-octave-band filters — Part 1: Specifications | Band-pass filter algorithm, attenuation shape, frequency math |
| **IEC 61672-1:2013** | Sound level meters — Part 1: Specifications | Core SLM signal chain: frequency weighting, time weighting, level linearity, tone bursts, overload |
| **IEC 61672-2:2013** | Sound level meters — Part 2: Pattern evaluation tests | Reference test procedures and acceptance criteria for full lab testing |
| **IEC 61672-3:2013** | Sound level meters — Part 3: Periodic tests | Simplified subset of key checks for in-service verification |

**Referenced standards that contain additional specifications important to test:**

- **IEC 60942** — Sound calibrators (class 1 and class 2). Relevant for calibration check frequency tests.
- **IEC 61183** — Random-incidence and diffuse-field calibration. Relevant for random-incidence frequency response tests.
- **IEC 62585** — Methods to determine free-field corrections. Provides max-permitted uncertainties for correction data.
- **IEC 61094-4/6** — Measurement microphones (working standard, electrostatic actuators). Referenced for coupler tests.
- **ISO/IEC Guide 98-3 (GUM)** — Uncertainty of measurement. Defines the 95 % coverage probability required in conformance assessments.
- **ISO/IEC Guide 98-4** — Role of measurement uncertainty in conformity assessment. Defines the acceptance-limit + max-permitted-uncertainty dual criterion.

---

## 2. Conformance Criterion (All Standards)

All four standards use the same dual criterion (from ISO/IEC Guide 98-4):

> **Conformance is demonstrated when BOTH of the following hold:**
> 1. Measured deviation from design goal ≤ applicable acceptance limit
> 2. Actual expanded uncertainty of measurement ≤ corresponding maximum-permitted expanded uncertainty (at 95 % coverage probability)

For software tests where measurement uncertainty is under programmatic control, criterion 2 is met by design, but the acceptance limits of criterion 1 must still be verified.

---

## 3. IEC 61260-1 — Octave-Band and Fractional-Octave-Band Filter Tests

### 3.1 Octave Frequency Ratio

**Specification (§5.2):**  
G = 10^(3/10) = 1.995 26 (base-10 filters). Base-2 (G=2) filters are not recommended for new designs.

**Test:**
- `test_octave_frequency_ratio`: Assert `G == 10**(3/10)` to required precision (6 significant digits: 1.995 26).
- Verify all frequency calculations use G = 10^(3/10), not G = 2.

---

### 3.2 Reference Frequency

**Specification (§5.3):**  
Reference frequency `f_r = 1000 Hz` exactly.

**Test:**
- `test_reference_frequency`: Assert `f_r == 1000.0` Hz.

---

### 3.3 Exact Mid-Band Frequencies

**Specification (§5.4):**
- Odd denominator bandwidth designator: `f_m = f_r × G^(x/b)`  
- Even denominator bandwidth designator: `f_m = f_r × G^((2x+1)/(2b))`

**Tests:**
- `test_exact_midband_freq_odd_denominator`: For 1/1 (octave) and 1/3 filters, verify `f_m` matches formula (2) for integer values of x from e.g. -10 to +10.
- `test_exact_midband_freq_even_denominator`: For 1/2 filters, verify `f_m` matches formula (3).
- `test_exact_midband_freq_table_E1`: Spot-check computed values against the reference table in Annex E. Example values: x=−15 → 31.623 Hz (nominal 31.5 Hz), x=0 → 1000 Hz, x=+15 → 31 623 Hz (nominal 31.5 kHz).

---

### 3.4 Nominal Mid-Band Frequencies

**Specification (Annex E, §5.5):**  
Filters shall be labelled by rounded nominal mid-band frequencies. Rounding rules:
- Most-significant digit 1–4: round to 3 significant figures.
- Most-significant digit 5–9: round to 2 significant figures.

**Tests:**
- `test_nominal_midband_rounding`: Verify nominal frequency rounding rules for a set of exact mid-band frequencies.
- `test_nominal_midband_table_compliance`: Check nominal values for standard octave and 1/3-octave series against Table E.1 (e.g., 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 Hz).

---

### 3.5 Band-Edge Frequencies

**Specification (§5.6):**
- Lower band-edge: `f_1 = f_m × G^(−1/(2b))`
- Upper band-edge: `f_2 = f_m × G^(+1/(2b))`
- Ratio: `f_2 / f_1 = G^(1/b)`
- Geometric mean: `f_m = sqrt(f_1 × f_2)`

**Tests:**
- `test_band_edge_lower`: Verify `f_1` formula for each filter in a set.
- `test_band_edge_upper`: Verify `f_2` formula.
- `test_band_edge_ratio`: Assert `f_2 / f_1 ≈ G^(1/b)` within floating-point tolerance.
- `test_band_edge_geometric_mean`: Assert `sqrt(f_1 × f_2) ≈ f_m`.

---

### 3.6 Relative Attenuation Shape — Acceptance Limits (Class 1 and Class 2)

**Specification (§5.10, Table 1):**  
At normalized frequency Ω = f/f_m, relative attenuation ΔA(Ω) must lie within the following bounds (Table 1, octave-band filters):

| Normalized frequency Ω | Class 1 min/max (dB) | Class 2 min/max (dB) |
|---|---|---|
| ≤ G^(−4) | +70; +∞ | +60; +∞ |
| G^(−3) | +60; +∞ | +54; +∞ |
| G^(−2) | +40.5; +∞ | +39.5; +∞ |
| G^(−1) | +16.6; +∞ | +15.6; +∞ |
| G^(−1/2) − ε | +1.2; +∞ | +0.8; +∞ |
| G^(−1/2) + ε | −0.4; +5.3 | −0.6; +5.8 |
| G^(−3/8) | −0.4; +1.4 | −0.6; +1.7 |
| G^(−1/4) | −0.4; +0.7 | −0.6; +0.9 |
| G^(−1/8) | −0.4; +0.5 | −0.6; +0.7 |
| G^(0) = 1 | −0.4; +0.4 | −0.6; +0.6 |
| G^(+1/8) | −0.4; +0.5 | −0.6; +0.7 |
| G^(+1/4) | −0.4; +0.7 | −0.6; +0.9 |
| G^(+3/8) | −0.4; +1.4 | −0.6; +1.7 |
| G^(+1/2) − ε | −0.4; +5.3 | −0.6; +5.8 |
| G^(+1/2) + ε | +1.2; +∞ | +0.8; +∞ |
| G^(+1) | +16.6; +∞ | +15.6; +∞ |
| G^(+2) | +40.5; +∞ | +39.5; +∞ |
| G^(+3) | +60; +∞ | +54; +∞ |
| ≥ G^(+4) | +70; +∞ | +60; +∞ |

Between breakpoints, limits are linearly interpolated in log frequency (Formula 11).

For fractional-octave-band filters with bandwidth 1/b, the normalized breakpoint frequencies are rescaled using Formulas (9) and (10).

**Tests:**
- `test_relative_attenuation_passband_class1`: Apply a swept sinusoidal input to each octave-band filter and verify ΔA at the exact mid-band frequency is within −0.4 to +0.4 dB.
- `test_relative_attenuation_passband_class2`: Same, limits −0.6 to +0.6 dB.
- `test_relative_attenuation_stopband_class1`: Verify ΔA ≥ +70 dB at Ω ≤ G^(−4) and Ω ≥ G^(+4).
- `test_relative_attenuation_breakpoints_class1`: Verify ΔA at each tabulated breakpoint of Table 1 for class 1.
- `test_relative_attenuation_breakpoints_class2`: Same for class 2.
- `test_relative_attenuation_1_3_octave_class1`: Repeat above tests for 1/3-octave-band filters using rescaled breakpoints per Formulas (9)/(10) and Annex F.
- `test_relative_attenuation_interpolation`: Verify that between two consecutive breakpoints, the acceptance limit is determined by linear interpolation in log frequency (Formula 11).

---

### 3.7 Effective Bandwidth Deviation

**Specification (§5.12):**
- Effective bandwidth `B_e` is integral of normalized response weighted by 1/Ω (Formula 13/14).
- Reference effective bandwidth: `B_r = ln(f_2/f_1)` (Formula 15).
  - Octave: B_r = 0.690 776
  - 1/3-octave: B_r = 0.230 259
- Effective bandwidth deviation: `ΔB = 10 × log10(B_e / B_r)`.
- Acceptance limits: ±0.4 dB for class 1, ±0.6 dB for class 2.

**Tests:**
- `test_reference_effective_bandwidth_octave`: Assert `B_r ≈ 0.690776` for octave filters.
- `test_reference_effective_bandwidth_third_octave`: Assert `B_r ≈ 0.230259` for 1/3-octave filters.
- `test_effective_bandwidth_deviation_class1`: Compute `ΔB` for each filter; assert within ±0.4 dB.
- `test_effective_bandwidth_deviation_class2`: Same, within ±0.6 dB.

---

### 3.8 Level Linearity

**Specification (§5.13):**
- Linear operating range: ≥ 60 dB (class 1), ≥ 50 dB (class 2).
- Level linearity deviation within upper 40 dB of linear range: ±0.5 dB (class 1), ±0.6 dB (class 2).
- Level linearity deviation in lower portion of linear range: ±0.7 dB (class 1), ±0.9 dB (class 2).
- Adjacent level ranges must overlap by ≥ 40 dB (class 1) or ≥ 30 dB (class 2).

**Tests:**
- `test_linear_operating_range_width`: Verify the range is ≥ 60 dB for class 1.
- `test_level_linearity_upper_40dB`: Sweep input levels over top 40 dB; assert deviation ≤ ±0.5 dB.
- `test_level_linearity_lower_range`: Sweep lower portion; assert deviation ≤ ±0.7 dB.

---

### 3.9 Time-Invariant Operation (Exponential Sweep)

**Specification (§5.14):**  
For a swept-frequency input sweeping one decade in 2–5 s, the deviation of measured output level from theoretical `L_c` (Formula 17) must be:
- ±0.4 dB (class 1), ±0.6 dB (class 2).

**Tests:**
- `test_time_invariant_operation_class1`: Generate exponential-swept sine (1 decade in 2–5 s), measure time-averaged output per filter, compare to theoretical level from Formula 17.
- `test_time_invariant_operation_sweep_rate`: Verify the test sweep rate corresponds to r in range 0.4605 s⁻¹ to 1.151 s⁻¹.

---

### 3.10 Summation of Output Signals

**Specification (§5.16):**  
For a sinusoidal input between two adjacent band mid-frequencies, the difference between (input level − ref attenuation) and (level of sum of mean-square outputs from adjacent filters) must be:
- +0.8 dB to −1.8 dB (class 1), +1.8 dB to −3.8 dB (class 2).

**Tests:**
- `test_output_summation_class1`: At frequencies between adjacent mid-bands, compute sum of adjacent filter output levels; check against input level minus reference attenuation.

---

### 3.11 Overload Indicator Timing

**Specification (§5.17):**
- Overload indication must persist for at least 1 s.
- For instruments displaying stored results, overload must be latched as long as the result is displayed.

**Tests:**
- `test_overload_indicator_minimum_duration`: Apply overload signal; verify overload flag remains set for ≥ 1 s after signal removal.
- `test_overload_indicator_latching`: Verify flag latches during measurement and persists while result is displayed.

---

## 4. IEC 61672-1 — Sound Level Meter Tests

### 4.1 Frequency Weighting A, C, Z

**Specification (§5.5, Table 3):**  
Design goals and acceptance limits at nominal frequencies (Hz) from 10 Hz to 20 kHz:

| Freq (Hz) | A (dB) | C (dB) | Z (dB) | Class 1 limits (dB) | Class 2 limits (dB) |
|---|---|---|---|---|---|
| 10 | −70.4 | −14.3 | 0.0 | +3.0; −∞ | +5.0; −∞ |
| 12.5 | −63.4 | −11.2 | 0.0 | +2.5; −∞ | +5.0; −∞ |
| 16 | −56.7 | −8.5 | 0.0 | +2.0; −4.0 | +5.0; −∞ |
| 20 | −50.5 | −6.2 | 0.0 | ±2.0 | ±3.0 |
| 25 | −44.7 | −4.4 | 0.0 | +2.0; −1.5 | ±3.0 |
| 31.5 | −39.4 | −3.0 | 0.0 | ±1.5 | ±3.0 |
| 40 | −34.6 | −2.0 | 0.0 | ±1.0 | ±2.0 |
| 50 | −30.2 | −1.3 | 0.0 | ±1.0 | ±2.0 |
| 63 | −26.2 | −0.8 | 0.0 | ±1.0 | ±2.0 |
| 80 | −22.5 | −0.5 | 0.0 | ±1.0 | ±2.0 |
| 100 | −19.1 | −0.3 | 0.0 | ±1.0 | ±1.5 |
| 125 | −16.1 | −0.2 | 0.0 | ±1.0 | ±1.5 |
| 160 | −13.4 | −0.1 | 0.0 | ±1.0 | ±1.5 |
| 200 | −10.9 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 250 | −8.6 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 315 | −6.6 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 400 | −4.8 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 500 | −3.2 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 630 | −1.9 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 800 | −0.8 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 1000 | 0.0 | 0.0 | 0.0 | ±0.7 | ±1.0 |
| 1250 | +0.6 | 0.0 | 0.0 | ±1.0 | ±1.5 |
| 1600 | +1.0 | −0.1 | 0.0 | ±1.0 | ±2.0 |
| 2000 | +1.2 | −0.2 | 0.0 | ±1.0 | ±2.0 |
| 2500 | +1.3 | −0.3 | 0.0 | ±1.0 | ±2.5 |
| 3150 | +1.2 | −0.5 | 0.0 | ±1.0 | ±2.5 |
| 4000 | +1.0 | −0.8 | 0.0 | ±1.0 | ±3.0 |
| 5000 | +0.5 | −1.3 | 0.0 | ±1.5 | ±3.5 |
| 6300 | −0.1 | −2.0 | 0.0 | +1.5; −2.0 | ±4.5 |
| 8000 | −1.1 | −3.0 | 0.0 | +1.5; −2.5 | ±5.0 |
| 10000 | −2.5 | −4.4 | 0.0 | +2.0; −3.0 | +5.0; −∞ |
| 12500 | −4.3 | −6.2 | 0.0 | +2.0; −5.0 | +5.0; −∞ |
| 16000 | −6.6 | −8.5 | 0.0 | +2.5; −16.0 | +5.0; −∞ |
| 20000 | −9.3 | −11.2 | 0.0 | +3.0; −∞ | +5.0; −∞ |

**Notes:**
- A-weighting analytical expression from Annex E of IEC 61672-1 (4th-order rational function of f).
- C-weighting from Annex E.
- Z-weighting = flat (0 dB at all frequencies).
- Design goal is 0 dB at 1 kHz for all weightings.
- Between tabulated frequencies: limits are the larger of the limits at the two adjacent frequencies.

**Tests:**
- `test_a_weighting_design_goals`: Compute A-weighting response at each nominal frequency using analytical expression (Annex E). Assert each matches the Table 3 design goal rounded to 0.1 dB.
- `test_a_weighting_acceptance_class1`: Apply sinusoidal signals at each Table 3 frequency; assert deviations ≤ class 1 limits.
- `test_a_weighting_acceptance_class2`: Same for class 2.
- `test_c_weighting_design_goals`: Same for C-weighting.
- `test_c_weighting_acceptance_class1/2`: Same for C limits.
- `test_z_weighting_flat`: Z-weighting response must be 0.0 dB (± applicable tolerance) at all frequencies.
- `test_weighting_at_1khz_zero`: Assert all weighting responses equal 0.0 dB at exactly 1 kHz.
- `test_c_vs_a_at_1khz`: The difference between C-weighted and A-weighted level at 1 kHz must be ≤ ±0.2 dB (§5.5.9).
- `test_z_vs_a_at_1khz`: Z-weighted minus A-weighted at 1 kHz must be ≤ ±0.2 dB (§5.5.9).

---

### 4.2 Time Weightings F and S

**Specification (§5.8):**
- F (Fast): exponential time constant τ = 0.125 s, decay rate design goal = 34.7 dB/s after cessation of steady 4 kHz signal. Acceptance limits: +3.8 dB/s; −3.7 dB/s.
- S (Slow): τ = 1 s, decay rate design goal = 4.3 dB/s. Acceptance limits: +0.8 dB/s; −0.7 dB/s.
- At 1 kHz, S-weighted level vs. F-weighted level for the same steady signal must not differ by more than ±0.1 dB (§5.8.3).

**Tests:**
- `test_time_weighting_F_constant`: Apply steady 4 kHz signal then abruptly remove; fit the decay slope and assert τ corresponds to 0.125 s (or check decay rate is within 31.0–38.5 dB/s).
- `test_time_weighting_S_constant`: Same for τ = 1 s (decay rate 3.6–5.1 dB/s).
- `test_F_vs_S_steady_1khz`: For a steady 1 kHz input, assert |L_S − L_F| ≤ 0.1 dB.

---

### 4.3 Toneburst Response

**Specification (§5.9, Table 4):**  
Reference 4 kHz toneburst response `δ_ref` for maximum F time-weighted level (relative to steady-state level):

| Duration (ms) | L_AFmax − L_A (dB) | L_AE − L_A (dB) | Class 1 limits | Class 2 limits |
|---|---|---|---|---|
| 1000 | 0.0 | 0.0 | ±0.5 | ±1.0 |
| 500 | −0.1 | −3.0 | ±0.5 | ±1.0 |
| 200 | −1.0 | −7.0 | ±0.5 | ±1.0 |
| 100 | −2.6 | −10.0 | ±1.0 | ±1.0 |
| 50 | −4.8 | −13.0 | ±1.0 | +1.0; −1.5 |
| 20 | −8.3 | −17.0 | ±1.0 | +1.0; −2.0 |
| 10 | −11.1 | −20.0 | ±1.0 | +1.0; −2.0 |
| 5 | −14.1 | −23.0 | ±1.0 | +1.0; −2.5 |
| 2 | −18.0 | −27.0 | +1.0; −1.5 | +1.0; −2.5 |
| 1 | −21.0 | −30.0 | +1.0; −2.0 | +1.0; −3.0 |
| 0.5 | −24.0 | −33.0 | +1.0; −2.5 | +1.0; −4.0 |
| 0.25 | −27.0 | −36.0 | +1.0; −3.0 | +1.5; −5.0 |

Reference approximation formulas:
- F-time-weighted: `δ_ref = 10 × log10(1 − e^(−T_b/τ))` dB (Formula 7)
- Sound exposure level: `δ_ref = 10 × log10(T_b / T_0)` dB where T_0 = 1 s (Formula 8)

**Tests:**
- `test_toneburst_response_F_1000ms`: Apply 4 kHz toneburst of 1000 ms; assert L_AFmax − L_A is within 0.0 ± 0.5 dB (class 1).
- `test_toneburst_response_F_parametric`: For each duration in Table 4, apply toneburst; compare measured δ to reference δ_ref (Formula 7); assert deviation within applicable limits.
- `test_toneburst_response_SEL_parametric`: Same but for sound exposure level using Formula 8.
- `test_toneburst_response_S_parametric`: Same for S-time weighting using lower half of Table 4.
- `test_toneburst_no_overload_in_range`: Verify no overload is triggered during toneburst tests within the specified input range (§5.9.6).

---

### 4.4 Response to Repeated Tonebursts

**Specification (§5.10):**  
For a sequence of n equal-amplitude 4 kHz tonebursts, the theoretical time-averaged level difference is:  
`δ_ref = 10 × log10(n × T_b / T_m)` dB  
Measured deviations must meet the acceptance limits of Table 4 for sound-exposure-level toneburst response.

**Tests:**
- `test_repeated_tonebursts`: Generate a sequence of n tonebursts, measure time-averaged sound level, compute expected value from Formula 9, assert deviation is within Table 4 SEL acceptance limits.

---

### 4.5 Level Linearity

**Specification (§5.6):**
- Linear operating range on reference level range: ≥ 60 dB at 1 kHz.
- Level linearity deviation: ≤ ±0.8 dB (class 1), ≤ ±1.1 dB (class 2) across total range.
- Any 1–10 dB change in input must produce the same change in displayed level; deviation ≤ ±0.3 dB (class 1), ±0.5 dB (class 2).
- Adjacent level ranges must overlap by ≥ 30 dB (time-weighted) or ≥ 40 dB (time-averaged/SEL).

**Tests:**
- `test_level_linearity_total_range_class1`: Sweep input level across full linear range; assert deviation ≤ ±0.8 dB for class 1.
- `test_level_linearity_incremental`: For each 1 dB and 10 dB input step, verify output changes by same amount ± 0.3 dB (class 1).
- `test_linear_range_width`: Assert linear operating range ≥ 60 dB at 1 kHz on reference level range.
- `test_level_range_overlap`: For multi-range implementations, verify overlap ≥ 40 dB.

---

### 4.6 Overload Indication

**Specification (§5.11):**
- Overload must be shown before acceptance limits are exceeded.
- For F/S time-weighted levels: display for ≥ 1 s or as long as overload persists (whichever is greater).
- For time-averaged/SEL measurements: overload must latch until measurement is reset.
- Difference between input levels causing overload on positive vs. negative half-cycles ≤ 1.5 dB.

**Tests:**
- `test_overload_shown_before_limit_exceeded`: Apply level just above upper boundary; verify overload flag is set before level linearity deviation exceeds limit.
- `test_overload_duration_minimum_1s`: Apply brief overload; confirm flag persists ≥ 1 s.
- `test_overload_latching_integrating_mode`: In time-averaging mode, confirm flag latches and remains until reset.
- `test_overload_half_cycle_symmetry`: Apply positive and negative half-cycles; assert difference in triggering levels ≤ 1.5 dB.

---

### 4.7 Under-Range Indication

**Specification (§5.12):**
- Under-range condition displayed when indicated level is below the lower boundary of linear operating range.
- Display persists ≥ 1 s or as long as condition exists.

**Tests:**
- `test_under_range_display_triggered`: Apply input below lower boundary; verify under-range flag is set.
- `test_under_range_duration_minimum_1s`: Confirm flag persists ≥ 1 s.

---

### 4.8 C-Weighted Peak Sound Level

**Specification (§5.13, Table 5):**  
Reference differences `L_Cpeak − L_C` for specified signals:

| Test signal | Freq (Hz) | Reference diff (dB) | Class 1 limits | Class 2 limits |
|---|---|---|---|---|
| 1 cycle | 31.5 | 2.5 | ±2.0 | ±3.0 |
| 1 cycle | 500 | 3.5 | ±1.0 | ±2.0 |
| 1 cycle | 8000 | 3.4 | ±2.0 | ±3.0 |
| positive half-cycle | 500 | 2.4 | ±1.0 | ±2.0 |
| negative half-cycle | 500 | 2.4 | ±1.0 | ±2.0 |

**Tests:**
- `test_c_peak_one_cycle_31Hz`: Apply 1-cycle burst at 31.5 Hz; assert `L_Cpeak − L_C` ≈ 2.5 ± 2.0 dB.
- `test_c_peak_one_cycle_500Hz`: Apply 1-cycle burst at 500 Hz; assert `L_Cpeak − L_C` ≈ 3.5 ± 1.0 dB.
- `test_c_peak_one_cycle_8kHz`: Apply 1-cycle burst at 8000 Hz; assert `L_Cpeak − L_C` ≈ 3.4 ± 2.0 dB.
- `test_c_peak_half_cycle_positive`: Apply positive half-cycle at 500 Hz; check reference difference.
- `test_c_peak_half_cycle_negative`: Apply negative half-cycle; assert same result ≈ positive within ≤ 1.5 dB difference.

---

### 4.9 Stability During Continuous Operation

**Specification (§5.14):**  
After 30 min of continuous operation, difference between initial and final A-weighted level indications at 1 kHz:
- ≤ ±0.1 dB (class 1), ≤ ±0.3 dB (class 2).

**Tests:**
- `test_stability_30min`: Run SLM for 30 min at steady 1 kHz input; compare final to initial indication; assert drift ≤ ±0.1 dB.

---

### 4.10 High-Level Stability

**Specification (§5.15):**  
After 5 min at a level 1 dB below upper boundary of the linear operating range, difference between initial and final indications ≤ ±0.1 dB (class 1), ±0.3 dB (class 2).

**Tests:**
- `test_high_level_stability_5min`: Hold input at 1 dB below upper boundary for 5 min; assert drift ≤ ±0.1 dB.

---

### 4.11 Analogue/Digital Output Match

**Specification (§5.19.4):**  
For any frequency in the SLM range, any weighting, and any level in the linear operating range, the difference between display-device level and output-port level: 0.0 dB design goal, acceptance limits ±0.1 dB.

**Tests:**
- `test_output_matches_display`: At multiple frequencies and levels, compare internal display value to digital output value; assert difference ≤ ±0.1 dB.

---

### 4.12 Time-Averaged Sound Level (L_Aeq) Formula

**Specification (§3.9, Equations 3–6):**

```
L_Aeq,T = 10 × log10( (1/T) × ∫ p_A²(t)dt / p_0² )
         = 10 × log10( E_A,T / (p_0² × T_0 × T/T_0) )
p_0 = 20 µPa, T_0 = 1 s
```

**Tests:**
- `test_laeq_formula`: With a known synthetic signal (e.g., 1 kHz sine of known amplitude), compute L_Aeq and compare to analytic expectation.
- `test_laeq_averaging_linearity`: Verify L_Aeq is consistent for varying integration times over a stationary signal.

---

### 4.13 Sound Exposure Level

**Specification (§3.12, Equation 4):**  
`L_AE,T = L_Aeq,T + 10 × log10(T/T_0)` where T_0 = 1 s.

**Tests:**
- `test_sel_from_laeq`: Verify L_AE computed from L_Aeq matches Formula 4.
- `test_sel_reference_exposure`: `E_0 = (20 µPa)² × 1 s = 400 × 10⁻¹² Pa²s`.

---

### 4.14 Crosstalk (Multi-Channel)

**Specification (§5.22):**  
For multi-channel systems at any frequency from 10 Hz to 20 kHz: channel crosstalk ≥ 70 dB (class 1 and class 2).

**Tests:**
- `test_crosstalk_multi_channel`: Apply signal to channel 1, measure on channel 2; assert signal is ≥ 70 dB below upper boundary of linear operating range.

---

### 4.15 Display Resolution and Range

**Specification (§5.18):**
- Display resolution: ≤ 0.1 dB.
- Display range: ≥ 60 dB.

**Tests:**
- `test_display_resolution`: Verify that incremental steps in output/reported levels are in multiples of 0.1 dB.
- `test_display_range`: Verify the display covers at least 60 dB.

---

### 4.16 Environmental Influence (Software-Relevant Parts)

**Specification (§6.2–6.4):**
- Static pressure 85–108 kPa: deviation ≤ ±0.4 dB (class 1), ±0.7 dB (class 2).
- Static pressure 65–85 kPa: ±0.9 dB (class 1), ±1.6 dB (class 2).
- Temperature −10 °C to +50 °C (class 1) or 0–40 °C (class 2): deviation ≤ ±0.5 dB (class 1), ±1.0 dB (class 2).
- Humidity 25–90 %: deviation ≤ ±0.5 dB (class 1), ±1.0 dB (class 2).

If the SLM implements any pressure, temperature, or humidity compensation algorithms, test those corrections:

**Tests:**
- `test_pressure_correction_algorithm`: If pressure correction is implemented, verify the corrected response stays within ±0.4 dB over 85–108 kPa.
- `test_temperature_correction_algorithm`: If implemented, verify within ±0.5 dB over the operating temperature range.

---

## 5. IEC 61672-2 — Pattern Evaluation Test Procedures

IEC 61672-2 describes the *test methods* used by accreditation laboratories, rather than adding new numerical specifications. However, it prescribes acceptance criteria identical to those in IEC 61672-1 and specifies alternative test procedures to choose from for software SLM validation:

**Key test procedures applicable to a software SLM:**

### 5.1 Frequency Weighting Tests with Electrical Signals (§9.5)

Two alternative procedures:
1. **Variable input level** (§9.5.2): Adjust input level at each frequency so that the output equals a fixed reference. Record deviation.
2. **Constant input level** (§9.5.3): Apply fixed input level at each frequency; the indication directly gives the frequency response deviation.

**Tests:**
- `test_freq_weighting_electrical_method1`: Implement procedure 1 to verify A-weighting.
- `test_freq_weighting_electrical_method2`: Implement procedure 2 as a cross-check.

### 5.2 Verification of Adjustment Data at Calibration Check Frequency (§9.2)

The difference between measured adjustment data and the Instruction Manual value must not exceed ±0.3 dB (§5.2.5 of IEC 61672-1).

**Tests:**
- `test_calibration_check_freq_adjustment`: At calibration check frequency (160–1250 Hz range), verify the displayed level after applying the nominal calibration adjustment is within ±0.3 dB.

### 5.3 Windscreen Correction (§9.3 and Table 1 of 61672-1)

The difference between measured and stated windscreen correction must not exceed acceptance limits of Table 1:
- 0.063–2 kHz: ±0.5 dB (class 1 and 2)
- 2–8 kHz: ±0.8 dB
- 8–12.5 kHz: ±1.0 dB (class 1 only)
- 12.5–16 kHz: ±1.5 dB (class 1 only)

**Tests (if windscreen correction is software-implemented):**
- `test_windscreen_correction_accuracy`: For each tabulated frequency band, compare measured windscreen correction to stated value; assert within Table 1 limits.

---

## 6. IEC 61672-3 — Periodic Test Requirements

Periodic tests are a reduced subset of pattern evaluation tests. For a software SLM, the following are the most directly testable:

### 6.1 Calibration Check (§10)

Apply sound calibrator at calibration check frequency; verify indication. Environmental correction must be applied.

**Tests:**
- `test_periodic_calibration_check`: Simulate calibrator signal at nominal level; verify SLM indication is within ±0.3 dB after corrections.

### 6.2 Self-Generated Noise (§11)

Report self-generated noise level; this is for information only and is not used to assess conformance.

**Tests:**
- `test_self_generated_noise_reported`: Verify SLM reports a valid self-noise level value (informational, not pass/fail on value itself).

### 6.3 Acoustical Signal Frequency Weighting Test (§12)

Apply known frequency to SLM through coupler or free-field at frequencies specified by the standard. Minimum required frequencies:
- 125 Hz, 1 kHz, and 8 kHz (from §5.3.5.3 of IEC 61672-1, referenced in IEC 61672-3).
- Full sweep at 1/3-octave intervals.

**Tests:**
- `test_periodic_freq_weighting_minimum_set`: Test A-weighting at 125 Hz, 1 kHz, 8 kHz minimum. Assert within class 1 or class 2 limits from Table 3 of IEC 61672-1.

### 6.4 Level Linearity Check (§13 of IEC 61672-3)

At 1 kHz, verify level linearity over the linear operating range on the reference level range.

**Tests:**
- `test_periodic_level_linearity_1khz`: Sweep input at 1 kHz; assert deviation ≤ ±0.8 dB (class 1).

---

## 7. Referenced Standards Summary (Further Test Implications)

| Standard | What it specifies | What to test |
|---|---|---|
| **IEC 60942** | Sound calibrator classes (class 1: SPL accuracy ±0.3 dB; frequency accuracy ±1 Hz at 1 kHz) | If software simulates calibration, verify calibration procedure absorbs ±0.3 dB offset correctly |
| **IEC 61183** | Random-incidence calibration using diffuse-field method | If random-incidence response is implemented, test against directivity index table |
| **IEC 62585** | Method to determine free-field corrections; maximum-permitted uncertainties for correction data | Verify correction data returned by SLM matches specifications; maximum ±0.3 dB for calibration check frequency |
| **IEC 61094-6** | Electrostatic actuators for frequency response testing | No software test; physical calibration equipment |
| **ISO/IEC Guide 98-3 (GUM)** | Method to determine 95 % coverage uncertainty | Verify that uncertainty estimates in conformance assessments use 95 % coverage intervals |
| **ISO/IEC Guide 98-4** | Conformance assessment protocol | Implement dual-criterion check: (a) deviation within acceptance limit AND (b) uncertainty within max-permitted uncertainty |
| **IEC 61000-4-2 / 61000-6-1/2** | EMC / ESD immunity | Hardware only; not directly testable in software SLM |
| **CISPR 22** | RF emission limits | Hardware only |

---

## 8. Maximum-Permitted Expanded Uncertainties (IEC 61260-1 Annex B)

These define the maximum laboratory measurement uncertainty allowed for conformance tests. They are constraints on the *test infrastructure*, not on the SLM output itself — but are relevant if the Python SLM reports uncertainty estimates:

| Requirement | Max permitted uncertainty |
|---|---|
| Frequency of input signal | 0.01 % |
| Input signal level | 0.10 dB |
| Output signal level (within 40 dB of upper limit) | 0.15 dB |
| Output signal level (>40 dB below upper limit) | 0.25 dB |
| Relative attenuation ΔA ≤ 2 dB | 0.20 dB |
| Relative attenuation 2 < ΔA ≤ 40 dB | 0.30 dB |
| Relative attenuation ΔA > 40 dB | 0.50 dB |
| Effective bandwidth deviation | 0.20 dB |
| Level linearity (within 40 dB of upper limit) | 0.20 dB |
| Level linearity (>40 dB below upper limit) | 0.35 dB |
| Time-invariant operation | 0.20 dB |
| Summation of output signals | 0.20 dB |
| Filter decay time | 10 % of indicated value |
| Influence of temperature/humidity | 0.15 dB |

---

## 9. Recommended Test Implementation Structure

```
tests/
├── test_61260_1_filters/
│   ├── test_frequency_math.py          # G, f_r, f_m, f_1, f_2, B_r
│   ├── test_nominal_frequencies.py     # Rounding, Table E.1
│   ├── test_relative_attenuation.py    # Table 1 acceptance limits (class 1 + 2)
│   ├── test_effective_bandwidth.py     # ΔB acceptance limits
│   ├── test_level_linearity.py         # ±0.5 dB / ±0.7 dB bounds
│   ├── test_time_invariant.py          # Swept-sine test, Formula 17
│   └── test_output_summation.py        # Adjacent filter summation
│
├── test_61672_1_slm/
│   ├── test_frequency_weightings.py    # A, C, Z vs Table 3, class 1 + 2
│   ├── test_time_weightings.py         # F (τ=0.125s), S (τ=1s), decay rates
│   ├── test_toneburst.py               # Table 4, Formulas 7 + 8
│   ├── test_repeated_tonebursts.py     # Formula 9
│   ├── test_level_linearity.py         # ±0.8 dB / ±0.3 dB incremental
│   ├── test_overload.py                # Timing, latching, half-cycle symmetry
│   ├── test_under_range.py             # Trigger and duration
│   ├── test_c_peak.py                  # Table 5 reference differences
│   ├── test_stability.py               # 30 min drift, 5 min high-level
│   ├── test_laeq_sel.py                # Equations 3–8
│   ├── test_output_match.py            # ±0.1 dB display vs digital out
│   └── test_crosstalk.py               # ≥70 dB isolation
│
├── test_61672_2_pattern/
│   ├── test_calibration_check.py       # ±0.3 dB after calibration
│   └── test_windscreen_correction.py   # Table 1 correction accuracy
│
├── test_61672_3_periodic/
│   ├── test_periodic_calibration.py    # Periodic calibration check
│   ├── test_periodic_freq_weighting.py # Min. set: 125 Hz, 1 kHz, 8 kHz
│   └── test_periodic_level_linearity.py
│
└── conftest.py                         # Shared fixtures (signal generators,
                                        # tolerance helpers, class 1/2 params)
```

---

## 10. Key Design Decisions for the SLM Implementation

Based on the standards, the following design choices directly determine which tests apply:

1. **Class 1 vs. Class 2**: Tighter limits throughout; class 1 also requires C-weighting and has stricter temperature range (−10 to +50 °C).
2. **Type**: Time-weighting (F/S only), integrating-averaging (L_Aeq), or integrating (L_AE, L_Aeq). Each type has different minimum required outputs.
3. **Number of level ranges**: If >1, overlap requirements apply.
4. **Multi-channel**: Crosstalk specification applies.
5. **Filter bandwidths supported** (IEC 61260-1): 1/1 octave, 1/3 octave, and/or narrower; all must individually pass attenuation shape tests.
6. **Digital implementation**: Anti-alias filter requirements apply (IEC 61260-1 §5.15).
