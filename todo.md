# Todo

## Open
- [ ] #2 **Real-time audio controller** — no soundcard/JACK/ALSA controller. `calibrate()` raises `NotImplementedError` everywhere.
- [ ] #4 **XL2 parser: missing Time section** — `util/xl2.py` `XL2_SLM_File` does not parse the Time section in `123_Report` files.
- [ ] #5 **README.md** — update to reflect current architecture (Engine, Bus, Reporter, plugins, meters)
- [ ] #6 **LICENSE** — add/update license file
- [ ] #19 **Implement LE** (exposure level) according to IEC 61672
- [ ] #18 **Test conformance with standards IEC 61260 and IEC 61672** — requirements summarized in `notes/SLM_Conformance_Test_Report.md`

## Optional
- [ ] #17 **FFT-based A-weighting (optional improvement)** — replace the bilinear-transform IIR in `PluginAWeighting` with a frequency-domain analytical A-weighting to reduce broadband LAeq error from −0.17 dB to ±0.05 dB vs XL2. Requires overlap-add block processing; incompatible with current real-time time-weighted plugins (LASmax etc.) on the same bus without a parallel path. Pre-requisite: split the bus into a Leq-only FFT path and a time-weighting IIR path.

## Done
- [x] #1 **Declarative measurement configuration** — `slm/assembly.py` (`parse_metric`, `build_chain`), `slm/config.py` (`SLMConfig` + TOML I/O), `slm/cli.py` (sensitivity helpers, calibration, `run_measurement`, `SLMShell`), `slm/__main__.py` (`python -m slm`). 154 tests pass.
- [x] #16 **Calibration routine** — `calibrate_sensitivity(wav_path, cal_level)` in `slm/cli.py`; also available as `python -m slm --calibrate --file ... --cal-level 94`.
- [x] #3 **Tighten broadband test tolerance below 0.18 dB** — Root cause identified: bilinear-transform IIR A-weighting (pyoctaveband) over-attenuates above 5 kHz (−0.54 dB at 8 kHz, −6.43 dB at 16 kHz); for broadband signals this causes −0.22 dB systematic LAeq error. FFT-based analytical computation confirms filter is sole culprit (gives ±0.07 dB vs XL2). `TOLERANCE_DB = 0.18` is the practical limit of the current IIR architecture at fs=48 kHz. Root cause documented in `tests/test_xl2_broadband.py`. FFT-based fix tracked in #17.
- [x] #7 Fit XL2 AC-coupling HPF cutoff `fc` — broadband Z scan: `PluginHPF(fc=5.0, order=1)` reduces LZeq error from ~0.27 dB → 0.009 dB; Z tests switched to `_PluginXL2Z = partial(PluginHPF, fc=5.0)`; tolerance unified to ±0.2 dB
- [x] #8 Switch to official `pyoctaveband==1.2.1` from PyPI
- [x] #9 `PluginHPF` added to `slm/frequency_weighting.py` (Butterworth HPF, parametrized fc/order)
- [x] #10 `test_xl2_rta.py` — octave band LZeq per-band vs XL2 RTA log (SLM_005), ±0.5 dB
- [x] #11 Time weighting decay rates: Fast=34.7 dB/s, Slow=4.3 dB/s (IEC 61672-1 §5.8)
- [x] #12 Reporter redesign: broadband/band-split split, 4 CSV files, console printing
- [x] #13 Nominal mid-band frequencies (IEC 61260-1 Annex E) via pyoctaveband fork
- [x] #14 Remove dead code: PluginTimeAveraging, ReadMode, commented blocks
- [x] #15 Structured Engine output — Reporter now retrieves and writes results
