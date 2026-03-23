# Todo

## Open
- [ ] #2 **Real-time audio controller** — no soundcard/JACK/ALSA controller. make sure that calibrate also works with realtime signal. maybe stop, once a stable tone was seen and averaged for long enough?
- [ ] #5 **README.md** — update to reflect current architecture (Engine, Bus, Reporter, plugins, meters)
- [ ] #6 **LICENSE** — add/update license file
- [ ] #20 **Add `__init__.py` re-exports for `slm/io/` and `slm/app/`** — flatten public API so callers can use e.g. `from slm.io import FileController` instead of the full submodule path


## Optional
- [ ] #17 **FFT-based A-weighting (optional improvement)** — replace the bilinear-transform IIR in `PluginAWeighting` with a frequency-domain analytical A-weighting to reduce broadband LAeq error from −0.17 dB to ±0.05 dB vs XL2. Requires overlap-add block processing; incompatible with current real-time time-weighted plugins (LASmax etc.) on the same bus without a parallel path. Pre-requisite: split the bus into a Leq-only FFT path and a time-weighting IIR path.
- [ ] #23 **use different filter for frequency-weighting** - use different design method (pre-warping, impulse invariance or so)
- 
## Done
- [x] #21 **Redo calibration routine** — `PluginBandpass` added; core `slm/calibration.py` is controller-agnostic; `calibrate_from_file` in `slm/app/cli.py`; `--cal-freq` CLI flag; 2 new unit tests.
- [x] #24 **conformance tests should record how "well" they passed** — `report=True` pattern on all conformance test methods; `scripts/conformance_report.py` calls them directly and prints margin tables
- [x] #22 **tidy up tests/ folder** - add sub folders and group tests together
- [x] #18 **Test conformance with standards IEC 61260 and IEC 61672** — requirements summarized in `notes/SLM_Conformance_Test_Report.md`
- [x] #4 **XL2 parser: missing Time section** — `_SectionTime` added to `util/xl2.py`; parses `Start`/`End` as `datetime` objects in all file types that carry `# Time` (123_Log, RTA_3rd_Log, RTA_3rd_Report, RTA_Oct_Log, RTA_Oct_Report); 6 new tests in `tests/test_xl2_parser.py`; 190 tests pass.
- [x] #19 **Implement LE** (exposure level) — `LEAccumulator` + `LEMovingMeter` in `slm/meter.py`; parser/builder extended in `slm/assembly.py` (`LAE`, `LCE`, `LZE`, window/band variants); engine end-of-file snapshot fix; 184 tests pass.
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
