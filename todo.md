# Todo

## Open
- [ ] #1 **Declarative measurement configuration** — user requests metrics by name (e.g. `LAeq_dt`, `LAFmax`, `LZeq_dt:octave:63-4000`); engine auto-assembles the plugin chain, shares upstream nodes, wires reporter. Possible entry points: `engine.request("LAeq_dt", "LAFmax")` or a CLI `open-spl --measure LAeq_dt LAFmax`. Open questions: how to express RTA requests; where sensitivity/dt/limits live; wrap vs replace manual API.
- [ ] #2 **Real-time audio controller** — no soundcard/JACK/ALSA controller. `calibrate()` raises `NotImplementedError` everywhere.
- [ ] #3 **Tighten broadband test tolerance below 0.18 dB** — current worst offenders are LAeq on SLM_003 and SLM_004 (~0.17 dB, systematic, same magnitude on both → likely sensitivity calibration offset or A-weighting filter accuracy). LCpeak on SLM_000 is 0.127 dB. Need to identify root cause before shipping. Current tolerance: `TOLERANCE_DB = 0.18` in `tests/test_xl2_broadband.py`.
- [ ] #4 **XL2 parser: missing Time section** — `util/xl2.py` `XL2_SLM_File` does not parse the Time section in `123_Report` files.
- [ ] #5 **README.md** — update to reflect current architecture (Engine, Bus, Reporter, plugins, meters)
- [ ] #6 **LICENSE** — add/update license file

## Done
- [x] #7 Fit XL2 AC-coupling HPF cutoff `fc` — broadband Z scan: `PluginHPF(fc=5.0, order=1)` reduces LZeq error from ~0.27 dB → 0.009 dB; Z tests switched to `_PluginXL2Z = partial(PluginHPF, fc=5.0)`; tolerance unified to ±0.2 dB
- [x] #8 Switch to official `pyoctaveband==1.2.1` from PyPI
- [x] #9 `PluginHPF` added to `slm/frequency_weighting.py` (Butterworth HPF, parametrized fc/order)
- [x] #10 `test_xl2_rta.py` — octave band LZeq per-band vs XL2 RTA log (SLM_005), ±0.5 dB
- [x] #11 Time weighting decay rates: Fast=34.7 dB/s, Slow=4.3 dB/s (IEC 61672-1 §5.8)
- [x] #12 Reporter redesign: broadband/band-split split, 4 CSV files, console printing
- [x] #13 Nominal mid-band frequencies (IEC 61260-1 Annex E) via pyoctaveband fork
- [x] #14 Remove dead code: PluginTimeAveraging, ReadMode, commented blocks
- [x] #15 Structured Engine output — Reporter now retrieves and writes results
