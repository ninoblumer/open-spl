# Running the tests

The test suite lives under `tests/` (unit tests plus IEC 61260-1 / 61672-1 conformance and
XL2 hardware-comparison integration tests). The IEC 61260-1 filter conformance tests are
parametric over filter bandwidth — they run for octave and 1/3-octave banks (extend
`BANDWIDTHS` in `tests/iec61260/test_61260_1_filters.py` to cover others). Run it with pytest:

```bash
python -m pytest tests/ -q
```

## Slow tests

A few long-running conformance tests are marked `@pytest.mark.slow` and are **skipped by
default**. Pass `--slow` to include them:

```bash
python -m pytest tests/ -q --slow
```

## Conformance margin report

`scripts/conformance_report.py` drives the IEC 61672-1 / IEC 61260-1 test suites and prints
a formatted table of measured values, limits, and margins from each limit — a quick way to
see how much headroom the implementation has against every conformance requirement:

```bash
python scripts/conformance_report.py
python scripts/conformance_report.py --no-color      # disable ANSI colour
python scripts/conformance_report.py --precision 4   # decimal places (default: 3)
python scripts/conformance_report.py --slow          # include §5.14/§5.15 stability tests
```
