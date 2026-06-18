"""IEC 61260-1:2014 — octave-band and fractional-octave-band filter conformance
tests (class 1).

Covers:
  §5.2  G = 10^(3/10) — base-10 octave frequency ratio
  §5.3  f_r = 1000 Hz — reference frequency
  §5.4  exact mid-band frequencies: f_m = f_r × G^(x/b)  (odd denominator b)
  §5.6  band-edge frequencies: f_1 = f_m × G^(-1/2b), f_2 = f_m × G^(+1/2b)
  §5.10 Table 1 / Table F.1 — relative attenuation acceptance limits, class 1
  §5.12 effective bandwidth deviation ΔB = 10·log₁₀(B_e/B_r) within ±0.4 dB

These tests are *parametric over filter bandwidth*.  Every test is run once per
entry in ``BANDWIDTHS`` (octave and one-third-octave by default); adding another
:class:`FilterConfig` to that list extends coverage to a new bandwidth with no
other change.

How the bandwidth scaling works (IEC 61260-1:2014):
  - Mid-band frequencies, band edges and the reference effective bandwidth all
    carry the bandwidth designator 1/b explicitly (§5.4, §5.6, §5.11).
  - The relative-attenuation *acceptance limits* (the dB column of Table 1) are
    identical for every bandwidth.  Only the normalized frequencies at which
    they apply move: a finite octave breakpoint Ω_h(1/1) maps to bandwidth 1/b
    via Formula (9):

        Ω_h(1/b) = 1 + (G^(1/2b) − 1)/(G^(1/2) − 1) · (Ω_h(1/1) − 1)

    and the low-frequency mirror by Formula (10): Ω_l(1/b) = 1/Ω_h(1/b).
    For b = 1 this is the identity, recovering Table 1; for b = 3 it reproduces
    the one-third-octave breakpoints tabulated in Annex F, Table F.1.

Frequency response is evaluated analytically via scipy.signal.sosfreqz on the
filter SOS coefficients — no audio processing or realtime simulation required.

Table 1 reference (class 1, relative attenuation in dB), in terms of the octave
normalized frequency Ω = f/f_m:

    Interior pass-band (G^{-3/8} … G^{+3/8}):
      G^0   : −0.4 to +0.4 dB
      G^±1/8: −0.4 to +0.5 dB
      G^±1/4: −0.4 to +0.7 dB
      G^±3/8: −0.4 to +1.4 dB

    Stop-band (beyond G^{±1/2}):
      G^±1  : ≥ 16.6 dB
      G^±2  : ≥ 40.5 dB
      G^±3  : ≥ 60.0 dB
      G^±4  : ≥ 70.0 dB
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest
from scipy import signal as sig

from slm.octave_band import PluginOctaveBand

# ---------------------------------------------------------------------------
# IEC 61260-1:2014 constants
# ---------------------------------------------------------------------------

G   = 10 ** (3 / 10)   # §5.2: base-10 octave frequency ratio ≈ 1.995 26
f_r = 1000.0            # §5.3: reference frequency (Hz)

SAMPLERATE = 48_000
BLOCKSIZE  = 4_096

# ---------------------------------------------------------------------------
# Table 1, class 1 — relative attenuation limits (§5.10), keyed by the OCTAVE
# normalized frequency Ω = f/f_m (= G^exponent).  These dB limits are
# bandwidth-independent; only the normalized frequencies are remapped per
# Formula (9)/(10) for fractional-octave bandwidths (see _omega_for_bandwidth).
#
# ΔA(Ω) = gain_dB(f_m) − gain_dB(Ω·f_m), so positive = attenuation.
# Pass-band:  lo ≤ ΔA ≤ hi  (negative lo allows slight gain above centre)
# Stop-band:  ΔA ≥ min  (no upper limit on attenuation)
# ---------------------------------------------------------------------------

_PASSBAND_CL1: dict[float, tuple[float, float]] = {
    # octave G-exponent : (min_ΔA, max_ΔA)
    -3/8: (-0.4, +1.4),
    -1/4: (-0.4, +0.7),
    -1/8: (-0.4, +0.5),
     0.0: (-0.4, +0.4),
    +1/8: (-0.4, +0.5),
    +1/4: (-0.4, +0.7),
    +3/8: (-0.4, +1.4),
}

_STOPBAND_CL1: dict[int, float] = {
    # octave G-exponent : min ΔA (dB)
    -4: 70.0,
    -3: 60.0,
    -2: 40.5,
    -1: 16.6,
    +1: 16.6,
    +2: 40.5,
    +3: 60.0,
    +4: 70.0,
}

# Band-edge attenuation window.
# NOTE: SELF-IMPOSED limit — NOT a literal IEC acceptance limit.  Table 1 has a
# discontinuity at the band edge (G^{±1/2}): just *outside* the min is +1.2 dB
# (G^{±1/2}-ε row), just *inside* the max is +5.3 dB (G^{±1/2}+ε row).  The
# standard specifies limits at those ε-offset frequencies, not a single band at
# the edge itself.  We evaluate at the exact edge and require [1.2, 5.3] dB as a
# pragmatic check spanning the discontinuity; it is a derived interpretation, so
# this section is intentionally excluded from scripts/conformance_report.py.
_EDGE_LIMIT_LO_CL1 = 1.2
_EDGE_LIMIT_HI_CL1 = 5.3

DELTA_B_LIMIT_CL1 = 0.4   # ±dB (§5.12 class 1)

# ---------------------------------------------------------------------------
# Annex E / Table E.1 — nominal (labelled) mid-band frequencies (Hz), keyed by
# the one-third-octave index x where the exact mid-band is f_m = f_r·G^(x/3)
# = 1000·10^(x/10).  Octave bands fall on every third index.  The nominal value
# depends only on x, so it is shared by octave and one-third-octave banks.
# ---------------------------------------------------------------------------

_NOMINAL_BY_INDEX: dict[int, float] = {
    -16: 25.0,    -15: 31.5,    -14: 40.0,    -13: 50.0,
    -12: 63.0,    -11: 80.0,    -10: 100.0,    -9: 125.0,
     -8: 160.0,    -7: 200.0,    -6: 250.0,    -5: 315.0,
     -4: 400.0,    -3: 500.0,    -2: 630.0,    -1: 800.0,
      0: 1000.0,    1: 1250.0,    2: 1600.0,    3: 2000.0,
      4: 2500.0,    5: 3150.0,    6: 4000.0,    7: 5000.0,
      8: 6300.0,    9: 8000.0,   10: 10000.0,  11: 12500.0,
     12: 16000.0,  13: 20000.0,
}


# ---------------------------------------------------------------------------
# Filter-bank configuration — one per bandwidth under test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterConfig:
    """A bandwidth configuration for the conformance suite.

    *b* is the denominator of the bandwidth designator 1/b (1 = octave,
    3 = one-third-octave).  *limits* is the (low, high) frequency span passed to
    the filter bank.
    """
    b: int
    limits: tuple[float, float]
    samplerate: int = SAMPLERATE

    @property
    def name(self) -> str:
        return "octave" if self.b == 1 else f"1_{self.b}-oct"


OCTAVE       = FilterConfig(b=1, limits=(63, 8000))
THIRD_OCTAVE = FilterConfig(b=3, limits=(50, 10000))

# Bandwidths exercised by every parametric test.  Append a FilterConfig here to
# extend conformance coverage to another bandwidth.
BANDWIDTHS: list[FilterConfig] = [OCTAVE, THIRD_OCTAVE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_bus(samplerate: int = SAMPLERATE, blocksize: int = BLOCKSIZE,
              sensitivity: float = 1.0, dt: float = 1.0) -> types.SimpleNamespace:
    mock = types.SimpleNamespace(
        samplerate=samplerate, blocksize=blocksize,
        sensitivity=sensitivity, dt=dt,
        width=1, get_chain=lambda: [],
    )
    mock.bus = mock
    return mock


def _make_plugin(limits: tuple[float, float] = (63, 8000),
                 fraction: int = 1,
                 samplerate: int = SAMPLERATE) -> PluginOctaveBand:
    bus = _mock_bus(samplerate=samplerate)
    return PluginOctaveBand(input=bus, limits=limits, bands_per_oct=fraction)


def _omega_for_bandwidth(omega_octave: float, b: int) -> float:
    """Map an octave normalized breakpoint Ω = f/f_m to bandwidth designator 1/b.

    Implements IEC 61260-1:2014 Formula (9) (high side, Ω ≥ 1) and Formula (10)
    (low side, Ω < 1).  Identity for b = 1.
    """
    if b == 1:
        return omega_octave
    factor = (G ** (1 / (2 * b)) - 1) / (G ** 0.5 - 1)
    if omega_octave >= 1.0:
        return 1.0 + factor * (omega_octave - 1.0)              # Formula (9)
    # Low side: map the mirror high-side breakpoint, then invert (Formula 10).
    omega_h = 1.0 + factor * (1.0 / omega_octave - 1.0)
    return 1.0 / omega_h


def _nominal_label_to_hz(label: str) -> float:
    """Parse a nominal-frequency label (e.g. '63', '31.5', '1k', '1.25k') to Hz."""
    s = label.strip().lower()
    return float(s[:-1]) * 1000.0 if s.endswith("k") else float(s)


def _reference_effective_bw(b: int) -> float:
    """Normalized reference effective bandwidth B_r = (1/b)·ln(G) (§5.11, Formula 15)."""
    return math.log(G) / b


def _gain_db(sos: np.ndarray, freqs_hz, samplerate: int) -> np.ndarray:
    """Filter gain in dB at each frequency in *freqs_hz* (array or scalar)."""
    _, h = sig.sosfreqz(sos, worN=np.atleast_1d(np.asarray(freqs_hz, float)),
                        fs=samplerate)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))


def _rel_atten(sos: np.ndarray, f_m: float, f_test: float,
               samplerate: int) -> float:
    """Relative attenuation ΔA = gain_dB(f_m) − gain_dB(f_test)  (positive = attenuated)."""
    gains = _gain_db(sos, [f_m, f_test], samplerate)
    return float(gains[0] - gains[1])


def _effective_bw_deviation(sos: np.ndarray, f_m: float, b: int, samplerate: int,
                            n_points: int = 8192) -> float:
    """ΔB = 10·log₁₀(B_e / B_r) where B_r = (1/b)·ln(G) (§5.12).

    B_e is computed by numeric log-trapezoidal integration of
    H²(f) / (H²(f_m) · f) df over [f_low, f_Nyquist).
    The change of variables Ω = f/f_m gives a dimensionless result matching
    the B_r reference.
    """
    f_nyq = samplerate / 2.0
    # Lower limit well into the stop-band (8 octaves below centre)
    f_low = max(f_m * G ** -8, 0.5)
    freqs  = np.geomspace(f_low, f_nyq * 0.9999, n_points)
    _, h   = sig.sosfreqz(sos, worN=freqs, fs=samplerate)
    h2     = np.abs(h) ** 2
    h2_c   = float(np.abs(sig.sosfreqz(sos, worN=[f_m], fs=samplerate)[1][0])) ** 2
    B_e    = float(np.trapezoid(h2 / (h2_c * freqs), freqs))
    B_r    = _reference_effective_bw(b)
    return 10.0 * math.log10(B_e / B_r)


# ---------------------------------------------------------------------------
# Shared filter banks — one per configuration, instantiated once and cached
# ---------------------------------------------------------------------------

_fb_cache: dict = {}


def _filterbank(cfg: FilterConfig = OCTAVE) -> tuple[list[float], list[np.ndarray]]:
    """Return (center_freqs, sos_list) for *cfg*'s filter bank (cached)."""
    entry = _fb_entry(cfg)
    return entry["centers"], entry["sos"]


def _nominal_labels(cfg: FilterConfig = OCTAVE) -> list[str]:
    """Return the bank's nominal (labelled) mid-band frequencies for *cfg* (cached)."""
    return _fb_entry(cfg)["nominal"]


def _fb_entry(cfg: FilterConfig) -> dict:
    key = (cfg.b, cfg.limits, cfg.samplerate)
    if key not in _fb_cache:
        plugin = _make_plugin(limits=cfg.limits, fraction=cfg.b,
                              samplerate=cfg.samplerate)
        fb     = plugin._filter_bank
        _fb_cache[key] = {
            "centers": [float(f) for f in fb.freq],
            "sos":     [fb.sos[i] for i in range(fb.num_bands)],
            "nominal": list(fb.nominal_freq),
        }
    return _fb_cache[key]


# ---------------------------------------------------------------------------
# Parametric test cases — built once per bandwidth at collection time
# ---------------------------------------------------------------------------

class AttenCase(NamedTuple):
    """A single relative-attenuation breakpoint test."""
    cfg: FilterConfig
    band_idx: int
    f_m: float
    f_test: float
    lo: float
    hi: float | None     # None = stop-band (no upper limit)
    label: str


class EdgeCase(NamedTuple):
    """A single band-edge attenuation test (side = -1 lower, +1 upper)."""
    cfg: FilterConfig
    band_idx: int
    f_m: float
    side: int
    label: str


class BandCase(NamedTuple):
    """A single per-band test (effective bandwidth / mid-band frequency)."""
    cfg: FilterConfig
    band_idx: int
    f_m: float
    label: str


def passband_cases(cfg: FilterConfig) -> list[AttenCase]:
    centers, _ = _filterbank(cfg)
    cases = []
    for band_idx, f_m in enumerate(centers):
        for exp, (lo, hi) in _PASSBAND_CL1.items():
            f_test = f_m * _omega_for_bandwidth(G ** exp, cfg.b)
            cases.append(AttenCase(cfg, band_idx, f_m, f_test, lo, hi,
                                   f"{int(round(f_m))} Hz G^{exp:+.3f}"))
    return cases


def stopband_cases(cfg: FilterConfig) -> list[AttenCase]:
    centers, _ = _filterbank(cfg)
    f_nyq = cfg.samplerate / 2.0
    cases = []
    for band_idx, f_m in enumerate(centers):
        for exp, min_da in _STOPBAND_CL1.items():
            f_test = f_m * _omega_for_bandwidth(G ** exp, cfg.b)
            if f_test <= 0.5 or f_test >= f_nyq:
                continue   # out-of-range stop-band frequency
            cases.append(AttenCase(cfg, band_idx, f_m, f_test, min_da, None,
                                   f"{int(round(f_m))} Hz G^{exp:+d}"))
    return cases


def band_edge_cases(cfg: FilterConfig) -> list[EdgeCase]:
    centers, _ = _filterbank(cfg)
    f_nyq = cfg.samplerate / 2.0
    cases = []
    for band_idx, f_m in enumerate(centers):
        for side, name in ((-1, "lower"), (+1, "upper")):
            f_edge = f_m * G ** (side / (2 * cfg.b))
            if f_edge < 0.5 or f_edge >= f_nyq:
                continue
            cases.append(EdgeCase(cfg, band_idx, f_m, side,
                                  f"{int(round(f_m))} Hz {name} edge"))
    return cases


def bandwidth_cases(cfg: FilterConfig) -> list[BandCase]:
    centers, _ = _filterbank(cfg)
    return [BandCase(cfg, i, f_m, f"{int(round(f_m))} Hz")
            for i, f_m in enumerate(centers)]


def _build(builder) -> tuple[list, list[str]]:
    """Flatten a case-builder over all BANDWIDTHS into (cases, ids)."""
    cases, ids = [], []
    for cfg in BANDWIDTHS:
        for case in builder(cfg):
            cases.append(case)
            ids.append(f"{cfg.name}-{case.label.replace(' ', '')}")
    return cases, ids


_PB_PARAMS,   _PB_IDS   = _build(passband_cases)
_SB_PARAMS,   _SB_IDS   = _build(stopband_cases)
_EDGE_PARAMS, _EDGE_IDS = _build(band_edge_cases)
_BW_PARAMS,   _BW_IDS   = _build(bandwidth_cases)


# ---------------------------------------------------------------------------
# §5.2–5.4: Frequency math
# ---------------------------------------------------------------------------

class TestOctaveFrequencyMath:
    """§5.2–5.4 — G ratio, reference frequency, and mid-band frequency formula."""

    def test_g_ratio(self):
        """G must equal 10^(3/10) within floating-point precision (§5.2)."""
        assert abs(G - 10 ** (3 / 10)) < 1e-12

    def test_reference_frequency(self):
        """Reference frequency f_r = 1000 Hz exactly (§5.3)."""
        assert f_r == 1000.0

    @pytest.mark.parametrize("omega_oct, expected", [
        # Octave breakpoint Ω → one-third-octave Ω, per Annex F (Table F.1 values).
        (G ** (+1/8), 1.026_67),
        (G ** (-1/8), 0.974_02),
        (G ** (+1/2), 1.122_02),   # upper band edge
        (G ** (-1/2), 0.891_25),   # lower band edge
    ], ids=["G^+1/8", "G^-1/8", "G^+1/2", "G^-1/2"])
    def test_third_octave_breakpoint_mapping(self, omega_oct: float, expected: float):
        """Formula (9)/(10) reproduce the Annex F Table F.1 1/3-octave breakpoints."""
        got = _omega_for_bandwidth(omega_oct, b=3)
        assert abs(got - expected) < 5e-5, (
            f"Ω(1/3) = {got:.6f}, expected ≈ {expected} (Table F.1)"
        )

    @pytest.mark.parametrize("cfg", BANDWIDTHS, ids=[c.name for c in BANDWIDTHS])
    def test_midband_formula(self, cfg: FilterConfig):
        """All filter-bank centres match f_r × G^(x/b) (§5.4 Formula 2).

        NOTE: IEC 61260-1 defines the exact mid-band frequencies by formula and
        does NOT specify an acceptance tolerance on f_m.  The ±0.01 % bound below
        is a SELF-IMPOSED implementation check (the bank realises the defined
        grid to within rounding), not a standard acceptance limit — which is why
        mid-band frequencies are not reported in scripts/conformance_report.py.
        """
        centers, _ = _filterbank(cfg)
        for f in centers:
            # Nearest grid index x on the f_r·G^(x/b) lattice (odd denominator b).
            x          = round(cfg.b * math.log(f / f_r) / math.log(G))
            f_expected = f_r * G ** (x / cfg.b)
            rel_err    = abs(f - f_expected) / f_expected
            assert rel_err < 1e-4, (   # self-imposed tolerance (see docstring)
                f"{cfg.name}: centre {f:.4f} Hz not on f_r·G^(x/{cfg.b}) grid "
                f"(nearest x={x}: {f_expected:.4f} Hz, rel error {rel_err:.2e})"
            )

    @pytest.mark.parametrize("case", _BW_PARAMS, ids=_BW_IDS)
    def test_nominal_midband_frequency(self, case: BandCase):
        """Each band's labelled nominal frequency matches Annex E / Table E.1 (§5.5)."""
        # Table E.1 index from the exact mid-band: f_m = f_r·G^(x/3) = 1000·10^(x/10).
        x = round(10.0 * math.log10(case.f_m / f_r))
        assert x in _NOMINAL_BY_INDEX, (
            f"{case.cfg.name} band {case.f_m:.2f} Hz → index x={x} outside Table E.1"
        )
        expected = _NOMINAL_BY_INDEX[x]
        got      = _nominal_label_to_hz(_nominal_labels(case.cfg)[case.band_idx])
        assert math.isclose(got, expected, rel_tol=1e-9), (
            f"{case.cfg.name} band (exact {case.f_m:.2f} Hz, index x={x}): "
            f"nominal label = {got:g} Hz, expected {expected:g} Hz (Table E.1)"
        )


# ---------------------------------------------------------------------------
# §5.6: Band-edge frequencies
# ---------------------------------------------------------------------------

class TestOctaveBandEdges:
    """§5.6 — band edges at f_1 = f_m × G^(-1/2b), f_2 = f_m × G^(+1/2b).

    IEC 61260-1 Table 1 specifies a range of [-0.4, +5.3] dB relative
    attenuation just inside each band edge (class 1).  For a Butterworth
    design the attenuation at the band-edge frequency is nominally 3 dB.
    We verify it falls within the [1.2, 5.3] dB window spanning the
    discontinuity at the edge.
    """

    @pytest.mark.parametrize("case", _EDGE_PARAMS, ids=_EDGE_IDS)
    def test_band_edge(self, case: EdgeCase):
        # NOTE: the [1.2, 5.3] dB window is a SELF-IMPOSED limit, not a literal
        # IEC acceptance limit — see the _EDGE_LIMIT_*_CL1 comment above.
        _, sos_list = _filterbank(case.cfg)
        f_edge = case.f_m * G ** (case.side / (2 * case.cfg.b))
        da = _rel_atten(sos_list[case.band_idx], case.f_m, f_edge,
                        case.cfg.samplerate)
        assert _EDGE_LIMIT_LO_CL1 <= da <= _EDGE_LIMIT_HI_CL1, (
            f"{case.cfg.name} band {case.f_m:.1f} Hz {case.label} @ {f_edge:.1f} Hz: "
            f"ΔA = {da:.3f} dB (class 1 at band edge: "
            f"[{_EDGE_LIMIT_LO_CL1}, {_EDGE_LIMIT_HI_CL1}] dB)"
        )


# ---------------------------------------------------------------------------
# §5.10 Table 1 / Table F.1: Relative attenuation
# ---------------------------------------------------------------------------

class TestOctaveRelativeAttenuation:
    """§5.10 Table 1 / Table F.1 — relative attenuation acceptance limits, class 1.

    Pass-band breakpoints are tested at (octave) G^{0, ±1/8, ±1/4, ±3/8}.
    Stop-band breakpoints are tested at (octave) G^{±1, ±2, ±3, ±4} for all
    bands whose test frequency falls within [0.5, Nyquist) Hz.  For
    fractional-octave bandwidths the normalized frequencies are remapped per
    Formula (9)/(10); the dB limits are unchanged.
    """

    @pytest.mark.parametrize("case", _PB_PARAMS, ids=_PB_IDS)
    def test_passband(self, case: AttenCase, report: bool = False) -> None:
        """Interior pass-band: lo ≤ ΔA(Ω) ≤ hi (dB)."""
        _, sos_list = _filterbank(case.cfg)
        da = _rel_atten(sos_list[case.band_idx], case.f_m, case.f_test,
                        case.cfg.samplerate)
        margin = min(da - case.lo, case.hi - da)
        if report:
            return {"label": case.label, "deviation": da,
                    "limit_lo": case.lo, "limit_hi": case.hi, "margin": margin}
        assert case.lo <= da <= case.hi, (
            f"{case.cfg.name} band {case.f_m:.1f} Hz @ {case.f_test:.1f} Hz: "
            f"ΔA = {da:+.4f} dB  (class 1: [{case.lo:+.1f}, {case.hi:+.1f}] dB)"
        )

    @pytest.mark.parametrize("case", _SB_PARAMS, ids=_SB_IDS)
    def test_stopband(self, case: AttenCase, report: bool = False) -> None:
        """Stop-band: ΔA(Ω) ≥ minimum attenuation (dB)."""
        _, sos_list = _filterbank(case.cfg)
        da = _rel_atten(sos_list[case.band_idx], case.f_m, case.f_test,
                        case.cfg.samplerate)
        margin = da - case.lo
        if report:
            return {"label": case.label, "deviation": da,
                    "limit_lo": case.lo, "limit_hi": None, "margin": margin}
        assert da >= case.lo, (
            f"{case.cfg.name} band {case.f_m:.1f} Hz @ {case.f_test:.1f} Hz: "
            f"ΔA = {da:.2f} dB  (class 1 minimum: {case.lo:.1f} dB)"
        )


# ---------------------------------------------------------------------------
# §5.12: Effective bandwidth deviation
# ---------------------------------------------------------------------------

class TestOctaveEffectiveBandwidth:
    """§5.12 — effective bandwidth deviation ΔB = 10·log₁₀(B_e/B_r) ≤ ±0.4 dB.

    B_r = (1/b)·ln(G) is the normalized reference effective bandwidth (§5.11):
    0.690 776 for octave, 0.230 259 for one-third-octave filters.
    B_e is computed numerically from the SOS frequency response.
    """

    @pytest.mark.parametrize("case", _BW_PARAMS, ids=_BW_IDS)
    def test_bandwidth_deviation(self, case: BandCase, report: bool = False) -> None:
        _, sos_list = _filterbank(case.cfg)
        db  = _effective_bw_deviation(sos_list[case.band_idx], case.f_m,
                                      case.cfg.b, case.cfg.samplerate)
        margin = DELTA_B_LIMIT_CL1 - abs(db)
        if report:
            return {"label": case.label, "deviation": db,
                    "limit_lo": -DELTA_B_LIMIT_CL1, "limit_hi": DELTA_B_LIMIT_CL1,
                    "margin": margin}
        assert abs(db) <= DELTA_B_LIMIT_CL1, (
            f"{case.cfg.name} band {case.f_m:.1f} Hz: ΔB = {db:+.4f} dB "
            f"(class 1 limit: ±{DELTA_B_LIMIT_CL1} dB)"
        )
