"""IEC 61260-1:2014 — octave-band filter conformance tests (class 1).

Covers:
  §5.2  G = 10^(3/10) — base-10 octave frequency ratio
  §5.3  f_r = 1000 Hz — reference frequency
  §5.4  exact mid-band frequencies: f_m = f_r × G^x  (b=1, odd denominator)
  §5.6  band-edge frequencies: f_1 = f_m × G^(-1/2), f_2 = f_m × G^(+1/2)
  §5.10 Table 1 — relative attenuation acceptance limits, class 1
  §5.12 effective bandwidth deviation ΔB = 10·log₁₀(B_e/B_r) within ±0.4 dB

Frequency response is evaluated analytically via scipy.signal.sosfreqz on
the filter SOS coefficients — no audio processing or realtime simulation
required.  All 8 standard octave bands (63–8000 Hz) at 48 kHz are tested.

Table 1 reference (class 1 relative attenuation limits):

  Normalised frequency Ω = f/f_m:

    Interior pass-band (G^{-3/8} to G^{+3/8}):
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
# Table 1, class 1 — relative attenuation limits (§5.10)
#
# ΔA(Ω) = gain_dB(f_m) − gain_dB(Ω·f_m), so positive = attenuation.
# Pass-band:  lo ≤ ΔA ≤ hi  (negative lo allows slight gain above centre)
# Stop-band:  ΔA ≥ min  (no upper limit on attenuation)
# ---------------------------------------------------------------------------

_PASSBAND_CL1: dict[float, tuple[float, float]] = {
    # G-exponent : (min_ΔA, max_ΔA)
    -3/8: (-0.4, +1.4),
    -1/4: (-0.4, +0.7),
    -1/8: (-0.4, +0.5),
     0.0: (-0.4, +0.4),
    +1/8: (-0.4, +0.5),
    +1/4: (-0.4, +0.7),
    +3/8: (-0.4, +1.4),
}

_STOPBAND_CL1: dict[int, float] = {
    # G-exponent : min ΔA (dB)
    -4: 70.0,
    -3: 60.0,
    -2: 40.5,
    -1: 16.6,
    +1: 16.6,
    +2: 40.5,
    +3: 60.0,
    +4: 70.0,
}

DELTA_B_LIMIT_CL1 = 0.4   # ±dB (§5.12 class 1)


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


def _gain_db(sos: np.ndarray, freqs_hz, samplerate: int) -> np.ndarray:
    """Filter gain in dB at each frequency in *freqs_hz* (array or scalar)."""
    _, h = sig.sosfreqz(sos, worN=np.atleast_1d(np.asarray(freqs_hz, float)),
                        fs=samplerate)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-300))


def _delta_A(sos: np.ndarray, f_m: float, omega_exp: float,
             samplerate: int) -> float:
    """Relative attenuation ΔA(Ω) = gain_dB(f_m) − gain_dB(G^omega_exp · f_m)."""
    f_test = f_m * G ** omega_exp
    gains  = _gain_db(sos, [f_m, f_test], samplerate)
    return float(gains[0] - gains[1])   # positive = attenuated at f_test


def _effective_bw_deviation(sos: np.ndarray, f_m: float, samplerate: int,
                             n_points: int = 8192) -> float:
    """ΔB = 10·log₁₀(B_e / B_r) where B_r = ln(G) (§5.12).

    B_e is computed by numeric log-trapezoidal integration of
    H²(f) / (H²(f_m) · f) df over [f_low, f_Nyquist).
    The change of variables Ω = f/f_m gives a dimensionless result
    matching the B_r = ln(G) reference.
    """
    f_nyq = samplerate / 2.0
    # Lower limit well into the stop-band (8 octaves below centre)
    f_low = max(f_m * G ** -8, 0.5)
    freqs  = np.geomspace(f_low, f_nyq * 0.9999, n_points)
    _, h   = sig.sosfreqz(sos, worN=freqs, fs=samplerate)
    h2     = np.abs(h) ** 2
    h2_c   = float(np.abs(sig.sosfreqz(sos, worN=[f_m], fs=samplerate)[1][0])) ** 2
    B_e    = float(np.trapezoid(h2 / (h2_c * freqs), freqs))
    B_r    = math.log(G)                # ln(G) ≈ 0.6908
    return 10.0 * math.log10(B_e / B_r)


# ---------------------------------------------------------------------------
# Shared filter bank — instantiated once at collection time
# ---------------------------------------------------------------------------

_fb_cache: dict = {}


def _get_filterbank() -> tuple[list[float], list[np.ndarray]]:
    """Return (center_freqs, sos_list) for the standard 63–8000 Hz filter bank."""
    if not _fb_cache:
        plugin = _make_plugin(limits=(63, 8000), fraction=1)
        fb     = plugin._filter_bank
        _fb_cache["centers"] = list(fb.freq)
        _fb_cache["sos"]     = [fb.sos[i] for i in range(fb.num_bands)]
    return _fb_cache["centers"], _fb_cache["sos"]


def _band_ids() -> list[str]:
    centers, _ = _get_filterbank()
    return [f"{int(round(f))}Hz" for f in centers]


# ---------------------------------------------------------------------------
# Build parametrize lists once (module-level)
# ---------------------------------------------------------------------------

def _passband_params() -> tuple[list, list[str]]:
    centers, _ = _get_filterbank()
    params, ids = [], []
    for band_idx, f_m in enumerate(centers):
        for exp, (lo, hi) in _PASSBAND_CL1.items():
            params.append((band_idx, f_m, exp, lo, hi))
            ids.append(f"band{int(round(f_m))}Hz_G^{exp:+.3f}")
    return params, ids


def _stopband_params() -> tuple[list, list[str]]:
    centers, _ = _get_filterbank()
    f_nyq = SAMPLERATE / 2.0
    params, ids = [], []
    for band_idx, f_m in enumerate(centers):
        for exp, min_da in _STOPBAND_CL1.items():
            f_test = f_m * G ** exp
            if f_test <= 0.5 or f_test >= f_nyq:
                continue   # skip out-of-range stop-band frequencies
            params.append((band_idx, f_m, exp, min_da))
            ids.append(f"band{int(round(f_m))}Hz_G^{exp:+d}")
    return params, ids


_PB_PARAMS, _PB_IDS = _passband_params()
_SB_PARAMS, _SB_IDS = _stopband_params()
_BAND_IDS           = _band_ids()


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

    @pytest.mark.parametrize("x", range(-4, 4),
                             ids=[f"G^{x:+d}" for x in range(-4, 4)])
    def test_midband_formula(self, x: int):
        """Filter-bank centre frequencies match f_r × G^x ± 0.01 % (§5.4 Formula 2)."""
        centers, _ = _get_filterbank()
        f_expected = f_r * G ** x
        diffs      = [abs(f - f_expected) for f in centers]
        best_match = centers[int(np.argmin(diffs))]
        rel_err    = abs(best_match - f_expected) / f_expected
        assert rel_err < 1e-4, (
            f"x={x:+d}: expected {f_expected:.4f} Hz, "
            f"filter bank has {best_match:.4f} Hz (rel error {rel_err:.2e})"
        )


# ---------------------------------------------------------------------------
# §5.6: Band-edge frequencies
# ---------------------------------------------------------------------------

class TestOctaveBandEdges:
    """§5.6 — band edges at f_1 = f_m × G^(-1/2), f_2 = f_m × G^(+1/2).

    IEC 61260-1 Table 1 specifies a range of [-0.4, +5.3] dB relative
    attenuation just inside each band edge (class 1).  For a Butterworth
    design the attenuation at the band-edge frequency is nominally 3 dB.
    We verify it falls within the less restrictive [1.2, 5.3] dB window
    (class 1 limits spanning both sides of the discontinuity).
    """

    @pytest.mark.parametrize("band_idx", range(8), ids=_BAND_IDS)
    def test_lower_band_edge(self, band_idx: int):
        centers, sos_list = _get_filterbank()
        f_m    = centers[band_idx]
        f_low  = f_m * G ** (-0.5)
        if f_low < 0.5:
            pytest.skip("Lower band edge below 0.5 Hz")
        da = _delta_A(sos_list[band_idx], f_m, -0.5, SAMPLERATE)
        assert 1.2 <= da <= 5.3, (
            f"Band {f_m:.1f} Hz lower edge @ {f_low:.1f} Hz: "
            f"ΔA = {da:.3f} dB (class 1 at band edge: [1.2, 5.3] dB)"
        )

    @pytest.mark.parametrize("band_idx", range(8), ids=_BAND_IDS)
    def test_upper_band_edge(self, band_idx: int):
        centers, sos_list = _get_filterbank()
        f_m    = centers[band_idx]
        f_high = f_m * G ** (+0.5)
        if f_high >= SAMPLERATE / 2:
            pytest.skip("Upper band edge above Nyquist")
        da = _delta_A(sos_list[band_idx], f_m, +0.5, SAMPLERATE)
        assert 1.2 <= da <= 5.3, (
            f"Band {f_m:.1f} Hz upper edge @ {f_high:.1f} Hz: "
            f"ΔA = {da:.3f} dB (class 1 at band edge: [1.2, 5.3] dB)"
        )


# ---------------------------------------------------------------------------
# §5.10 Table 1: Relative attenuation
# ---------------------------------------------------------------------------

class TestOctaveRelativeAttenuation:
    """§5.10 Table 1 — relative attenuation acceptance limits, class 1.

    Pass-band breakpoints are tested at G^{0, ±1/8, ±1/4, ±3/8}.
    Stop-band breakpoints are tested at G^{±1, ±2, ±3, ±4} for all bands
    whose test frequency falls within [0.5, Nyquist) Hz.
    """

    @pytest.mark.parametrize("row", _PB_PARAMS, ids=_PB_IDS)
    def test_passband(self, row: tuple) -> None:
        """Interior pass-band: lo ≤ ΔA(Ω) ≤ hi (dB)."""
        band_idx, f_m, exp, lo, hi = row
        _, sos_list = _get_filterbank()
        da = _delta_A(sos_list[band_idx], f_m, exp, SAMPLERATE)
        assert lo <= da <= hi, (
            f"Band {f_m:.1f} Hz @ Ω=G^{exp:+.3f} "
            f"(f={f_m * G**exp:.1f} Hz): "
            f"ΔA = {da:+.4f} dB  (class 1: [{lo:+.1f}, {hi:+.1f}] dB)"
        )

    @pytest.mark.parametrize("row", _SB_PARAMS, ids=_SB_IDS)
    def test_stopband(self, row: tuple) -> None:
        """Stop-band: ΔA(Ω) ≥ minimum attenuation (dB)."""
        band_idx, f_m, exp, min_da = row
        _, sos_list = _get_filterbank()
        da = _delta_A(sos_list[band_idx], f_m, exp, SAMPLERATE)
        assert da >= min_da, (
            f"Band {f_m:.1f} Hz @ Ω=G^{exp:+d} "
            f"(f={f_m * G**exp:.1f} Hz): "
            f"ΔA = {da:.2f} dB  (class 1 minimum: {min_da:.1f} dB)"
        )


# ---------------------------------------------------------------------------
# §5.12: Effective bandwidth deviation
# ---------------------------------------------------------------------------

class TestOctaveEffectiveBandwidth:
    """§5.12 — effective bandwidth deviation ΔB = 10·log₁₀(B_e/B_r) ≤ ±0.4 dB.

    B_r = ln(G) ≈ 0.6908 (reference effective bandwidth for octave filters).
    B_e is computed numerically from the SOS frequency response.
    """

    @pytest.mark.parametrize("band_idx", range(8), ids=_BAND_IDS)
    def test_bandwidth_deviation(self, band_idx: int) -> None:
        centers, sos_list = _get_filterbank()
        f_m = centers[band_idx]
        db  = _effective_bw_deviation(sos_list[band_idx], f_m, SAMPLERATE)
        assert abs(db) <= DELTA_B_LIMIT_CL1, (
            f"Band {f_m:.1f} Hz: ΔB = {db:+.4f} dB "
            f"(class 1 limit: ±{DELTA_B_LIMIT_CL1} dB)"
        )
