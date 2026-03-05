"""
Unit tests: step-response time constants for Fast, Slow, and Impulse time weightings.

Each filter is a first-order IIR exponential moving average on the squared pressure
signal. For a step from steady-state A² down to 0, the output must decay as:
    y(t) = A² * exp(-t / tau)
i.e. after exactly tau seconds the output is e^{-1} ≈ 36.8 % of the peak.

The Impulse weighting uses separate rise (35 ms) and fall (1500 ms) time constants
and is implemented in the numba function `asymmetric_time_weighting`.
"""
import numpy as np
import pytest
from scipy.signal import lfilter

from slm.time_weighting import asymmetric_time_weighting

FS = 48_000          # sample rate used throughout
RTOL = 1e-3          # 0.1 % tolerance — tight enough to catch the wrong formula


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_coeffs(tau: float) -> tuple[list, list]:
    """Return (b, a) for the symmetric 1st-order IIR used by PluginSymmetricTimeWeighting."""
    alpha = 1.0 - np.exp(-1.0 / (tau * FS))
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    return b, a


def _sym_decay_db_per_s(tau: float) -> float:
    """Drive symmetric filter to steady state, then apply 1 s of silence.

    Returns dB drop per second.
    IEC 61672-1 §5.8: Fast=34.7 dB/s, Slow=4.3 dB/s.
    """
    b, a = _sym_coeffs(tau)
    n_rise = int(5 * tau * FS)
    y_rise, zi = lfilter(b, a, np.ones(n_rise), zi=np.zeros(1))
    y0 = float(y_rise[-1])
    y_fall, _ = lfilter(b, a, np.zeros(FS), zi=zi)   # exactly 1 s at 48 kHz
    y1 = float(y_fall[-1])
    return 10 * np.log10(y0 / y1)


def _sym_decay_ratio(tau: float) -> float:
    """
    Drive the symmetric filter toward steady state with x²=1, record y₀ (last
    sample of rise phase), then feed x²=0 for exactly tau seconds and record y₁.
    Returns y₁ / y₀.  Expected: e^{-1} ≈ 0.3679.

    Using the ratio avoids sensitivity to whether the filter has fully converged
    to steady state during the rise phase.
    """
    b, a = _sym_coeffs(tau)
    n_rise = int(5 * tau * FS)
    y_rise, zi = lfilter(b, a, np.ones(n_rise), zi=np.zeros(1))
    y0 = float(y_rise[-1])
    n_fall = int(tau * FS)
    y_fall, _ = lfilter(b, a, np.zeros(n_fall), zi=zi)
    return float(y_fall[-1]) / y0


def _asym_rise_ratio(tau_rise: float, tau_fall: float) -> float:
    """
    Drive the asymmetric filter from 0 with x²=1 for exactly tau_rise seconds,
    then continue for another tau_rise seconds and record y₁.
    Returns y₁ / (1 - y₁), normalised to the complement so we can compare
    rise progress.

    Simpler: just return y(tau_rise) / y_ss where y_ss is approximated from a
    very long rise phase.  We instead return the ratio y(tau_rise)/y(5*tau_rise)
    which should equal (1 - e^{-1}) / (1 - e^{-5}) ≈ 0.632 / 0.993 ≈ 0.636.

    Actually, the cleanest check: starting from 0, after tau_rise seconds the
    output should be (1-e^{-1}) * y_ss.  Approximate y_ss with 10×tau_rise run.
    Returns y(tau_rise) / y_ss.  Expected: (1 - e^{-1}) ≈ 0.6321.
    """
    alpha_rise = 1.0 - np.exp(-1.0 / (tau_rise * FS))
    alpha_fall = 1.0 - np.exp(-1.0 / (tau_fall * FS))
    # Approximate steady-state with a long rise (10×tau_rise)
    n_long = int(10 * tau_rise * FS)
    y_long, _ = asymmetric_time_weighting(
        np.ones(n_long), zi=0.0, alpha_rise=alpha_rise, alpha_fall=alpha_fall
    )
    y_ss = float(y_long[-1])
    # Now measure rise from 0 for exactly tau_rise
    n = int(tau_rise * FS)
    y, _ = asymmetric_time_weighting(
        np.ones(n), zi=0.0, alpha_rise=alpha_rise, alpha_fall=alpha_fall
    )
    return float(y[-1]) / y_ss


def _asym_fall_ratio(tau_rise: float, tau_fall: float) -> float:
    """
    Drive the asymmetric filter toward steady state, record y₀ (last sample of
    rise), then feed x²=0 for exactly tau_fall seconds and record y₁.
    Returns y₁ / y₀.  Expected: e^{-1} ≈ 0.3679.
    """
    alpha_rise = 1.0 - np.exp(-1.0 / (tau_rise * FS))
    alpha_fall = 1.0 - np.exp(-1.0 / (tau_fall * FS))
    n_rise = int(5 * tau_rise * FS)
    y_rise, zi = asymmetric_time_weighting(
        np.ones(n_rise), zi=0.0, alpha_rise=alpha_rise, alpha_fall=alpha_fall
    )
    y0 = float(y_rise[-1])
    n_fall = int(tau_fall * FS)
    y_fall, _ = asymmetric_time_weighting(
        np.zeros(n_fall), zi=zi, alpha_rise=alpha_rise, alpha_fall=alpha_fall
    )
    return float(y_fall[-1]) / y0


# ---------------------------------------------------------------------------
# Fast (tau = 125 ms)
# ---------------------------------------------------------------------------

class TestFastDecayRate:
    def test_decay_ratio_is_1_over_e(self):
        """y(tau) / y(0) must equal e^{-1} to confirm tau=125 ms."""
        ratio = _sym_decay_ratio(tau=0.125)
        np.testing.assert_allclose(ratio, np.exp(-1), rtol=RTOL)

    def test_decay_rate_db_per_second(self):
        """IEC 61672-1 §5.8: Fast time weighting shall decay at 34.7 dB/s."""
        rate = _sym_decay_db_per_s(tau=0.125)
        np.testing.assert_allclose(rate, 10 / (np.log(10) * 0.125), rtol=RTOL)


# ---------------------------------------------------------------------------
# Slow (tau = 1000 ms)
# ---------------------------------------------------------------------------

class TestSlowDecayRate:
    def test_decay_ratio_is_1_over_e(self):
        """y(tau) / y(0) must equal e^{-1} to confirm tau=1000 ms."""
        ratio = _sym_decay_ratio(tau=1.0)
        np.testing.assert_allclose(ratio, np.exp(-1), rtol=RTOL)

    def test_decay_rate_db_per_second(self):
        """IEC 61672-1 §5.8: Slow time weighting shall decay at 4.3 dB/s."""
        rate = _sym_decay_db_per_s(tau=1.0)
        np.testing.assert_allclose(rate, 10 / (np.log(10) * 1.0), rtol=RTOL)


# ---------------------------------------------------------------------------
# Impulse (tau_rise = 35 ms, tau_fall = 1500 ms)
# ---------------------------------------------------------------------------

class TestImpulseDecayRate:

    tau_rise = 0.035
    tau_fall = 1.500

    def test_rise_ratio_is_1_minus_1_over_e(self):
        """y(tau_rise) / y_ss must equal (1-e^{-1}) to confirm tau_rise=35 ms."""
        ratio = _asym_rise_ratio(self.tau_rise, self.tau_fall)
        np.testing.assert_allclose(ratio, 1.0 - np.exp(-1), rtol=RTOL)

    def test_fall_ratio_is_1_over_e(self):
        """y(tau_fall) / y(0) must equal e^{-1} to confirm tau_fall=1500 ms."""
        ratio = _asym_fall_ratio(self.tau_rise, self.tau_fall)
        np.testing.assert_allclose(ratio, np.exp(-1), rtol=RTOL)
