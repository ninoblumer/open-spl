"""Core calibration routine — controller-agnostic.

Calibration follows IEC 61672-1's two steps: **measure** the calibrator tone,
then **adjust** the sensitivity.  :func:`calibrate_sensitivity` is the measure
step — it returns the sensitivity the tone implies without mutating anything; the
caller performs the adjust step (``controller.set_sensitivity(...)`` or storing
the value for later measurements).
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from slm.constants import CALIBRATION_FREQ_HZ, CALIBRATION_LEVEL_DB, REFERENCE_PRESSURE

if TYPE_CHECKING:
    from datetime import timedelta

    from slm.io.controller import Controller
    from slm.plugin_meter import PluginMeter


class _StabilityMonitor:
    """``on_record`` observer that auto-stops a real-time calibration.

    The engine fires ``on_record`` every block; this monitor samples the moving
    Leq once per *dt* seconds (gating the per-block ticks itself), keeps a rolling
    window of *window* readings, and stops the controller once their standard
    deviation falls below *threshold* dB — i.e. the tone has settled.
    """

    def __init__(self, plugin: "PluginMeter", meter_name: str, controller: "Controller",
                 *, window: int, threshold: float, dt: float) -> None:
        self._plugin = plugin
        self._meter_name = meter_name
        self._controller = controller
        self._window = window
        self._threshold = threshold
        self._dt = dt
        self._history: deque[float] = deque(maxlen=window)
        self._last_sample: "timedelta | None" = None

    def __call__(self, timestamp: "timedelta", dt: float) -> None:
        # Gate the per-block ticks down to one reading per dt seconds.
        if (self._last_sample is not None
                and (timestamp - self._last_sample).total_seconds() < self._dt):
            return
        self._last_sample = timestamp

        val_sq = self._plugin.read_lin(self._meter_name)[0]
        if val_sq > 0:
            self._history.append(10.0 * np.log10(val_sq / REFERENCE_PRESSURE ** 2))
        if (len(self._history) == self._window
                and float(np.std(self._history)) < self._threshold):
            self._controller.stop()


def calibrate_sensitivity(
    controller,
    cal_freq: float = CALIBRATION_FREQ_HZ,
    cal_level: float = CALIBRATION_LEVEL_DB,
    stability_window: int | None = None,
    stability_threshold: float = 0.1,
) -> float:
    """Measure a calibrator tone and return the sensitivity it implies (V/Pa).

    This is the **measure** half of calibration: it bandpass-filters the input at
    *cal_freq*, integrates the tone's RMS at the controller's current (raw)
    sensitivity, and returns the sensitivity that would make that tone read
    *cal_level* dB.  It does **not** mutate the controller — the caller performs
    the **adjust** step::

        proposed = calibrate_sensitivity(controller)     # step 1: measure
        controller.set_sensitivity(proposed, unit="V")   # step 2: adjust

    The controller must have a raw sensitivity set (e.g. 1.0 V) beforehand.

    Parameters
    ----------
    controller:
        Any :class:`~slm.io.controller.Controller` instance.
    cal_freq:
        Centre frequency of the calibrator tone in Hz (default 1000.0).
    cal_level:
        Known SPL of the calibrator tone in dB (default 94.0).
    stability_window:
        ``None`` (default) runs until the controller raises ``StopIteration``
        (file sources).  An integer *N* enables auto-stop for real-time sources:
        a :class:`_StabilityMonitor` (an ``on_record`` observer) samples the
        moving Leq every 0.5 s and stops once *N* readings agree within
        *stability_threshold* dB.
    stability_threshold:
        Max rolling standard deviation (dB) considered stable (default 0.1).
        Only used when *stability_window* is given.
    """
    from slm.engine import Engine
    from slm.frequency_weighting import PluginZWeighting, PluginBandpass
    from slm.time_weighting import PluginSquare
    from slm.meter import LeqAccumulator, LeqMovingMeter

    use_stability = stability_window is not None
    dt = 0.5   # stability sampling cadence (no observer is attached on the file path)

    engine = Engine(controller, dt=dt)
    bus = engine.add_bus("cal", PluginZWeighting)
    bp = PluginBandpass(fc=cal_freq, input=bus.frequency_weighting, width=1, bus=bus)
    bus.add_plugin(bp)
    # Square the bandpass output so the Leq meters get Pa² (they no longer square).
    sq = PluginSquare(input=bp, width=1)
    bus.add_plugin(sq)
    sq.create_meter(LeqAccumulator, name="leq")

    if use_stability:
        sq.create_meter(LeqMovingMeter, name="leq_moving", t=1.0)
        engine.on_record = _StabilityMonitor(
            sq, "leq_moving", controller,
            window=stability_window, threshold=stability_threshold, dt=dt,
        )

    engine.run()

    mean_sq = sq.read_lin("leq")[0]
    rms = mean_sq ** 0.5
    return rms / (REFERENCE_PRESSURE * 10 ** (cal_level / 20))
