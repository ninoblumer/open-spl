"""Core calibration routine — controller-agnostic."""
from __future__ import annotations

from slm.constants import REFERENCE_PRESSURE


def calibrate_sensitivity(
    controller,
    cal_freq: float = 1000.0,
    cal_level: float = 94.0,
    stability_window: int | None = None,
    stability_threshold: float = 0.1,
) -> float:
    """Derive controller sensitivity from a known-level calibrator tone.

    Builds an Engine/Bus pipeline with a bandpass filter at *cal_freq*, runs it
    against the provided controller, then derives the sensitivity from the
    measured RMS and the expected pressure level.

    The controller must already have its raw sensitivity set (e.g. 1.0 V) before
    calling this function.  Returns a value suitable for
    ``controller.set_sensitivity(result, unit="V")``.

    Parameters
    ----------
    controller:
        Any :class:`~slm.io.controller.Controller` instance.
    cal_freq:
        Centre frequency of the calibrator tone in Hz (default 1000.0).
    cal_level:
        Known SPL of the calibrator tone in dB (default 94.0).
    stability_window:
        When ``None`` (default) the engine runs until the controller raises
        ``StopIteration`` — suitable for file-based controllers.  When set to
        an integer *N*, a rolling window of *N* half-second Leq readings is
        tracked; the controller is stopped automatically once the standard
        deviation of those readings drops below *stability_threshold* dB.
        Use this for real-time controllers where there is no natural end.
    stability_threshold:
        Maximum rolling standard deviation (dB) to consider the tone stable
        (default 0.1 dB).  Only used when *stability_window* is not ``None``.
    """
    from collections import deque

    import numpy as np

    from slm.engine import Engine
    from slm.frequency_weighting import PluginZWeighting, PluginBandpass
    from slm.meter import LeqAccumulator, LeqMovingMeter

    use_stability = stability_window is not None
    dt = 0.5 if use_stability else 1e9   # dt=1e9 → reporter never fires (file path)

    if use_stability:
        from slm.io.reporter import Reporter

        _history: deque[float] = deque(maxlen=stability_window)

        def _on_report(timestamp, bb, bands):
            val_sq = bp.read_lin("leq_moving")[0]
            if val_sq > 0:
                _history.append(10.0 * np.log10(val_sq / REFERENCE_PRESSURE ** 2))
            if (len(_history) == stability_window
                    and float(np.std(_history)) < stability_threshold):
                controller.stop()

        engine = Engine(controller, dt=dt, reporter=Reporter(display_fn=_on_report))
    else:
        engine = Engine(controller, dt=dt)
    bus = engine.add_bus("cal", PluginZWeighting)
    bp = PluginBandpass(fc=cal_freq, input=bus.frequency_weighting, width=1, bus=bus)
    bus.add_plugin(bp)
    bp.create_meter(LeqAccumulator, name="leq")

    if use_stability:
        bp.create_meter(LeqMovingMeter, name="leq_moving", t=1.0)

    engine.run()

    mean_sq = bp.read_lin("leq")[0]
    rms = mean_sq ** 0.5
    return rms / (REFERENCE_PRESSURE * 10 ** (cal_level / 20))
