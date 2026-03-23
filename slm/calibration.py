"""Core calibration routine — controller-agnostic."""
from __future__ import annotations

from slm.constants import REFERENCE_PRESSURE


def calibrate_sensitivity(
    controller,
    cal_freq: float = 1000.0,
    cal_level: float = 94.0,
) -> float:
    """Derive controller sensitivity from a known-level calibrator tone.

    Builds an Engine/Bus pipeline with a bandpass filter at cal_freq, runs it
    against the provided controller, then derives the sensitivity from the
    measured RMS and the expected pressure level.

    The controller must already have its raw sensitivity set (e.g. 1.0 V) before
    calling this function.  Returns a value suitable for
    ``controller.set_sensitivity(result, unit="V")``.
    """
    from slm.engine import Engine
    from slm.frequency_weighting import PluginZWeighting, PluginBandpass
    from slm.meter import LeqAccumulator

    engine = Engine(controller, dt=1e9)          # dt=1e9 → reporter never fires
    bus = engine.add_bus("cal", PluginZWeighting)
    bp = PluginBandpass(fc=cal_freq, input=bus.frequency_weighting, width=1, bus=bus)
    bus.add_plugin(bp)
    bp.create_meter(LeqAccumulator, name="leq")

    engine.run()

    mean_sq = bp.read_lin("leq")[0]
    rms = mean_sq ** 0.5
    return rms / (REFERENCE_PRESSURE * 10 ** (cal_level / 20))
