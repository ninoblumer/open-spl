"""Declarative metric parsing and plugin-chain assembly.

Usage::

    from slm.assembly import parse_metric, build_chain

    specs  = [parse_metric(name) for name in ["LAeq", "LAFmax", "LZeq:bands:63-8000"]]
    build_chain(specs, engine, reporter)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slm.engine import Engine
    from slm.reporter import Reporter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_UNIT_SECONDS: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}

# L  weighting  [time-weighting]  measure  [_window]  [:bands:[1/3:]fmin-fmax]
_PATTERN = re.compile(
    r"^L([ACZ])([FSI]?)(eq|max|min)"
    r"(?:_(dt|\d+[smh]))?"
    r"(?::bands:(?:(1/3):)?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?))?$"
)


# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    """Parsed representation of a single metric name."""

    name: str
    weighting: str               # 'A', 'C', or 'Z'
    time_weighting: str | None   # 'F', 'S', 'I', or None
    measure: str                 # 'eq', 'max', or 'min'
    window_is_dt: bool           # True when the suffix was '_dt'
    window_seconds: float | None # explicit window in seconds, or None (accumulating)
    bands: tuple[float, float] | None  # (fmin, fmax), or None for broadband
    bands_per_oct: float         # 1.0 for 1/1-oct, 3.0 for 1/3-oct


# ---------------------------------------------------------------------------
# parse_metric
# ---------------------------------------------------------------------------

def parse_metric(name: str) -> MetricSpec:
    """Parse a metric name string into a :class:`MetricSpec`.

    Raises :exc:`ValueError` for any invalid or inconsistent name.
    """
    m = _PATTERN.match(name)
    if not m:
        raise ValueError(f"Invalid metric name: {name!r}")

    weighting, tw, measure, window_str, third_oct, fmin_str, fmax_str = m.groups()

    # Leq must not have a time-weighting letter; max/min must have one
    if measure == "eq" and tw:
        raise ValueError(
            f"Leq cannot have a time-weighting letter (got {name!r}). "
            f"Did you mean L{weighting}eq?"
        )
    if measure in ("max", "min") and not tw:
        raise ValueError(
            f"L{weighting}{measure} requires a time-weighting letter (F, S, or I): {name!r}"
        )

    # Parse window suffix
    window_is_dt = False
    window_seconds: float | None = None
    if window_str is not None:
        if window_str == "dt":
            window_is_dt = True
        else:
            n = float(window_str[:-1])
            unit = window_str[-1]
            window_seconds = n * _WINDOW_UNIT_SECONDS[unit]

    # Parse band limits
    bands: tuple[float, float] | None = None
    bands_per_oct = 1.0
    if fmin_str is not None:
        bands = (float(fmin_str), float(fmax_str))
        bands_per_oct = 3.0 if third_oct == "1/3" else 1.0

    return MetricSpec(
        name=name,
        weighting=weighting,
        time_weighting=tw if tw else None,
        measure=measure,
        window_is_dt=window_is_dt,
        window_seconds=window_seconds,
        bands=bands,
        bands_per_oct=bands_per_oct,
    )


# ---------------------------------------------------------------------------
# build_chain
# ---------------------------------------------------------------------------

def build_chain(
    specs: list[MetricSpec],
    engine: "Engine",
    reporter: "Reporter",
) -> None:
    """Wire buses, plugins, and meters for *specs*; register each with *reporter*.

    Shared upstream nodes (buses, time-weighting plugins, octave-band plugins)
    are created lazily and reused across specs with identical parameters.
    """
    from slm.frequency_weighting import (
        PluginAWeighting, PluginCWeighting, PluginZWeighting,
    )
    from slm.time_weighting import (
        PluginFastTimeWeighting, PluginSlowTimeWeighting, PluginImpulseTimeWeighting,
    )
    from slm.octave_band import PluginOctaveBand
    from slm.meter import (
        LeqAccumulator, MaxAccumulator, MinAccumulator,
        LeqMovingMeter, MaxMovingMeter, MinMovingMeter,
    )

    _w_cls = {
        "A": PluginAWeighting,
        "C": PluginCWeighting,
        "Z": PluginZWeighting,
    }
    _tw_cls = {
        "F": PluginFastTimeWeighting,
        "S": PluginSlowTimeWeighting,
        "I": PluginImpulseTimeWeighting,
    }
    _acc_cls = {"eq": LeqAccumulator, "max": MaxAccumulator, "min": MinAccumulator}
    _mov_cls = {"eq": LeqMovingMeter, "max": MaxMovingMeter, "min": MinMovingMeter}

    buses: dict[str, object] = {}
    tw_plugins: dict[tuple, object] = {}
    band_plugins: dict[tuple, object] = {}

    def get_bus(w: str):
        if w not in buses:
            buses[w] = engine.add_bus(w, _w_cls[w])
        return buses[w]

    def get_tw_plugin(w: str, tw_letter: str):
        key = (w, tw_letter)
        if key not in tw_plugins:
            bus = get_bus(w)
            freq_w = bus.frequency_weighting
            plugin = _tw_cls[tw_letter](input=freq_w, zero_zi=True)
            bus.add_plugin(plugin)
            tw_plugins[key] = plugin
        return tw_plugins[key]

    def get_band_plugin(w: str, bands: tuple[float, float], bpo: float):
        key = (w, bands, bpo)
        if key not in band_plugins:
            bus = get_bus(w)
            freq_w = bus.frequency_weighting
            plugin = PluginOctaveBand(
                input=freq_w, limits=bands, bands_per_oct=bpo, zero_zi=True,
            )
            bus.add_plugin(plugin)
            band_plugins[key] = plugin
        return band_plugins[key]

    for spec in specs:
        # Resolve the upstream plugin this metric reads from
        if spec.bands is not None:
            plugin = get_band_plugin(spec.weighting, spec.bands, spec.bands_per_oct)
        elif spec.time_weighting is not None:
            plugin = get_tw_plugin(spec.weighting, spec.time_weighting)
        else:
            bus = get_bus(spec.weighting)
            plugin = bus.frequency_weighting

        # Select meter class and build kwargs
        is_moving = spec.window_is_dt or spec.window_seconds is not None
        if not is_moving:
            meter_cls = _acc_cls[spec.measure]
            meter_kwargs: dict = {}
        else:
            meter_cls = _mov_cls[spec.measure]
            # window_is_dt=True → no 't' kwarg → MovingMeter defaults to bus.dt
            meter_kwargs = {} if spec.window_is_dt else {"t": spec.window_seconds}

        plugin.create_meter(meter_cls, name=spec.name, **meter_kwargs)

        # Register with reporter
        center_freqs = plugin.center_frequencies if spec.bands is not None else None
        reporter.add_column(spec.name, plugin, spec.name, center_frequencies=center_freqs)
