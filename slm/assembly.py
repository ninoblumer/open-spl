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
    from slm.bus import Bus
    from slm.engine import Engine
    from slm.io.reporter import Reporter
    from slm.plugin_meter import PluginMeter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_UNIT_SECONDS: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}

# L  weighting  [time-weighting]  [measure]  [_window]  [:bands:[1/3:]fmin-fmax]
_PATTERN = re.compile(
    r"^L([ACZ])([FSI]?)(eq|max|min|E)?"
    r"(?:_(dt|\d+[smh]))?"
    r"(?::bands:(?:(1/3):)?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?))?$"
)


# ---------------------------------------------------------------------------
# MetricSpec
# ---------------------------------------------------------------------------

@dataclass
class MetricSpec:
    """Parsed representation of a single metric name.

    Produced by :func:`parse_metric` and consumed by :func:`build_chain`.
    All fields are derived solely from the metric name string — no engine
    state is needed to construct one.
    """

    name: str
    """Original metric name string, e.g. ``'LAFmax'``."""

    weighting: str
    """Frequency-weighting letter: ``'A'``, ``'C'``, or ``'Z'``."""

    time_weighting: str | None
    """Time-weighting letter (``'F'``, ``'S'``, ``'I'``), or ``None`` for Leq/LE/bare."""

    measure: str
    """Aggregation kind: ``'eq'``, ``'max'``, ``'min'``, ``'E'`` (sound exposure), or
    ``'last'`` (most-recent time-weighted sample, bare metric syntax)."""

    window_is_dt: bool
    """``True`` when the window suffix was ``_dt`` (use the engine's block interval)."""

    window_seconds: float | None
    """Explicit moving-window duration in seconds, or ``None`` for an accumulating meter."""

    bands: tuple[float, float] | None
    """``(fmin, fmax)`` band limits in Hz, or ``None`` for broadband."""

    bands_per_oct: float
    """Filter density: ``1.0`` for 1/1-octave, ``3.0`` for 1/3-octave."""


# ---------------------------------------------------------------------------
# parse_metric
# ---------------------------------------------------------------------------

def parse_metric(name: str) -> MetricSpec:
    """Parse a metric name string into a :class:`MetricSpec`.

    Supported syntax::

        L[ACZ][FSI?](eq|max|min|E)[_(dt|Ns|Nm|Nh)][:bands:[1/3:]fmin-fmax]

    Examples::

        parse_metric("LAeq")              # broadband A-weighted Leq, accumulating
        parse_metric("LAFmax_dt")         # A-weighted fast-max, moving (engine dt window)
        parse_metric("LZeq_30s")          # Z-weighted Leq, 30-second moving window
        parse_metric("LZeq:bands:63-8000")  # Z-weighted per 1/1-oct band, accumulating
        parse_metric("LAF")               # bare metric: most-recent A-fast sample

    Raises :exc:`ValueError` for any invalid or inconsistent name.
    """
    m = _PATTERN.match(name)
    if not m:
        raise ValueError(f"Invalid metric name: {name!r}")

    weighting, tw, measure, window_str, third_oct, fmin_str, fmax_str = m.groups()

    # Leq must not have a time-weighting letter; max/min must have one
    # No measure → "last" (just the most-recent time-weighted sample); requires tw
    if measure == "eq" and tw:
        raise ValueError(
            f"Leq cannot have a time-weighting letter (got {name!r}). "
            f"Did you mean L{weighting}eq?"
        )
    if measure in ("max", "min") and not tw:
        raise ValueError(
            f"L{weighting}{measure} requires a time-weighting letter (F, S, or I): {name!r}"
        )
    # bare metric with no time-weighting letter → allowed (uses PluginSquare in build_chain)
    if measure == "E" and tw:
        raise ValueError(
            f"LE does not use a time-weighting letter: {name!r}"
        )
    if measure is None:
        measure = "last"

    # "last" is a single-sample snapshot — a window suffix makes no sense
    # Check window_str before we parse it
    if measure == "last" and window_str is not None:
        raise ValueError(
            f"Bare metric {name!r} (no eq/max/min) cannot have a window suffix."
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
            if unit not in _WINDOW_UNIT_SECONDS:
                valid = ", ".join(_WINDOW_UNIT_SECONDS)
                raise ValueError(
                    f"Unknown window unit {unit!r} in {name!r}; expected one of: {valid}"
                )
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
    engine: Engine,
    reporter: Reporter,
) -> None:
    """Wire buses, plugins, and meters for *specs*; register each with *reporter*.

    Shared upstream nodes (buses, time-weighting plugins, octave-band plugins)
    are created lazily and reused across specs with identical parameters.
    For example, ``LAFmax`` and ``LAFeq_dt`` share the same A-weighted bus and
    the same fast time-weighting plugin — only their meters differ.

    The signal chain for a broadband metric is::

        Bus(freq-weighting) → [time-weighting | PluginSquare] → Meter

    For a band metric::

        Bus(freq-weighting) → PluginOctaveBand → [time-weighting | PluginSquare] → Meter

    Args:
        specs:    List of parsed metric descriptors, typically from :func:`parse_metric`.
        engine:   The :class:`~slm.engine.Engine` instance to attach buses to.
        reporter: The :class:`~slm.io.reporter.Reporter` that collects meter readings.
    """
    from slm.frequency_weighting import (
        PluginAWeighting, PluginCWeighting, PluginZWeighting,
    )
    from slm.time_weighting import (
        PluginFastTimeWeighting, PluginSlowTimeWeighting, PluginImpulseTimeWeighting,
        PluginSquare,
    )
    from slm.octave_band import PluginOctaveBand
    from slm.meter import (
        LeqAccumulator, MaxAccumulator, MinAccumulator, LastAccumulatingMeter,
        LeqMovingMeter, MaxMovingMeter, MinMovingMeter,
        LEAccumulator, LEMovingMeter,
    )

    # Maps weighting letter → frequency-weighting plugin class
    _w_cls: dict[str, type[PluginMeter]] = {
        "A": PluginAWeighting,
        "C": PluginCWeighting,
        "Z": PluginZWeighting,
    }
    # Maps time-weighting letter → time-weighting plugin class
    _tw_cls: dict[str, type[PluginMeter]] = {
        "F": PluginFastTimeWeighting,
        "S": PluginSlowTimeWeighting,
        "I": PluginImpulseTimeWeighting,
    }
    # Maps measure string → accumulating meter class (no moving window)
    _acc_cls: dict[str, type[PluginMeter]] = {
        "eq": LeqAccumulator,
        "max": MaxAccumulator,
        "min": MinAccumulator,
        "last": LastAccumulatingMeter,
        "E": LEAccumulator,
    }
    # Maps measure string → moving-window meter class
    # Note: "last" is intentionally absent — bare metrics always use an accumulating meter.
    _mov_cls: dict[str, type[PluginMeter]] = {
        "eq": LeqMovingMeter,
        "max": MaxMovingMeter,
        "min": MinMovingMeter,
        "E": LEMovingMeter,
    }

    # Lazy-creation caches keyed by the parameters that uniquely identify each node.
    buses: dict[str, Bus] = {}
    tw_plugins: dict[tuple[str, str], PluginMeter] = {}
    sq_plugins: dict[str, PluginMeter] = {}
    band_plugins: dict[tuple[str, tuple[float, float], float], PluginMeter] = {}
    band_tw_plugins: dict[tuple[str, tuple[float, float], float, str], PluginMeter] = {}
    band_sq_plugins: dict[tuple[str, tuple[float, float], float], PluginMeter] = {}

    def get_bus(w: str) -> Bus:
        """Return the frequency-weighted bus for weighting letter *w*, creating it if needed."""
        if w not in buses:
            buses[w] = engine.add_bus(w, _w_cls[w])
        return buses[w]

    def get_tw_plugin(w: str, tw_letter: str) -> PluginMeter:
        """Return the broadband time-weighting plugin for (*w*, *tw_letter*), creating if needed."""
        key = (w, tw_letter)
        if key not in tw_plugins:
            bus = get_bus(w)
            freq_w = bus.frequency_weighting
            plugin = _tw_cls[tw_letter](input=freq_w, zero_zi=True)
            bus.add_plugin(plugin)
            tw_plugins[key] = plugin
        return tw_plugins[key]

    def get_band_plugin(w: str, bands: tuple[float, float], bpo: float) -> PluginMeter:
        """Return the octave-band filter bank for (*w*, *bands*, *bpo*), creating if needed."""
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

    def get_sq_plugin(w: str) -> PluginMeter:
        """Return the broadband squaring plugin for *w*, creating if needed.

        Used for bare metrics (no time-weighting) so the meter receives Pa² input.
        """
        if w not in sq_plugins:
            bus = get_bus(w)
            plugin = PluginSquare(input=bus.frequency_weighting)
            bus.add_plugin(plugin)
            sq_plugins[w] = plugin
        return sq_plugins[w]

    def get_band_sq_plugin(w: str, bands: tuple[float, float], bpo: float) -> PluginMeter:
        """Return the per-band squaring plugin for (*w*, *bands*, *bpo*), creating if needed.

        Used for bare per-band metrics so each band output is in Pa².
        """
        key = (w, bands, bpo)
        if key not in band_sq_plugins:
            band_plugin = get_band_plugin(w, bands, bpo)
            plugin = PluginSquare(input=band_plugin, width=band_plugin.width)
            get_bus(w).add_plugin(plugin)
            band_sq_plugins[key] = plugin
        return band_sq_plugins[key]

    def get_band_tw_plugin(
        w: str, bands: tuple[float, float], bpo: float, tw_letter: str
    ) -> PluginMeter:
        """Return the per-band time-weighting plugin for (*w*, *bands*, *bpo*, *tw_letter*).

        The plugin is inserted after the octave-band filter bank so each band
        is time-weighted independently.
        """
        key = (w, bands, bpo, tw_letter)
        if key not in band_tw_plugins:
            band_plugin = get_band_plugin(w, bands, bpo)
            plugin = _tw_cls[tw_letter](input=band_plugin, zero_zi=True, width=band_plugin.width)
            get_bus(w).add_plugin(plugin)
            band_tw_plugins[key] = plugin
        return band_tw_plugins[key]

    for spec in specs:
        # Resolve the upstream plugin this metric reads from
        if spec.bands is not None:
            if spec.time_weighting is not None:
                plugin = get_band_tw_plugin(
                    spec.weighting, spec.bands, spec.bands_per_oct, spec.time_weighting
                )
            elif spec.measure == "last":
                # no TW, bare metric per band: square first so output is Pa²
                plugin = get_band_sq_plugin(spec.weighting, spec.bands, spec.bands_per_oct)
            else:
                plugin = get_band_plugin(spec.weighting, spec.bands, spec.bands_per_oct)
        elif spec.time_weighting is not None:
            plugin = get_tw_plugin(spec.weighting, spec.time_weighting)
        elif spec.measure == "last":
            # no TW, broadband bare metric: square first so output is Pa²
            plugin = get_sq_plugin(spec.weighting)
        else:
            bus = get_bus(spec.weighting)
            plugin = bus.frequency_weighting

        # Select meter class and build kwargs
        is_moving = spec.window_is_dt or spec.window_seconds is not None
        if not is_moving:
            meter_cls = _acc_cls[spec.measure]
            meter_kwargs: dict[str, float] = {}
        else:
            meter_cls = _mov_cls[spec.measure]
            # window_is_dt=True → no 't' kwarg → MovingMeter defaults to bus.dt
            meter_kwargs = {} if spec.window_is_dt else {"t": spec.window_seconds}

        plugin.create_meter(meter_cls, name=spec.name, **meter_kwargs)

        # Register with reporter; band metrics also pass centre frequencies for column labels
        if spec.bands is not None:
            band_plugin = get_band_plugin(spec.weighting, spec.bands, spec.bands_per_oct)
            center_freqs = band_plugin.center_frequencies
        else:
            center_freqs = None
        reporter.add_column(spec.name, plugin, spec.name, center_frequencies=center_freqs)
