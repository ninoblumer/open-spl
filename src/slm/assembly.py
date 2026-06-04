"""Declarative metric parsing and plugin-chain assembly.

Usage::

    from slm.assembly import parse_metric, build_chain

    specs  = [parse_metric(name) for name in ["LAeq", "LAFmax", "LZeq:bands:63-8000"]]
    build_chain(specs, engine)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slm.bus import Bus
    from slm.engine import Engine
    from slm.plugin_meter import PluginMeter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_UNIT_SECONDS: dict[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}

# L  weighting  [time-weighting]  [measure]  [_window]  [:bands:[N/M:]fmin-fmax]
_PATTERN = re.compile(
    r"^L([ACZ])([FSI]?)(eq|max|min|E)?"
    r"(?:_(dt|\d+[smh]))?"
    r"(?::bands:(?:(\d+/\d+):)?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?))?$"
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
    """Filter density in bands per octave.  For an ``N/M``-octave filter bank
    this equals ``M/N`` — e.g. ``1.0`` for 1/1-octave, ``3.0`` for 1/3-octave,
    ``6.0`` for 1/6-octave."""


# ---------------------------------------------------------------------------
# parse_metric
# ---------------------------------------------------------------------------

def parse_metric(name: str) -> MetricSpec:
    """Parse a metric name string into a :class:`MetricSpec`.

    Supported syntax::

        L[ACZ][FSI?](eq|max|min|E)[_(dt|Ns|Nm|Nh)][:bands:[N/M:]fmin-fmax]

    Examples::

        parse_metric("LAeq")                   # broadband A-weighted Leq, accumulating
        parse_metric("LAFmax_dt")              # A-weighted fast-max, moving (engine dt window)
        parse_metric("LZeq_30s")              # Z-weighted Leq, 30-second moving window
        parse_metric("LZeq:bands:63-8000")    # Z-weighted 1/1-oct Leq, 63–8000 Hz
        parse_metric("LAeq:bands:1/3:31-16000") # A-weighted 1/3-oct Leq, 31–16000 Hz
        parse_metric("LAeq:bands:1/6:63-8000") # A-weighted 1/6-oct Leq, 63–8000 Hz
        parse_metric("LAF")                    # bare metric: most-recent A-fast sample

    Raises :exc:`ValueError` for any invalid or inconsistent name.
    """
    m = _PATTERN.match(name)
    if not m:
        raise ValueError(f"Invalid metric name: {name!r}")

    weighting, tw, measure, window_str, frac_str, fmin_str, fmax_str = m.groups()

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
            if unit not in _WINDOW_UNIT_SECONDS:  # pragma: no cover
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
        if frac_str is not None:
            num, den = (int(p) for p in frac_str.split("/"))
            if num == 0:
                raise ValueError(
                    f"Octave fraction numerator cannot be zero in {name!r}"
                )
            bands_per_oct = den / num

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
# ChainPlan — structural intermediate representation
# ---------------------------------------------------------------------------

# Display names mapping structural node / meter kinds to the class that
# implements them, as *strings*.  build_chain holds the parallel kind -> class
# mapping for actual instantiation; these string versions drive read-only
# renderers (the REPL ``tree`` / ``inspect`` commands) without importing any
# plugin classes.
_WEIGHTING_PLUGIN_NAME: dict[str, str] = {
    "A": "PluginAWeighting",
    "C": "PluginCWeighting",
    "Z": "PluginZWeighting",
}
_TIME_WEIGHTING_PLUGIN_NAME: dict[str, str] = {
    "F": "PluginFastTimeWeighting",
    "S": "PluginSlowTimeWeighting",
    "I": "PluginImpulseTimeWeighting",
}
_ACC_METER_NAME: dict[str, str] = {
    "eq": "LeqAccumulator", "max": "MaxAccumulator", "min": "MinAccumulator",
    "last": "LastAccumulatingMeter", "E": "LEAccumulator",
}
_MOV_METER_NAME: dict[str, str] = {
    "eq": "LeqMovingMeter", "max": "MaxMovingMeter", "min": "MinMovingMeter",
    "E": "LEMovingMeter",
}


@dataclass(frozen=True)
class NodeReq:
    """One node in a metric's processing chain, from the frequency-weighting bus
    down to the terminal plugin the meter attaches to.

    Structural only: a :class:`NodeReq` carries node *kinds* and parameters, not
    plugin classes or runtime values (samplerate, ``dt``).  This lets a
    :class:`ChainPlan` be built from a :class:`MetricSpec` alone — no engine —
    which is what the REPL preview (``tree``) needs.  The :attr:`key` is a
    hashable dedup identifier: two metrics whose chains share a node share its
    key, so the same upstream plugin is created once.
    """

    kind: str
    """Node kind: ``'freq_weighting'``, ``'band'``, ``'time_weighting'``, ``'square'``."""

    key: tuple
    """Hashable dedup key, unique across the whole set of chains."""

    weighting: str
    """Frequency-weighting letter the node lives under (``'A'``/``'C'``/``'Z'``)."""

    bands: tuple[float, float] | None = None
    """``(fmin, fmax)`` for band / band-derived nodes, else ``None``."""

    bands_per_oct: float | None = None
    """Filter density for band / band-derived nodes, else ``None``."""

    time_weighting: str | None = None
    """Time-weighting letter for ``'time_weighting'`` nodes, else ``None``."""


@dataclass(frozen=True)
class MeterReq:
    """The terminal meter of a chain — what to read and how to aggregate it."""

    measure: str
    """Aggregation kind: ``'eq'``/``'max'``/``'min'``/``'last'``/``'E'``."""

    moving: bool
    """``True`` for a moving-window meter, ``False`` for an accumulating one."""

    window_is_dt: bool
    """``True`` when the moving window is the engine block interval (``_dt``)."""

    window_seconds: float | None
    """Explicit moving-window length in seconds, or ``None``."""

    name: str
    """Metric name — used both as the meter name and the report-column label."""

    is_band: bool
    """``True`` for a per-band metric (the report column needs centre frequencies)."""


@dataclass(frozen=True)
class ChainPlan:
    """Structural plan for one metric: the ordered node path plus its meter.

    Produced by :func:`plan_chain` and consumed by every chain *backend* —
    :func:`build_chain` (instantiation) and the REPL ``tree``/``inspect``
    renderers.  The terminal plugin is the last entry of :attr:`nodes`; the
    meter attaches there.
    """

    name: str
    """Metric name, e.g. ``'LAFmax'``."""

    nodes: tuple[NodeReq, ...]
    """Chain path from the frequency-weighting bus to the terminal plugin."""

    meter: MeterReq
    """The meter created on the terminal node."""


@dataclass(frozen=True)
class ColumnBinding:
    """A report column to register: which meter to read and how to label it.

    Produced by :func:`build_chain` / :func:`assemble_engine` and consumed by
    ``Reporter.add_columns``.  This decouples chain assembly from the reporter:
    the assembler says *what* can be read; the caller decides *which* sink reads
    it (and when, via ``engine.on_record``).
    """

    label: str
    """Column label (the metric name)."""

    plugin: "PluginMeter"
    """Terminal plugin holding the meter."""

    meter_name: str
    """Name of the meter on *plugin* to read."""

    center_frequencies: list[float] | None = None
    """Band centre frequencies for per-band columns, else ``None``."""


def plan_chain(spec: MetricSpec) -> ChainPlan:
    """Lower a :class:`MetricSpec` into a structural :class:`ChainPlan`.

    Pure function of the spec — needs no engine, samplerate, or ``dt`` — so it
    can run at REPL-preview time.  The node sequence mirrors the wiring resolved
    by :func:`build_chain`:

    - always a ``freq_weighting`` node (the bus);
    - then an optional ``band`` node for per-band metrics;
    - then an optional ``time_weighting`` node (F/S/I) or ``square`` node (bare
      metrics, which need Pa² input);

    with the meter attached to whichever node ends up last.
    """
    w = spec.weighting
    nodes: list[NodeReq] = [
        NodeReq(kind="freq_weighting", key=("bus", w), weighting=w)
    ]

    if spec.bands is not None:
        nodes.append(NodeReq(
            kind="band", key=("band", w, spec.bands, spec.bands_per_oct),
            weighting=w, bands=spec.bands, bands_per_oct=spec.bands_per_oct,
        ))
        if spec.time_weighting is not None:
            nodes.append(NodeReq(
                kind="time_weighting",
                key=("band_tw", w, spec.bands, spec.bands_per_oct, spec.time_weighting),
                weighting=w, bands=spec.bands, bands_per_oct=spec.bands_per_oct,
                time_weighting=spec.time_weighting,
            ))
        elif spec.measure == "last":
            nodes.append(NodeReq(
                kind="square", key=("band_sq", w, spec.bands, spec.bands_per_oct),
                weighting=w, bands=spec.bands, bands_per_oct=spec.bands_per_oct,
            ))
    elif spec.time_weighting is not None:
        nodes.append(NodeReq(
            kind="time_weighting", key=("tw", w, spec.time_weighting),
            weighting=w, time_weighting=spec.time_weighting,
        ))
    elif spec.measure == "last":
        nodes.append(NodeReq(kind="square", key=("sq", w), weighting=w))

    moving = spec.window_is_dt or spec.window_seconds is not None
    meter = MeterReq(
        measure=spec.measure, moving=moving,
        window_is_dt=spec.window_is_dt, window_seconds=spec.window_seconds,
        name=spec.name, is_band=spec.bands is not None,
    )
    return ChainPlan(name=spec.name, nodes=tuple(nodes), meter=meter)


def _bpo_label(bands_per_oct: float | None) -> str:
    """Return a ``'1/N'`` fraction label for a bands-per-octave value."""
    if bands_per_oct is None:
        return ""
    if float(bands_per_oct).is_integer():
        return f"1/{int(bands_per_oct)}"
    return f"{bands_per_oct:g}/oct"


def node_label(node: NodeReq) -> str:
    """Human-readable one-line label for a chain node (REPL renderers)."""
    if node.kind == "freq_weighting":
        return f"Bus [{node.weighting}]  {_WEIGHTING_PLUGIN_NAME[node.weighting]}"
    if node.kind == "band":
        assert node.bands is not None
        return (f"PluginOctaveBand  limits=({node.bands[0]:.0f}, {node.bands[1]:.0f} Hz)"
                f"  bpo={_bpo_label(node.bands_per_oct)}")
    if node.kind == "time_weighting":
        assert node.time_weighting is not None
        return _TIME_WEIGHTING_PLUGIN_NAME[node.time_weighting]
    if node.kind == "square":
        return "PluginSquare"
    raise ValueError(f"Unknown node kind: {node.kind!r}")  # pragma: no cover


def meter_class_name(meter: MeterReq) -> str:
    """Return the meter class name a :class:`MeterReq` maps to."""
    return (_MOV_METER_NAME if meter.moving else _ACC_METER_NAME)[meter.measure]


# ---------------------------------------------------------------------------
# build_chain
# ---------------------------------------------------------------------------

def build_chain(
    specs: list[MetricSpec],
    engine: Engine,
) -> list[ColumnBinding]:
    """Wire buses, plugins, and meters for *specs* into *engine*; return the
    report-column bindings (the caller registers them with a sink).

    Shared upstream nodes (buses, time-weighting plugins, octave-band plugins)
    are created lazily and reused across specs with identical parameters.
    For example, ``LAFmax`` and ``LAFeq_dt`` share the same A-weighted bus and
    the same fast time-weighting plugin — only their meters differ.

    The signal chain for a broadband metric is::

        Bus(freq-weighting) → [time-weighting | PluginSquare] → Meter

    For a band metric::

        Bus(freq-weighting) → PluginOctaveBand → [time-weighting | PluginSquare] → Meter

    Args:
        specs:  List of parsed metric descriptors, typically from :func:`parse_metric`.
        engine: The :class:`~slm.engine.Engine` instance to attach buses to.

    Returns:
        One :class:`ColumnBinding` per spec, describing the report column it
        exposes.  Pass these to ``Reporter.add_columns`` (this function does not
        touch any reporter itself).
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

    # Codegen maps: structural node / meter kinds → concrete classes (the
    # instantiation counterpart of the string label maps the renderers use).
    _w_cls: dict[str, type[PluginMeter]] = {
        "A": PluginAWeighting, "C": PluginCWeighting, "Z": PluginZWeighting,
    }
    _tw_cls: dict[str, type[PluginMeter]] = {
        "F": PluginFastTimeWeighting, "S": PluginSlowTimeWeighting,
        "I": PluginImpulseTimeWeighting,
    }
    _acc_cls: dict[str, type[PluginMeter]] = {
        "eq": LeqAccumulator, "max": MaxAccumulator, "min": MinAccumulator,
        "last": LastAccumulatingMeter, "E": LEAccumulator,
    }
    # "last" is intentionally absent — bare metrics always use an accumulating meter.
    _mov_cls: dict[str, type[PluginMeter]] = {
        "eq": LeqMovingMeter, "max": MaxMovingMeter, "min": MinMovingMeter,
        "E": LEMovingMeter,
    }

    buses: dict[str, Bus] = {}
    plugins: dict[tuple, PluginMeter] = {}   # NodeReq.key → plugin (shared across specs)
    bindings: list[ColumnBinding] = []

    def get_bus(w: str) -> Bus:
        """Return the frequency-weighted bus for letter *w*, creating it if needed."""
        if w not in buses:
            buses[w] = engine.add_bus(w, _w_cls[w])
        return buses[w]

    def build_node(node: NodeReq, pred: "PluginMeter | None") -> PluginMeter:
        """Instantiate (or reuse, by :attr:`NodeReq.key`) the plugin for *node*.

        *pred* is the upstream plugin (the previous node's output); it is unused
        for the ``freq_weighting`` node, which is the bus's own weighting plugin.
        """
        if node.key in plugins:
            return plugins[node.key]
        bus = get_bus(node.weighting)
        if node.kind == "freq_weighting":
            plugin = bus.frequency_weighting
        elif node.kind == "band":
            plugin = PluginOctaveBand(
                input=pred, limits=node.bands, bands_per_oct=node.bands_per_oct,
                zero_zi=True,
            )
            bus.add_plugin(plugin)
        elif node.kind == "time_weighting":
            plugin = _tw_cls[node.time_weighting](
                input=pred, zero_zi=True, width=pred.width,
            )
            bus.add_plugin(plugin)
        elif node.kind == "square":
            plugin = PluginSquare(input=pred, width=pred.width)
            bus.add_plugin(plugin)
        else:  # pragma: no cover
            raise ValueError(f"Unknown node kind: {node.kind!r}")
        plugins[node.key] = plugin
        return plugin

    for spec in specs:
        plan = plan_chain(spec)

        # Walk the node path, deduping by key; the last plugin is the terminal.
        pred: PluginMeter | None = None
        for node in plan.nodes:
            pred = build_node(node, pred)
        terminal = pred
        assert terminal is not None   # every plan has at least the freq_weighting node

        meter = plan.meter
        meter_cls = (_mov_cls if meter.moving else _acc_cls)[meter.measure]
        meter_kwargs: dict[str, float] = {}
        if meter.moving and not meter.window_is_dt:
            # window_is_dt → no 't' kwarg → MovingMeter defaults to bus.dt
            meter_kwargs = {"t": meter.window_seconds}
        terminal.create_meter(meter_cls, name=meter.name, **meter_kwargs)

        # Record the report column this metric exposes (the caller registers it
        # with a sink); band metrics carry centre frequencies for labels.
        if meter.is_band:
            band_node = next(n for n in plan.nodes if n.kind == "band")
            center_freqs = plugins[band_node.key].center_frequencies
        else:
            center_freqs = None
        bindings.append(ColumnBinding(meter.name, terminal, meter.name,
                                      center_frequencies=center_freqs))

    return bindings


def assemble_engine(
    specs: list[MetricSpec],
    controller,
    dt: float = 0.1,
) -> tuple[Engine, list[ColumnBinding]]:
    """Create an :class:`~slm.engine.Engine` for *controller* and wire *specs*.

    The factory counterpart of :func:`build_chain`: it constructs the engine
    (so a half-built engine is never exposed), wires the chain, and returns the
    engine together with its report-column bindings.  The caller attaches a
    sink::

        engine, bindings = assemble_engine(specs, controller, dt)
        reporter.add_columns(bindings)
        engine.on_record = reporter.record
        engine.run()
    """
    from slm.engine import Engine
    engine = Engine(controller, dt=dt)
    bindings = build_chain(specs, engine)
    return engine, bindings
