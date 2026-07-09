# Architecture

```
Controller (FileController | ArrayController | SounddeviceController | NoiseController)
    │  reads audio blocks
    ▼
Engine
    │  routes blocks to each Bus, samples meters every dt seconds
    ├─► Bus(A) ──► PluginAWeighting ──► [plugins…] ──► [meters…]
    ├─► Bus(C) ──► PluginCWeighting ──► [plugins…] ──► [meters…]
    └─► Bus(Z) ──► PluginZWeighting ──► [plugins…] ──► [meters…]
                                                           │
Reporter ◄─────────────────────────────────────────────────┘
    │  writes CSV output files
```

## Key components

- **`Engine`** — main processing loop; owns buses; emits an `on_record(timestamp, dt)` tick each block to a sink callback (it holds no reference to the reporter)
- **`Bus`** — one frequency weighting + a chain of downstream plugins and meters. An optional signal-conditioning input filter sits in front of the weighting (see [Signal conditioning](signal-conditioning.md))
- **`PluginAWeighting` / `PluginCWeighting` / `PluginZWeighting`** — IIR frequency-weighting filters
- **`PluginFastTimeWeighting` / `PluginSlowTimeWeighting` / `PluginImpulseTimeWeighting`** — exponential time-weighting filters
- **`PluginSquare`** — instantaneous squaring (Pa → Pa²); the squaring stage for `eq`/`E`/peak chains, so every meter consumes Pa²
- **`PluginOctaveBand`** — arbitrary N/M-octave filter bank; outputs N channels
- **`LeqAccumulator` / `MaxAccumulator` / `MinAccumulator`** — whole-file/stream integrating meters (Pa² input)
- **`LeqMovingMeter` / `MaxMovingMeter` / `MinMovingMeter`** — sliding-window meters
- **`Reporter`** — a sink: columns registered via `add_columns` are sampled when `engine.on_record` fires, then written as CSV

## Assembly

`assemble_engine()` in `slm/assembly.py` is the factory that builds an engine and wires the
above components from a list of metric specs, returning `(engine, bindings)`; the caller
attaches a sink with `reporter.add_columns(bindings)` and `engine.on_record = reporter.record`.
Internally each metric is lowered `parse_metric → MetricSpec → plan_chain → ChainPlan`, and
`build_chain` walks that IR, reusing shared buses and plugins.

See the [Python API](python-api.md) for the mid- and low-level entry points that use these
components directly.
