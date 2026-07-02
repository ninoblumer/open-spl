"""SLM measurement configuration: dataclass + TOML round-trip."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from slm.io.realtime_controller import DEFAULT_QUEUE_MAXSIZE


def _normalize_signal_conditioning(value) -> str | None:
    """Validate and canonicalize a signal-conditioning spec; ``None``/``"none"`` → ``None``.

    Accepts a registered preset name (e.g. ``"xl2"``) or a custom
    ``"F_HPF N_HPF F_LPF N_LPF"`` spec. Delegates to
    :func:`slm.app.cli.normalize_signal_conditioning`, so a bad config value
    fails fast at load time.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"signal_conditioning must be a string, got {value!r}")
    from slm.app.cli import normalize_signal_conditioning
    return normalize_signal_conditioning(value)


@dataclass
class SLMConfig:
    """Measurement configuration."""

    metrics: list[str] = field(default_factory=list)
    dt: float = 1.0
    output: str = "output/measurement"
    warmup: float = 0.0
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE
    signal_conditioning: str | None = None
    """Signal-conditioning filter to insert in front of every frequency weighting:
    a preset name (e.g. ``"xl2"``) or a custom ``"F_HPF N_HPF F_LPF N_LPF"``
    band-limiting spec, or ``None`` for an unconditioned chain. Resolved to a
    plugin factory by ``slm.app.cli.resolve_signal_conditioning``."""

    # ------------------------------------------------------------------
    # TOML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "SLMConfig":
        """Load configuration from a TOML file.

        Raises :exc:`ValueError` for unknown keys (strict validation).
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)

        unknown_sections = set(data.keys()) - {"measurement", "metrics"}
        if unknown_sections:
            raise ValueError(f"Unknown TOML sections: {unknown_sections}")

        meas = data.get("measurement", {})
        unknown_meas = set(meas.keys()) - {
            "dt", "output", "warmup", "queue_maxsize", "signal_conditioning"
        }
        if unknown_meas:
            raise ValueError(f"Unknown keys in [measurement]: {unknown_meas}")

        metrics_sec = data.get("metrics", {})
        unknown_metrics = set(metrics_sec.keys()) - {"require"}
        if unknown_metrics:
            raise ValueError(f"Unknown keys in [metrics]: {unknown_metrics}")

        require = metrics_sec.get("require", [])
        if not isinstance(require, list) or not all(isinstance(m, str) for m in require):
            raise ValueError(
                f"[metrics] require must be a list of strings, got {require!r}"
            )

        dt = float(meas.get("dt", 1.0))
        if dt <= 0:
            raise ValueError(f"[measurement] dt must be positive, got {dt}")

        queue_maxsize = int(meas.get("queue_maxsize", DEFAULT_QUEUE_MAXSIZE))
        if queue_maxsize < 0:
            raise ValueError(f"[measurement] queue_maxsize must be >= 0, got {queue_maxsize}")

        warmup = float(meas.get("warmup", 0.0))
        if warmup < 0:
            raise ValueError(f"[measurement] warmup must be non-negative, got {warmup}")

        conditioning = _normalize_signal_conditioning(meas.get("signal_conditioning"))

        return cls(
            metrics=list(require),
            dt=dt,
            output=str(meas.get("output", "output/measurement")),
            warmup=warmup,
            queue_maxsize=queue_maxsize,
            signal_conditioning=conditioning,
        )

    def to_toml(self, path: str | Path) -> None:
        """Write configuration to a TOML file (no external library required)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.metrics:
            items = ",\n".join(f'    "{m}"' for m in self.metrics)
            metrics_value = f"[\n{items},\n]"
        else:
            metrics_value = "[]"

        conditioning_line = (
            f'signal_conditioning = "{self.signal_conditioning}"\n'
            if self.signal_conditioning else ""
        )
        content = (
            "[measurement]\n"
            f"dt            = {self.dt}\n"
            f'output        = "{self.output}"\n'
            f"warmup        = {self.warmup}\n"
            f"queue_maxsize = {self.queue_maxsize}\n"
            f"{conditioning_line}"
            "\n"
            "[metrics]\n"
            f"require = {metrics_value}\n"
        )
        path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, metrics: list[str], dt: float, output: str,
                  queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
                  warmup: float = 0.0,
                  signal_conditioning: str | None = None) -> "SLMConfig":
        """Construct from parsed command-line arguments."""
        return cls(metrics=list(metrics), dt=dt, output=output,
                   queue_maxsize=queue_maxsize, warmup=warmup,
                   signal_conditioning=_normalize_signal_conditioning(signal_conditioning))
