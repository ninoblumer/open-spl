"""SLM measurement configuration: dataclass + TOML round-trip."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SLMConfig:
    """Measurement configuration."""

    metrics: list[str] = field(default_factory=list)
    dt: float = 1.0
    output: str = "output/measurement"

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
        unknown_meas = set(meas.keys()) - {"dt", "output"}
        if unknown_meas:
            raise ValueError(f"Unknown keys in [measurement]: {unknown_meas}")

        metrics_sec = data.get("metrics", {})
        unknown_metrics = set(metrics_sec.keys()) - {"require"}
        if unknown_metrics:
            raise ValueError(f"Unknown keys in [metrics]: {unknown_metrics}")

        return cls(
            metrics=list(metrics_sec.get("require", [])),
            dt=float(meas.get("dt", 1.0)),
            output=str(meas.get("output", "output/measurement")),
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

        content = (
            "[measurement]\n"
            f"dt     = {self.dt}\n"
            f'output = "{self.output}"\n'
            "\n"
            "[metrics]\n"
            f"require = {metrics_value}\n"
        )
        path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, metrics: list[str], dt: float, output: str) -> "SLMConfig":
        """Construct from parsed command-line arguments."""
        return cls(metrics=list(metrics), dt=dt, output=output)
