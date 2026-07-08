"""In-memory measurement results returned by ``run_measurement``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeasurementResults:
    """Measured levels held in memory (no disk round-trip).

    Produced by :meth:`slm.io.reporter.Reporter.results`.  Broadband metrics and
    band-split (RTA) metrics are kept separate, mirroring the two CSV families the
    reporter writes.  All timestamps are float seconds from the measurement start.

    Attributes
    ----------
    report:
        Final broadband value per metric label (the last logged row), e.g.
        ``{"LAeq": 64.6, "LZeq": 70.1}``.
    log:
        Per-``dt`` broadband rows.  Each row is a dict with ``"timestamp"`` (float
        seconds) plus one entry per metric label.
    rta_report:
        Final band-split value array per band-column label, e.g.
        ``{"LZeq": array([...])}`` aligned with :attr:`band_frequencies`.
    rta_log:
        Per-``dt`` band-split rows: ``"timestamp"`` (float seconds) plus one numpy
        array per band-column label.
    band_frequencies:
        Band center-frequency labels (e.g. ``"1k"``, ``"8k"``) per band-column
        label, aligned with the arrays in :attr:`rta_report` / :attr:`rta_log`.
    """

    report: dict[str, float] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    rta_report: dict[str, np.ndarray] = field(default_factory=dict)
    rta_log: list[dict] = field(default_factory=list)
    band_frequencies: dict[str, list[str]] = field(default_factory=dict)
