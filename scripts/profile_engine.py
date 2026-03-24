"""Profile the SLM engine with a wide variety of meters over synthetic noise.

Usage:
    python scripts/profile_engine.py
    python scripts/profile_engine.py --seconds 10 --blocksize 512
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import numpy as np

from slm.io.controller import Controller
from slm.engine import Engine
from slm.assembly import parse_metric, build_chain
from slm.io.reporter import Reporter


# ---------------------------------------------------------------------------
# Synthetic noise controller
# ---------------------------------------------------------------------------

class NoiseController(Controller):
    """Generates blocks of white noise indefinitely, stopping after n_blocks."""

    def __init__(self, samplerate: int, blocksize: int, n_blocks: int, sensitivity: float = 1.0):
        super().__init__()
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._sensitivity = sensitivity
        self._n_blocks = n_blocks
        self._block_index = 0
        rng = np.random.default_rng(42)
        # Pre-generate all noise to avoid RNG overhead in the hot loop
        self._noise = rng.standard_normal((n_blocks, blocksize)).astype(np.float32) * 0.01

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    def read_block(self) -> tuple[np.ndarray, int]:
        if self._block_index >= self._n_blocks:
            raise StopIteration
        block = self._noise[self._block_index]
        idx = self._block_index
        self._block_index += 1
        return block[:, np.newaxis], idx   # shape (blocksize, 1) — matches FileController

    def stop(self):
        pass

    def calibrate(self, target_spl=94.0):
        pass


# ---------------------------------------------------------------------------
# Metric set
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    # Broadband accumulating Leq
    "LAeq", "LCeq", "LZeq",
    # Broadband accumulating max/min
    "LAFmax", "LASmax",
    "LCFmax", "LCSmax",
    "LZFmax", "LZSmax",
    # Broadband accumulating min
    "LAFmin", "LASmin",
    # Moving Leq (engine dt window)
    "LAeq_dt", "LCeq_dt", "LZeq_dt",
    # Moving max
    "LAFmax_dt", "LASmax_dt",
    # Longer moving windows
    "LAeq_1s", "LAeq_10s",
    # Sound exposure
    "LAE",
    # 1/1-octave band Leq (Z-weighted)
    "LZeq:bands:63-8000",
    # 1/1-octave band Leq (A-weighted)
    "LAeq:bands:63-8000",
    # 1/3-octave band Leq (A-weighted)
    "LAeq:bands:1/3:100-8000",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seconds: float, samplerate: int, blocksize: int):
    n_blocks = int(seconds * samplerate / blocksize)
    print(f"Profiling: {seconds:.0f}s audio | fs={samplerate} | blocksize={blocksize} | "
          f"n_blocks={n_blocks} | {len(METRIC_NAMES)} metrics")

    controller = NoiseController(samplerate=samplerate, blocksize=blocksize, n_blocks=n_blocks)
    engine = Engine(controller, dt=0.1)
    reporter = Reporter(precision=2)
    engine.reporter = reporter

    specs = [parse_metric(m) for m in METRIC_NAMES]
    build_chain(specs, engine, reporter)

    pr = cProfile.Profile()
    pr.enable()
    engine.run()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(40)
    print(s.getvalue())

    # Also print tottime-sorted for hot inner loops
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats(pstats.SortKey.TIME)
    ps2.print_stats(30)
    print("=== Sorted by self time ===")
    print(s2.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--samplerate", type=int, default=48000)
    parser.add_argument("--blocksize", type=int, default=4800)
    args = parser.parse_args()
    run(args.seconds, args.samplerate, args.blocksize)