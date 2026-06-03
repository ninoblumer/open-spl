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
from slm.engine import Engine
from slm.assembly import parse_metric, build_chain
from slm.io.reporter import Reporter
from slm.io.noise_controller import NoiseController


# ---------------------------------------------------------------------------
# K2 loadout — mirrors realtime_benchmark.py LOADOUTS["K2"]
# ---------------------------------------------------------------------------

_K1 = [
    "LAeq", "LAFmax", "LASmax", "LAF", "LAS",
    "LCeq", "LCFmax", "LCSmax", "LCF", "LCS",
    "LZeq", "LZFmax", "LZSmax", "LZF", "LZS",
]

METRIC_NAMES = _K1 + [
    "LAeq:bands:1/3:31-16000", "LAFmax:bands:1/3:31-16000", "LASmax:bands:1/3:31-16000",
    "LAF:bands:1/3:31-16000",  "LAS:bands:1/3:31-16000",
    "LCeq:bands:1/3:31-16000", "LCFmax:bands:1/3:31-16000", "LCSmax:bands:1/3:31-16000",
    "LCF:bands:1/3:31-16000",  "LCS:bands:1/3:31-16000",
    "LZeq:bands:1/3:31-16000", "LZFmax:bands:1/3:31-16000", "LZSmax:bands:1/3:31-16000",
    "LZF:bands:1/3:31-16000",  "LZS:bands:1/3:31-16000",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seconds: float, samplerate: int, blocksize: int):
    n_blocks = int(seconds * samplerate / blocksize)
    print(f"Profiling: {seconds:.0f}s audio | fs={samplerate} | blocksize={blocksize} | "
          f"n_blocks={n_blocks} | {len(METRIC_NAMES)} metrics")

    controller = NoiseController(
        samplerate=samplerate, blocksize=blocksize,
        n_blocks=n_blocks, realtime=False, seed=42,
    )
    controller.set_sensitivity(1.0, unit="V")
    reporter = Reporter(precision=2)
    engine = Engine(controller, dt=0.1, reporter=reporter)

    specs = [parse_metric(m) for m in METRIC_NAMES]
    build_chain(specs, engine)

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