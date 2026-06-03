"""
Real-time load benchmark for the SLM engine.

Measures two quantities across a matrix of loadout × block size:

Quantity A — processing utilisation (realtime=False)
    Per-block rho = t_proc / t_budget, where t_budget = blocksize / samplerate.
    Reports median, p99, and max rho.
    Motivating standard: IEC 61260-1 §5.14.4 (real-time sampled-data filters must
    complete on average within the block time budget so every sample has equal weight).

Quantity B — real-time deadline violations (realtime=True)
    Overrun count and rate after running for rt_duration_s of audio time at real-time
    pace.  Run only for cells where Quantity A indicates borderline behaviour (rho_median
    roughly 50%–100%) — cells well below that range cannot overrun by construction.
    Motivating standard: IEC 61260-1 §5.14.4.

Usage
-----
    python -m slm.tests.realtime_load [--samplerate 48000] [--blocksizes 128 256 …]
        [--loadouts K0 K1 K2] [--duration 300] [--rt-duration 3600]
        [--out results/] [--run-b K2:256 K2:512] [--skip-a]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Loadout definitions
# ---------------------------------------------------------------------------

# NOTE: IEC 61672-1 §3.3.52 defines LCpeak as the maximum instantaneous
# C-weighted level (no time-weighting).  The current metric-name syntax has no
# "peak" type, so K1 uses LCFmax / LCSmax as the closest available proxy.
# When a native LCpeak metric is added to the assembly, update K1 here.

_K0 = [
    # IEC 61672-1 §5.1.9 floor: A-weighting mandatory; at minimum one
    # time-weighted level (LAF) and/or one integrating-averaging level (LAeq).
    "LAF",
    "LAeq",
]

_K1 = [
    # Typical Class-1 handheld: A, C, Z with F- and S-time-weighting, Leq, max.
    "LAeq", "LAFmax", "LASmax", "LAF", "LAS",
    "LCeq", "LCFmax", "LCSmax", "LCF", "LCS",
    "LZeq", "LZFmax", "LZSmax", "LZF", "LZS",
]

_K2 = _K1 + [
    # K1 plus the full IEC 61260-1 third-octave filter bank on each weighting,
    # with all the same metric types as the broadband section (Leq, Fmax, Smax, F, S).
    "LAeq:bands:1/3:31-16000", "LAFmax:bands:1/3:31-16000", "LASmax:bands:1/3:31-16000",
    "LAF:bands:1/3:31-16000",  "LAS:bands:1/3:31-16000",
    "LCeq:bands:1/3:31-16000", "LCFmax:bands:1/3:31-16000", "LCSmax:bands:1/3:31-16000",
    "LCF:bands:1/3:31-16000",  "LCS:bands:1/3:31-16000",
    "LZeq:bands:1/3:31-16000", "LZFmax:bands:1/3:31-16000", "LZSmax:bands:1/3:31-16000",
    "LZF:bands:1/3:31-16000",  "LZS:bands:1/3:31-16000",
]

LOADOUTS: dict[str, list[str]] = {
    "K0": _K0,
    "K1": _K1,
    "K2": _K2,
}

DEFAULT_BLOCKSIZES: list[int] = [128, 256, 512, 1024, 4096]

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(loadout_name: str, blocksize: int, samplerate: int,
                 dt: float = 1.0, realtime: bool = False):
    from slm.io.noise_controller import NoiseController
    from slm.engine import Engine
    from slm.io.reporter import Reporter
    from slm.assembly import parse_metric, build_chain

    controller = NoiseController(samplerate=samplerate, blocksize=blocksize,
                                  realtime=realtime)
    controller.set_sensitivity(1.0, unit="V")
    reporter = Reporter()
    engine = Engine(controller, dt=dt, reporter=reporter)
    specs = [parse_metric(m) for m in LOADOUTS[loadout_name]]
    build_chain(specs, engine)
    return engine, controller, reporter


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

def _run_loop(
    engine,
    controller,
    reporter,
    target_blocks: int,
    timed: bool,
    record_queue: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drive the processing loop for target_blocks blocks.

    read_block() is always outside the timed region.  If timed=True, every
    block's processing time is recorded.  If record_queue=True, the queue
    depth is sampled after each read_block() call.

    Returns (t_procs, queue_depths), each None if not requested.
    """
    engine._last_timestamp = None
    t_procs: list[float] = []
    queue_depths: list[int] = []

    for _ in range(target_blocks):
        block, block_idx = controller.read_block()

        if record_queue:
            queue_depths.append(controller._queue.qsize())

        # ── timed region ──────────────────────────────────────────────────
        if timed:
            t0 = time.perf_counter()

        b = block.transpose()
        for bus in engine._busses.values():
            bus.process(b)
        ts = timedelta(seconds=block_idx * engine.blocksize / engine.samplerate)
        engine._last_timestamp = ts
        reporter.record(ts, engine._dt)

        if timed:
            t_procs.append(time.perf_counter() - t0)
        # ─────────────────────────────────────────────────────────────────

    return (
        np.array(t_procs) if timed else None,
        np.array(queue_depths, dtype=np.int32) if record_queue else None,
    )


# ---------------------------------------------------------------------------
# Overhead calibration
# ---------------------------------------------------------------------------

def calibrate_overhead(n: int = 2000) -> float:
    """Measure the per-call overhead of perf_counter() bookends with a no-op body.

    Returns the median elapsed time in seconds.  Used to flag when timer
    contamination is non-negligible relative to t_budget (worst case: K0 at
    the smallest block size).
    """
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
    return float(np.median(samples))


# ---------------------------------------------------------------------------
# Quantity A — processing utilisation
# ---------------------------------------------------------------------------

def measure_quantity_a(
    loadout_name: str,
    blocksize: int,
    samplerate: int,
    duration_s: float,
    overhead_s: float,
) -> tuple[dict, np.ndarray]:
    """Per-block processing utilisation at full speed (realtime=False).

    rho = t_proc / t_budget, where t_budget = blocksize / samplerate.
    GC is NOT suppressed and no warm-up blocks are discarded — all blocks from
    block 0 count, because a real SLM must keep up from the first sample.
    Motivating standard: IEC 61260-1 §5.14.4.
    """
    engine, controller, reporter = _make_engine(
        loadout_name, blocksize, samplerate, realtime=False
    )
    t_budget = blocksize / samplerate
    target_blocks = int(duration_s * samplerate / blocksize)

    controller.start()
    try:
        t_procs, _ = _run_loop(engine, controller, reporter, target_blocks, timed=True)
    finally:
        controller.stop()

    assert t_procs is not None
    rho = t_procs / t_budget
    n = len(rho)

    summary = {
        "loadout": loadout_name,
        "blocksize": blocksize,
        "samplerate": samplerate,
        "block_count": n,
        "overhead_s": f"{overhead_s:.3e}",
        "overhead_frac": f"{overhead_s / t_budget:.4f}",
        "rho_median": float(np.median(rho)),
        "rho_p99": float(np.percentile(rho, 99)),
        "rho_max": float(np.max(rho)),
        "warn_low_n": n < 3500,
    }
    return summary, rho


# ---------------------------------------------------------------------------
# Quantity B — real-time deadline violations
# ---------------------------------------------------------------------------

def measure_quantity_b(
    loadout_name: str,
    blocksize: int,
    samplerate: int,
    rt_duration_s: float,
) -> tuple[dict, np.ndarray]:
    """Real-time overrun count and per-block queue depth for rt_duration_s of audio.

    Uses NoiseController(realtime=True), which paces block delivery to
    wall-clock time and increments overruns each time a block arrives late.
    Queue depth is sampled after each read_block() call.
    Motivating standard: IEC 61260-1 §5.14.4.

    Returns (summary_row, queue_depths).
    """
    engine, controller, reporter = _make_engine(
        loadout_name, blocksize, samplerate, dt=1.0, realtime=True
    )
    target_blocks = int(rt_duration_s * samplerate / blocksize)

    controller.start()
    try:
        _, queue_depths = _run_loop(
            engine, controller, reporter, target_blocks,
            timed=False, record_queue=True,
        )
    finally:
        controller.stop()

    assert queue_depths is not None
    overruns = controller.overruns
    summary = {
        "loadout": loadout_name,
        "blocksize": blocksize,
        "samplerate": samplerate,
        "block_count": target_blocks,
        "overruns": overruns,
        "overrun_rate": overruns / target_blocks if target_blocks > 0 else float("nan"),
        "mean_queue_depth": float(np.mean(queue_depths)),
        "rt_duration_s": rt_duration_s,
    }
    return summary, queue_depths


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_rho_csv(rho: np.ndarray, path: Path) -> None:
    """Write per-block rho values; row number is the block index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("rho\n")
        for v in rho:
            f.write(f"{v:.6f}\n")
    print(f"  wrote {path}", file=sys.stderr)


def _write_queue_csv(queue_depths: np.ndarray, path: Path) -> None:
    """Write per-block queue depth values; row number is the block index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("queue_depth\n")
        for v in queue_depths:
            f.write(f"{v}\n")
    print(f"  wrote {path}", file=sys.stderr)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}", file=sys.stderr)


def _print_table_a(rows_a: list[dict], blocksizes: list[int]) -> None:
    loadout_names = list(dict.fromkeys(r["loadout"] for r in rows_a))
    cell: dict[tuple[str, int], str] = {}
    for r in rows_a:
        flag = "*" if r["warn_low_n"] else " "
        cell[(r["loadout"], r["blocksize"])] = (
            f"{r['rho_median']:.2f}/{r['rho_p99']:.2f}{flag}"
        )

    col_w = 13
    header = f"{'Loadout':<8} |" + "".join(
        f" {'bs='+str(bs):<{col_w}}" for bs in blocksizes
    )
    sep = "-" * len(header)
    print()
    print("Quantity A  rho_median / rho_p99   (* = fewer than 3500 blocks)")
    print(sep)
    print(header)
    print(sep)
    for lo in loadout_names:
        row_str = f"{lo:<8} |"
        for bs in blocksizes:
            row_str += f" {cell.get((lo, bs), '--/--'):<{col_w}}"
        print(row_str)
    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tests/benchmark/realtime_load.py",
        description="SLM real-time capability benchmark (IEC 61260-1 §5.14.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--samplerate", type=int, default=48_000, metavar="HZ",
        help="Sample rate used for all measurements.",
    )
    p.add_argument(
        "--blocksizes", nargs="+", type=int, default=DEFAULT_BLOCKSIZES, metavar="N",
        help="Block sizes to sweep.",
    )
    p.add_argument(
        "--loadouts", nargs="+", default=list(LOADOUTS),
        choices=list(LOADOUTS), metavar="K",
        help="Loadouts to run (K0 K1 K2).",
    )
    p.add_argument(
        "--duration", type=float, default=300.0, metavar="S",
        help="Quantity A: audio duration per cell in seconds.",
    )
    p.add_argument(
        "--rt-duration", type=float, default=3600.0, metavar="S",
        help="Quantity B: realtime audio duration per cell in seconds.",
    )
    p.add_argument(
        "--out", default="results", metavar="DIR",
        help="Output directory for CSV files.",
    )
    p.add_argument(
        "--run-b", nargs="*", default=None, metavar="LOADOUT:BLOCKSIZE",
        help=(
            "Quantity B cells to run. "
            "Bare --run-b (no tokens) runs B for all loadout×blocksize combinations. "
            "Pass specific cells to limit scope, e.g. --run-b K2:256 K2:512. "
            "If omitted entirely, Quantity B is skipped."
        ),
    )
    p.add_argument(
        "--skip-a", action="store_true",
        help=(
            "Skip Quantity A entirely.  Requires --run-b so that at least one "
            "quantity is measured."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.skip_a and args.run_b is None:
        sys.exit("--skip-a requires --run-b (nothing to measure otherwise).")

    out_dir = Path(args.out)
    blocksizes_sorted = sorted(args.blocksizes)

    # Overhead calibration — once per run (skipped when A is not measured)
    overhead_s = 0.0
    if not args.skip_a:
        print("Calibrating timer overhead …", file=sys.stderr)
        overhead_s = calibrate_overhead()
        t_budget_min = min(blocksizes_sorted) / args.samplerate
        if overhead_s > 0.02 * t_budget_min:
            print(
                f"WARNING: perf_counter overhead {overhead_s * 1e6:.1f} µs exceeds 2 % of"
                f" t_budget for bs={min(blocksizes_sorted)}"
                f" ({t_budget_min * 1e3:.3f} ms). Load figures for small block"
                " sizes may be inflated.",
                file=sys.stderr,
            )
        else:
            print(f"  overhead = {overhead_s * 1e6:.2f} µs  (negligible)", file=sys.stderr)

    # ── Quantity A ──────────────────────────────────────────────────────────
    rows_a: list[dict] = []
    if args.skip_a:
        print("Quantity A skipped (--skip-a).", file=sys.stderr)
    else:
        total_cells = len(args.loadouts) * len(blocksizes_sorted)
        cell_idx = 0
        for loadout in args.loadouts:
            for bs in blocksizes_sorted:
                cell_idx += 1
                print(
                    f"[{cell_idx}/{total_cells}] A  {loadout}  bs={bs}"
                    f"  duration={args.duration:.0f}s …",
                    end="  ", flush=True, file=sys.stderr,
                )
                row, rho = measure_quantity_a(loadout, bs, args.samplerate,
                                              args.duration, overhead_s)
                rows_a.append(row)
                _write_rho_csv(rho, out_dir / "rho" / f"{loadout}_{bs}.csv")
                print(
                    f"rho {row['rho_median']:.3f} / {row['rho_p99']:.3f}"
                    f" / {row['rho_max']:.3f}  n={row['block_count']}"
                    + ("  [LOW N — p99 unreliable]" if row["warn_low_n"] else ""),
                    file=sys.stderr,
                )

        _write_csv(rows_a, out_dir / "quantity_a.csv")
        _print_table_a(rows_a, blocksizes_sorted)

    # ── Quantity B ──────────────────────────────────────────────────────────
    if args.run_b is None:
        print(
            "Quantity B skipped.  Re-run with --run-b LOADOUT:BLOCKSIZE to measure"
            " realtime overruns for borderline cells.",
            file=sys.stderr,
        )
        return

    if not args.run_b:
        # bare --run-b with no tokens → run B for all loadout×blocksize combinations
        b_cells = [(lo, bs) for lo in args.loadouts for bs in blocksizes_sorted]
    else:
        b_cells = []
        for token in args.run_b:
            try:
                lo, bs_str = token.split(":", 1)
                bs = int(bs_str)
            except ValueError:
                sys.exit(
                    f"Invalid --run-b token {token!r}. Expected LOADOUT:BLOCKSIZE, e.g. K2:256"
                )
            if lo not in LOADOUTS:
                sys.exit(f"Unknown loadout {lo!r} in --run-b. Valid: {list(LOADOUTS)}")
            b_cells.append((lo, bs))

    rows_b: list[dict] = []
    for i, (lo, bs) in enumerate(b_cells, 1):
        print(
            f"[{i}/{len(b_cells)}] B  {lo}  bs={bs}"
            f"  rt_duration={args.rt_duration:.0f}s …",
            end="  ", flush=True, file=sys.stderr,
        )
        row, queue_depths = measure_quantity_b(lo, bs, args.samplerate, args.rt_duration)
        rows_b.append(row)
        _write_queue_csv(queue_depths, out_dir / "queue_depth" / f"{lo}_{bs}.csv")
        print(
            f"overruns={row['overruns']}  rate={row['overrun_rate']:.2e}"
            f"  mean_queue={row['mean_queue_depth']:.2f}",
            file=sys.stderr,
        )

    _write_csv(rows_b, out_dir / "quantity_b.csv")


if __name__ == "__main__":
    main()
