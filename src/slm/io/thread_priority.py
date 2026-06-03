"""Thread scheduling priority helpers for reducing OS preemption during measurement."""
from __future__ import annotations

import contextlib
import platform
import warnings
from typing import Callable, Generator


@contextlib.contextmanager
def high_priority() -> Generator[None, None, None]:
    """Raise the current thread's scheduling priority for the duration of the block.

    Reduces OS-scheduler preemption during the real-time processing loop.
    Falls back with a :mod:`warnings` warning if the process lacks the required
    permissions (e.g. SCHED_FIFO on Linux without CAP_SYS_NICE, or SCHED_RR on
    macOS without root).

    Platform behaviour
    ------------------
    Windows : ``SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)`` — no elevated
              permissions required.
    Linux   : ``SCHED_FIFO`` at maximum priority via ``os.sched_setscheduler``.
              Requires ``CAP_SYS_NICE`` or root.
    macOS   : ``SCHED_RR`` at maximum priority via ``pthread_setschedparam``.
              Requires root.
    """
    restore: Callable[[], None] | None = None
    system = platform.system()
    try:
        if system == "Windows":
            restore = _raise_windows()
        elif system == "Linux":
            restore = _raise_linux()
        elif system == "Darwin":
            restore = _raise_macos()
    except Exception as exc:
        warnings.warn(
            f"Could not raise thread priority on {system}: {exc}. "
            "Measurement may experience scheduler jitter.",
            RuntimeWarning,
            stacklevel=3,
        )

    try:
        yield
    finally:
        if restore is not None:
            try:
                restore()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Platform implementations
# ---------------------------------------------------------------------------

def _raise_windows() -> Callable[[], None]:
    import ctypes

    THREAD_PRIORITY_TIME_CRITICAL = 15
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetCurrentThread()
    old = kernel32.GetThreadPriority(handle)
    kernel32.SetThreadPriority(handle, THREAD_PRIORITY_TIME_CRITICAL)
    return lambda: kernel32.SetThreadPriority(handle, old)


def _raise_linux() -> Callable[[], None]:
    import os

    SCHED_FIFO = 1
    old_policy = os.sched_getscheduler(0)
    old_param = os.sched_getparam(0)
    max_prio = os.sched_get_priority_max(SCHED_FIFO)
    os.sched_setscheduler(0, SCHED_FIFO, os.sched_param(max_prio))
    return lambda: os.sched_setscheduler(0, old_policy, old_param)


def _raise_macos() -> Callable[[], None]:
    import ctypes
    import ctypes.util

    libpthread = ctypes.CDLL(ctypes.util.find_library("pthread"))
    libc = ctypes.CDLL(ctypes.util.find_library("c"))

    class _SchedParam(ctypes.Structure):
        _fields_ = [("sched_priority", ctypes.c_int)]

    SCHED_RR = 2

    libpthread.pthread_self.restype = ctypes.c_ulong
    tid = libpthread.pthread_self()

    old_policy = ctypes.c_int(0)
    old_param = _SchedParam(0)
    libpthread.pthread_getschedparam(
        tid, ctypes.byref(old_policy), ctypes.byref(old_param)
    )

    max_prio = libc.sched_get_priority_max(SCHED_RR)
    new_param = _SchedParam(max_prio)
    libpthread.pthread_setschedparam(tid, SCHED_RR, ctypes.byref(new_param))

    captured_old_policy = old_policy.value
    captured_old_param = _SchedParam(old_param.sched_priority)
    return lambda: libpthread.pthread_setschedparam(
        tid, captured_old_policy, ctypes.byref(captured_old_param)
    )
