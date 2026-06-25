from __future__ import annotations
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Callable

from slm.bus import Bus

if TYPE_CHECKING:
    from slm.frequency_weighting import PluginFrequencyWeighting
    from slm.io.controller import Controller


# A record callback is invoked once per processed block with (timestamp, dt).  It
# is how the engine signals "snapshot now" to a sink (e.g. a Reporter) without
# knowing what the sink is.  The default is a no-op, so an engine with no sink
# attached simply runs and updates its meters.
RecordCallback = Callable[[timedelta, float], None]


def _noop_record(timestamp: timedelta, dt: float) -> None:
    pass


class Engine:
    samplerate: int = property(lambda self: self._controller.samplerate)
    blocksize: int = property(lambda self: self._controller.blocksize)
    sensitivity: float = property(lambda self: self._controller.sensitivity)
    dt: float = property(lambda self: self._dt)

    def __init__(self, controller, dt: float = 0.1,
                 on_record: "RecordCallback | None" = None):
        self._controller: Controller = controller
        self._busses: dict[str, Bus] = dict()
        self._dt = dt
        self._timestamp_offset: timedelta = timedelta(0)
        # Sink callback invoked each block; the director (or a test) may also
        # assign ``engine.on_record`` after construction.  Defaults to a no-op.
        self.on_record: RecordCallback = on_record or _noop_record

    def add_bus(self, name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None) -> Bus:
        bus = Bus(engine=self, name=name, frequency_weighting=frequency_weighting)
        self._busses[name] = bus
        return bus

    def get_bus(self, name: str) -> Bus:
        try:
            return self._busses[name]
        except KeyError:
            raise KeyError(f"No bus named '{name}'")

    def run(self, duration: float | None = None, warmup: float = 0.0):
        """Process blocks until the source is exhausted or *duration* is reached.

        *warmup* (seconds), if positive, runs a settling phase first: blocks are
        processed without being logged, then every meter is reset so the
        measurement starts from a settled filter/time-weighting state with the
        accumulators back at zero.  Logged timestamps restart at 0 after warm-up,
        and *duration* measures the post-warm-up length.

        *duration* (seconds), if given, stops the run once the accumulated signal
        spans at least that long.  Stopping happens only on block edges, so the
        actual measured length is rounded up to the next whole block — the run
        covers at least *duration* seconds.
        """
        block_duration = self.blocksize / self.samplerate
        if self._dt < block_duration:
            warnings.warn(
                f"dt={self._dt:.4g}s is shorter than one block ({block_duration:.4g}s at "
                f"blocksize={self.blocksize}, fs={self.samplerate}Hz). "
                f"Logging resolution is limited to one entry per block.",
                UserWarning,
                stacklevel=2,
            )
        self._last_timestamp: timedelta | None = None
        self._timestamp_offset = timedelta(0)

        # Block counts drive warm-up/duration cut-offs (not the µs-rounded
        # timestamp), so the stop block is exact regardless of blocksize.

        # Warm-up phase: settle the chain, log nothing, then zero the meters and
        # rebase timestamps so the measurement proper begins at t=0.
        if warmup and warmup > 0:
            processed = 0
            while True:
                try:
                    self._process_block(record=False)
                except StopIteration:
                    # Source ended during warm-up: nothing left to measure.
                    self.reset_meters()
                    return
                processed += 1
                if processed * block_duration >= warmup:
                    break
            self.reset_meters()
            self._timestamp_offset = self._last_timestamp + timedelta(seconds=block_duration)
            self._last_timestamp = None

        processed = 0
        while True:
            try:
                self._process_block()
            except StopIteration:
                break
            processed += 1
            if duration is not None and processed * block_duration >= duration:
                break
        # Force a final snapshot so the report always reflects the fully-accumulated state,
        # even when the file duration is not an exact multiple of dt.
        if self._last_timestamp is not None:
            self.on_record(self._last_timestamp, 0)

    def reset_meters(self) -> None:
        """Reset every meter's accumulated state, leaving filter/time-weighting
        state intact.  Used after a warm-up period."""
        for bus in self._busses.values():
            for plugin in bus.plugins:
                for meter in getattr(plugin, "meters", {}).values():
                    meter.reset()

    def _process_block(self, record: bool = True) -> None:
        block, block_index = self._controller.read_block()
        block = block.transpose()

        if block.shape[0] > 1:
            raise ValueError(
                f"Engine received a {block.shape[0]}-channel block; only mono (1 channel) "
                "is supported. Use a controller that extracts a single channel, or create "
                "one engine per channel."
            )

        for bus in self._busses.values():
            bus.process(block)

        timestamp = (timedelta(seconds=block_index * self.blocksize / self.samplerate)
                     - self._timestamp_offset)
        self._last_timestamp = timestamp
        if record:
            self.on_record(timestamp, self._dt)

    def stop(self):
        self._controller.stop()


