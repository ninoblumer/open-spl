from __future__ import annotations
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING

from slm.bus import Bus

if TYPE_CHECKING:
    from slm.frequency_weighting import PluginFrequencyWeighting
    from slm.io.controller import Controller
    from slm.io.reporter import Reporter


class Engine:
    samplerate: int = property(lambda self: self._controller.samplerate)
    blocksize: int = property(lambda self: self._controller.blocksize)
    sensitivity: float = property(lambda self: self._controller.sensitivity)
    dt: float = property(lambda self: self._dt)

    def __init__(self, controller, dt: float=0.1):
        self._controller: Controller = controller
        self._busses: dict[str, Bus] = dict()
        self._dt = dt
        self.reporter: Reporter | None = None

    def add_bus(self, name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None) -> Bus:
        bus = Bus(engine=self, name=name, frequency_weighting=frequency_weighting)
        self._busses[name] = bus
        return bus

    def get_bus(self, name: str) -> Bus:
        try:
            return self._busses[name]
        except KeyError:
            raise KeyError(f"No bus named '{name}'")

    def run(self):
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
        while True:
            try:
                self._process_block()
            except StopIteration:
                break
        # Force a final snapshot so the report always reflects the fully-accumulated state,
        # even when the file duration is not an exact multiple of dt.
        if self.reporter is not None and self._last_timestamp is not None:
            self.reporter.record(self._last_timestamp, 0)

    def _process_block(self) -> None:
        block, block_index = self._controller.read_block()
        block = block.transpose()

        for bus in self._busses.values():
            bus.process(block)

        if self.reporter is not None:
            timestamp = timedelta(seconds=block_index * self.blocksize / self.samplerate)
            self._last_timestamp = timestamp
            self.reporter.record(timestamp, self._dt)

    def stop(self):
        self._controller.stop()


