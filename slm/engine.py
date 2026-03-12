from __future__ import annotations
from datetime import timedelta
from typing import TYPE_CHECKING

from slm.bus import Bus
from slm.meter import Meter
from slm.plugin_meter import PluginMeter

if TYPE_CHECKING:
    from slm.frequency_weighting import PluginFrequencyWeighting
    from slm.plugin import Plugin
    from slm.controller import Controller
    from slm.reporter import Reporter


class Engine:
    samplerate: int = property(lambda self: self._controller.samplerate)
    blocksize: int = property(lambda self: self._controller.blocksize)
    sensitivity: float = property(lambda self: self._controller.sensitivity)
    dt: float = property(lambda self: self._dt)

    def __init__(self, controller, dt: float=0.1):
        self._controller: Controller = controller
        self._busses: dict[str, Bus] = dict()
        self._supported_functions: list[tuple[str]] = []
        self._dt = dt
        self.reporter: Reporter | None = None

    def add_bus(self, name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None) -> Bus:
        bus = Bus(engine=self, name=name, frequency_weighting=frequency_weighting)
        self._busses[name] = bus
        return bus

    def get_bus(self, name: str) -> Bus:
        # if name is None:
        #     name = "Z"

        try:
            return self._busses[name]
        except KeyError:
            raise KeyError(f"No bus named '{name}'")

    def add_plugin(self, ptype: type[Plugin], bus: str, **kwargs) -> Plugin:
        if bus not in self._busses:
            raise Exception(f"Unknown bus {bus}")

        return self._busses[bus].add_plugin(ptype, **kwargs)

    def add_meter(self, plugin: PluginMeter, **kwargs) -> Meter:
        bus = plugin.bus
        if bus not in self._busses.values():
            raise Exception(f"Unknown bus {bus.name}")

        return bus.add_meter(input=plugin, **kwargs)


    # def require(self, requirement: tuple[str]):
    #     if requirement in self._supported_functions:
    #         return
    #
    #     # select bus with weighting
    #     try:
    #         bus = self.get_bus(requirement[0])
    #     except KeyError:
    #         bus = self.add_bus(requirement[0], find_plugin(requirement[0]))
    #
    #     last_plugin = bus.frequency_weighting
    #     last_requirement = 1
    #     for i, req in enumerate(requirement[1:], start=1):
    #         for plugin in bus.plugins:
    #             if plugin.input == last_plugin and plugin.function == req:
    #                 last_plugin = plugin
    #                 break
    #         else:  # loop finished with no break -> no matching plugin found
    #             satisfied = False
    #             last_requirement = i
    #             break
    #     else:
    #         # loop finished with no break -> all requirements were satisfied
    #         satisfied = True
    #
    #     if not satisfied:
    #         # requirement req at index i is not satisfied
    #         for j in range(last_requirement, len(requirement)):
    #             req = requirement[j]
    #             # resolve Plugin
    #             PType = find_plugin(req)
    #             last_plugin = self.add_plugin(PType, bus.name, input=last_plugin)
    #
    #     self._supported_functions.append(requirement)

    # def require(self, frequency_weighting: tuple[type[PluginFrequencyWeighting],  | None = None, time_reduction: type[PluginMeter] | None = None) -> None:
    #     if time_reduction is None:
    #         raise RequestRejected(f"Time-reduction must be specified")

    def require(self, *args, **kwargs):
        raise NotImplementedError()


    def run(self):
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

        if block is None:
            raise StopIteration

        for bus in self._busses.values():
            bus.process(block)

        if self.reporter is not None:
            timestamp = timedelta(seconds=block_index * self.blocksize / self.samplerate)
            self._last_timestamp = timestamp
            self.reporter.record(timestamp, self._dt)

    def stop(self):
        self._controller.stop()


