from __future__ import annotations

import numpy as np

from slm.plugins.frequency_weighting import PluginFrequencyWeighting
from slm.plugins.plugin import Plugin
from slm.bus import Bus
from slm.controller import Controller
from slm.plugins.time_weighting import PluginTimeWeighting
from slm.exceptions import ExecutionError, RequestRejected

class Engine:
    samplerate: int = property(lambda self: self._controller.samplerate)
    blocksize: int = property(lambda self: self._controller.blocksize)
    sensitivity: float = property(lambda self: self._controller.sensitivity)

    def __init__(self, controller):
        self._controller: Controller = controller
        self._busses: dict[str, Bus] = dict()
        self._supported_functions: list[tuple[str]] = []

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

    def add_plugin(self, ptype: type[Plugin], bus: str, input: Plugin, **kwargs) -> Plugin:
        if bus not in self._busses:
            raise Exception(f"Unknown bus {bus}")

        return self._busses[bus].add_plugin(ptype, input, **kwargs)

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
        while True:
            try:
                self._process_block()
            except StopIteration:
                break

        # hook for reporting
        # TODO

    def _process_block(self) -> None:
        block, block_index = self._controller.read_block()

        if block is None:
            raise StopIteration

        for bus in self._busses.values():
            bus.process(block)

        # hook for logging
        for bus in self._busses.values():
            # todo use yield ore so
            bus.log_block()


