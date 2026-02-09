from __future__ import annotations
from typing import Optional


import numpy as np

from plugin import Plugin, ID, find_plugin
from bus import Bus
from controller import Controller

class RequestRejected(Exception):
    pass

class ExecutionError(Exception):
    pass


class ExecutionContext:
    samplerate: int
    blocksize: int
    block_index: int
    block: np.ndarray
    cache: dict[ID, np.ndarray]

    @property
    def timestamp(self) -> float:
        """seconds passed since the beginning of the execution until the start of the block"""
        return self.block_index * self.blocksize / self.samplerate

    def __init__(self, samplerate, blocksize, block_index):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.block_index = block_index
        self.cache = dict()


class Engine:
    max_ctx: int = 10


    def __init__(self, controller):
        self._controller: Controller = controller
        self._busses: dict[str, Bus] = dict()
        self._supported_functions: list[tuple[str]] = []
        self._ctxs: list[ExecutionContext] = []


    def add_bus(self, name: str, root_type: type[Plugin]|None) -> Bus:
        bus = Bus(name=name, root_type=root_type)
        self._busses[name] = bus
        return bus

    def get_bus(self, name: str) -> Bus:
        # if name is None:
        #     name = "Z"

        try:
            return self._busses[name]
        except KeyError:
            raise KeyError(f"No bus named '{name}'")

    def add_plugin(self, ptype: type[Plugin], bus: str, source: Plugin | None) -> Plugin:
        if bus not in self._busses:
            raise Exception(f"Unknown bus {bus}")

        return self._busses[bus].add_plugin(ptype, source)


    def require(self, requirement: tuple[str]):
        if requirement in self._supported_functions:
            return

        # select bus with weighting
        try:
            bus = self.get_bus(requirement[0])
        except KeyError:
            bus = self.add_bus(requirement[0], find_plugin(requirement[0]))

        last = bus.root
        for i, req in enumerate(requirement[1:], start=1):
            for plugin in bus.plugins: # TODO: optimise by only searching through the outputs of the last plugin
                if plugin.input == last and plugin.function == req:
                    last = plugin
                    break
            else: # loop finished with no break -> no matching plugin found
                satisfied = False
                break
        else:
            # loop finished with no break -> all requirements were satisfied
            satisfied = True

        if not satisfied:
            # requirement req at index i is not satisfied
            for j in range(i, len(requirement)):
                req = requirement[j]
                # resolve Plugin
                PType = find_plugin(req)
                last = self.add_plugin(PType, bus.name, source=last)

        self._supported_functions.append(requirement)



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

        self._ctxs.append(
            ExecutionContext(blocksize=self._controller.blocksize,
                             samplerate=self._controller.samplerate,
                             block_index=block_index)
        )
        ctx = self._ctxs[-1]

        for bus in self._busses.values():
            for plugin in bus.plugins:
                plugin.process(ctx)

        # hook for logging
        # TODO

        # discard old contexts
        while len(self._ctxs) > self.max_ctx:
            self._ctxs.pop(0)













