from __future__ import annotations
import itertools
from typing import TYPE_CHECKING

import numpy as np


from slm.plugins.plugin import Plugin, TPlugin, PluginMeter
from slm.plugins.frequency_weighting import PluginFrequencyWeighting, PluginZWeighting

if TYPE_CHECKING:
    from slm.engine import Engine



class Bus:
    name: str
    frequency_weighting: PluginFrequencyWeighting
    plugins: set[Plugin]
    block: np.ndarray
    engine: "Engine"

    samplerate: int = property(lambda self: self.engine.samplerate)
    blocksize: int = property(lambda self: self.engine.blocksize)
    sensitivity: float = property(lambda self: self.engine.sensitivity)

    def __init__(self, engine: "Engine", name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None):
        self.engine = engine
        self.name = name
        self.plugins = set()
        self._counter = itertools.count(1)  # for numbering plugins with unique id
        self.block = np.zeros((1, self.blocksize))

        if frequency_weighting is None:
            frequency_weighting = PluginZWeighting

        self.frequency_weighting = self.add_plugin(frequency_weighting, input=self)

    def process(self, block: np.ndarray):
        self.block = block
        for plugin in self.plugins: # relies on the plugins being inserted in the order they need to be executed
            plugin.process()

    def get(self) -> np.ndarray:
        return self.block

    def add_plugin(self, ptype: type[TPlugin], input, **kwargs) -> TPlugin:
        plugin = ptype(id=f"{self.name}{next(self._counter)}", input=input, bus=self, **kwargs)
        self.plugins.add(plugin)
        return plugin

    def log_block(self):
        for plugin in self.plugins:
            if isinstance(plugin, PluginMeter):
                reading = plugin.read_db()
                if len(reading) == 1:
                    reading = reading[0]
                else:
                    reading = list(reading)
                print(f"{plugin}: {reading:.1f} dB")
