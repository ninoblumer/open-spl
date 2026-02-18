from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from datetime import timedelta

import numpy as np

from slm.plugins.plugin import Plugin, TPlugin, PluginMeter, Meter
from slm.plugins.frequency_weighting import PluginFrequencyWeighting, PluginZWeighting

if TYPE_CHECKING:
    from slm.engine import Engine



class Bus:
    name: str
    frequency_weighting: PluginFrequencyWeighting
    plugins: list[Plugin]
    meters: list[Meter]
    block: np.ndarray
    engine: "Engine"
    dt: float = property(lambda self: self.engine.dt)

    samplerate: int = property(lambda self: self.engine.samplerate)
    blocksize: int = property(lambda self: self.engine.blocksize)
    sensitivity: float = property(lambda self: self.engine.sensitivity)

    def __init__(self, engine: "Engine", name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None):
        self.engine = engine
        self.name = name
        self.plugins = []
        self.meters = []
        self._counter = itertools.count(1)  # for numbering plugins with unique id
        self.block = np.zeros((1, self.blocksize))

        if frequency_weighting is None:
            frequency_weighting = PluginZWeighting

        self.frequency_weighting = self.add_plugin(frequency_weighting, width=1, input=self, zero_zi=True)

    def process(self, block: np.ndarray):
        self.frequency_weighting.process(block)

    def get(self) -> np.ndarray:
        return self.block

    def add_plugin(self, ptype: type[TPlugin], **kwargs) -> TPlugin:
        plugin = ptype(id=f"{self.name}{next(self._counter)}", bus=self, **kwargs)
        self.plugins.append(plugin)
        return plugin

    def add_meter(self, input: PluginMeter, **kwargs) -> Meter:
        meter = input.add_meter(**kwargs)
        self.meters.append(meter)
        return meter

    def log_block(self, block_index: int):
        timestamp = timedelta(seconds=block_index * self.blocksize / self.samplerate)
        for plugin in self.plugins:
            if isinstance(plugin, PluginMeter):
                for name, meter in plugin.meters.items():
                    reading = plugin.read_db(name)
                    if len(reading) == 1:
                        reading = reading[0]
                    else:
                        reading = list(reading)
                    print(f"{timestamp} {meter}: {reading:.1f} dB")

    def get_chain(self) -> list[Plugin | Bus | Meter]:
        return [self]

    def to_str(self):
        return f"Bus(name={self.name})"

    def __str__(self):
        return self.to_str()
