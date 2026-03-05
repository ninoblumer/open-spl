from __future__ import annotations
import itertools
from typing import TYPE_CHECKING

import numpy as np

from slm.processing_element import ProcessingElement
from slm.frequency_weighting import PluginFrequencyWeighting, PluginZWeighting

if TYPE_CHECKING:
    from slm.plugin import Plugin, TPlugin
    from slm.meter import Meter, TMeter




class Bus(ProcessingElement):
    name: str
    frequency_weighting: PluginFrequencyWeighting
    plugins: list[Plugin]
    # meters: list[Meter] # meters are handled by plugins
    block: np.ndarray
    engine: "Engine"
    dt: float = property(lambda self: self.engine.dt)

    samplerate: int = property(lambda self: self.engine.samplerate)
    blocksize: int = property(lambda self: self.engine.blocksize)
    sensitivity: float = property(lambda self: self.engine.sensitivity)

    def __init__(self, engine: "Engine", name: str, frequency_weighting: type[PluginFrequencyWeighting] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.name = name
        self.plugins = []
        self.meters = []
        # self._counter = itertools.count(1)  # for numbering plugins with unique id
        self.block = np.zeros((1, self.blocksize))

        if frequency_weighting is None:
            frequency_weighting = PluginZWeighting

        self.frequency_weighting = self.add_plugin(frequency_weighting(width=1, input=self, bus=self, zero_zi=True))

    def process(self, block: np.ndarray):
        self.frequency_weighting.process(block)

    def get(self) -> np.ndarray:
        return self.block

    def create_plugin(self, ptype: type[TPlugin], **kwargs) -> TPlugin:
        plugin = ptype(id=f"{self.name}{next(self._counter)}", bus=self, **kwargs)
        return self.add_plugin(plugin)

    def add_plugin(self, plugin: TPlugin) -> TPlugin:
        if plugin.bus != self:
            raise Exception(f"Plugin {plugin.bus} does not belong to {self}")
        self.plugins.append(plugin)
        return plugin


    # Meters are handled by Plugins
    # def create_meter(self, plugin: PluginMeter, mtype: type[TMeter], **kwargs) -> TMeter:
    #     meter = plugin.create_meter(mtype, **kwargs)
    #     return self.add_meter(meter)
    #
    # def add_meter(self, meter: TMeter) -> TMeter:
    #     self.meters.append(meter)
    #     return meter



    def get_chain(self) -> list[ProcessingElement]:
        return [self]

    def to_str(self):
        return f"Bus(name={self.name})"

    def __str__(self):
        return self.to_str()
