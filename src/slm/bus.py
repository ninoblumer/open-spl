from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from slm.processing_element import ProcessingElement
from slm.frequency_weighting import PluginFrequencyWeighting, PluginZWeighting

if TYPE_CHECKING:
    from slm.plugin import Plugin, TPlugin



class Bus(ProcessingElement):
    name: str
    input_filter: Plugin | None
    frequency_weighting: PluginFrequencyWeighting
    plugins: list[Plugin]
    # meters: list[Meter] # meters are handled by plugins
    block: np.ndarray
    engine: "Engine"
    dt: float = property(lambda self: self.engine.dt)

    samplerate: int = property(lambda self: self.engine.samplerate)
    blocksize: int = property(lambda self: self.engine.blocksize)
    sensitivity: float = property(lambda self: self.engine.sensitivity)

    def __init__(self, engine: "Engine", name: str,
                 frequency_weighting: type[PluginFrequencyWeighting] | None = None,
                 input_filter: type[Plugin] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.name = name
        self.plugins = []
        self.block = np.zeros((1, self.blocksize))

        if frequency_weighting is None:
            frequency_weighting = PluginZWeighting

        # Optional signal-conditioning filter (e.g. a band-limiting analog input filter)
        # sits at the head of the bus, in front of the frequency weighting, so the
        # whole metering chain sees the band-limited signal. When absent the
        # weighting reads the raw bus block directly.
        if input_filter is not None:
            self.input_filter = self.add_plugin(
                input_filter(width=1, input=self, bus=self))
            weighting_input = self.input_filter
        else:
            self.input_filter = None
            weighting_input = self

        self.frequency_weighting = self.add_plugin(
            frequency_weighting(width=1, input=weighting_input, bus=self))

    def process(self, block: np.ndarray):
        # Drive the chain from its head: the input filter (if any) pushes its
        # output on to the frequency weighting via the subscriber mechanism.
        if self.input_filter is not None:
            self.input_filter.process(block)
        else:
            self.frequency_weighting.process(block)

    def get(self) -> np.ndarray:
        return self.block

    def add_plugin(self, plugin: TPlugin) -> TPlugin:
        if plugin.bus != self:
            raise Exception(f"Plugin {plugin.bus} does not belong to {self}")
        self.plugins.append(plugin)
        return plugin



    def get_chain(self) -> list[ProcessingElement]:
        return [self]

    def to_str(self):
        return f"Bus(name={self.name})"

    def __str__(self):
        return self.to_str()
