from __future__ import annotations
from abc import ABC
import numpy as np


from slm.plugin import Plugin

from slm.meter import Meter, TMeter


REFERENCE_PRESSURE = 20e-6
class PluginMeter(Plugin, ABC):
    meters: dict[str, Meter]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meters = {}

    def add_meter(self, mtype:type[TMeter], **kwargs) -> TMeter:
        meter = mtype(parent=self, **kwargs)
        self.meters[meter.name] = meter
        return meter

    def process(self, block: np.ndarray):
        super().process(block)
        self.process_meters()

    def process_meters(self):
        for meter in self.meters.values():
            meter.process(self.output)

    def read_lin(self, name: str):
        return self.meters[name].read()

    def read_db(self, name):
        return 10*np.log10(self.read_lin(name)/(REFERENCE_PRESSURE*self.sensitivity)**2)
