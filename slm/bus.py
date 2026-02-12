from slm.plugins.base import Plugin
from slm.engine import Engine

import itertools


class Bus:
    name: str
    root: Plugin | None
    plugins: set[Plugin]

    samplerate: int = property(lambda self: self.engine.samplerate)
    blocksize: int = property(lambda self: self.engine.blocksize)
    sensitivity: float = property(lambda self: self.engine.sensitivity)

    def __init__(self, engine: Engine, name: str, root_type: type[Plugin] | None = None):
        self.engine = engine
        self.name = name
        self.plugins = set()
        self._counter = itertools.count(1)  # for numbering plugins with unique id

        if root_type:
            self.root = self.add_plugin(root_type, source=None)

    def add_plugin(self, ptype: type[Plugin], source) -> Plugin:
        plugin = ptype(id=f"{self.name}{next(self._counter)}", input=source)
        self.plugins.add(plugin)
        return plugin
