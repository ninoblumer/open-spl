from plugin import Plugin, ID
import itertools

class Bus:
    name: str
    root: Plugin | None
    plugins: set[Plugin]

    def __init__(self, name: str, root_type: type[Plugin] | None = None):
        self.name = name
        self.plugins = set()
        self._counter = itertools.count(1)  # for numbering plugins with unique id

        if root_type:
            self.root = self.add_plugin(root_type, source=None)

    def add_plugin(self, ptype: type[Plugin], source) -> Plugin:
        plugin = ptype(id=f"{self.name}{next(self._counter)}", input=source)
        self.plugins.add(plugin)
        return plugin
