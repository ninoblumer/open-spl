from __future__ import annotations
from abc import ABC, abstractmethod

from engine import ExecutionContext, ExecutionError, RequestRejected

ID = str

def find_plugin(function) -> type[Plugin]:
    raise RequestRejected(
        f"Function {function} is not implemented."
    )


class Plugin(ABC):
    id: ID
    function: str = ""
    input: Plugin
    outputs: list[Plugin]

    def __init__(self, id: str, input: Plugin | None, outputs: list[Plugin] | None=None):
        self.id = id
        self.input = input
        if outputs:
            self.outputs = outputs.copy()
        else:
            self.outputs = list()

    def process(self, ctx: ExecutionContext):
        data = ctx.cache.get(self.input.id) if self.input else ctx.block
        result = self.function(data)

        if ctx.cache.get(self.id) is not None:
            raise ExecutionError
        ctx.cache[self.id] = result

    @abstractmethod
    def function(self):
        ...