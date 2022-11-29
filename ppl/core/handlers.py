from copy import copy
from .message import Message
from typing import Dict, Any, Union, Callable

class Handler:
    def __init__(self, fn: Union["AbstractModel", "Handler"]):
        self.fn = fn

    def stack(self):
        return self.fn.stack()

    def _push(self):
        self.stack().append(self)

    def _pop(self):
        assert self.stack().pop() is self

    def run(self, **kwargs):
        self._push()
        result = self.fn.run(**kwargs)
        self._pop()
        return result

    def process(self, msg: Message):
        pass

    def postprocess(self, msg: Message):
        pass

class trace(Handler):
    def _push(self):
        super()._push()
        self.result = dict()

    def postprocess(self, msg: Message):
        assert (
            msg.type != "rv" or msg.name not in self.result
        ), "sample sites must have unique names"
        self.result[msg.name] = copy(msg)

    def get(self, **kwargs):
        self.run(**kwargs)
        return self.result

class condition(Handler):
    def __init__(self, fn, substate: Dict[str, Any]):
        self.substate = substate
        super().__init__(fn)

    def process(self, msg: Message):
        if msg.name in self.substate:
            msg.value = self.substate[msg.name]
