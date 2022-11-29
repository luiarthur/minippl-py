from copy import copy
from .message import Message
from .handlers import Handler
import numpy as np
from ppl.distributions import Distribution
from typing import List

class AbstractModel:
    def __init__(self):
        self._stack: List[Handler] = []

    def model(self):
        pass

    def run(self, **kwargs):
        return self.model(**kwargs)

    def stack(self):
        return self._stack

    def apply_stack(self, msg: Message):
        for handler in reversed(self._stack):
            handler.process(msg)

        if msg.value is None:
            msg.value = msg.fn.run()

        for handler in self._stack:
            handler.postprocess(msg)

        return msg

    def rv(self, name: str, dist: Distribution, **kwargs):
        obs = kwargs.pop("obs", None)

        if not self._stack:
            return dist.sample()

        observed = not obs is None
        msg = Message(
            name = name,
            fn = dist,
            value = obs,
            observed = observed
        )

        msg = self.apply_stack(msg)
        return msg.value
