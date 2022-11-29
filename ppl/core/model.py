from copy import copy
from .message import Message
from .handlers import Handler, trace, condition
import numpy as np
from ppl.distributions import Distribution
from typing import List, Dict, Any

class AbstractModel:
    def __init__(self):
        self._stack: List[Handler] = []

    def model(self):
        pass

    def run(self, **kwargs):
        return self.model(**kwargs)

    def stack(self):
        # This is needed to avoid a handler pushing itself onto a stack, with a
        # stack field, causing recursive pushing issues. 
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

        msg = Message(
            name = name,
            fn = dist,
            value = obs,
            observed = (not obs is None)
        )

        msg = self.apply_stack(msg)
        return msg.value

    def logpdf(self, state: Dict[str, Any], **kwargs):
        t = trace(condition(self, state)).get(**kwargs)

        lp = 0
        for param in t.values():
            if param.type == "rv":
                lp += np.sum(param.fn.logpdf(param.value))

                # Don't do the remaining computation if the log joint density
                # is -inf.
                if np.isneginf(lp):
                    return np.NINF

        return lp
