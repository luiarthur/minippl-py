from copy import copy
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np
from .distributions import Distribution

# Global effect handler stack. Multiple models can be fit in parallel (e.g., via `multiprocessing`) because `_stack` will be copied onto each processor.
_stack = []

@dataclass
class Message:
    name: str
    value: Any
    dist: Distribution
    observed: bool = False

class Handler:
    def __init__(self, fn):
        self.fn = fn

    def _push(self):
        _stack.append(self)

    def _pop(self):
        assert _stack.pop() is self

    def __call__(self, **kwargs):
        self._push()
        result = self.fn(**kwargs)
        self._pop()
        return result

    def process(self, msg: Message):
        pass
        # See apply_stack. This is invoked before the default effect.

    def postprocess(self, msg: Message):
        # See apply_stack. This is invoked after the default effect.
        pass

class trace(Handler):
    def __init__(self, fn):
        self.result = {}
        super().__init__(fn)

    def postprocess(self, msg: Message):
        assert (msg.name not in self.result), "Sample sites must have unique names!"
        self.result[msg.name] = copy(msg)

    def get(self, **kwargs):
        self(**kwargs)
        return self.result

class condition(Handler):
    def __init__(self, fn, substate: Dict[str, Any]):
        self.substate = substate
        super().__init__(fn)

    def process(self, msg: Message):
        if msg.name in self.substate:
            msg.value = self.substate[msg.name]

def apply_stack(msg: Message) -> Message:
    # Handlers most recently added on the handler stack are first applied.
    for handler in reversed(_stack):
        handler.process(msg)

    # Default value assignment effect: if value is unassigned, sample from the provided distribution.
    if msg.value is None:
        msg.value = msg.dist.sample()

    # At the end, do post-processing. This is needed because trace should be
    # applied only at the end. We expect only trace to be invoked here. If we
    # didn't have postprocess, then the recorded msg.value may be None.
    for handler in _stack:
        handler.postprocess(msg)

    return msg

def sample(name: str, dist: Distribution, obs=None):
    if not _stack:
        return dist.sample()
    else:
        msg = Message(
            name = name,
            dist = dist,
            value = obs,
            observed = (not obs is None)
        )
        return apply_stack(msg).value

def logpdf(model, state: Dict[str, Any], **kwargs) -> float:
    t = trace(condition(model, state)).get(**kwargs)
    lp = 0.0
    for param in t.values():
        lp += np.sum(param.dist.logpdf(param.value))
        # Skip remaining computation if the log joint density is -inf.
        if np.isneginf(lp):
            return np.NINF
    return lp
