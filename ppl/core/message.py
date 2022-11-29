from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass
class Message:
    name: str
    value: Any
    fn: Any = None
    type: str = "rv"
    observed: bool = False
