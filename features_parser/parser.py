from dataclasses import dataclass
from typing import Any, DefaultDict, List, NamedTuple, Optional
import re


@dataclass
class BlockStatToken:
    poc: int
    x: int
    y: int
    w: int
    h: int
    param: str
    value: Any


class MotionVector(NamedTuple):
    x: float = 0.0
    y: float = 0.0


VTM_DECODER_BLOCK_REGEX = (
    r"BlockStat: POC (\d+) @\(\s*(\d+),\s*(\d+)\) \[\s*(\d+)x\s*(\d+)\] {param}=(.+)"
)


class BaseHandler:
    def __init__(self, param_name):
        self.param_name = param_name
        formatted_regex = VTM_DECODER_BLOCK_REGEX.format(param=self.param_name)
        self.base_regex = re.compile(formatted_regex)

    def parse(self, line: str) -> Optional[BlockStatToken]:
        match = self.base_regex.search(line)
        if match:
            poc, x, y, w, h, raw_val = match.groups()
            clean_val = self.process_value(raw_val.strip())
            return BlockStatToken(
                int(poc), int(x), int(y), int(w), int(h), self.param_name, clean_val
            )
        return None

    def process_value(self, raw_val: str):
        raise NotImplementedError


class ScalarHandler(BaseHandler):
    """Handles scalar values like QP"""

    def process_value(self, raw_val: str):
        try:
            return float(raw_val)
        except ValueError:
            return 0.0


class VectorHandler(BaseHandler):
    """Handles vector like values '{x, y}' converting it to tuple (float, float)"""

    def process_value(self, raw_val: str):
        nums = re.findall(r"-?\d+", raw_val)
        if len(nums) >= 2:
            return MotionVector(float(nums[0]), float(nums[1]))
        return MotionVector()


class VTMParser:
    def __init__(self):
        self.handlers: List[BaseHandler] = [
            ScalarHandler("QP"),
            ScalarHandler("PredMode"),
            ScalarHandler("Depth"),
            VectorHandler("MVL0"),
            VectorHandler("MVL1"),
        ]
        self.tokens: List[BlockStatToken] = []

    def group_on_poc(self):
        if not self.tokens:
            return {}

        self.tokens.sort(key=lambda t: (t.poc, t.y, t.x))
        grouped = DefaultDict(list)
        for token in self.tokens:
            grouped[token.poc].append(token)

        return dict(grouped)

    def parse(self, line_iterator):
        for line in line_iterator:
            if not line.startswith("BlockStat:"):
                continue
            for handler in self.handlers:
                token = handler.parse(line)
                if token:
                    self.tokens.append(token)
                    break
        return self.tokens

    def parse_file(self, file_path: str):
        with open(file_path, "r") as f:
            self.parse(f)
        return self.group_on_poc()
