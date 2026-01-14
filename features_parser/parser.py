from dataclasses import dataclass
from typing import Any, List, Optional
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


class BaseHandler:
    def __init__(self, param_name):
        self.param_name = param_name
        self.base_regex = re.compile(
            rf"BlockStat: POC (\d+) @\(\s*(\d+),\s*(\d+)\) \[\s*(\d+)x\s*(\d+)\] {self.param_name}=(.+)"
        )

    def match_and_create(self, line: str) -> Optional[BlockStatToken]:
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
            return (float(nums[0]), float(nums[1]))
        return (0.0, 0.0)


class VTMParser:
    def __init__(self):
        self.handlers = [
            ScalarHandler("QP"),
            ScalarHandler("PredMode"),
            ScalarHandler("Depth"),
            VectorHandler("MVL0"),
            VectorHandler("MVL1"),
        ]
        self.tokens: List[BlockStatToken] = []

    def parse(self, line_iterator):
        for line in line_iterator:
            if not line.startswith("BlockStat:"):
                continue
            for handler in self.handlers:
                token = handler.match_and_create(line)
                if token:
                    self.tokens.append(token)
                    break
        return self.tokens

    def parse_file(self, file_path: str):
        with open(file_path, "r") as f:
            return self.parse(f)
