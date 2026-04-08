from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedValue:
  name: str
  type_text: str = ''


@dataclass
class ParsedBlock:
  arguments: list[ParsedValue] = field(default_factory=list)
  operations: list['ParsedOperation'] = field(default_factory=list)


@dataclass
class ParsedRegion:
  blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass
class ParsedOperation:
  name: str
  result_names: list[str] = field(default_factory=list)
  result_count: int = 0
  attributes_text: str = ''
  type_text: str = ''
  operand_text: str = ''
  regions: list[ParsedRegion] = field(default_factory=list)
  attributes: dict[str, str] = field(default_factory=dict)
  result_types: list[Any] = field(default_factory=list)


@dataclass
class ParsedFunction:
  name: str
  arguments: list[ParsedValue] = field(default_factory=list)
  result_types: list[str] = field(default_factory=list)
  body: ParsedBlock = field(default_factory=ParsedBlock)


@dataclass
class ParsedModule:
  functions: list[ParsedFunction] = field(default_factory=list)
