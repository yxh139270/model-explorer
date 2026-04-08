from __future__ import annotations

from model_explorer.custom_dialect_tokenizer import split_top_level


RECOGNIZED_METADATA_KEYS = [
    'opName',
    'sym_name',
    'cluster',
    'ggType',
    'gtgType',
    'hashKey',
    'size',
    'memLoc',
    'dynInfo',
]


def _find_top_level_colon(text: str) -> int:
  stack: list[str] = []
  delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}
  for index, ch in enumerate(text):
    if ch in delimiters:
      stack.append(delimiters[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
    elif ch == ':' and not stack:
      return index
  return -1


def _normalize_value(value: str) -> str:
  value = value.strip()
  if value.startswith('"') and value.endswith('"'):
    return value[1:-1]

  colon = _find_top_level_colon(value)
  if colon != -1 and not value.startswith('{') and not value.startswith('#'):
    return value[:colon].strip()
  return value


def extract_operation_metadata(op_name: str, attributes_text: str) -> dict[str, str]:
  del op_name
  metadata: dict[str, str] = {}
  if not attributes_text:
    return metadata

  for entry in split_top_level(attributes_text, ','):
    if '=' not in entry:
      continue
    key, raw_value = entry.split('=', 1)
    key = key.strip()
    if key in RECOGNIZED_METADATA_KEYS:
      metadata[key] = _normalize_value(raw_value)
  return metadata
