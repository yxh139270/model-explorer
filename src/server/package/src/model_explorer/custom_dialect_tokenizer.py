from __future__ import annotations


DELIMITER_PAIRS = {
    '(': ')',
    '{': '}',
    '[': ']',
    '<': '>',
}


def _is_closing_delimiter(text: str, index: int, ch: str) -> bool:
  # Ignore `->` arrows so `>` does not incorrectly close `<...>` scopes.
  if ch == '>' and index > 0 and text[index - 1] == '-':
    return False
  return True


def split_top_level(text: str, delimiter: str) -> list[str]:
  parts: list[str] = []
  start = 0
  stack: list[str] = []
  for index, ch in enumerate(text):
    if ch in DELIMITER_PAIRS:
      stack.append(DELIMITER_PAIRS[ch])
    elif stack and ch == stack[-1] and _is_closing_delimiter(text, index, ch):
      stack.pop()
    elif ch == delimiter and not stack:
      part = text[start:index].strip()
      if part:
        parts.append(part)
      start = index + 1
  tail = text[start:].strip()
  if tail:
    parts.append(tail)
  return parts


def find_matching_brace(text: str, start_index: int) -> int:
  stack: list[str] = []
  for index in range(start_index, len(text)):
    ch = text[index]
    if ch in DELIMITER_PAIRS:
      stack.append(DELIMITER_PAIRS[ch])
    elif stack and ch == stack[-1] and _is_closing_delimiter(text, index, ch):
      stack.pop()
      if not stack:
        return index
  raise ValueError(f'unmatched delimiter starting at {start_index}')
