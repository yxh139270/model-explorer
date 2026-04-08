from __future__ import annotations

from dataclasses import dataclass

from model_explorer.custom_dialect_tokenizer import split_top_level


@dataclass
class ParsedType:
  raw_text: str
  kind: str = ''
  element_type: str = ''
  shape: list[str] | None = None
  memory_space: str = ''


def parse_type(type_text: str) -> ParsedType:
  text = type_text.strip()
  if not text:
    return ParsedType(raw_text='')

  kind = text.split('<', 1)[0].strip() if '<' in text else text
  element_type = ''
  shape: list[str] | None = None
  if '<' in text and '>' in text:
    inner = text[text.index('<') + 1:text.rfind('>')]
    segments = split_top_level(inner, ',')
    if segments:
      shape_and_element = segments[0]
      parts = [part for part in shape_and_element.split('x') if part]
      if parts:
        element_type = parts[-1]
        if len(parts) > 1:
          shape = parts[:-1]

  memory_space = ''
  if '#dlgpu<' in text:
    memory_space = text[text.index('#dlgpu<'):]

  return ParsedType(
      raw_text=text,
      kind=kind,
      element_type=element_type,
      shape=shape,
      memory_space=memory_space,
  )
