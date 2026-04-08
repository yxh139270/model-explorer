from __future__ import annotations

import re

from model_explorer.custom_dialect_ir import (
    ParsedBlock,
    ParsedFunction,
    ParsedModule,
    ParsedOperation,
    ParsedRegion,
    ParsedValue,
)
from model_explorer.custom_dialect_tokenizer import find_matching_brace, split_top_level


def _find_matching_paren(text: str, start_index: int) -> int:
  depth = 0
  for index in range(start_index, len(text)):
    ch = text[index]
    if ch == '(':
      depth += 1
    elif ch == ')':
      depth -= 1
      if depth == 0:
        return index
  raise ValueError(f'unmatched parenthesis starting at {start_index}')


def _find_top_level_token(text: str, token: str) -> int:
  stack: list[str] = []
  delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}
  index = 0
  while index < len(text):
    ch = text[index]
    if not stack and text.startswith(token, index):
      return index
    if ch in delimiters:
      stack.append(delimiters[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
    index += 1
  return -1


def _split_top_level_lines(text: str) -> list[str]:
  ops: list[str] = []
  start = 0
  stack: list[str] = []
  delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}
  for index, ch in enumerate(text):
    if ch in delimiters:
      stack.append(delimiters[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
    elif ch == '\n' and not stack:
      op_text = text[start:index].strip()
      if op_text:
        ops.append(op_text)
      start = index + 1
  tail = text[start:].strip()
  if tail:
    ops.append(tail)
  return ops


def _parse_value(value_text: str) -> ParsedValue:
  if ':' not in value_text:
    return ParsedValue(name=value_text.strip(), type_text='')
  name, type_text = value_text.split(':', 1)
  return ParsedValue(name=name.strip(), type_text=type_text.strip())


def _parse_arguments(args_text: str) -> list[ParsedValue]:
  arguments: list[ParsedValue] = []
  for arg in split_top_level(args_text.strip(), ','):
    if arg:
      arguments.append(_parse_value(arg))
  return arguments


def _parse_result_types(result_text: str) -> list[str]:
  result_text = result_text.strip()
  if not result_text:
    return []
  if result_text.startswith('(') and result_text.endswith(')'):
    inner = result_text[1:-1].strip()
    return split_top_level(inner, ',') if inner else []
  return [result_text]


def _parse_block_arguments(block_header: str) -> list[ParsedValue]:
  lparen = block_header.find('(')
  if lparen == -1:
    return []
  rparen = _find_matching_paren(block_header, lparen)
  return _parse_arguments(block_header[lparen + 1:rparen])


def _parse_operation(statement: str) -> ParsedOperation:
  line = statement.strip()
  result_names: list[str] = []
  result_count = 0
  if '=' in line:
    lhs, rhs = line.split('=', 1)
    lhs = lhs.strip()
    if lhs.startswith('%'):
      if ':' in lhs:
        base_name, count_text = lhs.rsplit(':', 1)
        if count_text.isdigit():
          result_count = int(count_text)
          result_names = [f'{base_name}#{i}' for i in range(result_count)]
        else:
          result_names = [lhs]
          result_count = 1
      else:
        result_names = [lhs]
        result_count = 1
      line = rhs.strip()

  paren_index = line.find('(')
  if paren_index == -1:
    op_name = line.split(' ', 1)[0].strip()
    return ParsedOperation(
        name=op_name,
        result_names=result_names,
        result_count=result_count,
    )

  op_name = line[:paren_index].strip()
  operand_end = _find_matching_paren(line, paren_index)
  operand_text = line[paren_index + 1:operand_end].strip()
  remainder = line[operand_end + 1:].strip()

  attributes_text = ''
  if remainder.startswith('{'):
    attr_end = find_matching_brace(remainder, 0)
    attributes_text = remainder[1:attr_end].strip()
    remainder = remainder[attr_end + 1:].strip()

  region_start = _find_top_level_token(remainder, '{')
  region_text = ''
  if region_start != -1:
    region_text = remainder[region_start:].strip()
    type_text = remainder[:region_start].strip()
  else:
    type_text = remainder
  if type_text.startswith(':'):
    type_text = type_text[1:].strip()

  regions: list[ParsedRegion] = []
  if region_text:
    region = _parse_region(region_text)
    regions.append(region)

  return ParsedOperation(
      name=op_name,
      result_names=result_names,
      result_count=result_count,
      attributes_text=attributes_text,
      type_text=type_text,
      operand_text=operand_text,
      regions=regions,
  )


def _parse_region(region_text: str) -> ParsedRegion:
  region_text = region_text.strip()
  if not region_text.startswith('{'):
    return ParsedRegion()
  end = find_matching_brace(region_text, 0)
  region_body = region_text[1:end].strip()

  block_matches = list(re.finditer(r'(?m)^\s*(\^bb[^\n]*:)', region_body))
  if not block_matches:
    return ParsedRegion(blocks=[_parse_block(region_body)])

  blocks: list[ParsedBlock] = []
  for index, match in enumerate(block_matches):
    block_header = match.group(1)
    start = match.end()
    end = block_matches[index + 1].start() if index + 1 < len(block_matches) else len(region_body)
    block_body = region_body[start:end]
    blocks.append(
        ParsedBlock(
            arguments=_parse_block_arguments(block_header),
            operations=_parse_block(block_body).operations,
        )
    )
  return ParsedRegion(blocks=blocks)


def _parse_block(body_text: str) -> ParsedBlock:
  operations: list[ParsedOperation] = []
  for statement in _split_top_level_lines(body_text):
    stripped = statement.strip()
    if not stripped or stripped.startswith('^bb'):
      continue
    operations.append(_parse_operation(stripped))
  return ParsedBlock(operations=operations)


def _parse_function(text: str, start_index: int) -> tuple[ParsedFunction, int]:
  name_match = re.match(r'func\.func\s+@([A-Za-z0-9_.$-]+)\s*\(', text[start_index:])
  if not name_match:
    raise ValueError('invalid function header')
  name = name_match.group(1)
  args_start = start_index + name_match.end() - 1
  args_end = _find_matching_paren(text, args_start)
  args_text = text[args_start + 1:args_end]
  arguments = _parse_arguments(args_text)

  body_start = _find_top_level_token(text[args_end + 1:], '{')
  if body_start == -1:
    raise ValueError('function body not found')
  body_start = args_end + 1 + body_start
  body_end = find_matching_brace(text, body_start)

  signature_tail = text[args_end + 1:body_start].strip()
  arrow_index = _find_top_level_token(signature_tail, '->')
  if arrow_index != -1:
    result_text = signature_tail[arrow_index + 2:].strip()
  else:
    result_text = ''

  body_text = text[body_start + 1:body_end]
  func = ParsedFunction(
      name=name,
      arguments=arguments,
      result_types=_parse_result_types(result_text),
      body=_parse_block(body_text),
  )
  return func, body_end + 1


def parse_mlir_text(text: str) -> ParsedModule:
  module = ParsedModule()
  index = 0
  while True:
    func_start = text.find('func.func', index)
    if func_start == -1:
      break
    func, index = _parse_function(text, func_start)
    module.functions.append(func)
  return module
