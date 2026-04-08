from __future__ import annotations

import re

from model_explorer import graph_builder
from model_explorer.custom_dialect_ir import (
    ParsedBlock,
    ParsedModule,
    ParsedOperation,
    ParsedRegion,
)
from model_explorer.custom_dialect_tokenizer import split_top_level
from model_explorer.custom_dialect_types import ParsedType, parse_type


def _sanitize_id(text: str) -> str:
  return re.sub(r'[^a-zA-Z0-9_]+', '_', text).strip('_')


def _extract_operands(operand_text: str) -> list[str]:
  if not operand_text:
    return []
  operands: list[str] = []
  for operand in split_top_level(operand_text, ','):
    candidate = operand.strip()
    if candidate.startswith('%'):
      operands.append(candidate)
  return operands


def _label_for_operation(op: ParsedOperation) -> str:
  if op.attributes.get('opName'):
    return op.attributes['opName']
  if op.attributes.get('sym_name'):
    return op.attributes['sym_name']
  return op.name


def _attrs_for_operation(op: ParsedOperation) -> list[graph_builder.KeyValue]:
  attrs = [
      graph_builder.KeyValue(key='op_type', value=op.name),
      graph_builder.KeyValue(key='raw_attributes', value=op.attributes_text),
      graph_builder.KeyValue(
          key='result_types',
          value=', '.join(result.raw_text for result in op.result_types),
      ),
  ]

  memory_spaces = [result.memory_space for result in op.result_types if result.memory_space]
  if memory_spaces:
    attrs.append(
        graph_builder.KeyValue(
            key='memory_space',
            value=' | '.join(sorted(set(memory_spaces))),
        )
    )

  for key, value in op.attributes.items():
    attrs.append(graph_builder.KeyValue(key=key, value=value))
  return attrs


def _group_attrs_for_operation(op: ParsedOperation) -> dict[str, str]:
  input_types = _input_types_for_operation(op)
  attrs: dict[str, str] = {
      'op_type': op.name,
      'raw_attributes': op.attributes_text,
      'result_types': ', '.join(result.raw_text for result in op.result_types),
      'input_count': str(len(input_types)),
      'output_count': str(len(op.result_types)),
  }

  for index, input_type in enumerate(input_types):
    attrs[f'input_{index}_type'] = input_type.raw_text
    if input_type.shape:
      attrs[f'input_{index}_shape'] = 'x'.join(input_type.shape)

  for index, output_type in enumerate(op.result_types):
    attrs[f'output_{index}_type'] = output_type.raw_text
    if output_type.shape:
      attrs[f'output_{index}_shape'] = 'x'.join(output_type.shape)

  memory_spaces = [result.memory_space for result in op.result_types if result.memory_space]
  if memory_spaces:
    attrs['memory_space'] = ' | '.join(sorted(set(memory_spaces)))
  attrs.update(op.attributes)
  return attrs


def _find_top_level_arrow(text: str) -> int:
  stack: list[str] = []
  delimiters = {'(': ')', '[': ']', '{': '}', '<': '>'}
  index = 0
  while index < len(text):
    ch = text[index]
    if ch in delimiters:
      stack.append(delimiters[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
    elif not stack and text.startswith('->', index):
      return index
    index += 1
  return -1


def _input_types_for_operation(op: ParsedOperation) -> list[ParsedType]:
  type_text = op.type_text.strip()
  if not type_text:
    return []

  arrow_index = _find_top_level_arrow(type_text)
  input_text = type_text[:arrow_index].strip() if arrow_index != -1 else type_text
  if input_text.startswith('(') and input_text.endswith(')'):
    input_text = input_text[1:-1].strip()

  if not input_text:
    return []

  return [
      parse_type(candidate)
      for candidate in split_top_level(input_text, ',')
      if candidate.strip()
  ]


def _outputs_metadata_for_operation(
    op: ParsedOperation,
) -> list[graph_builder.MetadataItem]:
  outputs: list[graph_builder.MetadataItem] = []
  for index, result in enumerate(op.result_types):
    attrs = [graph_builder.KeyValue(key='type', value=result.raw_text)]
    if result.shape:
      attrs.append(graph_builder.KeyValue(key='shape', value='x'.join(result.shape)))
    outputs.append(graph_builder.MetadataItem(id=str(index), attrs=attrs))
  return outputs


def _is_return_operation(op_name: str) -> bool:
  return op_name == 'return' or op_name.endswith('.return')


def _extract_return_operands(op: ParsedOperation) -> list[str]:
  operands = _extract_operands(op.operand_text)
  if operands:
    return operands

  # `return` is often written as `dlgpu.return %x : type` (without parens),
  # where parser stores `%x : type` in `type_text`.
  if ':' in op.type_text:
    head = op.type_text.split(':', 1)[0].strip()
  else:
    head = op.type_text.strip()
  if not head:
    return []
  return [item for item in _extract_operands(head) if item.startswith('%')]


def _find_region_return_operands(region: ParsedRegion) -> list[str]:
  for block in region.blocks:
    for candidate in reversed(block.operations):
      if _is_return_operation(candidate.name):
        return _extract_return_operands(candidate)
  return []


def build_graph(module: ParsedModule) -> graph_builder.Graph:
  graph = graph_builder.Graph(id='custom_dialect_graph')
  producer_map: dict[str, tuple[str, str]] = {}
  argument_nodes: dict[str, str] = {}
  node_counter = 0

  def ensure_argument_node(value_name: str, namespace: str, is_function_arg: bool) -> str:
    key = value_name if is_function_arg else f'{value_name}@{namespace}'
    if key in argument_nodes:
      return argument_nodes[key]

    if is_function_arg:
      node_id = f'arg_{value_name}'
    else:
      suffix = _sanitize_id(namespace) or 'root'
      node_id = f'arg_{value_name}_{suffix}'

    graph.nodes.append(
      graph_builder.GraphNode(
          id=node_id,
          label=value_name,
          namespace=namespace,
          attrs=[graph_builder.KeyValue(key='op_type', value='argument')],
      )
    )
    argument_nodes[key] = node_id
    return node_id

  def add_edges(op_node: graph_builder.GraphNode, op: ParsedOperation, namespace: str):
    for input_index, operand in enumerate(_extract_operands(op.operand_text)):
      if operand in producer_map:
        source_id, source_output_id = producer_map[operand]
      else:
        source_id = ensure_argument_node(operand, namespace, is_function_arg=False)
        source_output_id = '0'
      op_node.incomingEdges.append(
          graph_builder.IncomingEdge(
              sourceNodeId=source_id,
              sourceNodeOutputId=source_output_id,
              targetNodeInputId=str(input_index),
          )
      )

  def resolve_operand_source(
      operand: str, namespace: str
  ) -> tuple[str, str]:
    if operand in producer_map:
      return producer_map[operand]
    source_id = ensure_argument_node(operand, namespace, is_function_arg=False)
    return source_id, '0'

  def walk_block(
      block: ParsedBlock,
      namespace: str,
      block_arg_bindings: dict[str, tuple[str, str]] | None = None,
  ):
    nonlocal node_counter
    for block_arg in block.arguments:
      if block_arg_bindings and block_arg.name in block_arg_bindings:
        producer_map[block_arg.name] = block_arg_bindings[block_arg.name]
      else:
        arg_node_id = ensure_argument_node(
            block_arg.name, namespace, is_function_arg=False
        )
        producer_map[block_arg.name] = (arg_node_id, '0')

    for op in block.operations:
      if _is_return_operation(op.name):
        continue

      if op.regions:
        region_name = (
            op.attributes.get('sym_name')
            or op.attributes.get('opName')
            or f'sub_{node_counter}'
        )
        child_namespace = f'{namespace}/{region_name}' if namespace else region_name

        if graph.groupNodeAttributes is None:
          graph.groupNodeAttributes = {}
        graph.groupNodeAttributes[child_namespace] = _group_attrs_for_operation(op)

        operand_sources = [
            resolve_operand_source(operand, namespace)
            for operand in _extract_operands(op.operand_text)
        ]

        for region in op.regions:
          for block_in_region in region.blocks:
            block_bindings: dict[str, tuple[str, str]] = {}
            for index, block_arg in enumerate(block_in_region.arguments):
              if index < len(operand_sources):
                block_bindings[block_arg.name] = operand_sources[index]
            walk_block(block_in_region, child_namespace, block_bindings)

        return_operands: list[str] = []
        for region in op.regions:
          return_operands = _find_region_return_operands(region)
          if return_operands:
            break

        for output_index, result_name in enumerate(op.result_names):
          if output_index < len(return_operands):
            producer_map[result_name] = resolve_operand_source(
                return_operands[output_index], child_namespace
            )
          elif output_index < len(operand_sources):
            producer_map[result_name] = operand_sources[output_index]
        continue

      node_id = f'op_{node_counter}_{_sanitize_id(op.name or "op")}'
      node_counter += 1

      op_node = graph_builder.GraphNode(
          id=node_id,
          label=_label_for_operation(op),
          namespace=namespace,
          attrs=_attrs_for_operation(op),
          outputsMetadata=_outputs_metadata_for_operation(op),
      )
      add_edges(op_node, op, namespace)
      graph.nodes.append(op_node)

      for output_index, result_name in enumerate(op.result_names):
        producer_map[result_name] = (node_id, str(output_index))

  for func in module.functions:
    namespace = func.name
    for arg in func.arguments:
      arg_node_id = ensure_argument_node(arg.name, namespace, is_function_arg=True)
      producer_map[arg.name] = (arg_node_id, '0')
    walk_block(func.body, namespace)

  return graph
