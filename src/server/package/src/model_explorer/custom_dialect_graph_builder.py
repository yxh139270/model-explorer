from __future__ import annotations

import re

from model_explorer import graph_builder
from model_explorer.custom_dialect_ir import ParsedBlock, ParsedModule, ParsedOperation
from model_explorer.custom_dialect_tokenizer import split_top_level


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

  def walk_block(block: ParsedBlock, namespace: str):
    nonlocal node_counter
    for block_arg in block.arguments:
      arg_node_id = ensure_argument_node(
          block_arg.name, namespace, is_function_arg=False
      )
      producer_map[block_arg.name] = (arg_node_id, '0')

    for op in block.operations:
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

      for region_index, region in enumerate(op.regions):
        region_name = (
            op.attributes.get('sym_name')
            or op.attributes.get('opName')
            or f'sub_{region_index}'
        )
        child_namespace = f'{namespace}/{region_name}' if namespace else region_name
        for block_in_region in region.blocks:
          walk_block(block_in_region, child_namespace)

  for func in module.functions:
    namespace = func.name
    for arg in func.arguments:
      arg_node_id = ensure_argument_node(arg.name, namespace, is_function_arg=True)
      producer_map[arg.name] = (arg_node_id, '0')
    walk_block(func.body, namespace)

  return graph
