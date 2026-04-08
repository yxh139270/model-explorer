import unittest

from model_explorer.custom_dialect_graph_builder import build_graph
from model_explorer.custom_dialect_parser import parse_mlir_text


class CustomDialectGraphBuilderTest(unittest.TestCase):

  def test_build_graph_preserves_labels_and_namespaces(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0 = dlgpu.scale_shape_variants(%arg0) {opName = "ScaleShape0"} '
        ': memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, '
        '#dlgpu<memory_space cluster = 0>>\n'
        '  %1 = dlgpu.launch_gtg(%0) {sym_name = "sub_0_infer_shape"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):\n'
        '    %2 = dlgpu.infer_shape(%arg1) {opName = "FusedConv0"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '    dlgpu.return %2 : memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '  }\n'
        '}'
    )
    graph = build_graph(parse_mlir_text(text))
    labels = {node.label for node in graph.nodes}
    self.assertIn('ScaleShape0', labels)
    self.assertIn('FusedConv0', labels)
    infer_shape_node = [node for node in graph.nodes if node.label == 'FusedConv0'][0]
    self.assertEqual(infer_shape_node.namespace, 'main/sub_0_infer_shape')

  def test_build_graph_creates_input_edges(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0 = dlgpu.infer_shape(%arg0) {opName = "InputMove"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '}'
    )
    graph = build_graph(parse_mlir_text(text))
    target = [node for node in graph.nodes if node.label == 'InputMove'][0]
    self.assertEqual(target.incomingEdges[0].sourceNodeId, 'arg_%arg0')

  def test_region_op_attrs_are_attached_to_expandable_group(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0 = dlgpu.launch_gtg(%arg0) {sym_name = "sub_0_infer_shape", '
        'opName = "LaunchInfer"} : memref<1xf32, #dlgpu<memory_space cluster '
        '= 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):\n'
        '    %1 = dlgpu.infer_shape(%arg1) {opName = "FusedConv0"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '    dlgpu.return %1 : memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '  }\n'
        '  %2 = dlgpu.infer_shape(%0) {opName = "AfterLaunch"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '}'
    )
    graph = build_graph(parse_mlir_text(text))

    # The expandable namespace should carry launch op attrs directly.
    self.assertIsNotNone(graph.groupNodeAttributes)
    group_attrs = graph.groupNodeAttributes['main/sub_0_infer_shape']
    self.assertEqual(group_attrs['op_type'], 'dlgpu.launch_gtg')
    self.assertEqual(group_attrs['sym_name'], 'sub_0_infer_shape')
    self.assertEqual(group_attrs['opName'], 'LaunchInfer')

    # No duplicate launch node with the same label as the expandable group.
    self.assertFalse(any(node.label == 'sub_0_infer_shape' for node in graph.nodes))

    # Downstream op still connects through launch result.
    after_launch = [node for node in graph.nodes if node.label == 'AfterLaunch'][0]
    self.assertTrue(after_launch.incomingEdges)

  def test_hide_return_nodes_and_map_region_result_to_return_input_producer(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0:2 = dlgpu.scale_shape_variants(%arg0) {opName = "ScaleShape0"} '
        ': memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>, '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>>\n'
        '  %1 = dlgpu.launch_gtg(%0#0) {sym_name = "gtg_0"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):\n'
        '    %10 = dlgpu.infer_shape(%arg1) {opName = "Inner0"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '    dlgpu.return %10 : memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '  }\n'
        '  %2 = dlgpu.launch_gtg(%0#1) {sym_name = "gtg_1"} : '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg2: memref<2xf32, #dlgpu<memory_space cluster = 0>>):\n'
        '    %11 = dlgpu.infer_shape(%arg2) {opName = "Inner1"} : '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>>\n'
        '    dlgpu.return %11 : memref<2xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '  }\n'
        '  %3 = dlgpu.select_variant_result(%1, %2) {opName = "SelectV"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>, '
        'memref<2xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>>\n'
        '}'
    )
    graph = build_graph(parse_mlir_text(text))

    self.assertFalse(any(node.label == 'dlgpu.return' for node in graph.nodes))

    by_id = {node.id: node for node in graph.nodes}
    select_v = [node for node in graph.nodes if node.label == 'SelectV'][0]
    source_labels = [by_id[edge.sourceNodeId].label for edge in select_v.incomingEdges]
    self.assertEqual(source_labels, ['Inner0', 'Inner1'])


if __name__ == '__main__':
  unittest.main()
