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


if __name__ == '__main__':
  unittest.main()
