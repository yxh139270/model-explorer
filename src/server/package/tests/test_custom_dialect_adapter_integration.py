import unittest

from model_explorer.custom_dialect_adapter import CustomDialectAdapter
from model_explorer.extension_manager import ExtensionManager


TARGET_PATH = '/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir'


class CustomDialectAdapterIntegrationTest(unittest.TestCase):

  def test_adapter_detects_custom_dialect_mlir(self):
    adapter = CustomDialectAdapter()
    self.assertTrue(adapter.can_handle_file(TARGET_PATH))

  def test_extension_manager_registers_custom_dialect_adapter(self):
    manager = ExtensionManager()
    manager.load_extensions()
    extension_ids = [ext.metadata.id for ext in manager.extensions]
    self.assertIn('custom_dialect_mlir', extension_ids)

  def test_target_dynamic_graph_builds_non_empty_graph(self):
    adapter = CustomDialectAdapter()
    graphs = adapter.convert(TARGET_PATH, settings={})
    graph = graphs['graphs'][0]
    self.assertGreater(len(graph.nodes), 50)
    labels = {node.label for node in graph.nodes}
    self.assertIn('ScaleShape0', labels)
    self.assertIn('FusedConv0', labels)
    self.assertTrue(any(node.namespace.startswith('main/sub_0') for node in graph.nodes))
    self.assertTrue(
        any(any(attr.key == 'memory_space' for attr in node.attrs) for node in graph.nodes)
    )

  def test_custom_adapter_handles_builtin_mlir_failure_case(self):
    adapter = CustomDialectAdapter()
    result = adapter.convert(TARGET_PATH, settings={})
    self.assertIn('graphs', result)
    self.assertEqual(len(result['graphs']), 1)


if __name__ == '__main__':
  unittest.main()
