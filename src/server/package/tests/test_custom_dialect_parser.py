import unittest

from model_explorer.custom_dialect_parser import parse_mlir_text


class CustomDialectParserTest(unittest.TestCase):

  def test_parse_func_signature_with_custom_memref(self):
    text = (
        'func.func @main(%arg0: memref<1x3xf32, #dlgpu<memory_space cluster = '
        '0>>) -> memref<1x3xf32, #dlgpu<memory_space cluster = 1>> {\n'
        '  dlgpu.return %arg0 : memref<1x3xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '}'
    )
    module = parse_mlir_text(text)
    self.assertEqual(len(module.functions), 1)
    func = module.functions[0]
    self.assertEqual(func.name, 'main')
    self.assertEqual(func.arguments[0].name, '%arg0')
    self.assertIn('#dlgpu<memory_space', func.arguments[0].type_text)

  def test_parse_block_argument_and_multi_result_op(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0:2 = dlgpu.scale_shape_variants(%arg0) {opName = "ScaleShape0"} '
        ': memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, '
        '#dlgpu<memory_space cluster = 0>>, memref<2xf32, '
        '#dlgpu<memory_space cluster = 0>>\n'
        '  %1 = dlgpu.launch_gtg(%0#0) {sym_name = "sub_0_infer_shape"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):\n'
        '    dlgpu.return %arg1 : memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>\n'
        '  }\n'
        '}'
    )
    module = parse_mlir_text(text)
    op = module.functions[0].body.operations[0]
    self.assertEqual(op.result_count, 2)
    nested = module.functions[0].body.operations[1].regions[0].blocks[0]
    self.assertEqual(nested.arguments[0].name, '%arg1')


if __name__ == '__main__':
  unittest.main()
