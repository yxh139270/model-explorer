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

  def test_extract_dlgpu_attributes(self):
    text = (
        'func.func @main(%arg0: memref<1x3x1920x1920xf32, '
        '#dlgpu<memory_space cluster = 0, memLoc = {type: DdrCuda, cluster: '
        '0}, dynInfo = {dynDims: [false, false, true, true], minShape: [1, '
        '3, 180, 1280]}>>) {\n'
        '  %0 = dlgpu.load {cluster = 0 : si32, opName = '
        '"Conv0_filter_per_channel_t0", size = 384 : i64} : '
        'memref<384xui8, #dlgpu<memory_space cluster = 0, memLoc = {type: '
        'DdrConstPublic, cluster: 0}>>\n'
        '}'
    )
    module = parse_mlir_text(text)
    op = module.functions[0].body.operations[0]
    self.assertEqual(op.attributes['opName'], 'Conv0_filter_per_channel_t0')
    self.assertEqual(op.attributes['cluster'], '0')
    self.assertIn('DdrConstPublic', op.result_types[0].memory_space)

  def test_block_argument_with_affine_map_and_memory_space_stays_single_arg(self):
    text = (
        'func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = '
        '0>>) {\n'
        '  %0 = dlgpu.launch_gtg(%arg0) {sym_name = "sub_0"} : '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> -> '
        'memref<1xf32, #dlgpu<memory_space cluster = 0>> {\n'
        '  ^bb0(%arg1: memref<1x3x540x360xf32, affine_map<(d0, d1, d2, d3) '
        '-> (d0 * 588612 + d1 * 196204 + d2 * 362 + d3)>, '
        '#dlgpu<memory_space cluster = 0, memLoc = {type: DdrCuda, c: 0}>>):\n'
        '    dlgpu.return %arg1 : memref<1xf32, #dlgpu<memory_space cluster '
        '= 0>>\n'
        '  }\n'
        '}'
    )

    module = parse_mlir_text(text)
    nested = module.functions[0].body.operations[0].regions[0].blocks[0]
    self.assertEqual(len(nested.arguments), 1)
    self.assertEqual(nested.arguments[0].name, '%arg1')
    self.assertIn('#dlgpu<memory_space', nested.arguments[0].type_text)

  def test_fallback_keeps_raw_attribute_text(self):
    text = (
        'func.func @main() { %0 = custom.foo {opaque = '
        '#relay<binding_attr {"x": 1}>} : () -> i32 }'
    )
    module = parse_mlir_text(text)
    op = module.functions[0].body.operations[0]
    self.assertIn('#relay<binding_attr', op.attributes_text)


if __name__ == '__main__':
  unittest.main()
