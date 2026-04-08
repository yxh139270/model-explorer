import unittest

from model_explorer.custom_dialect_tokenizer import (
    find_matching_brace,
    split_top_level,
)


class CustomDialectTokenizerTest(unittest.TestCase):

  def test_split_top_level_ignores_nested_commas(self):
    text = (
        'memref<1x3xf32, #dlgpu<memory_space cluster = 0, dynInfo = {minShape: '
        '[1, 3]}>>, %arg1, affine_map<(d0, d1) -> (d0, d1)>'
    )
    self.assertEqual(
        split_top_level(text, ','),
        [
            (
                'memref<1x3xf32, #dlgpu<memory_space cluster = 0, dynInfo = '
                '{minShape: [1, 3]}>>'
            ),
            '%arg1',
            'affine_map<(d0, d1) -> (d0, d1)>',
        ],
    )

  def test_find_matching_brace_handles_nested_regions(self):
    text = (
        '{ %0 = dlgpu.launch_gtg(%arg0) '
        '{ ^bb0(%arg1: memref<1xf32>): dlgpu.return %arg1 : memref<1xf32> } }'
    )
    start = text.index('{')
    end = find_matching_brace(text, start)
    self.assertEqual(text[end], '}')
    self.assertEqual(end, len(text) - 1)


if __name__ == '__main__':
  unittest.main()
