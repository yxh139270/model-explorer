from __future__ import annotations

from typing import Dict

from .adapter import Adapter, AdapterMetadata
from .custom_dialect_graph_builder import build_graph
from .custom_dialect_parser import parse_mlir_text


class CustomDialectAdapter(Adapter):
  metadata = AdapterMetadata(
      id='custom_dialect_mlir',
      name='Custom dialect MLIR adapter',
      description=(
          'Parses MLIR files with custom dialects unsupported by the built-in '
          'converter.'
      ),
      fileExts=['mlir'],
  )

  def can_handle_file(self, model_path: str) -> bool:
    with open(model_path, 'r', encoding='utf-8') as f:
      text = f.read(8192)
    return '#dlgpu<' in text or 'dlgpu.' in text or 'dlhlo.' in text

  def convert(self, model_path: str, settings: Dict):
    del settings
    with open(model_path, 'r', encoding='utf-8') as f:
      text = f.read()
    module = parse_mlir_text(text)
    return {'graphs': [build_graph(module)]}
