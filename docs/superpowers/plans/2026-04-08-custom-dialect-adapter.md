# Custom Dialect Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure Python custom-dialect MLIR adapter that can parse `/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir` into a navigable Model Explorer graph without relying on the built-in MLIR converter.

**Architecture:** Build a small parser pipeline inside the Python package: text scanning, structural MLIR parsing, dialect-aware metadata extraction, and graph conversion. Keep the parser generic enough for future custom dialect handlers while making `dlgpu` the first fully supported dialect.

**Tech Stack:** Python 3, existing `model_explorer` adapter API, `graph_builder`, `unittest`, pure-text MLIR parsing.

---

## File Structure

### New files

- `src/server/package/src/model_explorer/custom_dialect_adapter.py`
  - Adapter entry point.
- `src/server/package/src/model_explorer/custom_dialect_ir.py`
  - Small internal IR dataclasses for parsed functions, operations, values, blocks, and regions.
- `src/server/package/src/model_explorer/custom_dialect_parser.py`
  - Parser orchestration and top-level parse entry point.
- `src/server/package/src/model_explorer/custom_dialect_tokenizer.py`
  - Delimiter-depth-aware scanners and split helpers.
- `src/server/package/src/model_explorer/custom_dialect_types.py`
  - Type and attribute parsing helpers.
- `src/server/package/src/model_explorer/custom_dialect_handlers.py`
  - Generic fallback plus `dlgpu` metadata extraction helpers.
- `src/server/package/src/model_explorer/custom_dialect_graph_builder.py`
  - Conversion from internal IR into `graph_builder.Graph`.
- `src/server/package/tests/test_custom_dialect_tokenizer.py`
  - Unit tests for splitting/token depth behavior.
- `src/server/package/tests/test_custom_dialect_parser.py`
  - Unit tests for operation/function parsing.
- `src/server/package/tests/test_custom_dialect_graph_builder.py`
  - Unit tests for IR-to-graph conversion.
- `src/server/package/tests/test_custom_dialect_adapter_integration.py`
  - Integration test for the target MLIR file.

### Modified files

- `src/server/package/src/model_explorer/extension_manager.py`
  - Register the new adapter alongside built-ins.
- `src/server/package/src/model_explorer/__init__.py`
  - Export any adapter-facing symbols if needed by tests or extension loading.

### Test data

- Reuse `/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir` for the primary integration target.
- Add small inline MLIR strings directly in unit tests for focused parser cases.

## Task 1: Add tokenizer and depth-aware split helpers

**Files:**
- Create: `src/server/package/src/model_explorer/custom_dialect_tokenizer.py`
- Test: `src/server/package/tests/test_custom_dialect_tokenizer.py`

- [ ] **Step 1: Write the failing tokenizer tests**

```python
import unittest

from model_explorer.custom_dialect_tokenizer import split_top_level, find_matching_brace


class CustomDialectTokenizerTest(unittest.TestCase):

  def test_split_top_level_ignores_nested_commas(self):
    text = 'memref<1x3xf32, #dlgpu<memory_space cluster = 0, dynInfo = {minShape: [1, 3]}>>, %arg1, affine_map<(d0, d1) -> (d0, d1)>'
    self.assertEqual(
        split_top_level(text, ','),
        [
            'memref<1x3xf32, #dlgpu<memory_space cluster = 0, dynInfo = {minShape: [1, 3]}>>',
            '%arg1',
            'affine_map<(d0, d1) -> (d0, d1)>',
        ],
    )

  def test_find_matching_brace_handles_nested_regions(self):
    text = '{ %0 = dlgpu.launch_gtg(%arg0) { ^bb0(%arg1: memref<1xf32>): dlgpu.return %arg1 : memref<1xf32> } }'
    start = text.index('{')
    end = find_matching_brace(text, start)
    self.assertEqual(text[end], '}')
    self.assertEqual(end, len(text) - 1)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_tokenizer -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors for `split_top_level` / `find_matching_brace`

- [ ] **Step 3: Write minimal tokenizer implementation**

```python
from __future__ import annotations


DELIMITER_PAIRS = {
    '(': ')',
    '{': '}',
    '[': ']',
    '<': '>',
}


def split_top_level(text: str, delimiter: str) -> list[str]:
  parts: list[str] = []
  start = 0
  stack: list[str] = []
  for index, ch in enumerate(text):
    if ch in DELIMITER_PAIRS:
      stack.append(DELIMITER_PAIRS[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
    elif ch == delimiter and not stack:
      part = text[start:index].strip()
      if part:
        parts.append(part)
      start = index + 1
  tail = text[start:].strip()
  if tail:
    parts.append(tail)
  return parts


def find_matching_brace(text: str, start_index: int) -> int:
  stack: list[str] = []
  for index in range(start_index, len(text)):
    ch = text[index]
    if ch in DELIMITER_PAIRS:
      stack.append(DELIMITER_PAIRS[ch])
    elif stack and ch == stack[-1]:
      stack.pop()
      if not stack:
        return index
  raise ValueError(f'unmatched delimiter starting at {start_index}')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_tokenizer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/src/model_explorer/custom_dialect_tokenizer.py src/server/package/tests/test_custom_dialect_tokenizer.py
git commit -m "feat: add custom dialect tokenizer helpers"
```

## Task 2: Add internal IR and parse function headers, results, and block arguments

**Files:**
- Create: `src/server/package/src/model_explorer/custom_dialect_ir.py`
- Create: `src/server/package/src/model_explorer/custom_dialect_parser.py`
- Test: `src/server/package/tests/test_custom_dialect_parser.py`

- [ ] **Step 1: Write the failing parser tests**

```python
import unittest

from model_explorer.custom_dialect_parser import parse_mlir_text


class CustomDialectParserTest(unittest.TestCase):

  def test_parse_func_signature_with_custom_memref(self):
    text = '''func.func @main(%arg0: memref<1x3xf32, #dlgpu<memory_space cluster = 0>>) -> memref<1x3xf32, #dlgpu<memory_space cluster = 1>> {
      dlgpu.return %arg0 : memref<1x3xf32, #dlgpu<memory_space cluster = 0>>
    }'''
    module = parse_mlir_text(text)
    self.assertEqual(len(module.functions), 1)
    func = module.functions[0]
    self.assertEqual(func.name, 'main')
    self.assertEqual(func.arguments[0].name, '%arg0')
    self.assertIn('#dlgpu<memory_space', func.arguments[0].type_text)

  def test_parse_block_argument_and_multi_result_op(self):
    text = '''func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = 0>>) {
      %0:2 = dlgpu.scale_shape_variants(%arg0) {opName = "ScaleShape0"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>>, memref<2xf32, #dlgpu<memory_space cluster = 0>>
      %1 = dlgpu.launch_gtg(%0#0) {sym_name = "sub_0_infer_shape"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>> {
      ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):
        dlgpu.return %arg1 : memref<1xf32, #dlgpu<memory_space cluster = 0>>
      }
    }'''
    module = parse_mlir_text(text)
    op = module.functions[0].body.operations[0]
    self.assertEqual(op.result_count, 2)
    nested = module.functions[0].body.operations[1].regions[0].blocks[0]
    self.assertEqual(nested.arguments[0].name, '%arg1')


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_parser -v`
Expected: FAIL with `ModuleNotFoundError`, missing `parse_mlir_text`, or missing IR attributes

- [ ] **Step 3: Write minimal IR and parser implementation**

```python
from dataclasses import dataclass, field


@dataclass
class ParsedValue:
  name: str
  type_text: str = ''


@dataclass
class ParsedBlock:
  arguments: list[ParsedValue] = field(default_factory=list)
  operations: list['ParsedOperation'] = field(default_factory=list)


@dataclass
class ParsedRegion:
  blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass
class ParsedOperation:
  name: str
  result_names: list[str] = field(default_factory=list)
  result_count: int = 0
  attributes_text: str = ''
  type_text: str = ''
  operand_text: str = ''
  regions: list[ParsedRegion] = field(default_factory=list)


@dataclass
class ParsedFunction:
  name: str
  arguments: list[ParsedValue] = field(default_factory=list)
  result_types: list[str] = field(default_factory=list)
  body: ParsedBlock = field(default_factory=ParsedBlock)


@dataclass
class ParsedModule:
  functions: list[ParsedFunction] = field(default_factory=list)
```

```python
from model_explorer.custom_dialect_ir import ParsedBlock, ParsedFunction, ParsedModule, ParsedOperation, ParsedRegion, ParsedValue
from model_explorer.custom_dialect_tokenizer import split_top_level


def parse_mlir_text(text: str) -> ParsedModule:
  # Minimal first pass for one top-level func and one nested block form.
  ...
```

Implementation requirement for this step:
- parse `func.func @name(...) -> ... { ... }`
- parse top-level operations inside the function body
- parse `%x:n = op(...) ...`
- parse one nested region block with `^bb0(...)`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_parser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/src/model_explorer/custom_dialect_ir.py src/server/package/src/model_explorer/custom_dialect_parser.py src/server/package/tests/test_custom_dialect_parser.py
git commit -m "feat: add custom dialect parser core"
```

## Task 3: Parse structured type and attribute metadata for custom dialects

**Files:**
- Create: `src/server/package/src/model_explorer/custom_dialect_types.py`
- Create: `src/server/package/src/model_explorer/custom_dialect_handlers.py`
- Modify: `src/server/package/src/model_explorer/custom_dialect_parser.py`
- Test: `src/server/package/tests/test_custom_dialect_parser.py`

- [ ] **Step 1: Write the failing metadata extraction tests**

```python
def test_extract_dlgpu_attributes(self):
  text = '''func.func @main(%arg0: memref<1x3x1920x1920xf32, #dlgpu<memory_space cluster = 0, memLoc = {type: DdrCuda, cluster: 0}, dynInfo = {dynDims: [false, false, true, true], minShape: [1, 3, 180, 1280]}>>) {
    %0 = dlgpu.load {cluster = 0 : si32, opName = "Conv0_filter_per_channel_t0", size = 384 : i64} : memref<384xui8, #dlgpu<memory_space cluster = 0, memLoc = {type: DdrConstPublic, cluster: 0}>>
  }'''
  module = parse_mlir_text(text)
  op = module.functions[0].body.operations[0]
  self.assertEqual(op.attributes['opName'], 'Conv0_filter_per_channel_t0')
  self.assertEqual(op.attributes['cluster'], '0')
  self.assertIn('DdrConstPublic', op.result_types[0].memory_space)

def test_fallback_keeps_raw_attribute_text(self):
  text = '''func.func @main() { %0 = custom.foo {opaque = #relay<binding_attr {"x": 1}>} : () -> i32 }'''
  module = parse_mlir_text(text)
  op = module.functions[0].body.operations[0]
  self.assertIn('#relay<binding_attr', op.attributes_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_parser -v`
Expected: FAIL because parsed operations do not yet expose structured `attributes` or parsed result types

- [ ] **Step 3: Write minimal type and handler implementation**

```python
from dataclasses import dataclass


@dataclass
class ParsedType:
  raw_text: str
  kind: str = ''
  element_type: str = ''
  shape: list[str] | None = None
  memory_space: str = ''


def parse_type(type_text: str) -> ParsedType:
  kind = type_text.split('<', 1)[0].strip() if '<' in type_text else type_text.strip()
  memory_space = ''
  if '#dlgpu<' in type_text:
    memory_space = type_text[type_text.index('#dlgpu<'):]
  return ParsedType(raw_text=type_text, kind=kind, memory_space=memory_space)
```

```python
def extract_operation_metadata(op_name: str, attributes_text: str) -> dict[str, str]:
  metadata: dict[str, str] = {}
  for key in ['opName', 'sym_name', 'cluster', 'ggType', 'gtgType', 'hashKey', 'size']:
    # implement minimal text extraction with top-level scanning
    ...
  return metadata
```

Implementation requirement for this step:
- parse and store `attributes` as a dict for recognized keys
- parse result types into objects or structured dicts
- preserve raw text for unknown/custom attributes

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_parser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/src/model_explorer/custom_dialect_types.py src/server/package/src/model_explorer/custom_dialect_handlers.py src/server/package/src/model_explorer/custom_dialect_parser.py src/server/package/tests/test_custom_dialect_parser.py
git commit -m "feat: extract custom dialect metadata"
```

## Task 4: Convert parsed IR into Model Explorer graph nodes and SSA edges

**Files:**
- Create: `src/server/package/src/model_explorer/custom_dialect_graph_builder.py`
- Test: `src/server/package/tests/test_custom_dialect_graph_builder.py`

- [ ] **Step 1: Write the failing graph builder tests**

```python
import unittest

from model_explorer.custom_dialect_graph_builder import build_graph
from model_explorer.custom_dialect_parser import parse_mlir_text


class CustomDialectGraphBuilderTest(unittest.TestCase):

  def test_build_graph_preserves_labels_and_namespaces(self):
    text = '''func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = 0>>) {
      %0 = dlgpu.scale_shape_variants(%arg0) {opName = "ScaleShape0"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>>
      %1 = dlgpu.launch_gtg(%0) {sym_name = "sub_0_infer_shape"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>> {
      ^bb0(%arg1: memref<1xf32, #dlgpu<memory_space cluster = 0>>):
        %2 = dlgpu.infer_shape(%arg1) {opName = "FusedConv0"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>>
        dlgpu.return %2 : memref<1xf32, #dlgpu<memory_space cluster = 0>>
      }
    }'''
    graph = build_graph(parse_mlir_text(text))
    labels = {node.label for node in graph.nodes}
    self.assertIn('ScaleShape0', labels)
    self.assertIn('FusedConv0', labels)
    infer_shape_node = [node for node in graph.nodes if node.label == 'FusedConv0'][0]
    self.assertEqual(infer_shape_node.namespace, 'main/sub_0_infer_shape')

  def test_build_graph_creates_input_edges(self):
    text = '''func.func @main(%arg0: memref<1xf32, #dlgpu<memory_space cluster = 0>>) {
      %0 = dlgpu.infer_shape(%arg0) {opName = "InputMove"} : memref<1xf32, #dlgpu<memory_space cluster = 0>> -> memref<1xf32, #dlgpu<memory_space cluster = 0>>
    }'''
    graph = build_graph(parse_mlir_text(text))
    target = [node for node in graph.nodes if node.label == 'InputMove'][0]
    self.assertEqual(target.incomingEdges[0].sourceNodeId, 'arg_%arg0')


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_graph_builder -v`
Expected: FAIL because `build_graph` does not exist or does not emit namespaces / edges yet

- [ ] **Step 3: Write minimal graph builder implementation**

```python
from model_explorer import graph_builder


def build_graph(module):
  graph = graph_builder.Graph(id='custom_dialect_graph')
  # create synthetic input nodes for function args
  # create one node per op
  # derive label from opName, sym_name, or op.name
  # derive namespace from nested region path
  # add incomingEdges from SSA producer map
  return graph
```

Implementation requirement for this step:
- create argument nodes with ids like `arg_%arg0`
- create deterministic op node ids
- attach `attrs` containing `op_type`, `raw_attributes`, `result_types`, and `memory_space`
- support nested namespace paths for `launch_gtg` / region-contained ops

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_graph_builder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/src/model_explorer/custom_dialect_graph_builder.py src/server/package/tests/test_custom_dialect_graph_builder.py
git commit -m "feat: build model explorer graphs for custom dialect mlir"
```

## Task 5: Add adapter entry point and register it in extension loading

**Files:**
- Create: `src/server/package/src/model_explorer/custom_dialect_adapter.py`
- Modify: `src/server/package/src/model_explorer/extension_manager.py`
- Modify: `src/server/package/src/model_explorer/__init__.py`
- Test: `src/server/package/tests/test_custom_dialect_adapter_integration.py`

- [ ] **Step 1: Write the failing adapter registration tests**

```python
import os
import unittest

from model_explorer.custom_dialect_adapter import CustomDialectAdapter
from model_explorer.extension_manager import ExtensionManager


class CustomDialectAdapterIntegrationTest(unittest.TestCase):

  def test_adapter_detects_custom_dialect_mlir(self):
    adapter = CustomDialectAdapter()
    path = '/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir'
    self.assertTrue(adapter.can_handle_file(path))

  def test_extension_manager_registers_custom_dialect_adapter(self):
    manager = ExtensionManager()
    manager.load_extensions()
    extension_ids = [ext.metadata.id for ext in manager.extensions]
    self.assertIn('custom_dialect_mlir', extension_ids)


if __name__ == '__main__':
  unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_adapter_integration -v`
Expected: FAIL because the adapter class and registration do not exist yet

- [ ] **Step 3: Write minimal adapter and registration implementation**

```python
from typing import Dict

from .adapter import Adapter, AdapterMetadata
from .custom_dialect_graph_builder import build_graph
from .custom_dialect_parser import parse_mlir_text


class CustomDialectAdapter(Adapter):
  metadata = AdapterMetadata(
      id='custom_dialect_mlir',
      name='Custom dialect MLIR adapter',
      description='Parses MLIR files with custom dialects unsupported by the built-in converter.',
      fileExts=['mlir'],
  )

  def can_handle_file(self, model_path: str) -> bool:
    with open(model_path, 'r') as f:
      text = f.read(8192)
    return '#dlgpu<' in text or 'dlgpu.' in text or 'dlhlo.' in text

  def convert(self, model_path: str, settings: Dict):
    with open(model_path, 'r') as f:
      text = f.read()
    module = parse_mlir_text(text)
    return {'graphs': [build_graph(module)]}
```

Registration requirement for this step:
- add `.custom_dialect_adapter` to `BUILTIN_ADAPTER_MODULES` in `extension_manager.py`
- export any symbols needed in `__init__.py` if tests import through `model_explorer`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_adapter_integration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/src/model_explorer/custom_dialect_adapter.py src/server/package/src/model_explorer/extension_manager.py src/server/package/src/model_explorer/__init__.py src/server/package/tests/test_custom_dialect_adapter_integration.py
git commit -m "feat: register custom dialect mlir adapter"
```

## Task 6: Prove end-to-end parsing of the target dynamic graph file

**Files:**
- Modify: `src/server/package/tests/test_custom_dialect_adapter_integration.py`

- [ ] **Step 1: Write the failing end-to-end target-file test**

```python
def test_target_dynamic_graph_builds_non_empty_graph(self):
  adapter = CustomDialectAdapter()
  path = '/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir'
  graphs = adapter.convert(path, settings={})
  graph = graphs['graphs'][0]
  self.assertGreater(len(graph.nodes), 50)
  labels = {node.label for node in graph.nodes}
  self.assertIn('ScaleShape0', labels)
  self.assertIn('FusedConv0', labels)
  self.assertTrue(any(node.namespace.startswith('main/sub_0') for node in graph.nodes))
  self.assertTrue(any(any(attr.key == 'memory_space' for attr in node.attrs) for node in graph.nodes))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_adapter_integration.CustomDialectAdapterIntegrationTest.test_target_dynamic_graph_builds_non_empty_graph -v`
Expected: FAIL because the parser still misses one or more real-world constructs in `dynamic_graph.mlir`

- [ ] **Step 3: Extend implementation minimally until the target test passes**

Implementation checklist for this step:
- handle additional real constructs discovered in `dynamic_graph.mlir`
- support `%value#0` style multi-result references if present
- preserve nested namespaces from `launch_graphgrid` and `launch_gtg`
- avoid crashing on unknown attrs by retaining raw text
- ensure graph node attributes include extracted memory-space-related fields

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.server.package.tests.test_custom_dialect_adapter_integration.CustomDialectAdapterIntegrationTest.test_target_dynamic_graph_builds_non_empty_graph -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/server/package/tests/test_custom_dialect_adapter_integration.py src/server/package/src/model_explorer/custom_dialect_*.py
git commit -m "feat: support dynamic graph custom dialect mlir"
```

## Task 7: Run full verification for the new adapter path

**Files:**
- Modify: `src/server/package/tests/test_custom_dialect_tokenizer.py`
- Modify: `src/server/package/tests/test_custom_dialect_parser.py`
- Modify: `src/server/package/tests/test_custom_dialect_graph_builder.py`
- Modify: `src/server/package/tests/test_custom_dialect_adapter_integration.py`

- [ ] **Step 1: Add one final regression test for the original failure mode**

```python
def test_custom_adapter_handles_builtin_mlir_failure_case(self):
  adapter = CustomDialectAdapter()
  path = '/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir'
  result = adapter.convert(path, settings={})
  self.assertIn('graphs', result)
  self.assertEqual(len(result['graphs']), 1)
```

- [ ] **Step 2: Run the full custom dialect test suite**

Run: `python3 -m unittest discover src/server/package/tests -v`
Expected: PASS

- [ ] **Step 3: If any test fails, make the minimal fix and re-run the suite**

Run: `python3 -m unittest discover src/server/package/tests -v`
Expected: PASS with no failures and no unexpected errors

- [ ] **Step 4: Smoke-test the adapter directly on the target file**

Run: `python3 - <<'PY'
from model_explorer.custom_dialect_adapter import CustomDialectAdapter

adapter = CustomDialectAdapter()
path = '/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir'
result = adapter.convert(path, settings={})
graph = result['graphs'][0]
print('nodes', len(graph.nodes))
print('sample', [node.label for node in graph.nodes[:10]])
PY`
Expected: prints a positive node count and sample labels including custom dialect operations

- [ ] **Step 5: Commit**

```bash
git add src/server/package/tests/test_custom_dialect_tokenizer.py src/server/package/tests/test_custom_dialect_parser.py src/server/package/tests/test_custom_dialect_graph_builder.py src/server/package/tests/test_custom_dialect_adapter_integration.py src/server/package/src/model_explorer/custom_dialect_*.py
git commit -m "test: cover custom dialect mlir adapter regression"
```
