# Custom Dialect Adapter Design

## Goal

Build a pure Python `custom dialect adapter` for Model Explorer that can parse MLIR text files containing custom dialects unsupported by the built-in MLIR converter. The first milestone is to successfully visualize `/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir`, while keeping the architecture extensible for additional custom dialects beyond `dlgpu`.

## Background

The current built-in MLIR adapter fails on the target file with:

```text
INVALID_ARGUMENT: Failed to parse MLIR module: unsupported memory space Attribute
```

The immediate root cause is that the built-in converter rejects custom memory space attributes such as `#dlgpu<memory_space ...>` during MLIR parsing.

The `dltc-viewer` project already contains a JavaScript MLIR parser that handles several custom dialect constructs, including `dlgpu`-specific syntax. This design does not embed or call that JavaScript parser. Instead, it uses the same broad parsing strategy in a pure Python implementation suitable for Model Explorer's adapter extension system.

## Scope

### In scope

- Add a new pure Python adapter extension named `custom dialect adapter`
- Implement a small MLIR text parser for a targeted, extensible subset of syntax
- Support the syntax needed by `dynamic_graph.mlir`
- Preserve operation hierarchy, SSA dependencies, and key type/attribute information
- Map parsed operations into `model_explorer.graph_builder.Graph`
- Structure the parser so future custom dialect handlers can be added cleanly

### Out of scope

- Full MLIR language compatibility
- MLIR bytecode support
- Replacing the built-in MLIR adapter for standard MLIR cases
- Semantic verification of custom dialects
- Complete support for every custom dialect already handled in `dltc-viewer`

## Product Direction

The adapter should be named generically because `dlgpu` is only the first supported custom dialect. The implementation should behave as a `custom dialect adapter` with:

- a generic parser core for MLIR-like text structure
- dialect-specific handlers for custom syntax and metadata extraction
- first-class support for `dlgpu` in milestone one
- room to add handlers such as `dlhlo` later without redesigning the adapter

## User Experience

When a user opens a custom-dialect MLIR file through Model Explorer:

- the file loads through the new adapter instead of the built-in MLIR converter
- the graph renders as an operation graph first
- nested regions such as `launch_graphgrid` and `launch_gtg` appear as namespaces/layers
- node details expose useful custom dialect metadata such as memory space, shapes, `dynInfo`, and memory location data

If parsing encounters unsupported syntax, the adapter should fail with a precise, debuggable message containing the phase and approximate location of failure.

## Architecture

### 1. Adapter layer

Create a new adapter extension that implements the standard `Adapter` interface and returns `{'graphs': [graph]}`.

Responsibilities:

- advertise itself as a custom adapter for `.mlir`
- decide whether a file should be handled by this adapter based on content heuristics for custom dialects
- invoke the parser and graph conversion pipeline
- surface parser errors as readable adapter errors

The adapter should be designed so it can explicitly target custom-dialect MLIR files without changing the built-in MLIR path for standard files.

### 2. Parser core

The parser core should be a pure Python structural parser, not a full MLIR frontend.

Responsibilities:

- read MLIR text
- maintain delimiter depth for `()`, `{}`, `[]`, `<>`
- split top-level constructs safely without breaking nested types or attributes
- parse operations, regions, blocks, block arguments, function signatures, results, and terminators
- record source offsets or line numbers for errors

The parser should follow the same high-level idea as `dltc-viewer/source/mlir.js`: treat MLIR as nested operations plus SSA values, not as flat lines matched by isolated regular expressions.

### 3. Intermediate representation

Introduce a small internal IR for the adapter pipeline. It should represent:

- module/function containers
- operations
- block arguments
- result values
- operand references
- nested regions/blocks
- parsed or raw attributes
- parsed or raw types

This internal IR is not user-facing. Its purpose is to decouple text parsing from Model Explorer graph generation.

### 4. Dialect handler layer

Add a dialect handler registry or dispatch layer.

Milestone one requires:

- a generic fallback handler for unknown operations and attributes
- a `dlgpu` handler for extracting better labels and custom metadata

Future handlers can specialize:

- operation labels
- namespace naming
- attribute decoding
- constant interpretation
- edge naming or output metadata

### 5. Graph conversion layer

Convert the internal IR into `graph_builder.Graph` and `graph_builder.GraphNode`.

Responsibilities:

- create one graph node per operation
- assign stable node ids
- derive labels from `opName`, `sym_name`, or operation name fallback
- derive namespaces from function and nested region structure
- connect SSA producer/consumer edges
- generate input nodes for function arguments and block arguments when needed
- attach attributes and output metadata for inspection in the UI

## Parsing Strategy

### Structural parsing

The parser should proceed in layers:

1. Token-aware scanning of the source text with bracket-depth tracking
2. Identification of top-level functions/modules and operation bodies
3. Parsing of individual operations into:
   - result list
   - operation name
   - operand list
   - attribute dictionary text
   - type signature text
   - nested regions
4. Parsing of selected types and attributes into structured forms where useful
5. Graph construction from SSA dependencies

This avoids fragile line-based parsing while remaining much smaller than a full MLIR implementation.

### Supported syntax in milestone one

The first version must correctly parse the subset already present in `dynamic_graph.mlir`, including:

- `func.func @main(...) -> ... { ... }`
- SSA results such as `%0 = ...`
- multi-result syntax such as `%75:8 = ...`
- operations named like `dlgpu.load`, `dlgpu.launch_graphgrid`, `dlgpu.launch_gtg`, `dlgpu.infer_shape`, `dlgpu.return`
- block labels and arguments such as `^bb0(%arg1: ...)`
- nested operation regions inside braces
- type forms including:
  - `memref<...>`
  - `tensor<...>`
  - `vector<...>`
  - `affine_map<...>`
  - custom attrs/types such as `#dlgpu<...>` and `#relay<...>`

### Attribute parsing

Attributes should be parsed with two levels of fidelity:

- structured extraction for fields needed by the UI or graph naming
- raw text preservation for everything else

Milestone-one structured extraction should cover at least:

- `opName`
- `sym_name`
- `cluster`
- `ggType`
- `gtgType`
- `hashKey`
- `size`
- `memLoc`
- `dynInfo`

Anything not explicitly decoded should remain available as raw attribute text.

### Type parsing

Type parsing should also use a hybrid strategy.

Structured extraction should cover enough to display:

- base type kind such as `memref`, `tensor`, or `vector`
- element type
- rank/shape where obvious
- memory space text for custom memory space annotations
- raw full type string for lossless inspection

The parser does not need to normalize all MLIR types into a canonical schema. It only needs enough structure for graph display and metadata inspection.

## Graph Semantics

### Node creation

Every parsed operation becomes a `GraphNode`.

Label priority:

1. `opName`
2. `sym_name`
3. canonical operation name like `dlgpu.load`

Node id should be deterministic and unique within the graph. A combination of operation location/order plus operation name is sufficient.

### Namespace mapping

Namespace should reflect nested operation structure so the visualizer can collapse custom subgraphs naturally.

Examples:

- `main`
- `main/sub_0`
- `main/sub_0/sub_0_infer_shape`

For `launch_graphgrid` and `launch_gtg`, the namespace should prefer semantic names like `sym_name` where available.

### Edges

Edges should be derived from SSA producer/consumer relationships.

- if an operation consumes `%87`, it should connect from the operation that produced `%87`
- if a value comes from a function argument or block argument, create a synthetic input node or argument node so the graph remains connected
- for multi-result ops, preserve output slot identity where practical using `sourceNodeOutputId`

### Node metadata

Each node should expose useful inspection details through `attrs` and `outputsMetadata`.

Candidate node attributes:

- `op_type`
- `raw_attributes`
- `result_types`
- `operand_types`
- `memory_space`
- `mem_shape`
- `dyn_info`
- `mem_loc`

Candidate output metadata:

- output type
- output shape
- output index

## Error Handling

Failures should be explicit and local.

Error messages should include:

- parser phase, such as `tokenize`, `parse operation`, `parse type`, or `build graph`
- line number or nearest location if available
- a short source snippet when practical
- the unsupported construct summary

The parser should prefer preserving unknown text over rejecting input. Only fail when core structure cannot be recovered.

## Testing Strategy

Testing should follow TDD.

### Unit tests

- delimiter-depth splitting for nested MLIR syntax
- parsing of function signatures with custom memref types
- parsing of representative `dlgpu` operations
- parsing of multi-result operations like `%75:8 = ...`
- block argument parsing for `^bb0(...)`
- extraction of custom attributes such as `opName`, `sym_name`, `dynInfo`, `memLoc`

### Integration tests

- end-to-end conversion of `dynamic_graph.mlir` into a non-empty `Graph`
- assertions that key node labels appear, such as `ScaleShape0` or `FusedConv0`
- assertions that namespaces include nested launch regions
- assertions that nodes carry memory-space-related attributes

### Regression goal

The original failure case should be covered by a test that proves the new adapter can parse the file without relying on the built-in MLIR converter.

## Risks and Mitigations

### Risk: MLIR syntax surface grows quickly

Mitigation:

- keep the parser scoped to structural MLIR plus targeted dialect features
- preserve unknown attributes/types as raw text
- add dialect handlers incrementally

### Risk: `.mlir` conflicts with built-in adapter selection

Mitigation:

- use clear adapter selection heuristics based on file contents
- only route files with custom-dialect signals such as `#dlgpu<`, `dlgpu.`, or similar markers into this adapter path

### Risk: block/region data flow is tricky

Mitigation:

- represent block arguments explicitly in the internal IR
- start with correct visual connectivity, then refine semantic edge labeling later

## Milestone Definition

Milestone one is complete when:

- the new `custom dialect adapter` exists as a pure Python adapter
- `/home/xuehua/project/dltc-viewer/samples/dltc_v2/dynamic_graph.mlir` loads successfully through it
- the resulting graph is navigable in Model Explorer
- major `dlgpu` operations and nested launch regions are visible
- key custom metadata is inspectable in the side panel
- the code structure clearly supports adding more custom dialect handlers later

## Open Follow-up Work

Not required for milestone one, but enabled by this design:

- add `dlhlo`-specific handler logic
- support more alias/attribute forms from `dltc-viewer`
- enrich edge naming with semantic tensor names
- improve synthetic input/output node presentation
- consider partial reuse of parser logic for other custom MLIR files
