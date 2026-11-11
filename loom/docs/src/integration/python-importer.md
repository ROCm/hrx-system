# Construct importer output in Python

**Complete example:**
[`loom/docs/examples/integration/python-importer/`](https://github.com/ROCm/hrx-system/tree/main/loom/docs/examples/integration/python-importer)

Loom's Python construction API is the target half of an importer. A frontend
owns its source language's parsing, traversal, diagnostics, and source model.
It translates the facts that survive that traversal into ordinary Loom IR
instead of adapting its foreign AST into a Loom-owned compatibility layer.

The example begins at that stable boundary. It imports no frontend package and
defines no source-language nodes: two buffer arguments, one exported kernel,
and a fixed launch shape are sufficient to construct and serialize a complete
module using the `hrx-loom` wheel.

## Construct the kernel envelope and body

`KernelModuleSpec` describes the source-independent ABI and launch facts.
`create_kernel_module` creates the target record, `kernel.def`, configuration
region, and body arguments, then returns typed handles for filling the body.

```python
--8<-- "examples/integration/python-importer/importer.py:importer"
```

The generated builder methods expose each Loom operation with its named
operands, attributes, and results. In the copy body,
`kernel.workitem.dispatch.id` produces a logical `index`. Buffer addressing
uses the physical `offset` domain, so `index.cast` makes that semantic boundary
visible before the load and store.

`kernel_module_ops` returns the operation declarations required by the selected
target preset. Passing the same declaration set to verification, printing, and
bytecode serialization keeps all three consumers on the module's exact
dialect surface. The example uses the portable `reference` target; an importer
that has selected an exact device supplies its target preset instead.

Running the example with an installed SDK prints canonical Loom text:

```shell
python importer.py > copy_64_bytes.loom
loom-format --check copy_64_bytes.loom
```

```loom title="copy_64_bytes.loom"
--8<-- "examples/integration/python-importer/expected.loom"
```

## Verify before crossing the serialization boundary

`verify_module(...).raise_if_errors()` rejects an invalid producer result while
the source-specific importer still owns the context needed to explain it. A
verified `Module` can then cross either of Loom's equivalent serialization
boundaries: canonical text for people and source control, or `.loombc` for
compact loading and composition.

The example test writes bytecode, reads it into a new module, verifies while
reading, and compares the recovered canonical text with the same exact golden:

```python
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module

bytecode = write_module(module, op_decls=MODULE_OPS)
loaded = read_module(bytecode, op_decls=MODULE_OPS, verify=True)
```

Source locations fit the same ownership split. A frontend records locations
while traversing its source and passes `location_id` to generated builder
methods. Loom retains those locations through text or bytecode and uses them in
later diagnostics; it does not need to own the frontend's AST after import.
