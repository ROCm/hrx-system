# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""External identities and caller dataflow through C++ source roundtrips."""

import subprocess
import sys

from loom.format.cxx import load_module, render_module
from loom.ir import Module


def _import_cxx(importer: str, source: bytes) -> Module:
    result = subprocess.run(
        [importer, "--to=bc", "-"],
        input=source,
        capture_output=True,
        check=True,
    )
    assert result.stderr == b""
    return load_module(result.stdout)


def _assert_external_calls(module: Module) -> None:
    declarations = {
        operation.attributes["callee"]: (
            operation.attributes["import_module"],
            operation.attributes.get("import_symbol"),
            tuple(repr(module.values[value].type) for value in operation.operands),
            tuple(repr(module.values[value].type) for value in operation.results),
        )
        for operation in module.body.ops
        if operation.name == "func.decl"
    }
    assert declarations == {
        "native.observe": ("native", None, ("i32",), ()),
        "sample": ("native", None, ("i32",), ("i32",)),
        "lookup": ('host"quoted', "lookup\\entry", ("i32",), ("i32",)),
    }
    functions = [op for op in module.body.ops if op.name == "func.def"]
    assert len(functions) == 1
    function = functions[0]
    assert function.attributes["callee"] == "run"
    body = function.regions[0].blocks[0]
    assert [op.name for op in body.ops] == [
        "func.call",
        "func.call",
        "func.call",
        "func.return",
    ]
    observe, sample, lookup, returned = body.ops
    assert [op.attributes["callee"] for op in (observe, sample, lookup)] == [
        "native.observe",
        "sample",
        "lookup",
    ]
    assert observe.operands == sample.operands == body.arg_ids
    assert lookup.operands == sample.results
    assert returned.operands == lookup.results


def main(importer: str) -> None:
    source = r"""[[loom::import("native")]] unsigned sample(unsigned value);
[[loom::symbol("native.observe"), loom::import("native")]]
void observe(unsigned value);
[[loom::import("host\"quoted", "lookup\\entry")]]
unsigned lookup(unsigned value);

unsigned run(unsigned value) {
  observe(value);
  return lookup(sample(value));
}
"""
    original = _import_cxx(importer, source.encode())
    _assert_external_calls(original)
    projected = _import_cxx(importer, render_module(original).encode())
    _assert_external_calls(projected)


if __name__ == "__main__":
    main(sys.argv[1])
