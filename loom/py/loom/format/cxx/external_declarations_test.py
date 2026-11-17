# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""External identities and caller dataflow through C++ source roundtrips."""

import subprocess
import sys
from collections import Counter

from loom.format.cxx import load_module, render_module
from loom.format.cxx.context import walk_operations
from loom.ir import Module, ReferenceOwnership, ReferenceType, ShapedType


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


def _assert_external_object_calls(module: Module) -> None:
    type_declarations = [
        operation for operation in module.body.ops if operation.name == "type.decl"
    ]
    assert len(type_declarations) == 1
    assert type_declarations[0].attributes == {
        "name": "Object",
        "namespace": "native",
        "type_name": "object.v1",
    }

    declarations = {
        operation.attributes["callee"]: operation
        for operation in module.body.ops
        if operation.name == "func.decl"
    }
    assert set(declarations) == {"native.create", "native.observe"}
    create = declarations["native.create"]
    assert create.attributes["import_module"] == "native"
    assert create.attributes["import_symbol"] == "create.v2"
    assert len(create.results) == 1
    created_type = module.values[create.results[0]].type
    assert isinstance(created_type, ReferenceType)
    assert created_type.ownership == ReferenceOwnership.OWNED
    assert (created_type.namespace, created_type.name) == ("native", "object.v1")

    observe = declarations["native.observe"]
    assert observe.attributes["import_module"] == "native"
    assert observe.attributes["import_symbol"] == "observe.v2"
    assert observe.attributes["signature_flags"] == b"\x00\x00\x01\x00"
    object_type = module.values[observe.operands[0]].type
    assert isinstance(object_type, ReferenceType)
    assert object_type.ownership == ReferenceOwnership.BORROWED
    assert (object_type.namespace, object_type.name) == ("native", "object.v1")
    objects_type = module.values[observe.operands[2]].type
    assert isinstance(objects_type, ShapedType)
    assert isinstance(objects_type.element_type, ReferenceType)
    assert objects_type.element_type.ownership == ReferenceOwnership.BORROWED
    assert (objects_type.element_type.namespace, objects_type.element_type.name) == (
        "native",
        "object.v1",
    )


def _operation_histogram(module: Module) -> Counter[str]:
    return Counter(operation.name for operation in walk_operations(module.body))


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

    object_source = r"""#include <loomcxx/object.h>
#include <loomcxx/view.h>

struct [[loom::import("native", "object.v1")]] Object;
using Objects = loom::type::view<loom::type::shape<loom::type::dynamic>, loom::ptr<Object>>;

[[loom::symbol("native.observe"), loom::import("native", "observe.v2")]]
unsigned observe(loom::ptr<Object> object,
                 [[loom::access("read")]] Objects objects);
[[loom::symbol("native.create"), loom::import("native", "create.v2")]]
loom::ref<Object> create();

unsigned run(loom::ptr<Object> object,
             [[loom::access("read")]] Objects objects) {
  auto owner = create();
  auto copy = owner;
  return observe(object, objects) + observe(owner.get(), objects) +
         observe(copy.get(), objects);
}
"""
    original = _import_cxx(importer, object_source.encode())
    _assert_external_object_calls(original)
    projected = _import_cxx(importer, render_module(original).encode())
    _assert_external_object_calls(projected)
    assert _operation_histogram(projected) == _operation_histogram(original)


if __name__ == "__main__":
    main(sys.argv[1])
