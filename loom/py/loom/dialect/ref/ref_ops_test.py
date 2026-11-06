# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Default Python authoring and serialization of managed handle lifetimes."""

import pytest

from loom.builders import default_ops, default_types, module_builder
from loom.builtin_types import ptr_type, ref_type
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.printer import Printer
from loom.ir import ScalarType, ScalarTypeKind
from loom.verify import verify_module


@pytest.mark.parametrize("handle_builder", [ref_type, ptr_type])
def test_author_and_serialize_independent_handles(handle_builder) -> None:
    module, builder = module_builder()
    builder.type.decl(namespace="media", type_name="frame", name="Frame")
    handle_type = handle_builder(module=module, declaration="Frame")
    source = builder.value("source", handle_type)
    body = builder.region()
    builder.func.def_(
        callee="copy",
        args=[source],
        results=[handle_type, handle_type],
        body=body,
    )
    with builder.insertion_block(body.blocks[0]):
        (duplicate,) = builder.ref.duplicate(sources=[source], results=[handle_type], names=["duplicate"])
        builder.func.return_(operands=[source, duplicate])

    ops = default_ops()
    assert not verify_module(module, ops=ops).has_errors
    serialized = write_module(module, op_decls=ops)
    restored = read_module(serialized, op_decls=ops)
    assert not verify_module(restored, ops=ops).has_errors

    printer = Printer()
    printer.register_ops(ops)
    printer.register_types(default_types())
    text = printer.print_module(module)
    assert "ref.duplicate %source" in text
    assert printer.print_module(restored) == text
    assert duplicate.type == source.type == handle_type
    assert duplicate.id != source.id


@pytest.mark.parametrize("left_builder", [ref_type, ptr_type])
@pytest.mark.parametrize("right_builder", [ref_type, ptr_type])
def test_author_and_serialize_null_and_identity(left_builder, right_builder) -> None:
    module, builder = module_builder()
    builder.type.decl(namespace="media", type_name="frame", name="Frame")
    left_type = left_builder(module=module, declaration="Frame")
    right_type = right_builder(module=module, declaration="Frame")
    predicate_type = ScalarType(ScalarTypeKind.I1)
    body = builder.region()
    builder.func.def_(callee="inspect", results=[predicate_type, predicate_type], body=body)
    with builder.insertion_block(body.blocks[0]):
        left = builder.ref.null(results=[left_type], name="left")
        right = builder.ref.null(results=[right_type], name="right")
        missing = builder.ref.compare_null(value=left, name="missing")
        same = builder.ref.compare_eq(left=left, right=right, name="same")
        builder.ref.release(value=left)
        builder.ref.release(value=right)
        builder.func.return_(operands=[missing, same])

    ops = default_ops()
    assert not verify_module(module, ops=ops).has_errors
    restored = read_module(write_module(module, op_decls=ops), op_decls=ops)
    assert not verify_module(restored, ops=ops).has_errors
    printer = Printer()
    printer.register_ops(ops)
    printer.register_types(default_types())
    text = printer.print_module(module)
    assert "ref.null" in text
    assert "ref.compare.null %left" in text
    assert "ref.compare.eq %left, %right" in text
    assert printer.print_module(restored) == text
    assert left.id != right.id
    assert missing.type == same.type == predicate_type


def test_author_and_serialize_borrowed_projection() -> None:
    module, builder = module_builder()
    builder.type.decl(namespace="media", type_name="frame", name="Frame")
    owned = ref_type(module=module, declaration="Frame")
    borrowed = ptr_type(module=module, declaration="Frame")
    owner = builder.value("owner", owned)
    body = builder.region()
    builder.func.def_(callee="project", args=[owner], results=[owned, borrowed], body=body)
    with builder.insertion_block(body.blocks[0]):
        pointer = builder.ref.borrow(owner=owner, results=[borrowed], name="pointer")
        builder.func.return_(operands=[owner, pointer])

    ops = default_ops()
    assert not verify_module(module, ops=ops).has_errors
    restored = read_module(write_module(module, op_decls=ops), op_decls=ops)
    assert not verify_module(restored, ops=ops).has_errors
    printer = Printer()
    printer.register_ops(ops)
    printer.register_types(default_types())
    text = printer.print_module(module)
    assert "ref.borrow %owner : ref<@Frame> -> ptr<@Frame>" in text
    assert printer.print_module(restored) == text
    assert owner.type == owned and pointer.type == borrowed
    assert owner.id != pointer.id


@pytest.mark.parametrize("handle_builder", [ref_type, ptr_type])
def test_author_and_serialize_release(handle_builder) -> None:
    module, builder = module_builder()
    builder.type.decl(namespace="media", type_name="frame", name="Frame")
    handle_type = handle_builder(module=module, declaration="Frame")
    source = builder.value("source", handle_type)
    body = builder.region()
    builder.func.def_(callee="release", args=[source], results=[], body=body)
    with builder.insertion_block(body.blocks[0]):
        builder.ref.release(value=source)
        builder.func.return_()

    ops = default_ops()
    assert not verify_module(module, ops=ops).has_errors
    restored = read_module(write_module(module, op_decls=ops), op_decls=ops)
    assert not verify_module(restored, ops=ops).has_errors
    printer = Printer()
    printer.register_ops(ops)
    printer.register_types(default_types())
    text = printer.print_module(module)
    assert "ref.release %source" in text
    assert printer.print_module(restored) == text
