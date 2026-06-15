# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.diagnostics import DiagnosticEngine
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import ANY, INTEGER, Op, Operand, Result
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.ir import (
    ENCODING_LAYOUT_TYPE,
    F32,
    I32,
    INDEX,
    Block,
    DynamicDim,
    DynamicEncoding,
    Module,
    Operation,
    Region,
    ShapedType,
    StaticDim,
    Symbol,
    TypeKind,
    Value,
)
from loom.verify import verify_module


def test_verifier_reports_missing_operand_value() -> None:
    module = _module_with_body_ops(Operation(name="test.use", operands=[7]))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert diagnostics.has_errors
    assert "outside [0, 0)" in str(diagnostics.diagnostics[0])


def test_verifier_reports_duplicate_symbols() -> None:
    module = Module()
    module.symbols.append(_symbol("same", _func()))
    module.symbols.append(_symbol("same", _func()))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert diagnostics.has_errors
    assert "duplicate symbol name" in str(diagnostics.diagnostics[0])


def test_verifier_reports_wrong_operand_count() -> None:
    module = Module()
    lhs = module.add_value(Value("lhs", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(name="test.addi", operands=[lhs], results=[result]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "wrong operand count")


def test_verifier_reports_type_constraint_failure() -> None:
    module = Module()
    lhs = module.add_value(Value("lhs", F32))
    rhs = module.add_value(Value("rhs", F32))
    result = module.add_value(Value("result", F32))
    module = _module_with_body_ops(
        Operation(name="test.addi", operands=[lhs, rhs], results=[result]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "operand type constraint violated")
    assert _diagnostic_text_contains(diagnostics, "expected integer")


def test_verifier_accepts_segmented_operand_counts() -> None:
    module = Module()
    root = module.add_value(Value("root", I32))
    lhs = module.add_value(Value("lhs", I32))
    rhs = module.add_value(Value("rhs", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.segmented",
            operands=[root, lhs, rhs],
            operand_segment_counts=(1, 0, 1, 1),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert not diagnostics.has_errors


def test_verifier_reports_missing_segmented_required_operand() -> None:
    module = Module()
    lhs = module.add_value(Value("lhs", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.segmented",
            operands=[lhs],
            operand_segment_counts=(0, 0, 1, 0),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "required operand segment count")
    assert _diagnostic_text_contains(diagnostics, "root")


def test_verifier_reports_segmented_count_sum_mismatch() -> None:
    module = Module()
    root = module.add_value(Value("root", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.segmented",
            operands=[root],
            operand_segment_counts=(1, 0, 1, 0),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "do not sum to operand count")


def test_verifier_reports_segmented_optional_operand_too_many_values() -> None:
    module = Module()
    root = module.add_value(Value("root", I32))
    guard0 = module.add_value(Value("guard0", I32))
    guard1 = module.add_value(Value("guard1", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.segmented",
            operands=[root, guard0, guard1],
            operand_segment_counts=(1, 2, 0, 0),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "optional operand segment count")
    assert _diagnostic_text_contains(diagnostics, "guard")


def test_verifier_reports_unexpected_segment_counts_on_non_segmented_op() -> None:
    module = Module()
    lhs = module.add_value(Value("lhs", I32))
    rhs = module.add_value(Value("rhs", I32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.addi",
            operands=[lhs, rhs],
            operand_segment_counts=(1, 1),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "unexpected operand segment counts")


def test_verifier_type_checks_segmented_operand_spans() -> None:
    segmented_integer = Op(
        "test.segmented_integer",
        operands=[
            Operand("root", INTEGER),
            Operand("guard", INTEGER, optional=True),
            Operand("lhs", INTEGER, variadic=True),
            Operand("rhs", INTEGER, variadic=True),
        ],
        results=[Result("result", ANY)],
    )
    module = Module()
    root = module.add_value(Value("root", I32))
    rhs = module.add_value(Value("rhs", F32))
    result = module.add_value(Value("result", I32))
    module = _module_with_body_ops(
        Operation(
            name="test.segmented_integer",
            operands=[root, rhs],
            operand_segment_counts=(1, 0, 0, 1),
            results=[result],
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=(*ALL_TEST_OPS, segmented_integer))

    assert _diagnostic_text_contains(diagnostics, "operand type constraint violated")
    assert _diagnostic_text_contains(diagnostics, "rhs[0]")
    assert _diagnostic_text_contains(diagnostics, "expected integer")


def test_verifier_runs_declarative_constraints() -> None:
    module = Module()
    input_value = module.add_value(Value("input", I32))
    result = module.add_value(Value("result", F32))
    module = _module_with_body_ops(
        Operation(name="test.convergent", operands=[input_value], results=[result]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "SameType constraint violated")


def test_verifier_reports_missing_region_terminator() -> None:
    module = Module()
    value = module.add_value(Value("value", I32))
    module = _module_with_body_ops(
        Operation(name="test.use", operands=[value]),
        module=module,
        append_yield=False,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics, "block is missing required terminator"
    )


def test_verifier_reports_empty_block_missing_region_terminator() -> None:
    module = Module()
    module.symbols.append(_symbol("f", _func()))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics, "block is missing required terminator"
    )


def test_verifier_reports_unresolved_symbol_ref() -> None:
    module = _module_with_body_ops(
        Operation(name="test.invoke", attributes={"callee": "missing"})
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "unresolved symbol reference")


def test_verifier_reports_missing_dynamic_dim_binding() -> None:
    module = Module()
    value_type = ShapedType(TypeKind.VIEW, F32, (DynamicDim(),))
    value = module.add_value(Value("view", value_type))
    module = _module_with_body_ops(
        Operation(name="test.use", operands=[value]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics, "dynamic dimension has no SSA binding"
    )


def test_verifier_reports_unexpected_dynamic_dim_binding() -> None:
    module = Module()
    size = module.add_value(Value("size", INDEX))
    value_type = ShapedType(TypeKind.VIEW, F32, (StaticDim(4),))
    value = module.add_value(Value("view", value_type, dim_bindings={0: size}))
    module = _module_with_body_ops(
        Operation(name="test.use", operands=[value]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics, "static dimension has unexpected SSA binding"
    )


def test_verifier_reports_missing_dynamic_encoding_binding() -> None:
    module = Module()
    value_type = ShapedType(
        TypeKind.VIEW,
        F32,
        (StaticDim(4),),
        encoding=DynamicEncoding(),
    )
    value = module.add_value(Value("view", value_type))
    module = _module_with_body_ops(
        Operation(name="test.use", operands=[value]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "dynamic encoding has no SSA binding")


def test_verifier_accepts_dynamic_encoding_binding() -> None:
    module = Module()
    layout = module.add_value(Value("layout", ENCODING_LAYOUT_TYPE))
    value_type = ShapedType(
        TypeKind.VIEW,
        F32,
        (StaticDim(4),),
        encoding=DynamicEncoding(),
    )
    value = module.add_value(Value("view", value_type, encoding_binding=layout))
    module = _module_with_body_ops(
        Operation(name="test.use", operands=[value]),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert not diagnostics.has_errors


def test_text_parser_can_verify_parsed_module() -> None:
    parser = _test_parser()

    module = parser.parse("test.func @f() {\n  test.yield\n}\n", verify=True)

    assert module.symbols[0].name == "f"


def test_bytecode_reader_can_verify_read_module() -> None:
    parser = _test_parser()
    module = parser.parse("test.func @f() {\n  test.yield\n}\n")
    data = write_module(module, op_decls=ALL_TEST_OPS)

    loaded = read_module(data, op_decls=ALL_TEST_OPS, verify=True)

    assert loaded.symbols[0].name == "f"


def _test_parser() -> Parser:
    parser = Parser()
    parser.register_ops(ALL_TEST_OPS)
    parser.register_types(ALL_BUILTIN_TYPES)
    return parser


def _module_with_body_ops(
    *ops: Operation,
    module: Module | None = None,
    append_yield: bool = True,
) -> Module:
    module = module if module is not None else Module()
    block_ops = list(ops)
    if append_yield:
        block_ops.append(Operation(name="test.yield"))
    func = _func(Region(blocks=[Block(ops=block_ops)]))
    module.symbols.append(_symbol("f", func))
    return module


def _func(body: Region | None = None) -> Operation:
    return Operation(
        name="test.func",
        attributes={"callee": "f"},
        regions=[body if body is not None else Region(blocks=[Block()])],
    )


def _symbol(name: str, op: Operation) -> Symbol:
    return Symbol(name=name, op=op)


def _diagnostic_text_contains(diagnostics: DiagnosticEngine, needle: str) -> bool:
    diagnostic_text = "\n".join(
        str(diagnostic) for diagnostic in diagnostics.diagnostics
    )
    return needle in diagnostic_text
