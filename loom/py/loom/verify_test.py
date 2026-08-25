# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import loom.ir as ir
from loom.builtin_types import ALL_BUILTIN_TYPES
from loom.diagnostics import DiagnosticEngine
from loom.dialect.func import ALL_FUNC_OPS
from loom.dialect.template import ALL_TEMPLATE_OPS
from loom.dialect.test import ALL_TEST_OPS
from loom.dsl import (
    ANY,
    INTEGER,
    ISOLATED_FROM_ABOVE,
    HasAncestor,
    InlinePolicy,
    Op,
    Operand,
    RegionDef,
    Result,
    TypeConstraint,
)
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
    SymbolName,
    SymbolNameArray,
    TypeKind,
    Value,
)
from loom.verify import type_satisfies_constraint, verify_module


def test_type_constraints_match_byte_pattern_scalars() -> None:
    accepted_kinds = (
        ir.ScalarTypeKind.I8,
        ir.ScalarTypeKind.I16,
        ir.ScalarTypeKind.I32,
        ir.ScalarTypeKind.I64,
        ir.ScalarTypeKind.F8E4M3,
        ir.ScalarTypeKind.F8E5M2,
        ir.ScalarTypeKind.F16,
        ir.ScalarTypeKind.BF16,
        ir.ScalarTypeKind.F32,
        ir.ScalarTypeKind.F64,
    )
    for kind in accepted_kinds:
        assert type_satisfies_constraint(
            ir.ScalarType(kind), TypeConstraint.BYTE_PATTERN_SCALAR
        )

    rejected_kinds = (
        ir.ScalarTypeKind.INDEX,
        ir.ScalarTypeKind.OFFSET,
        ir.ScalarTypeKind.I1,
    )
    for kind in rejected_kinds:
        assert not type_satisfies_constraint(
            ir.ScalarType(kind), TypeConstraint.BYTE_PATTERN_SCALAR
        )


def test_type_constraints_match_exact_i32() -> None:
    assert type_satisfies_constraint(I32, TypeConstraint.I32)
    assert not type_satisfies_constraint(F32, TypeConstraint.I32)


def test_python_verifier_handles_existing_bitwise_constraints() -> None:
    assert type_satisfies_constraint(INDEX, TypeConstraint.BITWISE_SCALAR)
    assert type_satisfies_constraint(F32, TypeConstraint.BITWISE_SCALAR)


def test_verifier_reports_missing_operand_value() -> None:
    module = _module_with_body_ops(Operation(name="test.use", operands=[7]))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert diagnostics.has_errors
    assert "outside [0, 0)" in str(diagnostics.diagnostics[0])


def test_verifier_reports_duplicate_symbols() -> None:
    module = Module()
    module.add_symbol(_symbol("same", _func()))
    module.add_symbol(_symbol("same", _func()))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert diagnostics.has_errors
    assert "duplicate symbol name" in str(diagnostics.diagnostics[0])


def test_verifier_reports_symbol_operation_missing_from_module_body() -> None:
    module = Module()
    module.symbols.append(_symbol("missing", _func()))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics,
        "symbol defining operation is not owned by the module body",
    )


def test_verifier_reports_module_body_symbol_missing_from_table() -> None:
    module = Module()
    module.body.ops.append(_func())

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics,
        "module body symbol definition is missing from the symbol table",
    )


def test_verifier_accepts_module_scope_operation() -> None:
    module = Module()
    module.add_top_level_operation(Operation(name="test.module_metadata"))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert not diagnostics.has_errors


def test_verifier_rejects_ordinary_operation_at_module_scope() -> None:
    module = Module()
    module.add_top_level_operation(Operation(name="test.yield"))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "op is not permitted at module scope")


def test_verifier_rejects_nested_module_scope_operation() -> None:
    module = _module_with_body_ops(Operation(name="test.module_metadata"))

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "module-scope op is nested")


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


def test_python_verifier_checks_condition_forwarding() -> None:
    from loom.builders import default_ops

    parser = Parser()
    parser.register_ops(default_ops())
    parser.register_types(ALL_BUILTIN_TYPES)
    module = parser.parse(
        """
func.def @f(%condition: i1, %initial: index) -> (index) {
  %result = scf.while(%before = %initial : index) -> (index) {
    scf.condition %condition : i1
  } do(%body: index) {
    scf.yield %body : index
  }
  func.return %result : index
}
"""
    )

    diagnostics = verify_module(module)

    assert _diagnostic_text_contains(
        diagnostics,
        "ConditionForwardedCountMatchesBlockArgs constraint violated",
    )


def test_verifier_defers_template_ancestor_requirement() -> None:
    diagnostics = _verify_required_ancestor_in_template(
        Operation(name="test.requires_context")
    )

    assert not diagnostics.has_errors


def test_verifier_defers_required_inline_ancestor_requirement() -> None:
    diagnostics = _verify_required_ancestor_in_func(
        InlinePolicy.INLINE,
        Operation(name="test.requires_context"),
    )

    assert not diagnostics.has_errors


def test_verifier_does_not_defer_noinline_ancestor_requirement() -> None:
    diagnostics = _verify_required_ancestor_in_func(
        InlinePolicy.NOINLINE,
        Operation(name="test.requires_context"),
    )

    assert _diagnostic_text_contains(diagnostics, "missing required ancestor")


def test_verifier_does_not_defer_inline_through_nested_isolation() -> None:
    isolated_op = Operation(
        name="test.isolated_region",
        regions=[
            Region(
                blocks=[
                    Block(
                        ops=[
                            Operation(name="test.requires_context"),
                            Operation(name="test.yield"),
                        ]
                    )
                ]
            )
        ],
    )
    diagnostics = _verify_required_ancestor_in_func(InlinePolicy.INLINE, isolated_op)

    assert _diagnostic_text_contains(diagnostics, "missing required ancestor")


def test_verifier_does_not_defer_through_nested_isolation() -> None:
    isolated = Op(
        "test.nested_isolated",
        traits=[ISOLATED_FROM_ABOVE],
        regions=[RegionDef("body")],
    )
    isolated_op = Operation(
        name="test.nested_isolated",
        regions=[
            Region(
                blocks=[
                    Block(ops=[Operation(name="test.requires_context")]),
                ]
            )
        ],
    )
    diagnostics = _verify_required_ancestor_in_template(
        isolated_op,
        extra_ops=(isolated,),
    )

    assert _diagnostic_text_contains(diagnostics, "missing required ancestor")


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
    module.add_symbol(_symbol("f", _func()))

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


def test_verifier_checks_each_symbol_array_dependency() -> None:
    module = _module_with_body_ops(
        Operation(
            name="test.symbol_array_attrs",
            attributes={
                "dependencies": SymbolNameArray(
                    [SymbolName("missing"), SymbolName("missing")]
                )
            },
        )
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert (
        sum(
            "unresolved symbol reference" in str(diagnostic)
            for diagnostic in diagnostics.diagnostics
        )
        == 2
    )
    assert _diagnostic_text_contains(
        diagnostics, "attribute 'dependencies' element 0 references @missing"
    )
    assert _diagnostic_text_contains(
        diagnostics, "attribute 'dependencies' element 1 references @missing"
    )


def test_verifier_accepts_unresolved_symbol_array_availability() -> None:
    module = _module_with_body_ops(
        Operation(
            name="test.symbol_array_attrs",
            attributes={
                "dependencies": SymbolNameArray(),
                "available": SymbolNameArray([SymbolName("provider")]),
            },
        )
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert not diagnostics.has_errors


def test_verifier_keeps_dependency_on_available_symbol_strict() -> None:
    module = _module_with_body_ops(
        Operation(
            name="test.symbol_array_attrs",
            attributes={
                "dependencies": SymbolNameArray([SymbolName("provider")]),
                "available": SymbolNameArray([SymbolName("provider")]),
            },
        )
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(diagnostics, "unresolved symbol reference")
    assert _diagnostic_text_contains(
        diagnostics, "attribute 'dependencies' element 0 references @provider"
    )


def test_verifier_accepts_unconstrained_symbol_array_target() -> None:
    module = Module()
    module.add_symbol(_symbol("function", _decl("function")))
    module = _module_with_body_ops(
        Operation(
            name="test.symbol_array_attrs",
            attributes={
                "dependencies": SymbolNameArray(),
                "available": SymbolNameArray([SymbolName("function")]),
            },
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert not diagnostics.has_errors


def test_verifier_checks_symbol_array_target_interface_when_defined() -> None:
    module = Module()
    module.add_symbol(_symbol("function", _decl("function")))
    module = _module_with_body_ops(
        Operation(
            name="test.symbol_array_attrs",
            attributes={
                "dependencies": SymbolNameArray([SymbolName("function")]),
            },
        ),
        module=module,
    )

    diagnostics = verify_module(module, ops=ALL_TEST_OPS)

    assert _diagnostic_text_contains(
        diagnostics, "symbol reference target has wrong interface"
    )
    assert _diagnostic_text_contains(
        diagnostics, "attribute 'dependencies' element 0 references @function"
    )


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
    module.add_symbol(_symbol("f", func))
    return module


def _func(body: Region | None = None) -> Operation:
    return Operation(
        name="test.func",
        attributes={"callee": "f"},
        regions=[body if body is not None else Region(blocks=[Block()])],
    )


def _decl(name: str) -> Operation:
    return Operation(
        name="test.decl",
        attributes={"callee": name},
    )


def _func_def_with_ops(
    name: str, inline_policy: InlinePolicy, *ops: Operation
) -> Operation:
    return Operation(
        name="func.def",
        attributes={
            "callee": name,
            "inline_policy": inline_policy.value,
        },
        regions=[
            Region(
                blocks=[
                    Block(ops=[*ops, Operation(name="func.return")]),
                ]
            )
        ],
    )


def _verify_required_ancestor_in_func(
    inline_policy: InlinePolicy, *ops: Operation
) -> DiagnosticEngine:
    requires_context = Op(
        "test.requires_context",
        traits=[HasAncestor("test.context")],
    )
    module = Module()
    module.add_symbol(
        _symbol(
            "helper",
            _func_def_with_ops("helper", inline_policy, *ops),
        )
    )
    return verify_module(
        module,
        ops=(*ALL_TEST_OPS, *ALL_FUNC_OPS, requires_context),
    )


def _verify_required_ancestor_in_template(
    *ops: Operation,
    extra_ops: tuple[Op, ...] = (),
) -> DiagnosticEngine:
    requires_context = Op(
        "test.requires_context",
        traits=[HasAncestor("test.context")],
    )
    module = Module()
    family = Operation(
        name="template.decl",
        attributes={"family": "family"},
    )
    module.add_symbol(_symbol("family", family))
    provider = Operation(
        name="template.def",
        attributes={"family": "family", "implementation": "provider"},
        regions=[
            Region(
                blocks=[
                    Block(ops=[*ops, Operation(name="template.return")]),
                ]
            )
        ],
    )
    module.add_symbol(_symbol("provider", provider))
    return verify_module(
        module,
        ops=(
            *ALL_TEST_OPS,
            *ALL_TEMPLATE_OPS,
            requires_context,
            *extra_ops,
        ),
    )


def _symbol(name: str, op: Operation) -> Symbol:
    return Symbol(name=name, op=op)


def _diagnostic_text_contains(diagnostics: DiagnosticEngine, needle: str) -> bool:
    diagnostic_text = "\n".join(
        str(diagnostic) for diagnostic in diagnostics.diagnostics
    )
    return needle in diagnostic_text
