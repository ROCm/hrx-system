# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom.fields -- field layout and resolution."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.dsl import (
    ANY,
    INTEGER,
    TENSOR,
    TERMINATOR,
    TILE,
    AttrDef,
    Dialect,
    Op,
    Operand,
    RegionDef,
    Result,
    Successor,
    binary_op,
    unary_op,
)
from loom.dsl import (
    INDEX as INDEX_CONSTRAINT,
)
from loom.fields import FieldKind, FieldLayout, compute_layout, resolve_fields
from loom.ir import (
    F32,
    I32,
    INDEX,
    Block,
    Module,
    Operation,
    Region,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
    Value,
)
from loom.ir import (
    TiedResult as IRTiedResult,
)

# ============================================================================
# Test helpers
# ============================================================================


def _make_module_with_values(
    *names_and_types: tuple[str, Type],
) -> tuple[Module, list[int]]:
    """Create a module with named values, returning (module, value_ids)."""
    module = Module(name="test")
    value_ids = []
    for name, value_type in names_and_types:
        vid = module.add_value(Value(name=name, type=value_type))
        value_ids.append(vid)
    return module, value_ids


_tile_4xf32 = ShapedType(TypeKind.TILE, F32, (StaticDim(4),))
_tensor_4xf32 = ShapedType(TypeKind.TENSOR, F32, (StaticDim(4),))


@contextmanager
def _raises(error_type: type[Exception], pattern: str) -> Iterator[None]:
    try:
        yield
    except error_type as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!s} did not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected {error_type.__name__} matching {pattern!r}")


# ============================================================================
# Layout computation
# ============================================================================


class TestComputeLayout:
    def test_binary_op(self) -> None:
        op = binary_op(
            "test.add",
            group=Dialect("test"),
            type_constraint=INTEGER,
            doc="Add.",
        )
        layout = compute_layout(op)
        assert "lhs" in layout.fields
        assert "rhs" in layout.fields
        assert "result" in layout.fields
        assert layout.fields["lhs"].kind == FieldKind.OPERAND
        assert layout.fields["lhs"].index == 0
        assert layout.fields["rhs"].index == 1
        assert layout.fields["result"].kind == FieldKind.RESULT
        assert layout.fields["result"].index == 0
        assert layout.fixed_operand_count == 2
        assert layout.fixed_result_count == 1
        assert layout.variadic_operand is None
        assert layout.variadic_result is None

    def test_trailing_variadic_operand(self) -> None:
        op = Op(
            "test.variadic",
            operands=[
                Operand("source", TILE),
                Operand("offsets", INDEX_CONSTRAINT, variadic=True),
            ],
            results=[Result("result", TILE)],
        )
        layout = compute_layout(op)
        assert layout.fields["source"].index == 0
        assert not layout.fields["source"].variadic
        assert layout.fields["offsets"].index == 1
        assert layout.fields["offsets"].variadic
        assert layout.fixed_operand_count == 1
        assert layout.variadic_operand == "offsets"

    def test_trailing_optional_operand(self) -> None:
        op = Op(
            "test.optional_operand",
            operands=[
                Operand("source", TILE),
                Operand("extent", INDEX_CONSTRAINT, optional=True),
            ],
            results=[Result("result", TILE)],
        )
        layout = compute_layout(op)
        assert layout.fields["source"].index == 0
        assert not layout.fields["source"].optional
        assert layout.fields["extent"].index == 1
        assert layout.fields["extent"].optional
        assert layout.fixed_operand_count == 1
        assert layout.variadic_operand is None

    def test_variadic_result(self) -> None:
        op = Op(
            "test.multi",
            results=[Result("results", ANY, variadic=True)],
        )
        layout = compute_layout(op)
        assert layout.fields["results"].variadic
        assert layout.variadic_result == "results"
        assert layout.fixed_result_count == 0

    def test_attrs_and_regions(self) -> None:
        op = Op(
            "test.complex",
            attrs=[AttrDef("axis", "i64"), AttrDef("label", "string")],
            regions=[RegionDef("body"), RegionDef("else_body")],
        )
        layout = compute_layout(op)
        assert layout.fields["axis"].kind == FieldKind.ATTR
        assert layout.fields["axis"].index == 0
        assert layout.fields["label"].index == 1
        assert layout.fields["body"].kind == FieldKind.REGION
        assert layout.fields["body"].index == 0
        assert layout.fields["else_body"].index == 1

    def test_successors(self) -> None:
        op = Op(
            "test.br",
            successors=[Successor("dest"), Successor("fallback")],
        )
        layout = compute_layout(op)
        assert layout.fields["dest"].kind == FieldKind.SUCCESSOR
        assert layout.fields["dest"].index == 0
        assert layout.fields["fallback"].index == 1
        assert layout.fixed_successor_count == 2
        assert layout.variadic_successor is None

    def test_trailing_variadic_successor(self) -> None:
        op = Op(
            "test.switch",
            successors=[Successor("default"), Successor("cases", variadic=True)],
        )
        layout = compute_layout(op)
        assert layout.fields["default"].index == 0
        assert not layout.fields["default"].variadic
        assert layout.fields["cases"].index == 1
        assert layout.fields["cases"].variadic
        assert layout.fixed_successor_count == 1
        assert layout.variadic_successor == "cases"

    def test_non_trailing_variadic_uses_segmented_operands(self) -> None:
        layout = compute_layout(
            Op(
                "test.segmented",
                operands=[
                    Operand("items", ANY, variadic=True),
                    Operand("extra", ANY),
                ],
            )
        )
        assert layout.segmented_operands
        assert layout.fields["items"].index == 0
        assert layout.fields["items"].variadic
        assert layout.fields["extra"].index == 1

    def test_multiple_variadic_operands_use_segmented_operands(self) -> None:
        layout = compute_layout(
            Op(
                "test.segmented",
                operands=[
                    Operand("a", ANY, variadic=True),
                    Operand("b", ANY, variadic=True),
                ],
            )
        )
        assert layout.segmented_operands
        assert layout.fields["a"].index == 0
        assert layout.fields["a"].variadic
        assert layout.fields["b"].index == 1
        assert layout.fields["b"].variadic

    def test_required_operand_after_optional_uses_segmented_operands(self) -> None:
        layout = compute_layout(
            Op(
                "test.segmented",
                operands=[
                    Operand("optional_extent", INDEX_CONSTRAINT, optional=True),
                    Operand("source", TILE),
                ],
            )
        )
        assert layout.segmented_operands
        assert layout.fields["optional_extent"].index == 0
        assert layout.fields["optional_extent"].optional
        assert layout.fields["source"].index == 1

    def test_optional_variadic_operand_rejected(self) -> None:
        with _raises(ValueError, "cannot be both optional and variadic"):
            compute_layout(
                Op(
                    "test.bad",
                    operands=[
                        Operand(
                            "values",
                            ANY,
                            optional=True,
                            variadic=True,
                        )
                    ],
                )
            )

    def test_variadic_operand_after_optional_uses_segmented_operands(self) -> None:
        layout = compute_layout(
            Op(
                "test.segmented",
                operands=[
                    Operand("optional_extent", INDEX_CONSTRAINT, optional=True),
                    Operand("values", ANY, variadic=True),
                ],
            )
        )
        assert layout.segmented_operands
        assert layout.fields["optional_extent"].index == 0
        assert layout.fields["optional_extent"].optional
        assert layout.fields["values"].index == 1
        assert layout.fields["values"].variadic

    def test_optional_region_cannot_precede_required_region(self) -> None:
        with _raises(ValueError, "cannot follow an optional region"):
            compute_layout(
                Op(
                    "test.bad",
                    regions=[
                        RegionDef("optional_region", optional=True),
                        RegionDef("required_region"),
                    ],
                )
            )

    def test_optional_variadic_region_rejected(self) -> None:
        with _raises(ValueError, "cannot be both optional and variadic"):
            compute_layout(
                Op(
                    "test.bad",
                    regions=[RegionDef("regions", optional=True, variadic=True)],
                )
            )

    def test_single_variadic_only_operand(self) -> None:
        """A single variadic operand (index 0) is valid."""
        op = Op(
            "test.yield",
            operands=[Operand("values", ANY, variadic=True)],
            traits=[TERMINATOR],
        )
        layout = compute_layout(op)
        assert layout.fields["values"].index == 0
        assert layout.fields["values"].variadic
        assert layout.fixed_operand_count == 0

    def test_all_test_ops_compute(self) -> None:
        """Every test dialect op computes a valid layout."""
        from loom.dialect.test import ALL_TEST_OPS

        for op in ALL_TEST_OPS:
            layout = compute_layout(op)
            assert isinstance(layout, FieldLayout), f"{op.name} failed"


# ============================================================================
# Field resolution — singular fields
# ============================================================================


class TestResolveSingular:
    def test_ref_operand(self) -> None:
        module, [lhs_id, rhs_id, result_id] = _make_module_with_values(
            ("lhs", I32),
            ("rhs", I32),
            ("result", I32),
        )
        op = Operation(
            kind=1,
            name="test.add",
            operands=[lhs_id, rhs_id],
            results=[result_id],
        )
        layout = compute_layout(
            binary_op(
                "test.add",
                group=Dialect("test"),
                type_constraint=INTEGER,
                doc="Add.",
            )
        )
        fields = resolve_fields(layout, op, module)

        assert fields.value_id("lhs") == lhs_id
        assert fields.value_id("rhs") == rhs_id
        assert fields.value_name("lhs") == "lhs"
        assert fields.value_name("rhs") == "rhs"

    def test_segmented_operand_fields(self) -> None:
        module, [root_id, guard_id, lhs0_id, lhs1_id, rhs_id, result_id] = (
            _make_module_with_values(
                ("root", I32),
                ("guard", I32),
                ("lhs0", I32),
                ("lhs1", I32),
                ("rhs", I32),
                ("result", I32),
            )
        )
        decl = Op(
            "test.segmented",
            operands=[
                Operand("root", ANY),
                Operand("guard", ANY, optional=True),
                Operand("lhs", ANY, variadic=True),
                Operand("rhs", ANY, variadic=True),
            ],
            results=[Result("result", ANY)],
        )
        op = Operation(
            kind=1,
            name="test.segmented",
            operands=[root_id, guard_id, lhs0_id, lhs1_id, rhs_id],
            operand_segment_counts=(1, 1, 2, 1),
            results=[result_id],
        )
        fields = resolve_fields(compute_layout(decl), op, module)

        assert fields.value_id("root") == root_id
        assert fields.value_id("guard") == guard_id
        assert fields.value_ids("lhs") == [lhs0_id, lhs1_id]
        assert fields.value_ids("rhs") == [rhs_id]
        assert fields.is_present("guard")
        assert fields.types_of("lhs") == [I32, I32]

    def test_absent_segmented_optional_operand(self) -> None:
        module, [root_id, lhs_id, result_id] = _make_module_with_values(
            ("root", I32),
            ("lhs", I32),
            ("result", I32),
        )
        decl = Op(
            "test.segmented",
            operands=[
                Operand("root", ANY),
                Operand("guard", ANY, optional=True),
                Operand("lhs", ANY, variadic=True),
                Operand("rhs", ANY, variadic=True),
            ],
            results=[Result("result", ANY)],
        )
        op = Operation(
            kind=1,
            name="test.segmented",
            operands=[root_id, lhs_id],
            operand_segment_counts=(1, 0, 1, 0),
            results=[result_id],
        )
        fields = resolve_fields(compute_layout(decl), op, module)

        assert not fields.is_present("guard")
        assert fields.value_ids("guard") == []
        assert fields.value_ids("lhs") == [lhs_id]
        assert fields.value_ids("rhs") == []

    def test_type_of(self) -> None:
        module, [vid] = _make_module_with_values(("x", I32))
        decl = unary_op(
            "test.neg",
            group=Dialect("test"),
            type_constraint=INTEGER,
            doc="Neg.",
        )
        op = Operation(kind=1, name="test.neg", operands=[vid], results=[vid])
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.type_of("input") == I32

    def test_attr(self) -> None:
        module = Module(name="test")
        decl = Op(
            "test.constant",
            attrs=[AttrDef("value", "any")],
            results=[Result("result", ANY)],
        )
        result_id = module.add_value(Value(name="c", type=I32))
        op = Operation(
            kind=1,
            name="test.constant",
            results=[result_id],
            attributes={"value": 42},
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.attr("value") == 42

    def test_region(self) -> None:
        module = Module(name="test")
        block = Block(ops=[Operation(kind=0, name="test.yield")])
        region = Region(blocks=[block])
        decl = Op(
            "test.branch",
            operands=[Operand("condition", INTEGER)],
            regions=[RegionDef("then_region"), RegionDef("else_region")],
        )
        cond_id = module.add_value(Value(name="cond", type=I32))
        op = Operation(
            kind=1,
            name="test.branch",
            operands=[cond_id],
            regions=[region, Region(blocks=[])],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        resolved_region = fields.region("then_region")
        assert resolved_region is not None
        assert len(resolved_region.blocks) == 1

    def test_optional_operand(self) -> None:
        module, [source_id, extent_id] = _make_module_with_values(
            ("source", _tile_4xf32),
            ("extent", INDEX),
        )
        decl = Op(
            "test.optional_operand",
            operands=[
                Operand("source", TILE),
                Operand("extent", INDEX_CONSTRAINT, optional=True),
            ],
            results=[Result("result", TILE)],
        )

        without_extent = Operation(
            kind=1,
            name="test.optional_operand",
            operands=[source_id],
            results=[source_id],
        )
        fields = resolve_fields(compute_layout(decl), without_extent, module)
        assert fields.value_id("source") == source_id
        assert fields.value_ids("extent") == []
        assert not fields.is_present("extent")

        with_extent = Operation(
            kind=1,
            name="test.optional_operand",
            operands=[source_id, extent_id],
            results=[source_id],
        )
        fields = resolve_fields(compute_layout(decl), with_extent, module)
        assert fields.value_id("extent") == extent_id
        assert fields.value_ids("extent") == [extent_id]
        assert fields.is_present("extent")

    def test_successor(self) -> None:
        module = Module(name="test")
        target = Block(label="exit")
        decl = Op("test.br", successors=[Successor("dest")])
        op = Operation(kind=1, name="test.br", successors=[target])
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.successor("dest") is target
        assert fields.successors("dest") == [target]

    def test_variadic_region(self) -> None:
        module = Module(name="test")
        case0 = Region(blocks=[Block(ops=[Operation(kind=0, name="test.yield")])])
        case1 = Region(blocks=[Block(ops=[Operation(kind=0, name="test.yield")])])
        default = Region(blocks=[Block(ops=[Operation(kind=0, name="test.yield")])])
        decl = Op(
            "test.region_table",
            operands=[Operand("selector", INTEGER)],
            regions=[
                RegionDef("default_region"),
                RegionDef("case_regions", variadic=True),
            ],
        )
        cond_id = module.add_value(Value(name="cond", type=I32))
        op = Operation(
            kind=1,
            name="test.region_table",
            operands=[cond_id],
            regions=[default, case0, case1],
        )
        layout = compute_layout(decl)
        assert layout.fixed_region_count == 1
        assert layout.variadic_region == "case_regions"
        fields = resolve_fields(layout, op, module)
        assert fields.region("default_region") is default
        assert fields.regions("case_regions") == [case0, case1]

    def test_optional_region(self) -> None:
        module = Module(name="test")
        body = Region(blocks=[Block(ops=[Operation(kind=0, name="test.yield")])])
        optional = Region(blocks=[Block(ops=[Operation(kind=0, name="test.yield")])])
        decl = Op(
            "test.optional_region",
            regions=[
                RegionDef("body"),
                RegionDef("else_region", optional=True),
            ],
        )
        layout = compute_layout(decl)
        assert layout.required_region_count == 1
        assert layout.fixed_region_count == 2
        assert layout.fields["else_region"].optional

        op_without_else = Operation(
            kind=1,
            name="test.optional_region",
            regions=[body],
        )
        fields = resolve_fields(layout, op_without_else, module)
        assert fields.region("body") is body
        assert fields.region("else_region") is None
        assert fields.regions("else_region") == []
        assert not fields.is_present("else_region")

        op_with_else = Operation(
            kind=1,
            name="test.optional_region",
            regions=[body, optional],
        )
        fields = resolve_fields(layout, op_with_else, module)
        assert fields.region("else_region") is optional
        assert fields.regions("else_region") == [optional]
        assert fields.is_present("else_region")

    def test_unknown_field_raises(self) -> None:
        module, [vid] = _make_module_with_values(("x", I32))
        decl = unary_op(
            "test.neg",
            group=Dialect("test"),
            type_constraint=INTEGER,
            doc="Neg.",
        )
        op = Operation(kind=1, name="test.neg", operands=[vid], results=[vid])
        fields = resolve_fields(compute_layout(decl), op, module)
        with _raises(KeyError, "Unknown field 'nonexistent'"):
            fields.value_id("nonexistent")

    def test_wrong_kind_raises(self) -> None:
        module = Module(name="test")
        decl = Op("test.attrs", attrs=[AttrDef("axis", "i64")])
        op = Operation(kind=1, name="test.attrs", attributes={"axis": 0})
        fields = resolve_fields(compute_layout(decl), op, module)
        with _raises(TypeError, "not an operand"):
            fields.value_id("axis")


# ============================================================================
# Field resolution — variadic fields
# ============================================================================


class TestResolveVariadic:
    def test_variadic_operand(self) -> None:
        module, vids = _make_module_with_values(
            ("src", _tile_4xf32),
            ("off0", INDEX),
            ("off1", INDEX),
            ("off2", INDEX),
        )
        decl = Op(
            "test.slice",
            operands=[
                Operand("source", TILE),
                Operand("offsets", INDEX_CONSTRAINT, variadic=True),
            ],
            results=[Result("result", TILE)],
        )
        op = Operation(
            kind=1,
            name="test.slice",
            operands=vids,
            results=[vids[0]],
        )
        layout = compute_layout(decl)
        fields = resolve_fields(layout, op, module)

        assert fields.value_id("source") == vids[0]
        offset_ids = fields.value_ids("offsets")
        assert offset_ids == vids[1:]
        assert len(fields.values("offsets")) == 3

    def test_variadic_result(self) -> None:
        module, vids = _make_module_with_values(
            ("a", I32),
            ("b", F32),
            ("c", INDEX),
        )
        decl = Op(
            "test.multi",
            results=[Result("results", ANY, variadic=True)],
        )
        op = Operation(kind=1, name="test.multi", results=vids)
        fields = resolve_fields(compute_layout(decl), op, module)

        result_ids = fields.value_ids("results")
        assert result_ids == vids
        types = fields.types_of("results")
        assert types == [I32, F32, INDEX]

    def test_variadic_successor(self) -> None:
        module = Module(name="test")
        default = Block(label="default")
        case0 = Block(label="case0")
        case1 = Block(label="case1")
        decl = Op(
            "test.switch",
            successors=[Successor("default"), Successor("cases", variadic=True)],
        )
        op = Operation(
            kind=1,
            name="test.switch",
            successors=[default, case0, case1],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.successor("default") is default
        assert fields.successors("cases") == [case0, case1]

    def test_empty_variadic(self) -> None:
        module, [src_id] = _make_module_with_values(("src", _tile_4xf32))
        decl = Op(
            "test.slice",
            operands=[
                Operand("source", TILE),
                Operand("offsets", INDEX_CONSTRAINT, variadic=True),
            ],
            results=[Result("result", TILE)],
        )
        op = Operation(
            kind=1,
            name="test.slice",
            operands=[src_id],
            results=[src_id],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.value_ids("offsets") == []

    def test_variadic_names(self) -> None:
        module, vids = _make_module_with_values(
            ("a", I32),
            ("b", I32),
        )
        decl = Op(
            "test.yield",
            operands=[Operand("values", ANY, variadic=True)],
            traits=[TERMINATOR],
        )
        op = Operation(kind=1, name="test.yield", operands=vids)
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.value_names("values") == ["a", "b"]


# ============================================================================
# Tied results
# ============================================================================


class TestTiedResults:
    def test_tied_result_map(self) -> None:
        module, vids = _make_module_with_values(
            ("tile", _tile_4xf32),
            ("tensor", _tensor_4xf32),
            ("result", _tensor_4xf32),
        )
        decl = Op(
            "test.update",
            operands=[Operand("source", TILE), Operand("target", TENSOR)],
            results=[Result("result", TENSOR)],
        )
        op = Operation(
            kind=1,
            name="test.update",
            operands=[vids[0], vids[1]],
            results=[vids[2]],
            tied_results=[IRTiedResult(result_index=0, operand_index=1)],
        )
        fields = resolve_fields(compute_layout(decl), op, module)

        tied_map = fields.tied_result_map()
        assert 0 in tied_map
        assert tied_map[0].operand_index == 1
        assert fields.operand_name_for_tied(tied_map[0]) == "%tensor"

    def test_no_tied_results(self) -> None:
        module, vids = _make_module_with_values(
            ("a", I32),
            ("b", I32),
            ("r", I32),
        )
        decl = binary_op(
            "test.add",
            group=Dialect("test"),
            type_constraint=INTEGER,
            doc="Add.",
        )
        op = Operation(
            kind=1,
            name="test.add",
            operands=[vids[0], vids[1]],
            results=[vids[2]],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.tied_result_map() == {}


# ============================================================================
# Presence checks
# ============================================================================


class TestPresence:
    def test_present_attr(self) -> None:
        module = Module(name="test")
        decl = Op("test.op", attrs=[AttrDef("visibility", "string", optional=True)])
        op = Operation(
            kind=1,
            name="test.op",
            attributes={"visibility": "public"},
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.is_present("visibility")

    def test_absent_attr(self) -> None:
        module = Module(name="test")
        decl = Op("test.op", attrs=[AttrDef("visibility", "string", optional=True)])
        op = Operation(kind=1, name="test.op", attributes={})
        fields = resolve_fields(compute_layout(decl), op, module)
        assert not fields.is_present("visibility")

    def test_present_variadic(self) -> None:
        module, vids = _make_module_with_values(("a", I32), ("b", I32))
        decl = Op(
            "test.yield",
            operands=[Operand("values", ANY, variadic=True)],
        )
        op = Operation(kind=1, name="test.yield", operands=vids)
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.is_present("values")

    def test_absent_variadic(self) -> None:
        module = Module(name="test")
        decl = Op(
            "test.yield",
            operands=[Operand("values", ANY, variadic=True)],
        )
        op = Operation(kind=1, name="test.yield", operands=[])
        fields = resolve_fields(compute_layout(decl), op, module)
        assert not fields.is_present("values")

    def test_present_successor(self) -> None:
        module = Module(name="test")
        decl = Op("test.br", successors=[Successor("dest")])
        op = Operation(kind=1, name="test.br", successors=[Block(label="exit")])
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.is_present("dest")

    def test_absent_successor(self) -> None:
        module = Module(name="test")
        decl = Op("test.br", successors=[Successor("dest")])
        op = Operation(kind=1, name="test.br", successors=[])
        fields = resolve_fields(compute_layout(decl), op, module)
        assert not fields.is_present("dest")

    def test_present_region(self) -> None:
        module = Module(name="test")
        decl = Op("test.op", regions=[RegionDef("body")])
        block = Block(ops=[Operation(kind=0, name="test.yield")])
        op = Operation(
            kind=1,
            name="test.op",
            regions=[Region(blocks=[block])],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert fields.is_present("body")

    def test_absent_region(self) -> None:
        module = Module(name="test")
        decl = Op("test.op", regions=[RegionDef("body")])
        op = Operation(
            kind=1,
            name="test.op",
            regions=[Region(blocks=[])],
        )
        fields = resolve_fields(compute_layout(decl), op, module)
        assert not fields.is_present("body")
