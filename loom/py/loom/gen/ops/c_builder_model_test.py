# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public builder parameter and presence contracts."""

import pytest

from loom.assembly import (
    ARROW,
    BINDING_TYPE_BLOCK_ARG,
    COLON,
    Attr,
    AttrDict,
    BindingList,
    BlockArgs,
    Flags,
    OptionalGroup,
    PredicateList,
    Ref,
    Region,
    ResultType,
    ResultTypeList,
    TypeOf,
    kw,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_FLAGS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    LoopLikeInterface,
    Op,
    Operand,
    RegionDef,
    Result,
    TiedResult,
    TypeConstraint,
)
from loom.gen.ops.c_builder_model import build_flag_params, build_flags_storage_type, detect_builder_pattern, extract_c_params
from loom.gen.ops.c_builder_source import generate_builders_c
from loom.gen.ops.c_ops_header import generate_ops_h


@pytest.mark.parametrize(("count", "storage_type"), [(0, "uint32_t"), (31, "uint32_t"), (32, "uint32_t"), (33, "uint64_t"), (64, "uint64_t")])
def test_presence_flags_fit_public_storage(count: int, storage_type: str) -> None:
    assert build_flags_storage_type(count) == storage_type


def test_builder_declarations_reject_presence_flags_beyond_capacity() -> None:
    for count in (64, 65):
        op = Op(
            "test.wide",
            group=Dialect("test"),
            attrs=[AttrDef(f"value_{i}", "i64", optional=True) for i in range(count)],
            format=[AttrDict()],
        )
        if count == 64:
            generate_ops_h("test", 0, [op])
        else:
            with pytest.raises(ValueError, match="exceeds the 64-bit build flag capacity"):
                generate_ops_h("test", 0, [op])


def test_projected_block_argument_builders_derive_group_boundary() -> None:
    op = Op(
        "test.partitioned_region",
        group=Dialect("test"),
        attrs=[AttrDef("actual_count", "i64")],
        regions=[RegionDef("body")],
        format=[
            BlockArgs(
                "body",
                group="actual",
                end_attr="actual_count",
            ),
            BlockArgs(
                "body",
                group="expected",
                start_attr="actual_count",
            ),
            Region("body"),
        ],
    )

    parameters = extract_c_params(op, {})
    block_parameters = [parameter for parameter in parameters if parameter["kind"] == "block_args"]
    assert [(parameter["name"], parameter["end_attr_index"]) for parameter in block_parameters] == [
        ("actual_arg_types", 0),
        ("expected_arg_types", 0xFF),
    ]
    source = generate_builders_c("test", [op])
    assert "_i < actual_arg_types_count" in source
    assert "_i < expected_arg_types_count" in source
    assert "actual_arg_types_count + expected_arg_types_count, UINT16_MAX" in source
    assert ("loom_op_attrs(*out_op)[0] = loom_attr_i64((int64_t)(actual_arg_types_count));") in source
    assert "loom_builder_define_value(" in source
    assert "loom_block_add_arg(" in source
    assert "loom_builder_define_block_arg(" not in source


def test_projected_block_argument_builders_omit_empty_optional_boundary() -> None:
    op = Op(
        "test.partitioned_region",
        group=Dialect("test"),
        attrs=[AttrDef("actual_count", "i64", optional=True)],
        regions=[RegionDef("body")],
        format=[
            BlockArgs(
                "body",
                group="actual",
                end_attr="actual_count",
            ),
            BlockArgs(
                "body",
                group="expected",
                start_attr="actual_count",
            ),
            Region("body"),
        ],
    )

    source = generate_builders_c("test", [op])
    assert "if ((actual_arg_types_count) != 0)" in source
    assert ("loom_op_attrs(*out_op)[0] = loom_attr_i64((int64_t)(actual_arg_types_count));") in source


def test_optional_aggregates_have_explicit_presence() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[
            AttrDef("dict", "dict", optional=True),
            AttrDef("values", "i64_array", optional=True),
            AttrDef("payload", "bytes", optional=True),
            AttrDef("predicates", "predicate_list", optional=True),
        ],
        format=[
            AttrDict("dict"),
            OptionalGroup([Attr("values")], anchor="values"),
            OptionalGroup([Attr("payload")], anchor="payload"),
            OptionalGroup([PredicateList("predicates")], anchor="predicates"),
        ],
    )
    assert [param["name"] for param in build_flag_params(extract_c_params(op, {}))] == ["dict", "values", "payload", "predicates"]


def test_optional_aggregates_preserve_established_flag_ordinals() -> None:
    op = Op(
        "test.attrs",
        group=Dialect("test"),
        attrs=[AttrDef("before", "i64", optional=True), AttrDef("dict", "dict", optional=True), AttrDef("after", "string", optional=True)],
        format=[Attr("before"), AttrDict("dict"), Attr("after")],
    )
    assert [param["name"] for param in build_flag_params(extract_c_params(op, {}))] == ["before", "after", "dict"]


def test_compact_builders_require_matching_parameter_names() -> None:
    for names in (("lhs", "rhs"), ("table", "indices")):
        op = Op(
            "test.binary",
            group=Dialect("test"),
            operands=[Operand(name, ANY) for name in names],
            results=[Result("result", ANY)],
            format=[Ref(name) for name in names],
        )
        assert (detect_builder_pattern(op) is not None) == (names == ("lhs", "rhs"))
        assert [param["name"] for param in extract_c_params(op, {}) if param["kind"] == "operand"] == list(names)


def test_compact_builders_require_an_exact_fixed_shape() -> None:
    optional = Op(
        "test.unary",
        group=Dialect("test"),
        operands=[Operand("input", ANY, optional=True)],
        results=[Result("result", ANY)],
        format=[
            OptionalGroup([Ref("input")], anchor="input"),
            ResultType("result"),
        ],
    )
    tied = Op(
        "test.unary",
        group=Dialect("test"),
        operands=[Operand("input", ANY)],
        results=[TiedResult("result", "input", ANY)],
        format=[Ref("input"), ResultType("result")],
    )
    predicate = EnumDef("Mode", [EnumCase("enabled", 0)])
    differently_named_comparison = Op(
        "test.compare",
        group=Dialect("test"),
        operands=[Operand("lhs", ANY), Operand("rhs", ANY)],
        results=[Result("result", ANY)],
        attrs=[AttrDef("mode", "enum", enum_def=predicate)],
        format=[Attr("mode"), Ref("lhs"), Ref("rhs"), ResultType("result")],
    )

    assert detect_builder_pattern(optional) is None
    assert detect_builder_pattern(tied) is None
    assert detect_builder_pattern(differently_named_comparison) is None


def test_flagged_cast_uses_a_flag_aware_compact_builder() -> None:
    flags = EnumDef("CastFlags", [EnumCase("relaxed", 1)])
    op = Op(
        "test.cast",
        group=Dialect("test"),
        operands=[Operand("input", ANY)],
        results=[Result("result", ANY)],
        attrs=[AttrDef("flags", ATTR_TYPE_FLAGS, optional=True, enum_def=flags)],
        format=[
            Flags("flags"),
            Ref("input"),
            COLON,
            TypeOf("input"),
            kw("to"),
            TypeOf("result"),
        ],
    )

    assert detect_builder_pattern(op) == "CAST_WITH_FLAGS"
    header = generate_ops_h("test", 0, [op])
    assert "loom_builder_t* builder, uint8_t instance_flags," in header
    source = generate_builders_c("test", [op])
    assert "iree_status_t loom_test_cast_build(" in source
    assert "(void)input_type;" in source
    assert "return loom_builder_build_unary_op(" in source


def test_fixed_shape_builders_share_implementation_and_preserve_public_names() -> None:
    binary = Op(
        "test.lookup",
        group=Dialect("test"),
        operands=[Operand("table", ANY), Operand("indices", ANY)],
        results=[Result("result", TypeConstraint.I32)],
        format=[Ref("table"), Ref("indices")],
    )
    nullary = Op(
        "test.source",
        group=Dialect("test"),
        results=[Result("result", ANY)],
        format=[ResultType("result")],
    )

    header = generate_ops_h("test", 0, [binary, nullary])
    assert "    loom_value_id_t table," in header
    assert "    loom_value_id_t indices," in header
    source = generate_builders_c("test", [binary, nullary])
    assert "builder_macros.h" not in source
    assert "LOOM_DEFINE_" not in source
    assert ("builder, LOOM_OP_TEST_LOOKUP, 0, table, indices,\n      loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, out_op);") in source
    assert ("builder, LOOM_OP_TEST_SOURCE, 0, result_type, location, out_op);") in source


@pytest.mark.parametrize("order", [("access", "view"), ("view", "access")])
def test_fixed_result_types_follow_fields_not_format_position(order: tuple[str, str]) -> None:
    op = Op(
        "test.selected_record",
        group=Dialect("test"),
        results=[Result("access", ANY), Result("view", ANY)],
        format=[ResultType(order[0]), ARROW, ResultType(order[1])],
    )
    parameters = [param for param in extract_c_params(op, {}) if param["kind"] == "result_type"]
    assert [(param["name"], param["result_index"]) for param in parameters] == [(f"{name}_type", ("access", "view").index(name)) for name in order]
    header = generate_ops_h("test", 0, [op])
    assert "loom_type_t access_type," in header
    assert "loom_type_t view_type," in header
    assert "loom_type_t result_type," not in header


def test_condition_loop_builder_exposes_independent_entry_scheme() -> None:
    op = Op(
        "test.while",
        group=Dialect("test"),
        operands=[Operand("iter_args", ANY, variadic=True)],
        results=[Result("results", ANY, variadic=True)],
        regions=[
            RegionDef("before", single_block=True, terminator="test.condition"),
            RegionDef(
                "after",
                single_block=True,
                terminator="test.yield",
                arg_source="results",
            ),
        ],
        interfaces=[
            LoopLikeInterface(
                body="after",
                condition_region="before",
                iter_args="iter_args",
                results="results",
            )
        ],
        format=[
            BindingList(
                "iter_args",
                type_source=BINDING_TYPE_BLOCK_ARG,
            ),
            Region("before"),
            ResultTypeList("results"),
            Region("after"),
        ],
    )

    header = generate_ops_h("test", 0, [op])
    assert "const loom_type_t* iter_args_types," in header
    source = generate_builders_c("test", [op])
    assert "if (result_count != iter_args_count)" in source
    assert "requires explicit result types when initial and result counts differ" in source
    assert ("result_types ? result_types[_i] : loom_module_value_type(builder->module, iter_args[_i])") in source
    assert "iter_args_types ? iter_args_types[_i] :" in source
    assert "loom_module_value_type(builder->module, iter_args[_i])" in source
    assert "loom_builder_define_value(builder, _arg_type, &_arg_id)" in source
    assert "loom_block_add_arg(builder->module, _block, _arg_id)" in source
    assert '#include "loom/ir/value_replacement.h"' in source
    assert "loom_module_assign_value_types(" in source
