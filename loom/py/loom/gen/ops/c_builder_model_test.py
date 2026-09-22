# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public builder parameter and presence contracts."""

import pytest

from loom.assembly import ARROW, Attr, AttrDict, OptionalGroup, PredicateList, Ref, ResultType
from loom.dsl import ANY, AttrDef, Dialect, Op, Operand, Result
from loom.gen.ops.c_builder_model import build_flag_params, build_flags_storage_type, detect_builder_pattern, extract_c_params
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
