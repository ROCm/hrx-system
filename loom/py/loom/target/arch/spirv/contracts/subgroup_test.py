# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.kernel import defs as kernel
from loom.target.arch.spirv.contracts.subgroup import SPIRV_SUBGROUP_CONTRACT_CASES
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.features import feature_bits_value
from loom.target.contracts import (
    DescriptorResultType,
    GuardKind,
    RecipeRule,
    SourceValueKind,
    ValueRef,
)


def _ballot_rules():
    return tuple(
        rule
        for rule in SPIRV_SUBGROUP_CONTRACT_CASES
        if rule.source_op is kernel.kernel_subgroup_vote_ballot
    )


def test_ballot_contracts_cover_both_public_mask_widths() -> None:
    assert len(_ballot_rules()) == 2
    by_mask_width = {
        next(
            guard.maximum
            for guard in rule.guards
            if guard.kind is GuardKind.TARGET_SUBGROUP_SIZE_RANGE
        ): rule
        for rule in _ballot_rules()
    }
    assert set(by_mask_width) == {32, 64}
    for mask_bit_count, rule in by_mask_width.items():
        size_guard = next(
            guard
            for guard in rule.guards
            if guard.kind is GuardKind.TARGET_SUBGROUP_SIZE_RANGE
        )
        assert size_guard.minimum == 1
        assert size_guard.maximum == mask_bit_count
        assert any(
            guard.kind is GuardKind.DESCRIPTOR_AVAILABLE for guard in rule.guards
        )


def test_ballot_contract_keeps_native_packing_explicit() -> None:
    i32_rule, i64_rule = _ballot_rules()
    assert tuple(emit.descriptor.key for emit in i32_rule.emit) == (
        "spirv.op_group_non_uniform_ballot.v4u32",
        "spirv.op_composite_extract.v4u32.u32",
        "spirv.op_bitcast.u32.i32",
    )
    assert tuple(emit.descriptor.key for emit in i64_rule.emit) == (
        "spirv.op_group_non_uniform_ballot.v4u32",
        "spirv.op_composite_extract.v4u32.u32",
        "spirv.op_composite_extract.v4u32.u32",
        "spirv.op_composite_construct.v2u32",
        "spirv.op_bitcast.v2u32.u64",
        "spirv.op_bitcast.u64.i64",
    )
    assert i32_rule.emit[1].immediates == {"component_index": 0}
    assert i64_rule.emit[1].immediates == {"component_index": 0}
    assert i64_rule.emit[2].immediates == {"component_index": 1}
    for emit in (*i32_rule.emit[:-1], *i64_rule.emit[:-1]):
        for result_name, result_ref in emit.results.items():
            if result_ref.kind is SourceValueKind.TEMPORARY:
                assert isinstance(emit.result_types[result_name], DescriptorResultType)


def test_active_masks_reuse_guarded_ballot_packing_with_a_true_predicate() -> None:
    masks = tuple(
        rule
        for rule in SPIRV_SUBGROUP_CONTRACT_CASES
        if rule.source_op is kernel.kernel_subgroup_active_mask
    )
    assert len(masks) == 2
    for mask, ballot in zip(masks, _ballot_rules(), strict=True):
        assert mask.emit[0].descriptor.key == "spirv.op_constant_true.bool"
        assert mask.emit[1].operands["predicate"] == mask.emit[0].results["dst"]
        assert tuple(emit.descriptor for emit in mask.emit[1:]) == tuple(
            emit.descriptor for emit in ballot.emit
        )
        assert tuple(
            guard
            for guard in mask.guards
            if guard.kind is GuardKind.TARGET_SUBGROUP_SIZE_RANGE
        ) == tuple(
            guard
            for guard in ballot.guards
            if guard.kind is GuardKind.TARGET_SUBGROUP_SIZE_RANGE
        )


def test_boolean_votes_are_single_native_operations() -> None:
    for operation, suffix in (
        (kernel.kernel_subgroup_vote_any, "any"),
        (kernel.kernel_subgroup_vote_all, "all"),
    ):
        rules = tuple(
            rule
            for rule in SPIRV_SUBGROUP_CONTRACT_CASES
            if rule.source_op is operation
        )
        assert len(rules) == 1
        rule = rules[0]
        assert len(rule.emit) == 1
        assert rule.descriptor.key == f"spirv.op_group_non_uniform_{suffix}.bool"
        assert any(
            guard.kind is GuardKind.DESCRIPTOR_AVAILABLE for guard in rule.guards
        )


def test_shuffle_family_covers_native_types_shapes_and_modes() -> None:
    fast = set()
    faithful = set()
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op is not kernel.kernel_subgroup_shuffle:
            continue
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "value"
        )
        mode = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "mode"
        )
        cell = (pattern.elements, pattern.lanes, mode)
        assert any(
            guard.kind is GuardKind.VALUE_I64_EQUALS_TARGET_SUBGROUP_SIZE
            and guard.field == "width"
            for guard in rule.guards
        )
        discards_valid = any(
            guard.kind is GuardKind.VALUE_NO_USES and guard.field == "valid"
            for guard in rule.guards
        )
        cells = fast if discards_valid else faithful
        assert cell not in cells, cell
        cells.add(cell)
        emit_keys = tuple(emit.descriptor.key for emit in rule.emit)
        if pattern.elements == ("bf16",):
            # Native BF16 transport is a pair-packed integer exchange. Its
            # payload recipe is checked independently below; validity is the
            # same source-lane calculation as ordinary native payloads.
            prefix_count = {None: 2, 1: 2, 2: 1, 3: 5, 4: 1}[pattern.lanes]
            suffix_count = {None: 2, 1: 2, 2: 1, 3: 5, 4: 1}[pattern.lanes]
            emit_keys = (
                *emit_keys[prefix_count : prefix_count + 2],
                *emit_keys[prefix_count + 2 + suffix_count :],
            )
        assert emit_keys[:2] == (
            "spirv.op_bitcast.i32.u32",
            f"spirv.op_group_non_uniform_shuffle.{mode}."
            f"{rule.descriptor.key.rsplit('.', 1)[-1]}",
        )
        if discards_valid:
            assert len(emit_keys) == 2
        elif mode == "index":
            assert emit_keys[2:] == (
                "spirv.op_load_builtin.subgroup_lane_id",
                "spirv.op_bitcast.i32.u32",
                "spirv.op_u_less_than.u32",
                "spirv.op_select.i32",
                "spirv.op_bitcast.i32.u32",
                "spirv.op_constant_true.bool",
                "spirv.op_group_non_uniform_ballot.v4u32",
                "spirv.op_group_non_uniform_ballot_bit_extract.bool",
                "spirv.op_logical_and.bool",
            )
        else:
            source_key = {
                "xor": "spirv.op_bitwise_xor.i32",
                "up": "spirv.op_isub.i32",
                "down": "spirv.op_iadd.i32",
            }[mode]
            assert emit_keys[2:] == (
                "spirv.op_load_builtin.subgroup_lane_id",
                source_key,
                "spirv.op_bitcast.i32.u32",
                "spirv.op_u_less_than.u32",
                "spirv.op_bitcast.i32.u32",
                "spirv.op_u_less_than.u32",
                "spirv.op_logical_and.bool",
                "spirv.op_select.i32",
                "spirv.op_bitcast.i32.u32",
                "spirv.op_constant_true.bool",
                "spirv.op_group_non_uniform_ballot.v4u32",
                "spirv.op_group_non_uniform_ballot_bit_extract.bool",
                "spirv.op_logical_and.bool",
            )

    expected = {
        (
            (*elements, "f8E4M3", "f8E5M2")
            if elements == ("i8",) and lanes in (None, 1)
            else (*elements, "bf16")
            if elements == ("i32", "index") and lanes in (None, 1)
            else elements,
            lanes,
            mode,
        )
        for elements in (
            ("i1",),
            ("i8",),
            ("i16",),
            ("i32", "index"),
            ("i64",),
            ("offset",),
            ("f16",),
            ("bf16",),
            ("f32",),
            ("f64",),
        )
        for lanes in (None, 1, 2, 3, 4)
        for mode in ("index", "xor", "up", "down")
    }
    assert fast == expected
    assert faithful == expected


def test_shuffle_contracts_require_the_matching_device_capability() -> None:
    indexed = feature_bits_value(("group_non_uniform_shuffle",))
    relative = feature_bits_value(("group_non_uniform_shuffle_relative",))
    ballot = feature_bits_value(("group_non_uniform_ballot",))
    extended = feature_bits_value(("subgroup_extended_types",))
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op is not kernel.kernel_subgroup_shuffle:
            continue
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "value"
        )
        mode = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "mode"
        )
        required_bits = 0
        for guard in rule.guards:
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE:
                for word in guard.descriptor.feature_mask_words:
                    required_bits |= word
        assert required_bits & (indexed | relative) == (
            relative if mode in ("up", "down") else indexed
        )
        assert bool(required_bits & extended) == (
            pattern.elements[0] in ("i8", "i16", "i64", "offset", "f16")
        )
        discards_valid = any(
            guard.kind is GuardKind.VALUE_NO_USES and guard.field == "valid"
            for guard in rule.guards
        )
        assert bool(required_bits & ballot) == (not discards_valid)


def test_broadcasts_share_transport_admission_and_preserve_lane_semantics() -> None:
    shapes = {}
    for operation in (
        kernel.kernel_subgroup_shuffle,
        kernel.kernel_subgroup_broadcast,
        kernel.kernel_subgroup_broadcast_first,
    ):
        shapes[operation] = set()
        for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
            if rule.source_op is not operation:
                continue
            rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
            pattern = next(
                guard.type_pattern
                for guard in rule.guards
                if guard.kind is GuardKind.VALUE_TYPE and guard.field == "value"
            )
            shapes[operation].add((pattern.elements, pattern.lanes))
            if operation is kernel.kernel_subgroup_shuffle:
                continue
            exchanges = [
                emit for emit in rule.emit if "group_non_uniform" in emit.descriptor.key
            ]
            assert len(exchanges) == 1
            exchange = exchanges[0]
            if operation is kernel.kernel_subgroup_broadcast_first:
                assert (
                    ".op_group_non_uniform_broadcast_first." in exchange.descriptor.key
                )
                assert set(exchange.operands) == {"value"}
                assert exchange.descriptor.feature_mask_words[0] & feature_bits_value(
                    ("group_non_uniform_ballot",)
                )
            else:
                # The selector may vary per invocation. OpBroadcast's dynamic
                # uniformity requirement is stronger than the High contract.
                assert ".op_group_non_uniform_shuffle.index." in exchange.descriptor.key
                selector = next(
                    emit
                    for emit in rule.emit
                    if emit.descriptor.key == "spirv.op_bitcast.i32.u32"
                )
                assert selector.operands["input"] == ValueRef.operand("lane")
                assert exchange.operands["offset"] == selector.results["dst"]
    assert (
        shapes[kernel.kernel_subgroup_shuffle]
        == shapes[kernel.kernel_subgroup_broadcast]
    )
    assert (
        shapes[kernel.kernel_subgroup_shuffle]
        == shapes[kernel.kernel_subgroup_broadcast_first]
    )


def test_native_bfloat_carriers_need_neither_int16_nor_extended_subgroup_types() -> (
    None
):
    observed = set()
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op not in (
            kernel.kernel_subgroup_shuffle,
            kernel.kernel_subgroup_broadcast,
            kernel.kernel_subgroup_broadcast_first,
        ):
            continue
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE and guard.field == "value"
        )
        if pattern.elements != ("bf16",):
            continue
        observed.add(pattern.lanes)
        required = 0
        for guard in rule.guards:
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE:
                required |= guard.descriptor.feature_mask_words[0]
        assert required & feature_bits_value(("bfloat16_type_khr",))
        assert not required & feature_bits_value(("int16", "subgroup_extended_types"))
        suffix = "v2i32" if pattern.lanes in (3, 4) else "i32"
        assert rule.descriptor.key.endswith("." + suffix)
        bitcasts = [
            emit
            for emit in rule.emit
            if "bf16" in emit.descriptor.key and ".op_bitcast." in emit.descriptor.key
        ]
        padded = "v4bf16" if pattern.lanes in (3, 4) else "v2bf16"
        assert [emit.descriptor.key for emit in bitcasts] == [
            f"spirv.op_bitcast.{padded}.{suffix}",
            f"spirv.op_bitcast.{suffix}.{padded}",
        ]
        exchange = next(
            emit for emit in rule.emit if emit.descriptor == rule.descriptor
        )
        assert exchange.operands["value"] == bitcasts[0].results["dst"]
        assert bitcasts[1].operands["input"] == exchange.results["dst"]
    assert observed == {None, 1, 2, 3, 4}


def test_reduction_family_covers_native_types_shapes_and_modes() -> None:
    actual = set()
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op is not kernel.kernel_subgroup_reduce:
            continue
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        )
        kinds = next(
            guard.enum_keywords
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_IN
        )
        mode = next(
            guard.kind
            for guard in rule.guards
            if guard.kind
            in (
                GuardKind.TARGET_WHOLE_SUBGROUP,
                GuardKind.TARGET_CONTIGUOUS_SUBGROUP_CLUSTER,
            )
        )
        for kind in kinds:
            cell = (pattern.elements[0], pattern.lanes, kind, mode)
            assert cell not in actual, cell
            actual.add(cell)
        group_emits = [
            emit
            for emit in (() if isinstance(rule, RecipeRule) else rule.emit)
            if emit.descriptor.key.startswith("spirv.op_group_non_uniform_")
        ]
        assert len(group_emits) == (0 if isinstance(rule, RecipeRule) else 1)
        if isinstance(rule, RecipeRule):
            continue
        clustered = mode is GuardKind.TARGET_CONTIGUOUS_SUBGROUP_CLUSTER
        assert bool(group_emits[0].immediates) == clustered
        if clustered:
            assert group_emits[0].descriptor.immediates[0].unsigned_max == (1 << 32) - 1
        if pattern.elements[0] == "i1" or kinds in (("addf",), ("mulf",)):
            assert len(rule.emit) == 1
    expected = {
        (element, lanes, kind, mode)
        for elements, kinds in (
            (
                ("i1", "i8", "i16", "i32", "i64"),
                (
                    "addi",
                    "muli",
                    "minsi",
                    "maxsi",
                    "minui",
                    "maxui",
                    "andi",
                    "ori",
                    "xori",
                ),
            ),
            (
                ("f16", "f32", "f64"),
                ("addf", "mulf", "minnumf", "maxnumf", "minimumf", "maximumf"),
            ),
        )
        for element in elements
        for lanes in (None, 1, 2, 3, 4)
        for kind in kinds
        for mode in (
            GuardKind.TARGET_WHOLE_SUBGROUP,
            GuardKind.TARGET_CONTIGUOUS_SUBGROUP_CLUSTER,
        )
    }
    assert actual == expected


def test_scan_family_covers_native_types_shapes_and_modes() -> None:
    actual = set()
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op is not kernel.kernel_subgroup_scan:
            continue
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        )
        kinds = next(
            guard.enum_keywords
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_IN
        )
        mode = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "mode"
        )
        direction = next(
            guard.enum_keyword
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_EQUALS and guard.field == "direction"
        )
        assert direction == "forward"
        assert any(
            guard.kind is GuardKind.TARGET_WHOLE_SUBGROUP for guard in rule.guards
        )
        for kind in kinds:
            cell = (pattern.elements[0], pattern.lanes, kind, mode)
            assert cell not in actual, cell
            actual.add(cell)
        group_emits = [
            emit
            for emit in (() if isinstance(rule, RecipeRule) else rule.emit)
            if emit.descriptor.key.startswith("spirv.op_group_non_uniform_")
        ]
        assert len(group_emits) == (0 if isinstance(rule, RecipeRule) else 1)
        if isinstance(rule, RecipeRule):
            continue
        assert not group_emits[0].immediates
        assert f".{mode}_scan." in group_emits[0].descriptor.key
    expected = {
        (element, lanes, kind, mode)
        for elements, kinds in (
            (
                ("i1", "i8", "i16", "i32", "i64"),
                (
                    "addi",
                    "muli",
                    "minsi",
                    "maxsi",
                    "minui",
                    "maxui",
                    "andi",
                    "ori",
                    "xori",
                ),
            ),
            (
                ("f16", "f32", "f64"),
                ("addf", "mulf", "minnumf", "maxnumf", "minimumf", "maximumf"),
            ),
        )
        for element in elements
        for lanes in (None, 1, 2, 3, 4)
        for kind in kinds
        for mode in ("inclusive", "exclusive")
    }
    assert actual == expected


def test_native_float_arithmetic_never_claims_source_preservation() -> None:
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op not in (
            kernel.kernel_subgroup_reduce,
            kernel.kernel_subgroup_scan,
        ):
            continue
        kinds = next(
            guard.enum_keywords
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_IN
        )
        native_only = any(
            guard.kind is GuardKind.INSTANCE_FLAGS_HAS_NONE
            and guard.field == "math"
            and guard.enum_keyword == "preserve"
            for guard in rule.guards
        )
        assert native_only == (kinds in (("addf",), ("mulf",)))


def test_workgroup_composition_has_one_topology_contract_for_all_payloads() -> None:
    rules = tuple(
        rule
        for rule in SPIRV_SUBGROUP_CONTRACT_CASES
        if rule.source_op is kernel.kernel_workgroup_reduce
    )
    assert len(rules) == 2
    kinds = set()
    for rule in rules:
        rule.validate(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET)
        assert isinstance(rule, RecipeRule)
        row_kinds = next(
            guard.enum_keywords
            for guard in rule.guards
            if guard.kind is GuardKind.ENUM_ATTR_IN
        )
        assert kinds.isdisjoint(row_kinds)
        kinds.update(row_kinds)
        assert not any(guard.kind is GuardKind.VALUE_TYPE for guard in rule.guards)
        assert any(
            guard.kind is GuardKind.INSTANCE_FLAGS_HAS_NONE
            and guard.field == "math"
            and guard.enum_keyword == "preserve"
            for guard in rule.guards
        ) == (set(row_kinds) == {"addf", "mulf"})
        topology = {
            guard.descriptor.key
            for guard in rule.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
        }
        assert "spirv.op_group_non_uniform_iadd.exclusive_scan.i32" in topology
        assert "spirv.op_group_non_uniform_iadd.reduce.i32" in topology
    assert kinds == {
        "addi",
        "muli",
        "minsi",
        "maxsi",
        "minui",
        "maxui",
        "andi",
        "ori",
        "xori",
        "addf",
        "mulf",
        "minnumf",
        "maxnumf",
        "minimumf",
        "maximumf",
    }


def test_arithmetic_types_require_logical_device_admission() -> None:
    extended = feature_bits_value(("subgroup_extended_types",))
    for rule in SPIRV_SUBGROUP_CONTRACT_CASES:
        if rule.source_op not in (
            kernel.kernel_subgroup_reduce,
            kernel.kernel_subgroup_scan,
        ):
            continue
        pattern = next(
            guard.type_pattern
            for guard in rule.guards
            if guard.kind is GuardKind.VALUE_TYPE
        )
        requires_extended = any(
            mask & extended
            for guard in rule.guards
            if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
            for mask in guard.descriptor.feature_mask_words
        )
        assert requires_extended == (
            pattern.elements[0] in ("i8", "i16", "i64", "f16")
            or (pattern.elements[0] == "f64" and isinstance(rule, RecipeRule))
        )
        if isinstance(rule, RecipeRule):
            assert pattern.elements[0] in ("f16", "f32", "f64")
            assert all(
                not guard.descriptor.key.startswith("spirv.op_f")
                for guard in rule.guards
                if guard.kind is GuardKind.DESCRIPTOR_AVAILABLE
            )
