# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V subgroup source-to-low contract rows."""

from __future__ import annotations

from dataclasses import dataclass

from loom.dialect.kernel import defs as kernel
from loom.target.arch.spirv.contracts.descriptor_rule import (
    IntegerRecipe,
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.ordinary_vector import (
    OrdinaryVectorComponentKind,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.scalar_alu import (
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    IntegerAluTypePair,
)
from loom.target.arch.spirv.subgroup import (
    SPIRV_COLLECTIVE_EXTREMA_TYPES,
    SPIRV_SUBGROUP_ARITHMETIC,
    SPIRV_SUBGROUP_BALLOT_BIT_EXTRACT_INSTRUCTION,
    SPIRV_SUBGROUP_BALLOT_INSTRUCTION,
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS,
    SPIRV_SUBGROUP_SHUFFLES,
    SPIRV_SUBGROUP_VOTE_INSTRUCTIONS,
    SpirvSubgroupArithmetic,
    SpirvSubgroupShuffle,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    RecipeRule,
    Scalar,
    SourceValueKind,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_SCAN_MODES = {
    "LOOM_SPIRV_GROUP_OPERATION_INCLUSIVE_SCAN": "inclusive",
    "LOOM_SPIRV_GROUP_OPERATION_EXCLUSIVE_SCAN": "exclusive",
}


def _integer_pair(source_type: str) -> IntegerAluTypePair:
    matches = tuple(
        pair
        for pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        if pair.source_type == source_type
    )
    if len(matches) != 1:
        raise ValueError(f"expected one SPIR-V integer pair for {source_type!r}")
    return matches[0]


_I32_PAIR = _integer_pair("i32")
_I64_PAIR = _integer_pair("i64")
_BALLOT = logical_core_descriptor(SPIRV_SUBGROUP_BALLOT_INSTRUCTION.key)
_EXTRACT_WORD = logical_core_descriptor(
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS[0].key
)
_CONSTRUCT_WORD_PAIR = logical_core_descriptor(
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS[1].key
)
_BITCAST_WORD_PAIR_TO_U64 = logical_core_descriptor(
    SPIRV_SUBGROUP_BALLOT_PACKING_INSTRUCTIONS[2].key
)
_BITCAST_I32_TO_U32 = logical_core_descriptor("spirv.op_bitcast.i32.u32")
_BALLOT_BIT_EXTRACT = logical_core_descriptor(
    SPIRV_SUBGROUP_BALLOT_BIT_EXTRACT_INSTRUCTION.key
)
_LOAD_SUBGROUP_LANE_ID = logical_core_descriptor(
    "spirv.op_load_builtin.subgroup_lane_id"
)
_IADD_I32 = logical_core_descriptor("spirv.op_iadd.i32")
_ISUB_I32 = logical_core_descriptor("spirv.op_isub.i32")
_BITWISE_XOR_I32 = logical_core_descriptor("spirv.op_bitwise_xor.i32")
_U_LESS_THAN_U32 = logical_core_descriptor("spirv.op_u_less_than.u32")
_SELECT_I32 = logical_core_descriptor("spirv.op_select.i32")
_CONSTANT_TRUE = logical_core_descriptor("spirv.op_constant_true.bool")
_LOGICAL_AND_BOOL = logical_core_descriptor("spirv.op_logical_and.bool")


def _integer_view_descriptor(pair: IntegerAluTypePair) -> Descriptor:
    return logical_core_descriptor(
        f"spirv.op_bitcast.{pair.unsigned.suffix}.{pair.signed.suffix}"
    )


def _ballot_emit(mask_type: str, predicate: ValueRef) -> tuple[EmitDescriptorOp, ...]:
    native_mask = ValueRef.temporary("native_mask")
    low_word = ValueRef.temporary("low_word")
    emits = [
        emit_descriptor_op(
            descriptor=_BALLOT,
            operands={"predicate": predicate},
            results={"dst": native_mask},
            result_types={"dst": DescriptorResultType()},
        ),
        emit_descriptor_op(
            descriptor=_EXTRACT_WORD,
            operands={"composite": native_mask},
            results={"dst": low_word},
            result_types={"dst": DescriptorResultType()},
            immediates={"component_index": 0},
        ),
    ]
    if mask_type == "i32":
        emits.append(
            emit_descriptor_op(
                descriptor=_integer_view_descriptor(_I32_PAIR),
                operands={"input": low_word},
                results={"dst": ValueRef.result("mask")},
            )
        )
        return tuple(emits)

    high_word = ValueRef.temporary("high_word")
    packed_words = ValueRef.temporary("packed_words")
    unsigned_mask = ValueRef.temporary("unsigned_mask")
    emits.extend(
        (
            emit_descriptor_op(
                descriptor=_EXTRACT_WORD,
                operands={"composite": native_mask},
                results={"dst": high_word},
                result_types={"dst": DescriptorResultType()},
                immediates={"component_index": 1},
            ),
            emit_descriptor_op(
                descriptor=_CONSTRUCT_WORD_PAIR,
                operands={"component0": low_word, "component1": high_word},
                results={"dst": packed_words},
                result_types={"dst": DescriptorResultType()},
            ),
            emit_descriptor_op(
                descriptor=_BITCAST_WORD_PAIR_TO_U64,
                operands={"input": packed_words},
                results={"dst": unsigned_mask},
                result_types={"dst": DescriptorResultType()},
            ),
            emit_descriptor_op(
                descriptor=_integer_view_descriptor(_I64_PAIR),
                operands={"input": unsigned_mask},
                results={"dst": ValueRef.result("mask")},
            ),
        )
    )
    return tuple(emits)


def _ballot_rule(
    mask_type: str, mask_bit_count: int, *, active_mask: bool = False
) -> DescriptorRule:
    descriptors = (
        _BALLOT,
        _EXTRACT_WORD,
        _integer_view_descriptor(_I32_PAIR),
    )
    if mask_type == "i64":
        descriptors = (
            *descriptors,
            _CONSTRUCT_WORD_PAIR,
            _BITCAST_WORD_PAIR_TO_U64,
            _integer_view_descriptor(_I64_PAIR),
        )
    predicate = ValueRef.operand("predicate")
    prefix = ()
    predicate_guards = (Guard.value_type("predicate", Scalar("i1")),)
    if active_mask:
        predicate = ValueRef.temporary("active")
        predicate_guards = ()
        constant = logical_core_descriptor("spirv.op_constant_true.bool")
        descriptors = (*descriptors, constant)
        prefix = (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": predicate},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.CONST,
            ),
        )
    return DescriptorRule(
        source_op=(
            kernel.kernel_subgroup_active_mask
            if active_mask
            else kernel.kernel_subgroup_vote_ballot
        ),
        descriptor=_BALLOT,
        guards=(
            *predicate_guards,
            Guard.value_type("mask", Scalar(mask_type)),
            Guard.target_subgroup_size_range(1, mask_bit_count),
            *descriptor_feature_guards(*descriptors),
        ),
        emit=(*prefix, *_ballot_emit(mask_type, predicate)),
    )


def _vote_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        DescriptorRule(
            source_op=source,
            descriptor=descriptor,
            guards=(
                Guard.value_type("predicate", Scalar("i1")),
                Guard.value_type("result", Scalar("i1")),
                *descriptor_feature_guards(descriptor),
            ),
            emit=(
                emit_descriptor_op(
                    descriptor=descriptor,
                    operands={"predicate": ValueRef.operand("predicate")},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        )
        for source, row in zip(
            (kernel.kernel_subgroup_vote_any, kernel.kernel_subgroup_vote_all),
            SPIRV_SUBGROUP_VOTE_INSTRUCTIONS,
            strict=True,
        )
        for descriptor in (logical_core_descriptor(row.key),)
    )


@dataclass(frozen=True, slots=True)
class _TransportPayload:
    # Authored scalar or rank-one vector value admitted by this carrier.
    pattern: TypePattern
    # Native subgroup operand/result type, independent of the source encoding.
    suffix: str
    # Bit-preserving preparation ending in the temporary named payload.
    pack: tuple[EmitDescriptorOp, ...] = ()
    # Bit-preserving reconstruction from the exchanged_payload temporary.
    unpack: tuple[EmitDescriptorOp, ...] = ()


def _transport_emits(
    payload: _TransportPayload, descriptor: Descriptor, selector: str | None
) -> list[EmitDescriptorOp]:
    emits = list(payload.pack)
    operands = {
        "value": ValueRef.temporary("payload")
        if payload.pack
        else ValueRef.operand("value")
    }
    if selector is not None:
        unsigned_offset = ValueRef.temporary("unsigned_offset")
        emits.append(
            emit_descriptor_op(
                descriptor=_BITCAST_I32_TO_U32,
                operands={"input": ValueRef.operand(selector)},
                results={"dst": unsigned_offset},
                result_types={"dst": DescriptorResultType()},
            )
        )
        operands["offset"] = unsigned_offset
    emits.append(
        emit_descriptor_op(
            descriptor=descriptor,
            operands=operands,
            results={
                "dst": (
                    ValueRef.temporary("exchanged_payload")
                    if payload.unpack
                    else ValueRef.result("result")
                )
            },
            result_types={"dst": DescriptorResultType()} if payload.unpack else None,
        )
    )
    emits.extend(payload.unpack)
    return emits


def _shuffle_rule(
    row: SpirvSubgroupShuffle,
    payload: _TransportPayload,
    *,
    preserve_valid: bool,
) -> DescriptorRule:
    descriptor = logical_core_descriptor(row.instruction.key)
    emits = _transport_emits(payload, descriptor, "offset")
    unsigned_offset = ValueRef.temporary("unsigned_offset")
    if preserve_valid:
        lane_i32 = ValueRef.temporary("lane_i32")
        source_i32 = ValueRef.temporary("source_i32")
        width_u32 = ValueRef.temporary("width_u32")
        offset_in_range = ValueRef.temporary("offset_in_range")
        safe_source_i32 = ValueRef.temporary("safe_source_i32")
        safe_source_u32 = ValueRef.temporary("safe_source_u32")
        active = ValueRef.temporary("active")
        active_mask = ValueRef.temporary("active_mask")
        source_active = ValueRef.temporary("source_active")
        source_descriptor = {
            "index": None,
            "xor": _BITWISE_XOR_I32,
            "up": _ISUB_I32,
            "down": _IADD_I32,
        }[row.mode]
        emits.append(
            emit_descriptor_op(
                descriptor=_LOAD_SUBGROUP_LANE_ID,
                results={"dst": lane_i32},
                result_types={"dst": DescriptorResultType()},
            )
        )
        if source_descriptor is None:
            source_i32 = ValueRef.operand("offset")
            source_u32 = unsigned_offset
            in_range = offset_in_range
        else:
            source_u32 = ValueRef.temporary("source_u32")
            source_in_range = ValueRef.temporary("source_in_range")
            in_range = ValueRef.temporary("offset_and_source_in_range")
            emits.append(
                emit_descriptor_op(
                    descriptor=source_descriptor,
                    operands={
                        "lhs": lane_i32,
                        "rhs": ValueRef.operand("offset"),
                    },
                    results={"dst": source_i32},
                    result_types={"dst": DescriptorResultType()},
                )
            )
        emits.extend(
            (
                emit_descriptor_op(
                    descriptor=_BITCAST_I32_TO_U32,
                    operands={"input": ValueRef.operand("width")},
                    results={"dst": width_u32},
                    result_types={"dst": DescriptorResultType()},
                ),
                emit_descriptor_op(
                    descriptor=_U_LESS_THAN_U32,
                    operands={"lhs": unsigned_offset, "rhs": width_u32},
                    results={"dst": offset_in_range},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        if source_descriptor is not None:
            emits.extend(
                (
                    emit_descriptor_op(
                        descriptor=_BITCAST_I32_TO_U32,
                        operands={"input": source_i32},
                        results={"dst": source_u32},
                        result_types={"dst": DescriptorResultType()},
                    ),
                    emit_descriptor_op(
                        descriptor=_U_LESS_THAN_U32,
                        operands={"lhs": source_u32, "rhs": width_u32},
                        results={"dst": source_in_range},
                        result_types={"dst": DescriptorResultType()},
                    ),
                    emit_descriptor_op(
                        descriptor=_LOGICAL_AND_BOOL,
                        operands={
                            "lhs": offset_in_range,
                            "rhs": source_in_range,
                        },
                        results={"dst": in_range},
                        result_types={"dst": DescriptorResultType()},
                    ),
                )
            )
        emits.extend(
            (
                emit_descriptor_op(
                    descriptor=_SELECT_I32,
                    operands={
                        "condition": in_range,
                        "true_value": source_i32,
                        "false_value": lane_i32,
                    },
                    results={"dst": safe_source_i32},
                    result_types={"dst": DescriptorResultType()},
                ),
                emit_descriptor_op(
                    descriptor=_BITCAST_I32_TO_U32,
                    operands={"input": safe_source_i32},
                    results={"dst": safe_source_u32},
                    result_types={"dst": DescriptorResultType()},
                ),
                EmitDescriptorOp(
                    descriptor=_CONSTANT_TRUE,
                    results={"dst": active},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.CONST,
                ),
                emit_descriptor_op(
                    descriptor=_BALLOT,
                    operands={"predicate": active},
                    results={"dst": active_mask},
                    result_types={"dst": DescriptorResultType()},
                ),
                emit_descriptor_op(
                    descriptor=_BALLOT_BIT_EXTRACT,
                    operands={"mask": active_mask, "index": safe_source_u32},
                    results={"dst": source_active},
                    result_types={"dst": DescriptorResultType()},
                ),
                emit_descriptor_op(
                    descriptor=_LOGICAL_AND_BOOL,
                    operands={"lhs": in_range, "rhs": source_active},
                    results={"dst": ValueRef.result("valid")},
                ),
            )
        )
    return DescriptorRule(
        source_op=kernel.kernel_subgroup_shuffle,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("mode", row.mode),
            Guard.value_type("value", payload.pattern),
            Guard.value_type("offset", Scalar("i32")),
            Guard.value_i64_equals_target_subgroup_size("width"),
            *((Guard.value_no_uses("valid"),) if not preserve_valid else ()),
            *descriptor_feature_guards(*(emit.descriptor for emit in emits)),
        ),
        emit=tuple(emits),
    )


def _native_bfloat_transport(
    source_lanes: int,
) -> tuple[str, tuple[EmitDescriptorOp, ...], tuple[EmitDescriptorOp, ...]]:
    """Packs BF16 pairs into integer words without requiring Int16."""
    lanes = max(1, source_lanes)
    padded_lanes = 2 if lanes <= 2 else 4
    carrier_suffix = "i32" if padded_lanes == 2 else "v2i32"
    pack: list[EmitDescriptorOp] = []
    unpack: list[EmitDescriptorOp] = []
    source = ValueRef.operand("value")
    padded = source

    def emit(
        steps: list[EmitDescriptorOp],
        key: str,
        operands: dict[str, ValueRef],
        result: ValueRef,
        **immediates: int,
    ) -> None:
        steps.append(
            emit_descriptor_op(
                descriptor=logical_core_descriptor(f"spirv.op_{key}"),
                operands=operands,
                results={"dst": result},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result.kind is SourceValueKind.TEMPORARY
                    else None
                ),
                immediates=immediates,
            )
        )

    if lanes in (1, 3):
        elements = [source]
        if lanes == 3:
            elements = [ValueRef.temporary(f"element{index}") for index in range(3)]
            for index, element in enumerate(elements):
                emit(
                    pack,
                    "composite_extract.v3bf16.bf16",
                    {"composite": source},
                    element,
                    component_index=index,
                )
        padded = ValueRef.temporary("padded_bfloat")
        emit(
            pack,
            f"composite_construct.v{padded_lanes}bf16",
            {
                f"component{index}": elements[min(index, lanes - 1)]
                for index in range(padded_lanes)
            },
            padded,
        )
    emit(
        pack,
        f"bitcast.v{padded_lanes}bf16.{carrier_suffix}",
        {"input": padded},
        ValueRef.temporary("payload"),
    )
    restored = (
        ValueRef.result("result")
        if lanes in (2, 4)
        else ValueRef.temporary("restored_bfloat")
    )
    emit(
        unpack,
        f"bitcast.{carrier_suffix}.v{padded_lanes}bf16",
        {"input": ValueRef.temporary("exchanged_payload")},
        restored,
    )
    if lanes == 1:
        emit(
            unpack,
            "composite_extract.v2bf16.bf16",
            {"composite": restored},
            ValueRef.result("result"),
            component_index=0,
        )
    elif lanes == 3:
        elements = [ValueRef.temporary(f"restored{index}") for index in range(3)]
        for index, element in enumerate(elements):
            emit(
                unpack,
                "composite_extract.v4bf16.bf16",
                {"composite": restored},
                element,
                component_index=index,
            )
        emit(
            unpack,
            "composite_construct.v3bf16",
            {f"component{index}": element for index, element in enumerate(elements)},
            ValueRef.result("result"),
        )
    return carrier_suffix, tuple(pack), tuple(unpack)


def _transport_payloads() -> tuple[_TransportPayload, ...]:
    payloads = []
    # Native BF16 must precede the scalar software-carrier alternative.
    for source_lanes in (0, 1, 2, 3, 4):
        suffix, pack, unpack = _native_bfloat_transport(source_lanes)
        pattern = (
            Scalar("bf16") if source_lanes == 0 else Vector("bf16", lanes=source_lanes)
        )
        payloads.append(_TransportPayload(pattern, suffix, pack, unpack))
    for row in SPIRV_SUBGROUP_SHUFFLES:
        if row.mode != "index":
            continue
        value_type = row.instruction.result_type
        component = (
            value_type.component_type
            if isinstance(value_type, OrdinaryVectorType)
            else value_type
        )
        lane_counts = (
            (value_type.lane_count,)
            if isinstance(value_type, OrdinaryVectorType)
            else (0, 1)
        )
        for source_lanes in lane_counts:
            source_types = component.source_types
            if source_lanes in (0, 1):
                if component.suffix == "i8":
                    source_types = (*source_types, "f8E4M3", "f8E5M2")
                elif component.suffix == "i32":
                    source_types = (*source_types, "bf16")
            pattern = (
                Scalar(source_types)
                if source_lanes == 0
                else Vector(source_types, lanes=source_lanes)
            )
            payloads.append(_TransportPayload(pattern, value_type.suffix))
    return tuple(payloads)


def _broadcast_rule(payload: _TransportPayload, *, first: bool) -> DescriptorRule:
    # A named source lane can vary across invocations in High. Indexed shuffle
    # preserves that contract; SPIR-V Broadcast requires a uniform selector.
    descriptor = logical_core_descriptor(
        f"spirv.op_group_non_uniform_broadcast_first.{payload.suffix}"
        if first
        else f"spirv.op_group_non_uniform_shuffle.index.{payload.suffix}"
    )
    emits = _transport_emits(payload, descriptor, None if first else "lane")
    return DescriptorRule(
        source_op=kernel.kernel_subgroup_broadcast_first
        if first
        else kernel.kernel_subgroup_broadcast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", payload.pattern),
            *((Guard.value_type("lane", Scalar("i32")),) if not first else ()),
            *descriptor_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=tuple(emits),
    )


def _transport_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    rows = {
        (row.mode, row.instruction.result_type.suffix): row
        for row in SPIRV_SUBGROUP_SHUFFLES
    }
    for payload in _transport_payloads():
        for mode in ("index", "xor", "up", "down"):
            rules.extend(
                _shuffle_rule(
                    rows[mode, payload.suffix],
                    payload,
                    preserve_valid=preserve_valid,
                )
                for preserve_valid in (False, True)
            )
        rules.append(_broadcast_rule(payload, first=False))
        rules.append(_broadcast_rule(payload, first=True))
    return tuple(rules)


def _arithmetic_rule(
    row: SpirvSubgroupArithmetic,
    source_lanes: int,
) -> DescriptorRule:
    instruction = row.instruction
    value_type = instruction.result_type
    component = (
        value_type.component_type
        if isinstance(value_type, OrdinaryVectorType)
        else value_type
    )
    source_type = row.source_type
    kinds = row.combining_kinds
    descriptor = logical_core_descriptor(instruction.key)
    scan_mode = _SCAN_MODES.get(instruction.group_operation)
    cluster_guard = (
        Guard.target_contiguous_subgroup_cluster
        if instruction.clustered
        else Guard.target_whole_subgroup
    )("cluster_size", "cluster_stride")
    immediates = (
        {"cluster_size": AttrProject.direct("cluster_size")}
        if instruction.clustered
        else {}
    )
    recipe = IntegerRecipe(component.bit_width, max(1, source_lanes))
    value = ValueRef.operand("value")
    result = ValueRef.result("result")
    if component.kind == OrdinaryVectorComponentKind.UNSIGNED_INTEGER:
        value = recipe.emit(
            f"bitcast.{recipe.integer_suffix}.{value_type.suffix}", {"input": value}
        )
        result = ValueRef.temporary("unsigned_result")
    recipe.emits.append(
        emit_descriptor_op(
            descriptor=descriptor,
            operands={"value": value},
            results={"dst": result},
            result_types={"dst": DescriptorResultType()},
            immediates=immediates,
        )
    )
    if component.kind == OrdinaryVectorComponentKind.UNSIGNED_INTEGER:
        recipe.emit(
            f"bitcast.{value_type.suffix}.{recipe.integer_suffix}",
            {"input": result},
            result=ValueRef.result("result"),
        )
    pattern = (
        Scalar(source_type)
        if source_lanes == 0
        else Vector(source_type, lanes=source_lanes)
    )
    source_op = (
        kernel.kernel_subgroup_scan
        if scan_mode is not None
        else kernel.kernel_subgroup_reduce
    )
    scan_guards = (
        (
            Guard.enum_attr_equals("mode", scan_mode),
            Guard.enum_attr_equals("direction", "forward"),
        )
        if scan_mode is not None
        else ()
    )
    math_guards = (
        (Guard.instance_flags_has_none("math", "preserve"),)
        if component.kind == OrdinaryVectorComponentKind.FLOAT
        else ()
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_in("kind", kinds),
            *math_guards,
            Guard.value_type("value", pattern),
            cluster_guard,
            *scan_guards,
            *descriptor_feature_guards(*(emit.descriptor for emit in recipe.emits)),
        ),
        emit=tuple(recipe.emits),
    )


def _arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    for row in SPIRV_SUBGROUP_ARITHMETIC:
        value_type = row.instruction.result_type
        lane_counts = (
            (value_type.lane_count,)
            if isinstance(value_type, OrdinaryVectorType)
            else (0, 1)
        )
        rules.extend(
            _arithmetic_rule(row, source_lanes) for source_lanes in lane_counts
        )
    return tuple(rules)


def _extrema_rules() -> tuple[RecipeRule, ...]:
    rules = []
    for row in SPIRV_COLLECTIVE_EXTREMA_TYPES:
        for source_lanes in (0, 1) if row.lane_count == 1 else (row.lane_count,):
            source_type = f"f{row.bit_count}"
            pattern = (
                Scalar(source_type)
                if source_lanes == 0
                else Vector(source_type, lanes=source_lanes)
            )
            for mode in (
                "reduce",
                "clustered_reduce",
                "inclusive_scan",
                "exclusive_scan",
            ):
                scan = mode.endswith("_scan")
                cluster_guard = (
                    Guard.target_contiguous_subgroup_cluster
                    if mode == "clustered_reduce"
                    else Guard.target_whole_subgroup
                )("cluster_size", "cluster_stride")
                descriptors = (
                    *(logical_core_descriptor(key) for key in row.descriptors.values()),
                    *(
                        logical_core_descriptor(row.subgroup_key(mode, minimum=minimum))
                        for minimum in (True, False)
                    ),
                )
                rules.append(
                    RecipeRule(
                        source_op=kernel.kernel_subgroup_scan
                        if scan
                        else kernel.kernel_subgroup_reduce,
                        guards=(
                            Guard.enum_attr_in(
                                "kind", ("minnumf", "maxnumf", "minimumf", "maximumf")
                            ),
                            Guard.value_type("value", pattern),
                            cluster_guard,
                            *(
                                (
                                    Guard.enum_attr_equals(
                                        "mode", mode.removesuffix("_scan")
                                    ),
                                    Guard.enum_attr_equals("direction", "forward"),
                                )
                                if scan
                                else ()
                            ),
                            *descriptor_feature_guards(*descriptors),
                        ),
                    )
                )
    return tuple(rules)


def _workgroup_rules() -> tuple[RecipeRule, ...]:
    # Every payload shares the same dynamic subgroup topology. The retained
    # typed plan checks the independent payload carriers and their capabilities.
    topology = descriptor_feature_guards(
        logical_core_descriptor("spirv.op_load_builtin.subgroup_id"),
        logical_core_descriptor("spirv.op_load_builtin.subgroup_count"),
        logical_core_descriptor("spirv.op_group_non_uniform_iadd.exclusive_scan.i32"),
        logical_core_descriptor("spirv.op_group_non_uniform_iadd.reduce.i32"),
        logical_core_descriptor("spirv.op_control_barrier.workgroup.workgroup.acq_rel"),
    )
    return (
        RecipeRule(
            source_op=kernel.kernel_workgroup_reduce,
            guards=(
                Guard.enum_attr_in("kind", ("addf", "mulf")),
                Guard.instance_flags_has_none("math", "preserve"),
                *topology,
            ),
        ),
        RecipeRule(
            source_op=kernel.kernel_workgroup_reduce,
            guards=(
                Guard.enum_attr_in(
                    "kind",
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
                        "minimumf",
                        "maximumf",
                        "minnumf",
                        "maxnumf",
                    ),
                ),
                *topology,
            ),
        ),
    )


SPIRV_SUBGROUP_CONTRACT_CASES: tuple[ContractCase, ...] = (
    _ballot_rule("i32", 32),
    _ballot_rule("i64", 64),
    _ballot_rule("i32", 32, active_mask=True),
    _ballot_rule("i64", 64, active_mask=True),
    *_vote_rules(),
    *_transport_rules(),
    *_arithmetic_rules(),
    *_extrema_rules(),
    *_workgroup_rules(),
)
