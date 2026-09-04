# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P scalar and vector memory selection rules."""

from __future__ import annotations

from enum import Enum, unique

from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.core_descriptors import AIE2P_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    SourceMemoryAddressLayout,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_SCALAR_MEMORY_SHAPES = (
    (
        "i8",
        1,
        -8,
        7,
        (_I1, _I8, _F8E4M3, _F8E5M2),
    ),
    (
        "i16",
        2,
        -16,
        14,
        (_I16, _F16, _BF16),
    ),
    (
        "i32",
        4,
        -32,
        28,
        (_I32, _F32, _INDEX, _OFFSET),
    ),
)
_F32X64_ACCUMULATOR = Vector("f32", lanes=64)
_ACCUMULATOR_CHUNK_BYTE_OFFSETS = (0, 64, 128, 192)
_ZERO_X_SPLAT_BY_ELEMENT_BYTE_COUNT = {
    1: "amd.xdna.aie2p.splat.i8x64",
    2: "amd.xdna.aie2p.splat.i16x32",
    4: "amd.xdna.aie2p.splat.i32x16",
}
_VECTOR_MEMORY_ELEMENT_TYPES = (
    ("i8", "i8", 8),
    ("f8E4M3", "i8", 8),
    ("f8E5M2", "i8", 8),
    ("i16", "i16", 16),
    ("f16", "i16", 16),
    ("bf16", "bf16", 16),
    ("i32", "i32", 32),
    ("f32", "f32", 32),
)
_VECTOR_MEMORY_SHAPES = tuple(
    (
        width_bits,
        element_type,
        descriptor_element_type,
        element_bits // 8,
        width_bits // element_bits,
        Vector(element_type, lanes=width_bits // element_bits),
    )
    for width_bits in (128, 256, 512)
    for element_type, descriptor_element_type, element_bits in (
        _VECTOR_MEMORY_ELEMENT_TYPES
    )
)

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1

_MEMORY_ROOTS = (
    (
        SourceMemoryRootKind.BLOCK_ARGUMENT,
        ("unknown", "generic", "workgroup"),
    ),
    (
        SourceMemoryRootKind.ALLOCA,
        ("private", "workgroup"),
    ),
)


@unique
class _MemoryAddressForm(Enum):
    """AIE2P memory address realization selected by one rule."""

    IMMEDIATE = "immediate"
    MATERIALIZED_STATIC = "materialized_static"
    DYNAMIC_ZERO_STATIC = "dynamic_zero_static"
    DYNAMIC_SMALL_STATIC = "dynamic_small_static"
    DYNAMIC_FULL_STATIC = "dynamic_full_static"


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _memory_constraint(
    operation: SourceMemoryOperation,
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    element_byte_count: int,
    vector_lane_count: int,
    minimum_alignment: int,
    immediate_offset_minimum: int,
    immediate_offset_maximum: int,
    maximum_additional_static_byte_offset: int = 0,
) -> SourceMemoryConstraint:
    dynamic = address_form in (
        _MemoryAddressForm.DYNAMIC_ZERO_STATIC,
        _MemoryAddressForm.DYNAMIC_SMALL_STATIC,
        _MemoryAddressForm.DYNAMIC_FULL_STATIC,
    )
    if address_form is _MemoryAddressForm.IMMEDIATE:
        static_minimum = immediate_offset_minimum
        static_maximum = (
            immediate_offset_maximum - maximum_additional_static_byte_offset
        )
    elif address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC:
        static_minimum = static_maximum = 0
    elif address_form is _MemoryAddressForm.DYNAMIC_SMALL_STATIC:
        static_minimum, static_maximum = -64, 63
    else:
        static_minimum, static_maximum = _I32_MIN, _I32_MAX
    static_maximum = min(
        static_maximum,
        _I32_MAX - maximum_additional_static_byte_offset,
    )
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=root_kind,
        address_layout=SourceMemoryAddressLayout.COMPACT_ROW_MAJOR,
        memory_spaces=memory_spaces,
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=static_minimum,
        static_byte_offset_maximum=static_maximum,
        minimum_alignment=minimum_alignment,
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        allow_dynamic_stride_values=dynamic,
    )


def _byte_offset_materializer() -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        const_i64=_descriptor("amd.xdna.aie2p.constant.i32"),
        add_i64=_descriptor("amd.xdna.aie2p.add.i32"),
        mul_i64=_descriptor("amd.xdna.aie2p.mul.i32"),
        shl_i64=_descriptor("amd.xdna.aie2p.lshl.i32"),
        const_i64_immediate="i",
    )


def _register_address_emits(
    source_memory: SourceMemoryConstraint,
    address_form: _MemoryAddressForm,
    *,
    additional_static_byte_offset: int = 0,
    temporary_suffix: str = "",
) -> tuple[tuple[EmitDescriptorOp, ...], ValueRef]:
    emits: list[EmitDescriptorOp] = []
    byte_offset = ValueRef.temporary(f"byte_offset{temporary_suffix}")
    static_byte_offset = (
        SourceMemoryProject.static_byte_offset_plus(additional_static_byte_offset)
        if additional_static_byte_offset
        else SourceMemoryProject.static_byte_offset()
    )
    if address_form is _MemoryAddressForm.MATERIALIZED_STATIC:
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.materialize.static-byte-offset.i32"
                ),
                results={"dst": byte_offset},
                result_types={"dst": _OFFSET},
                immediates={"i": static_byte_offset},
                source_memory=source_memory,
                form=DescriptorEmitForm.OP,
            )
        )
    elif (
        address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC
        and additional_static_byte_offset == 0
    ):
        byte_offset = ValueRef.source_memory_dynamic_byte_offset()
    elif (
        address_form is _MemoryAddressForm.DYNAMIC_SMALL_STATIC
        and additional_static_byte_offset == 0
    ):
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.add.i32.immediate"),
                operands={"s0": ValueRef.source_memory_dynamic_byte_offset()},
                results={"d0": byte_offset},
                result_types={"d0": _OFFSET},
                immediates={"imm": static_byte_offset},
                source_memory=source_memory,
                source_memory_byte_offset_materializer=_byte_offset_materializer(),
                form=DescriptorEmitForm.OP,
            )
        )
    else:
        static_offset = ValueRef.temporary(f"static_byte_offset{temporary_suffix}")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=_descriptor(
                        "amd.xdna.aie2p.materialize.static-byte-offset.i32"
                    ),
                    results={"dst": static_offset},
                    result_types={"dst": _OFFSET},
                    immediates={"i": static_byte_offset},
                    source_memory=source_memory,
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=_descriptor("amd.xdna.aie2p.add.i32"),
                    operands={
                        "s0": ValueRef.source_memory_dynamic_byte_offset(),
                        "s1": static_offset,
                    },
                    results={"d0": byte_offset},
                    result_types={"d0": _OFFSET},
                    source_memory=source_memory,
                    source_memory_byte_offset_materializer=(
                        _byte_offset_materializer()
                    ),
                    form=DescriptorEmitForm.OP,
                ),
            )
        )

    address_index = ValueRef.temporary(f"address_index{temporary_suffix}")
    materialize_dynamic_offset = (
        address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC
        and additional_static_byte_offset == 0
    )
    emits.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.move.to.address-index"),
            operands={"src": byte_offset},
            results={"dst": address_index},
            result_types={"dst": DescriptorResultType()},
            source_memory=source_memory if materialize_dynamic_offset else None,
            source_memory_byte_offset_materializer=(
                _byte_offset_materializer() if materialize_dynamic_offset else None
            ),
            form=DescriptorEmitForm.OP,
        )
    )
    return tuple(emits), address_index


def _zero_x_emits(element_byte_count: int) -> tuple[ContractEmit, ...]:
    scalar_zero = ValueRef.temporary("scalar_zero")
    return (
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.constant.i32.short"),
            results={"dst": scalar_zero},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 0},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor(
                _ZERO_X_SPLAT_BY_ELEMENT_BYTE_COUNT[element_byte_count]
            ),
            operands={"src": scalar_zero},
            results={"dst": ValueRef.temporary("vector_zero")},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
    )


def _memory_rule(
    operation: SourceMemoryOperation,
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    source_op: Op,
    value_type: TypePattern,
    immediate_descriptor_key: str,
    register_descriptor_key: str,
    element_byte_count: int,
    vector_lane_count: int,
    minimum_alignment: int,
    immediate_offset_minimum: int,
    immediate_offset_maximum: int,
    volatile: bool,
    expand_to_x_carrier: bool = False,
) -> DescriptorRule:
    is_load = operation is SourceMemoryOperation.LOAD
    immediate_memory = address_form is _MemoryAddressForm.IMMEDIATE
    memory_descriptor_key = (
        immediate_descriptor_key if immediate_memory else register_descriptor_key
    )
    if volatile:
        memory_descriptor_key = f"{memory_descriptor_key}.volatile"
    memory_descriptor = _descriptor(memory_descriptor_key)
    source_memory = _memory_constraint(
        operation,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        minimum_alignment=minimum_alignment,
        immediate_offset_minimum=immediate_offset_minimum,
        immediate_offset_maximum=immediate_offset_maximum,
    )
    source_value = ValueRef.operand("value")
    result_value = ValueRef.result("result")
    descriptor_value = (
        ValueRef.temporary("memory_value")
        if expand_to_x_carrier
        else (result_value if is_load else source_value)
    )
    memory_operands = (
        {"ptr": ValueRef.operand("view")}
        if is_load
        else {
            "src": descriptor_value,
            "ptr": ValueRef.operand("view"),
        }
    )
    memory_results = {"dst": descriptor_value} if is_load else {}
    memory_result_types = (
        {"dst": DescriptorResultType()} if is_load and expand_to_x_carrier else None
    )
    emits: list[ContractEmit] = []
    storage_continuation = next(
        (
            operand
            for operand in memory_descriptor.operands
            if operand.field_name == "storage"
        ),
        None,
    )
    padding_value = ValueRef.temporary("memory_padding")
    # Ordinary vector SSA values use a full X carrier even when memory moves
    # only one W unit. Seed the untouched storage once so partial 128-bit loads
    # preserve defined high bits and narrower loads expose zeroed tail units.
    if is_load and (expand_to_x_carrier or storage_continuation is not None):
        vector_zero = ValueRef.temporary("vector_zero")
        emits.extend(_zero_x_emits(element_byte_count))
        if storage_continuation is not None:
            storage_value = ValueRef.temporary("memory_storage")
            emits.append(
                EmitRegisterSlice(
                    source=vector_zero,
                    result=storage_value,
                    unit_count=1,
                )
            )
            memory_operands["storage"] = storage_value
        if expand_to_x_carrier:
            emits.append(
                EmitRegisterSlice(
                    source=vector_zero,
                    result=padding_value,
                    unit_offset=1,
                    unit_count=1,
                )
            )
    if not is_load and expand_to_x_carrier:
        emits.append(
            EmitRegisterSlice(
                source=source_value,
                result=descriptor_value,
                unit_count=1,
            )
        )
    if immediate_memory:
        emits.append(
            EmitDescriptorOp(
                descriptor=memory_descriptor,
                operands=memory_operands,
                results=memory_results,
                result_types=memory_result_types,
                immediates={"imm": SourceMemoryProject.static_byte_offset()},
                source_memory=source_memory,
                form=DescriptorEmitForm.OP,
            )
        )

    else:
        address_emits, address_index = _register_address_emits(
            source_memory, address_form
        )
        emits.extend(address_emits)
        memory_operands["dj"] = address_index
        emits.append(
            EmitDescriptorOp(
                descriptor=memory_descriptor,
                operands=memory_operands,
                results=memory_results,
                result_types=memory_result_types,
                source_memory=source_memory,
                form=DescriptorEmitForm.OP,
            )
        )
    if is_load and expand_to_x_carrier:
        emits.append(
            EmitRegisterConcat(
                sources=(descriptor_value, padding_value),
                result=result_value,
            )
        )

    return DescriptorRule(
        source_op=source_op,
        descriptor=memory_descriptor,
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result" if is_load else "value", value_type),
        ),
        emit=tuple(emits),
    )


def _accumulator_memory_rule(
    operation: SourceMemoryOperation,
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    volatile: bool,
) -> DescriptorRule:
    is_load = operation is SourceMemoryOperation.LOAD
    immediate_memory = address_form is _MemoryAddressForm.IMMEDIATE
    source_op = vector.vector_load if is_load else vector.vector_store
    descriptor_family = "load" if is_load else "store"
    address_family = "immediate" if immediate_memory else "register"
    descriptor_key = (
        f"amd.xdna.aie2p.{descriptor_family}.accumulator."
        f"f32x16.indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _memory_constraint(
        operation,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=4,
        vector_lane_count=64,
        minimum_alignment=64,
        immediate_offset_minimum=-512,
        immediate_offset_maximum=448,
        maximum_additional_static_byte_offset=(_ACCUMULATOR_CHUNK_BYTE_OFFSETS[-1]),
    )

    emits: list[ContractEmit] = []
    loaded_chunks: list[ValueRef] = []
    for chunk_index, chunk_byte_offset in enumerate(_ACCUMULATOR_CHUNK_BYTE_OFFSETS):
        chunk = ValueRef.temporary(f"chunk_{chunk_index}")
        if is_load:
            loaded_chunks.append(chunk)
        else:
            emits.append(
                EmitRegisterSlice(
                    source=ValueRef.operand("value"),
                    result=chunk,
                    unit_offset=chunk_index,
                    unit_count=1,
                )
            )

        memory_operands = (
            {"ptr": ValueRef.operand("view")}
            if is_load
            else {"src": chunk, "ptr": ValueRef.operand("view")}
        )
        memory_results = {"dst": chunk} if is_load else {}
        if immediate_memory:
            static_byte_offset = (
                SourceMemoryProject.static_byte_offset_plus(chunk_byte_offset)
                if chunk_byte_offset
                else SourceMemoryProject.static_byte_offset()
            )
            emits.append(
                EmitDescriptorOp(
                    descriptor=memory_descriptor,
                    operands=memory_operands,
                    results=memory_results,
                    result_types=({"dst": DescriptorResultType()} if is_load else None),
                    immediates={"imm": static_byte_offset},
                    source_memory=source_memory,
                    form=DescriptorEmitForm.OP,
                )
            )
        else:
            address_emits, address_index = _register_address_emits(
                source_memory,
                address_form,
                additional_static_byte_offset=chunk_byte_offset,
                temporary_suffix=f"_{chunk_index}",
            )
            emits.extend(address_emits)
            memory_operands["dj"] = address_index
            emits.append(
                EmitDescriptorOp(
                    descriptor=memory_descriptor,
                    operands=memory_operands,
                    results=memory_results,
                    result_types=({"dst": DescriptorResultType()} if is_load else None),
                    source_memory=source_memory,
                    form=DescriptorEmitForm.OP,
                )
            )

    if is_load:
        emits.append(
            EmitRegisterConcat(
                sources=loaded_chunks,
                result=ValueRef.result("result"),
            )
        )

    return DescriptorRule(
        source_op=source_op,
        descriptor=memory_descriptor,
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result" if is_load else "value", _F32X64_ACCUMULATOR),
        ),
        emit=tuple(emits),
    )


def _scalar_memory_rules(*, volatile: bool) -> tuple[DescriptorRule, ...]:
    return tuple(
        _memory_rule(
            operation,
            address_form,
            root_kind=root_kind,
            memory_spaces=memory_spaces,
            source_op=source_op,
            value_type=value_type,
            immediate_descriptor_key=immediate_descriptor_key,
            register_descriptor_key=register_descriptor_key,
            element_byte_count=element_byte_count,
            vector_lane_count=1,
            minimum_alignment=element_byte_count,
            immediate_offset_minimum=immediate_offset_minimum,
            immediate_offset_maximum=immediate_offset_maximum,
            volatile=volatile,
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for (
            descriptor_type,
            element_byte_count,
            immediate_offset_minimum,
            immediate_offset_maximum,
            value_types,
        ) in _SCALAR_MEMORY_SHAPES
        for value_type in value_types
        for operation, source_op, immediate_descriptor_key, register_descriptor_key in (
            (
                SourceMemoryOperation.LOAD,
                view.view_load,
                f"amd.xdna.aie2p.load.scalar.{descriptor_type}.indexed.immediate",
                f"amd.xdna.aie2p.load.scalar.{descriptor_type}.indexed.register",
            ),
            (
                SourceMemoryOperation.STORE,
                view.view_store,
                f"amd.xdna.aie2p.store.scalar.{descriptor_type}.indexed.immediate",
                f"amd.xdna.aie2p.store.scalar.{descriptor_type}.indexed.register",
            ),
        )
        for address_form in _MemoryAddressForm
    )


def _two_lane_16bit_load_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    element_type: str,
    volatile: bool,
) -> DescriptorRule:
    memory_descriptor_key = (
        "amd.xdna.aie2p.load.scalar.i16.indexed.immediate"
        if address_form is _MemoryAddressForm.IMMEDIATE
        else "amd.xdna.aie2p.load.scalar.i16.indexed.register"
    )
    if volatile:
        memory_descriptor_key = f"{memory_descriptor_key}.volatile"
    memory_descriptor = _descriptor(memory_descriptor_key)
    source_memory = _memory_constraint(
        SourceMemoryOperation.LOAD,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=2,
        vector_lane_count=2,
        minimum_alignment=2,
        immediate_offset_minimum=-16,
        immediate_offset_maximum=14,
        maximum_additional_static_byte_offset=2,
    )

    emits: list[ContractEmit] = []
    loaded_lanes: list[ValueRef] = []
    for lane_index, lane_byte_offset in enumerate((0, 2)):
        loaded_lane = ValueRef.temporary(f"lane_{lane_index}")
        loaded_lanes.append(loaded_lane)
        memory_operands = {"ptr": ValueRef.operand("view")}
        if address_form is _MemoryAddressForm.IMMEDIATE:
            static_byte_offset = (
                SourceMemoryProject.static_byte_offset_plus(lane_byte_offset)
                if lane_byte_offset
                else SourceMemoryProject.static_byte_offset()
            )
            emits.append(
                EmitDescriptorOp(
                    descriptor=memory_descriptor,
                    operands=memory_operands,
                    results={"dst": loaded_lane},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"imm": static_byte_offset},
                    source_memory=source_memory,
                    form=DescriptorEmitForm.OP,
                )
            )
        else:
            address_emits, address_index = _register_address_emits(
                source_memory,
                address_form,
                additional_static_byte_offset=lane_byte_offset,
                temporary_suffix=f"_{lane_index}",
            )
            emits.extend(address_emits)
            memory_operands["dj"] = address_index
            emits.append(
                EmitDescriptorOp(
                    descriptor=memory_descriptor,
                    operands=memory_operands,
                    results={"dst": loaded_lane},
                    result_types={"dst": DescriptorResultType()},
                    source_memory=source_memory,
                    form=DescriptorEmitForm.OP,
                )
            )

    splat = _descriptor("amd.xdna.aie2p.splat.i16x32")
    seed = ValueRef.temporary("seed")
    emits.append(
        EmitDescriptorOp(
            descriptor=splat,
            operands={"src": loaded_lanes[0]},
            results={"dst": seed},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    insert_index = ValueRef.temporary("insert_index")
    emits.append(
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": insert_index},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 1},
            form=DescriptorEmitForm.CONST,
        )
    )
    insert = _descriptor("amd.xdna.aie2p.insert.i16.register")
    emits.append(
        EmitDescriptorOp(
            descriptor=insert,
            operands={
                "s1": seed,
                "idx": insert_index,
                "src": loaded_lanes[1],
            },
            results={"dst": ValueRef.result("result")},
            copy_operands=("idx",),
            form=DescriptorEmitForm.OP,
        )
    )

    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=memory_descriptor,
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result", Vector(element_type, lanes=2)),
        ),
        emit=tuple(emits),
    )


def _two_lane_16bit_load_rules(*, volatile: bool) -> tuple[DescriptorRule, ...]:
    return tuple(
        _two_lane_16bit_load_rule(
            address_form,
            root_kind=root_kind,
            memory_spaces=memory_spaces,
            element_type=element_type,
            volatile=volatile,
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for element_type in ("i16", "f16", "bf16")
        for address_form in _MemoryAddressForm
    )


def _vector_memory_rules(*, volatile: bool) -> tuple[DescriptorRule, ...]:
    return tuple(
        _memory_rule(
            operation,
            address_form,
            root_kind=root_kind,
            memory_spaces=memory_spaces,
            source_op=(
                vector.vector_load
                if operation is SourceMemoryOperation.LOAD
                else vector.vector_store
            ),
            value_type=vector_type,
            immediate_descriptor_key=(
                f"amd.xdna.aie2p.{descriptor_family}.{shape}.indexed.immediate"
            ),
            register_descriptor_key=(
                f"amd.xdna.aie2p.{descriptor_family}.{shape}.indexed.register"
            ),
            element_byte_count=element_byte_count,
            vector_lane_count=vector_lane_count,
            minimum_alignment=width_bits // 8,
            immediate_offset_minimum=-(width_bits),
            immediate_offset_maximum=width_bits - width_bits // 8,
            volatile=volatile,
            expand_to_x_carrier=(
                width_bits < 512 and not (width_bits == 128 and element_type == "bf16")
            ),
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for (
            width_bits,
            element_type,
            descriptor_element_type,
            element_byte_count,
            vector_lane_count,
            vector_type,
        ) in _VECTOR_MEMORY_SHAPES
        for shape in (f"{descriptor_element_type}x{vector_lane_count}",)
        for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
        for descriptor_family in (
            "load.a" if operation is SourceMemoryOperation.LOAD else "store",
        )
        for address_form in _MemoryAddressForm
    )


def _accumulator_memory_rules(*, volatile: bool) -> tuple[DescriptorRule, ...]:
    return tuple(
        _accumulator_memory_rule(
            operation,
            address_form,
            root_kind=root_kind,
            memory_spaces=memory_spaces,
            volatile=volatile,
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
        for address_form in _MemoryAddressForm
    )


AIE2P_MEMORY_RULES: tuple[DescriptorRule, ...] = (
    *_scalar_memory_rules(volatile=True),
    *_two_lane_16bit_load_rules(volatile=True),
    *_vector_memory_rules(volatile=True),
    *_accumulator_memory_rules(volatile=True),
    *_scalar_memory_rules(volatile=False),
    *_two_lane_16bit_load_rules(volatile=False),
    *_vector_memory_rules(volatile=False),
    *_accumulator_memory_rules(volatile=False),
)
