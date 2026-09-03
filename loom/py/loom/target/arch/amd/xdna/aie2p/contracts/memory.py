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
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
    AIE2P_VECTOR_MEMORY_ELEMENT_TYPES,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
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

_I32 = Scalar("i32")
_OFFSET = Scalar("offset")
_VECTOR_MEMORY_SHAPES = tuple(
    (
        width_bits,
        element_type,
        element_bits // 8,
        width_bits // element_bits,
        Vector(element_type, lanes=width_bits // element_bits),
    )
    for width_bits in (128, 256, 512)
    for element_type, element_bits in AIE2P_VECTOR_MEMORY_ELEMENT_TYPES
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
) -> SourceMemoryConstraint:
    dynamic = address_form in (
        _MemoryAddressForm.DYNAMIC_ZERO_STATIC,
        _MemoryAddressForm.DYNAMIC_SMALL_STATIC,
        _MemoryAddressForm.DYNAMIC_FULL_STATIC,
    )
    if address_form is _MemoryAddressForm.IMMEDIATE:
        static_minimum = immediate_offset_minimum
        static_maximum = immediate_offset_maximum
    elif address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC:
        static_minimum = static_maximum = 0
    elif address_form is _MemoryAddressForm.DYNAMIC_SMALL_STATIC:
        static_minimum, static_maximum = -64, 63
    else:
        static_minimum, static_maximum = _I32_MIN, _I32_MAX
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
) -> tuple[tuple[EmitDescriptorOp, ...], ValueRef]:
    emits: list[EmitDescriptorOp] = []
    byte_offset = ValueRef.temporary("byte_offset")
    if address_form is _MemoryAddressForm.MATERIALIZED_STATIC:
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.materialize.static-byte-offset.i32"
                ),
                results={"dst": byte_offset},
                result_types={"dst": _OFFSET},
                immediates={"i": SourceMemoryProject.static_byte_offset()},
                source_memory=source_memory,
                form=DescriptorEmitForm.OP,
            )
        )
    elif address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC:
        byte_offset = ValueRef.source_memory_dynamic_byte_offset()
    elif address_form is _MemoryAddressForm.DYNAMIC_SMALL_STATIC:
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.add.i32.immediate"),
                operands={"s0": ValueRef.source_memory_dynamic_byte_offset()},
                results={"d0": byte_offset},
                result_types={"d0": _OFFSET},
                immediates={"imm": SourceMemoryProject.static_byte_offset()},
                source_memory=source_memory,
                source_memory_byte_offset_materializer=_byte_offset_materializer(),
                form=DescriptorEmitForm.OP,
            )
        )
    else:
        static_offset = ValueRef.temporary("static_byte_offset")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=_descriptor(
                        "amd.xdna.aie2p.materialize.static-byte-offset.i32"
                    ),
                    results={"dst": static_offset},
                    result_types={"dst": _OFFSET},
                    immediates={"i": SourceMemoryProject.static_byte_offset()},
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

    address_index = ValueRef.temporary("address_index")
    materialize_dynamic_offset = address_form is _MemoryAddressForm.DYNAMIC_ZERO_STATIC
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
    memory_operands = (
        {"ptr": ValueRef.operand("view")}
        if is_load
        else {
            "src": ValueRef.operand("value"),
            "ptr": ValueRef.operand("view"),
        }
    )
    memory_results = {"dst": ValueRef.result("result")} if is_load else {}
    emits: list[EmitDescriptorOp] = []
    if immediate_memory:
        emits.append(
            EmitDescriptorOp(
                descriptor=memory_descriptor,
                operands=memory_operands,
                results=memory_results,
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
                source_memory=source_memory,
                form=DescriptorEmitForm.OP,
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
            *(
                (Guard.operand_segment_count("indices", 0),)
                if address_form
                in (
                    _MemoryAddressForm.IMMEDIATE,
                    _MemoryAddressForm.MATERIALIZED_STATIC,
                )
                else ()
            ),
            Guard.value_type("result" if is_load else "value", value_type),
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
            value_type=_I32,
            immediate_descriptor_key=immediate_descriptor_key,
            register_descriptor_key=register_descriptor_key,
            element_byte_count=4,
            vector_lane_count=1,
            minimum_alignment=4,
            immediate_offset_minimum=-32,
            immediate_offset_maximum=28,
            volatile=volatile,
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for operation, source_op, immediate_descriptor_key, register_descriptor_key in (
            (
                SourceMemoryOperation.LOAD,
                view.view_load,
                "amd.xdna.aie2p.load.scalar.indexed.immediate",
                "amd.xdna.aie2p.load.scalar.indexed.register",
            ),
            (
                SourceMemoryOperation.STORE,
                view.view_store,
                "amd.xdna.aie2p.store.scalar.indexed.immediate",
                "amd.xdna.aie2p.store.scalar.indexed.register",
            ),
        )
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
        )
        for root_kind, memory_spaces in _MEMORY_ROOTS
        for (
            width_bits,
            element_type,
            element_byte_count,
            vector_lane_count,
            vector_type,
        ) in _VECTOR_MEMORY_SHAPES
        for shape in (f"{element_type}x{vector_lane_count}",)
        for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
        for descriptor_family in (
            "load.a" if operation is SourceMemoryOperation.LOAD else "store",
        )
        for address_form in _MemoryAddressForm
    )


AIE2P_MEMORY_RULES: tuple[DescriptorRule, ...] = (
    *_scalar_memory_rules(volatile=True),
    *_vector_memory_rules(volatile=True),
    *_scalar_memory_rules(volatile=False),
    *_vector_memory_rules(volatile=False),
)
