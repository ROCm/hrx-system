# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact physical-buffer transfers through naturally aligned integer packets."""

from dataclasses import replace

from loom.dialect.view import defs as view
from loom.target.arch.spirv.contracts.bfloat import (
    bfloat_carrier_from_i16_emits,
    bfloat_carrier_to_i16_emits,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards,
    emit_descriptor_op,
    logical_core_descriptor,
)
from loom.target.arch.spirv.contracts.memory import (
    source_memory_address_feature_guards,
    storage_buffer_address_materializer,
    storage_buffer_alignment_diagnostic,
    storage_buffer_source_memory,
)
from loom.target.arch.spirv.scalar_memory import (
    PACKED_STORAGE_BUFFER_VECTORS,
    SOURCE_STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
    StorageBufferVector,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    SourceMemoryOperation,
    SourceValueKind,
    ValueRef,
    View,
)


def _bitcast(
    source_suffix: str, result_suffix: str, source: ValueRef, result: ValueRef
) -> EmitDescriptorOp:
    return emit_descriptor_op(
        descriptor=logical_core_descriptor(
            f"spirv.op_bitcast.{source_suffix}.{result_suffix}"
        ),
        operands={"input": source},
        results={"dst": result},
        result_types=(
            {"dst": DescriptorResultType()}
            if result.kind is SourceValueKind.TEMPORARY
            else None
        ),
    )


def _packed_rule(
    scalar: StorageBufferScalar,
    packet: StorageBufferVector,
    operation: SourceMemoryOperation,
    *,
    bfloat_carrier: bool = False,
) -> DescriptorRule:
    is_load = operation is SourceMemoryOperation.LOAD
    memory_descriptor = logical_core_descriptor(
        f"spirv.op_{'load' if is_load else 'store'}.storage_buffer.{packet.suffix}"
    )
    address_materializer = storage_buffer_address_materializer(packet)
    alignment = packet.component.byte_width
    source_memory = replace(
        storage_buffer_source_memory(operation, scalar),
        minimum_alignment=alignment,
        alignment_diagnostic=storage_buffer_alignment_diagnostic(alignment),
    )
    packet_count = scalar.byte_width // packet.byte_width
    if packet_count > 1:
        # Materialize the retained source address once. Each packet then adds
        # its byte displacement and forms its own concrete vector pointer.
        address_materializer = replace(
            address_materializer,
            address=logical_core_descriptor("spirv.op_iadd.storage_buffer.byte_offset"),
        )
    source_value = ValueRef.operand("value")
    result_value = ValueRef.result("result")
    payload_suffix = "i16" if bfloat_carrier else scalar.suffix
    if bfloat_carrier:
        payload = ValueRef.temporary("bfloat_bits")
        source_value = payload
        result_value = payload

    def transfer(value: ValueRef, address: ValueRef) -> EmitDescriptorOp:
        # Both packets of a byte-aligned 64-bit value retain the enclosing
        # source footprint. Each issued access is a subset of that interval.
        return emit_descriptor_op(
            descriptor=memory_descriptor,
            operands={"ptr": address} | ({} if is_load else {"value": value}),
            results={"dst": value} if is_load else None,
            result_types={"dst": DescriptorResultType()} if is_load else None,
            source_memory=source_memory,
            source_memory_address_materializer=address_materializer,
        )

    emits: list[EmitDescriptorOp] = []
    if bfloat_carrier and not is_load:
        emits.extend(
            bfloat_carrier_to_i16_emits(ValueRef.operand("value"), source_value)
        )
    if packet_count == 1:
        packed = ValueRef.temporary("packed_value")
        memory = transfer(packed, ValueRef.source_memory_address())
        bitcast = (
            _bitcast(packet.suffix, payload_suffix, packed, result_value)
            if is_load
            else _bitcast(payload_suffix, packet.suffix, source_value, packed)
        )
        emits.extend((memory, bitcast) if is_load else (bitcast, memory))
    else:
        # SPIR-V vectors have at most four components. A byte-aligned 64-bit
        # payload therefore uses two four-byte packets, without overreading
        # either end or imposing Int64 on an F64 value.
        words = ValueRef.temporary("words")
        if not is_load:
            emits.append(_bitcast(payload_suffix, "v2i32", source_value, words))
        word_values = []
        for part in range(packet_count):
            displacement = ValueRef.temporary(f"packet_displacement_{part}")
            address = ValueRef.temporary(f"packet_address_{part}")
            emits.extend(
                (
                    EmitDescriptorOp(
                        descriptor=logical_core_descriptor(
                            "spirv.op_constant.offset64"
                        ),
                        results={"dst": displacement},
                        result_types={"dst": DescriptorResultType()},
                        immediates={"offset64_value": part * packet.byte_width},
                        form=DescriptorEmitForm.CONST,
                    ),
                    emit_descriptor_op(
                        descriptor=logical_core_descriptor(
                            f"spirv.op_ptr_access_chain.storage_buffer.{packet.suffix}.byte_offset"
                        ),
                        operands={
                            "base": ValueRef.source_memory_address(),
                            "byte_offset": displacement,
                        },
                        results={"ptr": address},
                        result_types={"ptr": DescriptorResultType()},
                        source_memory=source_memory,
                        source_memory_address_materializer=address_materializer,
                    ),
                )
            )
            packed = ValueRef.temporary(f"packed_value_{part}")
            word = ValueRef.temporary(f"word_{part}")
            word_values.append(word)
            if is_load:
                emits.extend(
                    (
                        transfer(packed, address),
                        _bitcast(packet.suffix, "i32", packed, word),
                    )
                )
            else:
                emits.extend(
                    (
                        emit_descriptor_op(
                            descriptor=logical_core_descriptor(
                                "spirv.op_composite_extract.v2i32.i32"
                            ),
                            operands={"composite": words},
                            results={"dst": word},
                            result_types={"dst": DescriptorResultType()},
                            immediates={"component_index": part},
                        ),
                        _bitcast("i32", packet.suffix, word, packed),
                        transfer(packed, address),
                    )
                )
        if is_load:
            emits.extend(
                (
                    emit_descriptor_op(
                        descriptor=logical_core_descriptor(
                            "spirv.op_composite_construct.v2i32"
                        ),
                        operands={
                            f"component{part}": value
                            for part, value in enumerate(word_values)
                        },
                        results={"dst": words},
                        result_types={"dst": DescriptorResultType()},
                    ),
                    _bitcast("v2i32", payload_suffix, words, result_value),
                )
            )
    if bfloat_carrier and is_load:
        emits.extend(
            bfloat_carrier_from_i16_emits(result_value, ValueRef.result("result"))
        )
    return DescriptorRule(
        source_op=view.view_load if is_load else view.view_store,
        descriptor=memory_descriptor,
        guards=(
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type(
                "result" if is_load else "value", Scalar(scalar.source_type)
            ),
            *source_memory_address_feature_guards(address_materializer),
            *descriptor_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=tuple(emits),
    )


def packed_storage_buffer_rules() -> tuple[DescriptorRule, ...]:
    # Native scalar transfers precede these cases. Within this family, prefer
    # the widest supported component allowed by the retained alignment facts.
    packets = {
        (packet.byte_width, packet.component.byte_width): packet
        for packet in PACKED_STORAGE_BUFFER_VECTORS
    }
    rules = []
    for scalar in SOURCE_STORAGE_BUFFER_SCALARS:
        for alignment in (4, 2, 1):
            if alignment >= scalar.byte_width:
                continue
            packet = packets[(min(scalar.byte_width, alignment * 4), alignment)]
            for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE):
                rules.append(_packed_rule(scalar, packet, operation))
                if scalar.source_type == "bf16":
                    rules.append(
                        _packed_rule(scalar, packet, operation, bfloat_carrier=True)
                    )
    return tuple(rules)
