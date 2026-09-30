# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P packet-converting memory selection rules."""

from __future__ import annotations

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.bfp import (
    BFP_ENCODE_GUARDS,
    bfp_encode_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    BF16_CONVERSION_ROUNDING,
)
from loom.target.arch.amd.xdna.aie2p.contracts.memory import (
    _descriptor,
    _memory_constraint,
    _memory_roots,
    _MemoryAddressForm,
    _register_address_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_conversion import (
    FLOAT_PACKET_LANE_COUNTS,
    I4_UNPACK_SOURCE_LANE_COUNTS,
    INTEGER_PACK_INSTRUCTIONS,
    INTEGER_WIDEN_INSTRUCTIONS,
    IntegerPackInstruction,
    IntegerWidenInstruction,
    integer_pack_state_emits,
    integer_unpack_state_emits,
    integer_widen_result_emits,
    integer_widen_state_emits,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    SourceNode,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor


def _fused_memory_access_emits(
    address_form: _MemoryAddressForm,
    memory_descriptor: Descriptor,
    source_memory: SourceMemoryConstraint,
    operands: dict[str, ValueRef],
    *,
    access_preamble: tuple[ContractEmit, ...] = (),
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, DescriptorResultType] | None = None,
    additional_static_byte_offset: int = 0,
    temporary_suffix: str = "",
) -> tuple[ContractEmit, ...]:
    """Builds one fused memory access after realizing its canonical address."""

    access_operands = dict(operands)
    address_emits: tuple[EmitDescriptorOp, ...] = ()
    immediates: dict[str, SourceMemoryProject] = {}
    if address_form is _MemoryAddressForm.IMMEDIATE:
        immediates["imm"] = (
            SourceMemoryProject.static_byte_offset_plus(additional_static_byte_offset)
            if additional_static_byte_offset
            else SourceMemoryProject.static_byte_offset()
        )
    else:
        address_emits, address_index = _register_address_emits(
            source_memory,
            address_form,
            additional_static_byte_offset=additional_static_byte_offset,
            temporary_suffix=temporary_suffix,
        )
        access_operands["dj"] = address_index
    return (
        *address_emits,
        *access_preamble,
        EmitDescriptorOp(
            descriptor=memory_descriptor,
            operands=access_operands,
            results={} if results is None else results,
            result_types=result_types,
            immediates=immediates,
            source_memory=source_memory,
            form=DescriptorEmitForm.OP,
        ),
    )


def _fused_memory_constraint(
    operation: SourceMemoryOperation,
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    element_byte_count: int,
    vector_lane_count: int,
    memory_width_bits: int,
) -> SourceMemoryConstraint:
    # Packet immediates are signed offsets stepped by the access byte width.
    # Thus c9s_step32 spans -256..224 bytes for a 256-bit access and
    # c10s_step64 spans -512..448 bytes for a 512-bit access.
    return _memory_constraint(
        operation,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        minimum_alignment=memory_width_bits // 8,
        immediate_offset_minimum=-memory_width_bits,
        immediate_offset_maximum=memory_width_bits - memory_width_bits // 8,
    )


def _fused_i4_unpack_load_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    source_op: Op,
    source_kind: str,
    source_lane_count: int,
    volatile: bool,
) -> DescriptorRule:
    result_lane_count = source_lane_count * 2
    source_type = Vector("i8", lanes=source_lane_count)
    result_type = Vector("i8", lanes=result_lane_count)
    signedness = "unsigned" if source_kind == "u" else "signed"
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.load.unpack.{source_kind}4x{result_lane_count}.to."
        f"{source_kind}8x{result_lane_count}.configured.indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _fused_memory_constraint(
        SourceMemoryOperation.LOAD,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=1,
        vector_lane_count=source_lane_count,
        memory_width_bits=source_lane_count * 8,
    )
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_unique_user(
                "convert",
                source_op=source_op,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("source"),
                guards=(
                    Guard.value_type("source", source_type),
                    Guard.value_type("result", result_type),
                    Guard.attr_kind("width", "i64"),
                    Guard.i64_range("width", 4, 4),
                ),
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result", source_type),
        ),
        emit=_fused_memory_access_emits(
            address_form,
            memory_descriptor,
            source_memory,
            {"ptr": ValueRef.operand("view")},
            access_preamble=integer_unpack_state_emits(),
            results={"dst": ValueRef.result("result", source_node="convert")},
        ),
        priority=1,
        report_key=(
            f"native_memory_load_{signedness}_i4x{result_lane_count}_to_"
            f"i8x{result_lane_count}"
        ),
    )


def _fused_float_load_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    lane_count: int,
    volatile: bool,
) -> DescriptorRule:
    source_type = Vector("bf16", lanes=lane_count)
    result_type = Vector("f32", lanes=lane_count)
    source_shape = f"bf16x{lane_count}"
    result_shape = f"f32x{lane_count}"
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.load.convert.{source_shape}.to.{result_shape}."
        f"indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _fused_memory_constraint(
        SourceMemoryOperation.LOAD,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=2,
        vector_lane_count=lane_count,
        memory_width_bits=lane_count * 16,
    )
    converted_result = ValueRef.result("result", source_node="convert")
    native_result = converted_result
    result_types = None
    output_emits: tuple[ContractEmit, ...] = ()
    if lane_count == 16:
        native_result = ValueRef.temporary("converted_accumulator")
        result_types = {"op": DescriptorResultType()}
        output_emits = (
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.move.accumulator512.to.vector512"
                ),
                operands={"src": native_result},
                results={"dst": converted_result},
                form=DescriptorEmitForm.OP,
            ),
        )
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_unique_user(
                "convert",
                source_op=vector.vector_extf,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("input"),
                guards=(
                    Guard.value_type("input", source_type),
                    Guard.value_type("result", result_type),
                ),
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result", source_type),
        ),
        emit=(
            *_fused_memory_access_emits(
                address_form,
                memory_descriptor,
                source_memory,
                {"ptr": ValueRef.operand("view")},
                results={"op": native_result},
                result_types=result_types,
            ),
            *output_emits,
        ),
        priority=1,
        report_key=f"native_memory_load_{source_shape}_to_{result_shape}",
    )


def _fused_bfp_encode_load_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    volatile: bool,
) -> DescriptorRule:
    source_type = Vector("bf16", lanes=64)
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.load.convert.bf16x32.to.f32x32.indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _memory_constraint(
        SourceMemoryOperation.LOAD,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=2,
        vector_lane_count=64,
        minimum_alignment=64,
        immediate_offset_minimum=-512,
        immediate_offset_maximum=448,
        maximum_additional_static_byte_offset=64,
    )
    emits: list[ContractEmit] = []
    float_halves = (
        ValueRef.temporary("float_half_0"),
        ValueRef.temporary("float_half_1"),
    )
    for half_index, float_half in enumerate(float_halves):
        emits.extend(
            _fused_memory_access_emits(
                address_form,
                memory_descriptor,
                source_memory,
                {"ptr": ValueRef.operand("view")},
                results={"op": float_half},
                result_types={"op": DescriptorResultType()},
                additional_static_byte_offset=half_index * 64,
                temporary_suffix=f"_{half_index}",
            )
        )
    emits.extend(
        bfp_encode_emits(float_halves, ValueRef.result("result", source_node="encode"))
    )
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_unique_user(
                "encode",
                source_op=vector.vector_encode,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("source"),
                guards=BFP_ENCODE_GUARDS,
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result", source_type),
        ),
        emit=tuple(emits),
        priority=1,
        report_key="native_memory_load_bf16x64_to_bfp16ebs8",
    )


def _fused_integer_widen_load_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    source_op: Op,
    signedness: str,
    instruction: IntegerWidenInstruction,
    volatile: bool,
) -> DescriptorRule:
    source_type = Vector(instruction.input_element, lanes=instruction.native_lane_count)
    result_type = Vector(
        instruction.result_element, lanes=instruction.native_lane_count
    )
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.load.widen.{instruction.physical_shape}."
        f"{signedness}.configured.indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _fused_memory_constraint(
        SourceMemoryOperation.LOAD,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=int(instruction.input_element[1:]) // 8,
        vector_lane_count=instruction.native_lane_count,
        memory_width_bits=instruction.memory_width_bits,
    )
    shift, state_emits = integer_widen_state_emits(instruction.ups_mode)
    converted_result = ValueRef.result("result", source_node="convert")
    native_result, output_emits = integer_widen_result_emits(
        instruction,
        converted_result,
    )
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_unique_user(
                "convert",
                source_op=source_op,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("input"),
                guards=(
                    Guard.value_type("input", source_type),
                    Guard.value_type("result", result_type),
                ),
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("result", source_type),
        ),
        emit=(
            *_fused_memory_access_emits(
                address_form,
                memory_descriptor,
                source_memory,
                {
                    "su": shift,
                    "ptr": ValueRef.operand("view"),
                },
                access_preamble=state_emits,
                results={"dst": native_result},
                result_types=(
                    None
                    if instruction.direct_accumulator_result
                    else {"dst": DescriptorResultType()}
                ),
            ),
            *output_emits,
        ),
        priority=1,
        report_key=(
            f"native_memory_load_{signedness}_{instruction.input_element}x"
            f"{instruction.native_lane_count}_to_{instruction.result_element}x"
            f"{instruction.native_lane_count}"
        ),
    )


def _fused_float_store_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    lane_count: int,
    volatile: bool,
) -> DescriptorRule:
    source_type = Vector("f32", lanes=lane_count)
    result_type = Vector("bf16", lanes=lane_count)
    source_shape = f"f32x{lane_count}"
    result_shape = f"bf16x{lane_count}"
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.store.convert.{source_shape}.to.{result_shape}."
        f"indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _fused_memory_constraint(
        SourceMemoryOperation.STORE,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=2,
        vector_lane_count=lane_count,
        memory_width_bits=lane_count * 16,
    )
    native_source = ValueRef.operand("input", source_node="convert")
    access_preamble: list[ContractEmit] = []
    if lane_count == 16:
        accumulator_source = ValueRef.temporary("converted_accumulator")
        access_preamble.append(
            EmitDescriptorOp(
                descriptor=_descriptor(
                    "amd.xdna.aie2p.move.vector512.to.accumulator512"
                ),
                operands={"src": native_source},
                results={"dst": accumulator_source},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            )
        )
        native_source = accumulator_source
    access_preamble.append(
        EmitDescriptorOp(
            descriptor=_descriptor("amd.xdna.aie2p.state.rounding.immediate"),
            immediates={"i": BF16_CONVERSION_ROUNDING},
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_definition(
                "convert",
                source_op=vector.vector_fptrunc,
                parent_operand=ValueRef.operand("value"),
                node_result=ValueRef.result("result"),
                guards=(
                    Guard.value_type("input", source_type),
                    Guard.value_type("result", result_type),
                ),
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("value", result_type),
        ),
        emit=_fused_memory_access_emits(
            address_form,
            memory_descriptor,
            source_memory,
            {
                "src": native_source,
                "ptr": ValueRef.operand("view"),
            },
            access_preamble=tuple(access_preamble),
        ),
        priority=1,
        report_key=f"native_memory_store_{source_shape}_to_{result_shape}",
    )


def _fused_integer_pack_store_rule(
    address_form: _MemoryAddressForm,
    *,
    root_kind: SourceMemoryRootKind,
    memory_spaces: tuple[str, ...],
    pack_instruction: IntegerPackInstruction,
    volatile: bool,
) -> DescriptorRule:
    source_type = Vector(
        pack_instruction.input_element,
        lanes=pack_instruction.native_lane_count,
    )
    result_type = Vector(
        pack_instruction.result_element,
        lanes=pack_instruction.result_lanes,
    )
    address_family = (
        "immediate" if address_form is _MemoryAddressForm.IMMEDIATE else "register"
    )
    descriptor_key = (
        f"amd.xdna.aie2p.store.pack.{pack_instruction.physical_width}.trunc."
        f"configured.indexed.{address_family}"
    )
    if volatile:
        descriptor_key = f"{descriptor_key}.volatile"
    memory_descriptor = _descriptor(descriptor_key)
    source_memory = _fused_memory_constraint(
        SourceMemoryOperation.STORE,
        address_form,
        root_kind=root_kind,
        memory_spaces=memory_spaces,
        element_byte_count=int(pack_instruction.result_element[1:]) // 8,
        vector_lane_count=pack_instruction.result_lanes,
        memory_width_bits=pack_instruction.memory_width_bits,
    )
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=memory_descriptor,
        source_nodes=(
            SourceNode.adjacent_definition(
                "convert",
                source_op=pack_instruction.source_op,
                parent_operand=ValueRef.operand("value"),
                node_result=ValueRef.result("result"),
                guards=(
                    Guard.value_type(pack_instruction.source_field, source_type),
                    Guard.value_type("result", result_type),
                    *(
                        (
                            Guard.attr_kind("width", "i64"),
                            Guard.i64_range(
                                "width",
                                pack_instruction.bit_width,
                                pack_instruction.bit_width,
                            ),
                        )
                        if pack_instruction.bit_width is not None
                        else ()
                    ),
                ),
            ),
        ),
        guards=(
            *(
                (Guard.instance_flags_has_all("memory_flags", "volatile"),)
                if volatile
                else ()
            ),
            Guard.value_type("value", result_type),
        ),
        emit=_fused_memory_access_emits(
            address_form,
            memory_descriptor,
            source_memory,
            {
                "src": ValueRef.operand(
                    pack_instruction.source_field,
                    source_node="convert",
                ),
                "ptr": ValueRef.operand("view"),
            },
            access_preamble=integer_pack_state_emits(pack_instruction.pack_size),
        ),
        priority=1,
        report_key=f"native_memory_store_{pack_instruction.report_key}",
    )


def _fused_memory_rules(*, volatile: bool) -> tuple[DescriptorRule, ...]:
    return (
        *(
            _fused_i4_unpack_load_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                source_op=source_op,
                source_kind=source_kind,
                source_lane_count=source_lane_count,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.LOAD)
            for source_op, source_kind in (
                (vector.vector_bitunpacku, "u"),
                (vector.vector_bitunpacks, "s"),
            )
            for source_lane_count in I4_UNPACK_SOURCE_LANE_COUNTS
            for address_form in _MemoryAddressForm
        ),
        *(
            _fused_float_load_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                lane_count=lane_count,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.LOAD)
            for lane_count in FLOAT_PACKET_LANE_COUNTS
            for address_form in _MemoryAddressForm
        ),
        *(
            _fused_bfp_encode_load_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.LOAD)
            for address_form in _MemoryAddressForm
        ),
        *(
            _fused_integer_widen_load_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                source_op=source_op,
                signedness=signedness,
                instruction=instruction,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.LOAD)
            for source_op, signedness in (
                (vector.vector_extui, "unsigned"),
                (vector.vector_extsi, "signed"),
            )
            for instruction in INTEGER_WIDEN_INSTRUCTIONS
            for address_form in _MemoryAddressForm
        ),
        *(
            _fused_float_store_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                lane_count=lane_count,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.STORE)
            for lane_count in FLOAT_PACKET_LANE_COUNTS
            for address_form in _MemoryAddressForm
        ),
        *(
            _fused_integer_pack_store_rule(
                address_form,
                root_kind=root_kind,
                memory_spaces=memory_spaces,
                pack_instruction=pack_instruction,
                volatile=volatile,
            )
            for root_kind, memory_spaces in _memory_roots(SourceMemoryOperation.STORE)
            for pack_instruction in INTEGER_PACK_INSTRUCTIONS
            for address_form in _MemoryAddressForm
        ),
    )


AIE2P_PACKET_MEMORY_RULES: tuple[DescriptorRule, ...] = (
    *_fused_memory_rules(volatile=True),
    *_fused_memory_rules(volatile=False),
)
