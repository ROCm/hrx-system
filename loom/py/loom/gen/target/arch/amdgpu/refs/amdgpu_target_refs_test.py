# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from loom.gen.target.arch.amdgpu.refs import amdgpu_target_refs
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_MUBUF,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
)
from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import (
    AsmForm,
    Descriptor,
    DescriptorSet,
    EncodingFieldValue,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    Resource,
    ResourceFlag,
    ResourceKind,
    ScheduleClass,
)

_RESOURCE_SALU = "salu"
_RESOURCE_VALU = "valu"
_RESOURCE_MATRIX = "matrix"
_SCHEDULE_NONE = "none"
_SCHEDULE_SALU = "salu"
_SCHEDULE_VALU = "valu"
_SCHEDULE_MATRIX = "matrix"


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(f"ValueError message {exc!s} did not match {match}") from exc
    else:
        raise AssertionError("expected ValueError")


def _descriptor(
    key: str,
    asm_forms: tuple[AsmForm, ...] = (),
    *,
    schedule_class: str = _SCHEDULE_NONE,
    encoding_format_id: int = 0,
    immediates: tuple[Immediate, ...] = (),
    instruction_classes: tuple[InstructionClass, ...] = (),
    operands: tuple[Operand, ...] = (),
    encoding_field_values: tuple[EncodingFieldValue, ...] = (),
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=None,
        semantic_tag=None,
        operands=operands,
        schedule_class=schedule_class,
        asm_forms=asm_forms,
        encoding_format_id=encoding_format_id,
        immediates=immediates,
        instruction_classes=instruction_classes,
        encoding_field_values=encoding_field_values,
    )


def _descriptor_set(*descriptors: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key="amdgpu.test.core",
        target_key="amdgpu",
        feature_key=None,
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=0,
        reg_classes=(),
        resources=(
            Resource(
                _RESOURCE_SALU,
                capacity_per_cycle=1,
                kind=ResourceKind.SCALAR_ALU,
            ),
            Resource(
                _RESOURCE_VALU,
                capacity_per_cycle=1,
                kind=ResourceKind.VECTOR_ALU,
                flags=(ResourceFlag.VECTOR_ISSUE,),
            ),
            Resource(
                _RESOURCE_MATRIX,
                capacity_per_cycle=1,
                kind=ResourceKind.MATRIX,
                flags=(
                    ResourceFlag.VECTOR_ISSUE,
                    ResourceFlag.MATRIX_COEXECUTION_SOURCE,
                ),
            ),
        ),
        schedule_classes=(
            ScheduleClass(
                _SCHEDULE_NONE,
                latency_kind=LatencyKind.EXACT,
                model_quality=ModelQuality.EXACT,
            ),
            ScheduleClass(
                _SCHEDULE_SALU,
                latency_kind=LatencyKind.EXACT,
                model_quality=ModelQuality.EXACT,
                issue_uses=(IssueUse(_RESOURCE_SALU, cycles=1, units=1),),
            ),
            ScheduleClass(
                _SCHEDULE_VALU,
                latency_kind=LatencyKind.EXACT,
                model_quality=ModelQuality.EXACT,
                issue_uses=(IssueUse(_RESOURCE_VALU, cycles=1, units=1),),
            ),
            ScheduleClass(
                _SCHEDULE_MATRIX,
                latency_kind=LatencyKind.EXACT,
                model_quality=ModelQuality.EXACT,
                issue_uses=(IssueUse(_RESOURCE_MATRIX, cycles=1, units=1),),
            ),
        ),
        descriptors=descriptors,
    )


def _spill_lowering_descriptor(
    descriptor_key: str,
    *,
    include_address_offset: bool = True,
    include_implicit_m0: bool = False,
) -> Descriptor:
    operand_count, result_count = amdgpu_target_refs._SPILL_LOWERING_DESCRIPTOR_SHAPES[descriptor_key]
    operands = (
        *(Operand(f"result{index}", OperandRole.RESULT, ()) for index in range(result_count)),
        *(Operand(f"operand{index}", OperandRole.OPERAND, ()) for index in range(operand_count)),
    )
    if include_implicit_m0:
        operands = (
            *operands,
            Operand(
                "m0",
                OperandRole.RESOURCE,
                (),
                flags=(OperandFlag.IMPLICIT,),
            ),
        )
    address_offset_immediates = (
        (
            Immediate(
                "offset",
                ImmediateKind.UNSIGNED,
                bit_width=12,
                unsigned_max=4095,
            ),
        )
        if include_address_offset and descriptor_key in amdgpu_target_refs._SPILL_LOWERING_SCRATCH_DESCRIPTOR_KEYS
        else ()
    )
    m0_immediates = (Immediate("imm32", ImmediateKind.UNSIGNED),) if descriptor_key == amdgpu_target_refs._SPILL_LOWERING_M0_DESCRIPTOR_KEY else ()
    return _descriptor(
        descriptor_key,
        immediates=(*address_offset_immediates, *m0_immediates),
        operands=operands,
    )


def _valid_spill_lowering_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        _spill_lowering_descriptor(descriptor_key)
        for descriptor_key in (
            *amdgpu_target_refs._SPILL_LOWERING_SCRATCH_DESCRIPTOR_KEYS,
            *amdgpu_target_refs._SPILL_LOWERING_HELPER_DESCRIPTOR_KEYS,
        )
    )


def _valid_contract_descriptors() -> tuple[Descriptor, ...]:
    rel32_operands = (
        Operand("dst", OperandRole.RESULT, ()),
        Operand("lhs", OperandRole.OPERAND, ()),
    )
    rel32_immediates = (
        Immediate(
            "symbol",
            ImmediateKind.ORDINAL,
            flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
            encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["LITERAL"],
        ),
        Immediate("byte_offset", ImmediateKind.UNSIGNED),
    )
    flat_load_descriptors = tuple(_descriptor(descriptor_key, (AsmForm(operands=("addr",)),)) for descriptor_key in amdgpu_target_refs._FLAT_LOAD_DESCRIPTOR_KEYS)
    flat_store_descriptors = tuple(_descriptor(descriptor_key, (AsmForm(operands=("addr", "value")),)) for descriptor_key in amdgpu_target_refs._FLAT_STORE_DESCRIPTOR_KEYS)
    return (
        _descriptor(
            "amdgpu.global_load_b32_saddr",
            (AsmForm(operands=("vaddr", "saddr")),),
        ),
        _descriptor(
            "amdgpu.global_load_b64_saddr",
            (AsmForm(operands=("vaddr", "saddr", "m0")),),
        ),
        *flat_load_descriptors,
        *flat_store_descriptors,
        *_valid_spill_lowering_descriptors(),
        _descriptor(
            "amdgpu.s_add_u32.rhs_symbol_rel32_lo",
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
            operands=rel32_operands,
            immediates=rel32_immediates,
        ),
        _descriptor(
            "amdgpu.s_addc_u32.rhs_symbol_rel32_hi",
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
            operands=rel32_operands,
            immediates=rel32_immediates,
        ),
    )


def test_target_refs_header_is_constant_fragment() -> None:
    source = amdgpu_target_refs._emit_tables_header()

    assert "typedef " not in source
    assert "extern " not in source
    assert "loom_amdgpu_descriptor_ref_ordinal" not in source
    assert "loom/codegen/low/descriptors.h" not in source
    assert "#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32" in source


def test_target_ref_tables_use_prebuilt_descriptor_sets() -> None:
    descriptor_set_info = sorted_descriptor_set_infos()[0]
    descriptor_set = replace(
        _descriptor_set(*_valid_contract_descriptors()),
        key=descriptor_set_info.key,
    )

    tables = amdgpu_target_refs._materialize_descriptor_ref_tables(
        (descriptor_set_info,),
        {descriptor_set_info.key: descriptor_set},
    )

    assert len(tables) == 1
    assert tables[0].descriptor_set_key == descriptor_set_info.key
    assert tables[0].descriptor_set_ordinal == amdgpu_descriptor_set_ordinal(descriptor_set_info.key)


def test_descriptor_trait_names_include_resource_and_encoding_facts() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_mov_b32_dpp",
            schedule_class=_SCHEDULE_VALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
    )


def test_descriptor_trait_names_include_destination_selection_forwarding() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_cvt_pk_fp8_f32.high",
            schedule_class=_SCHEDULE_VALU,
            encoding_field_values=(
                EncodingFieldValue(
                    AMDGPU_ENCODING_FIELD_IDS["OP_SEL"],
                    amdgpu_target_refs._DESTINATION_OP_SEL_MASK,
                ),
            ),
        ),
        _descriptor(
            "amdgpu.v_cvt_pk_fp8_f32.low",
            schedule_class=_SCHEDULE_VALU,
            encoding_field_values=(EncodingFieldValue(AMDGPU_ENCODING_FIELD_IDS["OP_SEL"], 0),),
        ),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING" in amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING" not in amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1])


def test_descriptor_trait_names_include_memory_and_ref_facts() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.global_load_b32",
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_MUBUF,
        ),
        _descriptor("amdgpu.v_exp_f32", schedule_class=_SCHEDULE_VALU),
        _descriptor("amdgpu.v_readfirstlane_b32", schedule_class=_SCHEDULE_SALU),
        _descriptor(
            "amdgpu.v_wmma",
            schedule_class=_SCHEDULE_MATRIX,
            instruction_classes=(InstructionClass.WMMA,),
        ),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[2]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[3]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_COEXECUTION_SOURCE",
    )


def test_vector_issue_traits_follow_issue_resource_contracts() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_wmma",
            schedule_class=_SCHEDULE_MATRIX,
            instruction_classes=(InstructionClass.WMMA,),
        ),
        _descriptor("amdgpu.v_add", schedule_class=_SCHEDULE_VALU),
        _descriptor("amdgpu.s_nop", schedule_class=_SCHEDULE_SALU),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_COEXECUTION_SOURCE",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
    )
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE" not in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[2]))


def test_ldsdma_instruction_class_advances_vector_issue() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.cluster_load_async_to_lds_b32",
            schedule_class=_SCHEDULE_SALU,
            instruction_classes=(InstructionClass.LDSDMA,),
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
    )


def test_descriptor_trait_names_include_xcnt_implicit_drain_families() -> None:
    descriptor_set = _descriptor_set(
        _descriptor("amdgpu.s_trap"),
        _descriptor("amdgpu.s_getreg_b32_cluster_workgroup_flat_id"),
        _descriptor("amdgpu.s_setreg_b32"),
        _descriptor("amdgpu.s_sendmsg_rtn_b32"),
        _descriptor("amdgpu.s_barrier_wait_all"),
        _descriptor("amdgpu.s_barrier_signal_all"),
        _descriptor("amdgpu.v_mov_b32"),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    for descriptor in descriptor_set.descriptors[:-1]:
        assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN" in amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor)
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN" not in amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[-1])


def test_vmem_result_order_classes_distinguish_memory_families() -> None:
    result = Operand("dst", OperandRole.RESULT, ())
    non_memory = _descriptor("amdgpu.v_mov_b32", operands=(result,))
    buffer_load = _descriptor(
        "amdgpu.buffer_load_dword",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_MUBUF,
        operands=(result,),
    )
    flat_load = _descriptor(
        "amdgpu.flat_load_dword",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT"],
        operands=(result,),
    )
    image_load = _descriptor(
        "amdgpu.image_load",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS["ENC_MIMG"],
        operands=(result,),
    )
    buffer_store = _descriptor(
        "amdgpu.buffer_store_dword",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_MUBUF,
    )

    assert amdgpu_target_refs._descriptor_vmem_result_order_class_name(non_memory) == "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"
    assert amdgpu_target_refs._descriptor_vmem_result_order_class_name(buffer_load) == "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER"
    assert amdgpu_target_refs._descriptor_vmem_result_order_class_name(flat_load) == "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER"
    assert amdgpu_target_refs._descriptor_vmem_result_order_class_name(image_load) == "LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN"
    assert amdgpu_target_refs._descriptor_vmem_result_order_class_name(buffer_store) == "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"


def test_reg_class_trait_names_classify_special_register_files() -> None:
    assert amdgpu_target_refs._reg_class_trait_names("amdgpu.agpr") == ("LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR",)
    assert amdgpu_target_refs._reg_class_trait_names("amdgpu.m0") == ("LOOM_AMDGPU_REG_CLASS_TRAIT_M0",)
    assert amdgpu_target_refs._reg_class_trait_names("amdgpu.vcc") == ("LOOM_AMDGPU_REG_CLASS_TRAIT_VCC",)
    assert amdgpu_target_refs._reg_class_trait_names("amdgpu.vgpr") == ()


def test_descriptor_traits_reject_missing_schedule_class() -> None:
    descriptor_set = _descriptor_set(_descriptor("amdgpu.test", schedule_class="missing"))
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    with _raises_value_error("descriptor 'amdgpu.test' references missing schedule class 'missing'"):
        amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])


def test_descriptor_traits_reject_missing_issue_resource() -> None:
    descriptor_set = DescriptorSet(
        key="amdgpu.test.core",
        target_key="amdgpu",
        feature_key=None,
        c_header_path=Path("test.h"),
        c_source_path=Path("test.c"),
        header_guard="TEST_H_",
        public_header="test.h",
        function_name="test",
        c_table_prefix="test",
        c_enum_prefix="TEST",
        generator_version=0,
        reg_classes=(),
        resources=(),
        schedule_classes=(
            ScheduleClass(
                _SCHEDULE_VALU,
                latency_kind=LatencyKind.EXACT,
                model_quality=ModelQuality.EXACT,
                issue_uses=(IssueUse("missing", cycles=1, units=1),),
            ),
        ),
        descriptors=(_descriptor("amdgpu.test", schedule_class=_SCHEDULE_VALU),),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    with _raises_value_error("schedule class 'valu' references missing resource 'missing'"):
        amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])


def test_descriptor_traits_reject_duplicate_matrix_source_issue_use() -> None:
    descriptor_set = _descriptor_set(_descriptor("amdgpu.test", schedule_class=_SCHEDULE_MATRIX))
    descriptor_set = replace(
        descriptor_set,
        schedule_classes=tuple(
            replace(
                schedule_class,
                issue_uses=(
                    IssueUse(_RESOURCE_MATRIX, cycles=1, units=1),
                    IssueUse(_RESOURCE_MATRIX, cycles=1, units=1),
                ),
            )
            if schedule_class.name == _SCHEDULE_MATRIX
            else schedule_class
            for schedule_class in descriptor_set.schedule_classes
        ),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    with _raises_value_error("descriptor 'amdgpu.test' must use at most one matrix coexecution source resource"):
        amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])


def test_descriptor_traits_reject_unclassified_matrix_source() -> None:
    descriptor_set = _descriptor_set(_descriptor("amdgpu.test", schedule_class=_SCHEDULE_MATRIX))
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    with _raises_value_error("descriptor 'amdgpu.test' matrix coexecution source must belong to exactly one WMMA or SWMMAC instruction class"):
        amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])


def test_descriptor_immediate_slots_publish_sdwa_dst_sel() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_mov_b32_sdwa",
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
            immediates=(
                Immediate("dst_unused", ImmediateKind.UNSIGNED),
                Immediate(
                    "dst_sel",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["DST_SEL"],
                ),
            ),
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
    descriptor = descriptor_set.descriptors[0]
    trait_names = amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor)

    assert amdgpu_target_refs._descriptor_sdwa_dst_sel_immediate_slot(descriptor_set, descriptor, trait_names) == 1


def test_descriptor_immediate_slots_reject_duplicate_sdwa_dst_sel() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_bad_sdwa",
            immediates=(
                Immediate(
                    "dst_sel",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["DST_SEL"],
                ),
                Immediate(
                    "dst_sel2",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["DST_SEL"],
                ),
            ),
        )
    )

    with _raises_value_error("descriptor 'amdgpu.v_bad_sdwa' has multiple SDWA dst_sel immediates"):
        amdgpu_target_refs._descriptor_sdwa_dst_sel_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
            (),
        )


def test_descriptor_immediate_slots_reject_sdwa_without_dst_sel() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_bad_sdwa",
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
    descriptor = descriptor_set.descriptors[0]
    trait_names = amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor)

    with _raises_value_error("SDWA descriptor 'amdgpu.v_bad_sdwa' has no dst_sel immediate"):
        amdgpu_target_refs._descriptor_sdwa_dst_sel_immediate_slot(
            descriptor_set,
            descriptor,
            trait_names,
        )


def test_descriptor_immediate_slots_publish_literal() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_mov_b32",
            immediates=(
                Immediate(
                    "imm32",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["LITERAL"],
                ),
            ),
        )
    )

    assert (
        amdgpu_target_refs._descriptor_literal_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )
        == 0
    )


def test_descriptor_immediate_slots_publish_address_offset() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.scratch_load_b32_offset_only",
            immediates=(
                Immediate("cache_policy", ImmediateKind.UNSIGNED),
                Immediate("offset", ImmediateKind.UNSIGNED),
            ),
        )
    )

    assert (
        amdgpu_target_refs._descriptor_address_offset_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )
        == 1
    )


def test_descriptor_immediate_slots_reject_duplicate_literal() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_bad_literal",
            immediates=(
                Immediate(
                    "literal0",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["LITERAL"],
                ),
                Immediate(
                    "literal1",
                    ImmediateKind.UNSIGNED,
                    encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["LITERAL"],
                ),
            ),
        )
    )

    with _raises_value_error("descriptor 'amdgpu.v_bad_literal' has multiple literal immediates"):
        amdgpu_target_refs._descriptor_literal_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )


def test_lowering_descriptor_contracts_accept_expected_asm_shapes() -> None:
    amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*_valid_contract_descriptors()))


def test_lowering_descriptor_contracts_reject_missing_canonical_form() -> None:
    with _raises_value_error("amdgpu.global_load_b32_saddr.*exactly one canonical asm form"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                _descriptor("amdgpu.global_load_b32_saddr", ()),
                *_valid_contract_descriptors()[1:],
            )
        )


def test_lowering_descriptor_contracts_reject_missing_helper_descriptor() -> None:
    with _raises_value_error("missing descriptor 'amdgpu.global_load_b64_saddr' required by target lowering"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*_valid_contract_descriptors()[:1]))


def test_lowering_descriptor_contracts_reject_bad_operand_count() -> None:
    descriptors = tuple(
        _descriptor(
            descriptor.key,
            (AsmForm(operands=("addr", "m0", "extra")),),
        )
        if descriptor.key == "amdgpu.flat_load_u8"
        else descriptor
        for descriptor in _valid_contract_descriptors()
    )
    with _raises_value_error("amdgpu.flat_load_u8.*expected one of: 1, 2"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*descriptors))


def test_lowering_descriptor_contracts_reject_bad_flat_store_operand_count() -> None:
    descriptors = tuple(
        _descriptor(
            descriptor.key,
            (AsmForm(operands=("addr", "value", "m0", "extra")),),
        )
        if descriptor.key == "amdgpu.flat_store_b32"
        else descriptor
        for descriptor in _valid_contract_descriptors()
    )
    with _raises_value_error("amdgpu.flat_store_b32.*expected one of: 2, 3"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(_descriptor_set(*descriptors))


def test_spill_lowering_contracts_require_every_packet() -> None:
    descriptors = _valid_spill_lowering_descriptors()

    with _raises_value_error("missing descriptor 'amdgpu.scratch_load_b128_vaddr' required by target lowering"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(_descriptor_set(*(descriptor for descriptor in descriptors if descriptor.key != "amdgpu.scratch_load_b128_vaddr")))


def test_spill_lowering_contracts_require_address_offset() -> None:
    descriptors = _valid_spill_lowering_descriptors()

    with _raises_value_error("spill descriptor 'amdgpu.scratch_load_b32_offset_only' has no address offset immediate"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
            _descriptor_set(
                _spill_lowering_descriptor(
                    "amdgpu.scratch_load_b32_offset_only",
                    include_address_offset=False,
                ),
                *descriptors[1:],
            )
        )


def test_spill_lowering_contracts_require_packet_shape() -> None:
    descriptors = _valid_spill_lowering_descriptors()

    with _raises_value_error(
        "spill descriptor 'amdgpu.scratch_load_b32_offset_only' has 0 explicit "
        r"operand\(s\) and 0 result\(s\); expected 0 operand\(s\) and 1 "
        r"result\(s\)"
    ):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
            _descriptor_set(
                _descriptor(
                    "amdgpu.scratch_load_b32_offset_only",
                    immediates=descriptors[0].immediates,
                ),
                *descriptors[1:],
            )
        )


def test_spill_lowering_contracts_require_encodable_zero_offset() -> None:
    descriptors = _valid_spill_lowering_descriptors()

    with _raises_value_error("spill descriptor 'amdgpu.scratch_load_b32_offset_only' address offset immediate does not encode zero"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
            _descriptor_set(
                _descriptor(
                    "amdgpu.scratch_load_b32_offset_only",
                    immediates=(
                        Immediate(
                            "offset",
                            ImmediateKind.SIGNED,
                            signed_min=1,
                            unsigned_max=4095,
                        ),
                    ),
                    operands=descriptors[0].operands,
                ),
                *descriptors[1:],
            )
        )


def test_spill_lowering_contracts_require_numeric_address_offset() -> None:
    descriptors = _valid_spill_lowering_descriptors()

    with _raises_value_error("spill descriptor 'amdgpu.scratch_load_b32_offset_only' address offset immediate has unsupported kind enum"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
            _descriptor_set(
                _descriptor(
                    "amdgpu.scratch_load_b32_offset_only",
                    immediates=(Immediate("offset", ImmediateKind.ENUM),),
                    operands=descriptors[0].operands,
                ),
                *descriptors[1:],
            )
        )


def test_spill_lowering_contracts_require_m0_materialization_when_consumed() -> None:
    descriptors = _valid_spill_lowering_descriptors()
    scratch_with_m0 = _spill_lowering_descriptor(
        "amdgpu.scratch_load_b32_offset_only",
        include_implicit_m0=True,
    )
    descriptor_set = _descriptor_set(scratch_with_m0, *descriptors[1:])

    with _raises_value_error("missing descriptor 'amdgpu.s_mov_b32_m0.imm' required by target lowering"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(descriptor_set)

    amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
        _descriptor_set(
            *descriptor_set.descriptors,
            _spill_lowering_descriptor(amdgpu_target_refs._SPILL_LOWERING_M0_DESCRIPTOR_KEY),
        )
    )


def test_spill_lowering_contracts_require_m0_materialization_immediate() -> None:
    descriptors = _valid_spill_lowering_descriptors()
    scratch_with_m0 = _spill_lowering_descriptor(
        "amdgpu.scratch_load_b32_offset_only",
        include_implicit_m0=True,
    )

    with _raises_value_error("spill helper 'amdgpu.s_mov_b32_m0.imm' has no imm32 immediate"):
        amdgpu_target_refs._validate_spill_lowering_descriptor_contracts(
            _descriptor_set(
                scratch_with_m0,
                *descriptors[1:],
                _descriptor(
                    amdgpu_target_refs._SPILL_LOWERING_M0_DESCRIPTOR_KEY,
                    operands=_spill_lowering_descriptor(amdgpu_target_refs._SPILL_LOWERING_M0_DESCRIPTOR_KEY).operands,
                ),
            )
        )


def test_lowering_descriptor_contracts_reject_bad_rel32_shape() -> None:
    bad_descriptor = _descriptor(
        "amdgpu.s_add_u32.rhs_symbol_rel32_lo",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        operands=(
            Operand("dst", OperandRole.RESULT, ()),
            Operand("lhs", OperandRole.OPERAND, ()),
        ),
        immediates=(
            Immediate("symbol", ImmediateKind.ORDINAL),
            Immediate("byte_offset", ImmediateKind.UNSIGNED),
        ),
    )
    with _raises_value_error("rel32 descriptor.*symbol immediate must be a symbolic relative literal"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                *(bad_descriptor if descriptor.key == bad_descriptor.key else descriptor for descriptor in _valid_contract_descriptors()),
            )
        )
