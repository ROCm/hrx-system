# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from loom.gen.target.arch.amdgpu.refs import amdgpu_target_refs
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_MUBUF,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
)
from loom.target.low_descriptors import (
    AsmForm,
    Descriptor,
    DescriptorSet,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    IssueUse,
    LatencyKind,
    ModelQuality,
    Operand,
    OperandRole,
    Resource,
    ResourceKind,
    ScheduleClass,
)

_RESOURCE_SALU = "salu"
_RESOURCE_VALU = "valu"
_SCHEDULE_NONE = "none"
_SCHEDULE_SALU = "salu"
_SCHEDULE_VALU = "valu"


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
    operands: tuple[Operand, ...] = (),
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
        ),
        descriptors=descriptors,
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
    return (
        _descriptor(
            "amdgpu.global_load_b32_saddr",
            (AsmForm(operands=("vaddr", "saddr")),),
        ),
        _descriptor(
            "amdgpu.global_load_b64_saddr",
            (AsmForm(operands=("vaddr", "saddr", "m0")),),
        ),
        _descriptor(
            "amdgpu.flat_load_u8",
            (AsmForm(operands=("addr",)),),
        ),
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
    )


def test_descriptor_trait_names_include_memory_and_ref_facts() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.global_load_b32",
            schedule_class=_SCHEDULE_SALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_MUBUF,
        ),
        _descriptor("amdgpu.v_exp_f32", schedule_class=_SCHEDULE_VALU),
        _descriptor("amdgpu.v_readfirstlane_b32", schedule_class=_SCHEDULE_SALU),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL",
    )
    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[2]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE",
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
    with _raises_value_error("amdgpu.flat_load_u8.*expected one of: 1, 2"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                *_valid_contract_descriptors()[:2],
                _descriptor(
                    "amdgpu.flat_load_u8",
                    (AsmForm(operands=("addr", "m0", "extra")),),
                ),
                *_valid_contract_descriptors()[3:],
            )
        )


def test_lowering_descriptor_contracts_reject_bad_rel32_shape() -> None:
    with _raises_value_error("rel32 descriptor.*symbol immediate must be a symbolic relative literal"):
        amdgpu_target_refs._validate_lowering_descriptor_contracts(
            _descriptor_set(
                *_valid_contract_descriptors()[:3],
                _descriptor(
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
                ),
                _valid_contract_descriptors()[4],
            )
        )
