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
from loom.target.arch.amdgpu.descriptors.cdna import (
    _AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET,
)
from loom.target.arch.amdgpu.descriptors.common import _amdgpu_mfma_schedule_class_name
from loom.target.arch.amdgpu.descriptors.matrix import _cdna2_matrix_rows
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx9_4_generic_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_MUBUF,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
)
from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import (
    AsmForm,
    Constraint,
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
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClassAlt,
    Resource,
    ResourceFlag,
    ResourceKind,
    ScheduleClass,
    StorageLease,
    StorageLeaseAttachment,
    StorageLeaseKind,
    StorageLeaseReleaseScope,
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
    semantic_tag: str | None = None,
    schedule_class: str = _SCHEDULE_NONE,
    encoding_format_id: int = 0,
    encoding_id: int = 0,
    immediates: tuple[Immediate, ...] = (),
    instruction_classes: tuple[InstructionClass, ...] = (),
    operands: tuple[Operand, ...] = (),
    encoding_field_values: tuple[EncodingFieldValue, ...] = (),
    storage_leases: tuple[StorageLease, ...] = (),
    constraints: tuple[Constraint, ...] = (),
) -> Descriptor:
    return Descriptor(
        key=key,
        mnemonic=None,
        semantic_tag=semantic_tag,
        operands=operands,
        schedule_class=schedule_class,
        asm_forms=asm_forms,
        encoding_format_id=encoding_format_id,
        encoding_id=encoding_id,
        immediates=immediates,
        instruction_classes=instruction_classes,
        encoding_field_values=encoding_field_values,
        storage_leases=storage_leases,
        constraints=constraints,
    )


def _descriptor_set(*descriptors: Descriptor) -> DescriptorSet:
    return DescriptorSet(
        key="amdgpu.test.core",
        inline_schedule_model_key="amdgpu.generic",
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
        Operand("carry", OperandRole.RESULT, ()),
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
    source = amdgpu_target_refs._emit_tables_header((_descriptor_set(*_valid_contract_descriptors()),))

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
        reg_classes=_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE.reg_classes,
    )

    tables = amdgpu_target_refs._materialize_descriptor_ref_tables(
        (descriptor_set_info,),
        {descriptor_set_info.key: descriptor_set},
    )

    assert len(tables) == 1
    assert tables[0].descriptor_set_key == descriptor_set_info.key
    assert tables[0].descriptor_set_ordinal == amdgpu_descriptor_set_ordinal(descriptor_set_info.key)
    assert len(tables[0].immediate_slots) == len(descriptor_set.descriptors)


def test_target_ref_tables_reject_descriptor_ordinals_wider_than_storage() -> None:
    descriptor_set_info = sorted_descriptor_set_infos()[0]
    descriptor_set = replace(
        _descriptor_set(*_valid_contract_descriptors()),
        key=descriptor_set_info.key,
        descriptors=(_valid_contract_descriptors()[0],) * (amdgpu_target_refs._COMPACT_DESCRIPTOR_ORDINAL_NONE + 1),
        reg_classes=_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE.reg_classes,
    )

    with _raises_value_error("do not fit compact target-ref ordinals"):
        amdgpu_target_refs._materialize_descriptor_ref_tables(
            (descriptor_set_info,),
            {descriptor_set_info.key: descriptor_set},
        )


def test_exact_target_materialization_includes_compatible_portable_view() -> None:
    rdna3_5_info = amdgpu_descriptor_set_info_by_generator_target("rdna3_5")
    gfx11_info = amdgpu_descriptor_set_info_by_generator_target("gfx11_generic")

    assert set(amdgpu_target_refs._materialization_descriptor_set_infos((rdna3_5_info,))) == {rdna3_5_info, gfx11_info}
    assert amdgpu_target_refs._materialization_descriptor_set_infos((gfx11_info,)) == (gfx11_info,)


def test_target_ref_source_shares_exact_tables() -> None:
    descriptor_set_infos = sorted_descriptor_set_infos()[:2]
    descriptor_set = _descriptor_set(*_valid_contract_descriptors())
    descriptor_sets_by_key = {
        descriptor_set_info.key: replace(
            descriptor_set,
            key=descriptor_set_info.key,
            reg_classes=_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE.reg_classes,
        )
        for descriptor_set_info in descriptor_set_infos
    }

    source = amdgpu_target_refs._emit_source(
        public_header="target_refs.h",
        descriptor_set_infos=descriptor_set_infos,
        descriptor_sets_by_key=descriptor_sets_by_key,
    )

    assert ".cache_coherence = LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE" in source
    assert ".literal = UINT8_C(" in source

    first_info, second_info = sorted(
        descriptor_set_infos,
        key=lambda info: amdgpu_descriptor_set_ordinal(info.key),
    )
    table_name_functions = (
        amdgpu_target_refs._descriptor_set_table_name,
        amdgpu_target_refs._descriptor_set_trait_table_name,
        amdgpu_target_refs._descriptor_set_memory_property_table_name,
        amdgpu_target_refs._descriptor_set_immediate_slot_ref_table_name,
        amdgpu_target_refs._reg_class_trait_table_name,
    )
    for table_name_function in table_name_functions:
        first_table_name = table_name_function(first_info.key)
        second_table_name = table_name_function(second_info.key)
        assert source.count(f" {first_table_name}[] = {{") == 1, (
            table_name_function.__name__,
            first_table_name,
            second_table_name,
        )
        assert f" {second_table_name}[] = {{" not in source
        assert (f"[{amdgpu_target_refs._u16_literal(amdgpu_descriptor_set_ordinal(second_info.key))}] = {first_table_name},") in source
    first_ordinal_table_name = amdgpu_target_refs._descriptor_set_table_name(first_info.key)
    assert f"static const uint16_t {first_ordinal_table_name}[] = {{" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_ORDINAL_NONE" in source
    assert "DescriptorRefsByOrdinal" not in source
    assert "kLoomAmdgpuDescriptorRefByOrdinalTables" not in source
    first_immediate_slot_ref_table_name = amdgpu_target_refs._descriptor_set_immediate_slot_ref_table_name(first_info.key)
    assert f"static const uint8_t {first_immediate_slot_ref_table_name}[] = {{" in source
    assert "const loom_amdgpu_descriptor_immediate_slots_t kLoomAmdgpuDescriptorImmediateSlotValues[] = {" in source


def test_accumulator_class_ids_follow_each_descriptor_set() -> None:
    for base in (
        _AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
        _AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE,
    ):
        descriptor_set = replace(_descriptor_set(*_valid_contract_descriptors()), key=base.key, reg_classes=base.reg_classes)
        info = next(info for info in sorted_descriptor_set_infos() if info.key == base.key)
        tables = amdgpu_target_refs._materialize_descriptor_ref_tables((info,), {base.key: descriptor_set})
        assert descriptor_set.reg_classes[tables[0].agpr_reg_class_id].name == "amdgpu.agpr"
        with _raises_value_error("requires an accumulator register class"):
            amdgpu_target_refs._materialize_descriptor_ref_tables((info,), {base.key: replace(descriptor_set, reg_classes=())})


def test_descriptor_trait_names_include_resource_and_encoding_facts() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_mov_b32.sdwa",
            schedule_class=_SCHEDULE_VALU,
            encoding_format_id=AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]) == (
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE",
    )


def test_matrix_pass_traits_are_independent_of_latency_estimates() -> None:
    for descriptor_set, overlays in (
        (_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE, tuple(overlay for overlay, _ in _cdna2_matrix_rows())),
        (_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, _gfx940_core_overlays()),
        (_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx950_core_overlays()),
        (_AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE, _gfx9_4_generic_core_overlays()),
    ):
        overlays_by_key = {overlay.descriptor_key: overlay for overlay in overlays}
        descriptor_set = replace(
            descriptor_set,
            schedule_classes=tuple(replace(row, latency_cycles=91) for row in descriptor_set.schedule_classes),
        )
        context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
        for key, timing in AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET[descriptor_set.key].items():
            overlay = overlays_by_key[key]
            descriptor = _descriptor(
                key,
                schedule_class=_amdgpu_mfma_schedule_class_name(key),
                semantic_tag=overlay.semantic_tag,
                operands=tuple(operand.descriptor_operand for operand in overlay.operands),
                constraints=overlay.constraints,
            )
            traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor)
            assert f"({timing.execution_passes}u << LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_SHIFT)" in traits


def test_matrix_pipeline_traits_preserve_arithmetic_semantics() -> None:
    context = amdgpu_target_refs._descriptor_trait_context(_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE)
    overlays_by_semantic = {overlay.semantic_tag: overlay for overlay in _gfx940_core_overlays()}
    # Physical result width cannot distinguish these pipelines. Sparse and
    # reduced-precision F32 instructions belong to XDL along with narrow inputs.
    for semantic, expected in (
        ("matrix.mfma.f32.16x16x4.f32", ()),
        ("matrix.mfma.f64.16x16x4.f64", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM",)),
        ("matrix.mfma.f32.16x16x16.f16", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL",)),
        ("matrix.mfma.f32.16x16x16.bf16.1k", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL",)),
        ("matrix.mfma.f32.16x16x8.xf32", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL",)),
        ("matrix.mfma.i32.16x16x32.i8", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL",)),
        ("matrix.smfmac.f32.16x16x32.bf16", ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL",)),
    ):
        overlay = overlays_by_semantic[semantic]
        descriptor = _descriptor(
            overlay.descriptor_key,
            semantic_tag=semantic,
            schedule_class=overlay.schedule_class,
            operands=tuple(operand.descriptor_operand for operand in overlay.operands),
            constraints=overlay.constraints,
        )
        traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor)
        assert tuple(trait for trait in traits if trait.endswith(("MATRIX_DGEMM", "MATRIX_XDL"))) == expected


def test_matrix_exact_forwarding_floors_require_indexable_result_shapes() -> None:
    context = amdgpu_target_refs._descriptor_trait_context(_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE)
    timings = AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET[context.descriptor_set.key]
    for overlay in _gfx940_core_overlays():
        timing = timings.get(overlay.descriptor_key)
        if timing is None or not (timing.execution_passes == 2 or (timing.execution_passes == 4 and (overlay.semantic_tag or "").startswith("matrix.mfma.f64."))):
            continue
        operands = tuple(operand.descriptor_operand for operand in overlay.operands)
        descriptor = _descriptor(overlay.descriptor_key, semantic_tag=overlay.semantic_tag, schedule_class=overlay.schedule_class, operands=operands)
        for invalid_operands in (
            (),
            tuple(replace(operand, unit_count=operand.unit_count + 1) if operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT) else operand for operand in operands),
        ):
            with _raises_value_error("exact forwarding floor requires one"):
                amdgpu_target_refs._descriptor_trait_names(context, replace(descriptor, operands=invalid_operands))


def test_matrix_input_retention_requires_matching_accumulator_width() -> None:
    for descriptor_set, overlays in (
        (_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE, tuple(overlay for overlay, _ in _cdna2_matrix_rows())),
        (_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, _gfx940_core_overlays()),
        (_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx950_core_overlays()),
        (_AMDGPU_GFX9_4_GENERIC_CORE_DESCRIPTOR_SET_BASE, _gfx9_4_generic_core_overlays()),
    ):
        context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
        for overlay in overlays:
            if overlay.descriptor_key not in AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET[descriptor_set.key]:
                continue
            descriptor = _descriptor(
                overlay.descriptor_key,
                semantic_tag=overlay.semantic_tag,
                schedule_class=overlay.schedule_class,
                operands=tuple(operand.descriptor_operand for operand in overlay.operands),
                constraints=overlay.constraints,
            )
            amdgpu_target_refs._descriptor_trait_names(context, descriptor)
            with _raises_value_error("same-width accumulator"):
                amdgpu_target_refs._descriptor_trait_names(context, replace(descriptor, constraints=()))
            with _raises_value_error("same-width accumulator"):
                amdgpu_target_refs._descriptor_trait_names(
                    context, replace(descriptor, operands=tuple(replace(operand, unit_count=operand.unit_count + 1) if operand.field_name == "acc" else operand for operand in descriptor.operands))
                )


def test_cdna_dot_identity_uses_native_opcode_and_preserves_aliases() -> None:
    overlay = next(overlay for overlay in _gfx940_core_overlays() if overlay.semantic_tag == "dot.f16f16.f32x1")
    descriptor = _descriptor(
        overlay.descriptor_key,
        semantic_tag=overlay.semantic_tag,
        schedule_class=overlay.schedule_class,
        operands=tuple(operand.descriptor_operand for operand in overlay.operands),
        encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS["ENC_VOP3P"],
        encoding_id=35,
    )
    descriptors = (
        descriptor,
        replace(descriptor, key="amdgpu.dot_alias"),
        replace(descriptor, key="amdgpu.other_dot", encoding_id=40),
    )
    context = amdgpu_target_refs._descriptor_trait_context(replace(_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=descriptors))
    for row, identity in zip(descriptors, (1, 1, 2), strict=True):
        traits = amdgpu_target_refs._descriptor_trait_names(context, row)
        assert f"({identity}u << LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_SHIFT)" in traits
    with _raises_value_error("six native DOT forwarding channels"):
        amdgpu_target_refs._descriptor_trait_context(
            replace(_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, descriptors=tuple(replace(descriptor, key=f"dot{index}", encoding_id=index) for index in range(7)))
        )
    with _raises_value_error("one DWORD result and lhs/rhs/acc inputs"):
        amdgpu_target_refs._descriptor_trait_names(context, replace(descriptor, operands=descriptor.operands[:-1]))
    # RDNA uses its own hardware/delay-ALU contract and has no CDNA channels.
    context = amdgpu_target_refs._descriptor_trait_context(
        replace(_descriptor_set(descriptor), schedule_classes=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE.schedule_classes, resources=_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE.resources)
    )
    assert not context.dot_identities
    assert not any("DOT_IDENTITY" in trait for trait in amdgpu_target_refs._descriptor_trait_names(context, descriptor))


def test_scalar_memory_trait_follows_native_encoding_independently_of_spelling() -> None:
    # SMEM includes scalar, descriptor-backed and scratch loads/stores plus
    # cache operations. Their common encoding class, not their mnemonic or
    # scheduling resource, defines replay-clause membership.
    for name, format_id in AMDGPU_ENCODING_FORMAT_IDS.items():
        descriptor = _descriptor("amdgpu.memory_alias", encoding_format_id=format_id)
        context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
        traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor)
        assert ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY" in traits) == (name == "ENC_SMEM"), name


def test_matrix_memory_read_traits_cover_all_native_memory_formats() -> None:
    for name in (
        "ENC_MUBUF",
        "ENC_MTBUF",
        "ENC_FLAT",
        "ENC_FLAT_GLBL",
        "ENC_FLAT_SCRATCH",
        "ENC_MIMG",
        "MIMG_NSA1",
        "MIMG_NSA2",
        "MIMG_NSA3",
        "ENC_DS",
        "ENC_EXP",
        "ENC_VBUFFER",
        "ENC_VFLAT",
        "ENC_VGLOBAL",
        "ENC_VSCRATCH",
        "ENC_VIMAGE",
        "ENC_VSAMPLE",
        "ENC_VDS",
        "ENC_VEXPORT",
    ):
        descriptor = _descriptor("amdgpu.memory", encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS[name])
        context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
        assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER" in amdgpu_target_refs._descriptor_trait_names(context, descriptor), name
        traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor)
        assert ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK" in traits) == (name not in ("ENC_EXP", "ENC_VEXPORT")), name
    for name in ("ENC_VOP1", "ENC_SMEM", "ENC_SOPP"):
        descriptor = _descriptor("amdgpu.other", encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS[name])
        context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
        assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER" not in amdgpu_target_refs._descriptor_trait_names(context, descriptor), name
        assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK" not in amdgpu_target_refs._descriptor_trait_names(context, descriptor), name


def test_fma64_pipeline_traits_follow_semantics_across_encoding_forms() -> None:
    for semantic in ("float.fma.f64", "float.fmac.f64", "float.fma.f32", "float.add.f64", "float.div.fmas.f64"):
        for format_name in ("ENC_VOP2", "ENC_VOP3"):
            descriptor = _descriptor("amdgpu.arithmetic", semantic_tag=semantic, encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS[format_name])
            context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
            traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor)
            assert ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_FMA64" in traits) == (semantic in ("float.fma.f64", "float.fmac.f64"))


def test_descriptor_trait_names_classify_dpp_encoding_families() -> None:
    for format_name, format_id in AMDGPU_ENCODING_FORMAT_IDS.items():
        if not format_name.endswith(("_DPP", "_DPP16", "_DPP8")):
            continue
        descriptor_set = _descriptor_set(
            _descriptor(
                f"amdgpu.test.{format_name.lower()}",
                schedule_class=_SCHEDULE_VALU,
                encoding_format_id=format_id,
            )
        )
        trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
        traits = amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])
        assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP" in traits, format_name


def test_descriptor_trait_names_classify_packed_f32_pair_spelling() -> None:
    for semantic_tag, expected in (
        ("float.add.pk2.f32", True),
        ("float.mul.pk2.f32", True),
        ("float.fma.pk2.f32", True),
        ("float.fma.pk2.f16", False),
        ("float.add.f32", False),
        ("float.fma.mix.f32", False),
    ):
        descriptor_set = _descriptor_set(_descriptor("amdgpu.test", semantic_tag=semantic_tag))
        context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)
        traits = amdgpu_target_refs._descriptor_trait_names(context, descriptor_set.descriptors[0])
        assert ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_PACKED_F32" in traits) == expected


def test_wide_shift_allocation_trait_requires_the_first_count_operand() -> None:
    count = Operand("shift", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),), address_map_kind=OperandAddressMapKind.LOW_SUBSET, addressable_unit_count=255)
    value = Operand("value", OperandRole.OPERAND, (RegClassAlt("amdgpu.vgpr"),), unit_count=2)
    for semantic_tag in ("integer.shl.u64", "integer.shr.u64", "integer.shr.s64"):
        for operands, expected in (
            ((count, value), True),
            ((replace(count, address_map_kind=OperandAddressMapKind.DIRECT, addressable_unit_count=0), value), False),
            ((value,), False),
        ):
            descriptor = _descriptor("amdgpu.shift", semantic_tag=semantic_tag, operands=operands)
            context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
            assert ("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SHIFT64_ALLOCATION" in amdgpu_target_refs._descriptor_trait_names(context, descriptor)) == expected
        for operands in ((value, count), (replace(count, unit_count=2), value)):
            descriptor = _descriptor("amdgpu.shift", semantic_tag=semantic_tag, operands=operands)
            context = amdgpu_target_refs._descriptor_trait_context(_descriptor_set(descriptor))
            with _raises_value_error("first one-register operand"):
                amdgpu_target_refs._descriptor_trait_names(context, descriptor)


def test_descriptor_trait_names_classify_permlane_semantics() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.v_permlane32_swap_b32",
            semantic_tag="lane.permlane32.swap.b32",
            schedule_class=_SCHEDULE_VALU,
        )
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_PERMLANE" in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]))


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


def test_descriptor_trait_names_include_retained_address_sources() -> None:
    source_lease = StorageLease(
        kind=StorageLeaseKind.SOURCE_READ,
        attachment=StorageLeaseAttachment.OPERAND,
        attachment_index=0,
        unit_offset=0,
        unit_count=1,
        release_scope=StorageLeaseReleaseScope.PROGRESS_CLASS,
        release_class_id=1,
        release_class_name="test.progress",
        release_action_id=1,
        release_action_name="test.release",
        release_reason_id=1,
        release_reason_name="test.retained",
    )
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.global_store_b32",
            operands=(
                Operand("addr", OperandRole.OPERAND, ()),
                Operand("value", OperandRole.OPERAND, ()),
            ),
            storage_leases=(source_lease,),
        ),
        _descriptor(
            "amdgpu.buffer_store_b32",
            operands=(
                Operand("value", OperandRole.OPERAND, ()),
                Operand("vaddr", OperandRole.OPERAND, ()),
            ),
            storage_leases=(replace(source_lease, attachment_index=1),),
        ),
        _descriptor(
            "amdgpu.global_store_b32.issue_consumed",
            operands=(
                Operand("addr", OperandRole.OPERAND, ()),
                Operand("value", OperandRole.OPERAND, ()),
            ),
            storage_leases=(replace(source_lease, attachment_index=1),),
        ),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED" in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]))
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED" in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1]))
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED" not in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[2]))


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
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER",
        "LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK",
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


def test_descriptor_trait_names_retain_exec_writes() -> None:
    exec_alt = (RegClassAlt("amdgpu.exec"),)
    scc_alt = (RegClassAlt("amdgpu.scc"),)
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.s_mov_b64_exec",
            operands=(
                Operand(
                    "exec",
                    OperandRole.IMPLICIT,
                    exec_alt,
                    flags=(OperandFlag.STATE_WRITE,),
                ),
            ),
        ),
        _descriptor(
            "amdgpu.exec_read",
            operands=(
                Operand(
                    "exec",
                    OperandRole.IMPLICIT,
                    exec_alt,
                    flags=(OperandFlag.STATE_READ,),
                ),
            ),
        ),
        _descriptor(
            "amdgpu.scc_write",
            operands=(
                Operand(
                    "scc",
                    OperandRole.IMPLICIT,
                    scc_alt,
                    flags=(OperandFlag.STATE_WRITE,),
                ),
            ),
        ),
    )
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_WRITES_EXEC" in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0]))
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_WRITES_EXEC" not in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[1]))
    assert "LOOM_AMDGPU_DESCRIPTOR_TRAIT_WRITES_EXEC" not in (amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[2]))


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


def test_wide_store_windows_preserve_generation_and_offset_contracts() -> None:
    # ISA section 4.5: CDNA2 has a uniform one-slot window. Later CDNA has
    # a second VALU slot; LLVM retains one VALU slot for SGPR-offset buffers.
    for target, ordinary, scalar_offset in (
        ("cdna2", (1, 1), (0, 0)),
        ("cdna3", (2, 1), (1, 0)),
        ("cdna4", (2, 1), (1, 0)),
        ("gfx9_4_generic", (2, 1), (1, 0)),
    ):
        info = amdgpu_descriptor_set_info_by_generator_target(target)
        for format_name, data_field, offsets in (
            ("ENC_FLAT", "DATA", (False,)),
            ("ENC_MUBUF", "VDATA", (False, True)),
        ):
            # Ordinary stores plus 32/64-bit compare-swap input/result shapes.
            for width, result_width in ((1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (4, 2)):
                for has_scalar_offset in offsets:
                    returned = result_width != 0
                    operands = [Operand("value", OperandRole.OPERAND, (), unit_count=width, encoding_field_id=AMDGPU_ENCODING_FIELD_IDS[data_field])]
                    if returned:
                        operands.insert(0, Operand("old", OperandRole.RESULT, (), unit_count=result_width))
                    if has_scalar_offset:
                        operands.append(Operand("offset", OperandRole.OPERAND, (), encoding_field_id=AMDGPU_ENCODING_FIELD_IDS["SOFFSET"]))
                    descriptor = _descriptor(
                        "amdgpu.payload",
                        encoding_format_id=AMDGPU_ENCODING_FORMAT_IDS[format_name],
                        operands=tuple(operands),
                    )
                    order = "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER" if returned else "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"
                    valu_cycles, other_cycles = scalar_offset if has_scalar_offset else ordinary
                    expected = order
                    if width > 2 and valu_cycles:
                        expected += f" | (({int(returned)} + 1) << 3) | ({other_cycles} << 6) | ({valu_cycles - 1} << 7)"
                    assert amdgpu_target_refs._descriptor_memory_properties(info, descriptor) == expected


def test_descriptor_traits_reject_missing_schedule_class() -> None:
    descriptor_set = _descriptor_set(_descriptor("amdgpu.test", schedule_class="missing"))
    trait_context = amdgpu_target_refs._descriptor_trait_context(descriptor_set)

    with _raises_value_error("descriptor 'amdgpu.test' references missing schedule class 'missing'"):
        amdgpu_target_refs._descriptor_trait_names(trait_context, descriptor_set.descriptors[0])


def test_descriptor_traits_reject_missing_issue_resource() -> None:
    descriptor_set = DescriptorSet(
        key="amdgpu.test.core",
        inline_schedule_model_key="amdgpu.generic",
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


def test_descriptor_immediate_slots_publish_cache_coherence() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.global_store_b32",
            immediates=(
                Immediate("offset", ImmediateKind.UNSIGNED),
                Immediate("scope", ImmediateKind.UNSIGNED),
            ),
        )
    )

    assert (
        amdgpu_target_refs._descriptor_cache_coherence_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )
        == 1
    )


def test_descriptor_immediate_slots_publish_glc_cache_coherence() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.global_load_b32",
            immediates=(
                Immediate("offset", ImmediateKind.UNSIGNED),
                Immediate("glc", ImmediateKind.UNSIGNED),
            ),
        )
    )

    assert (
        amdgpu_target_refs._descriptor_cache_coherence_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )
        == 1
    )


def test_descriptor_immediate_slots_reject_duplicate_cache_coherence() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.bad_global_store_b32",
            immediates=(
                Immediate("glc", ImmediateKind.UNSIGNED),
                Immediate("scope", ImmediateKind.UNSIGNED),
            ),
        )
    )

    with _raises_value_error("descriptor 'amdgpu.bad_global_store_b32' has multiple cache coherence immediates"):
        amdgpu_target_refs._descriptor_cache_coherence_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
        )


def test_descriptor_immediate_slots_reject_unrepresentable_slot() -> None:
    descriptor_set = _descriptor_set(
        _descriptor(
            "amdgpu.bad_global_store_b32",
            immediates=tuple(
                Immediate(
                    "scope" if index == 255 else f"unused{index}",
                    ImmediateKind.UNSIGNED,
                )
                for index in range(256)
            ),
        )
    )

    with _raises_value_error("cache coherence immediate slot 255 does not fit compact target refs"):
        amdgpu_target_refs._descriptor_cache_coherence_immediate_slot(
            descriptor_set,
            descriptor_set.descriptors[0],
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
