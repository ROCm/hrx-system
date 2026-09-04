# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from loom.gen.target.low import compiler, views
from loom.gen.target.low.low_descriptors import (
    DescriptorAllowlist,
    generate_descriptor_set,
    generate_descriptor_set_family,
)
from loom.ir import ScalarTypeKind
from loom.target.low_descriptors import (
    LOW_DESCRIPTOR_ENCODING_ID_NONE,
    AsmForm,
    AsmImmediate,
    AsmOperandSegment,
    AsmOperandSegmentDelimiter,
    AsmResultValueType,
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorAsmSurface,
    DescriptorCategory,
    DescriptorFlag,
    Effect,
    EffectKind,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    Hazard,
    HazardKind,
    Immediate,
    ImmediateEncodingSlice,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    IssueUseKind,
    NativeAsmValue,
    NativeAsmValueKind,
    OperandAddressMapKind,
    OperandFlag,
    OperandForm,
    OperandFormMatch,
    OperandFormMatchKind,
    OperandRole,
    OperandSourceBinding,
    PhysicalRegister,
    PhysicalRegisterView,
    RegClassAlt,
    RegClassAltFlag,
    RegClassFlag,
    RegisterPackingResource,
    RegisterPackingResourceMember,
    StorageLease,
    StorageLeaseAttachment,
    StorageLeaseFlag,
    StorageLeaseKind,
    StorageLeaseReleaseScope,
    descriptor_stable_id,
    operand_source_binding,
)
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_ADD_PHYS_DESCRIPTOR,
    TEST_LOW_BARRIER_DESCRIPTOR,
    TEST_LOW_COND_BR_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
    TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
    TEST_LOW_STATE_ADD_SCHEDULE_STATE_DESCRIPTOR,
    TEST_LOW_WRITE_HIGH16_I32_DESCRIPTOR,
    TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR,
)


def _explicit_physical_descriptor_set():
    physical_registers = (
        PhysicalRegister("test.r0", (0, 1)),
        PhysicalRegister("test.r1", (2, 3)),
        PhysicalRegister("test.r2", (4, 5)),
        PhysicalRegister("test.pair", (0, 1, 4, 5)),
    )
    register_classes = tuple(
        replace(
            register_class,
            flags=(
                RegClassFlag.PHYSICAL,
                RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS,
            ),
            allocatable_count=0,
            fixed_location_base=0,
            fixed_location_count=0,
            physical_registers=("test.r2", "test.r0"),
        )
        if register_class.name == "test.phys"
        else register_class
        for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes
        if register_class.name not in ("test.explicit32", "test.packed.narrow", "test.packed.wide")
    )
    return replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        physical_registers=physical_registers,
        physical_register_views=(PhysicalRegisterView("test.pair", "test.phys", ("test.r2", "test.r0")),),
        register_packing_resources=(),
        reg_classes=register_classes,
        descriptors=(TEST_LOW_ADD_PHYS_DESCRIPTOR,),
    )


def _compiled_slice(
    rows: Sequence[object],
    start: int,
    count: int,
) -> tuple[object, ...]:
    return tuple(rows[start : start + count])


def test_descriptor_category_validates_stable_key_spelling() -> None:
    category = DescriptorCategory("memory.atomic", doc="Atomic memory.")

    assert category.key == "memory.atomic"
    assert category.doc == "Atomic memory."

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor category key 'Memory/Atomic' must contain only "
            r"lowercase letters, digits, '.', '_', or '-'"
        ),
    ):
        DescriptorCategory("Memory/Atomic")


def test_descriptor_set_validates_category_membership() -> None:
    category = DescriptorCategory("control")
    descriptor = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[0],
        category=category,
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        categories=(category,),
        descriptors=(descriptor,),
    )

    assert descriptor_set.categories == (category,)
    assert descriptor_set.descriptors[0].category == category


def test_descriptor_set_rejects_unknown_default_category() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"DescriptorSet 'test.low.core': default_category 'memory' "
            r"is not declared in categories"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            categories=(DescriptorCategory("control"),),
            default_category=DescriptorCategory("memory"),
        )


def test_descriptor_set_rejects_unknown_descriptor_category() -> None:
    descriptor = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[0],
        category=DescriptorCategory("memory"),
    )

    with pytest.raises(
        ValueError,
        match=(
            rf"DescriptorSet 'test.low.core': descriptor '{descriptor.key}' "
            r"category 'memory' is not declared in categories"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            categories=(DescriptorCategory("control"),),
            descriptors=(descriptor,),
        )


def test_descriptor_set_requires_canonical_supported_target_contracts() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"DescriptorSet 'test.low.core': supported target contract keys "
            r"must be sorted and unique"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            supported_target_contract_keys=(
                "test.low.exact_b",
                "test.low.exact_a",
            ),
        )


def test_descriptor_set_rejects_explicit_self_target_contract() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"DescriptorSet 'test.low.core': support for its own target "
            r"contract is implicit"
        ),
    ):
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            supported_target_contract_keys=("test.low.core",),
        )


def test_descriptor_set_emits_supported_target_contract_identities() -> None:
    target_contract_keys = (
        "test.low.exact_a",
        "test.low.exact_b",
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        supported_target_contract_keys=target_contract_keys,
    )

    generated = generate_descriptor_set(descriptor_set)

    assert ("static const uint64_t kTestLowCoreSupportedTargetContractStableIds[]") in generated.source
    for key in target_contract_keys:
        assert f"UINT64_C(0x{descriptor_stable_id(key):x})" in generated.source
    assert (".supported_target_contract_stable_ids = kTestLowCoreSupportedTargetContractStableIds") in generated.source
    assert (".supported_target_contract_count = IREE_ARRAYSIZE(kTestLowCoreSupportedTargetContractStableIds)") in generated.source


def test_descriptor_set_requires_canonical_asm_for_authorable_surface() -> None:
    descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, asm_forms=())
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is authorable asm but does not declare exactly one canonical asm form; found 0"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_rejects_authorable_surface_reason() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_surface_reason="temporary note that should not be in authorable policy",
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is authorable asm but has an asm surface reason"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_rejects_authorable_pseudo_surface() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
        flags=(DescriptorFlag.PSEUDO,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is pseudo but classified as authorable asm"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_requires_reason_for_non_authorable_surface() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(),
        asm_surface=DescriptorAsmSurface.STRUCTURAL,
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is structural asm but does not explain the non-authorable surface"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_rejects_non_authorable_surface_with_asm_forms() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_surface=DescriptorAsmSurface.STRUCTURAL,
        asm_surface_reason="represented by structural low asm syntax",
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is structural asm but still declares 1 asm form(s)"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_rejects_non_generated_only_pseudo_surface() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(),
        asm_surface=DescriptorAsmSurface.STRUCTURAL,
        asm_surface_reason="represented by structural low asm syntax",
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
        flags=(DescriptorFlag.PSEUDO,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' descriptor 'test.add.i32' is pseudo but not classified as generated-only asm"),
    ):
        generate_descriptor_set(descriptor_set)


def test_descriptor_set_accepts_explained_generated_only_pseudo_surface() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(),
        asm_surface=DescriptorAsmSurface.GENERATED_ONLY,
        asm_surface_reason="lowering synthesizes this pseudo before packet selection",
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
        flags=(DescriptorFlag.PSEUDO,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
        requires_explicit_asm_surface=True,
    )

    generated = generate_descriptor_set(descriptor_set)

    assert ".canonical_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE" in generated.source


def test_allowlist_closes_over_referenced_descriptor_tables() -> None:
    generated = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(keys=("test.add.i32",)),
    )

    assert '"test.add.i32"' in generated.source
    assert '"test.i32"' in generated.source
    assert '"test.scalar.alu"' in generated.source
    assert '"test.scalar"' in generated.source
    assert "test.call.i32" not in generated.source


def test_compiler_descriptor_rows_span_source_tables() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)

    assert len(compiled.descriptor_rows) == len(compiled.descriptors)
    rows_by_key = {
        descriptor.key: row
        for descriptor, row in zip(
            compiled.descriptors,
            compiled.descriptor_rows,
            strict=True,
        )
    }
    assert rows_by_key["test.add.i32"]["minimum_packet_operand_count"] == 2
    for descriptor, row in zip(
        compiled.descriptors,
        compiled.descriptor_rows,
        strict=True,
    ):
        assert (
            _compiled_slice(
                compiled.operands,
                row["operand_start"],
                row["operand_count"],
            )
            == descriptor.operands
        )
        assert (
            _compiled_slice(
                compiled.immediates,
                row["immediate_start"],
                row["immediate_count"],
            )
            == descriptor.immediates
        )
        assert (
            _compiled_slice(
                compiled.effects,
                row["effect_start"],
                row["effect_count"],
            )
            == descriptor.effects
        )
        assert (
            _compiled_slice(
                compiled.constraints,
                row["constraint_start"],
                row["constraint_count"],
            )
            == descriptor.constraints
        )
        assert (
            _compiled_slice(
                compiled.storage_leases,
                row["storage_lease_start"],
                row["storage_lease_count"],
            )
            == descriptor.storage_leases
        )
        assert (
            _compiled_slice(
                compiled.feature_mask_words,
                row["feature_mask_word_start"],
                row["feature_mask_word_count"],
            )
            == descriptor.feature_mask_words
        )
        assert (
            _compiled_slice(
                compiled.encoding_field_values,
                row["encoding_field_value_start"],
                row["encoding_field_value_count"],
            )
            == descriptor.encoding_field_values
        )


def test_compiler_rejects_contradictory_storage_lease_boundary_flags() -> None:
    lease = StorageLease(
        kind=StorageLeaseKind.RESULT_WRITE,
        attachment=StorageLeaseAttachment.RESULT,
        attachment_index=0,
        unit_offset=0,
        unit_count=1,
        release_scope=StorageLeaseReleaseScope.PROGRESS_CLASS,
        release_class_id=1,
        release_class_name="test.progress",
        release_action_id=1,
        release_action_name="test.release",
        release_reason_id=1,
        release_reason_name="test.result_reuse",
        flags=(
            StorageLeaseFlag.RELEASE_BEFORE_BOUNDARY,
            StorageLeaseFlag.MAY_CARRY_ACROSS_BOUNDARY,
        ),
    )
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        storage_leases=(lease,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' storage lease 0 cannot both release before and carry across a boundary"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_sparse_register_alias_set_ids() -> None:
    register_classes = tuple(
        replace(register_class, alias_set_id=register_class.alias_set_id + 1) if register_class.alias_set_id != 0 else register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' alias-set IDs must be dense from 1; found [2, 3]"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_register_alias_set_location_kind_mismatch() -> None:
    register_classes = tuple(
        replace(
            register_class,
            flags=(RegClassFlag.VIRTUAL_ONLY,),
            allocatable_count=0,
        )
        if register_class.name == "test.alias64"
        else register_class
        for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' alias set 1 mixes physical and virtual location classes 'test.alias32' and 'test.alias64'"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_register_alias_set_target_bank_mismatch() -> None:
    register_classes = tuple(replace(register_class, target_bank_id=1) if register_class.name == "test.alias64" else register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' alias set 1 classes 'test.alias32' and 'test.alias64' use different target banks"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_register_alias_set_capacity_mismatch() -> None:
    register_classes = tuple(replace(register_class, allocatable_count=2) if register_class.name == "test.alias64" else register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' alias set 1 classes 'test.alias32' and 'test.alias64' have different allocatable counts"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_virtual_register_fixed_locations() -> None:
    register_classes = tuple(replace(register_class, fixed_location_count=1) if register_class.name == "test.i32" else register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register class 'test.i32' has fixed physical locations but is virtual-only"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_register_fixed_location_overlap() -> None:
    register_classes = tuple(replace(register_class, fixed_location_base=31) if register_class.name == "test.phys" else register_class for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes)
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register class 'test.phys' fixed-location range overlaps its allocatable locations"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_register_alias_set_fixed_location_mismatch() -> None:
    register_classes = tuple(
        replace(register_class, fixed_location_base=1, fixed_location_count=1) if register_class.name == "test.alias64" else register_class
        for register_class in TEST_LOW_CORE_DESCRIPTOR_SET.reg_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        reg_classes=register_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' alias set 1 classes 'test.alias32' and 'test.alias64' have different fixed-location ranges"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_emits_explicit_physical_register_candidates() -> None:
    descriptor_set = _explicit_physical_descriptor_set()

    compiled = compiler.compile_descriptor_set(descriptor_set)
    generated = generate_descriptor_set(descriptor_set)

    physical_class_index = next(index for index, register_class in enumerate(compiled.reg_classes) if register_class.name == "test.phys")
    candidate_start = compiled.physical_register_candidate_starts[physical_class_index]
    candidate_count = len(compiled.reg_classes[physical_class_index].physical_registers)
    assert compiled.physical_register_candidate_ids[candidate_start : candidate_start + candidate_count] == [2, 0]
    assert compiled.physical_register_atomic_units == [
        0,
        1,
        2,
        3,
        4,
        5,
        0,
        1,
        4,
        5,
    ]
    assert ".allocatable_count = 2," in generated.source
    assert "kTestLowCorePhysicalRegisterCandidates" in generated.source
    assert "kTestLowCorePhysicalRegisterAtomicUnits" in generated.source


def test_compiler_derives_ordered_physical_register_views() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    register_ids = {physical_register.name: index for index, physical_register in enumerate(compiled.physical_registers)}
    explicit_class_id = compiled.reg_class_ids["test.explicit32"]

    l0_view = next(view for view in compiled.physical_register_views if view.physical_register_id == register_ids["test.l0"] and view.reg_class_id == explicit_class_id)
    l0_ordinals = compiled.physical_register_view_unit_candidate_ordinals[l0_view.unit_candidate_ordinal_start : l0_view.unit_candidate_ordinal_start + l0_view.unit_count]

    assert l0_ordinals == [0, 2]
    assert compiled.physical_registers[0].atomic_units == (1,)
    assert compiled.physical_registers[1].atomic_units == (0,)


def test_compiler_emits_register_packing_resources() -> None:
    compiled = compiler.compile_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)

    assert len(compiled.register_packing_resources) == 1
    resource = compiled.register_packing_resources[0]
    assert resource.source.name == "test.pair_slots"
    assert resource.source.capacity == 2
    assert resource.member_start == 0
    assert resource.member_count == 2
    assert [compiled.reg_classes[member.reg_class_id].name for member in compiled.register_packing_resource_members] == ["test.packed.narrow", "test.packed.wide"]
    assert [(member.register_unit_count, member.resource_unit_count) for member in compiled.register_packing_resource_members] == [(1, 1), (2, 1)]

    generated = generate_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)
    assert "kTestLowCoreRegisterPackingResources" in generated.source
    assert "kTestLowCoreRegisterPackingResourceMembers" in generated.source
    assert ".register_packing_resource_count = IREE_ARRAYSIZE(" in generated.source


def test_compiler_rejects_invalid_register_packing_resources() -> None:
    resource = TEST_LOW_CORE_DESCRIPTOR_SET.register_packing_resources[0]
    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register packing resource 'test.pair_slots' has zero capacity"),
    ):
        compiler.compile_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                register_packing_resources=(replace(resource, capacity=0),),
            )
        )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register packing resource 'test.pair_slots' member 'test.i32' references a non-physical register class"),
    ):
        compiler.compile_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                register_packing_resources=(
                    RegisterPackingResource(
                        name=resource.name,
                        capacity=resource.capacity,
                        members=(RegisterPackingResourceMember("test.i32"),),
                    ),
                ),
            )
        )


def test_compiler_rejects_unknown_explicit_physical_register() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    register_classes = tuple(replace(register_class, physical_registers=("test.missing",)) if register_class.name == "test.phys" else register_class for register_class in descriptor_set.reg_classes)

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register class 'test.phys' references unknown physical registers: test.missing"),
    ):
        compiler.compile_descriptor_set(replace(descriptor_set, reg_classes=register_classes))


def test_compiler_rejects_overlapping_candidates_in_one_class() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    overlapping_registers = (
        PhysicalRegister("test.r0", (0, 1)),
        PhysicalRegister("test.r1", (1, 2)),
    )
    register_classes = tuple(
        replace(
            register_class,
            physical_registers=("test.r0", "test.r1"),
        )
        if register_class.name == "test.phys"
        else register_class
        for register_class in descriptor_set.reg_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register class 'test.phys' candidates 'test.r0' and 'test.r1' overlap atomic storage unit 1"),
    ):
        compiler.compile_descriptor_set(
            replace(
                descriptor_set,
                physical_registers=overlapping_registers,
                reg_classes=register_classes,
            )
        )


def test_compiler_rejects_mixed_width_explicit_register_candidates() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    physical_registers = (
        PhysicalRegister("test.r0", (0,)),
        *descriptor_set.physical_registers[1:],
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' register class 'test.phys' explicit physical register candidates must occupy the same number of atomic storage units"),
    ):
        compiler.compile_descriptor_set(replace(descriptor_set, physical_registers=physical_registers))


def test_compiler_rejects_noncanonical_physical_register_units() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    physical_registers = (
        PhysicalRegister("test.r0", (1, 0)),
        *descriptor_set.physical_registers[1:],
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' physical register 'test.r0' atomic storage units must be sorted and unique"),
    ):
        compiler.compile_descriptor_set(replace(descriptor_set, physical_registers=physical_registers))


def test_compiler_rejects_duplicate_physical_register_view() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    duplicate_view = descriptor_set.physical_register_views[0]

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' physical register view 'test.pair' as 'test.phys' is duplicated"),
    ):
        compiler.compile_descriptor_set(
            replace(
                descriptor_set,
                physical_register_views=(duplicate_view, duplicate_view),
            )
        )


def test_compiler_rejects_physical_register_view_non_candidate_unit() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    view = replace(
        descriptor_set.physical_register_views[0],
        units=("test.r2", "test.r1"),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' physical register view 'test.pair' as 'test.phys' units are not candidates of the register class: test.r1"),
    ):
        compiler.compile_descriptor_set(replace(descriptor_set, physical_register_views=(view,)))


def test_compiler_rejects_inexact_physical_register_view_storage() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    physical_registers = (
        *descriptor_set.physical_registers[:-1],
        PhysicalRegister("test.pair", (0, 1, 2, 4, 5)),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' physical register view 'test.pair' as 'test.phys' units do not exactly cover aggregate atomic storage"),
    ):
        compiler.compile_descriptor_set(replace(descriptor_set, physical_registers=physical_registers))


def test_compiler_emits_ordered_multiunit_explicit_register_view() -> None:
    descriptor_set = _explicit_physical_descriptor_set()
    descriptor = replace(
        TEST_LOW_ADD_PHYS_DESCRIPTOR,
        operands=(
            replace(TEST_LOW_ADD_PHYS_DESCRIPTOR.operands[0], unit_count=2),
            *TEST_LOW_ADD_PHYS_DESCRIPTOR.operands[1:],
        ),
    )

    compiled = compiler.compile_descriptor_set(replace(descriptor_set, descriptors=(descriptor,)))
    generated = generate_descriptor_set(replace(descriptor_set, descriptors=(descriptor,)))

    assert len(compiled.physical_register_views) == 1
    view = compiled.physical_register_views[0]
    assert view.physical_register_id == 3
    assert view.reg_class_id == compiled.reg_class_ids["test.phys"]
    assert view.unit_count == 2
    assert compiled.physical_register_view_unit_candidate_ordinals == [0, 1]
    assert "kTestLowCorePhysicalRegisterViews" in generated.source
    assert "kTestLowCorePhysicalRegisterViewUnitCandidateOrdinals" in generated.source


def test_compiler_derives_barrier_descriptor_flag() -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_BARRIER_DESCRIPTOR,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert DescriptorFlag.BARRIER in compiled.descriptors[0].flags


def test_compiler_rejects_barrier_flag_without_effect() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        flags=(*TEST_LOW_ADD_I32_DESCRIPTOR.flags, DescriptorFlag.BARRIER),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' has the barrier flag without a barrier effect"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_derives_early_clobber_descriptor_flag() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        constraints=(Constraint(ConstraintKind.EARLY_CLOBBER, 0),),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert DescriptorFlag.EARLY_CLOBBER in compiled.descriptors[0].flags
    assert OperandFlag.EARLY_CLOBBER in compiled.descriptors[0].operands[0].flags


def test_compiler_derives_tied_operand_projection_flags() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert OperandFlag.TIED in compiled.descriptors[0].operands[0].flags
    assert OperandFlag.TIED in compiled.descriptors[0].operands[1].flags
    assert OperandFlag.TIED not in compiled.descriptors[0].operands[2].flags


def _storage_continuation_descriptor() -> Descriptor:
    operands = list(TEST_LOW_WRITE_HIGH16_I32_DESCRIPTOR.operands)
    operands[1] = replace(
        operands[1],
        flags=(
            OperandFlag.IMPLICIT,
            OperandFlag.STORAGE_CONTINUATION,
        ),
    )
    return replace(
        TEST_LOW_WRITE_HIGH16_I32_DESCRIPTOR,
        operands=tuple(operands),
    )


def test_compiler_projects_storage_continuation_to_tied_result() -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(_storage_continuation_descriptor(),),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    result, source = compiled.descriptors[0].operands[:2]
    assert OperandFlag.STORAGE_CONTINUATION in result.flags
    assert OperandFlag.STORAGE_CONTINUATION in source.flags
    assert OperandFlag.TIED in result.flags
    assert OperandFlag.TIED in source.flags


def test_compiler_rejects_authored_storage_continuation_result_projection() -> None:
    descriptor = _storage_continuation_descriptor()
    operands = list(descriptor.operands)
    operands[0] = replace(
        operands[0],
        flags=(OperandFlag.STORAGE_CONTINUATION,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(replace(descriptor, operands=tuple(operands)),),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("authors a result projection; mark only the tied packet input"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_overlapping_storage_continuation_parts() -> None:
    descriptor = _storage_continuation_descriptor()
    operands = list(descriptor.operands)
    operands[1] = replace(
        operands[1],
        register_part=operands[0].register_part,
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(replace(descriptor, operands=tuple(operands)),),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("and tied result have overlapping register parts"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


@pytest.mark.parametrize(
    "projection_flag",
    [OperandFlag.TIED, OperandFlag.EARLY_CLOBBER],
)
def test_compiler_rejects_authored_operand_projection_flags(
    projection_flag: OperandFlag,
) -> None:
    operands = list(TEST_LOW_ADD_I32_DESCRIPTOR.operands)
    operands[0] = replace(
        operands[0],
        flags=(*operands[0].flags, projection_flag),
    )
    descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, operands=tuple(operands))
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor, TEST_LOW_MUL_I32_DESCRIPTOR),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(f"descriptor 'test.add.i32' operand 'dst' authors derived projection flag(s): {projection_flag.name.lower()}"),
    ):
        compiler.compile_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=(TEST_LOW_MUL_I32_DESCRIPTOR.key,)),
        )


def test_compiler_derives_instruction_classes_from_structured_metadata() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        semantic_tag="dot.s8s8.i32x1",
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert compiled.instruction_classes == [(InstructionClass.SCALAR_ALU, InstructionClass.DOT)]


def test_compiler_closes_instruction_class_hierarchies() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        semantic_tag=None,
        instruction_classes=(InstructionClass.SMFMAC,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert compiled.instruction_classes == [
        (
            InstructionClass.SCALAR_ALU,
            InstructionClass.MATRIX,
            InstructionClass.MFMA,
            InstructionClass.SMFMAC,
        )
    ]


def test_compiler_requires_explicit_other_instruction_class() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        instruction_classes=(),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' has no generated instruction class; classify it explicitly or use OTHER"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_other_with_derived_instruction_class() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        semantic_tag="control.return",
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' combines the exclusive OTHER instruction class"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_contradictory_memory_instruction_classes() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        semantic_tag=None,
        effects=(Effect(EffectKind.READ),),
        instruction_classes=(
            InstructionClass.GLOBAL_MEMORY,
            InstructionClass.PRIVATE_MEMORY,
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' combines private and global memory instruction classes"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_load_instruction_class_without_read_effect() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        semantic_tag=None,
        instruction_classes=(InstructionClass.GLOBAL_LOAD,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' has a load instruction class without a read effect"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_early_clobber_flag_without_constraint() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        flags=(
            *TEST_LOW_ADD_I32_DESCRIPTOR.flags,
            DescriptorFlag.EARLY_CLOBBER,
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' has the early-clobber flag without an early-clobber constraint"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_projects_validated_rematerializable_results() -> None:
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_CONST_I32_DESCRIPTOR,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert compiled.operand_rematerializable == [True]


def test_compiler_rejects_rematerializable_result_without_dead_removal() -> None:
    descriptor = replace(TEST_LOW_CONST_I32_DESCRIPTOR, flags=())
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' rematerializable result 0 requires the dead-removable flag"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_effectful_rematerializable_result() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        effects=(Effect(EffectKind.READ),),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' rematerializable result 0 requires an effect-free descriptor"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_compiler_rejects_rematerialization_across_target_state() -> None:
    descriptor = replace(
        TEST_LOW_STATE_ADD_SCHEDULE_STATE_DESCRIPTOR,
        constraints=(Constraint(ConstraintKind.REMATERIALIZABLE, 0),),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.state.add.schedule_state' rematerializable result 0 cannot replay target state operand 'state_out'"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_allowlist_closes_over_operand_form_replacements() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    replacement_descriptor = replace(
        base_descriptor,
        key="test.add.i32.rhs_zero",
        mnemonic="test.add.i32.rhs_zero",
        operands=base_descriptor.operands[:2],
        asm_forms=(AsmForm(results=("dst",), operands=("lhs",)),),
    )
    source_descriptor = replace(
        base_descriptor,
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement_descriptor.key,
                matches=(
                    OperandFormMatch(
                        source_operand="rhs",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor, replacement_descriptor),
    )

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=(source_descriptor.key,)),
    )

    assert source_descriptor.key in generated.source
    assert replacement_descriptor.key in generated.source
    assert ".match_kind = LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64" in generated.source


def test_schedule_alternatives_are_validated_closed_and_remapped() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    alternative_descriptor = replace(
        base_descriptor,
        key="test.add.i32.pipeline_b",
        mnemonic="test.add.i32.pipeline_b",
        encoding_id=2,
    )
    source_descriptor = replace(
        base_descriptor,
        schedule_alternatives=(alternative_descriptor.key,),
        encoding_id=1,
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor, alternative_descriptor),
    )

    compiled = compiler.compile_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=(source_descriptor.key,)),
    )

    assert [descriptor.key for descriptor in compiled.descriptors] == [
        source_descriptor.key,
        alternative_descriptor.key,
    ]
    assert compiled.schedule_alternative_rows == [(0, 1)]

    view_spec = replace(
        descriptor_set,
        key="test.low.schedule_alternative_view",
        c_table_prefix="TestLowScheduleAlternativeView",
        descriptors=(alternative_descriptor, source_descriptor),
    )
    compiled_view = views.descriptor_set_view_for_spec(compiled, view_spec)
    assert compiled_view.schedule_alternative_rows == [(1, 0)]
    assert not compiled_view.uses_storage_schedule_alternative_tables

    generated_family = generate_descriptor_set_family(
        descriptor_set,
        (view_spec,),
    )
    assert "kTestLowScheduleAlternativeViewScheduleAlternatives" in (generated_family.source)
    assert ".source_descriptor_ordinal = 1" in generated_family.source
    assert ".alternative_descriptor_ordinal = 0" in generated_family.source

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=(source_descriptor.key,)),
    )
    assert "loom_low_schedule_alternative_t" in generated.source
    assert ".source_descriptor_ordinal = 0" in generated.source
    assert ".alternative_descriptor_ordinal = 1" in generated.source


def test_schedule_alternative_rejects_semantic_changes() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    alternative_descriptor = replace(
        base_descriptor,
        key="test.sub.i32.pipeline_b",
        mnemonic="test.sub.i32.pipeline_b",
        semantic_tag="integer.sub.i32",
    )
    source_descriptor = replace(
        base_descriptor,
        schedule_alternatives=(alternative_descriptor.key,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor, alternative_descriptor),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' schedule alternative 'test.sub.i32.pipeline_b' changes semantic, allocation, or timing facts"),
    ):
        compiler.compile_descriptor_set(descriptor_set)


def test_operand_forms_preserve_assembly_implicit_packet_sources() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    hidden_lhs = replace(base_descriptor.operands[1], flags=(OperandFlag.IMPLICIT,))
    replacement_descriptor = replace(
        base_descriptor,
        key="test.add.i32.hidden_lhs",
        mnemonic="test.add.i32.hidden_lhs",
        operands=(base_descriptor.operands[0], hidden_lhs),
        asm_forms=(AsmForm(results=("dst",), operands=("lhs",)),),
    )
    source_descriptor = replace(
        base_descriptor,
        operands=(
            base_descriptor.operands[0],
            hidden_lhs,
            base_descriptor.operands[2],
        ),
        operand_forms=(
            OperandForm(
                replacement_descriptor=replacement_descriptor.key,
                matches=(
                    OperandFormMatch(
                        source_operand="rhs",
                        match_kind=OperandFormMatchKind.ALL_EQUAL_I64,
                        match_i64=0,
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor, replacement_descriptor),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)

    assert compiled.operand_form_matches[0].source_packet_operand_index == 1
    assert compiled.operand_form_operand_indices == [0]


def test_descriptor_set_family_emits_one_storage_table_and_ordered_headers() -> None:
    base_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[:1],
    )
    extension_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.extension.core",
        function_name="loom_test_low_extension_core_descriptor_set",
        c_table_prefix="TestLowExtensionCore",
        c_enum_prefix="TEST_LOW_EXTENSION_CORE",
        descriptors=TEST_LOW_CORE_DESCRIPTOR_SET.descriptors[:2],
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=extension_view.descriptors,
    )

    compiled_view = views.descriptor_set_view_for_spec(
        compiler.compile_descriptor_set(storage_set),
        base_view,
    )
    assert compiled_view.uses_storage_descriptor_tables
    assert compiled_view.uses_storage_descriptor_view_tables
    assert compiled_view.uses_storage_asm_form_tables
    assert compiled_view.uses_storage_operand_form_tables
    assert len(compiled_view.canonical_asm_form_ordinals) == 1

    generated = generate_descriptor_set_family(
        storage_set,
        (base_view, extension_view),
    )
    source = generated.source

    assert source.count("static const loom_low_descriptor_t kTestLowCoreDescriptors[]") == 1
    assert "kTestLowExtensionCoreDescriptors" not in source
    assert ".descriptors = kTestLowCoreDescriptors," in source
    assert ".descriptor_refs = kTestLowCoreDescriptorRefs," in source
    assert ".descriptor_refs = kTestLowExtensionCoreDescriptorRefs," in source
    assert "kTestLowExtensionCoreAsmForms" not in source
    assert "kTestLowExtensionCoreOperandForms" not in source
    assert source.count(".asm_forms = kTestLowCoreAsmForms,") == 2
    assert ".descriptor_count = 1," in source
    assert ".descriptor_count = 2," in source
    assert ("const loom_low_descriptor_set_t* loom_test_low_extension_core_descriptor_set(void)") in source
    assert "loom_test_low_core_descriptor_set(void)" in generated.view_headers[0]
    assert "loom_test_low_extension_core_descriptor_set(void)" in generated.view_headers[1]


def test_descriptor_set_view_selects_shared_schedule_class() -> None:
    scalar_schedule = TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes[1]
    vector_schedule = TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes[2]
    assert TEST_LOW_ADD_I32_DESCRIPTOR.schedule_class == scalar_schedule.name

    vector_multiply = replace(
        TEST_LOW_MUL_I32_DESCRIPTOR,
        schedule_class=vector_schedule.name,
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR, vector_multiply),
    )
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.schedule_view.core",
        function_name="loom_test_low_schedule_view_core_descriptor_set",
        c_table_prefix="TestLowScheduleViewCore",
        c_enum_prefix="TEST_LOW_SCHEDULE_VIEW_CORE",
        descriptors=(
            replace(
                TEST_LOW_ADD_I32_DESCRIPTOR,
                schedule_class=vector_schedule.name,
            ),
        ),
    )

    compiled_view = views.descriptor_set_view_for_spec(
        compiler.compile_descriptor_set(storage_set),
        view,
    )

    assert compiled_view.uses_storage_descriptor_tables
    assert not compiled_view.uses_storage_descriptor_view_tables
    assert compiled_view.descriptors[0].schedule_class == vector_schedule.name
    assert compiled_view.instruction_classes == ((InstructionClass.VECTOR_ALU,),)


def test_descriptor_set_view_reuses_schedule_independent_storage_tables() -> None:
    vector_schedule = TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes[2]
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
            TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
        ),
    )
    view = replace(
        storage_set,
        key="test.low.schedule_view.core",
        function_name="loom_test_low_schedule_view_core_descriptor_set",
        c_table_prefix="TestLowScheduleViewCore",
        c_enum_prefix="TEST_LOW_SCHEDULE_VIEW_CORE",
        descriptors=(
            replace(
                TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
                schedule_class=vector_schedule.name,
            ),
            TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
        ),
    )
    compiled = compiler.compile_descriptor_set(
        storage_set,
        required_schedule_class_names=(vector_schedule.name,),
    )

    compiled_view = views.descriptor_set_view_for_spec(compiled, view)

    assert compiled_view.uses_storage_descriptor_tables
    assert not compiled_view.uses_storage_descriptor_view_tables
    assert compiled_view.uses_storage_asm_form_tables
    assert compiled_view.uses_storage_operand_form_tables
    assert compiled_view.asm_forms is compiled.asm_forms
    assert compiled_view.operand_forms is compiled.operand_forms

    source = generate_descriptor_set_family(storage_set, (view,)).source
    assert "kTestLowScheduleViewCoreDescriptors" not in source
    assert "kTestLowScheduleViewCoreDescriptorViews" in source
    assert "kTestLowScheduleViewCoreAsmForms" not in source
    assert "kTestLowScheduleViewCoreOperandForms" not in source
    assert ".asm_forms = kTestLowCoreAsmForms," in source
    assert ".operand_forms = kTestLowCoreOperandForms," in source


def test_descriptor_set_view_rejects_local_schedule_definition() -> None:
    vector_schedule = TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes[2]
    vector_multiply = replace(
        TEST_LOW_MUL_I32_DESCRIPTOR,
        schedule_class=vector_schedule.name,
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR, vector_multiply),
    )
    view_schedule_classes = tuple(
        replace(schedule_class, latency_cycles=3) if schedule_class.name == vector_schedule.name else schedule_class for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.schedule_view.core",
        function_name="loom_test_low_schedule_view_core_descriptor_set",
        c_table_prefix="TestLowScheduleViewCore",
        c_enum_prefix="TEST_LOW_SCHEDULE_VIEW_CORE",
        schedule_classes=view_schedule_classes,
        descriptors=(
            replace(
                TEST_LOW_ADD_I32_DESCRIPTOR,
                schedule_class=vector_schedule.name,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set view 'test.low.schedule_view.core' schedule class 'test.vector.alu' differs from storage set 'test.low.core'"),
    ):
        views.descriptor_set_view_for_spec(
            compiler.compile_descriptor_set(storage_set),
            view,
        )


def test_descriptor_set_family_emits_prefix_view_local_asm_forms() -> None:
    storage_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                mnemonic="storage.add.i32",
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(AsmResultValueType(ScalarTypeKind.I32),),
            ),
        ),
    )
    view_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                mnemonic="add.i32",
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(AsmResultValueType(ScalarTypeKind.I32),),
            ),
        ),
    )
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.view.core",
        function_name="loom_test_low_view_core_descriptor_set",
        c_table_prefix="TestLowViewCore",
        c_enum_prefix="TEST_LOW_VIEW_CORE",
        descriptors=(view_descriptor,),
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(storage_descriptor,),
    )

    source = generate_descriptor_set_family(
        storage_set,
        (view, storage_set),
    ).source

    assert "storage.add.i32" in source
    assert '"add.i32"' in source
    assert "static const loom_low_descriptor_t kTestLowViewCoreDescriptors[]" not in source
    assert "kTestLowViewCoreDescriptorViews" not in source
    assert "static const loom_low_asm_form_t kTestLowViewCoreAsmForms[]" in source
    assert ".descriptors = kTestLowCoreDescriptors," in source
    assert ".descriptor_views = kTestLowCoreDescriptorViews," in source
    assert ".asm_forms = kTestLowViewCoreAsmForms," in source
    assert source.count(".kind = LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_SCALAR,") == 1
    assert source.count(".result_value_type_start = 0,") == 2


def test_descriptor_set_family_compares_derived_descriptor_projections() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        constraints=(Constraint(ConstraintKind.EARLY_CLOBBER, 0),),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    source = generate_descriptor_set_family(
        descriptor_set,
        (descriptor_set,),
    ).source

    assert ".flags = LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER," in source


def test_descriptor_set_family_rejects_view_descriptor_contract_drift() -> None:
    view_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        semantic_tag="test.changed.add.i32",
    )
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.view.core",
        descriptors=(view_descriptor,),
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "descriptor set view 'test.low.view.core' descriptor 'test.add.i32' differs from storage descriptor 'test.add.i32' outside of asm forms, asm surface policy, or schedule class"
        ),
    ):
        generate_descriptor_set_family(storage_set, (view,))


def test_descriptor_set_family_requires_view_canonical_asm_for_authorable_surface() -> None:
    view_descriptor = replace(TEST_LOW_ADD_I32_DESCRIPTOR, asm_forms=())
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.view.core",
        function_name="loom_test_low_view_core_descriptor_set",
        c_table_prefix="TestLowViewCore",
        c_enum_prefix="TEST_LOW_VIEW_CORE",
        descriptors=(view_descriptor,),
        requires_explicit_asm_surface=True,
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set view 'test.low.view.core' descriptor 'test.add.i32' is authorable asm but does not declare exactly one canonical asm form; found 0"),
    ):
        generate_descriptor_set_family(
            storage_set,
            (view,),
        )


def test_descriptor_set_family_allows_view_local_non_authorable_surface() -> None:
    view_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(),
        asm_surface=DescriptorAsmSurface.STRUCTURAL,
        asm_surface_reason="printed by structural low asm syntax",
    )
    view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.view.core",
        function_name="loom_test_low_view_core_descriptor_set",
        c_table_prefix="TestLowViewCore",
        c_enum_prefix="TEST_LOW_VIEW_CORE",
        descriptors=(view_descriptor,),
        requires_explicit_asm_surface=True,
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR,),
    )

    source = generate_descriptor_set_family(
        storage_set,
        (view,),
    ).source

    assert "static const loom_low_descriptor_t kTestLowViewCoreDescriptors[]" not in source
    assert ("static const loom_low_descriptor_view_t kTestLowViewCoreDescriptorViews[]") in source
    assert ".canonical_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE" in source


def test_descriptor_set_family_emits_sibling_view_descriptor_surfaces() -> None:
    first_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(TEST_LOW_CONST_I32_DESCRIPTOR,),
    )
    sibling_view = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        key="test.low.sibling.core",
        function_name="loom_test_low_sibling_core_descriptor_set",
        c_table_prefix="TestLowSiblingCore",
        c_enum_prefix="TEST_LOW_SIBLING_CORE",
        descriptors=(TEST_LOW_ADD_I32_DESCRIPTOR,),
    )
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            TEST_LOW_CONST_I32_DESCRIPTOR,
            TEST_LOW_ADD_I32_DESCRIPTOR,
        ),
    )

    source = generate_descriptor_set_family(
        storage_set,
        (first_view, sibling_view, storage_set),
    ).source

    assert source.count("static const loom_low_operand_t kTestLowCoreOperands[]") == 1
    assert "static const loom_low_descriptor_t kTestLowSiblingCoreDescriptors[]" in source
    assert "static const loom_low_asm_form_t kTestLowSiblingCoreAsmForms[]" in source
    assert ".descriptors = kTestLowSiblingCoreDescriptors," in source
    assert ".asm_forms = kTestLowSiblingCoreAsmForms," in source
    assert ".descriptor_refs = kTestLowSiblingCoreDescriptorRefs," in source
    assert ".descriptor_count = 1," in source
    assert ".descriptor_ordinal = 0," in source
    assert "test.mul.i32" not in source


def test_descriptor_set_family_shares_exact_sibling_view_tables() -> None:
    storage_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            TEST_LOW_CONST_I32_DESCRIPTOR,
            TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
            TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
        ),
    )
    first_view = replace(
        storage_set,
        key="test.low.first.core",
        function_name="loom_test_low_first_core_descriptor_set",
        c_table_prefix="TestLowFirstCore",
        c_enum_prefix="TEST_LOW_FIRST_CORE",
        descriptors=(
            TEST_LOW_STATE_ADD_I32_DESCRIPTOR,
            TEST_LOW_STATE_ADD_I32_RHS_ZERO_DESCRIPTOR,
        ),
    )
    second_view = replace(
        first_view,
        key="test.low.second.core",
        function_name="loom_test_low_second_core_descriptor_set",
        c_table_prefix="TestLowSecondCore",
        c_enum_prefix="TEST_LOW_SECOND_CORE",
    )

    source = generate_descriptor_set_family(
        storage_set,
        (first_view, second_view),
    ).source

    table_fields = (
        ("loom_low_descriptor_t", "Descriptors", "descriptors"),
        (
            "loom_low_descriptor_view_t",
            "DescriptorViews",
            "descriptor_views",
        ),
        ("loom_low_operand_form_t", "OperandForms", "operand_forms"),
        ("loom_low_descriptor_ref_t", "DescriptorRefs", "descriptor_refs"),
        ("loom_low_asm_form_t", "AsmForms", "asm_forms"),
    )
    for c_type, table_suffix, field_name in table_fields:
        first_table_symbol = f"kTestLowFirstCore{table_suffix}"
        second_table_symbol = f"kTestLowSecondCore{table_suffix}"
        assert source.count(f"static const {c_type} {first_table_symbol}[]") == 1
        assert f"static const {c_type} {second_table_symbol}[]" not in source
        assert source.count(f".{field_name} = {first_table_symbol},") == 2


def test_generate_test_low_core_descriptor_set() -> None:
    generated = generate_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)

    assert '"test.low.core"' in generated.source
    assert '"test.low"' in generated.source
    assert '"test.spv.op_iadd.i32"' in generated.source
    assert '"OpIAdd"' in generated.source
    assert ".op_kind = LOOM_LOW_DESCRIPTOR_OP_KIND_CONST," in generated.source
    assert (".instruction_class_flags = LOOM_LOW_INSTRUCTION_CLASS_FLAG_SCALAR_ALU") in generated.source
    assert ".kind = LOOM_LOW_ISSUE_USE_KIND_REQUIRED," in generated.source
    assert ".kind = LOOM_LOW_ISSUE_USE_KIND_RESERVED," in generated.source


def test_generator_resolves_symbolic_hazard_resources() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(
                Hazard(
                    HazardKind.MIN_DISTANCE,
                    resource="test.scalar",
                    producer_stage=0,
                    consumer_stage=1,
                    distance=2,
                ),
            ),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    generated = generate_descriptor_set(
        descriptor_set,
        DescriptorAllowlist(keys=("test.add.i32",)),
    )

    assert ".reference_kind = LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE" in generated.source
    assert ".reference_id = 0" in generated.source


def test_generator_rejects_unknown_hazard_resource() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="missing"),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("schedule class 'test.scalar.alu' hazard references unknown resource 'missing'"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_generator_rejects_ambiguous_hazard_reference() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            hazards=(Hazard(HazardKind.MIN_DISTANCE, resource="test.scalar", counter_id=0),),
        )
        if schedule_class.name == "test.scalar.alu"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        schedule_classes=schedule_classes,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("schedule class 'test.scalar.alu' hazard must reference exactly one resource, counter, or target id"),
    ):
        generate_descriptor_set(
            descriptor_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_allowlist_accepts_semantic_tags() -> None:
    generated = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(semantic_tags=("control.return.void",)),
    )

    assert "test.return.void" in generated.source
    assert "test.add.i32" not in generated.source


def test_allowlist_rejects_unknown_descriptor_key() -> None:
    with pytest.raises(ValueError, match="allowlist references unknown descriptor key 'missing'"):
        generate_descriptor_set(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            DescriptorAllowlist(keys=("missing",)),
        )


def test_generator_rejects_asm_form_unknown_operand_field() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "missing")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' operand references unknown operand field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_trailing_variadic_operand_segment() -> None:
    lhs, rhs = TEST_LOW_ADD_I32_DESCRIPTOR.operands[1:]
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            lhs,
            replace(rhs, flags=(OperandFlag.VARIADIC,)),
        ),
        asm_forms=(
            AsmForm(
                results=("dst",),
                operand_segments=(
                    AsmOperandSegment(
                        AsmOperandSegmentDelimiter.PAREN,
                        ("lhs", "rhs"),
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)
    generated = generate_descriptor_set(descriptor_set)

    assert compiled.asm_forms[0].operand_indices == (1, 2)
    assert compiled.asm_forms[0].operand_segment_start == 0
    assert compiled.asm_table_storage.operand_segments[0].operand_count == 2
    assert compiled.asm_table_storage.operand_segments[0].has_variadic_operand
    assert compiled.descriptor_rows[0]["minimum_packet_operand_count"] == 1
    assert DescriptorFlag.VARIADIC_OPERANDS in compiled.descriptors[0].flags
    assert "LOOM_LOW_OPERAND_FLAG_VARIADIC" in generated.source
    assert ".minimum_packet_operand_count = 1," in generated.source
    assert "LOOM_LOW_DESCRIPTOR_FLAG_VARIADIC_OPERANDS" in generated.source
    assert "static const loom_low_asm_operand_segment_t kTestLowCoreAsmOperandSegments[]" in generated.source
    assert ".delimiter = LOOM_LOW_ASM_OPERAND_SEGMENT_DELIMITER_PAREN," in generated.source
    assert ".flags = LOOM_LOW_ASM_OPERAND_SEGMENT_FLAG_VARIADIC," in generated.source
    assert ".asm_operand_segments = kTestLowCoreAsmOperandSegments," in generated.source


def test_generator_rejects_non_trailing_variadic_operand() -> None:
    lhs, rhs = TEST_LOW_ADD_I32_DESCRIPTOR.operands[1:]
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            replace(lhs, flags=(OperandFlag.VARIADIC,)),
            rhs,
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' variadic operand 'lhs' must be the final descriptor operand"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_result_with_operand_role() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("lhs",), operands=("rhs",)),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' result field 'lhs' names a non-result operand"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_exact_asm_result_value_type() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(AsmResultValueType(ScalarTypeKind.I32, vector_lane_count=4),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    compiled = compiler.compile_descriptor_set(descriptor_set)
    generated = generate_descriptor_set(descriptor_set)

    assert compiled.asm_forms[0].result_value_type_start == 0
    assert compiled.asm_table_storage.result_value_types == [AsmResultValueType(ScalarTypeKind.I32, vector_lane_count=4)]
    assert "static const loom_low_asm_result_value_type_t kTestLowCoreAsmResultValueTypes[]" in generated.source
    assert ".kind = LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_VECTOR," in generated.source
    assert ".element_type = LOOM_SCALAR_TYPE_I32," in generated.source
    assert ".vector_lane_count = 4," in generated.source
    assert ".result_value_type_start = 0," in generated.source
    assert ".asm_result_value_types = kTestLowCoreAsmResultValueTypes," in generated.source


def test_generator_emits_partial_multi_result_value_type_recipe() -> None:
    base = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base,
        operands=(
            base.operands[0],
            replace(base.operands[0], field_name="carry"),
            *base.operands[1:],
        ),
        asm_forms=(
            AsmForm(
                results=("dst", "carry"),
                operands=("lhs", "rhs"),
                result_value_types=(
                    AsmResultValueType(ScalarTypeKind.I32),
                    None,
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    compiled = compiler.compile_descriptor_set(descriptor_set)
    generated = generate_descriptor_set(descriptor_set)

    assert compiled.asm_table_storage.result_value_types == [
        AsmResultValueType(ScalarTypeKind.I32),
        None,
    ]
    assert ".kind = LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_SCALAR," in generated.source
    assert ".kind = LOOM_LOW_ASM_RESULT_VALUE_TYPE_KIND_NONE," in generated.source


def test_generator_rejects_misaligned_asm_result_value_types() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(
                    AsmResultValueType(ScalarTypeKind.I32),
                    AsmResultValueType(ScalarTypeKind.I32),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' has 2 result value types for 1 results"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_empty_asm_result_value_type_recipe() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(None,),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' has an empty result value type recipe"),
    ):
        generate_descriptor_set(descriptor_set)


@pytest.mark.parametrize("lane_count", [-1, 2**16])
def test_asm_result_value_type_rejects_lane_count_outside_u16(
    lane_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="asm result vector lane count must fit u16",
    ):
        AsmResultValueType(ScalarTypeKind.I32, lane_count)


def test_asm_result_value_type_requires_scalar_type_kind() -> None:
    with pytest.raises(
        ValueError,
        match="asm result element type must be a ScalarTypeKind",
    ):
        AsmResultValueType(5)  # type: ignore[arg-type]


def test_generator_rejects_exact_type_for_tied_asm_result() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                result_value_types=(AsmResultValueType(ScalarTypeKind.I32),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' result 0 is operand-inferred and must use the exact operand type"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_low_const_with_no_result() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        operands=(),
        asm_forms=(AsmForm(),),
        constraints=(),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' uses low.const but declares 0 results instead of exactly one"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_low_const_with_packet_operand() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_CONST_I32_DESCRIPTOR.operands[0],
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' uses low.const but declares packet operands"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_accepts_low_const_with_implicit_schedule_state() -> None:
    result = TEST_LOW_CONST_I32_DESCRIPTOR.operands[0]
    schedule_state = replace(
        result,
        field_name="exec_in",
        role=OperandRole.IMPLICIT,
        flags=(
            OperandFlag.IMPLICIT,
            OperandFlag.STATE_READ,
            OperandFlag.SCHEDULE_ONLY_STATE,
        ),
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        operands=(result, schedule_state),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generate_descriptor_set(descriptor_set)


def test_generator_rejects_effectful_low_const() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        constraints=(),
        effects=(Effect(EffectKind.CONVERGENT),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' uses low.const but declares effects"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_low_const_asm_operand() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("dst",), operands=("dst",)),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' low.const asm form 'test.const.i32' must expose exactly one result and no operands"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_accepts_asm_form_implicit_packet_operand() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            base_descriptor.operands[0],
            replace(base_descriptor.operands[1], flags=(OperandFlag.IMPLICIT,)),
            *base_descriptor.operands[2:],
        ),
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "LOOM_LOW_OPERAND_FLAG_IMPLICIT" in generated.source
    assert ".operand_index_count = 2," in generated.source


def test_compiler_indexes_every_descriptor_source_value_role() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    source_descriptor = replace(
        base_descriptor,
        key="test.source.coordinates",
        mnemonic="test.source.coordinates",
        semantic_tag="test.source.coordinates",
        operands=(
            replace(base_descriptor.operands[0], field_name="dst0"),
            replace(base_descriptor.operands[0], field_name="dst1"),
            replace(base_descriptor.operands[1], field_name="value"),
            replace(
                TEST_LOW_COND_BR_I32_DESCRIPTOR.operands[0],
                field_name="predicate",
            ),
            replace(
                TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR.operands[1],
                field_name="hidden_resource",
                flags=(OperandFlag.IMPLICIT,),
            ),
            replace(
                TEST_LOW_STATE_ADD_SCHEDULE_STATE_DESCRIPTOR.operands[-1],
                field_name="target_state",
            ),
        ),
        constraints=(
            Constraint(ConstraintKind.TIED, 0, 2),
            Constraint(ConstraintKind.EARLY_CLOBBER, 1),
        ),
        asm_forms=(
            AsmForm(
                results=("dst0", "dst1"),
                operands=("value", "predicate", "hidden_resource"),
            ),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(source_descriptor,),
    )

    compiled = compiler.compile_descriptor_set(descriptor_set)
    generated = generate_descriptor_set(descriptor_set)

    assert compiled.operand_source_value_indices == [0, 1, 0, 1, 2, None]
    assert ".source_value_index = 2," in generated.source
    assert ".source_value_index = LOOM_LOW_ID_NONE," in generated.source


@pytest.mark.parametrize(
    ("field_name", "role", "expected_binding"),
    [
        ("lhs", OperandRole.OPERAND, OperandSourceBinding.LHS),
        ("rhs", OperandRole.OPERAND, OperandSourceBinding.RHS),
        ("acc", OperandRole.OPERAND, OperandSourceBinding.ACCUMULATOR),
        (
            "sparse_metadata",
            OperandRole.OPERAND,
            OperandSourceBinding.SPARSE_METADATA,
        ),
        ("lhs_scale", OperandRole.OPERAND, OperandSourceBinding.LHS_SCALE),
        ("rhs_scale", OperandRole.OPERAND, OperandSourceBinding.RHS_SCALE),
        ("value", OperandRole.OPERAND, OperandSourceBinding.NONE),
        ("lhs", OperandRole.RESULT, OperandSourceBinding.NONE),
    ],
)
def test_operand_source_binding_uses_canonical_field_names(field_name: str, role: OperandRole, expected_binding: OperandSourceBinding) -> None:
    assert operand_source_binding(field_name, role) is expected_binding


def test_generator_compiles_canonical_operand_source_bindings() -> None:
    generated = generate_descriptor_set(TEST_LOW_CORE_DESCRIPTOR_SET)

    assert ".source_binding = LOOM_LOW_OPERAND_SOURCE_BINDING_LHS," in generated.source
    assert ".source_binding = LOOM_LOW_OPERAND_SOURCE_BINDING_RHS," in generated.source
    assert ".source_binding = LOOM_LOW_OPERAND_SOURCE_BINDING_NONE," in generated.source


def test_generator_rejects_ambiguous_asm_mnemonics() -> None:
    first = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    second = replace(
        TEST_LOW_MUL_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="dup", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(first, second))

    with pytest.raises(
        ValueError,
        match=re.escape("asm mnemonic 'dup' is ambiguous between descriptors 'test.add.i32' and 'test.mul.i32'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_unknown_immediate_field() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        asm_forms=(AsmForm(results=("dst",), immediates=(AsmImmediate("missing"),)),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' asm form 'test.const.i32' immediate references unknown immediate field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_duplicate_asm_immediate_name() -> None:
    descriptor = replace(
        TEST_LOW_COND_BR_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                operands=("cond",),
                immediates=(
                    AsmImmediate("true_block", name="target"),
                    AsmImmediate("false_block", name="target"),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.cond_br.i32' asm form 'test.cond_br.i32' uses immediate name 'target' more than once"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_asm_form_without_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        mnemonic=None,
        asm_forms=(AsmForm(results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form must specify a mnemonic because the descriptor has no default mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_empty_asm_form_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(AsmForm(mnemonic="", results=("dst",), operands=("lhs", "rhs")),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form specifies an empty mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_asm_form_native_assembly_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                mnemonic="test.add.i32_low",
                native_assembly_mnemonic="test.add.i32",
                results=("dst",),
                operands=("lhs", "rhs"),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert '"test.add.i32_low"' in generated.source
    assert '"test.add.i32"' in generated.source
    assert ".native_assembly_mnemonic_string_offset = " in generated.source
    assert ".native_assembly_mnemonic_string_offset = LOOM_LOW_STRING_OFFSET_NONE" not in generated.source


def test_generator_emits_asm_form_native_assembly_values() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                mnemonic="test.add.i32.native",
                native_assembly_mnemonic="test.add.i32",
                results=("dst",),
                operands=("lhs", "rhs"),
                native_assembly_values=(
                    NativeAsmValue(NativeAsmValueKind.RESULT, field_name="dst"),
                    NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="lhs"),
                    NativeAsmValue(NativeAsmValueKind.LITERAL, literal="literal"),
                    NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="rhs"),
                    NativeAsmValue(
                        NativeAsmValueKind.MODIFIER_LITERAL,
                        literal="modifier:1",
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "static const loom_low_native_asm_value_t kTestLowCoreNativeAsmValues[]" in generated.source
    assert ".native_assembly_value_count = 5," in generated.source
    assert ".native_asm_values = kTestLowCoreNativeAsmValues," in generated.source
    assert "LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT" in generated.source
    assert "LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL" in generated.source
    assert "LOOM_LOW_NATIVE_ASM_VALUE_KIND_MODIFIER_LITERAL" in generated.source
    assert '"literal"' in generated.source
    assert '"modifier:1"' in generated.source


def test_generator_emits_native_register_part_values() -> None:
    descriptor = replace(
        TEST_LOW_WRITE_LOW16_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("address",),
                native_assembly_values=(
                    NativeAsmValue(
                        NativeAsmValueKind.REGISTER_PART,
                        field_name="dst",
                    ),
                    NativeAsmValue(
                        NativeAsmValueKind.OPERAND,
                        field_name="address",
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "LOOM_LOW_NATIVE_ASM_VALUE_KIND_REGISTER_PART" in generated.source


def test_generator_emits_target_native_asm_immediate_values() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        immediates=(
            Immediate(
                "delay",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                unsigned_max=0x07FF,
            ),
        ),
        asm_forms=(
            AsmForm(
                native_assembly_values=(
                    NativeAsmValue(
                        NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                        field_name="delay",
                        literal="delay_bits",
                        bit_width=16,
                        target_format_id=1,
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_TARGET_FORMAT" in generated.source
    assert ".index = 0," in generated.source
    assert ".bit_width = 16," in generated.source
    assert ".target_format_id = 1," in generated.source
    assert '"delay_bits"' in generated.source


def test_generator_rejects_target_native_asm_immediate_oversized_bit_width() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        immediates=(
            Immediate(
                "delay",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                unsigned_max=0x07FF,
            ),
        ),
        asm_forms=(
            AsmForm(
                native_assembly_values=(
                    NativeAsmValue(
                        NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                        field_name="delay",
                        bit_width=256,
                        target_format_id=1,
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' native target-format immediate 'delay' bit width must be in [0, 255]"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_target_native_asm_immediate_missing_format() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        immediates=(
            Immediate(
                "delay",
                ImmediateKind.UNSIGNED,
                bit_width=16,
                unsigned_max=0x07FF,
            ),
        ),
        asm_forms=(
            AsmForm(
                native_assembly_values=(
                    NativeAsmValue(
                        NativeAsmValueKind.IMMEDIATE_TARGET_FORMAT,
                        field_name="delay",
                    ),
                ),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' native target-format immediate 'delay' target format must be in [1, 255]"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_native_asm_form_unknown_operand_field() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                native_assembly_values=(NativeAsmValue(NativeAsmValueKind.OPERAND, field_name="missing"),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' native operand references unknown operand field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_native_asm_form_unknown_immediate_field() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                results=("dst",),
                operands=("lhs", "rhs"),
                native_assembly_values=(NativeAsmValue(NativeAsmValueKind.IMMEDIATE_I64, field_name="missing"),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' native immediate references unknown immediate field 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_empty_asm_form_native_assembly_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                native_assembly_mnemonic="",
                results=("dst",),
                operands=("lhs", "rhs"),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' has an empty native assembly mnemonic"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_redundant_asm_form_native_assembly_mnemonic() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        asm_forms=(
            AsmForm(
                native_assembly_mnemonic="test.add.i32",
                results=("dst",),
                operands=("lhs", "rhs"),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' asm form 'test.add.i32' repeats its low asm mnemonic as a native assembly override"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_resource() -> None:
    bad_resource_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, resources=())

    with pytest.raises(
        ValueError,
        match=re.escape("schedule class 'test.scalar.alu' references unknown resource 'test.scalar'"),
    ):
        generate_descriptor_set(
            bad_resource_set,
            DescriptorAllowlist(keys=("test.add.i32",)),
        )


def test_generator_emits_compact_timing_event_tables() -> None:
    compiled = compiler.compile_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(
            keys=(
                "test.event.fast.i32",
                "test.event.consume.early.i32",
                "test.event.consume.late.i32",
            )
        ),
    )
    generated = generate_descriptor_set(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        DescriptorAllowlist(
            keys=(
                "test.event.fast.i32",
                "test.event.consume.early.i32",
                "test.event.consume.late.i32",
            )
        ),
    )

    assert [event.name for event in compiled.timing_events] == [
        "test.write.fast",
        "test.read.early",
        "test.read.late",
    ]
    assert [separation.minimum_issue_separation_cycles for separation in compiled.event_separations] == [0, -2]
    assert "kTestLowCoreTimingEvents" in generated.source
    assert "kTestLowCoreEventSeparations" in generated.source
    assert ".minimum_issue_separation_cycles = -2," in generated.source


def test_generator_rejects_duplicate_schedule_resource() -> None:
    resource = TEST_LOW_CORE_DESCRIPTOR_SET.resources[0]
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        resources=(*TEST_LOW_CORE_DESCRIPTOR_SET.resources, resource),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' resource 'test.scalar' is duplicated"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_zero_schedule_resource_capacity() -> None:
    resources = tuple(replace(resource, capacity_per_cycle=0) if resource.name == "test.shared_a" else resource for resource in TEST_LOW_CORE_DESCRIPTOR_SET.resources)

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' resource 'test.shared_a' has zero per-cycle capacity"),
    ):
        generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, resources=resources))


def test_generator_rejects_mismatched_contention_group_capacity() -> None:
    resources = tuple(replace(resource, capacity_per_cycle=3) if resource.name == "test.shared_b" else resource for resource in TEST_LOW_CORE_DESCRIPTOR_SET.resources)

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' resource 'test.shared_b' capacity 3 does not match contention group 1 capacity 2"),
    ):
        generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, resources=resources))


def test_generator_rejects_duplicate_timing_event() -> None:
    timing_event = TEST_LOW_CORE_DESCRIPTOR_SET.timing_events[0]

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' timing event 'test.write.fast' is duplicated"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                timing_events=(
                    *TEST_LOW_CORE_DESCRIPTOR_SET.timing_events,
                    timing_event,
                ),
            )
        )


def test_generator_rejects_unknown_event_separation_producer() -> None:
    separation = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[0],
        producer_event="test.missing",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' event separation 'test.missing' -> 'test.read.early' references an unknown producer event"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                event_separations=(
                    separation,
                    *TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[1:],
                ),
            )
        )


def test_generator_rejects_unknown_event_separation_consumer() -> None:
    separation = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[0],
        consumer_event="test.missing",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' event separation 'test.write.fast' -> 'test.missing' references an unknown consumer event"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                event_separations=(
                    separation,
                    *TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[1:],
                ),
            )
        )


def test_generator_rejects_duplicate_event_separation() -> None:
    separation = TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[0]

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' event separation 'test.write.fast' -> 'test.read.early' is duplicated"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                event_separations=(
                    *TEST_LOW_CORE_DESCRIPTOR_SET.event_separations,
                    separation,
                ),
            )
        )


def test_generator_rejects_out_of_range_event_separation() -> None:
    separation = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[0],
        minimum_issue_separation_cycles=1 << 31,
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' event separation 'test.write.fast' -> 'test.read.early' minimum issue separation does not fit i32"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                event_separations=(
                    separation,
                    *TEST_LOW_CORE_DESCRIPTOR_SET.event_separations[1:],
                ),
            )
        )


def test_generator_rejects_out_of_range_schedule_class_separation() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            minimum_issue_separation_cycles=-(1 << 31) - 1,
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' schedule class 'test.event.fast' minimum issue separation does not fit i32"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                schedule_classes=schedule_classes,
            )
        )


def test_generator_rejects_issue_use_exceeding_resource_capacity() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            issue_uses=(replace(schedule_class.issue_uses[0], units=3),),
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' schedule class 'test.event.fast' consumes 3 units of resource 'test.shared_a' with capacity 2"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                schedule_classes=schedule_classes,
            )
        )


def test_generator_rejects_aggregate_issue_use_exceeding_shared_capacity() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            issue_uses=(
                schedule_class.issue_uses[0],
                schedule_class.issue_uses[0],
                schedule_class.issue_uses[0],
            ),
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' schedule class 'test.event.fast' consumes 3 units from group:1 at relative cycle 0, exceeding capacity 2"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                schedule_classes=schedule_classes,
            )
        )


def test_generator_allows_overlapping_resource_reservations() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            issue_uses=tuple(
                replace(issue_use, kind=IssueUseKind.RESERVED)
                for issue_use in (
                    schedule_class.issue_uses[0],
                    schedule_class.issue_uses[0],
                    schedule_class.issue_uses[0],
                )
            ),
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    generate_descriptor_set(
        replace(
            TEST_LOW_CORE_DESCRIPTOR_SET,
            schedule_classes=schedule_classes,
        )
    )


def test_generator_rejects_required_uses_overlapping_a_reservation() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            issue_uses=(
                schedule_class.issue_uses[0],
                schedule_class.issue_uses[0],
                replace(
                    schedule_class.issue_uses[0],
                    kind=IssueUseKind.RESERVED,
                ),
            ),
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' schedule class 'test.event.fast' consumes 3 units from group:1 at relative cycle 0, exceeding capacity 2"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                schedule_classes=schedule_classes,
            )
        )


def test_generator_rejects_invalid_issue_use_kind() -> None:
    schedule_classes = tuple(
        replace(
            schedule_class,
            issue_uses=(
                replace(
                    schedule_class.issue_uses[0],
                    kind=cast(IssueUseKind, "invalid"),
                ),
            ),
        )
        if schedule_class.name == "test.event.fast"
        else schedule_class
        for schedule_class in TEST_LOW_CORE_DESCRIPTOR_SET.schedule_classes
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor set 'test.low.core' schedule class 'test.event.fast' has invalid issue-use kind 'invalid'"),
    ):
        generate_descriptor_set(
            replace(
                TEST_LOW_CORE_DESCRIPTOR_SET,
                schedule_classes=schedule_classes,
            )
        )


def test_generator_rejects_unknown_operand_timing_event() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            replace(
                TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
                write_event="test.missing",
            ),
            *TEST_LOW_ADD_I32_DESCRIPTOR.operands[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' write references unknown timing event 'test.missing'"),
    ):
        generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_generator_rejects_unknown_effect_timing_event() -> None:
    descriptor = next(descriptor for descriptor in TEST_LOW_CORE_DESCRIPTOR_SET.descriptors if descriptor.key == "test.event.memory.read.i32")
    descriptor = replace(
        descriptor,
        effects=(replace(descriptor.effects[0], timing_event="test.missing"),),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.event.memory.read.i32' effect 0 references unknown timing event 'test.missing'"),
    ):
        generate_descriptor_set(replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,)))


def test_generator_emits_enum_immediate_domains() -> None:
    domain = EnumDomain(
        "test.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="test.condition",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    generated = generate_descriptor_set(descriptor_set)

    assert "loom_low_enum_domain_t" in generated.source
    assert "test.condition" in generated.source
    assert "eq" in generated.source
    assert "ne" in generated.source


def test_generator_rejects_missing_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain=None,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' enum immediate 'i32_value' has no enum domain"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_unknown_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="missing",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' enum immediate 'i32_value' references unknown enum domain 'missing'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_non_enum_immediate_domain() -> None:
    enum_immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        enum_domain="test.condition",
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(enum_immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' non-enum immediate 'i32_value' references enum domain 'test.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_defaulted_immediate() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE" in generated.source
    assert ".default_value = INT64_C(7)" in generated.source


def test_generator_emits_immediate_value_step() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        value_step=64,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert ".value_step = UINT64_C(64)" in generated.source


def test_generator_rejects_zero_immediate_value_step() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        value_step=0,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' has zero value step"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_misaligned_immediate_default() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        value_step=4,
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' default value is not a multiple of 4"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_default_without_default_flag() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' has a default value without the default-value flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_default_outside_unsigned_range() -> None:
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.UNSIGNED,
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        unsigned_max=3,
        default_value=4,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' default value is out of unsigned range"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_default_outside_enum_domain() -> None:
    domain = EnumDomain(
        "test.condition",
        values=(EnumValue("ne", 1), EnumValue("eq", 0)),
    )
    immediate = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR.immediates[0],
        kind=ImmediateKind.ENUM,
        enum_domain="test.condition",
        flags=(ImmediateFlag.DEFAULT_VALUE,),
        default_value=7,
    )
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(immediate,),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        enum_domains=(domain,),
        descriptors=(descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' default value is not in enum domain 'test.condition'"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_missing_schedule_class() -> None:
    bad_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        schedule_class=cast(str, None),
    )
    bad_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' has no schedule class"),
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_result_after_operand() -> None:
    bad_descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
        ),
    )
    bad_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(bad_descriptor,),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' has result operand 'dst' after non-result operands"),
    ):
        generate_descriptor_set(bad_set)


def test_generator_rejects_operand_result_role() -> None:
    destructive_result = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
        role=OperandRole.OPERAND_RESULT,
    )
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            destructive_result,
            *TEST_LOW_ADD_I32_DESCRIPTOR.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' uses OPERAND_RESULT; use separate result and operand rows plus an explicit constraint"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_implicit_operand_without_implicit_flag() -> None:
    implicit_operand = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
        role=OperandRole.IMPLICIT,
        flags=(),
    )
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            implicit_operand,
            *TEST_LOW_ADD_I32_DESCRIPTOR.operands[2:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' implicit operand 'lhs' must set the implicit flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_accepts_tied_duplicate_operand_encoding_field() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            replace(base_descriptor.operands[1], encoding_field_id=7),
            *base_descriptor.operands[2:],
        ),
        constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "test.add.i32" in generated.source


def test_generator_emits_operand_low_subset_address_map() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=16,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert ".address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET" in generated.source
    assert ".addressable_unit_count = 16" in generated.source


def test_generator_emits_operand_target_state_address_map() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=256,
                address_state_slot=3,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert ".address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE" in generated.source
    assert ".addressable_unit_count = 256" in generated.source
    assert ".address_state_slot = 3" in generated.source


def test_generator_rejects_target_state_address_map_without_slot() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                address_map_kind=OperandAddressMapKind.TARGET_STATE,
                addressable_unit_count=256,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(ValueError, match="target-state address map must set"):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_direct_address_map_with_unit_count() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], addressable_unit_count=16),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' direct address map must not set an addressable unit count"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_bounded_address_map_without_unit_count() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' bounded address map must set an addressable unit count"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_bounded_address_map_on_implicit_operand() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            base_descriptor.operands[0],
            replace(
                base_descriptor.operands[1],
                role=OperandRole.IMPLICIT,
                flags=(OperandFlag.IMPLICIT,),
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=1,
            ),
            *base_descriptor.operands[2:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'lhs' bounded address map must apply to an SSA value operand"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_bounded_address_map_smaller_than_operand() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                unit_count=4,
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=2,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' bounded address map covers fewer units than the operand consumes"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_bounded_address_map_without_concrete_alt() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(
                base_descriptor.operands[0],
                reg_alts=(RegClassAlt(None, flags=(RegClassAltFlag.IMMEDIATE,)),),
                address_map_kind=OperandAddressMapKind.LOW_SUBSET,
                addressable_unit_count=1,
            ),
            *base_descriptor.operands[1:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' bounded address map requires a concrete register-class alternative"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_untied_duplicate_operand_encoding_field() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            replace(base_descriptor.operands[1], encoding_field_id=7),
            *base_descriptor.operands[2:],
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operands 'dst' and 'lhs' share encoding field id 7 without a tied constraint"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_operand_fixed_encoding_field_overlap() -> None:
    base_descriptor = TEST_LOW_ADD_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        operands=(
            replace(base_descriptor.operands[0], encoding_field_id=7),
            *base_descriptor.operands[1:],
        ),
        encoding_field_values=(EncodingFieldValue(7, 0),),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' operand 'dst' shares fixed encoding field id 7"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_emits_sliced_immediate_encoding_rows() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        immediates=(
            Immediate(
                "i32_value",
                ImmediateKind.SIGNED,
                bit_width=32,
                encoding_slices=(
                    ImmediateEncodingSlice(7, 0, 16),
                    ImmediateEncodingSlice(8, 16, 16),
                ),
                signed_min=-(2**31),
                unsigned_max=(2**31) - 1,
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    generated = generate_descriptor_set(descriptor_set)

    assert "loom_low_immediate_encoding_slice_t" in generated.source
    assert ".encoding_slice_start = 0," in generated.source
    assert ".encoding_slice_count = 2," in generated.source
    assert ".encoding_field_id = 7," in generated.source
    assert ".source_bit_offset = 16," in generated.source
    assert ".signed_min = (-INT64_C(2147483648))," in generated.source


def test_generator_rejects_immediate_with_direct_and_sliced_encoding() -> None:
    base_descriptor = TEST_LOW_CONST_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        immediates=(
            replace(
                base_descriptor.immediates[0],
                encoding_field_id=7,
                encoding_slices=(ImmediateEncodingSlice(8, 0, 32),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' uses both direct and sliced encoding fields"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_incomplete_sliced_immediate_encoding() -> None:
    base_descriptor = TEST_LOW_CONST_I32_DESCRIPTOR
    descriptor = replace(
        base_descriptor,
        immediates=(
            replace(
                base_descriptor.immediates[0],
                encoding_slices=(ImmediateEncodingSlice(7, 0, 16),),
            ),
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.const.i32' immediate 'i32_value' encoding slices cover 0xffff instead of 0xffffffff"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_absent_encoding_without_pseudo_flag() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' uses absent encoding id without the pseudo flag"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_pseudo_flag_with_target_encoding() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        flags=(
            *TEST_LOW_ADD_I32_DESCRIPTOR.flags,
            DescriptorFlag.PSEUDO,
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' uses the pseudo flag with a target encoding id"),
    ):
        generate_descriptor_set(descriptor_set)


def test_generator_rejects_pseudo_flag_with_target_encoding_format() -> None:
    descriptor = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        encoding_format_id=1,
        encoding_id=LOW_DESCRIPTOR_ENCODING_ID_NONE,
        flags=(
            *TEST_LOW_ADD_I32_DESCRIPTOR.flags,
            DescriptorFlag.PSEUDO,
        ),
    )
    descriptor_set = replace(TEST_LOW_CORE_DESCRIPTOR_SET, descriptors=(descriptor,))

    with pytest.raises(
        ValueError,
        match=re.escape("descriptor 'test.add.i32' uses the pseudo flag with a target encoding format id"),
    ):
        generate_descriptor_set(descriptor_set)
