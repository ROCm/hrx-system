# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from pathlib import Path
from re import Pattern
from tempfile import TemporaryDirectory
from types import TracebackType

from loom.gen.support.string_pool import CStringBank
from loom.gen.target.arch.x86 import x86_descriptors
from loom.gen.target.low import compiler
from loom.target.arch.x86 import descriptors as x86_descriptor_data
from loom.target.arch.x86 import native_vector
from loom.target.arch.x86.descriptors.encoding import Flag, Form, encoding
from loom.target.arch.x86.scalar_families import (
    X86_SCALAR_READ_MEMORY_FAMILIES,
)
from loom.target.arch.x86.target_info import (
    sorted_descriptor_set_infos,
    x86_descriptor_set_info_by_generator_target,
    x86_descriptor_set_ordinal,
)
from loom.target.arch.x86.vector_encoding import (
    VECTOR_ENCODING_FORMAT_MARKER,
    VectorEncodingPrefix,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    DescriptorAsmSurface,
    DescriptorFlag,
    DescriptorOpKind,
    EffectFlag,
    EffectKind,
    ImmediateKind,
    InstructionClass,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    RegClassFlag,
    ScheduleClassFlag,
)


class _RaisesValueError:
    def __init__(self, pattern: str):
        self.pattern: Pattern[str] = re.compile(pattern)

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        if exc_type is None or exc is None:
            raise AssertionError("expected ValueError")
        if not issubclass(exc_type, ValueError):
            return False
        if not self.pattern.search(str(exc)):
            raise AssertionError(f"ValueError {exc!r} did not match {self.pattern.pattern!r}")
        return True


def _assert_descriptor_ref(header: str, macro_name: str) -> None:
    assert re.search(rf"#define {re.escape(macro_name)} \d+u", header), macro_name


def _assert_reg_class_id(header: str, constant_name: str) -> None:
    assert re.search(rf"  {re.escape(constant_name)} = \d+u,", header), constant_name
    assert f"#define {constant_name} " not in header


def test_scalar_physical_ownership_is_shared_by_core_profiles() -> None:
    names = (
        "rax",
        "rcx",
        "rdx",
        "rbx",
        "rsp",
        "rbp",
        "rsi",
        "rdi",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
    )
    for target in ("scalar", "simd128", "avx2", "avx512", "avx512_features"):
        spec = x86_descriptors._descriptor_set_for_info(x86_descriptor_set_info_by_generator_target(target))
        # Native profiles are views over the composite storage inventory.
        storage = x86_descriptors._shared_storage_descriptor_set(x86_descriptor_data.X86_AVX512_FEATURES_DESCRIPTOR_SET, (spec,))
        compiled = compiler.compile_descriptor_set(storage)
        assert tuple(register.name for register in compiled.physical_registers) == names
        assert tuple(register.atomic_units for register in compiled.physical_registers) == tuple((i,) for i in range(16))
        for name, width, physical_ids in (
            ("x86.gpr32", 32, tuple(range(16))),
            ("x86.gpr64", 64, tuple(range(16))),
            ("x86.rax", 64, (0,)),
            ("x86.rdx", 64, (2,)),
        ):
            class_id = compiled.reg_class_ids[name]
            reg_class = compiled.reg_classes[class_id]
            assert reg_class.alloc_unit_bits == width
            assert RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS in reg_class.flags
            start = compiled.physical_register_candidate_starts[class_id]
            assert tuple(compiled.physical_register_candidate_ids[start : start + len(physical_ids)]) == physical_ids
            if len(physical_ids) == 1:
                assert RegClassFlag.UNSPILLABLE in reg_class.flags
        multiply = next(descriptor for descriptor in spec.descriptors if descriptor.key == "x86.scalar.mul.high.gpr64")
        operands = {operand.field_name: operand for operand in multiply.operands}
        assert OperandFlag.IMPLICIT in operands["lhs"].flags
        assert OperandFlag.IMPLICIT in operands["dst"].flags
        assert operands["low"].role == OperandRole.IMPLICIT
        assert OperandFlag.STATE_WRITE in operands["low"].flags


def test_scalar_speculation_facts_cover_total_integer_forms() -> None:
    total_forms = {
        Form.BINARY_RM_R,
        Form.BINARY_R_RM,
        Form.MULTIPLY_IMMEDIATE,
        Form.BINARY_IMMEDIATE,
        Form.SHIFT_IMMEDIATE,
        Form.SHIFT_COUNT,
        Form.UNARY_R_RM,
        Form.SELECT,
        Form.SUBTRACT_IF_UGE,
        Form.COMPARE,
        Form.COMPARE_IMMEDIATE,
        Form.COMPARE_PREDICATE,
        Form.COMPARE_IMMEDIATE_PREDICATE,
        Form.COMPARE_SELECT,
        Form.COMPARE_IMMEDIATE_SELECT,
        Form.CONSTANT,
        Form.ADDRESS_ADD,
        Form.ADDRESS_DISPLACEMENT,
        Form.ADDRESS_SCALE,
        Form.ADDRESS_ADD_SCALE,
        Form.ADDRESS_PC_RELATIVE,
        Form.UNARY_RM,
    }
    trapping_or_effecting_forms = {
        Form.MULTIPLY_HIGH,
        Form.LOAD,
        Form.STORE,
        Form.BINARY_R_MEMORY,
        Form.ZERO_REGISTER,
        Form.SIGN_EXTEND_DIVIDEND,
        Form.DIVIDE,
        Form.COMPARE_BRANCH,
        Form.COMPARE_IMMEDIATE_BRANCH,
    }
    descriptors = x86_descriptor_data.X86_SCALAR_DESCRIPTOR_SET.descriptors
    observed_forms = {Form(descriptor.encoding_format_id) for descriptor in descriptors}
    assert total_forms.isdisjoint(trapping_or_effecting_forms)
    assert observed_forms == total_forms | trapping_or_effecting_forms

    for descriptor in descriptors:
        form = Form(descriptor.encoding_format_id)
        writes_state = any(OperandFlag.STATE_WRITE in operand.flags for operand in descriptor.operands)
        expected_safe = form in total_forms and not writes_state
        assert (DescriptorFlag.SAFE_TO_SPECULATE in descriptor.flags) is expected_safe, descriptor.key
        if expected_safe:
            assert DescriptorFlag.DEAD_REMOVABLE in descriptor.flags
            assert not descriptor.effects
            assert not descriptor.storage_leases


def test_native_facts_cover_core_profiles() -> None:
    for descriptor_set in (
        x86_descriptor_data.X86_AVX2_DESCRIPTOR_SET,
        x86_descriptor_data.X86_AVX512_CORE_DESCRIPTOR_SET,
    ):
        assert all(descriptor.encoding_format_id for descriptor in descriptor_set.descriptors)

    core_keys = {descriptor.key for descriptor in x86_descriptor_data.X86_AVX512_CORE_DESCRIPTOR_SET.descriptors}
    missing_composite_facts = {descriptor.key for descriptor in x86_descriptor_data.X86_AVX512_FEATURES_DESCRIPTOR_SET.descriptors if not descriptor.encoding_format_id}
    assert missing_composite_facts.isdisjoint(core_keys)


def test_native_vector_binding_rejects_cross_wired_facts() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in x86_descriptor_data.X86_AVX2_DESCRIPTOR_SET.descriptors}
    to_gpr = descriptors["x86.avx2.vmovd.gpr32.xmm"]
    to_xmm = descriptors["x86.avx2.vmovd.xmm.gpr32"]
    with _RaisesValueError(r"native operand shape mismatch"):
        native_vector.VMOVD_TO_GPR32.bind(to_xmm, VectorEncodingPrefix.VEX)
    assert to_gpr.encoding_format_id & VECTOR_ENCODING_FORMAT_MARKER


def test_explicit_writemasks_use_seven_location_window() -> None:
    descriptors = x86_descriptor_data.X86_AVX512_CORE_DESCRIPTOR_SET.descriptors
    k_operands = tuple(operand for descriptor in descriptors for operand in descriptor.operands if any(alternative.reg_class == "x86.k" for alternative in operand.reg_alts))
    assert k_operands

    explicit_writemasks = tuple(operand for operand in k_operands if operand.field_name == "mask")
    assert explicit_writemasks
    assert all(operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET and operand.addressable_unit_count == 7 for operand in explicit_writemasks)

    ordinary_k_operands = tuple(operand for operand in k_operands if operand.field_name != "mask")
    assert ordinary_k_operands
    assert all(operand.address_map_kind is OperandAddressMapKind.DIRECT and operand.addressable_unit_count == 0 for operand in ordinary_k_operands)


def test_native_vector_prefix_owns_simd_register_addressability() -> None:
    for info in sorted_descriptor_set_infos():
        spec = x86_descriptors._descriptor_set_for_info(info)
        for descriptor in spec.descriptors:
            if not descriptor.encoding_format_id & VECTOR_ENCODING_FORMAT_MARKER:
                continue
            # The extension bit denotes EVEX for maps 1/2/3. Maps 5/6 use
            # encoded map zero and repurpose that bit to distinguish the maps.
            evex = bool(descriptor.encoding_id & 0x2000) or not (descriptor.encoding_id & 0x300)
            for operand in descriptor.operands:
                if not any(alternative.reg_class in ("x86.xmm", "x86.ymm", "x86.zmm") for alternative in operand.reg_alts):
                    continue
                context = (spec.key, descriptor.key, operand.field_name)
                if evex:
                    assert operand.address_map_kind is OperandAddressMapKind.DIRECT, context
                    assert operand.addressable_unit_count == 0, context
                else:
                    assert operand.address_map_kind is OperandAddressMapKind.LOW_SUBSET, context
                    assert operand.addressable_unit_count == 16, context


def test_narrow_memory_widths_are_independent_of_scalar_carriers() -> None:
    for info in sorted_descriptor_set_infos():
        if "x86.gpr32" not in info.register_classes:
            continue
        spec = x86_descriptors._descriptor_set_for_info(info)
        descriptors = {descriptor.key: descriptor for descriptor in spec.descriptors}
        for width in (8, 16):
            for operation, mnemonic, role in (
                ("load", "movzx", OperandRole.RESULT),
                ("store", "mov", OperandRole.OPERAND),
            ):
                for addressing in ("", ".indexed"):
                    descriptor = descriptors[f"x86.scalar.{mnemonic}.{operation}{addressing}.u{width}.gpr32"]
                    value = descriptor.operands[0]
                    assert value.role is role
                    assert tuple(alt.reg_class for alt in value.reg_alts) == ("x86.gpr32",)
                    assert len(descriptor.effects) == 1
                    assert descriptor.effects[0].width_bits == width
                    assert descriptor.mnemonic == mnemonic


def test_bitwise_immediate_forms_are_shared_by_scalar_profiles() -> None:
    for info in sorted_descriptor_set_infos():
        # Packed-dot-only profiles do not expose scalar register classes.
        if "x86.gpr32" not in info.register_classes:
            continue
        spec = x86_descriptors._descriptor_set_for_info(info)
        descriptors = {descriptor.key: descriptor for descriptor in spec.descriptors}
        for width in (32, 64):
            for operation in ("and", "or", "xor"):
                descriptor = descriptors[f"x86.scalar.{operation}.imm.gpr{width}"]
                assert descriptor.constraints == (
                    Constraint(ConstraintKind.TIED, 0, 1),
                    Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
                )
                assert len(descriptor.operands) == 2
                assert descriptor.operands[0].reg_alts == descriptor.operands[1].reg_alts
                assert len(descriptor.immediates) == 1
                immediate = descriptor.immediates[0]
                assert immediate.kind == ImmediateKind.SIGNED
                assert immediate.bit_width == 32
                assert immediate.signed_min == -(2**31)
                assert immediate.unsigned_max == 2**31 - 1


def test_integer_comparisons_use_width_parametric_predicates() -> None:
    compare_branch_schedule = next(schedule for schedule in x86_descriptor_data.X86_SCALAR_DESCRIPTOR_SET.schedule_classes if schedule.name == "x86.scalar.control")
    assert tuple(use.resource for use in compare_branch_schedule.issue_uses) == (
        "x86.scalar",
        "x86.control",
    )
    assert compare_branch_schedule.flags == (ScheduleClassFlag.CONTROL,)

    predicate_descriptors = {
        descriptor.key: descriptor
        for descriptor in x86_descriptor_data.X86_SCALAR_DESCRIPTOR_SET.descriptors
        if descriptor.key.startswith("x86.scalar.cmp.") and ".select." not in descriptor.key and ".branch." not in descriptor.key
    }
    branch_descriptors = {descriptor.key: descriptor for descriptor in x86_descriptor_data.X86_SCALAR_DESCRIPTOR_SET.descriptors if descriptor.key.startswith("x86.scalar.cmp.branch.")}
    dynamic_keys = {f"x86.scalar.cmp.{form_name}gpr{bit_count}" for bit_count in (8, 16, 32, 64) for form_name in ("", "imm.")}
    branch_keys = {f"x86.scalar.cmp.branch.{form_name}gpr{bit_count}" for bit_count in (8, 16, 32, 64) for form_name in ("", "imm.")}
    fixed_recipe_keys = {f"x86.scalar.cmp.{predicate}.gpr32" for predicate in ("slt", "sge", "sgt")} | {
        f"x86.scalar.cmp.{predicate}.imm.gpr{bit_count}"
        for bit_count, predicate in (
            (32, "eq"),
            (32, "slt"),
            (32, "sge"),
            (32, "sgt"),
            (32, "uge"),
            (64, "eq"),
        )
    }
    assert set(predicate_descriptors) == dynamic_keys | fixed_recipe_keys
    assert set(branch_descriptors) == branch_keys

    width_flags = {
        8: Flag.BYTE,
        16: Flag.OPERAND_16,
        32: Flag(0),
        64: Flag.REX_W,
    }
    for bit_count, flags in width_flags.items():
        for immediate in (False, True):
            form_name = "imm." if immediate else ""
            descriptor = predicate_descriptors[f"x86.scalar.cmp.{form_name}gpr{bit_count}"]
            assert descriptor.encoding_format_id == (Form.COMPARE_IMMEDIATE_PREDICATE if immediate else Form.COMPARE_PREDICATE)
            opcode = (0x80 if bit_count == 8 else 0x81) if immediate else (0x38 if bit_count == 8 else 0x39)
            assert descriptor.encoding_id == encoding(
                opcode,
                extension=7 if immediate else 0,
                flags=flags,
            )
            immediate_fields = (f"imm{min(bit_count, 32)}", "predicate") if immediate else ("predicate",)
            assert tuple(field.field_name for field in descriptor.immediates) == immediate_fields

            branch = branch_descriptors[f"x86.scalar.cmp.branch.{form_name}gpr{bit_count}"]
            assert branch.encoding_format_id == (Form.COMPARE_IMMEDIATE_BRANCH if immediate else Form.COMPARE_BRANCH)
            assert branch.encoding_id == descriptor.encoding_id
            assert tuple(field.field_name for field in branch.immediates) == immediate_fields
            assert branch.op_kind is DescriptorOpKind.COND_BR
            assert branch.asm_surface is DescriptorAsmSurface.STRUCTURAL
            assert branch.effects[0].kind is EffectKind.CONTROL
            assert branch.effects[0].flags == (EffectFlag.ORDERED,)
            assert DescriptorFlag.SIDE_EFFECTING in branch.flags
            assert DescriptorFlag.TERMINATOR in branch.flags
            assert branch.schedule_class == compare_branch_schedule.name
            assert branch.instruction_classes == (InstructionClass.BRANCH,)


def test_scalar_read_memory_binary_forms_are_shared_by_scalar_profiles() -> None:
    assert {(family.descriptor_operation, family.opcode) for family in X86_SCALAR_READ_MEMORY_FAMILIES} == {
        ("add", 0x03),
        ("sub", 0x2B),
        ("imul", 0x0FAF),
        ("and", 0x23),
        ("or", 0x0B),
        ("xor", 0x33),
    }
    for info in sorted_descriptor_set_infos():
        if "x86.gpr32" not in info.register_classes:
            continue
        spec = x86_descriptors._descriptor_set_for_info(info)
        descriptors = {descriptor.key: descriptor for descriptor in spec.descriptors}
        schedules = {schedule.name: schedule for schedule in spec.schedule_classes}
        for width in (32, 64):
            register_class = f"x86.gpr{width}"
            schedule_name = f"x86.memory.alu.gpr{width}"
            schedule = schedules[schedule_name]
            assert schedule.latency_cycles == 4
            assert schedule.minimum_issue_separation_cycles == 4
            assert tuple(use.resource for use in schedule.issue_uses) == (
                "x86.load",
                "x86.scalar",
            )
            assert schedule.flags == (ScheduleClassFlag.MAY_LOAD,)
            for family in X86_SCALAR_READ_MEMORY_FAMILIES:
                for address_suffix, immediate_fields in (
                    ("", ("disp32",)),
                    (".indexed", ("disp32", "scale")),
                ):
                    descriptor = descriptors[f"x86.scalar.{family.descriptor_operation}.load{address_suffix}.gpr{width}"]
                    expected_encoding = encoding(
                        family.opcode,
                        flags=(Flag.REX_W if width == 64 else 0) | (Flag.INDEXED if address_suffix else 0),
                    )
                    assert descriptor.encoding_format_id == Form.BINARY_R_MEMORY
                    assert descriptor.encoding_id == expected_encoding
                    assert descriptor.constraints == (
                        Constraint(ConstraintKind.TIED, 0, 1),
                        Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
                    )
                    assert tuple(operand.role for operand in descriptor.operands) == (
                        OperandRole.RESULT,
                        OperandRole.OPERAND,
                        OperandRole.RESOURCE,
                        *((OperandRole.RESOURCE,) if address_suffix else ()),
                    )
                    assert {alternative.reg_class for operand in descriptor.operands[:2] for alternative in operand.reg_alts} == {register_class}
                    assert all({alternative.reg_class for alternative in operand.reg_alts} == {"x86.gpr64"} for operand in descriptor.operands[2:])
                    assert tuple(immediate.field_name for immediate in descriptor.immediates) == immediate_fields
                    assert descriptor.flags == (DescriptorFlag.SIDE_EFFECTING,)
                    assert len(descriptor.effects) == 1
                    effect = descriptor.effects[0]
                    assert effect.kind == EffectKind.READ
                    assert effect.flags == (EffectFlag.DEPENDENCY,)
                    assert effect.width_bits == width
                    assert descriptor.schedule_class == schedule_name


def test_storage_generation_emits_current_public_views() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        bank = CStringBank.build(frozenset({"x86.avx2.vaddps.xmm"}), symbol="test_instruction_strings", header="test/instruction_strings.h")
        manifest = tmp_path / "instruction_strings.json"
        manifest.write_text(bank.manifest())
        assert (
            x86_descriptors.main(
                [
                    "--target=avx512_features",
                    f"--shared-strings={manifest}",
                    f"--header={tmp_path / 'avx512_features_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_features_descriptors.c'}",
                    f"--view-header=avx512={tmp_path / 'avx512_descriptors.h'}",
                    f"--view-header=avx2={tmp_path / 'avx2_descriptors.h'}",
                    f"--view-header=avx2_features={tmp_path / 'avx2_features_descriptors.h'}",
                    f"--view-header=avx10_2={tmp_path / 'avx10_2_descriptors.h'}",
                    f"--view-header=avx512_bf16={tmp_path / 'avx512_bf16_descriptors.h'}",
                    f"--view-header=avx512_fp16={tmp_path / 'avx512_fp16_descriptors.h'}",
                    f"--view-header=avx512_vnni={tmp_path / 'avx512_vnni_descriptors.h'}",
                    f"--view-header=avx_vnni={tmp_path / 'avx_vnni_descriptors.h'}",
                    f"--view-header=avx_vnni_int8={tmp_path / 'avx_vnni_int8_descriptors.h'}",
                    f"--view-header=avx_vnni_int16={tmp_path / 'avx_vnni_int16_descriptors.h'}",
                    f"--view-header=packed_dot={tmp_path / 'packed_dot_descriptors.h'}",
                    f"--view-header=scalar={tmp_path / 'scalar_descriptors.h'}",
                    f"--view-header=simd128={tmp_path / 'simd128_descriptors.h'}",
                ]
            )
            == 0
        )

        source = (tmp_path / "avx512_features_descriptors.c").read_text(encoding="utf-8")
        composite_header = (tmp_path / "avx512_features_descriptors.h").read_text(encoding="utf-8")
        avx512_header = (tmp_path / "avx512_descriptors.h").read_text(encoding="utf-8")
        avx512_bf16_header = (tmp_path / "avx512_bf16_descriptors.h").read_text(encoding="utf-8")
        avx2_header = (tmp_path / "avx2_descriptors.h").read_text(encoding="utf-8")
        avx2_features_header = (tmp_path / "avx2_features_descriptors.h").read_text(encoding="utf-8")
        avx512_fp16_header = (tmp_path / "avx512_fp16_descriptors.h").read_text(encoding="utf-8")
        avx_vnni_header = (tmp_path / "avx_vnni_descriptors.h").read_text(encoding="utf-8")
        packed_dot_header = (tmp_path / "packed_dot_descriptors.h").read_text(encoding="utf-8")
        scalar_header = (tmp_path / "scalar_descriptors.h").read_text(encoding="utf-8")
        simd128_header = (tmp_path / "simd128_descriptors.h").read_text(encoding="utf-8")

    assert '#include "test/instruction_strings.h"' in source
    assert "test_instruction_strings}" in source
    assert "loom_x86_avx512_core_descriptor_set" in source
    assert "loom_x86_avx2_core_descriptor_set" in source
    assert "loom_x86_avx2_features_core_descriptor_set" in source
    assert "loom_x86_avx512_fp16_core_descriptor_set" in source
    assert "loom_x86_packed_dot_core_descriptor_set" in source
    assert "loom_x86_avx_vnni_core_descriptor_set" in source
    assert "loom_x86_avx512_features_core_descriptor_set" in source
    assert "loom_x86_scalar_core_descriptor_set" in source
    assert "loom_x86_simd128_core_descriptor_set" in source
    assert "static const loom_low_operand_t kX86Avx512FeaturesCoreStorageOperands[]" in source
    assert "static const loom_low_operand_t kX86Avx512CoreOperands[]" not in source
    assert "static const loom_low_operand_t kX86PackedDotCoreOperands[]" not in source
    assert "static const loom_low_descriptor_t kX86Avx512FeaturesCoreStorageDescriptors[]" in source
    for view_prefix in (
        "X86Avx512Core",
        "X86Avx512Bf16Core",
        "X86Avx2Core",
        "X86Avx2FeaturesCore",
        "X86PackedDotCore",
        "X86ScalarCore",
        "X86Simd128Core",
    ):
        assert f"static const loom_low_descriptor_t k{view_prefix}Descriptors[]" not in source
        assert f"static const loom_low_descriptor_view_t k{view_prefix}DescriptorViews[]" not in source
        assert f"static const uint32_t k{view_prefix}DescriptorRefs[]" not in source
        assert f"static const loom_low_asm_form_t k{view_prefix}AsmForms[]" not in source
    assert "static const loom_low_asm_form_t kX86AvxVnniCoreAsmForms[]" in source
    assert source.count(".descriptors = kX86Avx512FeaturesCoreStorageDescriptors,") == 9
    assert source.count(".descriptor_views = kX86Avx512FeaturesCoreStorageDescriptorViews,") == 9
    assert ".asm_forms = kX86AvxVnniCoreAsmForms," in source
    assert source.count(".asm_forms = kX86Avx512FeaturesCoreStorageAsmForms,") == 9
    assert source.count(".descriptor_refs = kX86Avx512FeaturesCoreStorageDescriptorRefs,") == 9
    assert "kX86Avx2FeaturesCoreDescriptorMembershipWords" in source
    assert "kX86PackedDotCoreDescriptorMembershipWords" in source
    assert source.count(".descriptor_membership_words = kX86ScalarCoreDescriptorMembershipWords,") == 2
    # A shared spelling may span adjacent literals in the compact byte pool.
    string_data = source.replace('"\n    "', "")
    assert "avx_vnni.vpdpbusd.ymm" in string_data
    assert "vpdpbusd.ymm" in string_data
    assert "loom_x86_avx512_core_descriptor_set" in avx512_header
    assert "loom_x86_avx512_bf16_core_descriptor_set" in avx512_bf16_header
    assert "loom_x86_avx2_core_descriptor_set" in avx2_header
    assert "loom_x86_avx2_features_core_descriptor_set" in avx2_features_header
    assert "loom_x86_avx512_fp16_core_descriptor_set" in avx512_fp16_header
    assert "loom_x86_avx_vnni_core_descriptor_set" in avx_vnni_header
    assert "loom_x86_packed_dot_core_descriptor_set" in packed_dot_header
    assert "loom_x86_scalar_core_descriptor_set" in scalar_header
    assert "loom_x86_simd128_core_descriptor_set" in simd128_header
    assert f"#define X86_AVX512_FEATURES_CORE_DESCRIPTOR_SET_ORDINAL UINT16_C({x86_descriptor_set_ordinal('x86.avx512_features.core')})" in composite_header
    assert f"#define X86_PACKED_DOT_CORE_DESCRIPTOR_SET_ORDINAL UINT16_C({x86_descriptor_set_ordinal('x86.packed_dot.core')})" in packed_dot_header
    _assert_descriptor_ref(avx512_header, "X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VADDPS_XMM")
    _assert_descriptor_ref(avx2_header, "X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VADDPS_XMM")
    _assert_descriptor_ref(
        avx2_features_header,
        "X86_AVX2_FEATURES_CORE_DESCRIPTOR_REF_AVX_VNNI_INT8_VPDPBSSD_YMM",
    )
    _assert_descriptor_ref(
        avx_vnni_header,
        "X86_AVX_VNNI_CORE_DESCRIPTOR_REF_AVX_VNNI_VPDPBUSD_YMM",
    )
    _assert_descriptor_ref(scalar_header, "X86_SCALAR_CORE_DESCRIPTOR_REF_SCALAR_LEA_ADD_GPR64")
    _assert_descriptor_ref(
        packed_dot_header,
        "X86_PACKED_DOT_CORE_DESCRIPTOR_REF_AVX512_BF16_VDPBF16PS_YMM",
    )
    _assert_descriptor_ref(
        composite_header,
        "X86_AVX512_FEATURES_CORE_DESCRIPTOR_REF_AVX512_BF16_VDPBF16PS_ZMM",
    )
    _assert_descriptor_ref(
        avx512_fp16_header,
        "X86_AVX512_FP16_CORE_DESCRIPTOR_REF_AVX512_FP16_VADDPH_ZMM",
    )
    _assert_reg_class_id(scalar_header, "X86_SCALAR_CORE_REG_CLASS_ID_GPR32")
    _assert_reg_class_id(scalar_header, "X86_SCALAR_CORE_REG_CLASS_ID_GPR64")
    _assert_reg_class_id(simd128_header, "X86_SIMD128_CORE_REG_CLASS_ID_XMM")
    _assert_reg_class_id(avx2_header, "X86_AVX2_CORE_REG_CLASS_ID_YMM")
    _assert_reg_class_id(avx2_features_header, "X86_AVX2_FEATURES_CORE_REG_CLASS_ID_YMM")
    assert "X86_AVX2_FEATURES_CORE_REG_CLASS_ID_ZMM" not in avx2_features_header
    _assert_reg_class_id(avx_vnni_header, "X86_AVX_VNNI_CORE_REG_CLASS_ID_YMM")
    assert "X86_AVX_VNNI_CORE_REG_CLASS_ID_ZMM" not in avx_vnni_header
    assert "X86_AVX_VNNI_CORE_DESCRIPTOR_REF_AVX512_VADDPS_ZMM" not in avx_vnni_header
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_XMM")
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_YMM")
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_ZMM")


def test_view_target_generation_rejects_direct_source_output() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        with _RaisesValueError(
            r"x86 descriptor target avx512 is a view of storage target "
            r"avx512_features"
        ):
            x86_descriptors.main(
                [
                    "--target=avx512",
                    f"--header={tmp_path / 'avx512_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_descriptors.c'}",
                ]
            )


def test_composite_descriptor_merge_rejects_duplicate_keys() -> None:
    descriptor_set = x86_descriptor_data.X86_PACKED_DOT_DESCRIPTOR_SET
    with _RaisesValueError(r"repeats descriptor 'x86.avx512_vnni.vpdpbusd.xmm'"):
        x86_descriptor_data._merge_component_descriptors(
            (
                (descriptor_set, frozenset()),
                (descriptor_set, frozenset()),
            )
        )


def test_unknown_view_header_is_rejected() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        with _RaisesValueError(
            r"x86 descriptor target avx512_features cannot emit view "
            r"headers for: missing"
        ):
            x86_descriptors.main(
                [
                    "--target=avx512_features",
                    f"--header={tmp_path / 'avx512_features_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_features_descriptors.c'}",
                    f"--view-header=missing={tmp_path / 'missing_descriptors.h'}",
                ]
            )
