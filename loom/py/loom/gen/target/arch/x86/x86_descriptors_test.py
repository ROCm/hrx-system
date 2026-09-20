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

from loom.gen.target.arch.x86 import x86_descriptors
from loom.gen.target.low import compiler
from loom.target.arch.x86 import descriptors as x86_descriptor_data
from loom.target.arch.x86.encoding import (
    X86_ENCODING_ID_FORCE_32_BIT,
    X86EncodingFormat,
    x86_legacy_opcode,
    x86_modrm_group_opcode,
)
from loom.target.arch.x86.target_info import (
    sorted_descriptor_set_infos,
    x86_descriptor_set_info_by_generator_target,
    x86_descriptor_set_ordinal,
)
from loom.target.low_descriptors import Constraint, ConstraintKind, ImmediateKind, OperandFlag, OperandRole, RegClassFlag


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
    for target in ("scalar", "simd128", "avx2", "avx512", "avx512_packed_dot"):
        spec = x86_descriptors._descriptor_set_for_info(x86_descriptor_set_info_by_generator_target(target))
        # Native profiles are views over the composite storage inventory.
        storage = x86_descriptors._shared_storage_descriptor_set(x86_descriptor_data.X86_AVX512_PACKED_DOT_DESCRIPTOR_SET, (spec,))
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


def test_storage_generation_emits_current_public_views() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        assert (
            x86_descriptors.main(
                [
                    "--target=avx512_packed_dot",
                    f"--header={tmp_path / 'avx512_packed_dot_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_packed_dot_descriptors.c'}",
                    f"--encoding-header={tmp_path / 'encoding_defs.h'}",
                    f"--view-header=avx512={tmp_path / 'avx512_descriptors.h'}",
                    f"--view-header=avx2={tmp_path / 'avx2_descriptors.h'}",
                    f"--view-header=avx10_2={tmp_path / 'avx10_2_descriptors.h'}",
                    f"--view-header=avx512_bf16={tmp_path / 'avx512_bf16_descriptors.h'}",
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

        source = (tmp_path / "avx512_packed_dot_descriptors.c").read_text(encoding="utf-8")
        encoding_header = (tmp_path / "encoding_defs.h").read_text(encoding="utf-8")
        composite_header = (tmp_path / "avx512_packed_dot_descriptors.h").read_text(encoding="utf-8")
        avx512_header = (tmp_path / "avx512_descriptors.h").read_text(encoding="utf-8")
        avx2_header = (tmp_path / "avx2_descriptors.h").read_text(encoding="utf-8")
        avx_vnni_header = (tmp_path / "avx_vnni_descriptors.h").read_text(encoding="utf-8")
        packed_dot_header = (tmp_path / "packed_dot_descriptors.h").read_text(encoding="utf-8")
        scalar_header = (tmp_path / "scalar_descriptors.h").read_text(encoding="utf-8")
        simd128_header = (tmp_path / "simd128_descriptors.h").read_text(encoding="utf-8")

    assert "loom_x86_avx512_core_descriptor_set" in source
    assert "LOOM_X86_ENCODING_FORMAT_COMPARE = 6u" in encoding_header
    assert "#define LOOM_X86_ENCODING_ID_FORCE_32_BIT 32768u" in encoding_header
    assert "loom_x86_avx2_core_descriptor_set" in source
    assert "loom_x86_packed_dot_core_descriptor_set" in source
    assert "loom_x86_avx_vnni_core_descriptor_set" in source
    assert "loom_x86_avx512_packed_dot_core_descriptor_set" in source
    assert "loom_x86_scalar_core_descriptor_set" in source
    assert "loom_x86_simd128_core_descriptor_set" in source
    assert "static const loom_low_operand_t kX86Avx512PackedDotCoreStorageOperands[]" in source
    assert "static const loom_low_operand_t kX86Avx512CoreOperands[]" not in source
    assert "static const loom_low_operand_t kX86PackedDotCoreOperands[]" not in source
    assert "static const loom_low_descriptor_t kX86Avx512PackedDotCoreStorageDescriptors[]" in source
    assert "static const loom_low_descriptor_t kX86Avx512CoreDescriptors[]" in source
    assert "static const loom_low_descriptor_t kX86PackedDotCoreDescriptors[]" in source
    assert "static const loom_low_asm_form_t kX86AvxVnniCoreAsmForms[]" in source
    assert ".descriptors = kX86Avx512CoreDescriptors," in source
    assert ".descriptors = kX86PackedDotCoreDescriptors," in source
    assert ".descriptors = kX86Avx512PackedDotCoreStorageDescriptors," in source
    assert ".descriptor_views = kX86Avx512CoreDescriptorViews," in source
    assert ".descriptor_views = kX86PackedDotCoreDescriptorViews," in source
    assert ".descriptor_views = kX86Avx512PackedDotCoreStorageDescriptorViews," in source
    assert ".asm_forms = kX86AvxVnniCoreAsmForms," in source
    assert ".asm_forms = kX86Avx512PackedDotCoreStorageAsmForms," in source
    assert ".descriptor_refs = kX86Avx512CoreDescriptorRefs," in source
    assert ".descriptor_refs = kX86PackedDotCoreDescriptorRefs," in source
    assert ".descriptor_refs = kX86ScalarCoreDescriptorRefs," in source
    assert "static const loom_low_descriptor_t kX86Simd128CoreDescriptors[]" not in source
    assert ("static const loom_low_descriptor_view_t kX86Simd128CoreDescriptorViews[]") not in source
    assert "static const loom_low_descriptor_ref_t kX86Simd128CoreDescriptorRefs[]" not in source
    assert "static const loom_low_asm_form_t kX86Simd128CoreAsmForms[]" not in source
    assert source.count(".descriptors = kX86ScalarCoreDescriptors,") == 2
    assert source.count(".descriptor_views = kX86ScalarCoreDescriptorViews,") == 2
    assert source.count(".descriptor_refs = kX86ScalarCoreDescriptorRefs,") == 2
    assert source.count(".asm_forms = kX86ScalarCoreAsmForms,") == 2
    assert '"avx_vnni.vpdpbusd.ymm"' in source
    assert '"vpdpbusd.ymm"' in source
    assert "loom_x86_avx512_core_descriptor_set" in avx512_header
    assert "loom_x86_avx2_core_descriptor_set" in avx2_header
    assert "loom_x86_avx_vnni_core_descriptor_set" in avx_vnni_header
    assert "loom_x86_packed_dot_core_descriptor_set" in packed_dot_header
    assert "loom_x86_scalar_core_descriptor_set" in scalar_header
    assert "loom_x86_simd128_core_descriptor_set" in simd128_header
    assert f"#define X86_AVX512_PACKED_DOT_CORE_DESCRIPTOR_SET_ORDINAL UINT16_C({x86_descriptor_set_ordinal('x86.avx512_packed_dot.core')})" in composite_header
    assert f"#define X86_PACKED_DOT_CORE_DESCRIPTOR_SET_ORDINAL UINT16_C({x86_descriptor_set_ordinal('x86.packed_dot.core')})" in packed_dot_header
    _assert_descriptor_ref(avx512_header, "X86_AVX512_CORE_DESCRIPTOR_REF_AVX2_VADDPS_XMM")
    _assert_descriptor_ref(avx2_header, "X86_AVX2_CORE_DESCRIPTOR_REF_AVX2_VADDPS_XMM")
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
        "X86_AVX512_PACKED_DOT_CORE_DESCRIPTOR_REF_AVX512_BF16_VDPBF16PS_ZMM",
    )
    _assert_reg_class_id(scalar_header, "X86_SCALAR_CORE_REG_CLASS_ID_GPR32")
    _assert_reg_class_id(scalar_header, "X86_SCALAR_CORE_REG_CLASS_ID_GPR64")
    _assert_reg_class_id(simd128_header, "X86_SIMD128_CORE_REG_CLASS_ID_XMM")
    _assert_reg_class_id(avx2_header, "X86_AVX2_CORE_REG_CLASS_ID_YMM")
    _assert_reg_class_id(avx_vnni_header, "X86_AVX_VNNI_CORE_REG_CLASS_ID_YMM")
    assert "X86_AVX_VNNI_CORE_REG_CLASS_ID_ZMM" not in avx_vnni_header
    assert "X86_AVX_VNNI_CORE_DESCRIPTOR_REF_AVX512_VADDPS_ZMM" not in avx_vnni_header
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_XMM")
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_YMM")
    _assert_reg_class_id(packed_dot_header, "X86_PACKED_DOT_CORE_REG_CLASS_ID_ZMM")


def test_scalar_descriptors_define_complete_native_encoding_recipes() -> None:
    descriptors = {descriptor.key: descriptor for descriptor in x86_descriptor_data.X86_SCALAR_DESCRIPTOR_SET.descriptors}
    assert len(descriptors) == 81
    assert all(descriptor.encoding_format_id != X86EncodingFormat.NONE for descriptor in descriptors.values())

    for bit_count in (32, 64):
        for mnemonic, opcode in (
            ("add", 0x01),
            ("sub", 0x29),
            ("and", 0x21),
            ("or", 0x09),
            ("xor", 0x31),
        ):
            descriptor = descriptors[f"x86.scalar.{mnemonic}.gpr{bit_count}"]
            assert descriptor.encoding_format_id == X86EncodingFormat.RM_REG
            assert descriptor.encoding_id == x86_legacy_opcode(opcode)
        imul = descriptors[f"x86.scalar.imul.gpr{bit_count}"]
        assert imul.encoding_format_id == X86EncodingFormat.REG_RM
        assert imul.encoding_id == x86_legacy_opcode(0x0F, 0xAF)
        for mnemonic, extension in (("shl", 4), ("sar", 7), ("shr", 5)):
            shift = descriptors[f"x86.scalar.{mnemonic}.imm.gpr{bit_count}"]
            assert shift.encoding_format_id == X86EncodingFormat.RM_IMM8
            assert shift.encoding_id == x86_modrm_group_opcode(0xC1, extension)
        for mnemonic, extension in (("and", 4), ("or", 1), ("xor", 6)):
            bitwise = descriptors[f"x86.scalar.{mnemonic}.imm.gpr{bit_count}"]
            assert bitwise.encoding_format_id == X86EncodingFormat.RM_IMM32
            assert bitwise.encoding_id == x86_modrm_group_opcode(0x81, extension)
        assert descriptors[f"x86.scalar.select.gpr{bit_count}"].encoding_format_id == X86EncodingFormat.SELECT

    condition_codes = {
        "eq": 0x4,
        "ne": 0x5,
        "slt": 0xC,
        "sle": 0xE,
        "sgt": 0xF,
        "sge": 0xD,
        "ult": 0x2,
        "ule": 0x6,
        "ugt": 0x7,
        "uge": 0x3,
    }
    for bit_count in (32, 64):
        for predicate, condition_code in condition_codes.items():
            descriptor = descriptors[f"x86.scalar.cmp.{predicate}.gpr{bit_count}"]
            assert descriptor.encoding_format_id == X86EncodingFormat.COMPARE
            assert descriptor.encoding_id == condition_code
            if bit_count == 32:
                immediate = descriptors[f"x86.scalar.cmp.{predicate}.imm.gpr32"]
                assert immediate.encoding_format_id == X86EncodingFormat.COMPARE
                assert immediate.encoding_id == condition_code

    high_product = descriptors["x86.scalar.mul.high.gpr64"]
    assert high_product.encoding_format_id == X86EncodingFormat.RM_GROUP
    assert high_product.encoding_id == x86_modrm_group_opcode(0xF7, 4)
    conditional_subtract = descriptors["x86.scalar.sub.if_uge.imm.gpr32"]
    assert conditional_subtract.encoding_format_id == X86EncodingFormat.CONDITIONAL_SUBTRACT
    assert conditional_subtract.encoding_id == x86_modrm_group_opcode(0x81, 5)

    assert descriptors["x86.scalar.movimm.gpr32"].encoding_format_id == X86EncodingFormat.MOV_IMMEDIATE
    assert descriptors["x86.scalar.movimm.gpr64"].encoding_format_id == X86EncodingFormat.MOV_IMMEDIATE
    zero_extend = descriptors["x86.scalar.movzx.gpr64.gpr32"]
    assert zero_extend.encoding_format_id == X86EncodingFormat.RM_REG
    assert zero_extend.encoding_id == (X86_ENCODING_ID_FORCE_32_BIT | 0x89)
    for descriptor in descriptors.values():
        if ".load" in descriptor.key:
            assert descriptor.encoding_format_id == X86EncodingFormat.MEMORY_REG_RM
        if ".store" in descriptor.key:
            assert descriptor.encoding_format_id == X86EncodingFormat.MEMORY_RM_REG
        if ".lea." in descriptor.key:
            assert descriptor.encoding_format_id == X86EncodingFormat.LEA
    assert descriptors["x86.scalar.jmp"].encoding_format_id == X86EncodingFormat.DIRECT_BRANCH


def test_view_target_generation_rejects_direct_source_output() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        with _RaisesValueError(
            r"x86 descriptor target avx512 is a view of storage target "
            r"avx512_packed_dot"
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
            r"x86 descriptor target avx512_packed_dot cannot emit view "
            r"headers for: missing"
        ):
            x86_descriptors.main(
                [
                    "--target=avx512_packed_dot",
                    f"--header={tmp_path / 'avx512_packed_dot_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_packed_dot_descriptors.c'}",
                    f"--view-header=missing={tmp_path / 'missing_descriptors.h'}",
                ]
            )
