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
from loom.target.arch.x86 import descriptors as x86_descriptor_data


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


def test_storage_generation_emits_current_public_views() -> None:
    with TemporaryDirectory() as temporary_directory:
        tmp_path = Path(temporary_directory)
        assert (
            x86_descriptors.main(
                [
                    "--target=avx512_packed_dot",
                    f"--header={tmp_path / 'avx512_packed_dot_descriptors.h'}",
                    f"--source={tmp_path / 'avx512_packed_dot_descriptors.c'}",
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
        composite_header = (tmp_path / "avx512_packed_dot_descriptors.h").read_text(encoding="utf-8")
        avx512_header = (tmp_path / "avx512_descriptors.h").read_text(encoding="utf-8")
        avx2_header = (tmp_path / "avx2_descriptors.h").read_text(encoding="utf-8")
        avx_vnni_header = (tmp_path / "avx_vnni_descriptors.h").read_text(encoding="utf-8")
        packed_dot_header = (tmp_path / "packed_dot_descriptors.h").read_text(encoding="utf-8")
        scalar_header = (tmp_path / "scalar_descriptors.h").read_text(encoding="utf-8")
        simd128_header = (tmp_path / "simd128_descriptors.h").read_text(encoding="utf-8")

    assert "loom_x86_avx512_core_descriptor_set" in source
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
    assert ".asm_forms = kX86AvxVnniCoreAsmForms," in source
    assert ".asm_forms = kX86Avx512PackedDotCoreStorageAsmForms," in source
    assert ".descriptor_refs = kX86Avx512CoreDescriptorRefs," in source
    assert ".descriptor_refs = kX86PackedDotCoreDescriptorRefs," in source
    assert ".descriptor_refs = kX86ScalarCoreDescriptorRefs," in source
    assert '"avx_vnni.vpdpbusd.ymm"' in source
    assert '"vpdpbusd.ymm"' in source
    assert "loom_x86_avx512_core_descriptor_set" in avx512_header
    assert "loom_x86_avx2_core_descriptor_set" in avx2_header
    assert "loom_x86_avx_vnni_core_descriptor_set" in avx_vnni_header
    assert "loom_x86_packed_dot_core_descriptor_set" in packed_dot_header
    assert "loom_x86_scalar_core_descriptor_set" in scalar_header
    assert "loom_x86_simd128_core_descriptor_set" in simd128_header
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
