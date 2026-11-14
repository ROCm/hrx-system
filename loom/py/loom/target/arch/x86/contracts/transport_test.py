# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dialect.scalar import conversion
from loom.dialect.vector import defs as vector
from loom.dialect.view import defs as view
from loom.target.arch.x86.contracts.avx2 import X86_AVX2_CONTRACT_FRAGMENT
from loom.target.arch.x86.contracts.avx512 import X86_AVX512_CONTRACT_FRAGMENT
from loom.target.arch.x86.descriptors import (
    X86_AVX2_DESCRIPTOR_SET,
    X86_AVX512_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.x86.feature_bits import FEATURE_AVX512_VL
from loom.target.arch.x86.vector_families import X86_VECTOR_STORAGE_FAMILIES
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    GuardKind,
    Scalar,
    Vector,
)
from loom.target.low_descriptors import EffectKind, OperandAddressMapKind

_PROFILES = (
    (
        "avx2",
        X86_AVX2_DESCRIPTOR_SET,
        X86_AVX2_CONTRACT_FRAGMENT,
        (8, 16, 32, 64, 128, 256),
    ),
    (
        "avx512",
        X86_AVX512_CORE_DESCRIPTOR_SET,
        X86_AVX512_CONTRACT_FRAGMENT,
        (8, 16, 32, 64, 128, 256, 512),
    ),
)


def _value_type(rule: DescriptorRule, field: str):
    return next(
        guard.type_pattern
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE and guard.field == field
    )


def _check_simd_domain(row, evex: bool) -> None:
    for operand in row.operands:
        if any(
            alt.reg_class in ("x86.xmm", "x86.ymm", "x86.zmm")
            for alt in operand.reg_alts
        ):
            assert operand.addressable_unit_count == (0 if evex else 16)
            assert operand.address_map_kind is (
                OperandAddressMapKind.DIRECT
                if evex
                else OperandAddressMapKind.LOW_SUBSET
            )


def test_bit_transfers_preserve_both_directions_and_scalar_feature_requirements() -> (
    None
):
    for profile, descriptor_set, _, _ in _PROFILES:
        rows = {row.key: row for row in descriptor_set.descriptors}
        evex = profile == "avx512"
        for bits, mnemonic in ((32, "vmovd"), (64, "vmovq")):
            for to_simd in (False, True):
                suffix = f"xmm.gpr{bits}" if to_simd else f"gpr{bits}.xmm"
                row = rows[f"x86.{profile}.{mnemonic}.{suffix}"]
                assert row.encoding_format_id == (0x8140 if to_simd else 0x8041)
                assert row.encoding_id == (
                    0x0500
                    | (0x1000 if bits == 64 else 0)
                    | (0x2000 if evex else 0)
                    | (0x6E if to_simd else 0x7E)
                )
                assert row.feature_mask_words == ()
                assert row.effects == ()
                _check_simd_domain(row, evex)


def test_memory_forms_keep_exact_footprints_and_tuple_widths() -> None:
    for profile, descriptor_set, _, widths in _PROFILES:
        rows = {row.key: row for row in descriptor_set.descriptors}
        evex = profile == "avx512"
        for bits in widths[2:]:
            suffix = "xmm" if bits <= 128 else "ymm" if bits == 256 else "zmm"
            mnemonic = {32: "vmovss", 64: "vmovsd"}.get(bits, "vmovdqu32")
            for indexed in (False, True):
                for load in (False, True):
                    operation = "load" if load else "store"
                    address = ".indexed" if indexed else ""
                    row = rows[
                        f"x86.{profile}.{mnemonic}.{operation}{address}.{suffix}"
                    ]
                    assert len(row.effects) == 1
                    assert row.effects[0].kind is (
                        EffectKind.READ if load else EffectKind.WRITE
                    )
                    assert row.effects[0].width_bits == bits
                    assert len(row.operands) == (3 if indexed else 2)
                    assert row.feature_mask_words == (
                        (FEATURE_AVX512_VL,) if evex and bits in (128, 256) else ()
                    )
                    tuple_selector = {32: 14, 64: 15}.get(bits, 12)
                    assert row.encoding_format_id == (
                        (0xC100 if load else 0xD201) | (tuple_selector << 4)
                    )
                    opcode = (
                        (0x10 if load else 0x11)
                        if bits < 128
                        else (0x6F if load else 0x7F)
                    )
                    assert row.encoding_id == (
                        opcode
                        | 0x100
                        | ((3 if bits == 64 else 2) << 10)
                        | (0x1000 if evex and bits == 64 else 0)
                        | (0x2000 if evex else 0)
                        | ({32: 0, 64: 0, 128: 0, 256: 1, 512: 2}[bits] << 14)
                    )
                    _check_simd_domain(row, evex)


def test_read_only_payloads_use_full_carriers_with_vl_gating() -> None:
    for profile, descriptor_set, _, widths in _PROFILES:
        rows = {row.key: row for row in descriptor_set.descriptors}
        evex = profile == "avx512"
        for bits in (width for width in widths if width >= 128):
            suffix = {128: "xmm", 256: "ymm", 512: "zmm"}[bits]
            mnemonic = "vmovdqu64" if evex else "vmovdqu"
            row = rows[f"x86.{profile}.{mnemonic}.rodata.{suffix}"]
            assert row.encoding_format_id == 0xE440
            assert row.encoding_id == (
                0x096F
                | (0x3000 if evex else 0)
                | ({128: 0, 256: 1, 512: 2}[bits] << 14)
            )
            assert row.feature_mask_words == (
                (FEATURE_AVX512_VL,) if evex and bits < 512 else ()
            )
            _check_simd_domain(row, evex)


def test_scalar_memory_keeps_one_exact_read_or_write_without_bitcasts() -> None:
    for profile, _, fragment, _ in _PROFILES:
        for operation, field in (
            (view.view_load, "result"),
            (view.view_store, "value"),
        ):
            for element, byte_count, mnemonic in (
                ("f32", 4, "vmovss"),
                ("f64", 8, "vmovsd"),
            ):
                rules = [
                    case
                    for case in fragment.cases
                    if isinstance(case, DescriptorRule)
                    and case.source_op is operation
                    and _value_type(case, field) == Scalar(element)
                ]
                assert len(rules) == 9
                for rule in rules:
                    memory = [
                        emit
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                        and emit.descriptor.semantic_tag.startswith("memory.")
                    ]
                    assert len(memory) == 1
                    assert memory[0].descriptor.key.startswith(
                        f"x86.{profile}.{mnemonic}."
                    )
                    assert memory[0].source_memory.element_byte_count == byte_count
                    assert memory[0].source_memory.vector_lane_count == 1
                    assert not any(
                        emit.descriptor.mnemonic in ("vmovd", "vmovq")
                        for emit in rule.emit
                        if isinstance(emit, EmitDescriptorOp)
                    )


def test_vector_memory_preserves_every_payload_width_and_source_address_plan() -> None:
    for profile, _, fragment, widths in _PROFILES:
        for operation, field in (
            (vector.vector_load, "result"),
            (vector.vector_store, "value"),
        ):
            for family in X86_VECTOR_STORAGE_FAMILIES:
                for bits in widths:
                    if bits < family.element_bit_width:
                        continue
                    lanes = bits // family.element_bit_width
                    value_type = Vector(family.element_names, lanes=lanes)
                    rules = [
                        case
                        for case in fragment.cases
                        if isinstance(case, DescriptorRule)
                        and case.source_op is operation
                        and _value_type(case, field) == value_type
                    ]
                    assert len(rules) == 9
                    for rule in rules:
                        memory = [
                            emit
                            for emit in rule.emit
                            if isinstance(emit, EmitDescriptorOp)
                            and emit.descriptor.semantic_tag.startswith("memory.")
                        ]
                        assert len(memory) == 1
                        assert (
                            memory[0].source_memory.element_byte_count
                            == family.element_bit_width // 8
                        )
                        assert memory[0].source_memory.vector_lane_count == lanes
                        if bits <= 16:
                            prefix = (
                                "x86.scalar.movzx."
                                if operation is vector.vector_load
                                else "x86.scalar.mov."
                            )
                            moves = [
                                emit.descriptor
                                for emit in rule.emit
                                if isinstance(emit, EmitDescriptorOp)
                                and emit.descriptor.mnemonic == "vmovd"
                            ]
                            assert len(moves) == 1
                            assert moves[0].key.startswith(f"x86.{profile}.")
                            _check_simd_domain(moves[0], profile == "avx512")
                        else:
                            mnemonic = {32: "vmovss", 64: "vmovsd"}.get(
                                bits, "vmovdqu32"
                            )
                            prefix = f"x86.{profile}.{mnemonic}."
                        assert memory[0].descriptor.key.startswith(prefix)


def test_scalar_constants_and_bitcasts_select_the_profile_transfer_family() -> None:
    for profile, _, fragment, _ in _PROFILES:
        for source_type, result_type, mnemonic, suffix in (
            ("i32", "f32", "vmovd", "xmm.gpr32"),
            ("f32", "i32", "vmovd", "gpr32.xmm"),
            ("i64", "f64", "vmovq", "xmm.gpr64"),
            ("f64", "i64", "vmovq", "gpr64.xmm"),
        ):
            rules = [
                case
                for case in fragment.cases
                if isinstance(case, DescriptorRule)
                and case.source_op is conversion.scalar_bitcast
                and _value_type(case, "input") == Scalar(source_type)
                and _value_type(case, "result") == Scalar(result_type)
            ]
            assert len(rules) == 1
            assert rules[0].descriptor.key == f"x86.{profile}.{mnemonic}.{suffix}"
        for element, mnemonic, suffix in (
            ("f32", "vmovd", "gpr32"),
            ("f64", "vmovq", "gpr64"),
        ):
            constants = [
                case
                for case in fragment.cases
                if isinstance(case, DescriptorRule)
                and case.source_op is conversion.scalar_constant
                and _value_type(case, "result") == Scalar(element)
            ]
            bit_recipes = [
                rule for rule in constants if rule.descriptor.mnemonic == mnemonic
            ]
            assert len(bit_recipes) == 1
            assert (
                bit_recipes[0].descriptor.key
                == f"x86.{profile}.{mnemonic}.xmm.{suffix}"
            )
            for zero in (
                rule for rule in constants if rule.descriptor.mnemonic == "vxorps"
            ):
                assert zero.priority > 1
