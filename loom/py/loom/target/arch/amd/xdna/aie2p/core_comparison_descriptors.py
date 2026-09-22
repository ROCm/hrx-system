# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native vector comparisons, fused extrema and predicate storage views."""

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import RegisterPart

_TARGET_KEY = "amd.xdna.aie2p"
_EL_LOW32_PART = "aie2p.elpredicate.low32"
_EL_HIGH32_PART = "aie2p.elpredicate.high32"
PREDICATE_REGISTER_PARTS = (
    RegisterPart(_EL_LOW32_PART, "aie2p.elpredicate", 0x1),
    RegisterPart(_EL_HIGH32_PART, "aie2p.elpredicate", 0x2),
)

INTEGER_EXTREMA_DESCRIPTOR_SPECS = (
    *(
        _DescriptorSpec(
            f"V{operation.upper()}_{comparison}_{width}_vaddSign{sign_bit}",
            f"{_TARGET_KEY}.{operation}.{signedness}.i{width}x{512 // width}",
            f"integer.{operation}.{signedness}.i{width}x{512 // width}",
            f"II_V{operation.upper()}_{comparison}_{width}_vaddSign{sign_bit}",
            implicit_outputs=("cmp",),
            asm_mnemonic=(
                f"{operation}.{'s' if signedness == 'signed' else 'u'}"
                f"{width}x{512 // width}"
            ),
        )
        for width in (8, 16, 32)
        for operation, comparison in (("min", "GE"), ("max", "LT"))
        for signedness, sign_bit in (("signed", 1), ("unsigned", 0))
    ),
)

PREDICATE_DESCRIPTOR_SPECS = (
    # Scalar AND reads either 32-bit word through its physical eR alias.
    *(
        _DescriptorSpec(
            "AND",
            f"{_TARGET_KEY}.predicate.mask.{word}",
            f"integer.predicate.mask.{word}",
            "II_AND",
            storage_overrides=(("s0", "eLPredicate"),),
            asm_mnemonic=f"predicate.mask.{word}",
            operand_register_parts=(("s0", register_part),),
            encoding_adapter_overrides=(("s0", f"LOOM_eL_{word}"),),
        )
        for word, register_part in (
            ("low32", _EL_LOW32_PART),
            ("high32", _EL_HIGH32_PART),
        )
    ),
    _DescriptorSpec(
        "VEQZ_8",
        f"{_TARGET_KEY}.cmp.eqz.i8x64",
        "integer.cmp.eq.i8x64",
        "II_VEQZ_8",
        storage_overrides=(("cmp", "eLPredicate"),),
    ),
    *(
        _DescriptorSpec(
            f"VEQZ_{width}",
            f"{_TARGET_KEY}.cmp.eqz.i{width}x{512 // width}.el.low32",
            f"integer.cmp.eq.i{width}x{512 // width}.low32",
            f"II_VEQZ_{width}",
            storage_overrides=(("cmp", "eLPredicate"),),
            asm_mnemonic=f"veqz.{width}.el.low32",
            operand_register_parts=(("cmp", _EL_LOW32_PART),),
            encoding_adapter_overrides=(("cmp", "LOOM_eL_low32"),),
        )
        for width in (16, 32)
    ),
    *(
        _DescriptorSpec(
            f"V{relation.upper()}_{width}_vaddSign{sign_bit}",
            (
                f"{_TARGET_KEY}.cmp.{relation}.{signedness}."
                f"i{width}x{512 // width}"
                f"{'.el.low32' if width != 8 else ''}"
            ),
            (
                f"integer.cmp.{relation}.{signedness}."
                f"i{width}x{512 // width}"
                f"{'.low32' if width != 8 else ''}"
            ),
            f"II_V{relation.upper()}_{width}_vaddSign{sign_bit}",
            storage_overrides=(("cmp", "eLPredicate"),),
            asm_mnemonic=(
                f"v{relation}.{'s' if signedness == 'signed' else 'u'}"
                f"{width}x{512 // width}"
                f"{'.el.low32' if width != 8 else ''}"
            ),
            operand_register_parts=((("cmp", _EL_LOW32_PART),) if width != 8 else ()),
            encoding_adapter_overrides=(
                (("cmp", "LOOM_eL_low32"),) if width != 8 else ()
            ),
        )
        for width in (8, 16, 32)
        for relation in ("lt", "ge")
        for signedness, sign_bit in (("signed", 1), ("unsigned", 0))
    ),
    _DescriptorSpec(
        "VSEL_8",
        f"{_TARGET_KEY}.select.i8x64",
        "integer.select.i8x64",
        "II_VSEL_8",
        storage_overrides=(("sel", "eLPredicate"),),
    ),
    _DescriptorSpec(
        "VSEL_32",
        f"{_TARGET_KEY}.select.i32x16",
        "integer.select.i32x16",
        "II_VSEL_32",
    ),
    *(
        _DescriptorSpec(
            f"VSEL_{width}",
            f"{_TARGET_KEY}.select.i{width}x{512 // width}.mask64",
            f"integer.select.i{width}x{512 // width}.mask64",
            f"II_VSEL_{width}",
            storage_overrides=(("sel", "eLPredicate"),),
            asm_mnemonic=f"vsel.{width}.mask64",
            operand_register_parts=(("sel", _EL_LOW32_PART),),
            encoding_adapter_overrides=(("sel", "LOOM_eL_low32"),),
        )
        for width in (16, 32)
    ),
    *(
        _DescriptorSpec(
            operation.upper(),
            f"{_TARGET_KEY}.predicate.{operation}.low32",
            f"integer.predicate.{operation}.low32",
            f"II_{operation.upper()}",
            storage_overrides=(
                ("d0", "eLPredicate"),
                ("s0", "eLPredicate"),
                ("s1", "eLPredicate"),
            ),
            asm_mnemonic=f"predicate.{operation}.low32",
            operand_register_parts=(
                ("d0", _EL_LOW32_PART),
                ("s0", _EL_LOW32_PART),
                ("s1", _EL_LOW32_PART),
            ),
            encoding_adapter_overrides=(
                ("d0", "LOOM_eL_low32"),
                ("s0", "LOOM_eL_low32"),
                ("s1", "LOOM_eL_low32"),
            ),
        )
        for operation in ("and", "or", "xor")
    ),
    *(
        _DescriptorSpec(
            operation.upper(),
            f"{_TARGET_KEY}.predicate.{operation}.high32",
            f"integer.predicate.{operation}.high32",
            f"II_{operation.upper()}",
            storage_overrides=(
                ("d0", "eLPredicate"),
                ("s0", "eLPredicate"),
                ("s1", "eLPredicate"),
            ),
            asm_mnemonic=f"predicate.{operation}.high32",
            operand_register_parts=(
                ("d0", _EL_HIGH32_PART),
                ("s0", _EL_HIGH32_PART),
                ("s1", _EL_HIGH32_PART),
            ),
            encoding_adapter_overrides=(
                ("d0", "LOOM_eL_high32"),
                ("s0", "LOOM_eL_high32"),
                ("s1", "LOOM_eL_high32"),
            ),
            storage_continuation_part=_EL_LOW32_PART,
        )
        for operation in ("and", "or", "xor")
    ),
    _DescriptorSpec(
        "MOVA",
        f"{_TARGET_KEY}.predicate.complete.zero.high32",
        "integer.predicate.complete.zero.high32",
        "II_MOVA_eR",
        storage_overrides=(("dst", "eLPredicate"),),
        asm_mnemonic="predicate.complete.zero.high32",
        operand_register_parts=(("dst", _EL_HIGH32_PART),),
        encoding_adapter_overrides=(("dst", "LOOM_eL_high32_OP_mLdaCg"),),
        storage_continuation_part=_EL_LOW32_PART,
    ),
)

# Native BF16 comparisons distinguish -0 from +0; NaNs make both LT and GE
# false. Fused extrema select the second operand where their comparison is
# true and preserve the first operand's bits elsewhere, including NaN payloads.
# Standalone comparisons write eRS16; fused extrema write the fixed r16 mask.
BF16_COMPARISON_DESCRIPTOR_SPECS = tuple(
    _DescriptorSpec(
        form,
        f"{_TARGET_KEY}.{semantic}.bf16x32.native",
        f"floating.{semantic}.bf16x32.native",
        f"II_{form}",
    )
    for form, semantic in (
        ("VLT_bf16", "cmp.lt"),
        ("VGE_bf16", "cmp.ge"),
        ("VMAX_LT_bf16", "max.lt"),
        ("VMIN_GE_bf16", "min.ge"),
    )
)

# Native standalone masks use every eL or eRS16 candidate. Fused arithmetic
# exposes both its vector and fixed mask results; ordinary Low copies preserve
# a mask when a subsequent fused operation needs the same physical register.
INTEGER_COMPARISON_DESCRIPTOR_SPECS = (
    *(
        _DescriptorSpec(
            f"VEQZ_{width}",
            f"{_TARGET_KEY}.cmp.eqz.i{width}x{512 // width}.native",
            f"integer.cmp.eqz.i{width}x{512 // width}.native",
            f"II_VEQZ_{width}",
            asm_mnemonic=f"veqz.{width}.native",
        )
        for width in (8, 16, 32)
    ),
    *(
        _DescriptorSpec(
            f"V{relation.upper()}_{width}_vaddSign{sign_bit}",
            f"{_TARGET_KEY}.cmp.{relation}.{signedness}.i{width}x{512 // width}.native",
            f"integer.cmp.{relation}.{signedness}.i{width}x{512 // width}.native",
            f"II_V{relation.upper()}_{width}_vaddSign{sign_bit}",
            asm_mnemonic=f"v{relation}.{sign}{width}x{512 // width}.native",
        )
        for width in (8, 16, 32)
        for relation in ("lt", "ge")
        for signedness, sign, sign_bit in (("signed", "s", 1), ("unsigned", "u", 0))
    ),
    *(
        _DescriptorSpec(
            f"VSEL_{width}",
            f"{_TARGET_KEY}.select.i{width}x{512 // width}.native",
            f"integer.select.i{width}x{512 // width}.native",
            f"II_VSEL_{width}",
            asm_mnemonic=f"vsel.{width}.native",
        )
        for width in (8, 16)
    ),
    *(
        _DescriptorSpec(
            f"V{operation.upper()}_{relation.upper()}_{width}_vaddSign{sign_bit}",
            (
                f"{_TARGET_KEY}.{operation}.{relation}.{signedness}."
                f"i{width}x{512 // width}.native"
            ),
            (
                f"integer.{operation}.{relation}.{signedness}."
                f"i{width}x{512 // width}.native"
            ),
            f"II_V{operation.upper()}_{relation.upper()}_{width}_vaddSign{sign_bit}",
            asm_mnemonic=f"v{operation}_{relation}.{sign}{width}x{512 // width}",
        )
        for width in (8, 16, 32)
        for operation, relation in (
            ("min", "ge"),
            ("max", "lt"),
            ("maxdiff", "lt"),
            ("sub", "lt"),
            ("sub", "ge"),
            ("abs", "gtz"),
        )
        for signedness, sign, sign_bit in (("signed", "s", 1), ("unsigned", "u", 0))
    ),
    *(
        _DescriptorSpec(
            form.format(width=width),
            f"{_TARGET_KEY}.{operation}.{relation}.i{width}x{512 // width}.native",
            f"integer.{operation}.{relation}.i{width}x{512 // width}.native",
            f"II_{form.format(width=width)}",
        )
        for width in (8, 16, 32)
        for form, operation, relation in (
            ("VNEG_GTZ_{width}", "neg", "gtz"),
            ("VBNEG_LTZ_s{width}", "bitnot", "ltz"),
        )
    ),
)
