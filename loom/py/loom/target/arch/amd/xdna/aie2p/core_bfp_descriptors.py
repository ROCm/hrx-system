# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native packed BFP mantissa and exponent storage transfers."""

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import RegisterPart

_TARGET_KEY = "amd.xdna.aie2p"
_MANTISSAS = "aie2p.mexa.mantissas"
_EXPONENTS_LOW = "aie2p.mexa.exponents.low32"
_EXPONENTS_HIGH = "aie2p.mexa.exponents.high32"
_MANTISSAS_AND_LOW = "aie2p.mexa.mantissas.exponents.low32"

BFP_REGISTER_PARTS = (
    RegisterPart(_MANTISSAS, "aie2p.mexa", 0x1),
    RegisterPart(_EXPONENTS_LOW, "aie2p.mexa", 0x2),
    RegisterPart(_EXPONENTS_HIGH, "aie2p.mexa", 0x4),
    RegisterPart(_MANTISSAS_AND_LOW, "aie2p.mexa", 0x3),
)

BFP_STORAGE_DESCRIPTOR_SPECS = (
    _DescriptorSpec(
        "VMOV_alu_mv_mv_x",
        f"{_TARGET_KEY}.bfp.mantissas.from.vector512",
        "register.bfp.mantissas.from.vector512",
        "II_VMOV_alu_mv_mv_x_eXe_eXe",
        storage_overrides=(("dst", "mEXa"), ("src", "VEC256")),
        operand_register_parts=(("dst", _MANTISSAS),),
        encoding_adapter_overrides=(
            ("dst", "LOOM_mEXa_mantissas_OP_mMvBMXDst"),
            ("src", "LOOM_mXm_OP_mMvBMXSrc"),
        ),
        asm_mnemonic="bfp.mantissas.from.vector512",
    ),
    *(
        _DescriptorSpec(
            f"MOV_alu_mv_mv_mv_e_mv_r_to_{exponent_class}",
            f"{_TARGET_KEY}.bfp.exponents.{half}.from.scalar",
            f"register.bfp.exponents.{half}.from.scalar",
            f"II_MOV_alu_mv_mv_mv_e_mv_r_to_{exponent_class}",
            storage_overrides=(("dst", "mEXa"),),
            operand_register_parts=(("dst", part),),
            encoding_adapter_overrides=(
                ("dst", f"LOOM_mEXa_exponents_{half}_OP_m{exponent_class.title()}m"),
            ),
            storage_continuation_part=continuation,
            asm_mnemonic=f"bfp.exponents.{half}.from.scalar",
        )
        for half, exponent_class, part, continuation in (
            ("low32", "el", _EXPONENTS_LOW, _MANTISSAS),
            ("high32", "eh", _EXPONENTS_HIGH, _MANTISSAS_AND_LOW),
        )
    ),
)
