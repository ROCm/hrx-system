# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

from loom.gen.target.arch.amdgpu.matrix import amdgpu_matrix_contract_tables
from loom.target.arch.amdgpu.matrix_contracts import (
    AMDGPU_MATRIX_CONTRACTS,
    AmdgpuMatrixContract,
    payload,
)


def _contract(name: str) -> AmdgpuMatrixContract:
    for contract in AMDGPU_MATRIX_CONTRACTS:
        if contract.name == name:
            return contract
    raise ValueError(f"unknown AMDGPU matrix contract {name!r}")


def _contract_initializer(contract: AmdgpuMatrixContract) -> str:
    return amdgpu_matrix_contract_tables._contract_initializer(
        contract,
        keys_by_semantic_tag=(amdgpu_matrix_contract_tables._matrix_descriptor_keys_by_semantic_tag()),
        descriptor_shapes_by_key=(amdgpu_matrix_contract_tables._matrix_descriptor_shapes_by_key()),
    )


def test_generation_derives_semantic_tag_descriptor_ref() -> None:
    initializer = _contract_initializer(_contract("swmmac.f32.16x16x32.f16"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_F16" in initializer


def test_generation_accepts_one_matching_descriptor_shape_variant() -> None:
    initializer = _contract_initializer(_contract("wmma.f32.16x16x16.f16"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16" in initializer


def test_generation_resolves_gfx1250_supplemental_matrix_descriptors() -> None:
    wmma = _contract_initializer(_contract("wmma.f32.16x16x128.fp8.bf8"))
    swmmac = _contract_initializer(_contract("swmmac.f16.16x16x128.bf8.fp8"))
    scaled_f4 = _contract_initializer(_contract("wmma.scale16.f32.32x16x128.f4"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_FP8_BF8" in wmma
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_BF8_FP8" in swmmac
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_32X16X128_F4" in scaled_f4


def test_generation_resolves_gfx12_wmma_abi_shape_variants() -> None:
    f16 = _contract_initializer(_contract("wmma.f32.16x16x16.f16.gfx12"))
    bf16 = _contract_initializer(_contract("wmma.bf16.16x16x16.bf16.gfx12"))
    iu4 = _contract_initializer(_contract("wmma.i32.16x16x16.iu4.gfx12"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16" in f16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_BF16_16X16X16_BF16" in bf16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4" in iu4


def test_generation_rejects_low_descriptor_payload_shape_drift() -> None:
    contract = _contract("swmmac.f32.16x16x32.f16")
    drifted_contract = replace(contract, lhs=payload("f16", 0, 0))

    try:
        _contract_initializer(drifted_contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'swmmac.f32.16x16x32.f16'" in message
        assert "payload shape" in message
        assert "descriptor key(s) amdgpu.v_swmmac_f32_16x16x32_f16" in message
    else:
        raise AssertionError("expected payload shape validation to fail")


def test_generation_rejects_ambiguous_shape_matched_descriptor_keys() -> None:
    contract = _contract("swmmac.f32.16x16x32.f16")
    descriptor_shapes_by_key = {
        "amdgpu.first": (amdgpu_matrix_contract_tables._contract_matrix_descriptor_shape(contract),),
        "amdgpu.second": (amdgpu_matrix_contract_tables._contract_matrix_descriptor_shape(contract),),
    }

    try:
        amdgpu_matrix_contract_tables._resolve_contract_descriptor_key(
            contract,
            keys_by_semantic_tag={
                "matrix.swmmac.f32.16x16x32.f16": (
                    "amdgpu.first",
                    "amdgpu.second",
                ),
            },
            descriptor_shapes_by_key=descriptor_shapes_by_key,
        )
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'swmmac.f32.16x16x32.f16'" in message
        assert "ambiguously matches descriptor key(s) amdgpu.first, amdgpu.second" in message
    else:
        raise AssertionError("expected ambiguous descriptor resolution to fail")
