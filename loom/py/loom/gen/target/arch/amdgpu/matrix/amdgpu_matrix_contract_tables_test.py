# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.gen.target.arch.amdgpu.matrix import amdgpu_matrix_contract_tables
from loom.target.arch.amdgpu.matrix_contracts import (
    AMDGPU_MATRIX_CONTRACTS,
    AmdgpuMatrixContract,
    payload,
)
from loom.target.arch.amdgpu.matrix_formats import (
    AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
)
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
)
from loom.target.low_descriptors import Immediate, ImmediateKind

_GLOBAL_MATRIX_DESCRIPTOR_CATALOG = amdgpu_matrix_contract_tables._global_matrix_descriptor_catalog()


def _contract(name: str) -> AmdgpuMatrixContract:
    for contract in AMDGPU_MATRIX_CONTRACTS:
        if contract.name == name:
            return contract
    raise ValueError(f"unknown AMDGPU matrix contract {name!r}")


def _is_rdna4_contract(contract: AmdgpuMatrixContract) -> bool:
    return any(
        feature
        in {
            "wmma_gfx12",
            "swmmac_gfx12",
            "wmma_gfx1250",
            "wmma_gfx1250_scale_f8f6f4",
            "swmmac_gfx1250",
        }
        for feature in contract.features
    )


def _is_cdna_dense_mfma_16x16x32_f32_contract(
    contract: AmdgpuMatrixContract,
) -> bool:
    return (
        contract.family == "mfma"
        and any(feature in {"mfma_gfx940_fp8", "mfma_gfx950"} for feature in contract.features)
        and contract.tile_shape == (16, 16, 32)
        and not contract.flags
        and contract.accumulator.numeric_type == "f32"
        and contract.result.numeric_type == "f32"
        and contract.lhs.numeric_type in {"f16", "bf16", "fp8", "bf8"}
        and contract.rhs.numeric_type in {"f16", "bf16", "fp8", "bf8"}
    )


def _is_cdna_dense_mfma_32x32x16_f32_contract(
    contract: AmdgpuMatrixContract,
) -> bool:
    return (
        contract.family == "mfma"
        and any(feature in {"mfma_gfx940_fp8", "mfma_gfx950"} for feature in contract.features)
        and contract.tile_shape == (32, 32, 16)
        and not contract.flags
        and contract.accumulator.numeric_type == "f32"
        and contract.result.numeric_type == "f32"
        and contract.lhs.numeric_type in {"f16", "bf16", "fp8", "bf8"}
        and contract.rhs.numeric_type in {"f16", "bf16", "fp8", "bf8"}
    )


def _is_cdna_dense_mfma_f32_contract(
    contract: AmdgpuMatrixContract,
) -> bool:
    return (
        contract.family,
        contract.flags,
        contract.scale_kind,
        contract.accumulator.numeric_type,
        contract.result.numeric_type,
    ) == ("mfma", (), "none", "f32", "f32")


def _payload_numeric_types(contract: AmdgpuMatrixContract) -> tuple[str, ...]:
    return (
        contract.lhs.numeric_type,
        contract.rhs.numeric_type,
        contract.accumulator.numeric_type,
        contract.result.numeric_type,
    )


def _contract_initializer(contract: AmdgpuMatrixContract) -> str:
    catalog = _GLOBAL_MATRIX_DESCRIPTOR_CATALOG
    return amdgpu_matrix_contract_tables._contract_initializer(
        contract,
        keys_by_semantic_tag=catalog.keys_by_semantic_tag,
        descriptor_shapes_by_key=catalog.shapes_by_key,
        descriptor_immediates_by_key=catalog.immediates_by_key,
    )


def _global_descriptor_keys() -> tuple[str | None, ...]:
    catalog = _GLOBAL_MATRIX_DESCRIPTOR_CATALOG
    return amdgpu_matrix_contract_tables._contract_descriptor_keys(
        keys_by_semantic_tag=catalog.keys_by_semantic_tag,
        descriptor_shapes_by_key=catalog.shapes_by_key,
        descriptor_immediates_by_key=catalog.immediates_by_key,
    )


def test_generation_rejects_gfx12_profile_for_rdna3_5_inventory() -> None:
    global_descriptor_keys = _global_descriptor_keys()
    catalog = amdgpu_matrix_contract_tables._matrix_descriptor_catalog_for_builder(
        "rdna3_5",
        descriptor_ref_keys=(descriptor_key for descriptor_key in global_descriptor_keys if descriptor_key is not None),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"descriptor set 'amdgpu\.rdna3_5\.core'.*profile 'wmma_gfx12'.*"
            r"contract 'swmmac\.f32\.16x16x32\.f16'.*no matching low descriptor"
        ),
    ):
        amdgpu_matrix_contract_tables._validate_matrix_profile_descriptor_catalog(
            descriptor_set_key="amdgpu.rdna3_5.core",
            profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
            catalog=catalog,
            global_descriptor_keys=global_descriptor_keys,
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
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_GFX1250_SWMMAC_16BIT_16X16X128_PACKED8" in swmmac
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_32X16X128_F4" in scaled_f4


def test_generation_resolves_every_gfx125x_selector_driven_wmma_abi() -> None:
    for scale_name in ("", "scale.", "scale16."):
        for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
            for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
                name = f"wmma.{scale_name}f32.16x16x128.f8f6f4.{lhs_format.token}.{rhs_format.token}"
                contract = _contract(name)
                initializer = _contract_initializer(contract)
                descriptor_suffix = name.removeprefix("wmma.").replace(".", "_").upper()
                layout_suffix = (f"GFX125X_WMMA_F32_16X16X128_{lhs_format.token}_{rhs_format.token}").upper()

                assert f".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_{descriptor_suffix}" in initializer
                assert f".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_{layout_suffix}" in initializer
                assert contract.lhs.numeric_type == (lhs_format.contract_numeric_type)
                assert contract.lhs.register_count == (lhs_format.register_count_for(64))
                assert contract.lhs.element_count == 64
                assert contract.rhs.numeric_type == (rhs_format.contract_numeric_type)
                assert contract.rhs.register_count == (rhs_format.register_count_for(64))
                assert contract.rhs.element_count == 64


def test_generation_resolves_every_cdna4_selector_driven_mfma_abi() -> None:
    for scale_name in ("", "scale."):
        for tile_shape in ("16x16x128", "32x32x64"):
            for lhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
                for rhs_format in AMDGPU_F8F6F4_MATRIX_PHYSICAL_FORMATS:
                    name = f"mfma.{scale_name}f32.{tile_shape}.f8f6f4.{lhs_format.token}.{rhs_format.token}"
                    contract = _contract(name)
                    initializer = _contract_initializer(contract)
                    descriptor_suffix = name.removeprefix("mfma.").replace(".", "_").upper()

                    assert (f".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MFMA_{descriptor_suffix}") in initializer
                    assert contract.lhs.numeric_type == (lhs_format.contract_numeric_type)
                    assert contract.lhs.register_count == (lhs_format.register_count_for(32))
                    assert contract.lhs.element_count == 32
                    assert contract.rhs.numeric_type == (rhs_format.contract_numeric_type)
                    assert contract.rhs.register_count == (rhs_format.register_count_for(32))
                    assert contract.rhs.element_count == 32


def test_generation_emits_gfx950_implicit_scale_format_masks() -> None:
    initializer = _contract_initializer(_contract("mfma.scale.f32.16x16x128.f8f6f4.f4.f4"))

    assert (".implicit_scale_format_selector_bits = (loom_amdgpu_matrix_scale_format_selector_bits_t)((1u << LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0))") in initializer


def test_generation_resolves_sparse_fragment_layouts() -> None:
    rdna = _contract_initializer(_contract("swmmac.f32.16x16x32.f16"))
    cdna = _contract_initializer(_contract("smfmac.f32.16x16x32.f16"))

    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED16" in rdna
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_SMFMAC_32BIT_16X16X32_PACKED16" in cdna
    assert ".source_requirement_flags = 0" in rdna
    assert ".source_requirement_flags = 0" in cdna


def test_generation_uses_static_aggregate_initializers() -> None:
    initializer = _contract_initializer(_contract("mfma.f32.16x16x1.f32"))

    assert ".tile_shape = {" in initializer
    assert ".lhs_payload = {" in initializer
    assert "_t){" not in initializer


def test_generation_resolves_gfx12_wmma_abi_shape_variants() -> None:
    f16 = _contract_initializer(_contract("wmma.f32.16x16x16.f16.gfx12"))
    bf16 = _contract_initializer(_contract("wmma.bf16.16x16x16.bf16.gfx12"))
    iu4 = _contract_initializer(_contract("wmma.i32.16x16x16.iu4.gfx12"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16" in f16
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_F16" in f16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_BF16_16X16X16_BF16" in bf16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4" in iu4
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU4" in iu4


def test_generation_resolves_gfx12_wave64_matrix_abi_shape_variants() -> None:
    fp8_bf8 = _contract_initializer(_contract("wmma.f32.16x16x16.fp8.bf8.w64"))
    iu4 = _contract_initializer(_contract("wmma.i32.16x16x16.iu4.gfx12.w64"))
    swmmac = _contract_initializer(_contract("swmmac.f32.16x16x32.f16.w64"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_FP8_BF8_W64" in fp8_bf8
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X16_PACKED8_W64" in fp8_bf8
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64" in iu4
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_I32_16X16X16_IU4_W64" in iu4
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_F16_W64" in swmmac
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_SWMMAC_32BIT_16X16X32_PACKED16_W64" in swmmac


def test_generation_resolves_gfx1250_wmma_f32_fragment_layouts() -> None:
    f32 = _contract_initializer(_contract("wmma.f32.16x16x4.f32"))
    f16 = _contract_initializer(_contract("wmma.f32.16x16x32.f16"))
    bf16 = _contract_initializer(_contract("wmma.f32.16x16x32.bf16"))

    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X4_F32" in f32
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X32_F16" in f16
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32_16X16X32_BF16" in bf16


def test_generation_resolves_gfx950_mfma_f32_fragment_layouts() -> None:
    f16_16x16 = _contract_initializer(_contract("mfma.f32.16x16x32.f16"))
    bf16_16x16 = _contract_initializer(_contract("mfma.f32.16x16x32.bf16"))
    f16_32x32 = _contract_initializer(_contract("mfma.f32.32x32x16.f16"))
    bf16_32x32 = _contract_initializer(_contract("mfma.f32.32x32x16.bf16"))
    fp8_bf8_32x32 = _contract_initializer(_contract("mfma.f32.32x32x16.fp8.bf8"))

    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_F16" in f16_16x16
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X32_BF16" in bf16_16x16
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_F16" in f16_32x32
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_BF16" in bf16_32x32
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_32X32X16_PACKED8" in fp8_bf8_32x32


def test_generation_resolves_f64_mfma_fragment_layout() -> None:
    f64 = _contract_initializer(_contract("mfma.f64.16x16x4.f64"))

    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F64_16X16X4_F64" in f64


def test_generation_emits_blocked_mfma_tile_shapes() -> None:
    f32 = _contract_initializer(_contract("mfma.f32.16x16x4.f16"))
    f64 = _contract_initializer(_contract("mfma.f64.4x4x4.f64"))
    i32 = _contract_initializer(_contract("mfma.i32.4x4x4.i8"))
    single_block = _contract_initializer(_contract("mfma.f32.16x16x16.f16"))

    assert ".block_count = 4" in f32
    assert ".block_count = 4" in f64
    assert ".block_count = 16" in i32
    assert ".block_count = 1" in single_block


def test_generation_rejects_invalid_block_count() -> None:
    contract = replace(_contract("mfma.f32.16x16x4.f16"), block_count=0)

    with pytest.raises(ValueError, match="invalid tile shape"):
        _contract_initializer(contract)


def test_generation_rejects_source_indistinguishable_contracts() -> None:
    contract = _contract("swmmac.f32.16x16x64.bf16")
    duplicate_contract = replace(
        contract,
        name="swmmac.unknown.16x16x64.bf16",
        intrinsic_name="llvm.amdgcn.swmmac.unknown.16x16x64.bf16",
        semantic_tag="matrix.swmmac.unknown.16x16x64.bf16",
    )

    with pytest.raises(
        ValueError,
        match=(f"feature profile '{AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250}'.*'{contract.name}'.*'{duplicate_contract.name}'.*wave32"),
    ):
        amdgpu_matrix_contract_tables._validate_matrix_source_contracts((contract, duplicate_contract))


def test_generation_accepts_source_contracts_with_disjoint_wave_sizes() -> None:
    contract = _contract("swmmac.f32.16x16x64.bf16")
    wave32_contract = replace(contract, wave_size="32")
    wave64_contract = replace(
        contract,
        name="swmmac.unknown.16x16x64.bf16",
        wave_size="64",
    )

    amdgpu_matrix_contract_tables._validate_matrix_source_contracts((wave32_contract, wave64_contract))


def test_generation_audits_cdna_dense_mfma_16x16x32_f32_layout_surface() -> None:
    missing = tuple(contract.name for contract in AMDGPU_MATRIX_CONTRACTS if _is_cdna_dense_mfma_16x16x32_f32_contract(contract) and contract.fragment_layout is None)

    assert missing == ()


def test_generation_audits_cdna_dense_mfma_32x32x16_f32_layout_surface() -> None:
    missing = tuple(contract.name for contract in AMDGPU_MATRIX_CONTRACTS if _is_cdna_dense_mfma_32x32x16_f32_contract(contract) and contract.fragment_layout is None)

    assert missing == ()


def test_generation_audits_cdna_dense_mfma_f32_layout_surface() -> None:
    missing = tuple(contract.name for contract in AMDGPU_MATRIX_CONTRACTS if _is_cdna_dense_mfma_f32_contract(contract) and contract.fragment_layout is None)

    assert missing == ()


def test_generation_validates_every_referenced_fragment_layout() -> None:
    referenced_layouts = {contract.fragment_layout for contract in AMDGPU_MATRIX_CONTRACTS if contract.fragment_layout is not None}
    canonical_layouts = {layout.key for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.values() if layout.canonical_key is None}

    assert referenced_layouts == canonical_layouts
    for contract in AMDGPU_MATRIX_CONTRACTS:
        amdgpu_matrix_contract_tables._validate_contract_fragment_layout(contract)


def test_gfx11_f16_transposed_result_layout_is_symmetric() -> None:
    canonical = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16"]
    transposed = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY["rdna3_wmmar3_f32_16x16x16_f16_transposed_result"]

    assert transposed.canonical_key == canonical.key
    assert transposed.instruction_operand_order == ("rhs", "lhs")
    assert transposed.lhs == canonical.lhs
    assert transposed.rhs == canonical.rhs
    assert canonical.lhs.axes[1] == canonical.rhs.axes[2]
    assert canonical.lhs.axes[3] == canonical.rhs.axes[3]
    expected_result_axes = (
        canonical.result.axes[0],
        canonical.result.axes[2],
        canonical.result.axes[1],
        canonical.result.axes[3],
    )
    assert transposed.accumulator.axes == expected_result_axes
    assert transposed.result.axes == expected_result_axes


def test_generation_audits_rdna4_float_fragment_layout_surface() -> None:
    memory_numeric_types = {"f16", "bf16", "f32"}
    missing = tuple(
        contract.name
        for contract in AMDGPU_MATRIX_CONTRACTS
        if _is_rdna4_contract(contract)
        and all(numeric_type in memory_numeric_types for numeric_type in _payload_numeric_types(contract))
        and contract.fragment_layout is None
        and "fragment_layout" not in contract.source_requirements
    )

    assert missing == ()


def test_generation_audits_rdna4_unknown_fragment_layout_exceptions() -> None:
    unknown_unrequired = tuple(
        contract for contract in AMDGPU_MATRIX_CONTRACTS if _is_rdna4_contract(contract) and contract.fragment_layout is None and "fragment_layout" not in contract.source_requirements
    )

    assert unknown_unrequired
    for contract in unknown_unrequired:
        numeric_types = set(_payload_numeric_types(contract))
        assert contract.family == "wmma", contract.name
        assert numeric_types - {"f16", "bf16", "f32"} or "matrix_formats" in contract.flags or "scale_formats" in contract.flags or "sign_select" in contract.flags, contract.name


def test_generation_audits_dense_integer_wmma_fragment_layout_surface() -> None:
    missing = tuple(contract.name for contract in AMDGPU_MATRIX_CONTRACTS if contract.family == "wmma" and contract.result.numeric_type == "i32" and contract.fragment_layout is None)

    assert missing == ()


def test_generation_resolves_gfx11_wmma_wave64_abi_shape_variants() -> None:
    f32_f16 = _contract_initializer(_contract("wmma.f32.16x16x16.f16.w64"))
    f16 = _contract_initializer(_contract("wmma.f16.16x16x16.f16.w64"))
    iu8 = _contract_initializer(_contract("wmma.i32.16x16x16.iu8.w64"))

    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16_W64" in f32_f16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X16_F16_W64" in f16
    assert ".low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64" in iu8
    assert ".fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_I32_16X16X16_IU8_W64" in iu8


def test_generation_rejects_low_descriptor_payload_shape_drift() -> None:
    contract = _contract("wmma.i32.16x16x32.iu4")
    drifted_contract = replace(
        contract,
        lhs=payload("iu4", 0, 0),
        fragment_layout=None,
    )

    try:
        _contract_initializer(drifted_contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'wmma.i32.16x16x32.iu4'" in message
        assert "payload shape" in message
        assert "descriptor key(s) amdgpu.v_wmma_i32_16x16x32_iu4" in message
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


def test_generation_rejects_unmapped_matrix_descriptor_immediates() -> None:
    contract = _contract("swmmac.f32.16x16x32.f16")

    try:
        amdgpu_matrix_contract_tables._validate_contract_descriptor_immediates(
            contract,
            "amdgpu.v_swmmac_f32_16x16x32_f16",
            descriptor_immediates_by_key={
                "amdgpu.v_swmmac_f32_16x16x32_f16": (Immediate("surprise", ImmediateKind.UNSIGNED),),
            },
        )
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'swmmac.f32.16x16x32.f16'" in message
        assert "unmapped immediate 'surprise'" in message
    else:
        raise AssertionError("expected unmapped immediate validation to fail")


def test_generation_requires_integer_matrix_control_immediates() -> None:
    contract = _contract("swmmac.i32.16x16x32.iu8")

    with pytest.raises(ValueError, match=r"required immediate field.*clamp"):
        amdgpu_matrix_contract_tables._validate_contract_descriptor_immediates(
            contract,
            "amdgpu.v_swmmac_i32_16x16x32_iu8",
            descriptor_immediates_by_key={
                "amdgpu.v_swmmac_i32_16x16x32_iu8": (Immediate("neg_lo", ImmediateKind.UNSIGNED),),
            },
        )


def test_generation_rejects_unsupported_wait_state_result_payload_count() -> None:
    contract = replace(
        _contract("mfma.f32.16x16x16.f16"),
        result=payload("f32", 3, 3),
    )

    try:
        _contract_initializer(contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'mfma.f32.16x16x16.f16'" in message
        assert "unsupported wait-state result payload register count 3" in message
        assert "expected one of 2, 4, 8, 16, 32" in message
    else:
        raise AssertionError("expected wait-state result payload validation to fail")


def test_generation_rejects_selector_and_implicit_scale_format_overlap() -> None:
    contract = replace(
        _contract("wmma.scale.f32.16x16x128.f8f6f4.f8.f8"),
        implicit_scale_formats=("e8m0",),
    )

    try:
        _contract_initializer(contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'wmma.scale.f32.16x16x128.f8f6f4.f8.f8'" in message
        assert "scale-format selector operands and implicit scale formats" in message
    else:
        raise AssertionError("expected selector/implicit scale validation to fail")


def test_generation_rejects_scaled_contract_without_scale_format_policy() -> None:
    contract = replace(
        _contract("mfma.scale.f32.16x16x128.f8f6f4.f4.f4"),
        implicit_scale_formats=(),
    )

    try:
        _contract_initializer(contract)
    except ValueError as exc:
        message = str(exc)
        assert "AMDGPU matrix contract 'mfma.scale.f32.16x16x128.f8f6f4.f4.f4'" in message
        assert "selector operands or implicit scale formats" in message
    else:
        raise AssertionError("expected scale format policy validation to fail")
