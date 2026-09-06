# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU matrix contract descriptor tables."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.support.native_layout import (  # noqa: E402
    NativeContractionFactTable,
    NativeTransitionFactTable,
)
from loom.target.arch.amdgpu.descriptor_overlay import (  # noqa: E402
    AmdgpuDescriptorOverlay,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
    _amdgpu_core_descriptor_set_bases,
    amdgpu_descriptor_ref_keys,
)
from loom.target.arch.amdgpu.matrix_contracts import (  # noqa: E402
    AMDGPU_MATRIX_CONTRACTS,
    AMDGPU_MATRIX_NUMERIC_TYPE_BIT_COUNTS,
    AmdgpuMatrixContract,
    AmdgpuMatrixPayload,
)
from loom.target.arch.amdgpu.matrix_fragment_layout import (  # noqa: E402
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentRoleLayout,
    layout_roles,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_adaptation import (  # noqa: E402
    matrix_fragment_native_contraction_facts,
    matrix_fragment_native_transition_facts,
    matrix_fragment_packed_element_axis,
    matrix_fragment_role_storage_projection_plan,
)
from loom.target.arch.amdgpu.matrix_fragment_layout_recipes import (  # noqa: E402
    MatrixFragmentPackedB16PublicationProjection,
    MatrixFragmentResultToLhsBf16Projection,
    MatrixFragmentResultToRhsPackedB16Projection,
    matrix_fragment_packed_b16_publication_projection,
    matrix_fragment_result_to_lhs_bf16_projection,
    matrix_fragment_result_to_rhs_packed_b16_projection,
)
from loom.target.arch.amdgpu.matrix_fragment_layouts import (  # noqa: E402
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS,
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY,
)
from loom.target.arch.amdgpu.matrix_fragment_realization import (  # noqa: E402
    AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG,
    MATRIX_CONTRACT_ORDINAL_NONE,
    MatrixContractRealizationChoices,
    MatrixFragmentRealizationCatalog,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_MATRIX_FEATURE_PROFILES,
    AMDGPU_MATRIX_FEATURES_BY_PROFILE,
    amdgpu_target_descriptor_set_key,
    sorted_processor_infos,
    sorted_target_infos,
)
from loom.target.low_descriptors import (  # noqa: E402
    Descriptor,
    DescriptorSet,
    Immediate,
    ImmediateFlag,
    Operand,
    target_relative_name,
)
from loom.target.native_coordinate_projection import (  # noqa: E402
    CoordinateProjectionPlan,
    CoordinateProjectionTerm,
)

_FAMILY_C_NAMES = {
    "mfma": "LOOM_AMDGPU_MATRIX_FAMILY_MFMA",
    "smfmac": "LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC",
    "wmma": "LOOM_AMDGPU_MATRIX_FAMILY_WMMA",
    "swmmac": "LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC",
}

_FEATURE_C_NAMES = {
    "mfma_gfx908": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908",
    "mfma_gfx908_gfx90a": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A",
    "mfma_gfx90a_bf16_1k": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K",
    "mfma_gfx90a_f64": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64",
    "mfma_gfx940_fp8": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8",
    "mfma_gfx940_i8": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_I8",
    "mfma_gfx940_xf32": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_XF32",
    "mfma_gfx950": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950",
    "mfma_gfx950_scale_f8f6f4": ("LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4"),
    "smfmac_gfx940": "LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940",
    "smfmac_gfx940_fp8": "LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940_FP8",
    "smfmac_gfx950": "LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950",
    "wmma_gfx11": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11",
    "wmma_gfx12": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12",
    "swmmac_gfx12": "LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12",
    "wmma_gfx1250": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250",
    "wmma_gfx1250_scale_f8f6f4": ("LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4"),
    "swmmac_gfx1250": "LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250",
}

_MATRIX_FEATURE_PROFILE_C_NAMES = {
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908"),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A"),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940"),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950"),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11"),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12"),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250"),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC"),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC: ("LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC"),
}

_WAVE_SIZE_C_NAMES = {
    "32": "LOOM_AMDGPU_MATRIX_WAVE_SIZE_32",
    "64": "LOOM_AMDGPU_MATRIX_WAVE_SIZE_64",
    "any": "LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY",
}

_NUMERIC_TYPE_C_NAMES = {
    "f64": "LOOM_AMDGPU_MATRIX_NUMERIC_F64",
    "f32": "LOOM_AMDGPU_MATRIX_NUMERIC_F32",
    "f16": "LOOM_AMDGPU_MATRIX_NUMERIC_F16",
    "bf16": "LOOM_AMDGPU_MATRIX_NUMERIC_BF16",
    "xf32": "LOOM_AMDGPU_MATRIX_NUMERIC_XF32",
    "i32": "LOOM_AMDGPU_MATRIX_NUMERIC_I32",
    "i8": "LOOM_AMDGPU_MATRIX_NUMERIC_I8",
    "iu8": "LOOM_AMDGPU_MATRIX_NUMERIC_IU8",
    "i4": "LOOM_AMDGPU_MATRIX_NUMERIC_I4",
    "iu4": "LOOM_AMDGPU_MATRIX_NUMERIC_IU4",
    "fp8": "LOOM_AMDGPU_MATRIX_NUMERIC_FP8",
    "bf8": "LOOM_AMDGPU_MATRIX_NUMERIC_BF8",
    "fp6": "LOOM_AMDGPU_MATRIX_NUMERIC_FP6",
    "bf6": "LOOM_AMDGPU_MATRIX_NUMERIC_BF6",
    "fp4": "LOOM_AMDGPU_MATRIX_NUMERIC_FP4",
    "f8": "LOOM_AMDGPU_MATRIX_NUMERIC_F8",
    "f6": "LOOM_AMDGPU_MATRIX_NUMERIC_F6",
    "f8f6f4": "LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4",
}

_SCALE_KIND_C_NAMES = {
    "none": "LOOM_AMDGPU_MATRIX_SCALE_NONE",
    "scale32": "LOOM_AMDGPU_MATRIX_SCALE_32",
    "scale16": "LOOM_AMDGPU_MATRIX_SCALE_16",
}

_SCALE_FORMAT_C_NAMES = {
    "e8m0": "LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0",
    "fp8_e4m3": "LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3",
}

_FLAG_C_NAMES = {
    "sparse": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE",
    "scaled": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED",
    "matrix_formats": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS",
    "reuse": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE",
    "clamp": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP",
    "sign_select": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT",
    "ab_modifiers": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS",
    "c_modifier": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER",
    "opsel": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL",
    "zero_scale_fallback": ("LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK"),
    "scale_formats": "LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS",
}

_SOURCE_REQUIREMENT_C_NAMES = {
    "fragment_layout": ("LOOM_AMDGPU_MATRIX_CONTRACT_SOURCE_REQUIREMENT_FRAGMENT_LAYOUT"),
}

_MATRIX_ATTR_IMMEDIATE_FIELDS = frozenset(
    (
        "matrix_a_fmt",
        "matrix_b_fmt",
        "matrix_a_scale",
        "matrix_b_scale",
        "matrix_a_scale_fmt",
        "matrix_b_scale_fmt",
        "matrix_a_reuse",
        "matrix_b_reuse",
        "neg_lo",
        "neg_hi",
        "clamp",
    )
)

_MATRIX_FLAG_REQUIRED_IMMEDIATE_FIELDS = {
    "sign_select": frozenset(("neg_lo",)),
    "clamp": frozenset(("clamp",)),
}

_FRAGMENT_LAYOUT_C_NAMES = {
    None: "LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN",
    **{key: layout.c_kind for key, layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.items()},
}

_MATRIX_WAIT_RESULT_FAMILIES = frozenset(("mfma", "smfmac"))
_MATRIX_WAIT_RESULT_REGISTER_COUNTS = frozenset((2, 4, 8, 16, 32))


@dataclass(frozen=True, slots=True)
class _MatrixDescriptorShape:
    lhs_register_count: int
    rhs_register_count: int
    accumulator_register_count: int
    result_register_count: int
    has_sparse_metadata: bool
    has_scale_operands: bool


@dataclass(frozen=True, slots=True)
class _MatrixDescriptorCatalog:
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]]
    shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]]
    immediates_by_key: Mapping[str, tuple[Immediate, ...]]


@dataclass(frozen=True, slots=True)
class _MatrixSourceContractSignature:
    block_count: int
    tile_shape: tuple[int, int, int]
    lhs: AmdgpuMatrixPayload
    rhs: AmdgpuMatrixPayload
    accumulator: AmdgpuMatrixPayload
    result: AmdgpuMatrixPayload
    scale_kind: str
    flags: frozenset[str]
    implicit_scale_formats: frozenset[str]


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{_c_identifier(target_relative_name('amdgpu', key))}"


def _validate_known_values(
    values: Iterable[str],
    known_values: Mapping[str, str],
    *,
    field_name: str,
    contract: AmdgpuMatrixContract,
) -> None:
    for value in values:
        if value not in known_values:
            raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown {field_name} '{value}'")


def _c_bitset(
    values: Sequence[str],
    c_names: Mapping[str, str],
    *,
    field_name: str,
    contract: AmdgpuMatrixContract,
) -> str:
    _validate_known_values(values, c_names, field_name=field_name, contract=contract)
    if not values:
        return "0"
    return " | ".join(c_names[value] for value in values)


def _c_selector_bitset(
    values: Sequence[str],
    c_names: Mapping[str, str],
    *,
    field_name: str,
    contract: AmdgpuMatrixContract,
) -> str:
    _validate_known_values(values, c_names, field_name=field_name, contract=contract)
    if not values:
        return "0"
    return "(loom_amdgpu_matrix_scale_format_selector_bits_t)(" + " | ".join(f"(1u << {c_names[value]})" for value in values) + ")"


def _contract_semantic_tag(contract: AmdgpuMatrixContract) -> str:
    return contract.semantic_tag or f"matrix.{contract.name}"


def _contract_intrinsic_name(contract: AmdgpuMatrixContract) -> str:
    return contract.intrinsic_name or f"llvm.amdgcn.{contract.name}"


def _matrix_descriptor_catalog(
    *,
    descriptor_ref_keys: Iterable[str],
    descriptor_sets: Iterable[DescriptorSet],
    overlay_sets: Iterable[Iterable[AmdgpuDescriptorOverlay]],
    extra_descriptors: Iterable[Descriptor] = (),
) -> _MatrixDescriptorCatalog:
    descriptor_ref_key_set = frozenset(descriptor_ref_keys)
    keys_by_semantic_tag: dict[str, set[str]] = {}
    shapes_by_key: dict[str, set[_MatrixDescriptorShape]] = {}
    immediates_by_key: dict[str, list[Immediate]] = {}

    def add_descriptor(
        descriptor_key: str,
        semantic_tag: str | None,
        operands: Iterable[Operand],
        immediates: Iterable[Immediate],
    ) -> None:
        if semantic_tag is None or not semantic_tag.startswith("matrix.") or descriptor_key not in descriptor_ref_key_set:
            return
        keys_by_semantic_tag.setdefault(semantic_tag, set()).add(descriptor_key)
        shape = _matrix_descriptor_shape_from_operands(operands)
        if shape is not None:
            shapes_by_key.setdefault(descriptor_key, set()).add(shape)
        immediates_by_key.setdefault(descriptor_key, []).extend(immediates)

    for descriptor_set in descriptor_sets:
        for descriptor in descriptor_set.descriptors:
            add_descriptor(
                descriptor.key,
                descriptor.semantic_tag,
                descriptor.operands,
                descriptor.immediates,
            )

    for overlays in overlay_sets:
        for overlay in overlays:
            add_descriptor(
                overlay.descriptor_key,
                overlay.semantic_tag,
                (operand_overlay.descriptor_operand for operand_overlay in overlay.operands),
                overlay.immediates,
            )

    for descriptor in extra_descriptors:
        add_descriptor(
            descriptor.key,
            descriptor.semantic_tag,
            descriptor.operands,
            descriptor.immediates,
        )

    return _MatrixDescriptorCatalog(
        keys_by_semantic_tag={semantic_tag: tuple(sorted(descriptor_keys)) for semantic_tag, descriptor_keys in keys_by_semantic_tag.items()},
        shapes_by_key={descriptor_key: tuple(sorted(shapes, key=_matrix_descriptor_shape_sort_key)) for descriptor_key, shapes in shapes_by_key.items()},
        immediates_by_key={descriptor_key: tuple(immediates) for descriptor_key, immediates in immediates_by_key.items()},
    )


def _global_matrix_descriptor_catalog() -> _MatrixDescriptorCatalog:
    return _matrix_descriptor_catalog(
        descriptor_ref_keys=amdgpu_descriptor_ref_keys(),
        descriptor_sets=_amdgpu_core_descriptor_set_bases(),
        overlay_sets=(builder.overlay_rows() for builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.values()),
        extra_descriptors=(descriptor for builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.values() for descriptor in builder.extra_descriptors),
    )


def _matrix_descriptor_catalog_for_builder(
    generator_target: str,
    *,
    descriptor_ref_keys: Iterable[str],
) -> _MatrixDescriptorCatalog:
    builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[generator_target]
    return _matrix_descriptor_catalog(
        descriptor_ref_keys=descriptor_ref_keys,
        descriptor_sets=(builder.base,),
        overlay_sets=(builder.overlay_rows(),),
        extra_descriptors=builder.extra_descriptors,
    )


def _matrix_descriptor_keys_by_semantic_tag() -> dict[str, tuple[str, ...]]:
    return dict(_global_matrix_descriptor_catalog().keys_by_semantic_tag)


def _matrix_descriptor_shape_from_operands(
    operands: Iterable[Operand],
) -> _MatrixDescriptorShape | None:
    operand_units = {operand.field_name: operand.unit_count for operand in operands}
    if any(field_name not in operand_units for field_name in ("lhs", "rhs", "acc", "dst")):
        return None
    has_scale_operands = "lhs_scale" in operand_units or "rhs_scale" in operand_units
    if has_scale_operands and ("lhs_scale" not in operand_units or "rhs_scale" not in operand_units):
        raise ValueError("AMDGPU matrix descriptor has incomplete lhs_scale/rhs_scale operands")
    return _MatrixDescriptorShape(
        lhs_register_count=operand_units["lhs"],
        rhs_register_count=operand_units["rhs"],
        accumulator_register_count=operand_units["acc"],
        result_register_count=operand_units["dst"],
        has_sparse_metadata="sparse_metadata" in operand_units,
        has_scale_operands=has_scale_operands,
    )


def _matrix_descriptor_shapes_by_key() -> dict[str, tuple[_MatrixDescriptorShape, ...]]:
    return dict(_global_matrix_descriptor_catalog().shapes_by_key)


def _matrix_descriptor_immediates_by_key() -> dict[str, tuple[Immediate, ...]]:
    return dict(_global_matrix_descriptor_catalog().immediates_by_key)


def _matrix_descriptor_shape_sort_key(
    shape: _MatrixDescriptorShape,
) -> tuple[int, int, int, int, bool, bool]:
    return (
        shape.lhs_register_count,
        shape.rhs_register_count,
        shape.accumulator_register_count,
        shape.result_register_count,
        shape.has_sparse_metadata,
        shape.has_scale_operands,
    )


def _resolve_contract_descriptor_key(
    contract: AmdgpuMatrixContract,
    *,
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]],
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
) -> str | None:
    descriptor_keys = keys_by_semantic_tag.get(_contract_semantic_tag(contract), ())
    if not descriptor_keys:
        return None

    contract_shape = _contract_matrix_descriptor_shape(contract)
    matching_keys = tuple(descriptor_key for descriptor_key in descriptor_keys if contract_shape in descriptor_shapes_by_key.get(descriptor_key, ()))
    if len(matching_keys) == 1:
        return matching_keys[0]
    if not matching_keys:
        descriptor_key_list = ", ".join(descriptor_keys)
        raise ValueError(
            f"AMDGPU matrix contract '{contract.name}' semantic tag "
            f"'{_contract_semantic_tag(contract)}' matches descriptor key(s) "
            f"{descriptor_key_list}, but none have payload shape "
            f"{_format_matrix_descriptor_shape(contract_shape)}"
        )
    descriptor_key_list = ", ".join(matching_keys)
    raise ValueError(f"AMDGPU matrix contract '{contract.name}' semantic tag '{_contract_semantic_tag(contract)}' ambiguously matches descriptor key(s) {descriptor_key_list}")


def _contract_matrix_descriptor_shape(
    contract: AmdgpuMatrixContract,
) -> _MatrixDescriptorShape:
    has_scale_operands = contract.scale_kind != "none" or "scaled" in contract.flags
    if has_scale_operands != ("scaled" in contract.flags):
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has inconsistent scaled flag and scale kind")
    return _MatrixDescriptorShape(
        lhs_register_count=contract.lhs.register_count,
        rhs_register_count=contract.rhs.register_count,
        accumulator_register_count=contract.accumulator.register_count,
        result_register_count=contract.result.register_count,
        has_sparse_metadata="sparse" in contract.flags,
        has_scale_operands=has_scale_operands,
    )


def _contract_has_abstract_matrix_format_payload(
    contract: AmdgpuMatrixContract,
) -> bool:
    return "matrix_formats" in contract.flags and contract.lhs.register_count == 0 and contract.lhs.element_count == 0 and contract.rhs.register_count == 0 and contract.rhs.element_count == 0


def _format_matrix_descriptor_shape(shape: _MatrixDescriptorShape) -> str:
    sparse_suffix = ", sparse" if shape.has_sparse_metadata else ""
    scale_suffix = ", scaled" if shape.has_scale_operands else ""
    return f"lhs={shape.lhs_register_count}, rhs={shape.rhs_register_count}, acc={shape.accumulator_register_count}, dst={shape.result_register_count}{sparse_suffix}{scale_suffix}"


def _validate_contract_descriptor_shape(
    contract: AmdgpuMatrixContract,
    descriptor_key: str,
    *,
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
) -> None:
    descriptor_shapes = descriptor_shapes_by_key.get(descriptor_key)
    if descriptor_shapes is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' references low descriptor '{descriptor_key}' without matrix operand shape metadata")
    contract_shape = _contract_matrix_descriptor_shape(contract)
    if contract_shape in descriptor_shapes:
        return
    descriptor_shape_list = ", ".join(_format_matrix_descriptor_shape(shape) for shape in descriptor_shapes)
    raise ValueError(
        f"AMDGPU matrix contract '{contract.name}' payload shape "
        f"{_format_matrix_descriptor_shape(contract_shape)} does not match "
        f"low descriptor '{descriptor_key}' operand shape(s): "
        f"{descriptor_shape_list}"
    )


def _validate_contract_descriptor_immediates(
    contract: AmdgpuMatrixContract,
    descriptor_key: str,
    *,
    descriptor_immediates_by_key: Mapping[str, tuple[Immediate, ...]],
) -> None:
    immediates = descriptor_immediates_by_key.get(descriptor_key)
    if immediates is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' references low descriptor '{descriptor_key}' without matrix immediate metadata")
    immediate_fields = frozenset(immediate.field_name for immediate in immediates)
    for flag, required_fields in _MATRIX_FLAG_REQUIRED_IMMEDIATE_FIELDS.items():
        if flag not in contract.flags:
            continue
        missing_fields = required_fields - immediate_fields
        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise ValueError(f"AMDGPU matrix contract '{contract.name}' selects low descriptor '{descriptor_key}' without required immediate field(s): {missing_text}")
    for immediate in immediates:
        if immediate.field_name in _MATRIX_ATTR_IMMEDIATE_FIELDS:
            continue
        if ImmediateFlag.DEFAULT_VALUE in immediate.flags:
            continue
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' selects low descriptor '{descriptor_key}' with unmapped immediate '{immediate.field_name}'")


def _validate_contract_wait_state_payload(contract: AmdgpuMatrixContract) -> None:
    if contract.family not in _MATRIX_WAIT_RESULT_FAMILIES:
        return
    if contract.result.register_count in _MATRIX_WAIT_RESULT_REGISTER_COUNTS:
        return
    expected_counts = ", ".join(str(count) for count in sorted(_MATRIX_WAIT_RESULT_REGISTER_COUNTS))
    raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unsupported wait-state result payload register count {contract.result.register_count}; expected one of {expected_counts}")


def _contract_descriptor_key(
    contract: AmdgpuMatrixContract,
    *,
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]],
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
    descriptor_immediates_by_key: Mapping[str, tuple[Immediate, ...]],
) -> str | None:
    descriptor_key = _resolve_contract_descriptor_key(
        contract,
        keys_by_semantic_tag=keys_by_semantic_tag,
        descriptor_shapes_by_key=descriptor_shapes_by_key,
    )
    if descriptor_key is None:
        return None
    _validate_contract_descriptor_shape(
        contract,
        descriptor_key,
        descriptor_shapes_by_key=descriptor_shapes_by_key,
    )
    _validate_contract_descriptor_immediates(
        contract,
        descriptor_key,
        descriptor_immediates_by_key=descriptor_immediates_by_key,
    )
    return descriptor_key


def _contract_descriptor_keys(
    *,
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]],
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
    descriptor_immediates_by_key: Mapping[str, tuple[Immediate, ...]],
) -> tuple[str | None, ...]:
    descriptor_keys: list[str | None] = []
    seen_keys: dict[str, AmdgpuMatrixContract] = {}
    for contract in AMDGPU_MATRIX_CONTRACTS:
        descriptor_key = _contract_descriptor_key(
            contract,
            keys_by_semantic_tag=keys_by_semantic_tag,
            descriptor_shapes_by_key=descriptor_shapes_by_key,
            descriptor_immediates_by_key=descriptor_immediates_by_key,
        )
        descriptor_keys.append(descriptor_key)
        if descriptor_key is None:
            continue
        signature = _contract_wait_state_signature(contract)
        if signature is None:
            continue
        previous_contract = seen_keys.get(descriptor_key)
        if previous_contract is not None:
            previous_signature = _contract_wait_state_signature(previous_contract)
            if previous_signature != signature:
                raise ValueError(f"AMDGPU matrix contracts '{previous_contract.name}' and '{contract.name}' both map to low descriptor '{descriptor_key}' but have different wait-state signatures")
            continue
        seen_keys[descriptor_key] = contract
    return tuple(descriptor_keys)


def _validate_matrix_feature_profiles() -> None:
    if tuple(AMDGPU_MATRIX_FEATURES_BY_PROFILE) != AMDGPU_MATRIX_FEATURE_PROFILES:
        raise ValueError("AMDGPU matrix feature profile rows must exactly match the ordered profile inventory")
    if tuple(_MATRIX_FEATURE_PROFILE_C_NAMES) != AMDGPU_MATRIX_FEATURE_PROFILES:
        raise ValueError("AMDGPU matrix feature profile C names must exactly match the ordered profile inventory")

    assigned_features: set[str] = set()
    for profile in AMDGPU_MATRIX_FEATURE_PROFILES:
        features = AMDGPU_MATRIX_FEATURES_BY_PROFILE[profile]
        if len(features) != len(set(features)):
            raise ValueError(f"AMDGPU matrix feature profile '{profile}' contains duplicate feature bits")
        if profile == AMDGPU_MATRIX_FEATURE_PROFILE_NONE:
            if features:
                raise ValueError("AMDGPU matrix feature profile 'none' must not enable feature bits")
        elif not features:
            raise ValueError(f"AMDGPU matrix feature profile '{profile}' has no feature bits")
        unknown_features = set(features) - set(_FEATURE_C_NAMES)
        if unknown_features:
            unknown_text = ", ".join(sorted(unknown_features))
            raise ValueError(f"AMDGPU matrix feature profile '{profile}' has unknown feature bit(s): {unknown_text}")
        assigned_features.update(features)

    unassigned_features = set(_FEATURE_C_NAMES) - assigned_features
    if unassigned_features:
        unassigned_text = ", ".join(sorted(unassigned_features))
        raise ValueError(f"AMDGPU matrix feature bit(s) have no profile: {unassigned_text}")


def _matrix_source_contract_signature(
    contract: AmdgpuMatrixContract,
) -> _MatrixSourceContractSignature:
    return _MatrixSourceContractSignature(
        block_count=contract.block_count,
        tile_shape=contract.tile_shape,
        lhs=contract.lhs,
        rhs=contract.rhs,
        accumulator=contract.accumulator,
        result=contract.result,
        scale_kind=contract.scale_kind,
        flags=frozenset(contract.flags),
        implicit_scale_formats=frozenset(contract.implicit_scale_formats),
    )


def _matrix_contract_wave_sizes(contract: AmdgpuMatrixContract) -> frozenset[int]:
    if contract.wave_size == "any":
        return frozenset((32, 64))
    if contract.wave_size in ("32", "64"):
        return frozenset((int(contract.wave_size),))
    raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown wave size '{contract.wave_size}'")


def _validate_matrix_source_contracts(
    contracts: Sequence[AmdgpuMatrixContract] = AMDGPU_MATRIX_CONTRACTS,
) -> None:
    for profile, profile_features in AMDGPU_MATRIX_FEATURES_BY_PROFILE.items():
        feature_set = frozenset(profile_features)
        contracts_by_signature: dict[_MatrixSourceContractSignature, list[AmdgpuMatrixContract]] = {}
        for contract in contracts:
            if not feature_set.issuperset(contract.features):
                continue
            signature = _matrix_source_contract_signature(contract)
            contracts_by_signature.setdefault(signature, []).append(contract)
        for signature_contracts in contracts_by_signature.values():
            for index, contract in enumerate(signature_contracts):
                contract_wave_sizes = _matrix_contract_wave_sizes(contract)
                for other_contract in signature_contracts[index + 1 :]:
                    overlapping_wave_sizes = contract_wave_sizes.intersection(_matrix_contract_wave_sizes(other_contract))
                    if not overlapping_wave_sizes:
                        continue
                    wave_sizes = ", ".join(f"wave{wave_size}" for wave_size in sorted(overlapping_wave_sizes))
                    raise ValueError(f"AMDGPU matrix feature profile '{profile}' exposes source-indistinguishable contracts '{contract.name}' and '{other_contract.name}' for {wave_sizes}")


def _validate_matrix_profile_descriptor_catalog(
    *,
    descriptor_set_key: str,
    profile: str,
    catalog: _MatrixDescriptorCatalog,
    global_descriptor_keys: Sequence[str | None],
) -> None:
    feature_set = frozenset(AMDGPU_MATRIX_FEATURES_BY_PROFILE[profile])
    for contract, global_descriptor_key in zip(AMDGPU_MATRIX_CONTRACTS, global_descriptor_keys, strict=True):
        if not feature_set.issuperset(contract.features):
            continue
        if global_descriptor_key is None:
            if _contract_has_abstract_matrix_format_payload(contract):
                continue
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set_key}' matrix profile '{profile}' exposes concrete contract '{contract.name}' without a global low descriptor")
        try:
            descriptor_key = _contract_descriptor_key(
                contract,
                keys_by_semantic_tag=catalog.keys_by_semantic_tag,
                descriptor_shapes_by_key=catalog.shapes_by_key,
                descriptor_immediates_by_key=catalog.immediates_by_key,
            )
        except ValueError as exc:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set_key}' matrix profile '{profile}' exposes contract '{contract.name}' with an incompatible descriptor ABI: {exc}") from exc
        if descriptor_key is None:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set_key}' matrix profile '{profile}' exposes contract '{contract.name}' but has no matching low descriptor")
        if descriptor_key != global_descriptor_key:
            raise ValueError(
                f"AMDGPU descriptor set '{descriptor_set_key}' matrix profile "
                f"'{profile}' resolves contract '{contract.name}' to "
                f"'{descriptor_key}', expected stable descriptor "
                f"'{global_descriptor_key}'"
            )


def _validate_matrix_profile_descriptor_sets(*, global_descriptor_keys: Sequence[str | None]) -> None:
    _validate_matrix_feature_profiles()
    _validate_matrix_source_contracts()
    descriptor_ref_keys = frozenset(descriptor_key for descriptor_key in global_descriptor_keys if descriptor_key is not None)
    builders_by_descriptor_set_key = {builder.base.key: generator_target for generator_target, builder in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS.items()}
    profiles_by_descriptor_set_key: dict[str, set[str]] = {}
    processors_by_name = {processor.processor: processor for processor in sorted_processor_infos()}
    for target_info in sorted_target_infos():
        processor_info = processors_by_name[target_info.processor]
        profile = processor_info.features.matrix
        if profile not in AMDGPU_MATRIX_FEATURES_BY_PROFILE:
            raise ValueError(f"AMDGPU processor '{processor_info.processor}' has unknown matrix feature profile '{profile}'")
        descriptor_set_key = amdgpu_target_descriptor_set_key(target_info, processor_info)
        if not descriptor_set_key:
            continue
        if descriptor_set_key not in builders_by_descriptor_set_key:
            raise ValueError(f"AMDGPU target '{target_info.target}' references descriptor set '{descriptor_set_key}' without a builder")
        profiles_by_descriptor_set_key.setdefault(descriptor_set_key, set()).add(profile)

    for descriptor_set_key in sorted(profiles_by_descriptor_set_key):
        generator_target = builders_by_descriptor_set_key[descriptor_set_key]
        catalog = _matrix_descriptor_catalog_for_builder(
            generator_target,
            descriptor_ref_keys=descriptor_ref_keys,
        )
        for profile in sorted(profiles_by_descriptor_set_key[descriptor_set_key]):
            _validate_matrix_profile_descriptor_catalog(
                descriptor_set_key=descriptor_set_key,
                profile=profile,
                catalog=catalog,
                global_descriptor_keys=global_descriptor_keys,
            )


def _contract_wait_state_signature(
    contract: AmdgpuMatrixContract,
) -> tuple[str, int] | None:
    if contract.family not in _MATRIX_WAIT_RESULT_FAMILIES:
        return None
    return (contract.family, contract.result.register_count)


def _contract_wait_state_ordinals_by_descriptor_ref(
    descriptor_keys: Sequence[str | None],
) -> list[int | None]:
    contract_ordinals_by_descriptor_key: dict[str, int] = {}
    for ordinal, (contract, descriptor_key) in enumerate(zip(AMDGPU_MATRIX_CONTRACTS, descriptor_keys, strict=True)):
        if descriptor_key is None:
            continue
        if _contract_wait_state_signature(contract) is None:
            continue
        contract_ordinals_by_descriptor_key.setdefault(descriptor_key, ordinal)
    return [contract_ordinals_by_descriptor_key.get(descriptor_key) for descriptor_key in amdgpu_descriptor_ref_keys()]


def _matrix_feature_profile_table_lines() -> list[str]:
    _validate_matrix_feature_profiles()
    lines = [
        "const loom_amdgpu_matrix_feature_bits_t",
        "    kLoomAmdgpuMatrixFeatureBitsByProfile[",
        "        LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_COUNT] = {",
    ]
    for profile in AMDGPU_MATRIX_FEATURE_PROFILES:
        profile_c_name = _MATRIX_FEATURE_PROFILE_C_NAMES[profile]
        feature_c_names = tuple(_FEATURE_C_NAMES[feature] for feature in AMDGPU_MATRIX_FEATURES_BY_PROFILE[profile])
        if not feature_c_names:
            lines.append(f"    [{profile_c_name}] = 0,")
            continue
        lines.append(f"    [{profile_c_name}] =")
        for index, feature_c_name in enumerate(feature_c_names):
            suffix = "," if index == len(feature_c_names) - 1 else " |"
            lines.append(f"        {feature_c_name}{suffix}")
    lines.extend(("};", ""))
    return lines


def _payload_initializer(payload: AmdgpuMatrixPayload) -> str:
    numeric_type = _NUMERIC_TYPE_C_NAMES.get(payload.numeric_type)
    if numeric_type is None:
        raise ValueError(f"unknown AMDGPU matrix numeric type '{payload.numeric_type}'")
    return "\n".join(
        [
            "{",
            f"    .numeric_type = {numeric_type},",
            f"    .register_count = {payload.register_count},",
            f"    .element_count = {payload.element_count},",
            "}",
        ]
    )


def _validate_contract_fragment_layout(contract: AmdgpuMatrixContract) -> None:
    if contract.fragment_layout is None:
        if "sparse" in contract.flags:
            raise ValueError(f"sparse AMDGPU matrix contract '{contract.name}' has no fragment layout")
        return
    layout = AMDGPU_MATRIX_FRAGMENT_LAYOUTS_BY_KEY.get(contract.fragment_layout)
    if layout is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown fragment layout '{contract.fragment_layout}'")
    block_count, *layout_tile_shape = layout.tile_shape
    if (block_count, *layout_tile_shape) != (
        contract.block_count,
        *contract.tile_shape,
    ):
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' shape {contract.tile_shape} disagrees with fragment layout '{layout.key}' shape {layout.tile_shape}")
    if contract.wave_size != "any" and int(contract.wave_size) != layout.wave_size:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' wave size {contract.wave_size} disagrees with fragment layout '{layout.key}' wave size {layout.wave_size}")
    compressed_roles = tuple(role.role for role in layout_roles(layout) if role.reduction_group is not None)
    if "sparse" in contract.flags:
        reduction_group = layout.lhs.reduction_group
        if compressed_roles != ("lhs",) or reduction_group is None or reduction_group.storage_element_count != 2 or reduction_group.logical_element_count != 4:
            raise ValueError(f"sparse AMDGPU matrix contract '{contract.name}' uses fragment layout '{layout.key}' without an exact 2:4 LHS")
    elif compressed_roles:
        raise ValueError(f"dense AMDGPU matrix contract '{contract.name}' uses compressed fragment layout '{layout.key}'")
    contract_payloads = (
        contract.lhs,
        contract.rhs,
        contract.accumulator,
        contract.result,
    )
    for contract_payload, role_layout in zip(contract_payloads, layout_roles(layout), strict=True):
        element_bit_count = AMDGPU_MATRIX_NUMERIC_TYPE_BIT_COUNTS.get(contract_payload.numeric_type)
        if element_bit_count is None:
            raise ValueError(f"AMDGPU matrix contract '{contract.name}' role '{role_layout.role}' uses fragment layout '{layout.key}' with unsupported numeric type '{contract_payload.numeric_type}'")
        actual_payload = (
            role_layout.register_count,
            role_layout.payload_element_count,
            role_layout.element_bit_count,
        )
        expected_payload = (
            contract_payload.register_count,
            contract_payload.element_count,
            element_bit_count,
        )
        if actual_payload != expected_payload:
            raise ValueError(f"AMDGPU matrix contract '{contract.name}' role '{role_layout.role}' payload {expected_payload} disagrees with fragment layout '{layout.key}' payload {actual_payload}")


def _validate_contract_tile_shape(contract: AmdgpuMatrixContract) -> None:
    tile_shape = (contract.block_count, *contract.tile_shape)
    if any(count <= 0 or count > 0xFFFF for count in tile_shape):
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has invalid tile shape {tile_shape}")


def _contract_initializer(
    contract: AmdgpuMatrixContract,
    realization: MatrixContractRealizationChoices,
    *,
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]],
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
    descriptor_immediates_by_key: Mapping[str, tuple[Immediate, ...]],
) -> str:
    _validate_known_values(
        contract.features,
        _FEATURE_C_NAMES,
        field_name="feature",
        contract=contract,
    )
    _validate_known_values(contract.flags, _FLAG_C_NAMES, field_name="flag", contract=contract)
    _validate_known_values(
        contract.source_requirements,
        _SOURCE_REQUIREMENT_C_NAMES,
        field_name="source requirement",
        contract=contract,
    )
    _validate_known_values(
        contract.implicit_scale_formats,
        _SCALE_FORMAT_C_NAMES,
        field_name="implicit scale format",
        contract=contract,
    )
    if contract.implicit_scale_formats and "scale_formats" in contract.flags:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' cannot have both scale-format selector operands and implicit scale formats")
    if contract.implicit_scale_formats and contract.scale_kind == "none":
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' cannot have implicit scale formats without scale operands")
    if contract.scale_kind != "none" and "scale_formats" not in contract.flags and not contract.implicit_scale_formats:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' with scale operands must have selector operands or implicit scale formats")
    _validate_contract_tile_shape(contract)
    _validate_contract_wait_state_payload(contract)
    _validate_contract_fragment_layout(contract)
    descriptor_key = _contract_descriptor_key(
        contract,
        keys_by_semantic_tag=keys_by_semantic_tag,
        descriptor_shapes_by_key=descriptor_shapes_by_key,
        descriptor_immediates_by_key=descriptor_immediates_by_key,
    )
    low_descriptor_ref = "LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE" if descriptor_key is None else _descriptor_ref_constant_name(descriptor_key)
    family = _FAMILY_C_NAMES.get(contract.family)
    if family is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown family '{contract.family}'")
    wave_size = _WAVE_SIZE_C_NAMES.get(contract.wave_size)
    if wave_size is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown wave size '{contract.wave_size}'")
    scale_kind = _SCALE_KIND_C_NAMES.get(contract.scale_kind)
    if scale_kind is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown scale kind '{contract.scale_kind}'")
    fragment_layout = _FRAGMENT_LAYOUT_C_NAMES.get(contract.fragment_layout)
    if fragment_layout is None:
        raise ValueError(f"AMDGPU matrix contract '{contract.name}' has unknown fragment layout '{contract.fragment_layout}'")
    implicit_scale_format_selector_bits = _c_selector_bitset(
        contract.implicit_scale_formats,
        _SCALE_FORMAT_C_NAMES,
        field_name="implicit scale format",
        contract=contract,
    )
    alternative_contract_value = (
        "LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE" if realization.operand_exchanged_contract_ordinal == MATRIX_CONTRACT_ORDINAL_NONE else f"UINT16_C({realization.operand_exchanged_contract_ordinal})"
    )
    result_row_count, result_column_count, reduction_count = contract.tile_shape
    return "\n".join(
        [
            "{",
            f'    .name = IREE_SVL("{contract.name}"),',
            f"    .low_descriptor_ref = {low_descriptor_ref},",
            "    .realization = {",
            f"        .operand_exchanged_contract_ordinal = {alternative_contract_value},",
            f"        .canonical_result_representation_id = UINT8_C({realization.canonical_result_representation_id}),",
            f"        .operand_exchanged_result_representation_id = UINT8_C({realization.operand_exchanged_result_representation_id}),",
            "    },",
            f'    .llvm_intrinsic_name = IREE_SVL("{_contract_intrinsic_name(contract)}"),',
            f"    .family = {family},",
            f"    .required_feature_bits = {_c_bitset(contract.features, _FEATURE_C_NAMES, field_name='feature', contract=contract)},",
            f"    .wave_size_bits = {wave_size},",
            f"    .flags = {_c_bitset(contract.flags, _FLAG_C_NAMES, field_name='flag', contract=contract)},",
            f"    .source_requirement_flags = {_c_bitset(contract.source_requirements, _SOURCE_REQUIREMENT_C_NAMES, field_name='source requirement', contract=contract)},",
            "    .tile_shape = {",
            f"        .block_count = {contract.block_count},",
            f"        .result_row_count = {result_row_count},",
            f"        .result_column_count = {result_column_count},",
            f"        .reduction_count = {reduction_count},",
            "    },",
            f"    .lhs_payload = {_payload_initializer(contract.lhs)},",
            f"    .rhs_payload = {_payload_initializer(contract.rhs)},",
            f"    .accumulator_payload = {_payload_initializer(contract.accumulator)},",
            f"    .result_payload = {_payload_initializer(contract.result)},",
            f"    .scale_kind = {scale_kind},",
            f"    .implicit_scale_format_selector_bits = {implicit_scale_format_selector_bits},",
            f"    .fragment_layout_kind = {fragment_layout},",
            "},",
        ]
    )


def _matrix_result_representation_table_lines(
    catalog: MatrixFragmentRealizationCatalog,
) -> list[str]:
    lines = [
        "const loom_amdgpu_matrix_result_representation_t",
        "    kLoomAmdgpuMatrixResultRepresentations[",
        "        LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_COUNT] = {",
        "    [LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE] = {",
        "        .fragment_layout_kind = LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN,",
        "        .numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN,",
        "        .flags = 0,",
        "    },",
    ]
    for representation_id, representation in enumerate(catalog.result_representations, start=1):
        flags = "LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_FLAG_TRANSPOSE_MN" if representation.transposed else "0"
        lines.extend(
            [
                f"    [UINT8_C({representation_id})] = {{",
                f"        .fragment_layout_kind = {representation.fragment_layout.c_kind},",
                f"        .numeric_type = {_NUMERIC_TYPE_C_NAMES[representation.payload.numeric_type]},",
                f"        .flags = {flags},",
                "    },",
            ]
        )
    lines.append("};")
    return lines


def _emit_header() -> str:
    result_representation_count = len(AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG.result_representations) + 1
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.matrix.amdgpu_matrix_contract_tables"),
        "",
        "// AMDGPU matrix contract descriptor tables.",
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_TABLES_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_TABLES_H_",
        "",
        '#include "loom/target/arch/amdgpu/matrix/types.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "extern const loom_amdgpu_matrix_contract_descriptor_t",
        "    kLoomAmdgpuMatrixContractDescriptors[];",
        "extern const iree_host_size_t kLoomAmdgpuMatrixContractDescriptorCount;",
        f"#define LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_COUNT {result_representation_count}",
        "extern const loom_amdgpu_matrix_result_representation_t",
        "    kLoomAmdgpuMatrixResultRepresentations[",
        "        LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_COUNT];",
        "extern const loom_amdgpu_matrix_feature_bits_t",
        "    kLoomAmdgpuMatrixFeatureBitsByProfile[",
        "        LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_COUNT];",
        "extern const uint16_t",
        "    kLoomAmdgpuMatrixWaitStateContractOrdinalsByDescriptorRef[];",
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        "#endif  // LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_TABLES_H_",
    ]
    return "\n".join(lines) + "\n"


def _deduplicate_repack_projections[ProjectionT: Hashable](
    project: Callable[[AmdgpuMatrixFragmentLayout], ProjectionT | None],
    source_role: str,
    destination_role: str,
    native_transitions: NativeTransitionFactTable,
) -> tuple[tuple[int, ...], tuple[tuple[ProjectionT, str], ...]]:
    rows: list[tuple[ProjectionT, str]] = []
    row_ordinals: dict[tuple[ProjectionT, str], int] = {}
    layout_ordinals: list[int] = []
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        projection = project(layout)
        if projection is None:
            layout_ordinals.append(0)
            continue
        transition = matrix_fragment_native_transition_facts(layout, source_role, destination_role)
        if transition is None:
            raise ValueError(f"AMDGPU matrix fragment layout '{layout.key}' has a repack projection without exact native transition facts")
        row = (projection, native_transitions.reference(transition))
        ordinal = row_ordinals.get(row)
        if ordinal is None:
            rows.append(row)
            ordinal = len(rows)
            if ordinal > 0xFF:
                raise ValueError("fragment repack projection ordinal exceeds uint8_t")
            row_ordinals[row] = ordinal
        layout_ordinals.append(ordinal)
    return tuple(layout_ordinals), tuple(rows)


def _fragment_repack_ordinal_table_lines(name: str, layout_ordinals: tuple[int, ...]) -> list[str]:
    lines = [
        f"static const uint8_t {name}[",
        "    LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {",
    ]
    for layout, ordinal in zip(AMDGPU_MATRIX_FRAGMENT_LAYOUTS, layout_ordinals, strict=True):
        if ordinal:
            lines.append(f"    [{layout.c_kind}] = UINT8_C({ordinal}),")
    lines.extend(["};", ""])
    return lines


def _fragment_repack_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.matrix.amdgpu_matrix_contract_tables"),
        ),
        "",
    ]


def _emit_result_to_lhs_repack_projections() -> str:
    native_transitions = NativeTransitionFactTable("kResultToLhsBf16NativeTransitions")
    layout_ordinals, projection_rows = _deduplicate_repack_projections(
        matrix_fragment_result_to_lhs_bf16_projection,
        "result",
        "lhs",
        native_transitions,
    )
    lines = _fragment_repack_header()
    lines.extend(native_transitions.definition_lines())
    lines.extend(_fragment_repack_ordinal_table_lines("kResultToLhsBf16ProjectionOrdinals", layout_ordinals))
    lines.extend(
        [
            "static const loom_amdgpu_result_to_lhs_bf16_projection_t",
            "    kResultToLhsBf16Projections[] = {",
            "    [0] = {0},",
        ]
    )
    for ordinal, (projection, native_transition_reference) in enumerate(projection_rows, start=1):
        if not isinstance(projection, MatrixFragmentResultToLhsBf16Projection):
            raise TypeError("unexpected result-to-LHS projection type")
        lines.extend(
            [
                f"    [{ordinal}] = {{",
                f"        .source_register_selector_and_mask = UINT16_C(0x{projection.source_register_selector.and_mask:x}),",
                f"        .source_lane_group_and_mask = UINT16_C(0x{projection.source_lane_group.and_mask:x}),",
                f"        .source_lane_group_byte_shift = UINT8_C({projection.source_lane_group_byte_shift}),",
                f"        .result_lane_div_byte_shift = UINT8_C({projection.result_lane_div_byte_shift}),",
                f"        .source_register_selector_right_shift = UINT8_C({projection.source_register_selector.right_shift}),",
                f"        .source_lane_group_right_shift = UINT8_C({projection.source_lane_group.right_shift}),",
                f"        .transpose_bit_count = UINT8_C({projection.transpose_bit_count}),",
                f"        .native_transition_facts = {native_transition_reference},",
                "    },",
            ]
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _emit_result_to_rhs_repack_projections() -> str:
    native_transitions = NativeTransitionFactTable("kResultToRhsPackedB16NativeTransitions")
    layout_ordinals, projection_rows = _deduplicate_repack_projections(
        matrix_fragment_result_to_rhs_packed_b16_projection,
        "result",
        "rhs",
        native_transitions,
    )
    lines = _fragment_repack_header()
    lines.extend(native_transitions.definition_lines())
    lines.extend(_fragment_repack_ordinal_table_lines("kResultToRhsPackedB16ProjectionOrdinals", layout_ordinals))
    lines.extend(
        [
            "static const loom_amdgpu_result_to_rhs_packed_b16_projection_t",
            "    kResultToRhsPackedB16Projections[] = {",
            "    [0] = {0},",
        ]
    )
    for ordinal, (projection, native_transition_reference) in enumerate(projection_rows, start=1):
        if not isinstance(projection, MatrixFragmentResultToRhsPackedB16Projection):
            raise TypeError("unexpected result-to-RHS projection type")
        lines.extend(
            [
                f"    [{ordinal}] = {{",
                f"        .exchange_lane_mask = UINT16_C(0x{projection.exchange_participant_xor_mask:x}),",
                f"        .reverse_lane_mask = UINT32_C(0x{projection.reverse_participant_mask:x}),",
                f"        .native_transition_facts = {native_transition_reference},",
                "    },",
            ]
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


_PACKED_ELEMENT_AXIS_C_NAMES = {
    "block": "LOOM_MATRIX_FRAGMENT_AXIS_BLOCK",
    "row": "LOOM_MATRIX_FRAGMENT_AXIS_ROW",
    "column": "LOOM_MATRIX_FRAGMENT_AXIS_COLUMN",
    "reduction": "LOOM_MATRIX_FRAGMENT_AXIS_REDUCTION",
}

_COORDINATE_DIMENSION_C_NAMES = {
    "participant": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_PARTICIPANT",
    "value": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_VALUE",
    "block": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_BLOCK",
    "m": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW",
    "n": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN",
    "k": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION",
    "row": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_ROW",
    "column": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_COLUMN",
    "reduction": "LOOM_MATRIX_FRAGMENT_COORDINATE_DIMENSION_REDUCTION",
}


def _fragment_coordinate_plan_catalog() -> tuple[tuple[CoordinateProjectionPlan, ...], Mapping[tuple[str, str], int]]:
    plans: list[CoordinateProjectionPlan] = []
    plan_indices: dict[CoordinateProjectionPlan, int] = {}
    role_plan_indices: dict[tuple[str, str], int] = {}
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        for role in layout_roles(layout):
            plan = matrix_fragment_role_storage_projection_plan(layout, role)
            participant_destinations = tuple(term.destination_dimension for term in plan.forward_terms if term.source_dimension == "participant")
            if len(participant_destinations) != len(set(participant_destinations)):
                raise ValueError(f"AMDGPU matrix fragment layout '{layout.key}' role '{role.role}' requires multiple participant digits for one stored coordinate")
            plan_index = plan_indices.get(plan)
            if plan_index is None:
                plan_index = len(plans)
                plans.append(plan)
                plan_indices[plan] = plan_index
            role_plan_indices[(layout.key, role.role)] = plan_index
    return tuple(plans), role_plan_indices


def _fragment_coordinate_term_initializer(
    term: CoordinateProjectionTerm,
) -> list[str]:
    for field_name, value in (
        ("source_divisor", term.source_divisor),
        ("source_modulus", term.source_modulus),
        ("destination_multiplier", term.destination_multiplier),
    ):
        if value > 0xFFFF:
            raise ValueError(f"AMDGPU matrix fragment coordinate projection {field_name} {value} exceeds uint16_t")
    return [
        "{",
        f"    .source_dimension = {_COORDINATE_DIMENSION_C_NAMES[term.source_dimension]},",
        f"    .destination_dimension = {_COORDINATE_DIMENSION_C_NAMES[term.destination_dimension]},",
        f"    .source_divisor = {term.source_divisor},",
        f"    .source_modulus = {term.source_modulus},",
        f"    .destination_multiplier = {term.destination_multiplier},",
        "},",
    ]


def _fragment_coordinate_plan_table_lines(
    plans: tuple[CoordinateProjectionPlan, ...],
) -> list[str]:
    plan_terms = tuple(plan.forward_terms + plan.inverse_terms for plan in plans)
    lines = [
        "static const loom_matrix_fragment_coordinate_projection_term_t",
        "    kLoomAmdgpuMatrixFragmentCoordinateProjectionTerms[] = {",
    ]
    for terms in plan_terms:
        for term in terms:
            lines.extend(_fragment_coordinate_term_initializer(term))
    lines.extend(
        [
            "};",
            "",
            "static const loom_matrix_fragment_coordinate_projection_plan_t",
            "    kLoomAmdgpuMatrixFragmentCoordinateProjectionPlans[] = {",
        ]
    )
    term_offset = 0
    for plan, terms in zip(plans, plan_terms, strict=True):
        if len(plan.forward_terms) > 0xFF or len(plan.inverse_terms) > 0xFF:
            raise ValueError("AMDGPU matrix fragment coordinate projection term count exceeds uint8_t")
        lines.extend(
            [
                "{",
                "    .terms =",
                "        &kLoomAmdgpuMatrixFragmentCoordinateProjectionTerms[",
                f"            {term_offset}],",
                f"    .forward_term_count = {len(plan.forward_terms)},",
                f"    .inverse_term_count = {len(plan.inverse_terms)},",
                "},",
            ]
        )
        term_offset += len(terms)
    lines.extend(["};", ""])
    return lines


def _fragment_role_initializer(
    layout: AmdgpuMatrixFragmentLayout,
    role: MatrixFragmentRoleLayout,
    coordinate_plan_index: int,
) -> list[str]:
    packed_publications = tuple(
        (
            packed_axis,
            matrix_fragment_packed_b16_publication_projection(layout, role, packed_axis),
        )
        for packed_axis in ("row", "column")
    )
    packed_element_axis = matrix_fragment_packed_element_axis(layout, role)
    packed_element_axis_c_name = "LOOM_MATRIX_FRAGMENT_AXIS_COUNT" if packed_element_axis is None else _PACKED_ELEMENT_AXIS_C_NAMES[packed_element_axis]
    lines = [
        "{",
        f"    .register_count = {role.register_count},",
        f"    .element_bit_count = {role.element_bit_count},",
        f"    .payload_element_count = {role.payload_element_count},",
        f"    .coordinate_element_count = {role.coordinate_element_count},",
        "    .reserved = 0,",
        f"    .coordinate_element_stride = {role.coordinate_element_stride},",
    ]
    for packed_axis, packed_publication in packed_publications:
        if packed_publication is None:
            continue
        if not isinstance(packed_publication, MatrixFragmentPackedB16PublicationProjection):
            raise TypeError("unexpected packed-B16 publication projection type")
        lines.extend(
            [
                f"    .packed_b16_publications.{packed_axis} = {{",
                f"        .publishing_participant_and_mask = UINT16_C(0x{packed_publication.publishing_participant_and_mask:x}),",
                f"        .publishing_participant_equal_value = UINT16_C(0x{packed_publication.publishing_participant_equal_value:x}),",
                f"        .paired_participant_xor_mask = UINT16_C(0x{packed_publication.paired_participant_xor_mask:x}),",
                "    },",
            ]
        )
    lines.extend(
        [
            f"    .packed_element_axis = {packed_element_axis_c_name},",
        ]
    )
    if role.reduction_group is not None:
        lines.extend(
            [
                "    .reduction_group = {",
                f"        .storage_element_count = {role.reduction_group.storage_element_count},",
                f"        .logical_element_count = {role.reduction_group.logical_element_count},",
                "    },",
            ]
        )
    lines.extend(
        [
            "    .coordinate_projection_plan =",
            "        &kLoomAmdgpuMatrixFragmentCoordinateProjectionPlans[",
            f"            {coordinate_plan_index}],",
        ]
    )
    lines.append("},")
    return lines


def _fragment_layout_initializer(
    layout: AmdgpuMatrixFragmentLayout,
    role_plan_indices: Mapping[tuple[str, str], int],
    native_contractions: NativeContractionFactTable,
) -> list[str]:
    block_count, row_count, column_count, reduction_count = layout.tile_shape
    lines = [
        f"[{layout.c_kind}] = {{",
        f"    .kind = {layout.c_kind},",
        f'    .name = IREE_SVL("{layout.name}"),',
        f"    .wave_size = {layout.wave_size},",
        "    .tile_shape = {",
        f"        .block_count = {block_count},",
        f"        .result_row_count = {row_count},",
        f"        .result_column_count = {column_count},",
        f"        .reduction_count = {reduction_count},",
        "    },",
    ]
    for field_name, role in zip(
        ("lhs", "rhs", "accumulator", "result"),
        layout_roles(layout),
        strict=True,
    ):
        role_lines = _fragment_role_initializer(layout, role, role_plan_indices[(layout.key, role.role)])
        lines.append(f"    .{field_name} = {role_lines[0]}")
        lines.extend(f"    {line}" for line in role_lines[1:])
    lines.append(f"    .native_contraction_facts = {native_contractions.reference(matrix_fragment_native_contraction_facts(layout))},")
    lines.append("},")
    return lines


def _emit_source(*, public_header: str) -> str:
    realization_catalog = AMDGPU_MATRIX_FRAGMENT_REALIZATION_CATALOG
    catalog = _global_matrix_descriptor_catalog()
    keys_by_semantic_tag = catalog.keys_by_semantic_tag
    descriptor_shapes_by_key = catalog.shapes_by_key
    descriptor_immediates_by_key = catalog.immediates_by_key
    descriptor_keys = _contract_descriptor_keys(
        keys_by_semantic_tag=keys_by_semantic_tag,
        descriptor_shapes_by_key=descriptor_shapes_by_key,
        descriptor_immediates_by_key=descriptor_immediates_by_key,
    )
    _validate_matrix_profile_descriptor_sets(global_descriptor_keys=descriptor_keys)
    coordinate_plans, role_plan_indices = _fragment_coordinate_plan_catalog()
    native_contractions = NativeContractionFactTable("kLoomAmdgpuNativeContractionFacts")
    fragment_layout_lines: list[str] = []
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        fragment_layout_lines.extend(_fragment_layout_initializer(layout, role_plan_indices, native_contractions))
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.matrix.amdgpu_matrix_contract_tables"),
        "",
        f'#include "{public_header}"',
        "",
        *_matrix_feature_profile_table_lines(),
        *_fragment_coordinate_plan_table_lines(coordinate_plans),
        *native_contractions.definition_lines(),
        "const loom_amdgpu_matrix_fragment_layout_t",
        "    kLoomAmdgpuMatrixFragmentLayouts[",
        "        LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {",
        *fragment_layout_lines,
    ]
    lines.extend(
        [
            "};",
            "",
            "const loom_amdgpu_matrix_contract_descriptor_t",
            "    kLoomAmdgpuMatrixContractDescriptors[] = {",
        ]
    )
    lines.extend(
        _contract_initializer(
            contract,
            realization,
            keys_by_semantic_tag=keys_by_semantic_tag,
            descriptor_shapes_by_key=descriptor_shapes_by_key,
            descriptor_immediates_by_key=descriptor_immediates_by_key,
        )
        for contract, realization in zip(
            AMDGPU_MATRIX_CONTRACTS,
            realization_catalog.contract_choices,
            strict=True,
        )
    )
    lines.extend(
        [
            "};",
            "",
            "const iree_host_size_t kLoomAmdgpuMatrixContractDescriptorCount =",
            "    IREE_ARRAYSIZE(kLoomAmdgpuMatrixContractDescriptors);",
            "",
            "static_assert(sizeof(loom_amdgpu_matrix_result_representation_t) == 3,",
            '              "matrix result representation row must remain compact");',
            "static_assert(sizeof(loom_amdgpu_matrix_contract_realization_choices_t) == 4,",
            '              "matrix contract realization row must remain compact");',
            "static_assert(sizeof(loom_amdgpu_matrix_contract_descriptor_t) <= 120,",
            '              "matrix contract descriptor rows must remain compact");',
            "static_assert(LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT <= UINT8_MAX + 1u,",
            '              "matrix fragment layout kinds must fit uint8");',
            "static_assert(LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_COUNT <=",
            "                  LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_MAX_ID + 1u,",
            '              "matrix result representation IDs must fit availability bitsets");',
            "",
            *_matrix_result_representation_table_lines(realization_catalog),
            "",
            "static_assert(",
            "    sizeof(kLoomAmdgpuMatrixResultRepresentations) < 2048,",
            '    "matrix realization catalog must remain below 2 KiB");',
            "",
            "const uint16_t kLoomAmdgpuMatrixWaitStateContractOrdinalsByDescriptorRef[] = {",
        ]
    )
    for contract_ordinal in _contract_wait_state_ordinals_by_descriptor_ref(descriptor_keys):
        if contract_ordinal is None:
            lines.append("    UINT16_MAX,")
        else:
            lines.append(f"    UINT16_C({contract_ordinal}),")
    lines.append("};")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU matrix contract descriptor tables.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated matrix contract table header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated matrix contract table source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/matrix/contract_tables.h",
        help="Public include path for the generated header.",
    )
    parser.add_argument(
        "--result-to-lhs",
        required=True,
        type=Path,
        help="Generated result-to-LHS fragment repack projection table path.",
    )
    parser.add_argument(
        "--result-to-rhs",
        required=True,
        type=Path,
        help="Generated result-to-RHS fragment repack projection table path.",
    )
    args = parser.parse_args(argv)

    write_text_file(args.header, _emit_header())
    write_text_file(
        args.source,
        _emit_source(public_header=args.public_header),
    )
    write_text_file(
        args.result_to_lhs,
        _emit_result_to_lhs_repack_projections(),
    )
    write_text_file(
        args.result_to_rhs,
        _emit_result_to_rhs_repack_projections(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
