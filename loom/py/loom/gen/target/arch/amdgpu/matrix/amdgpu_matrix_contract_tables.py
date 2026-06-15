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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _amdgpu_core_descriptor_set_bases,
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
    _gfx1250_core_overlays,
    amdgpu_descriptor_ref_keys,
)
from loom.target.arch.amdgpu.matrix_contracts import (  # noqa: E402
    AMDGPU_MATRIX_CONTRACTS,
    AmdgpuMatrixContract,
    AmdgpuMatrixPayload,
)
from loom.target.low_descriptors import (  # noqa: E402
    Operand,
    target_relative_name,
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
    "mfma_gfx950": "LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950",
    "mfma_gfx950_scale_f8f6f4": ("LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4"),
    "smfmac_gfx940": "LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940",
    "smfmac_gfx950": "LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950",
    "wmma_gfx11": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11",
    "wmma_gfx12": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12",
    "swmmac_gfx12": "LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12",
    "wmma_gfx1250": "LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250",
    "wmma_gfx1250_scale_f8f6f4": ("LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4"),
    "swmmac_gfx1250": "LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250",
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
    "f8f6f4": "LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4",
}

_SCALE_KIND_C_NAMES = {
    "none": "LOOM_AMDGPU_MATRIX_SCALE_NONE",
    "scale32": "LOOM_AMDGPU_MATRIX_SCALE_32",
    "scale16": "LOOM_AMDGPU_MATRIX_SCALE_16",
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

_FRAGMENT_LAYOUT_C_NAMES = {
    None: "LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN",
    "rdna3_wmmar3_f32_16x16x16_f16": ("LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16"),
    "rdna3_wmmar3_f32_16x16x16_bf16": ("LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16"),
    "cdna_mfma_f32_16x16x16_f16": ("LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_F16"),
    "cdna_mfma_f32_16x16x16_bf16": ("LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16"),
    "cdna_mfma_f32_16x16x4_f32": ("LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32"),
}


@dataclass(frozen=True, slots=True)
class _MatrixDescriptorShape:
    lhs_register_count: int
    rhs_register_count: int
    accumulator_register_count: int
    result_register_count: int
    has_sparse_index: bool
    has_scale_operands: bool


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


def _contract_semantic_tag(contract: AmdgpuMatrixContract) -> str:
    return contract.semantic_tag or f"matrix.{contract.name}"


def _contract_intrinsic_name(contract: AmdgpuMatrixContract) -> str:
    return contract.intrinsic_name or f"llvm.amdgcn.{contract.name}"


def _matrix_descriptor_keys_by_semantic_tag() -> dict[str, tuple[str, ...]]:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    keys_by_semantic_tag: dict[str, set[str]] = {}

    def add_descriptor_key(descriptor_key: str, semantic_tag: str | None) -> None:
        if semantic_tag is None or not semantic_tag.startswith("matrix.") or descriptor_key not in descriptor_ref_key_set:
            return
        keys_by_semantic_tag.setdefault(semantic_tag, set()).add(descriptor_key)

    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        for descriptor in descriptor_set.descriptors:
            add_descriptor_key(descriptor.key, descriptor.semantic_tag)

    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        for overlay in overlays:
            add_descriptor_key(overlay.descriptor_key, overlay.semantic_tag)

    return {semantic_tag: tuple(sorted(descriptor_keys)) for semantic_tag, descriptor_keys in keys_by_semantic_tag.items()}


def _matrix_descriptor_shape_from_operands(
    operands: Iterable[Operand],
) -> _MatrixDescriptorShape | None:
    operand_units = {operand.field_name: operand.unit_count for operand in operands}
    if any(field_name not in operand_units for field_name in ("a", "b", "acc", "dst")):
        return None
    has_scale_a_pair = "scale_a" in operand_units or "scale_b" in operand_units
    has_scale_src_pair = "scale_src0" in operand_units or "scale_src1" in operand_units
    if has_scale_a_pair and ("scale_a" not in operand_units or "scale_b" not in operand_units):
        raise ValueError("AMDGPU matrix descriptor has incomplete scale_a/scale_b operands")
    if has_scale_src_pair and ("scale_src0" not in operand_units or "scale_src1" not in operand_units):
        raise ValueError("AMDGPU matrix descriptor has incomplete scale_src0/scale_src1 operands")
    return _MatrixDescriptorShape(
        lhs_register_count=operand_units["a"],
        rhs_register_count=operand_units["b"],
        accumulator_register_count=operand_units["acc"],
        result_register_count=operand_units["dst"],
        has_sparse_index="index" in operand_units,
        has_scale_operands=has_scale_a_pair or has_scale_src_pair,
    )


def _matrix_descriptor_shapes_by_key() -> dict[str, tuple[_MatrixDescriptorShape, ...]]:
    shapes_by_key: dict[str, set[_MatrixDescriptorShape]] = {}

    def add_descriptor_shape(
        descriptor_key: str,
        operands: Iterable[Operand],
    ) -> None:
        shape = _matrix_descriptor_shape_from_operands(operands)
        if shape is None:
            return
        shapes_by_key.setdefault(descriptor_key, set()).add(shape)

    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        for descriptor in descriptor_set.descriptors:
            add_descriptor_shape(descriptor.key, descriptor.operands)

    for overlays in (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    ):
        for overlay in overlays:
            add_descriptor_shape(
                overlay.descriptor_key,
                (operand_overlay.descriptor_operand for operand_overlay in overlay.operands),
            )

    return {descriptor_key: tuple(sorted(shapes, key=_matrix_descriptor_shape_sort_key)) for descriptor_key, shapes in shapes_by_key.items()}


def _matrix_descriptor_shape_sort_key(
    shape: _MatrixDescriptorShape,
) -> tuple[int, int, int, int, bool, bool]:
    return (
        shape.lhs_register_count,
        shape.rhs_register_count,
        shape.accumulator_register_count,
        shape.result_register_count,
        shape.has_sparse_index,
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
        has_sparse_index="sparse" in contract.flags,
        has_scale_operands=has_scale_operands,
    )


def _format_matrix_descriptor_shape(shape: _MatrixDescriptorShape) -> str:
    sparse_suffix = ", sparse" if shape.has_sparse_index else ""
    scale_suffix = ", scaled" if shape.has_scale_operands else ""
    return f"a={shape.lhs_register_count}, b={shape.rhs_register_count}, acc={shape.accumulator_register_count}, dst={shape.result_register_count}{sparse_suffix}{scale_suffix}"


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


def _payload_initializer(payload: AmdgpuMatrixPayload) -> str:
    numeric_type = _NUMERIC_TYPE_C_NAMES.get(payload.numeric_type)
    if numeric_type is None:
        raise ValueError(f"unknown AMDGPU matrix numeric type '{payload.numeric_type}'")
    return "\n".join(
        [
            "(loom_amdgpu_matrix_payload_shape_t){",
            f"    .numeric_type = {numeric_type},",
            f"    .register_count = {payload.register_count},",
            f"    .element_count = {payload.element_count},",
            "}",
        ]
    )


def _contract_initializer(
    contract: AmdgpuMatrixContract,
    *,
    keys_by_semantic_tag: Mapping[str, tuple[str, ...]],
    descriptor_shapes_by_key: Mapping[str, tuple[_MatrixDescriptorShape, ...]],
) -> str:
    _validate_known_values(
        contract.features,
        _FEATURE_C_NAMES,
        field_name="feature",
        contract=contract,
    )
    _validate_known_values(contract.flags, _FLAG_C_NAMES, field_name="flag", contract=contract)
    descriptor_key = _resolve_contract_descriptor_key(
        contract,
        keys_by_semantic_tag=keys_by_semantic_tag,
        descriptor_shapes_by_key=descriptor_shapes_by_key,
    )
    if descriptor_key is not None:
        _validate_contract_descriptor_shape(
            contract,
            descriptor_key,
            descriptor_shapes_by_key=descriptor_shapes_by_key,
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
    result_row_count, result_column_count, reduction_count = contract.tile_shape
    return "\n".join(
        [
            "{",
            f'    .name = IREE_SVL("{contract.name}"),',
            f"    .low_descriptor_ref = {low_descriptor_ref},",
            f'    .llvm_intrinsic_name = IREE_SVL("{_contract_intrinsic_name(contract)}"),',
            f"    .family = {family},",
            f"    .required_feature_bits = {_c_bitset(contract.features, _FEATURE_C_NAMES, field_name='feature', contract=contract)},",
            f"    .wave_size_bits = {wave_size},",
            f"    .flags = {_c_bitset(contract.flags, _FLAG_C_NAMES, field_name='flag', contract=contract)},",
            "    .tile_shape = (loom_amdgpu_matrix_tile_shape_t){",
            f"        .result_row_count = {result_row_count},",
            f"        .result_column_count = {result_column_count},",
            f"        .reduction_count = {reduction_count},",
            "    },",
            f"    .lhs_payload = {_payload_initializer(contract.lhs)},",
            f"    .rhs_payload = {_payload_initializer(contract.rhs)},",
            f"    .accumulator_payload = {_payload_initializer(contract.accumulator)},",
            f"    .result_payload = {_payload_initializer(contract.result)},",
            f"    .scale_kind = {scale_kind},",
            f"    .fragment_layout_kind = {fragment_layout},",
            "},",
        ]
    )


def _emit_header() -> str:
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
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        "#endif  // LOOM_TARGET_ARCH_AMDGPU_MATRIX_CONTRACT_TABLES_H_",
    ]
    return "\n".join(lines) + "\n"


def _emit_source(*, public_header: str) -> str:
    keys_by_semantic_tag = _matrix_descriptor_keys_by_semantic_tag()
    descriptor_shapes_by_key = _matrix_descriptor_shapes_by_key()
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
        "const loom_amdgpu_matrix_contract_descriptor_t",
        "    kLoomAmdgpuMatrixContractDescriptors[] = {",
    ]
    lines.extend(
        _contract_initializer(
            contract,
            keys_by_semantic_tag=keys_by_semantic_tag,
            descriptor_shapes_by_key=descriptor_shapes_by_key,
        )
        for contract in AMDGPU_MATRIX_CONTRACTS
    )
    lines.extend(
        [
            "};",
            "",
            "const iree_host_size_t kLoomAmdgpuMatrixContractDescriptorCount =",
            "    IREE_ARRAYSIZE(kLoomAmdgpuMatrixContractDescriptors);",
        ]
    )
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
    args = parser.parse_args(argv)

    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_header(),
        encoding="utf-8",
    )
    args.source.write_text(
        _emit_source(public_header=args.public_header),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
