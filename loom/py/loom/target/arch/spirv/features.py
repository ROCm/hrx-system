# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth SPIR-V feature atom catalog."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

SPIRV_VERSION_1_0 = 0x00010000
SPIRV_VERSION_1_3 = 0x00010300

ADDRESSING_MODEL_UNSPECIFIED = "LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED"
MEMORY_MODEL_UNSPECIFIED = "LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED"

_C_SUFFIX_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class FeatureAtom:
    key: str
    c_suffix: str
    name: str
    doc: str
    minimum_spirv_version: int
    required: tuple[str, ...] = ()
    addressing_model: str = ADDRESSING_MODEL_UNSPECIFIED
    memory_model: str = MEMORY_MODEL_UNSPECIFIED
    extensions: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    opcodes: tuple[str, ...] = ()
    storage_classes: tuple[str, ...] = ()
    decorations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    c_suffix: str
    doc: str
    atoms: tuple[str, ...]


FEATURE_ATOMS = (
    FeatureAtom(
        key="vulkan_shader",
        c_suffix="VULKAN_SHADER",
        name="spirv.vulkan.shader",
        doc="Vulkan shader-module baseline.",
        required=(),
        minimum_spirv_version=SPIRV_VERSION_1_3,
        memory_model="LOOM_SPIRV_MEMORY_MODEL_VULKAN",
        extensions=("SPV_KHR_vulkan_memory_model",),
        capabilities=(
            "LOOM_SPIRV_CAPABILITY_SHADER",
            "LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL",
        ),
    ),
    FeatureAtom(
        key="physical_storage_buffer",
        c_suffix="PHYSICAL_STORAGE_BUFFER",
        name="spirv.physical_storage_buffer",
        doc="SPV_KHR_physical_storage_buffer support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        addressing_model="LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64",
        extensions=("SPV_KHR_physical_storage_buffer",),
        capabilities=(
            "LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES",
            "LOOM_SPIRV_CAPABILITY_INT64",
        ),
        storage_classes=("LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER",),
        decorations=(
            "LOOM_SPIRV_DECORATION_RESTRICT_POINTER",
            "LOOM_SPIRV_DECORATION_ALIASED_POINTER",
        ),
    ),
    FeatureAtom(
        key="float16",
        c_suffix="FLOAT16",
        name="spirv.float16",
        doc="16-bit floating-point scalar support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        capabilities=("LOOM_SPIRV_CAPABILITY_FLOAT16",),
    ),
    FeatureAtom(
        key="float64",
        c_suffix="FLOAT64",
        name="spirv.float64",
        doc="64-bit floating-point scalar support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        capabilities=("LOOM_SPIRV_CAPABILITY_FLOAT64",),
    ),
    FeatureAtom(
        key="int8",
        c_suffix="INT8",
        name="spirv.int8",
        doc="8-bit integer scalar support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        capabilities=("LOOM_SPIRV_CAPABILITY_INT8",),
    ),
    FeatureAtom(
        key="int16",
        c_suffix="INT16",
        name="spirv.int16",
        doc="16-bit integer scalar support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        capabilities=("LOOM_SPIRV_CAPABILITY_INT16",),
    ),
    FeatureAtom(
        key="storage_buffer_8bit_access",
        c_suffix="STORAGE_BUFFER_8BIT_ACCESS",
        name="spirv.storage_buffer_8bit_access",
        doc="8-bit storage-buffer access support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_8bit_storage",),
        capabilities=("LOOM_SPIRV_CAPABILITY_STORAGE_BUFFER8_BIT_ACCESS",),
    ),
    FeatureAtom(
        key="storage_buffer_16bit_access",
        c_suffix="STORAGE_BUFFER_16BIT_ACCESS",
        name="spirv.storage_buffer_16bit_access",
        doc="16-bit storage-buffer access support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_16bit_storage",),
        capabilities=("LOOM_SPIRV_CAPABILITY_STORAGE_BUFFER16_BIT_ACCESS",),
    ),
    FeatureAtom(
        key="cooperative_vector_nv",
        c_suffix="COOPERATIVE_VECTOR_NV",
        name="spirv.cooperative_vector.nv",
        doc="SPV_NV_cooperative_vector support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_NV_cooperative_vector",),
        capabilities=("LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_NV",),
        opcodes=(
            "LOOM_SPIRV_OP_TYPE_COOPERATIVE_VECTOR_NV",
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_MATRIX_MUL_NV",
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_MATRIX_MUL_ADD_NV",
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_LOAD_NV",
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_STORE_NV",
        ),
    ),
    FeatureAtom(
        key="cooperative_vector_training_nv",
        c_suffix="COOPERATIVE_VECTOR_TRAINING_NV",
        name="spirv.cooperative_vector.training.nv",
        doc="SPV_NV_cooperative_vector training-operation support.",
        required=("vulkan_shader", "cooperative_vector_nv"),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_NV_cooperative_vector",),
        capabilities=("LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_TRAINING_NV",),
        opcodes=(
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_OUTER_PRODUCT_ACCUMULATE_NV",
            "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_REDUCE_SUM_ACCUMULATE_NV",
        ),
    ),
    FeatureAtom(
        key="cooperative_matrix_khr",
        c_suffix="COOPERATIVE_MATRIX_KHR",
        name="spirv.cooperative_matrix.khr",
        doc="SPV_KHR_cooperative_matrix support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_cooperative_matrix",),
        capabilities=("LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR",),
        opcodes=(
            "LOOM_SPIRV_OP_TYPE_COOPERATIVE_MATRIX_KHR",
            "LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            "LOOM_SPIRV_OP_COOPERATIVE_MATRIX_STORE_KHR",
            "LOOM_SPIRV_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR",
            "LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LENGTH_KHR",
        ),
    ),
    FeatureAtom(
        key="bfloat16_type_khr",
        c_suffix="BFLOAT16_TYPE_KHR",
        name="spirv.bfloat16.type.khr",
        doc="SPV_KHR_bfloat16 scalar type support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_bfloat16",),
        capabilities=("LOOM_SPIRV_CAPABILITY_B_FLOAT16_TYPE_KHR",),
    ),
    FeatureAtom(
        key="bfloat16_dot_product_khr",
        c_suffix="BFLOAT16_DOT_PRODUCT_KHR",
        name="spirv.bfloat16.dot_product.khr",
        doc="SPV_KHR_bfloat16 dot-product support.",
        required=("vulkan_shader", "bfloat16_type_khr"),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_bfloat16",),
        capabilities=("LOOM_SPIRV_CAPABILITY_B_FLOAT16_DOT_PRODUCT_KHR",),
    ),
    FeatureAtom(
        key="bfloat16_cooperative_matrix_khr",
        c_suffix="BFLOAT16_COOPERATIVE_MATRIX_KHR",
        name="spirv.bfloat16.cooperative_matrix.khr",
        doc="SPV_KHR_bfloat16 cooperative-matrix support.",
        required=(
            "vulkan_shader",
            "cooperative_matrix_khr",
            "bfloat16_type_khr",
        ),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        extensions=("SPV_KHR_bfloat16",),
        capabilities=("LOOM_SPIRV_CAPABILITY_B_FLOAT16_COOPERATIVE_MATRIX_KHR",),
    ),
    FeatureAtom(
        key="int64",
        c_suffix="INT64",
        name="spirv.int64",
        doc="64-bit integer scalar support.",
        required=("vulkan_shader",),
        minimum_spirv_version=SPIRV_VERSION_1_0,
        capabilities=("LOOM_SPIRV_CAPABILITY_INT64",),
    ),
)

FEATURE_PROFILES = (
    FeatureProfile(
        c_suffix="VULKAN_1_3_BDA",
        doc="Feature bits selected by the built-in Vulkan 1.3 BDA profile.",
        atoms=("vulkan_shader", "physical_storage_buffer", "int64"),
    ),
)


def atom_by_key(atoms: Iterable[FeatureAtom] = FEATURE_ATOMS) -> dict[str, FeatureAtom]:
    return {atom.key: atom for atom in atoms}


def feature_bit_constant(atom: FeatureAtom) -> str:
    return f"LOOM_SPIRV_FEATURE_{atom.c_suffix}"


def feature_bit_value(
    atom_key: str,
    *,
    atoms: Iterable[FeatureAtom] = FEATURE_ATOMS,
) -> int:
    for index, atom in enumerate(atoms, start=1):
        if atom.key == atom_key:
            return 1 << index
    raise ValueError(f"unknown SPIR-V feature atom {atom_key!r}")


def feature_atom_enum(atom: FeatureAtom) -> str:
    return f"LOOM_SPIRV_FEATURE_ATOM_{atom.c_suffix}"


def feature_bit_constants(
    atom_keys: Iterable[str],
    *,
    atoms: Iterable[FeatureAtom] = FEATURE_ATOMS,
) -> tuple[str, ...]:
    atoms_by_key = atom_by_key(atoms)
    return tuple(feature_bit_constant(atoms_by_key[key]) for key in atom_keys)


def feature_bits_expression(
    atom_keys: Iterable[str],
    *,
    atoms: Iterable[FeatureAtom] = FEATURE_ATOMS,
) -> str:
    parts = feature_bit_constants(atom_keys, atoms=atoms)
    return " | ".join(parts) if parts else "0"


def feature_bits_value(
    atom_keys: Iterable[str],
    *,
    atoms: Iterable[FeatureAtom] = FEATURE_ATOMS,
) -> int:
    bits = 0
    for atom_key in atom_keys:
        bits |= feature_bit_value(atom_key, atoms=atoms)
    return bits


def feature_row_capacity(
    field_name: str,
    *,
    atoms: Iterable[FeatureAtom] = FEATURE_ATOMS,
) -> int:
    unique_values = {value for atom in atoms for value in getattr(atom, field_name)}
    return len(unique_values)


def parse_isa_symbols(isa_header: str) -> frozenset[str]:
    return frozenset(re.findall(r"\bLOOM_SPIRV_[A-Z0-9_]+\b", isa_header))


def validate_feature_catalog(
    *,
    atoms: tuple[FeatureAtom, ...] = FEATURE_ATOMS,
    profiles: tuple[FeatureProfile, ...] = FEATURE_PROFILES,
    isa_symbols: frozenset[str] | None = None,
) -> None:
    _validate_unique_atoms(atoms)
    atoms_by_key = atom_by_key(atoms)
    _validate_dependencies(atoms, atoms_by_key)
    _validate_profiles(profiles, atoms_by_key)
    _validate_row_sets(atoms)
    _validate_model_consistency(atoms)
    if isa_symbols is not None:
        _validate_isa_symbols(atoms, isa_symbols)
    if len(atoms) >= 64:
        raise ValueError("SPIR-V feature atom bitset supports at most 63 atoms")
    for field_name in _ROW_FIELD_NAMES:
        row_count = feature_row_capacity(field_name, atoms=atoms)
        if row_count > 255:
            raise ValueError(
                f"SPIR-V feature {field_name} rows exceed uint8_t capacity"
            )


_ROW_FIELD_NAMES = (
    "extensions",
    "capabilities",
    "opcodes",
    "storage_classes",
    "decorations",
)

_ISA_SYMBOL_FIELD_NAMES = (
    "capabilities",
    "opcodes",
    "storage_classes",
    "decorations",
)


def _validate_unique_atoms(atoms: tuple[FeatureAtom, ...]) -> None:
    _ensure_unique("feature atom key", (atom.key for atom in atoms))
    _ensure_unique("feature atom C suffix", (atom.c_suffix for atom in atoms))
    _ensure_unique("feature atom diagnostic name", (atom.name for atom in atoms))
    for atom in atoms:
        if not _KEY_PATTERN.match(atom.key):
            raise ValueError(f"feature atom key {atom.key!r} is not a stable key")
        if not _C_SUFFIX_PATTERN.match(atom.c_suffix):
            raise ValueError(
                f"feature atom {atom.key!r} has invalid C suffix {atom.c_suffix!r}"
            )
        if not atom.doc.endswith("."):
            raise ValueError(
                f"feature atom {atom.key!r} doc must be a complete sentence"
            )


def _validate_dependencies(
    atoms: tuple[FeatureAtom, ...],
    atoms_by_key: Mapping[str, FeatureAtom],
) -> None:
    for atom in atoms:
        for required_key in atom.required:
            if required_key not in atoms_by_key:
                raise ValueError(
                    f"feature atom {atom.key!r} requires unknown atom {required_key!r}"
                )
            if required_key == atom.key:
                raise ValueError(f"feature atom {atom.key!r} requires itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(atom: FeatureAtom) -> None:
        if atom.key in visited:
            return
        if atom.key in visiting:
            raise ValueError(
                f"feature atom {atom.key!r} participates in a dependency cycle"
            )
        visiting.add(atom.key)
        for required_key in atom.required:
            visit(atoms_by_key[required_key])
        visiting.remove(atom.key)
        visited.add(atom.key)

    for atom in atoms:
        visit(atom)


def _validate_profiles(
    profiles: tuple[FeatureProfile, ...],
    atoms_by_key: Mapping[str, FeatureAtom],
) -> None:
    _ensure_unique(
        "feature profile C suffix", (profile.c_suffix for profile in profiles)
    )
    for profile in profiles:
        if not _C_SUFFIX_PATTERN.match(profile.c_suffix):
            raise ValueError(
                f"feature profile has invalid C suffix {profile.c_suffix!r}"
            )
        if not profile.doc.endswith("."):
            raise ValueError(
                f"feature profile {profile.c_suffix!r} doc must be a complete sentence"
            )
        selected = set(profile.atoms)
        for atom_key in profile.atoms:
            if atom_key not in atoms_by_key:
                raise ValueError(
                    f"feature profile {profile.c_suffix!r} references "
                    f"unknown atom {atom_key!r}"
                )
            missing_dependencies = set(atoms_by_key[atom_key].required) - selected
            if missing_dependencies:
                missing_text = ", ".join(sorted(missing_dependencies))
                raise ValueError(
                    f"feature profile {profile.c_suffix!r} atom {atom_key!r} "
                    f"misses dependencies: {missing_text}"
                )


def _validate_row_sets(atoms: tuple[FeatureAtom, ...]) -> None:
    for atom in atoms:
        for field_name in _ROW_FIELD_NAMES:
            _ensure_unique(
                f"feature atom {atom.key!r} {field_name}", getattr(atom, field_name)
            )


def _validate_model_consistency(atoms: tuple[FeatureAtom, ...]) -> None:
    _ensure_single_model(
        atoms,
        field_name="addressing_model",
        unspecified_value=ADDRESSING_MODEL_UNSPECIFIED,
    )
    _ensure_single_model(
        atoms,
        field_name="memory_model",
        unspecified_value=MEMORY_MODEL_UNSPECIFIED,
    )


def _validate_isa_symbols(
    atoms: tuple[FeatureAtom, ...], isa_symbols: frozenset[str]
) -> None:
    for atom in atoms:
        for field_name in _ISA_SYMBOL_FIELD_NAMES:
            for symbol in getattr(atom, field_name):
                if symbol not in isa_symbols:
                    raise ValueError(
                        f"feature atom {atom.key!r} {field_name} references "
                        f"unknown SPIR-V ISA symbol {symbol}"
                    )
        for field_name, unspecified_value in (
            ("addressing_model", ADDRESSING_MODEL_UNSPECIFIED),
            ("memory_model", MEMORY_MODEL_UNSPECIFIED),
        ):
            symbol = getattr(atom, field_name)
            if symbol != unspecified_value and symbol not in isa_symbols:
                raise ValueError(
                    f"feature atom {atom.key!r} {field_name} references "
                    f"unknown SPIR-V ISA symbol {symbol}"
                )


def _ensure_unique(label: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{label} repeats {value!r}")
        seen.add(value)


def _ensure_single_model(
    atoms: tuple[FeatureAtom, ...],
    *,
    field_name: str,
    unspecified_value: str,
) -> None:
    specified_values = {
        getattr(atom, field_name)
        for atom in atoms
        if getattr(atom, field_name) != unspecified_value
    }
    if len(specified_values) > 1:
        raise ValueError(
            f"SPIR-V feature atoms specify conflicting {field_name} values: "
            f"{', '.join(sorted(specified_values))}"
        )
