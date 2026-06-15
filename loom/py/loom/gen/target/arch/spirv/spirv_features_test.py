# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

from loom.gen.target.arch.spirv.spirv_features import generate_tables
from loom.target.arch.spirv.features import (
    FEATURE_ATOMS,
    FEATURE_PROFILES,
    FeatureAtom,
    FeatureProfile,
    parse_isa_symbols,
    validate_feature_catalog,
)


@contextmanager
def _raises_value_error(pattern: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{exc!s} did not match {pattern!r}") from exc
    else:
        raise AssertionError(f"expected ValueError matching {pattern!r}")


def test_generation_emits_compact_feature_tables() -> None:
    tables = generate_tables()

    assert "static const loom_spirv_feature_atom_descriptor_t kSpirvFeatureAtoms[]" in tables
    assert 'IREE_SVL("SPV_KHR_8bit_storage")' in tables
    assert 'IREE_SVL("SPV_KHR_16bit_storage")' in tables
    assert 'IREE_SVL("SPV_KHR_bfloat16")' in tables
    assert "LOOM_SPIRV_CAPABILITY_B_FLOAT16_COOPERATIVE_MATRIX_KHR" in tables
    assert "LOOM_SPIRV_OP_COOPERATIVE_VECTOR_MATRIX_MUL_ADD_NV" in tables
    assert "#ifndef LOOM_TARGET_ARCH_SPIRV_FEATURES_H_" not in tables
    assert "typedef enum loom_spirv_feature_bit_e" not in tables
    assert "loom_spirv_feature_set_prepare" not in tables


def test_validation_rejects_unknown_dependency() -> None:
    bad_atoms = (
        FeatureAtom(
            key="bad_atom",
            c_suffix="BAD_ATOM",
            name="spirv.bad",
            doc="Bad atom.",
            required=("missing_atom",),
            minimum_spirv_version=0x00010000,
        ),
    )

    with _raises_value_error(r"requires unknown atom 'missing_atom'"):
        validate_feature_catalog(atoms=bad_atoms, profiles=())


def test_validation_rejects_profile_missing_atom_dependency() -> None:
    bad_profiles = (
        FeatureProfile(
            c_suffix="BAD_PROFILE",
            doc="Bad profile.",
            atoms=("physical_storage_buffer",),
        ),
    )

    with _raises_value_error(r"misses dependencies: vulkan_shader"):
        validate_feature_catalog(atoms=FEATURE_ATOMS, profiles=bad_profiles)


def test_validation_rejects_duplicate_atom_rows() -> None:
    bad_atoms = (
        replace(
            FEATURE_ATOMS[0],
            extensions=("SPV_KHR_vulkan_memory_model", "SPV_KHR_vulkan_memory_model"),
        ),
    )

    with _raises_value_error(r"extensions repeats 'SPV_KHR_vulkan_memory_model'"):
        validate_feature_catalog(atoms=bad_atoms, profiles=())


def test_validation_rejects_unknown_isa_symbol() -> None:
    bad_atoms = (
        replace(
            FEATURE_ATOMS[0],
            capabilities=("LOOM_SPIRV_CAPABILITY_DOES_NOT_EXIST",),
        ),
        *FEATURE_ATOMS[1:],
    )

    with _raises_value_error(r"unknown SPIR-V ISA symbol"):
        validate_feature_catalog(
            atoms=bad_atoms,
            profiles=FEATURE_PROFILES,
            isa_symbols=parse_isa_symbols(""),
        )
