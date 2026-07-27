# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from loom.target.arch.amdgpu.target_info import (
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    validate_amdgpu_descriptor_set_isa_xml,
)


@dataclass(frozen=True, slots=True)
class _IsaArchitecture:
    source_name: str
    architecture_name: str
    architecture_id: int


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(
                f"ValueError message {exc!s} did not match {match}"
            ) from exc
    else:
        raise AssertionError("expected ValueError")


def test_descriptor_set_isa_xml_validation_accepts_matching_architecture() -> None:
    spec = _IsaArchitecture(
        source_name="rdna4.xml",
        architecture_name="AMD RDNA 4",
        architecture_id=10,
    )

    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4"), spec
    )
    validate_amdgpu_descriptor_set_isa_xml(
        amdgpu_descriptor_set_info_by_generator_target("rdna4_gfx125x"), spec
    )


def test_descriptor_set_isa_xml_validation_rejects_mismatched_architecture() -> None:
    spec = _IsaArchitecture(
        source_name="rdna4.xml",
        architecture_name="AMD RDNA 4",
        architecture_id=10,
    )

    with _raises_value_error(
        "amdgpu.rdna3.core expects AMD RDNA 3 architecture id 8, "
        "found AMD RDNA 4 architecture id 10"
    ):
        validate_amdgpu_descriptor_set_isa_xml(
            amdgpu_descriptor_set_info_by_generator_target("rdna3"), spec
        )


def test_descriptor_set_generator_target_lookup_rejects_unknown_target() -> None:
    with _raises_value_error("unknown AMDGPU descriptor generator target"):
        amdgpu_descriptor_set_info_by_generator_target("gfx999")


def test_descriptor_set_storage_target_lookup_classifies_storage_targets() -> None:
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target(
            "rdna4_gfx125x"
        ).generator_target
        == "rdna4_gfx125x"
    )
    assert (
        amdgpu_descriptor_set_storage_info_by_generator_target("cdna3").generator_target
        == "cdna3"
    )

    assert amdgpu_descriptor_set_view_infos_by_storage_generator_target("rdna4") == ()
    assert (
        amdgpu_descriptor_set_view_infos_by_storage_generator_target("rdna4_gfx125x")
        == ()
    )
    assert amdgpu_descriptor_set_view_infos_by_storage_generator_target("cdna3") == ()
