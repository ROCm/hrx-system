# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.gen.target.arch.amdgpu.records import amdgpu_target_records
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX9,
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX10_12,
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX125X,
    AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32,
    AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AmdgpuDescriptorSetBufferResourceInfo,
    AmdgpuDescriptorSetInfo,
    AmdgpuTargetRecordInfo,
    processor_info,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    sorted_target_record_infos,
)


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(f"ValueError message {exc!s} did not match {match}") from exc
    else:
        raise AssertionError("expected ValueError")


def _descriptor_set_info() -> AmdgpuDescriptorSetInfo:
    return AmdgpuDescriptorSetInfo(
        generator_target="test",
        key="amdgpu.test.core",
        isa_xml_key="test",
        isa_architecture_name="AMDGPU Test",
        isa_architecture_id=1,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        buffer_resource=AmdgpuDescriptorSetBufferResourceInfo(
            layout=AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32,
            intrinsic_flags=AMDGPU_BUFFER_RESOURCE_FLAGS_GFX9,
        ),
    )


def _row(
    processor_name: str,
    *,
    enum_value: int = 1,
    default_for_descriptor_set: bool = True,
    max_workgroup_storage_bytes: int = (AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES),
) -> amdgpu_target_records._AmdgpuTargetRecordRow:
    descriptor_set = _descriptor_set_info()
    processor = processor_info(
        processor_name,
        0x001,
        descriptor_set_key=descriptor_set.key,
        max_workgroup_storage_bytes=max_workgroup_storage_bytes,
    )
    info = AmdgpuTargetRecordInfo(
        processor=processor_name,
        enum_value=enum_value,
        doc=f"Test target record for {processor_name}.",
        default_for_descriptor_set=default_for_descriptor_set,
    )
    return amdgpu_target_records._AmdgpuTargetRecordRow(
        info=info,
        processor=processor,
        descriptor_set=descriptor_set,
        descriptor_set_ordinal=0,
    )


def test_target_records_materialize_current_rows() -> None:
    rows = amdgpu_target_records._materialize_rows(
        sorted_target_record_infos(),
        sorted_processor_infos(),
        sorted_descriptor_set_infos(),
    )

    amdgpu_target_records._validate_target_record_infos(rows)
    source = amdgpu_target_records._emit_tables(rows)

    assert "typedef " not in source
    assert "static " not in source
    assert "LOOM_AMDGPU_TARGET_RECORD_INFO(Gfx1250" in source
    assert "LOOM_AMDGPU_TARGET_RECORD_DEFAULT(" in source
    assert (f"LOOM_AMDGPU_TARGET_BUFFER_RESOURCE(Cdna3, UINT32_C({AMDGPU_BUFFER_RESOURCE_FLAGS_GFX9}))") in source
    assert (f"LOOM_AMDGPU_TARGET_BUFFER_RESOURCE(Rdna3, UINT32_C({AMDGPU_BUFFER_RESOURCE_FLAGS_GFX10_12}))") in source
    assert (f"LOOM_AMDGPU_TARGET_BUFFER_RESOURCE(Rdna4Gfx125x, UINT32_C({AMDGPU_BUFFER_RESOURCE_FLAGS_GFX125X}))") in source
    assert "Gfx1250)" in source
    assert f"UINT64_C({AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES})" in source


def test_target_records_reject_unknown_processor() -> None:
    with _raises_value_error("does not name a known processor"):
        amdgpu_target_records._materialize_rows(
            (
                AmdgpuTargetRecordInfo(
                    processor="gfx9999",
                    enum_value=1,
                    doc="Unknown test target record.",
                ),
            ),
            sorted_processor_infos(),
            sorted_descriptor_set_infos(),
        )


def test_target_records_reject_sparse_enum_values() -> None:
    with _raises_value_error("dense one-based range"):
        amdgpu_target_records._validate_target_record_infos((_row("gfx-test", enum_value=2),))


def test_target_records_reject_duplicate_processors() -> None:
    with _raises_value_error("processors must be unique"):
        amdgpu_target_records._validate_target_record_infos(
            (
                _row("gfx-test", enum_value=1),
                _row("gfx-test", enum_value=2, default_for_descriptor_set=False),
            )
        )


def test_target_records_require_one_default_per_descriptor_set() -> None:
    with _raises_value_error("requires exactly one default target record, found 0"):
        amdgpu_target_records._validate_target_record_infos((_row("gfx-test", default_for_descriptor_set=False),))

    with _raises_value_error("requires exactly one default target record, found 2"):
        amdgpu_target_records._validate_target_record_infos(
            (
                _row("gfx-test-a", enum_value=1),
                _row("gfx-test-b", enum_value=2),
            )
        )


def test_target_records_require_consistent_descriptor_set_storage_limits() -> None:
    with _raises_value_error("requires a max workgroup storage limit"):
        amdgpu_target_records._validate_target_record_infos((_row("gfx-test", max_workgroup_storage_bytes=0),))

    with _raises_value_error("inconsistent max workgroup storage limits"):
        amdgpu_target_records._validate_target_record_infos(
            (
                _row("gfx-test-a", enum_value=1),
                _row(
                    "gfx-test-b",
                    enum_value=2,
                    default_for_descriptor_set=False,
                    max_workgroup_storage_bytes=32 * 1024,
                ),
            )
        )
