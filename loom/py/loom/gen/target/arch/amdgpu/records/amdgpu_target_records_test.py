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
    AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_SOPP_OPCODE_INFO_RDNA,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
    AmdgpuDescriptorSetInfo,
    AmdgpuDescriptorSetIsaInfo,
    AmdgpuTargetInfo,
    processor_info,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    sorted_target_infos,
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
        isa_infos=(
            AmdgpuDescriptorSetIsaInfo(
                isa_xml_key="test",
                isa_architecture_name="AMDGPU Test",
                isa_architecture_id=1,
                sopp_opcodes=AMDGPU_SOPP_OPCODE_INFO_RDNA,
            ),
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
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
    info = AmdgpuTargetInfo(
        target=processor_name,
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
    processors = sorted_processor_infos()
    rows = amdgpu_target_records._materialize_rows(
        sorted_target_infos(),
        processors,
        sorted_descriptor_set_infos(),
    )

    amdgpu_target_records._validate_target_record_infos(rows)
    amdgpu_target_records._validate_target_record_coverage(rows, processors)
    source = amdgpu_target_records._emit_tables(rows)

    assert "typedef " not in source
    assert "static " not in source
    for row in rows:
        suffix = amdgpu_target_records._c_symbol_suffix(row.info.target)
        descriptor_suffix = amdgpu_target_records._c_symbol_suffix(row.descriptor_set.generator_target)
        assert (f'LOOM_AMDGPU_TARGET_RECORD_INFO({suffix}, UINT32_C({row.info.enum_value}), "{row.info.target}", UINT16_C({row.descriptor_set_ordinal}), {descriptor_suffix})') in source
        if row.info.default_for_descriptor_set:
            assert (f"LOOM_AMDGPU_TARGET_RECORD_DEFAULT(UINT16_C({row.descriptor_set_ordinal}), {suffix})") in source

    assert "LOOM_AMDGPU_TARGET_PROFILE(Gfx942SrameccAnyXnackAny, UINT32_C(1), Cdna3, LOOM_AMDGPU_TARGET_FEATURE_ANY, LOOM_AMDGPU_TARGET_FEATURE_ANY)" in source
    assert "LOOM_AMDGPU_TARGET_PROFILE(Gfx942SrameccOnXnackOff, UINT32_C(1), Cdna3, LOOM_AMDGPU_TARGET_FEATURE_ON, LOOM_AMDGPU_TARGET_FEATURE_OFF)" in source
    assert "LOOM_AMDGPU_TARGET_PROFILE(Gfx1151SrameccUnsupportedXnackUnsupported, UINT32_C(15), Rdna35, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED, LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED)" in source

    expected_profile_count = 0
    for row in rows:
        supported_features = row.processor.target_id.supported_features
        sramecc_state_count = 3 if supported_features & AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC else 1
        xnack_state_count = 3 if supported_features & AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK else 1
        expected_profile_count += sramecc_state_count * xnack_state_count
    assert source.count("\nLOOM_AMDGPU_TARGET_PROFILE(") == expected_profile_count

    rows_by_target = {row.info.target: row for row in rows}
    assert rows_by_target["gfx1250"].descriptor_set.key == "amdgpu.rdna4.gfx125x.core"
    assert rows_by_target["gfx1250-a0"].descriptor_set.key == "amdgpu.rdna4.gfx1250_a0.core"


def test_target_record_enum_values_are_stable_and_explicit() -> None:
    assert {info.target: info.enum_value for info in sorted_target_infos()} == {
        "gfx942": 1,
        "gfx950": 2,
        "gfx1100": 3,
        "gfx1200": 4,
        "gfx1250": 5,
        "gfx1150": 6,
        "gfx11-generic": 7,
        "gfx12-generic": 8,
        "gfx12-5-generic": 9,
        "gfx940": 10,
        "gfx941": 11,
        "gfx1101": 12,
        "gfx1102": 13,
        "gfx1103": 14,
        "gfx1151": 15,
        "gfx1152": 16,
        "gfx1153": 17,
        "gfx1170": 18,
        "gfx1171": 19,
        "gfx1172": 20,
        "gfx1201": 21,
        "gfx1251": 22,
        "gfx9-4-generic": 23,
        "gfx1250-a0": 24,
    }


def test_target_records_reject_missing_descriptor_backed_processor() -> None:
    processors = sorted_processor_infos()
    rows = amdgpu_target_records._materialize_rows(
        sorted_target_infos(),
        processors,
        sorted_descriptor_set_infos(),
    )
    missing_processor = next(row.info.processor for row in reversed(rows) if row.info.target == row.info.processor)
    incomplete_rows = tuple(row for row in rows if not (row.info.processor == missing_processor and row.info.target == row.info.processor))

    with _raises_value_error(rf"missing=\['{re.escape(missing_processor)}'\]"):
        amdgpu_target_records._validate_target_record_coverage(incomplete_rows, processors)


def test_target_records_reject_unknown_processor() -> None:
    with _raises_value_error("does not name a known processor"):
        amdgpu_target_records._materialize_rows(
            (
                AmdgpuTargetInfo(
                    target="gfx9999",
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


def test_target_records_reject_duplicate_identities() -> None:
    with _raises_value_error("identities must be unique"):
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
