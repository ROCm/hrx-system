# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
from pathlib import Path

from loom.gen.target.arch.amd.xdna.aie2p import encoding_tables


def test_outputs_contain_owned_tables() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        encoding_path = Path(temp_dir) / "encoding_tables.inl"
        machine_path = Path(temp_dir) / "machine_tables.inl"
        descriptor_header_path = Path(temp_dir) / "core_descriptors.h"
        descriptor_source_path = Path(temp_dir) / "core_descriptors.c"
        array_descriptor_header_path = Path(temp_dir) / "array_descriptors.h"
        array_descriptor_source_path = Path(temp_dir) / "array_descriptors.c"
        assert (
            encoding_tables.main(
                [
                    "--encoding-output",
                    str(encoding_path),
                    "--machine-output",
                    str(machine_path),
                    "--descriptor-header-output",
                    str(descriptor_header_path),
                    "--descriptor-source-output",
                    str(descriptor_source_path),
                    "--array-descriptor-header-output",
                    str(array_descriptor_header_path),
                    "--array-descriptor-source-output",
                    str(array_descriptor_source_path),
                ]
            )
            == 0
        )
        encoding_contents = encoding_path.read_text(encoding="utf-8")
        machine_contents = machine_path.read_text(encoding="utf-8")
        descriptor_header_contents = descriptor_header_path.read_text(encoding="utf-8")
        descriptor_source_contents = descriptor_source_path.read_text(encoding="utf-8")
        array_descriptor_header_contents = array_descriptor_header_path.read_text(encoding="utf-8")
        array_descriptor_source_contents = array_descriptor_source_path.read_text(encoding="utf-8")

    assert "kLoomAie2pInstructionLayouts" in encoding_contents
    assert "kLoomAie2pEncodingFieldNames" in encoding_contents
    assert "kLoomAie2pSlotBitCounts" in encoding_contents
    assert "LOOM_AIE2P_INSTRUCTION_VADD_32" not in encoding_contents
    assert "MOV_OR" not in encoding_contents
    assert "ce8c0f8fd66bff15b347351c67e9fb4fe0a17205" in encoding_contents
    assert "kLoomAie2pPhysicalRegisters" in machine_contents
    assert "kLoomAie2pRegisterClasses" in machine_contents
    assert "kLoomAie2pMachineForms" in machine_contents
    assert "MOV_OR" not in machine_contents
    assert "loom_aie2p_core_descriptor_set" in descriptor_header_contents
    assert ".encoding_adapter_id = " in descriptor_source_contents
    assert ".physical_register_count = IREE_ARRAYSIZE(kAie2pCorePhysicalRegisters)" in descriptor_source_contents
    assert "loom_aie2p_array_descriptor_set" in array_descriptor_header_contents
    assert "amd.xdna.aie2p.array.worker" in array_descriptor_source_contents
    assert "aie2p.array.binding_access" in array_descriptor_source_contents


def test_check_validates_without_output() -> None:
    assert encoding_tables.main(["--check"]) == 0
