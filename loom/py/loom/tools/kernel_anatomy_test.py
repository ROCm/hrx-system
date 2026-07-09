# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
from pathlib import Path

from loom.tools.kernel_anatomy import (
    NamedPath,
    build_kernel_anatomy_report,
    format_text_report,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_compile_report(path: Path) -> None:
    _write(
        path,
        json.dumps(
            {
                "compile_report": {
                    "function": "loom_kernel",
                    "target_key": "gfx1100",
                    "executable_format": "amdgcn-amd-amdhsa--gfx1100",
                    "entries": {
                        "rows": [
                            {
                                "function": "loom_kernel",
                                "instruction_count": 123,
                                "code_byte_count": 456,
                                "private_memory_bytes": 0,
                                "local_memory_bytes": 2048,
                                "static_instruction_mix": {
                                    "vector_alu_count": 17,
                                    "wmma_count": 2,
                                    "mfma_count": 0,
                                },
                                "target_resources": {
                                    "vector": {
                                        "final": {
                                            "register_count": 64,
                                        },
                                    },
                                    "occupancy_percent": 50,
                                },
                                "workload": {
                                    "workgroup_size": {
                                        "flat": 128,
                                    },
                                },
                            },
                        ],
                    },
                },
            }
        ),
    )


def test_build_kernel_anatomy_report_merges_disassembly_and_compile_report(
    tmp_path: Path,
) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <small>:
      global_load_b128 v[0:3], v0
0000000000000100 <big>:
      global_load_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        top_symbol_count=1,
    )

    assert report["schema"] == "loom.kernel_anatomy"
    assert report["schema_version"] == 1
    assert report["loom_compile_reports"]["loom"]["function"] == "loom_kernel"
    assert report["loom_compile_reports"]["loom"]["local_memory_bytes"] == 2048
    assert report["disassemblies"]["asm"]["whole_file"]["family_counts"]["v_wmma"] == 2
    assert report["disassemblies"]["asm"]["top_symbols"][0]["symbol"] == "big"


def test_symbol_regex_filters_disassembly_blocks(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <skip_me>:
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <keep_me>:
      global_store_b128 v0, v[0:3]
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[],
        symbol_regex="keep",
        top_symbol_count=8,
    )

    symbols = report["disassemblies"]["asm"]["top_symbols"]
    assert [symbol["symbol"] for symbol in symbols] == ["keep_me"]


def test_amdhsa_metadata_extracts_kernel_resources(tmp_path: Path) -> None:
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
        .size:           8
    .group_segment_fixed_size: 25600
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 128
    .name:           selected_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         selected_kernel.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
        .size:           8
    .group_segment_fixed_size: 4096
    .name:           skipped_kernel
    .symbol:         skipped_kernel.kd
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[],
        compile_report_paths=[],
        amdhsa_metadata_paths=[NamedPath("hsaco", metadata_path)],
        amdhsa_metadata_regex="selected",
    )

    metadata = report["amdhsa_metadata"]["hsaco"]
    assert metadata["kernel_count"] == 1
    assert metadata["kernels"][0]["symbol"] == "selected_kernel.kd"
    assert metadata["kernels"][0]["group_segment_fixed_size"] == 25600
    assert metadata["kernels"][0]["vgpr_count"] == 256


def test_text_report_contains_kernel_economics(tmp_path: Path) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <kernel>:
      global_load_b128 v[0:3], v0
      global_store_b128 v0, v[0:3]
      buffer_load_b64 v[0:1], v0, s[0:3], 0 offen
      buffer_store_b64 v[0:1], v0, s[0:3], 0 offen
      s_waitcnt vmcnt(0)
      s_barrier
""",
    )
    compile_report_path = tmp_path / "report.json"
    _write_compile_report(compile_report_path)
    metadata_path = tmp_path / "notes.txt"
    _write(
        metadata_path,
        """
amdhsa.kernels:
  - .args:
      - .name:           Tensor2dSizeA
        .offset:         0
    .group_segment_fixed_size: 25600
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 128
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
""",
    )

    report = build_kernel_anatomy_report(
        disassembly_paths=[NamedPath("asm", disassembly_path)],
        compile_report_paths=[NamedPath("loom", compile_report_path)],
        amdhsa_metadata_paths=[NamedPath("hsaco", metadata_path)],
    )
    text = format_text_report(report)

    assert "Kernel anatomy report" in text
    assert "loom: instructions=123 code_bytes=456 local_bytes=2048" in text
    assert "asm: symbols=1 instructions=6" in text
    assert "global_load=1 global_store=1 buffer_load=1 buffer_store=1" in text
    assert "wait=1 barrier=1 read_bytes=24 write_bytes=24" in text
    assert "hsaco: kernels=1" in text
    assert "kernel.kd: lds=25600 private=0 vgpr=256 sgpr=66" in text


def test_main_emits_json(tmp_path: Path, capsys) -> None:
    disassembly_path = tmp_path / "kernel.s"
    _write(
        disassembly_path,
        """
0000000000000000 <kernel>:
      s_endpgm
""",
    )

    assert main(["--disassembly", f"asm={disassembly_path}", "--format", "json"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["schema"] == "loom.kernel_anatomy"
    assert report["disassemblies"]["asm"]["symbol_count"] == 1
