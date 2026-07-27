# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from loom.importers.tilelang.differential import (
    ABI_PROLOGUE,
    INSTRUCTION_SELECTION,
    MEMORY_SHAPE,
    SCHEDULING_WAIT,
    AmdgpuDifferentialArtifact,
    LoomAmdgpuArtifactUnavailable,
    LoomAmdgpuToolchain,
    capture_loom_amdgpu_artifact,
    classify_amdgpu_instruction_family,
    compare_amdgpu_artifacts,
)
from loom.tools.amdgpu_asm import summarize_amdgpu_disassembly


def test_compare_amdgpu_artifacts_reports_family_deltas() -> None:
    tilelang_summary = summarize_amdgpu_disassembly(
        """
global_load_b128 v[0:3], v0
global_store_b128 v0, v[0:3]
s_waitcnt vmcnt(0)
s_endpgm
"""
    )
    loom_summary = summarize_amdgpu_disassembly(
        """
s_load_b128 s[0:3], s[0:1], null
global_load_b128 v[0:3], v0
global_store_b128 v0, v[0:3]
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
v_add_u32_e32 v0, v1, v2
s_endpgm
"""
    )

    report = compare_amdgpu_artifacts(
        target_text="hip -mcpu=gfx1100",
        tilelang=AmdgpuDifferentialArtifact(
            producer="tilelang",
            summary=tilelang_summary,
            code_object_path=Path("tilelang.co"),
            disassembly_path=Path("tilelang.disasm"),
            metadata={"oracle": "code-object"},
        ),
        loom=AmdgpuDifferentialArtifact(
            producer="loom",
            summary=loom_summary,
            code_object_path=Path("loom.co"),
            disassembly_path=Path("loom.disasm"),
            metadata={"compiler": "amdgpu-hal"},
        ),
    )

    deltas = {delta.family: delta for delta in report.family_deltas}
    assert deltas["global_load"].delta == 0
    assert deltas["global_store"].category == MEMORY_SHAPE
    assert deltas["s_load"].delta == 1
    assert deltas["s_load"].category == ABI_PROLOGUE
    assert deltas["s_waitcnt"].delta == 1
    assert deltas["s_waitcnt"].category == SCHEDULING_WAIT
    assert deltas["v_alu"].category == INSTRUCTION_SELECTION
    assert report.changed_categories == (
        ABI_PROLOGUE,
        INSTRUCTION_SELECTION,
        SCHEDULING_WAIT,
    )


def test_differential_report_metadata_is_json_serializable() -> None:
    summary = summarize_amdgpu_disassembly("s_endpgm\n")
    report = compare_amdgpu_artifacts(
        target_text="hip -mcpu=gfx1100",
        tilelang=AmdgpuDifferentialArtifact("tilelang", summary),
        loom=AmdgpuDifferentialArtifact("loom", summary),
    )

    encoded = json.dumps(dict(report.metadata()), sort_keys=True)
    assert '"changed_categories": []' in encoded
    assert '"target": "hip -mcpu=gfx1100"' in encoded


def test_classify_amdgpu_instruction_family_has_stable_review_buckets() -> None:
    assert classify_amdgpu_instruction_family("global_load") == MEMORY_SHAPE
    assert classify_amdgpu_instruction_family("buffer_store") == MEMORY_SHAPE
    assert classify_amdgpu_instruction_family("s_waitcnt") == SCHEDULING_WAIT
    assert classify_amdgpu_instruction_family("v_mfma") == INSTRUCTION_SELECTION
    assert classify_amdgpu_instruction_family("s_load") == ABI_PROLOGUE


def test_capture_loom_amdgpu_artifact_runs_production_compiler(
    tmp_path: Path,
) -> None:
    loom_compile = tmp_path / "loom-compile"
    loom_compile.write_text("fake loom-compile", encoding="utf-8")
    llvm_objdump = tmp_path / "llvm-objdump"
    llvm_objdump.write_text("fake llvm-objdump", encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command == (str(llvm_objdump), "--version"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="LLVM (http://llvm.org/):\n  LLVM version 22.1.3\n",
                stderr="",
            )
        if command[0] == str(loom_compile):
            _write_arg_path(command, "--output=", b"vmfb")
            _write_arg_path(command, "--emit-target-artifact=", b"hsaco")
            _write_arg_path(
                command,
                "--compile-report-output=",
                json.dumps(
                    {
                        "backend": "amdgpu-hal",
                        "target_key": "gfx1100",
                        "status": "OK",
                    }
                ).encode(),
            )
            _write_arg_path(
                command,
                "--emit-artifact-manifest=",
                json.dumps(
                    {
                        "kind": "loom.artifact_manifest",
                        "artifact": {"format": "elf", "byte_length": 5},
                    }
                ).encode(),
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == str(llvm_objdump):
            assert command[1:3] == ("-d", "--no-show-raw-insn")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "s_load_b128 s[0:3], s[0:1], null\n"
                    "global_store_b32 v0, v1, off\n"
                    "s_waitcnt vmcnt(0)\n"
                    "s_endpgm\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    artifact = capture_loom_amdgpu_artifact(
        "kernel.def @copy() {}\n",
        target_text="gfx1100",
        output_directory=tmp_path / "out",
        stem="copy",
        toolchain=LoomAmdgpuToolchain(loom_compile, llvm_objdump),
        runner=runner,
    )

    compile_command = commands[0]
    assert compile_command[:3] == (
        str(loom_compile),
        str(tmp_path / "out" / "copy.gfx1100.loom"),
        "--backend=amdgpu-hal",
    )
    assert "--target=gfx1100" in compile_command
    assert "--compile-report=summary" in compile_command
    assert "--artifact-manifest=summary" in compile_command
    assert commands[1] == (
        str(llvm_objdump),
        "-d",
        "--no-show-raw-insn",
        str(tmp_path / "out" / "copy.gfx1100.hsaco"),
    )
    assert commands[2] == (str(llvm_objdump), "--version")

    assert artifact.producer == "loom"
    assert artifact.code_object_path == tmp_path / "out" / "copy.gfx1100.hsaco"
    assert artifact.disassembly_path == tmp_path / "out" / "copy.gfx1100.disasm"
    assert artifact.summary.family_counts["global_store"] == 1
    assert artifact.summary.family_counts["s_load"] == 1
    metadata = dict(artifact.metadata)
    assert metadata["backend"] == "amdgpu-hal"
    assert metadata["target"] == "gfx1100"
    assert dict(metadata["compile_report"])["target_key"] == "gfx1100"
    assert dict(metadata["artifact_manifest"])["kind"] == "loom.artifact_manifest"
    assert dict(metadata["llvm_objdump"])["version"] == "LLVM version 22.1.3"
    assert (
        (tmp_path / "out" / "copy.gfx1100.disasm")
        .read_text(encoding="utf-8")
        .startswith("s_load_b128")
    )
    json.loads((tmp_path / "out" / "copy.gfx1100.metadata.json").read_text())
    json.loads((tmp_path / "out" / "copy.gfx1100.instructions.json").read_text())


def test_capture_loom_amdgpu_artifact_reports_missing_tool(
    tmp_path: Path,
) -> None:
    try:
        capture_loom_amdgpu_artifact(
            "kernel.def @copy() {}\n",
            target_text="gfx1100",
            output_directory=tmp_path,
            stem="copy",
            toolchain=LoomAmdgpuToolchain(loom_compile=None, llvm_objdump=None),
        )
    except LoomAmdgpuArtifactUnavailable as exc:
        metadata = dict(exc.metadata())
        assert metadata["dependency"] == "loom-compile"
        assert metadata["target"] == "gfx1100"
    else:
        raise AssertionError("expected missing loom-compile to be unavailable")


def _write_arg_path(
    command: tuple[str, ...],
    prefix: str,
    content: bytes,
) -> None:
    for argument in command:
        if argument.startswith(prefix):
            Path(argument.removeprefix(prefix)).write_bytes(content)
            return
    raise AssertionError(f"missing argument prefix {prefix}: {command}")
