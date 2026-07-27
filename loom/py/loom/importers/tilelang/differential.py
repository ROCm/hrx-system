# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU differential reports for TileLang-vs-Loom artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from loom.tools.amdgpu_asm import (
    AmdgpuDisassemblySummary,
    summarize_amdgpu_disassembly,
)

ABI_PROLOGUE = "abi/prologue"
ALLOCATION_PRESSURE = "allocation/register-pressure"
INSTRUCTION_SELECTION = "instruction-selection"
LDS_TRAFFIC = "lds-traffic"
MEMORY_SHAPE = "memory-addressing"
SCHEDULING_WAIT = "scheduling/wait-planning"
TARGET_COVERAGE = "target-coverage"

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


class LoomAmdgpuArtifactError(RuntimeError):
    """Raised when Loom AMDGPU artifact capture cannot complete."""


class LoomAmdgpuArtifactUnavailable(LoomAmdgpuArtifactError):
    """Raised when an optional Loom differential dependency is unavailable."""

    def __init__(
        self,
        dependency: str,
        reason: str,
        *,
        target_text: str,
    ) -> None:
        super().__init__(reason)
        self.dependency: str = dependency
        self.reason: str = reason
        self.target_text: str = target_text

    def metadata(self) -> Mapping[str, object]:
        """Returns structured unavailable metadata for check results."""

        return MappingProxyType(
            {
                "status": "unavailable",
                "dependency": self.dependency,
                "reason": self.reason,
                "target": self.target_text,
            }
        )


@dataclass(frozen=True, slots=True)
class LoomAmdgpuToolchain:
    """External tools used to capture and disassemble Loom AMDGPU artifacts."""

    loom_compile: Path | None = None
    llvm_objdump: Path | None = None

    @classmethod
    def probe(
        cls,
        *,
        loom_compile: Path | None = None,
        llvm_objdump: Path | None = None,
    ) -> LoomAmdgpuToolchain:
        """Discovers optional Loom and LLVM tools from explicit paths or PATH."""

        return cls(
            loom_compile=loom_compile or _which("loom-compile"),
            llvm_objdump=llvm_objdump or _which("llvm-objdump"),
        )


@dataclass(frozen=True, slots=True)
class AmdgpuDifferentialArtifact:
    """One externally disassembled AMDGPU artifact in a differential report."""

    producer: str
    summary: AmdgpuDisassemblySummary
    code_object_path: Path | None = None
    disassembly_path: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def report_metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable artifact summary."""

        metadata: dict[str, object] = {
            "producer": self.producer,
            "instruction_summary": dict(self.summary.metadata()),
            "metadata": dict(self.metadata),
        }
        if self.code_object_path is not None:
            metadata["code_object_path"] = str(self.code_object_path)
        if self.disassembly_path is not None:
            metadata["disassembly_path"] = str(self.disassembly_path)
        return MappingProxyType(metadata)


@dataclass(frozen=True, slots=True)
class AmdgpuInstructionFamilyDelta:
    """Instruction-family count delta between TileLang and Loom artifacts."""

    family: str
    tilelang_count: int
    loom_count: int
    category: str

    @property
    def delta(self) -> int:
        """Returns `loom_count - tilelang_count` for this family."""

        return self.loom_count - self.tilelang_count

    def metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable family-delta object."""

        return MappingProxyType(
            {
                "family": self.family,
                "category": self.category,
                "tilelang_count": self.tilelang_count,
                "loom_count": self.loom_count,
                "delta": self.delta,
            }
        )


@dataclass(frozen=True, slots=True)
class AmdgpuDifferentialReport:
    """Side-by-side TileLang and Loom AMDGPU disassembly comparison."""

    target_text: str
    tilelang: AmdgpuDifferentialArtifact
    loom: AmdgpuDifferentialArtifact
    family_deltas: tuple[AmdgpuInstructionFamilyDelta, ...]

    @property
    def changed_categories(self) -> tuple[str, ...]:
        """Returns stable mismatch categories present in the report."""

        return tuple(
            sorted({delta.category for delta in self.family_deltas if delta.delta != 0})
        )

    def metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable differential report."""

        return MappingProxyType(
            {
                "target": self.target_text,
                "tilelang": dict(self.tilelang.report_metadata()),
                "loom": dict(self.loom.report_metadata()),
                "family_deltas": [
                    dict(delta.metadata()) for delta in self.family_deltas
                ],
                "changed_categories": list(self.changed_categories),
            }
        )


def compare_amdgpu_artifacts(
    *,
    target_text: str,
    tilelang: AmdgpuDifferentialArtifact,
    loom: AmdgpuDifferentialArtifact,
) -> AmdgpuDifferentialReport:
    """Builds a compact TileLang-vs-Loom AMDGPU instruction-family report."""

    families = sorted(
        set(tilelang.summary.family_counts) | set(loom.summary.family_counts)
    )
    family_deltas = tuple(
        AmdgpuInstructionFamilyDelta(
            family=family,
            tilelang_count=int(tilelang.summary.family_counts.get(family, 0)),
            loom_count=int(loom.summary.family_counts.get(family, 0)),
            category=classify_amdgpu_instruction_family(family),
        )
        for family in families
    )
    return AmdgpuDifferentialReport(
        target_text=target_text,
        tilelang=tilelang,
        loom=loom,
        family_deltas=family_deltas,
    )


def capture_loom_amdgpu_artifact(
    loom_module_text: str,
    *,
    target_text: str,
    output_directory: Path,
    stem: str,
    toolchain: LoomAmdgpuToolchain | None = None,
    runner: CommandRunner | None = None,
) -> AmdgpuDifferentialArtifact:
    """Compiles Loom IR through AMDGPU HAL and disassembles the target artifact."""

    toolchain = toolchain or LoomAmdgpuToolchain.probe()
    loom_compile = _require_tool(
        toolchain.loom_compile,
        dependency="loom-compile",
        target_text=target_text,
    )
    llvm_objdump = _require_tool(
        toolchain.llvm_objdump,
        dependency="llvm-objdump",
        target_text=target_text,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    file_stem = f"{stem}.{_file_safe_target(target_text)}"
    source_path = output_directory / f"{file_stem}.loom"
    vmfb_path = output_directory / f"{file_stem}.vmfb"
    code_object_path = output_directory / f"{file_stem}.hsaco"
    compile_report_path = output_directory / f"{file_stem}.compile.json"
    manifest_path = output_directory / f"{file_stem}.manifest.json"
    disassembly_path = output_directory / f"{file_stem}.disasm"
    metadata_path = output_directory / f"{file_stem}.metadata.json"
    instruction_summary_path = output_directory / f"{file_stem}.instructions.json"

    source_path.write_text(loom_module_text, encoding="utf-8")
    _run_command(
        (
            str(loom_compile),
            str(source_path),
            "--backend=amdgpu-hal",
            f"--target={target_text}",
            f"--output={vmfb_path}",
            f"--emit-target-artifact={code_object_path}",
            "--compile-report=summary",
            f"--compile-report-output={compile_report_path}",
            "--artifact-manifest=summary",
            f"--emit-artifact-manifest={manifest_path}",
        ),
        runner=runner,
    )
    disassembly = _run_command(
        (
            str(llvm_objdump),
            "-d",
            "--no-show-raw-insn",
            str(code_object_path),
        ),
        runner=runner,
    ).stdout
    disassembly_path.write_text(disassembly or "", encoding="utf-8")
    summary = summarize_amdgpu_disassembly(disassembly or "")
    metadata = {
        "backend": "amdgpu-hal",
        "target": target_text,
        "source_path": str(source_path),
        "vmfb_path": str(vmfb_path),
        "compile_report_path": str(compile_report_path),
        "artifact_manifest_path": str(manifest_path),
        "loom_compile": dict(_tool_identity(loom_compile)),
        "llvm_objdump": dict(
            _tool_identity(
                llvm_objdump,
                version_command=(str(llvm_objdump), "--version"),
                runner=runner,
            )
        ),
        "compile_report": dict(_read_json_mapping(compile_report_path)),
        "artifact_manifest": dict(_read_json_mapping(manifest_path)),
    }
    artifact = AmdgpuDifferentialArtifact(
        producer="loom",
        summary=summary,
        code_object_path=code_object_path,
        disassembly_path=disassembly_path,
        metadata=metadata,
    )
    metadata_path.write_text(
        json.dumps(dict(artifact.report_metadata()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    instruction_summary_path.write_text(
        json.dumps(dict(summary.metadata()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def classify_amdgpu_instruction_family(family: str) -> str:
    """Classifies an AMDGPU instruction family into a review bucket."""

    if family in {
        "global_load",
        "global_store",
        "global_atomic",
        "buffer_load",
        "buffer_store",
        "buffer_atomic",
        "flat_load",
        "flat_store",
        "flat_atomic",
    }:
        return MEMORY_SHAPE
    if family in {"ds_read", "ds_write", "ds_other"}:
        return LDS_TRAFFIC
    if family in {"s_waitcnt", "s_barrier"}:
        return SCHEDULING_WAIT
    if family in {"v_mfma", "v_wmma", "v_smfmac", "v_dot", "v_alu"}:
        return INSTRUCTION_SELECTION
    if family in {"s_load", "s_alu", "s_branch", "s_endpgm"}:
        return ABI_PROLOGUE
    if "scratch" in family or "spill" in family:
        return ALLOCATION_PRESSURE
    return TARGET_COVERAGE


def _require_tool(
    path: Path | None,
    *,
    dependency: str,
    target_text: str,
) -> Path:
    if path is None:
        raise LoomAmdgpuArtifactUnavailable(
            dependency,
            f"{dependency} is not available",
            target_text=target_text,
        )
    if not path.exists():
        raise LoomAmdgpuArtifactUnavailable(
            dependency,
            f"{dependency} path does not exist: {path}",
            target_text=target_text,
        )
    return path


def _run_command(
    command: tuple[str, ...],
    *,
    runner: CommandRunner | None,
) -> subprocess.CompletedProcess[str]:
    if runner is None:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        completed = runner(command)
    if completed.returncode != 0:
        output = completed.stdout or completed.stderr or ""
        raise LoomAmdgpuArtifactError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{output}"
        )
    return completed


def _tool_identity(
    path: Path,
    *,
    version_command: tuple[str, ...] | None = None,
    runner: CommandRunner | None = None,
) -> Mapping[str, object]:
    identity: dict[str, object] = {"path": str(path)}
    if path.is_file():
        identity["sha256"] = _sha256_file(path)
    if version_command is not None:
        try:
            version_output = _run_command(version_command, runner=runner).stdout or ""
        except LoomAmdgpuArtifactError as exc:
            identity["version_probe_error"] = str(exc)
        else:
            identity["version"] = _version_line(version_output)
    return MappingProxyType(identity)


def _version_line(version_output: str) -> str:
    for line in version_output.splitlines():
        stripped = line.strip()
        if stripped and "version" in stripped.lower():
            return stripped
    return version_output.strip().splitlines()[0] if version_output.strip() else ""


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise LoomAmdgpuArtifactError(f"expected JSON object at {path}")
    return MappingProxyType(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_safe_target(target_text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", target_text).strip("_") or "target"


def _which(tool_name: str) -> Path | None:
    path = shutil.which(tool_name)
    return Path(path) if path is not None else None
