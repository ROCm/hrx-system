# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang-generated source and code-object oracle capture."""

from __future__ import annotations

import ctypes
import importlib
import json
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, cast

from loom.importers.tilelang.model import (
    normalize_tilelang_input,
    resolve_tilelang_input,
)
from loom.tools.amdgpu_asm import (
    AmdgpuDisassemblySummary,
    summarize_amdgpu_disassembly,
)


class TileLangOracleError(RuntimeError):
    """Raised when TileLang oracle capture cannot complete."""


class TileLangOracleUnavailable(TileLangOracleError):
    """Raised when an optional TileLang oracle dependency is unavailable."""

    def __init__(
        self,
        dependency: str,
        reason: str,
        *,
        target_text: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.dependency: str = dependency
        self.reason: str = reason
        self.target_text: str | None = target_text

    def metadata(self) -> Mapping[str, object]:
        """Returns structured unavailable metadata for check results."""

        metadata: dict[str, object] = {
            "status": "unavailable",
            "dependency": self.dependency,
            "reason": self.reason,
        }
        if self.target_text is not None:
            metadata["target"] = self.target_text
        return metadata


@dataclass(frozen=True, slots=True)
class TileLangOracleToolchain:
    """Optional external tools used by TileLang oracle capture."""

    hipcc: Path | None = None
    clang_offload_bundler: Path | None = None
    llvm_objdump: Path | None = None
    loaded_rocm_libraries: tuple[Path, ...] = ()

    @classmethod
    def probe(cls) -> TileLangOracleToolchain:
        """Discovers optional ROCm/LLVM tools without importing TileLang."""

        loaded_libraries = preload_rocm_runtime()
        return cls(
            hipcc=_which("hipcc"),
            clang_offload_bundler=_find_clang_offload_bundler(loaded_libraries),
            llvm_objdump=_which("llvm-objdump"),
            loaded_rocm_libraries=loaded_libraries,
        )


@dataclass(frozen=True, slots=True)
class TileLangGeneratedSource:
    """Generated device source emitted by TileLang/TVM for one target."""

    target_text: str
    arch: str
    source: str
    source_summary: tuple[str, ...]
    tilelang_version: str
    tvm_version: str
    pass_config_keys: tuple[str, ...]
    loaded_rocm_libraries: tuple[Path, ...] = ()

    def metadata(self) -> Mapping[str, object]:
        """Returns stable metadata for artifact files and JSON reports."""

        return MappingProxyType(
            {
                "target": self.target_text,
                "arch": self.arch,
                "tilelang_version": self.tilelang_version,
                "tvm_version": self.tvm_version,
                "pass_config_keys": list(self.pass_config_keys),
                "loaded_rocm_libraries": [
                    str(path) for path in self.loaded_rocm_libraries
                ],
                "source_summary": list(self.source_summary),
            }
        )


@dataclass(frozen=True, slots=True)
class TileLangCodeObjectOracle:
    """Compiled TileLang code-object oracle plus external disassembly."""

    generated_source: TileLangGeneratedSource
    bundled_object_path: Path
    code_object_path: Path
    disassembly: str
    disassembly_summary: AmdgpuDisassemblySummary

    @property
    def instruction_counts(self) -> Mapping[str, int]:
        """Returns stable instruction-family counts for compatibility."""

        return self.disassembly_summary.family_counts

    def metadata(self) -> Mapping[str, object]:
        """Returns stable metadata for artifact files and JSON reports."""

        return MappingProxyType(
            {
                **self.generated_source.metadata(),
                "bundled_object_path": str(self.bundled_object_path),
                "code_object_path": str(self.code_object_path),
                "disassembly_bytes": len(self.disassembly.encode()),
                "instruction_counts": dict(self.instruction_counts),
                "instruction_summary": dict(self.disassembly_summary.metadata()),
            }
        )


@dataclass(frozen=True, slots=True)
class _TileLangModules:
    """TileLang modules needed for final source codegen."""

    tilelang: ModuleType
    tvm: ModuleType
    tvm_ffi: ModuleType
    lower: ModuleType
    target: ModuleType
    pass_config: ModuleType


def capture_generated_source(
    source: object,
    *,
    target_text: str | None = None,
    toolchain: TileLangOracleToolchain | None = None,
) -> TileLangGeneratedSource:
    """Captures TileLang's generated device source for an explicit target."""

    toolchain = toolchain or TileLangOracleToolchain.probe()
    normalized = normalize_tilelang_input(source)
    resolved_target_text = target_text or normalized.target
    if not resolved_target_text:
        raise TileLangOracleError(
            "TileLang oracle capture requires explicit target text"
        )
    arch = target_arch(resolved_target_text)
    modules = _import_tilelang_modules()
    _require_codegen_hook(modules, resolved_target_text)
    resolved = resolve_tilelang_input(normalized)
    pass_configs = _pass_configs(modules, normalized.source)
    try:
        target = _target_object(modules, resolved_target_text)
    except Exception as exc:
        raise TileLangOracleUnavailable(
            "tilelang-target",
            f"TileLang target construction failed: {type(exc).__name__}: {exc}",
            target_text=resolved_target_text,
        ) from exc
    target_host = modules.tvm.target.Target("c")
    with (
        modules.tvm.transform.PassContext(
            opt_level=3,
            config=pass_configs,
        ),
        target,
    ):
        artifact = modules.lower.lower(
            resolved.source,
            target=target,
            target_host=target_host,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source_text = str(artifact.kernel_source)
    return TileLangGeneratedSource(
        target_text=resolved_target_text,
        arch=arch,
        source=source_text,
        source_summary=summarize_source(source_text),
        tilelang_version=str(getattr(modules.tilelang, "__version__", "<unknown>")),
        tvm_version=str(getattr(modules.tvm, "__version__", "<unknown>")),
        pass_config_keys=tuple(sorted(str(key) for key in pass_configs)),
        loaded_rocm_libraries=toolchain.loaded_rocm_libraries,
    )


def compile_hip_code_object(
    generated_source: TileLangGeneratedSource,
    *,
    output_directory: Path,
    stem: str,
    toolchain: TileLangOracleToolchain | None = None,
) -> TileLangCodeObjectOracle:
    """Compiles generated HIP source and disassembles the unbundled code object."""

    toolchain = toolchain or TileLangOracleToolchain.probe()
    if toolchain.hipcc is None:
        raise TileLangOracleUnavailable(
            "hipcc",
            "hipcc is not available",
            target_text=generated_source.target_text,
        )
    if toolchain.clang_offload_bundler is None:
        raise TileLangOracleUnavailable(
            "clang-offload-bundler",
            "clang-offload-bundler is not available",
            target_text=generated_source.target_text,
        )
    if toolchain.llvm_objdump is None:
        raise TileLangOracleUnavailable(
            "llvm-objdump",
            "llvm-objdump is not available",
            target_text=generated_source.target_text,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    source_path = output_directory / f"{stem}.{generated_source.arch}.hip.cpp"
    source_path.write_text(generated_source.source, encoding="utf-8")

    bundled_object_path = output_directory / f"{stem}.{generated_source.arch}.hsaco"
    _compile_hip_source(
        generated_source.source,
        bundled_object_path,
        arch=generated_source.arch,
    )
    code_object_path = output_directory / f"{stem}.{generated_source.arch}.co"
    _unbundle_code_object(
        bundled_object_path,
        code_object_path,
        arch=generated_source.arch,
        bundler=toolchain.clang_offload_bundler,
    )
    disassembly = _disassemble_code_object(
        code_object_path,
        objdump=toolchain.llvm_objdump,
    )
    result = TileLangCodeObjectOracle(
        generated_source=generated_source,
        bundled_object_path=bundled_object_path,
        code_object_path=code_object_path,
        disassembly=disassembly,
        disassembly_summary=summarize_amdgpu_disassembly(disassembly),
    )
    metadata = dict(result.metadata())
    (output_directory / f"{stem}.{generated_source.arch}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / f"{stem}.{generated_source.arch}.disasm").write_text(
        disassembly,
        encoding="utf-8",
    )
    (output_directory / f"{stem}.{generated_source.arch}.instructions.json").write_text(
        json.dumps(
            dict(result.disassembly_summary.metadata()),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def target_arch(target_text: str) -> str:
    """Extracts the explicit AMDGPU architecture from TileLang target text."""

    for token in shlex.split(target_text):
        if token.startswith("-mcpu="):
            return token.split("=", 1)[1]
        if token.startswith("--offload-arch="):
            return token.split("=", 1)[1]
    raise TileLangOracleError(
        f"TileLang oracle target must specify -mcpu=<gfx>, got `{target_text}`"
    )


def summarize_source(source: str) -> tuple[str, ...]:
    """Returns compact, stable facts about generated source text."""

    lines = source.splitlines()
    summary = [
        f"source_bytes={len(source.encode())}",
        f"source_lines={len(lines)}",
        f"has_global={'__global__' in source}",
        f"has_mfma={'mfma' in source.lower()}",
        f"has_wmma={'wmma' in source.lower()}",
        f"has_waitcnt={'waitcnt' in source.lower()}",
    ]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("extern", "__global__")):
            summary.append(f"kernel_decl={stripped[:160]}")
            break
    return tuple(summary)


def summarize_disassembly(disassembly: str) -> Mapping[str, int]:
    """Counts instruction-family markers in external disassembly text."""

    return summarize_amdgpu_disassembly(disassembly).family_counts


def preload_rocm_runtime() -> tuple[Path, ...]:
    """Loads ROCm runtime libraries when discoverable from the HIP toolchain."""

    loaded: list[Path] = []
    for library_name in ("libhsa-runtime64.so", "libamdhip64.so"):
        for candidate in _rocm_library_candidates(library_name):
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            except OSError:
                continue
            loaded.append(candidate)
            break
    return tuple(loaded)


def _import_tilelang_modules() -> _TileLangModules:
    return _TileLangModules(
        tilelang=importlib.import_module("tilelang"),
        tvm=importlib.import_module("tvm"),
        tvm_ffi=importlib.import_module("tvm_ffi"),
        lower=importlib.import_module("tilelang.engine.lower"),
        target=importlib.import_module("tilelang.utils.target"),
        pass_config=importlib.import_module("tilelang.transform.pass_config"),
    )


def _pass_configs(modules: _TileLangModules, source: object) -> Mapping[str, Any]:
    raw_pass_configs = getattr(source, "pass_configs", None)
    if raw_pass_configs is None:
        raw_pass_configs = {}
    return cast(
        Mapping[str, Any],
        modules.pass_config.normalize_pass_configs(dict(raw_pass_configs)),
    )


def tilelang_target_config(target_text: str) -> str | Mapping[str, str]:
    """Returns a TileLang/TVM target config for the current target API."""

    if not target_text.startswith("hip"):
        return target_text
    for token in shlex.split(target_text):
        if token.startswith("-mcpu="):
            return {"kind": "hip", "mcpu": token.split("=", 1)[1]}
        if token.startswith("--offload-arch="):
            return {"kind": "hip", "mcpu": token.split("=", 1)[1]}
    return target_text


def _target_object(modules: _TileLangModules, target_text: str) -> Any:
    target = modules.target.determine_target(
        tilelang_target_config(target_text),
        return_object=True,
    )
    return modules.tvm.target.Target(target)


def _require_codegen_hook(modules: _TileLangModules, target_text: str) -> None:
    if not target_text.startswith("hip"):
        return
    required_hook = "target.build.tilelang_hip_without_compile"
    global_func_names = set(modules.tvm_ffi.registry.list_global_func_names())
    if required_hook not in global_func_names:
        raise TileLangOracleUnavailable(
            "tilelang-hip-source-codegen",
            f"TileLang HIP source codegen hook `{required_hook}` is not registered",
            target_text=target_text,
        )


def _compile_hip_source(source: str, path: Path, *, arch: str) -> None:
    hipcc = importlib.import_module("tilelang.contrib.hipcc")
    env = importlib.import_module("tilelang.env")
    options = [
        "-std=c++17",
        "-include",
        "__clang_hip_runtime_wrapper.h",
        "-I" + str(env.TILELANG_TEMPLATE_PATH),
        "-I" + str(env.COMPOSABLE_KERNEL_INCLUDE_DIR),
    ]
    include_directory = _rocm_include_directory()
    if include_directory is not None:
        options.append("-I" + str(include_directory))
    hipcc.compile_hip(
        source,
        target_format="hsaco",
        arch=arch,
        options=options,
        path_target=str(path),
        verbose=False,
    )


def _unbundle_code_object(
    bundled_object_path: Path,
    code_object_path: Path,
    *,
    arch: str,
    bundler: Path,
) -> None:
    subprocess.run(
        [
            str(bundler),
            "-unbundle",
            "-type=o",
            f"-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--{arch}",
            f"-input={bundled_object_path}",
            "-output=/dev/null",
            f"-output={code_object_path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _disassemble_code_object(path: Path, *, objdump: Path) -> str:
    completed = subprocess.run(
        [str(objdump), "-d", "--no-show-raw-insn", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _rocm_library_candidates(library_name: str) -> tuple[Path, ...]:
    hipcc_path = _which("hipcc")
    if hipcc_path is None:
        return ()
    prefix = hipcc_path.resolve().parent.parent
    candidates = [
        prefix / "lib" / library_name,
        prefix / "lib64" / library_name,
        prefix.parent / "lib" / library_name,
        prefix.parent / "lib64" / library_name,
    ]
    candidates.extend(
        prefix.glob(f"lib/python*/site-packages/_rocm_sdk_devel/lib/{library_name}")
    )
    candidates.extend(
        prefix.glob(f"lib/python*/site-packages/_rocm_sdk_core/lib/{library_name}")
    )
    return tuple(candidates)


def _rocm_include_directory() -> Path | None:
    for library_name in ("libhsa-runtime64.so", "libamdhip64.so"):
        for candidate in _rocm_library_candidates(library_name):
            include_directory = candidate.parent.parent / "include"
            if (include_directory / "hip" / "hip_runtime.h").exists():
                return include_directory
    return None


def _find_clang_offload_bundler(
    loaded_libraries: tuple[Path, ...],
) -> Path | None:
    for library_path in loaded_libraries:
        candidate = library_path.parent.parent / "lib" / "llvm" / "bin"
        candidate = candidate / "clang-offload-bundler"
        if candidate.exists():
            return candidate
    return _which("clang-offload-bundler")


def _which(tool_name: str) -> Path | None:
    path = shutil.which(tool_name)
    return Path(path) if path is not None else None
