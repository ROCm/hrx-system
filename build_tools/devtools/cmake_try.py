# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CMake scratch-snippet support."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from build_tools.devtools import cmake_file_api
from build_tools.devtools.command_plan import CommandPlan, quote_command
from build_tools.devtools.environment import LOCAL_TMP_ROOT, REPO_ROOT, ToolEnvironment

CMAKE_TRY_ROOT = LOCAL_TMP_ROOT / "iree-cmake-try"
DEFAULT_TRY_BINARY_NAME = "snippet"
TRY_TARGET_NAME = "iree_cmake_try_snippet"
PROJECT_CACHE_PREFIXES = ("IREE_", "LIBHRX_", "HRX_")
PROJECT_CACHE_NAMES = {
    "BUILD_SHARED_LIBS",
    "BUILD_TESTING",
    "CMAKE_BUILD_TYPE",
    "CMAKE_CONFIGURATION_TYPES",
    "CMAKE_EXPORT_COMPILE_COMMANDS",
    "CMAKE_INSTALL_PREFIX",
    "CMAKE_PREFIX_PATH",
    "CMAKE_SYSROOT",
    "CMAKE_TOOLCHAIN_FILE",
    "Python3_EXECUTABLE",
}
PROJECT_CACHE_NAME_PATTERNS = (
    re.compile(r"^CMAKE_(?:C|CXX|ASM)_(?:COMPILER|COMPILER_LAUNCHER|FLAGS.*)$"),
    re.compile(r"^CMAKE_(?:EXE|SHARED|MODULE)_LINKER_FLAGS.*$"),
    re.compile(r"^CMAKE_OSX_.*$"),
)
CACHE_ENTRY_PATTERN = re.compile(r"^([^:#=]+):([^=]+)=(.*)$")
CMAKE_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CMAKE_HEADER_ROOTS: tuple[tuple[str, Path, str], ...] = (
    ("iree/", REPO_ROOT / "runtime/src", "iree"),
    ("hrx_", REPO_ROOT / "libhrx/include", "hrx"),
)


@dataclass(frozen=True)
class CMakeCacheEntry:
    name: str
    type: str
    value: str


@dataclass(frozen=True)
class CMakeTryCommand:
    files: list[Path] = field(default_factory=list)
    inline_sources: list[str] = field(default_factory=list)
    language: str | None = None
    compile_only: bool = False
    output: Path | None = None
    explicit_deps: list[str] = field(default_factory=list)
    infer_deps: bool = True
    keep: bool = False
    program_args: list[str] = field(default_factory=list)
    run_cwd: Path = field(default_factory=Path.cwd)


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator_index = arguments.index("--")
    return arguments[:separator_index], arguments[separator_index + 1 :]


def parse_try_args(
    arguments: list[str],
    *,
    run_cwd: Path | None = None,
) -> CMakeTryCommand:
    tool_args, program_args = split_program_args(arguments)
    files = []
    inline_sources = []
    explicit_deps = []
    language = None
    compile_only = False
    output = None
    infer_deps = True
    keep = False

    index = 0
    while index < len(tool_args):
        arg = tool_args[index]
        if arg in ("-e", "--execute"):
            index += 1
            if index >= len(tool_args):
                raise ValueError(f"{arg} requires source text")
            inline_sources.append(tool_args[index])
        elif arg.startswith("--execute="):
            inline_sources.append(arg.split("=", 1)[1])
        elif arg == "-x":
            index += 1
            if index >= len(tool_args):
                raise ValueError("-x requires c or c++")
            language = normalize_language(tool_args[index])
        elif arg.startswith("-x="):
            language = normalize_language(arg.split("=", 1)[1])
        elif arg in ("-c", "--compile-only", "--compile_only"):
            compile_only = True
        elif arg in ("-o", "--output"):
            index += 1
            if index >= len(tool_args):
                raise ValueError(f"{arg} requires a path")
            output = Path(tool_args[index])
        elif arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1])
        elif arg == "--dep":
            index += 1
            if index >= len(tool_args):
                raise ValueError("--dep requires a CMake target")
            explicit_deps.append(tool_args[index])
        elif arg.startswith("--dep="):
            explicit_deps.append(arg.split("=", 1)[1])
        elif arg in ("--no-infer", "--no_infer"):
            infer_deps = False
        elif arg in ("-k", "--keep"):
            keep = True
        elif arg.startswith("-"):
            raise ValueError(f"unknown cmake try option: {arg}")
        else:
            files.append(Path(arg))
        index += 1

    return CMakeTryCommand(
        files=files,
        inline_sources=inline_sources,
        language=language,
        compile_only=compile_only,
        output=output,
        explicit_deps=explicit_deps,
        infer_deps=infer_deps,
        keep=keep,
        program_args=program_args,
        run_cwd=run_cwd or Path.cwd(),
    )


def normalize_language(value: str) -> str:
    if value in ("c", "C"):
        return "c"
    if value in ("c++", "cc", "cpp", "cxx", "C++"):
        return "c++"
    raise ValueError(f"-x expects c or c++, got {value!r}")


def try_plan(
    tool_env: ToolEnvironment,
    *,
    configured_build_dir: Path,
    backend_args: list[str],
    run_cwd: Path | None = None,
) -> CommandPlan:
    command = parse_try_args(backend_args, run_cwd=run_cwd)
    return CommandPlan(
        [
            CMakeTryStep(
                tool_env.tool("cmake"),
                command,
                configured_build_dir,
                env=tool_env.path_env(),
            )
        ]
    )


def run_quietly(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    verbose: bool,
) -> int:
    if verbose:
        print("dev.py:", quote_command(argv), flush=True)
        return subprocess.run(argv, cwd=cwd, env=env).returncode

    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0 and completed.stdout:
        print(completed.stdout.rstrip(), file=sys.stderr)
    return completed.returncode


@dataclass(frozen=True)
class CMakeTryStep:
    cmake: str
    command: CMakeTryCommand
    configured_build_dir: Path
    env: dict[str, str] | None = None

    def describe(self) -> str:
        scratch = CMAKE_TRY_ROOT / "run-<pid>"
        try:
            generator_args = cmake_generator_args(self.configured_build_dir)
        except cmake_file_api.FileApiError:
            generator_args = []
        lines = [
            f"# write {scratch}/try.cmake",
            quote_command(
                [
                    self.cmake,
                    "-S",
                    str(REPO_ROOT),
                    "-B",
                    str(scratch / "build"),
                    *generator_args,
                    "-C",
                    str(scratch / "cache.cmake"),
                    f"-DIREE_CMAKE_TRY_FILE={scratch / 'try.cmake'}",
                ]
            ),
            quote_command(
                [
                    self.cmake,
                    "--build",
                    str(scratch / "build"),
                    "--target",
                    TRY_TARGET_NAME,
                ]
            ),
        ]
        if self.command.compile_only:
            lines.append("# compile only")
        else:
            lines.append(
                "run " + quote_command(["<built snippet>", *self.command.program_args])
            )
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        scratch_dir = CMAKE_TRY_ROOT / f"run-{os.getpid()}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        source_dir = scratch_dir / "src"
        scratch_build_dir = scratch_dir / "build"
        scratch_dir.mkdir(parents=True)
        try:
            try:
                source_names, source_texts = self.materialize_sources(source_dir)
                deps = self.deps_for_sources(source_texts)
                write_try_cmake_file(
                    scratch_dir / "try.cmake",
                    source_paths=[source_dir / name for name in source_names],
                    deps=deps,
                )
                write_initial_cache_file(
                    scratch_dir / "cache.cmake",
                    load_cmake_cache(self.configured_build_dir),
                )
                cmake_file_api.codemodel_query_path(scratch_build_dir).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                cmake_file_api.codemodel_query_path(scratch_build_dir).write_text(
                    "",
                    encoding="utf-8",
                )
            except (FileNotFoundError, ValueError, cmake_file_api.FileApiError) as exc:
                print(f"dev.py: {exc}", file=sys.stderr)
                return 2

            configure_result = run_quietly(
                [
                    self.cmake,
                    "-S",
                    str(REPO_ROOT),
                    "-B",
                    str(scratch_build_dir),
                    *cmake_generator_args(self.configured_build_dir),
                    "-C",
                    str(scratch_dir / "cache.cmake"),
                    f"-DIREE_CMAKE_TRY_FILE={scratch_dir / 'try.cmake'}",
                ],
                cwd=REPO_ROOT,
                env=self.env,
                verbose=verbose,
            )
            if configure_result != 0:
                return configure_result
            build_result = run_quietly(
                [
                    self.cmake,
                    "--build",
                    str(scratch_build_dir),
                    "--target",
                    TRY_TARGET_NAME,
                ],
                cwd=REPO_ROOT,
                env=self.env,
                verbose=verbose,
            )
            if build_result != 0:
                return build_result
            try:
                target = cmake_file_api.resolve_executable(
                    scratch_build_dir,
                    TRY_TARGET_NAME,
                )
            except cmake_file_api.FileApiError as exc:
                print(f"dev.py: {exc}", file=sys.stderr)
                return 1
            if self.command.output is not None:
                output_path = self.command.output
                if not output_path.is_absolute():
                    output_path = self.command.run_cwd / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target.path, output_path)
                output_path.chmod(output_path.stat().st_mode | 0o755)
            if self.command.compile_only:
                return 0
            if not os.access(target.path, os.X_OK):
                print(
                    f"dev.py: built output is not executable: {target.path}",
                    file=sys.stderr,
                )
                return 1
            return subprocess.run(
                [str(target.path), *self.command.program_args],
                cwd=self.command.run_cwd,
                env=self.env,
            ).returncode
        finally:
            if not self.command.keep:
                shutil.rmtree(scratch_dir, ignore_errors=True)
                try:
                    CMAKE_TRY_ROOT.rmdir()
                except OSError:
                    pass

    def materialize_sources(self, scratch_dir: Path) -> tuple[list[str], list[str]]:
        source_names = []
        source_texts = []
        for index, source in enumerate(self.command.inline_sources):
            suffix = (
                "cc" if inline_source_is_cxx(self.command.language, source) else "c"
            )
            source_name = f"inline_{index}.{suffix}"
            (scratch_dir / source_name).parent.mkdir(parents=True, exist_ok=True)
            (scratch_dir / source_name).write_text(source + "\n", encoding="utf-8")
            source_names.append(source_name)
            source_texts.append(source)
        for file_index, source_path in enumerate(self.command.files):
            resolved_path = source_path
            if not resolved_path.is_absolute():
                resolved_path = self.command.run_cwd / resolved_path
            if not resolved_path.is_file():
                raise FileNotFoundError(f"source file not found: {source_path}")
            source_name = scratch_input_name(
                resolved_path,
                self.command.run_cwd,
                file_index,
                source_names,
            )
            scratch_source_path = scratch_dir / source_name
            scratch_source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_path, scratch_source_path)
            text = resolved_path.read_text(encoding="utf-8", errors="ignore")
            source_names.append(source_name)
            source_texts.append(text)
        if not source_names:
            if sys.stdin.isatty():
                raise ValueError("cmake try requires a file, -e source, or stdin")
            stdin_source = sys.stdin.read()
            if not stdin_source:
                raise ValueError("cmake try received empty stdin")
            suffix = (
                "cc"
                if inline_source_is_cxx(self.command.language, stdin_source)
                else "c"
            )
            source_name = f"stdin.{suffix}"
            (scratch_dir / source_name).write_text(stdin_source, encoding="utf-8")
            source_names.append(source_name)
            source_texts.append(stdin_source)
        return source_names, source_texts

    def deps_for_sources(self, source_texts: list[str]) -> list[str]:
        deps = list(dict.fromkeys(self.command.explicit_deps))
        if not self.command.infer_deps:
            return deps
        aliases = cmake_file_api.load_target_aliases(self.configured_build_dir)
        inferred_deps = [
            dep
            for header in sorted(extract_quoted_includes(source_texts))
            if (dep := infer_dep_for_header(header, aliases)) is not None
        ]
        return list(dict.fromkeys([*deps, *inferred_deps]))


def load_cmake_cache(build_dir: Path) -> list[CMakeCacheEntry]:
    cache_path = build_dir / "CMakeCache.txt"
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise cmake_file_api.FileApiError(
            f"CMake cache is missing; run iree-cmake-configure first: {cache_path}"
        ) from exc
    entries = []
    for line in lines:
        match = CACHE_ENTRY_PATTERN.match(line)
        if match is None:
            continue
        name, entry_type, value = match.groups()
        entries.append(CMakeCacheEntry(name=name, type=entry_type, value=value))
    return entries


def cmake_generator_args(build_dir: Path) -> list[str]:
    by_name = {entry.name: entry.value for entry in load_cmake_cache(build_dir)}
    args = []
    generator = by_name.get("CMAKE_GENERATOR")
    if generator:
        args.extend(["-G", generator])
    platform = by_name.get("CMAKE_GENERATOR_PLATFORM")
    if platform:
        args.extend(["-A", platform])
    toolset = by_name.get("CMAKE_GENERATOR_TOOLSET")
    if toolset:
        args.extend(["-T", toolset])
    return args


def write_initial_cache_file(path: Path, entries: list[CMakeCacheEntry]) -> None:
    lines = [
        "# Generated by iree-cmake-try from the configured build tree.",
        "",
    ]
    for entry in entries:
        if not should_preserve_cache_entry(entry):
            continue
        lines.append(
            "set("
            + entry.name
            + " "
            + cmake_bracket_quote(entry.value)
            + " CACHE "
            + entry.type
            + ' "Copied from configured build tree." FORCE)'
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def should_preserve_cache_entry(entry: CMakeCacheEntry) -> bool:
    if not CMAKE_VARIABLE_NAME_PATTERN.match(entry.name):
        return False
    if entry.name == "IREE_CMAKE_TRY_FILE":
        return False
    if entry.type in ("INTERNAL", "STATIC"):
        return False
    if entry.name in PROJECT_CACHE_NAMES:
        return True
    if entry.name.startswith(PROJECT_CACHE_PREFIXES):
        return True
    return any(pattern.match(entry.name) for pattern in PROJECT_CACHE_NAME_PATTERNS)


def cmake_bracket_quote(value: str) -> str:
    delimiter = ""
    while f"]{delimiter}]" in value:
        delimiter += "="
    return f"[{delimiter}[{value}]{delimiter}]"


def write_try_cmake_file(
    path: Path, *, source_paths: list[Path], deps: list[str]
) -> None:
    lines = [
        "# Generated by iree-cmake-try.",
        f"add_executable({TRY_TARGET_NAME}",
    ]
    for source_path in source_paths:
        lines.append(f"  {cmake_bracket_quote(str(source_path))}")
    lines.append(")")
    lines.append(
        f"set_target_properties({TRY_TARGET_NAME} "
        f"PROPERTIES OUTPUT_NAME {DEFAULT_TRY_BINARY_NAME})"
    )
    if deps:
        lines.append(f"target_link_libraries({TRY_TARGET_NAME} PRIVATE")
        for dep in deps:
            lines.append(f"  {cmake_bracket_quote(dep)}")
        lines.append(")")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def infer_language(source_names: list[str], source_texts: list[str]) -> str:
    if any(Path(name).suffix in (".cc", ".cpp", ".cxx", ".C") for name in source_names):
        return "c++"
    joined_source = "\n".join(source_texts)
    if (
        "iree/testing/gtest_harness.h" in joined_source
        or "iree/testing/gbenchmark_harness.h" in joined_source
        or re.search(r"\b(TEST|BENCHMARK)\s*\(", joined_source)
    ):
        return "c++"
    return "c"


def inline_source_is_cxx(language: str | None, source: str) -> bool:
    if language is not None:
        return language == "c++"
    return infer_language(["inline.c"], [source]) == "c++"


def extract_quoted_includes(source_texts: list[str]) -> set[str]:
    headers = set()
    include_pattern = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
    for source in source_texts:
        headers.update(include_pattern.findall(source))
    return headers


def scratch_input_name(
    resolved_path: Path, run_cwd: Path, file_index: int, existing_names: list[str]
) -> str:
    try:
        relative_path = resolved_path.resolve().relative_to(run_cwd.resolve())
    except ValueError:
        relative_path = Path("external") / str(file_index) / resolved_path.name

    if not relative_path.parts or any(part == ".." for part in relative_path.parts):
        relative_path = Path("external") / str(file_index) / resolved_path.name

    source_name = relative_path.as_posix()
    if source_name in existing_names:
        raise ValueError(f"multiple try inputs map to {source_name}")
    return source_name


def infer_dep_for_header(header: str, aliases: dict[str, str]) -> str | None:
    for prefix, root, namespace in CMAKE_HEADER_ROOTS:
        if not header.startswith(prefix):
            continue
        header_path = root / header
        if namespace == "hrx":
            return "hrx::hrx" if "hrx::hrx" in aliases else None
        if header_path.suffix:
            header_path = header_path.parent
        try:
            relative_path = header_path.relative_to(REPO_ROOT / "runtime/src/iree")
        except ValueError:
            return None
        parts = relative_path.parts
        while parts:
            scoped = "::".join((namespace, *parts))
            if scoped in aliases:
                return scoped
            parts = parts[:-1]
    return None
