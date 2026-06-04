# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel command helpers for dev.py."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from build_tools.devtools.command_plan import quote_command
from build_tools.devtools.environment import REPO_ROOT

BAZEL_TRY_ROOT = REPO_ROOT / ".iree-bazel-try"
DEFAULT_TRY_BINARY_NAME = "snippet"
FUZZ_CACHE_DIR = Path(os.environ.get("IREE_FUZZ_CACHE", "~/.cache/iree-fuzz-cache"))
HEADER_ROOTS: tuple[tuple[str, Path], ...] = (
    ("iree/", REPO_ROOT / "runtime/src"),
    ("loom/", REPO_ROOT / "loom/src"),
    ("loomc/", REPO_ROOT / "loom/binding/c/include"),
)
SPECIAL_HEADER_DEPS = {
    "iree/testing/gtest_harness.h": "//runtime/src/iree/testing:gtest_harness",
    "iree/testing/gbenchmark_harness.h": (
        "//runtime/src/iree/testing:gbenchmark_harness"
    ),
}


@dataclass(frozen=True)
class BazelTargetCommand:
    target: str
    bazel_args: list[str] = field(default_factory=list)
    program_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BazelRunCommand(BazelTargetCommand):
    print_path: bool = False
    run_cwd: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True)
class BazelTryCommand:
    files: list[Path] = field(default_factory=list)
    inline_sources: list[str] = field(default_factory=list)
    language: str | None = None
    compile_only: bool = False
    output: Path | None = None
    explicit_deps: list[str] = field(default_factory=list)
    infer_deps: bool = True
    keep: bool = False
    bazel_args: list[str] = field(default_factory=list)
    program_args: list[str] = field(default_factory=list)
    run_cwd: Path = field(default_factory=Path.cwd)


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator_index = arguments.index("--")
    return arguments[:separator_index], arguments[separator_index + 1 :]


def parse_bazel_run_args(
    arguments: list[str], *, run_cwd: Path | None = None
) -> BazelRunCommand:
    tool_args, program_args = split_program_args(arguments)
    target = ""
    bazel_args = []
    print_path = False
    for arg in tool_args:
        if arg in ("-p", "--print-path", "--print_path"):
            print_path = True
        elif arg.startswith("-"):
            bazel_args.append(arg)
        elif not target:
            target = arg
        else:
            bazel_args.append(arg)
    if not target:
        raise ValueError("Target is required for bazel run")
    return BazelRunCommand(
        target=target,
        bazel_args=bazel_args,
        program_args=program_args,
        print_path=print_path,
        run_cwd=run_cwd or Path.cwd(),
    )


def parse_bazel_fuzz_args(arguments: list[str]) -> BazelTargetCommand:
    tool_args, fuzzer_args = split_program_args(arguments)
    target = ""
    bazel_args = []
    for arg in tool_args:
        if arg.startswith("-"):
            bazel_args.append(arg)
        elif not target:
            target = arg
        else:
            bazel_args.append(arg)
    if not target:
        raise ValueError("Target is required for bazel fuzz")
    return BazelTargetCommand(
        target=target,
        bazel_args=bazel_args,
        program_args=fuzzer_args,
    )


def parse_bazel_try_args(
    arguments: list[str], *, run_cwd: Path | None = None
) -> BazelTryCommand:
    tool_args, program_args = split_program_args(arguments)
    files = []
    inline_sources = []
    explicit_deps = []
    bazel_args = []
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
                raise ValueError("--dep requires a Bazel label")
            explicit_deps.append(tool_args[index])
        elif arg.startswith("--dep="):
            explicit_deps.append(arg.split("=", 1)[1])
        elif arg in ("--no-infer", "--no_infer"):
            infer_deps = False
        elif arg in ("-k", "--keep"):
            keep = True
        elif arg.startswith("-"):
            bazel_args.append(arg)
        else:
            files.append(Path(arg))
        index += 1

    return BazelTryCommand(
        files=files,
        inline_sources=inline_sources,
        language=language,
        compile_only=compile_only,
        output=output,
        explicit_deps=explicit_deps,
        infer_deps=infer_deps,
        keep=keep,
        bazel_args=bazel_args,
        program_args=program_args,
        run_cwd=run_cwd or Path.cwd(),
    )


def normalize_language(value: str) -> str:
    if value in ("c", "C"):
        return "c"
    if value in ("c++", "cc", "cpp", "cxx", "C++"):
        return "c++"
    raise ValueError(f"-x expects c or c++, got {value!r}")


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


def run_captured(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def resolve_bazel_output_path(
    *,
    bazel: str,
    target: str,
    bazel_args: list[str],
    cwd: Path,
    env: dict[str, str] | None,
) -> Path | None:
    cquery = run_captured(
        [bazel, "cquery", "--output=files", *bazel_args, target],
        cwd=cwd,
        env=env,
    )
    if cquery.returncode != 0:
        print_process_failure(cquery)
        return None
    output_paths = [
        line.strip()
        for line in cquery.stdout.splitlines()
        if line.strip() and not line.startswith("INFO:")
    ]
    if not output_paths:
        return None

    execution_root = bazel_execution_root(bazel=bazel, cwd=cwd, env=env)
    candidates = []
    for output_path in output_paths:
        path = Path(output_path)
        if not path.is_absolute() and execution_root is not None:
            path = execution_root / path
        elif not path.is_absolute():
            path = cwd / path
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def bazel_execution_root(
    *, bazel: str, cwd: Path, env: dict[str, str] | None
) -> Path | None:
    completed = run_captured([bazel, "info", "execution_root"], cwd=cwd, env=env)
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            return Path(line)
    return None


def print_process_failure(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout.rstrip(), file=sys.stderr)
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)


def exec_path(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
) -> int:
    try:
        os.chdir(cwd)
        os.execvpe(argv[0], argv, env or os.environ)
    except OSError as exc:
        print(f"dev.py: failed to exec {quote_command(argv)}: {exc}", file=sys.stderr)
        return 127
    raise AssertionError("os.execvpe returned unexpectedly")


def cleanup_try_scratch(scratch_dir: Path) -> None:
    shutil.rmtree(scratch_dir, ignore_errors=True)
    try:
        BAZEL_TRY_ROOT.rmdir()
    except OSError:
        pass


def run_fuzzers(
    fuzzer_commands: list[tuple[str, list[str]]],
    *,
    env: dict[str, str] | None,
) -> int:
    processes = [
        (target, subprocess.Popen(argv, cwd=REPO_ROOT, env=env))
        for target, argv in fuzzer_commands
    ]
    result = 0
    try:
        while processes:
            for target, process in list(processes):
                process_result = process.poll()
                if process_result is None:
                    continue
                processes.remove((target, process))
                if process_result != 0 and result == 0:
                    print(
                        f"dev.py: fuzzer failed: {target} exited {process_result}",
                        file=sys.stderr,
                    )
                    result = process_exit_code(process_result)
            if result != 0:
                stop_fuzzers(processes)
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_fuzzers(processes)
        result = 130
    for _, process in processes:
        process.wait()
    return result


def process_exit_code(process_result: int) -> int:
    if process_result < 0:
        return 128 + abs(process_result)
    return process_result


def stop_fuzzers(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _, process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)


@dataclass(frozen=True)
class BazelRunStep:
    bazel: str
    command: BazelRunCommand
    env: dict[str, str] | None = None

    def describe(self) -> str:
        lines = [
            f"# bazel run {self.command.target}",
            quote_command(
                [
                    self.bazel,
                    "build",
                    *self.command.bazel_args,
                    self.command.target,
                ]
            ),
            quote_command(
                [
                    self.bazel,
                    "cquery",
                    "--output=files",
                    *self.command.bazel_args,
                    self.command.target,
                ]
            ),
        ]
        if self.command.print_path:
            lines.append("# print built executable path")
        else:
            lines.append(
                "exec "
                + quote_command(["<built executable>", *self.command.program_args])
            )
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        build_argv = [
            self.bazel,
            "build",
            *self.command.bazel_args,
            self.command.target,
        ]
        build_result = run_quietly(
            build_argv,
            cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
        )
        if build_result != 0:
            return build_result
        binary_path = resolve_bazel_output_path(
            bazel=self.bazel,
            target=self.command.target,
            bazel_args=self.command.bazel_args,
            cwd=REPO_ROOT,
            env=self.env,
        )
        if binary_path is None or not binary_path.is_file():
            print(
                f"dev.py: could not find built binary for {self.command.target}",
                file=sys.stderr,
            )
            return 1
        if self.command.print_path:
            print(binary_path)
            return 0
        if not os.access(binary_path, os.X_OK):
            print(
                f"dev.py: built output is not executable: {binary_path}",
                file=sys.stderr,
            )
            return 1
        return exec_path(
            [str(binary_path), *self.command.program_args],
            cwd=self.command.run_cwd,
            env=self.env,
        )


@dataclass(frozen=True)
class BazelFuzzStep:
    bazel: str
    command: BazelTargetCommand
    env: dict[str, str] | None = None

    @property
    def bazel_args(self) -> list[str]:
        return ["--config=fuzzer", *self.command.bazel_args]

    def describe(self) -> str:
        lines = [f"# bazel fuzz {self.command.target}"]
        if "..." in self.command.target:
            lines.append(quote_command([self.bazel, "query", self.command.target]))
            lines.append(
                quote_command([self.bazel, "build", *self.bazel_args, "<fuzz targets>"])
            )
            lines.append("run each built fuzzer without holding the Bazel lock")
        else:
            lines.append(
                quote_command(
                    [self.bazel, "build", *self.bazel_args, self.command.target]
                )
            )
            lines.append("exec " + quote_command(["<built fuzzer>", "<corpus>"]))
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        if "..." in self.command.target:
            return self.run_target_pattern(verbose=verbose)
        return self.build_and_run_target(
            self.command.target, exec_single=True, verbose=verbose
        )

    def run_target_pattern(self, *, verbose: bool) -> int:
        discovered = run_captured(
            [self.bazel, "query", self.command.target],
            cwd=REPO_ROOT,
            env=self.env,
        )
        if discovered.returncode != 0:
            print_process_failure(discovered)
            return discovered.returncode
        targets = sorted(
            line.strip()
            for line in discovered.stdout.splitlines()
            if line.strip().endswith("_fuzz")
        )
        if not targets:
            print(
                f"dev.py: no fuzz targets found under {self.command.target}",
                file=sys.stderr,
            )
            return 1
        build_result = run_quietly(
            [self.bazel, "build", *self.bazel_args, *targets],
            cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
        )
        if build_result != 0:
            return build_result
        fuzzer_commands = []
        for target in targets:
            fuzzer_argv = self.fuzzer_argv_for_target(target)
            if fuzzer_argv is None:
                return 1
            fuzzer_commands.append((target, fuzzer_argv))
        return run_fuzzers(fuzzer_commands, env=self.env)

    def build_and_run_target(
        self, target: str, *, exec_single: bool, verbose: bool
    ) -> int:
        build_result = run_quietly(
            [self.bazel, "build", *self.bazel_args, target],
            cwd=REPO_ROOT,
            env=self.env,
            verbose=verbose,
        )
        if build_result != 0:
            return build_result
        return self.run_built_target(target, exec_single=exec_single)

    def run_built_target(self, target: str, *, exec_single: bool) -> int:
        binary_path = resolve_bazel_output_path(
            bazel=self.bazel,
            target=target,
            bazel_args=self.bazel_args,
            cwd=REPO_ROOT,
            env=self.env,
        )
        if binary_path is None or not binary_path.is_file():
            print(f"dev.py: could not find built fuzzer for {target}", file=sys.stderr)
            return 1
        argv = self.fuzzer_argv(target, binary_path)
        if exec_single:
            return exec_path(argv, cwd=REPO_ROOT, env=self.env)
        return subprocess.run(argv, cwd=REPO_ROOT, env=self.env).returncode

    def fuzzer_argv_for_target(self, target: str) -> list[str] | None:
        binary_path = resolve_bazel_output_path(
            bazel=self.bazel,
            target=target,
            bazel_args=self.bazel_args,
            cwd=REPO_ROOT,
            env=self.env,
        )
        if binary_path is None or not binary_path.is_file():
            print(f"dev.py: could not find built fuzzer for {target}", file=sys.stderr)
            return None
        return self.fuzzer_argv(target, binary_path)

    def fuzzer_argv(self, target: str, binary_path: Path) -> list[str]:
        target_dir = fuzz_target_dir(target)
        corpus_dir = target_dir / "corpus"
        artifact_dir = target_dir / "artifacts"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return [
            str(binary_path),
            str(corpus_dir),
            f"-artifact_prefix={artifact_dir}/",
            *self.command.program_args,
        ]


@dataclass(frozen=True)
class BazelTryStep:
    bazel: str
    command: BazelTryCommand
    env: dict[str, str] | None = None

    def describe(self) -> str:
        scratch = BAZEL_TRY_ROOT / "run-<pid>"
        label = "//.iree-bazel-try/run-<pid>:snippet"
        lines = [
            f"# write {scratch}/BUILD.bazel",
            quote_command([self.bazel, "build", *self.command.bazel_args, label]),
        ]
        if self.command.compile_only:
            lines.append("# compile only")
        else:
            lines.append(
                "exec " + quote_command(["<built snippet>", *self.command.program_args])
            )
        return "\n".join(lines)

    def run(self, verbose: bool = False) -> int:
        scratch_dir = BAZEL_TRY_ROOT / f"run-{os.getpid()}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True)
        try:
            try:
                source_names, source_texts = self.materialize_sources(scratch_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(f"dev.py: {exc}", file=sys.stderr)
                return 2
            deps = list(dict.fromkeys(self.command.explicit_deps))
            if self.command.infer_deps:
                deps = list(dict.fromkeys([*deps, *self.infer_deps(source_texts)]))
            write_try_build_file(
                scratch_dir / "BUILD.bazel",
                source_names=source_names,
                deps=deps,
                testonly=True,
            )
            label = f"//.iree-bazel-try/{scratch_dir.name}:{DEFAULT_TRY_BINARY_NAME}"
            build_result = run_quietly(
                [self.bazel, "build", *self.command.bazel_args, label],
                cwd=REPO_ROOT,
                env=self.env,
                verbose=verbose,
            )
            if build_result != 0:
                return build_result
            binary_path = resolve_bazel_output_path(
                bazel=self.bazel,
                target=label,
                bazel_args=self.command.bazel_args,
                cwd=REPO_ROOT,
                env=self.env,
            )
            if binary_path is None or not binary_path.is_file():
                print("dev.py: could not find built snippet binary", file=sys.stderr)
                return 1
            if self.command.output is not None:
                output_path = self.command.output
                if not output_path.is_absolute():
                    output_path = self.command.run_cwd / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary_path, output_path)
                output_path.chmod(output_path.stat().st_mode | 0o755)
            if self.command.compile_only:
                return 0
            if not self.command.keep:
                cleanup_try_scratch(scratch_dir)
            return exec_path(
                [str(binary_path), *self.command.program_args],
                cwd=self.command.run_cwd,
                env=self.env,
            )
        finally:
            if not self.command.keep:
                cleanup_try_scratch(scratch_dir)

    def materialize_sources(self, scratch_dir: Path) -> tuple[list[str], list[str]]:
        source_names = []
        source_texts = []
        for index, source in enumerate(self.command.inline_sources):
            suffix = (
                "cc" if inline_source_is_cxx(self.command.language, source) else "c"
            )
            source_name = f"inline_{index}.{suffix}"
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
                resolved_path, self.command.run_cwd, file_index, source_names
            )
            scratch_source_path = scratch_dir / source_name
            scratch_source_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_path, scratch_source_path)
            text = resolved_path.read_text(encoding="utf-8", errors="ignore")
            source_names.append(source_name)
            source_texts.append(text)
        if not source_names:
            if sys.stdin.isatty():
                raise ValueError("bazel try requires a file, -e source, or stdin")
            stdin_source = sys.stdin.read()
            if not stdin_source:
                raise ValueError("bazel try received empty stdin")
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

    def infer_deps(self, source_texts: list[str]) -> list[str]:
        deps = []
        for header in sorted(extract_quoted_includes(source_texts)):
            dep = infer_dep_for_header(self.bazel, header, env=self.env)
            if dep is not None:
                deps.append(dep)
        return deps


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


def infer_dep_for_header(
    bazel: str, header: str, *, env: dict[str, str] | None
) -> str | None:
    special_dep = SPECIAL_HEADER_DEPS.get(header)
    if special_dep is not None:
        return special_dep

    header_path = header_path_for_include(header)
    if header_path is None:
        return None
    return infer_dep_for_header_path(bazel, header_path, env=env)


def header_path_for_include(header: str) -> Path | None:
    for prefix, root in HEADER_ROOTS:
        if header.startswith(prefix):
            return root / header
    return None


def infer_dep_for_header_path(
    bazel: str, header_path: Path, *, env: dict[str, str] | None
) -> str | None:
    package_dir = header_path.parent
    if not header_path.is_file():
        return fallback_dep_for_header_dir(package_dir)
    package_dir = nearest_package_dir(package_dir)
    if package_dir is None:
        return None
    package_label = "//" + package_dir.relative_to(REPO_ROOT).as_posix()
    header_label_path = re.escape(header_path.relative_to(package_dir).as_posix())
    query_expression = (
        f'kind(".* rule", attr("hdrs", "{header_label_path}", {package_label}:*))'
    )
    completed = run_captured([bazel, "query", query_expression], cwd=REPO_ROOT, env=env)
    if completed.returncode == 0:
        labels = sorted(
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        )
        if labels:
            return labels[0]
    return fallback_dep_for_header_dir(package_dir)


def nearest_package_dir(path: Path) -> Path | None:
    current = path
    while current != REPO_ROOT and current != current.parent:
        if (current / "BUILD.bazel").is_file():
            return current
        current = current.parent
    return None


def fallback_dep_for_header_dir(package_dir: Path) -> str | None:
    current = nearest_package_dir(package_dir)
    if current is not None:
        package = current.relative_to(REPO_ROOT).as_posix()
        target_name = current.name
        return f"//{package}:{target_name}"
    return None


def write_try_build_file(
    path: Path,
    *,
    source_names: list[str],
    deps: list[str],
    testonly: bool,
) -> None:
    lines = [
        'load("//build_tools/bazel:cc.bzl", "iree_cc_binary")',
        "",
        "iree_cc_binary(",
        f'    name = "{DEFAULT_TRY_BINARY_NAME}",',
        "    srcs = [",
    ]
    lines.extend(f'        "{source_name}",' for source_name in source_names)
    lines.append("    ],")
    if testonly:
        lines.append("    testonly = True,")
    if deps:
        lines.append("    deps = [")
        lines.extend(f'        "{dep}",' for dep in deps)
        lines.append("    ],")
    lines.append(")")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fuzz_target_dir(target: str) -> Path:
    package, target_name = split_target_label(target)
    return FUZZ_CACHE_DIR.expanduser() / package / target_name


def split_target_label(target: str) -> tuple[str, str]:
    label = target
    if label.startswith("@"):
        label = label.split("//", 1)[-1]
    elif label.startswith("//"):
        label = label[2:]
    if ":" in label:
        package, target_name = label.split(":", 1)
    else:
        package = label
        target_name = Path(package).name
    return package, target_name
