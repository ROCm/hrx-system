# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""JSON execution test runner for command-line tooling tests."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class ExecutionError(Exception):
    """Base class for execution test errors."""


class SchemaError(ExecutionError):
    """Raised when a manifest does not match the supported schema."""


class CaseFailure(ExecutionError):
    """Raised when a manifest case fails while executing."""


@dataclasses.dataclass(frozen=True)
class RunSummary:
    case_count: int = 0
    failed_count: int = 0

    def merge(self, other: "RunSummary") -> "RunSummary":
        return RunSummary(
            case_count=self.case_count + other.case_count,
            failed_count=self.failed_count + other.failed_count,
        )


def resolve_runfile(path: str) -> Path:
    """Resolves a Bazel runfile path or ordinary filesystem path."""
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    suffixes = ["", ".exe"] if os.name == "nt" else [""]
    workspace = os.environ.get("TEST_WORKSPACE")
    roots = [
        os.environ.get("RUNFILES_DIR"),
        os.environ.get("TEST_SRCDIR"),
        os.getcwd(),
    ]
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        prefixes = [root_path]
        if workspace:
            prefixes.insert(0, root_path / workspace)
        for prefix in prefixes:
            for suffix in suffixes:
                resolved = prefix / (path + suffix)
                if resolved.exists():
                    return resolved
    raise FileNotFoundError(path)


_SANITIZER_OPTION_ENV_NAMES = (
    "ASAN_OPTIONS",
    "LSAN_OPTIONS",
    "MSAN_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
)

_SANITIZER_RUNFILE_OPTIONS = frozenset({"suppressions"})


def _resolve_sanitizer_options(value: str) -> str:
    """Resolves runfile paths inside sanitizer option strings."""
    entries = value.split(os.pathsep)
    for index, entry in enumerate(entries):
        key, separator, option_value = entry.partition("=")
        if (
            separator
            and key in _SANITIZER_RUNFILE_OPTIONS
            and option_value
            and not Path(option_value).is_absolute()
        ):
            try:
                option_value = str(resolve_runfile(option_value))
            except FileNotFoundError:
                pass
            entries[index] = key + separator + option_value
    return os.pathsep.join(entries)


def _resolve_sanitizer_env(env: dict[str, str]) -> None:
    """Resolves sanitizer option runfiles before launching child tools."""
    for name in _SANITIZER_OPTION_ENV_NAMES:
        if name in env:
            env[name] = _resolve_sanitizer_options(env[name])


def _load_manifest(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}: invalid JSON: {exc}") from exc


def _as_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{location}: expected mapping, got {type(value).__name__}")
    return value


def _as_sequence(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{location}: expected list, got {type(value).__name__}")
    return value


def _as_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{location}: expected string, got {type(value).__name__}")
    return value


def _safe_case_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "case"


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _normalize_text(text: str, normalizers: list[Any] | None) -> str:
    normalized = text.replace("\r\n", "\n")
    for entry in normalizers or []:
        kind = entry
        if isinstance(entry, dict):
            kind = entry.get("kind")
        if kind == "line_endings":
            normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        elif kind == "path_separators":
            normalized = normalized.replace("\\", "/")
        elif kind == "regex":
            pattern = _as_string(entry.get("pattern"), "normalize.pattern")
            replacement = _as_string(
                entry.get("replacement", ""), "normalize.replacement"
            )
            normalized = re.sub(pattern, replacement, normalized)
        else:
            raise SchemaError(f"unknown normalizer {kind!r}")
    return normalized


class _Substituter:
    def __init__(self, values: dict[str, str], tools: dict[str, Path]):
        self._values = values
        self._tools = tools

    def substitute(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._substitute_string(value)
        if isinstance(value, list):
            return [self.substitute(item) for item in value]
        if isinstance(value, dict):
            return {key: self.substitute(item) for key, item in value.items()}
        return value

    def _substitute_string(self, value: str) -> str:
        result = value
        for key, replacement in self._values.items():
            result = result.replace("{" + key + "}", replacement)

        def replace_tool(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self._tools:
                raise SchemaError(f"unknown tool substitution {name!r}")
            return str(self._tools[name])

        return re.sub(r"\{tool:([^}]+)\}", replace_tool, result)


class ExecutionRunner:
    """Runs one or more JSON execution manifests."""

    def __init__(
        self,
        *,
        tools: dict[str, Path],
        case_filters: set[str] | None = None,
        keep_temp: bool = False,
        list_only: bool = False,
        output_stream: Any = sys.stderr,
    ):
        self._tools = tools
        self._case_filters = case_filters or set()
        self._keep_temp = keep_temp
        self._list_only = list_only
        self._output_stream = output_stream

    def run_manifest(self, manifest_path: Path) -> RunSummary:
        manifest_path = resolve_runfile(str(manifest_path))
        manifest = _as_mapping(_load_manifest(manifest_path), str(manifest_path))
        version = manifest.get("version")
        if version != 1:
            raise SchemaError(f"{manifest_path}: expected version: 1")
        case_values = _as_sequence(manifest.get("cases", []), f"{manifest_path}: cases")
        summary = RunSummary()
        for index, case in enumerate(case_values):
            case_mapping = _as_mapping(case, f"{manifest_path}: cases[{index}]")
            name = _as_string(
                case_mapping.get("name"), f"{manifest_path}: cases[{index}].name"
            )
            qualified_name = f"{manifest_path.name}:{name}"
            if (
                self._case_filters
                and name not in self._case_filters
                and qualified_name not in self._case_filters
            ):
                continue
            if self._list_only:
                print(qualified_name, file=self._output_stream)
                summary = summary.merge(RunSummary(case_count=1))
                continue
            try:
                self._run_case(manifest_path, case_mapping)
                summary = summary.merge(RunSummary(case_count=1))
            except CaseFailure as exc:
                print(str(exc), file=self._output_stream)
                summary = summary.merge(RunSummary(case_count=1, failed_count=1))
        return summary

    def _run_case(self, manifest_path: Path, case: dict[str, Any]) -> None:
        case_name = _as_string(case.get("name"), "case.name")
        temp_root = os.environ.get("TEST_TMPDIR")
        temp_dir = Path(
            tempfile.mkdtemp(prefix=_safe_case_name(case_name) + "-", dir=temp_root)
        )
        substituter = _Substituter(
            {
                "case": _safe_case_name(case_name),
                "manifest": str(manifest_path),
                "srcdir": str(manifest_path.parent),
                "tmp": str(temp_dir),
            },
            self._tools,
        )
        step_outputs: dict[str, subprocess.CompletedProcess[bytes]] = {}
        failed = False
        try:
            for index, step in enumerate(self._case_steps(case)):
                step_name = _as_string(
                    step.get("name", f"step-{index}"),
                    f"{case_name}: steps[{index}].name",
                )
                self._run_step(
                    case_name=case_name,
                    step_name=step_name,
                    step=substituter.substitute(step),
                    temp_dir=temp_dir,
                    step_outputs=step_outputs,
                )
        except CaseFailure:
            failed = True
            self._preserve_failed_temp(case_name, temp_dir)
            raise
        finally:
            if not self._keep_temp and not failed:
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif self._keep_temp:
                print(
                    f"kept temp directory for {case_name}: {temp_dir}",
                    file=self._output_stream,
                )

    def _case_steps(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        if "steps" in case:
            return [
                _as_mapping(step, "case.steps[]")
                for step in _as_sequence(case["steps"], "case.steps")
            ]
        if "run" in case:
            return [{key: value for key, value in case.items() if key not in ("name",)}]
        raise SchemaError(f"{case.get('name', '<unnamed>')}: expected 'run' or 'steps'")

    def _run_step(
        self,
        *,
        case_name: str,
        step_name: str,
        step: dict[str, Any],
        temp_dir: Path,
        step_outputs: dict[str, subprocess.CompletedProcess[bytes]],
    ) -> None:
        if "write" in step:
            self._run_write_step(
                case_name, step_name, _as_mapping(step["write"], f"{step_name}.write")
            )
            return
        if "run" not in step:
            raise SchemaError(f"{case_name}:{step_name}: expected 'run' or 'write'")
        run = _as_mapping(step["run"], f"{case_name}:{step_name}.run")
        tool_name = _as_string(run.get("tool"), f"{case_name}:{step_name}.run.tool")
        if tool_name not in self._tools:
            raise SchemaError(f"{case_name}:{step_name}: unknown tool {tool_name!r}")
        args = [
            _as_string(arg, f"{case_name}:{step_name}.run.args[]")
            for arg in _as_sequence(
                run.get("args", []), f"{case_name}:{step_name}.run.args"
            )
        ]
        argv = [str(self._tools[tool_name]), *args]
        stdin = self._stdin_bytes(
            case_name, step_name, run.get("stdin", step.get("stdin")), step_outputs
        )
        env = dict(os.environ)
        env.update(
            {
                key: _as_string(value, f"{case_name}:{step_name}.env[{key!r}]")
                for key, value in _as_mapping(
                    step.get("env", {}), f"{case_name}:{step_name}.env"
                ).items()
            }
        )
        _resolve_sanitizer_env(env)
        cwd = Path(
            _as_string(step.get("cwd", str(temp_dir)), f"{case_name}:{step_name}.cwd")
        )
        try:
            completed = subprocess.run(
                argv,
                check=False,
                cwd=cwd,
                env=env,
                input=stdin,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        except OSError as exc:
            raise CaseFailure(
                f"{case_name}:{step_name}: failed to launch command: {exc}\n"
                "argv:\n"
                f"  {' '.join(argv)}\n"
                f"cwd: {cwd}"
            ) from exc
        self._check_exit(case_name, step_name, argv, completed, step.get("exit", 0))
        self._check_stream(
            case_name, step_name, argv, "stdout", completed.stdout, step.get("stdout")
        )
        self._check_stream(
            case_name, step_name, argv, "stderr", completed.stderr, step.get("stderr")
        )
        self._check_files(case_name, step_name, step.get("files", []))
        step_outputs[step_name] = completed

    def _run_write_step(
        self, case_name: str, step_name: str, write: dict[str, Any]
    ) -> None:
        path = Path(
            _as_string(write.get("path"), f"{case_name}:{step_name}.write.path")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if "text" in write:
            path.write_text(
                _as_string(write["text"], f"{case_name}:{step_name}.write.text"),
                encoding="utf-8",
            )
        elif "bytes" in write:
            path.write_bytes(
                bytes(
                    _as_sequence(write["bytes"], f"{case_name}:{step_name}.write.bytes")
                )
            )
        else:
            raise SchemaError(
                f"{case_name}:{step_name}.write: expected 'text' or 'bytes'"
            )

    def _stdin_bytes(
        self,
        case_name: str,
        step_name: str,
        stdin: Any,
        step_outputs: dict[str, subprocess.CompletedProcess[bytes]],
    ) -> bytes | None:
        if stdin is None:
            return None
        spec = _as_mapping(stdin, f"{case_name}:{step_name}.stdin")
        if "text" in spec:
            return _as_string(
                spec["text"], f"{case_name}:{step_name}.stdin.text"
            ).encode("utf-8")
        if "file" in spec:
            return Path(
                _as_string(spec["file"], f"{case_name}:{step_name}.stdin.file")
            ).read_bytes()
        if "step_stdout" in spec:
            source = _as_string(
                spec["step_stdout"], f"{case_name}:{step_name}.stdin.step_stdout"
            )
            if source not in step_outputs:
                raise SchemaError(
                    f"{case_name}:{step_name}: unknown stdout source step {source!r}"
                )
            return step_outputs[source].stdout
        raise SchemaError(
            f"{case_name}:{step_name}.stdin: expected 'text', 'file', or 'step_stdout'"
        )

    def _check_exit(
        self,
        case_name: str,
        step_name: str,
        argv: list[str],
        completed: subprocess.CompletedProcess[bytes],
        expected: Any,
    ) -> None:
        if isinstance(expected, int):
            if completed.returncode == expected:
                return
            self._fail_command(
                case_name, step_name, argv, completed, f"expected exit {expected}"
            )
        spec = _as_mapping(expected, f"{case_name}:{step_name}.exit")
        if spec.get("nonzero") is True and completed.returncode != 0:
            return
        if "code" in spec and completed.returncode == spec["code"]:
            return
        self._fail_command(
            case_name,
            step_name,
            argv,
            completed,
            f"unexpected exit {completed.returncode}",
        )

    def _check_stream(
        self,
        case_name: str,
        step_name: str,
        argv: list[str],
        stream_name: str,
        raw: bytes,
        spec_value: Any,
    ) -> None:
        if spec_value is None:
            return
        spec = _as_mapping(spec_value, f"{case_name}:{step_name}.{stream_name}")
        if "save" in spec:
            save_path = Path(
                _as_string(spec["save"], f"{case_name}:{step_name}.{stream_name}.save")
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(raw)
        text = _normalize_text(_decode_text(raw), spec.get("normalize"))
        if spec.get("empty") is True and text:
            self._fail_command(
                case_name,
                step_name,
                argv,
                None,
                f"expected empty {stream_name}, got:\n{text}",
            )
        if "contains" in spec:
            self._check_contains(
                case_name, step_name, stream_name, text, spec["contains"]
            )
        if "not_contains" in spec:
            self._check_not_contains(
                case_name, step_name, stream_name, text, spec["not_contains"]
            )

    def _check_contains(
        self,
        case_name: str,
        step_name: str,
        stream_name: str,
        text: str,
        value: Any,
    ) -> None:
        ordered, entries = self._contains_entries(
            value, f"{case_name}:{step_name}.{stream_name}.contains"
        )
        position = 0
        for entry in entries:
            match = self._find_match(text, entry, position if ordered else 0)
            if match is None:
                raise CaseFailure(
                    f"{case_name}:{step_name}: missing {stream_name} pattern "
                    f"{entry!r}\n{stream_name}:\n{text}"
                )
            if ordered:
                position = match

    def _check_not_contains(
        self,
        case_name: str,
        step_name: str,
        stream_name: str,
        text: str,
        value: Any,
    ) -> None:
        _, entries = self._contains_entries(
            value, f"{case_name}:{step_name}.{stream_name}.not_contains"
        )
        for entry in entries:
            if self._find_match(text, entry, 0) is not None:
                raise CaseFailure(
                    f"{case_name}:{step_name}: forbidden {stream_name} pattern "
                    f"{entry!r}\n{stream_name}:\n{text}"
                )

    def _contains_entries(self, value: Any, location: str) -> tuple[bool, list[Any]]:
        if isinstance(value, list):
            return True, value
        mapping = _as_mapping(value, location)
        if "ordered" in mapping:
            return True, _as_sequence(mapping["ordered"], location + ".ordered")
        if "unordered" in mapping:
            return False, _as_sequence(mapping["unordered"], location + ".unordered")
        raise SchemaError(f"{location}: expected list, ordered, or unordered")

    def _find_match(self, text: str, entry: Any, position: int) -> int | None:
        if isinstance(entry, str):
            index = text.find(entry, position)
            return None if index == -1 else index + len(entry)
        mapping = _as_mapping(entry, "contains[]")
        if "regex" not in mapping:
            raise SchemaError("contains[] mapping expected 'regex'")
        match = re.search(
            _as_string(mapping["regex"], "contains[].regex"), text[position:]
        )
        if not match:
            return None
        return position + match.end()

    def _check_files(self, case_name: str, step_name: str, value: Any) -> None:
        for index, entry in enumerate(
            _as_sequence(value, f"{case_name}:{step_name}.files")
        ):
            spec = _as_mapping(entry, f"{case_name}:{step_name}.files[{index}]")
            path = Path(
                _as_string(
                    spec.get("path"), f"{case_name}:{step_name}.files[{index}].path"
                )
            )
            if spec.get("exists", True) and not path.exists():
                raise CaseFailure(
                    f"{case_name}:{step_name}: expected file {path} to exist"
                )
            if spec.get("non_empty") is True and path.stat().st_size == 0:
                raise CaseFailure(
                    f"{case_name}:{step_name}: expected file {path} to be non-empty"
                )

    def _fail_command(
        self,
        case_name: str,
        step_name: str,
        argv: list[str],
        completed: subprocess.CompletedProcess[bytes] | None,
        reason: str,
    ) -> None:
        message = [
            f"{case_name}:{step_name}: {reason}",
            "argv:",
            "  " + " ".join(argv),
        ]
        if completed is not None:
            message.extend(
                [
                    f"exit: {completed.returncode}",
                    "stdout:",
                    _decode_text(completed.stdout),
                    "stderr:",
                    _decode_text(completed.stderr),
                ]
            )
        raise CaseFailure("\n".join(message))

    def _preserve_failed_temp(self, case_name: str, temp_dir: Path) -> None:
        undeclared_outputs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if undeclared_outputs:
            destination = Path(undeclared_outputs) / (
                _safe_case_name(case_name) + "_temp"
            )
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(temp_dir, destination)
            print(
                f"preserved temp directory for {case_name}: {destination}",
                file=self._output_stream,
            )
        else:
            print(
                f"preserved temp directory for {case_name}: {temp_dir}",
                file=self._output_stream,
            )


def parse_tool_bindings(values: list[str]) -> dict[str, Path]:
    tools: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name:
            raise SchemaError(f"invalid --tool binding {value!r}; expected name=path")
        tools[name] = resolve_runfile(path)
    return tools


def run_from_args(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", action="append", default=[], help="JSON execution manifest."
    )
    parser.add_argument(
        "--tool", action="append", default=[], help="Tool binding as name=path."
    )
    parser.add_argument(
        "--case", action="append", default=[], help="Only run a case name."
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep per-case temporary directories."
    )
    parser.add_argument(
        "--list", action="store_true", help="List selected cases without running them."
    )
    args = parser.parse_args(argv)
    if not args.manifest:
        raise SchemaError("at least one --manifest is required")
    runner = ExecutionRunner(
        tools=parse_tool_bindings(args.tool),
        case_filters=set(args.case),
        keep_temp=args.keep_temp,
        list_only=args.list,
    )
    summary = RunSummary()
    for manifest in args.manifest:
        summary = summary.merge(runner.run_manifest(Path(manifest)))
    if args.list:
        return 0
    print(
        f"execution tests: {summary.case_count - summary.failed_count}/{summary.case_count} passed",
        file=sys.stderr,
    )
    return 1 if summary.failed_count else 0
