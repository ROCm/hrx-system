# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Applies a loom-analyze refactor plan to produce smaller Loom sources.

loom-analyze finds duplicated operation sequences and reports, for every copy,
its exact source range and the values crossing its boundary. loom-reduce turns
each applicable finding into one `func.def inline` motif and replaces the copies
with calls. Sources are never edited in place: the source tree is copied to an
output directory and only the copy is rewritten.

With --scope=file, a motif is defined in the same file as its copies, so no
declarations or link inputs change. Every rewritten file is parsed and verified
with loom-analyze; a file that fails keeps only the edits that verify on their
own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# Terminators end a block and can never move into a motif.
_TERMINATOR_PATTERN = re.compile(
    r"^(?:%[^=]*=\s*)?[a-z_][a-z0-9_]*\.(?:return|yield|launch\.config|br|cond_br|exit)\b"
)
_SSA_REFERENCE_PATTERN = re.compile(r"%([A-Za-z0-9_.$#]+)")
_DIAGNOSTIC_FILE_PATTERN = re.compile(r"^(?P<file>.+?):\d+:\d+: error")


class SourceReduceError(Exception):
    """A usage, input, or verification-tool failure."""


@dataclass(frozen=True)
class Binding:
    """One value crossing a copy boundary."""

    name: str
    type: str


@dataclass(frozen=True)
class Copy:
    """One occurrence of a finding that can be replaced by a call."""

    file: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    inputs: tuple[Binding, ...]
    outputs: tuple[Binding, ...]


@dataclass
class Job:
    """One motif defined in one file and the copies it replaces."""

    name: str
    file: str
    fingerprint: str
    op_count: int
    copies: list[Copy]
    status: str = "pending"
    reason: str = ""

    @property
    def removed_ops(self) -> int:
        # Each copy becomes one call; one motif body and its func.return are
        # added.
        count = len(self.copies)
        return max(0, count * self.op_count - (self.op_count + 1 + count))


@dataclass
class Edit:
    """Replaces [start, end) of a file with text."""

    start: int
    end: int
    text: str


@dataclass
class ReduceResult:
    """Everything loom-reduce did, for reporting."""

    jobs: list[Job] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------


def _bindings(entries: Sequence[dict]) -> tuple[Binding, ...]:
    return tuple(Binding(entry["name"], entry["type"]) for entry in entries)


def _load_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceReduceError(f"cannot read plan {path}: {error}") from error
    if plan.get("tool") != "loom-analyze" or "findings" not in plan:
        raise SourceReduceError(f"{path} is not a loom-analyze --format=json report")
    return plan


def _plan_jobs(plan: dict, result: ReduceResult) -> list[Job]:
    """Builds file-scope jobs from the selected, exactly matching findings."""

    jobs: list[Job] = []
    for finding in plan["findings"]:
        if not finding.get("selected"):
            continue
        fingerprint = finding["fingerprint"]
        if finding.get("parameterized_by"):
            result.skipped.append(
                {
                    "fingerprint": fingerprint,
                    "reason": "copies differ in constants or types",
                }
            )
            continue
        if finding["rule"].endswith("reimplemented-motif"):
            result.skipped.append(
                {
                    "fingerprint": fingerprint,
                    "reason": "calling an existing motif needs --scope=corpus",
                }
            )
            continue
        by_file: dict[str, list[Copy]] = {}
        for occurrence in finding["occurrences"]:
            if not occurrence.get("callable", True):
                continue
            copy = Copy(
                file=occurrence["file"],
                start_line=occurrence["start_line"],
                start_column=occurrence["start_column"],
                end_line=occurrence["end_line"],
                end_column=occurrence["end_column"],
                inputs=_bindings(occurrence["inputs"]),
                outputs=_bindings(occurrence["outputs"]),
            )
            by_file.setdefault(copy.file, []).append(copy)
        for file, copies in sorted(by_file.items()):
            if len(copies) < 2:
                continue
            file_hash = hashlib.sha1(
                file.replace("\\", "/").encode("utf-8")
            ).hexdigest()
            jobs.append(
                Job(
                    name=f"loom_reduce_{fingerprint[:12]}_{file_hash[:6]}",
                    file=file,
                    fingerprint=fingerprint,
                    op_count=finding["op_count"],
                    copies=copies,
                )
            )
    return jobs


# ---------------------------------------------------------------------------
# Source text
# ---------------------------------------------------------------------------


class SourceText:
    """One source file with (line, column) to offset conversion."""

    def __init__(self, text: str):
        self.text = text
        self._line_starts = [0]
        for index, char in enumerate(text):
            if char == "\n":
                self._line_starts.append(index + 1)

    def offset(self, line: int, column: int) -> int:
        """Converts one-based line and code-point column to a string offset."""

        if line < 1 or line > len(self._line_starts) or column < 1:
            raise SourceReduceError(f"location {line}:{column} is outside the source")
        return self._line_starts[line - 1] + column - 1

    def slice(self, copy: Copy) -> str:
        return self.text[
            self.offset(copy.start_line, copy.start_column) : self.offset(
                copy.end_line, copy.end_column
            )
        ]


def _validate_job(job: Job, sources: dict[str, SourceText]) -> str:
    """Returns why |job| cannot be applied, or an empty string."""

    template = job.copies[0]
    body = sources[job.file].slice(template)
    if "buffer.alloca" in body:
        return "moving buffer.alloca into a motif would allocate once per call"
    base_indent = template.start_column - 1
    top_level_lines = [body.splitlines()[0]] + [
        line[base_indent:]
        for line in body.splitlines()[1:]
        if len(line) - len(line.lstrip(" ")) == base_indent
    ]
    if any(_TERMINATOR_PATTERN.match(line.strip()) for line in top_level_lines):
        return "sequence contains a block terminator"
    for copy in job.copies:
        if len(copy.inputs) != len(template.inputs) or len(copy.outputs) != len(
            template.outputs
        ):
            return "copies disagree on their boundary values"
        if any(not binding.name for binding in copy.inputs + copy.outputs):
            return "a boundary value has no source name"
    # A parameter type may only mention earlier parameters; result types may
    # mention parameters.
    seen: set[str] = set()
    for binding in template.inputs:
        if any(
            name not in seen for name in _SSA_REFERENCE_PATTERN.findall(binding.type)
        ):
            return (
                "a parameter type depends on a value that is not an earlier parameter"
            )
        seen.add(binding.name.lstrip("%"))
    for binding in template.outputs:
        if any(
            name not in seen for name in _SSA_REFERENCE_PATTERN.findall(binding.type)
        ):
            return "a result type depends on a value defined inside the sequence"
    return ""


def _signature_types(bindings: Sequence[Binding]) -> str:
    return ", ".join(binding.type for binding in bindings)


def _call_text(job: Job, copy: Copy) -> str:
    arguments = ", ".join(binding.name for binding in copy.inputs)
    call = f"func.call @{job.name}({arguments}) : ({_signature_types(copy.inputs)})"
    if not copy.outputs:
        return call
    results = ", ".join(binding.name for binding in copy.outputs)
    return f"{results} = {call} -> ({_signature_types(copy.outputs)})"


def _motif_text(job: Job, source: SourceText) -> str:
    template = job.copies[0]
    body_lines = source.slice(template).split("\n")
    base_indent = template.start_column - 1
    indented = ["  " + body_lines[0]]
    for line in body_lines[1:]:
        stripped = (
            line[base_indent:] if line[:base_indent].strip() == "" else line.lstrip(" ")
        )
        indented.append("  " + stripped if stripped else "")
    parameters = ", ".join(
        f"{binding.name}: {binding.type}" for binding in template.inputs
    )
    header = f"func.def inline @{job.name}({parameters})"
    if template.outputs:
        header += f" -> ({_signature_types(template.outputs)})"
        names = ", ".join(binding.name for binding in template.outputs)
        terminator = f"  func.return {names} : {_signature_types(template.outputs)}"
    else:
        terminator = "  func.return"
    return (
        f"// Motif extracted by loom-reduce from {len(job.copies)} copies"
        f" ({job.op_count} ops each).\n"
        f"{header} {{\n" + "\n".join(indented) + f"\n{terminator}\n}}\n"
    )


def _rewrite(source: SourceText, jobs: Sequence[Job]) -> str:
    """Returns |source| with every copy of |jobs| replaced and motifs appended."""

    edits = [
        Edit(
            source.offset(copy.start_line, copy.start_column),
            source.offset(copy.end_line, copy.end_column),
            _call_text(job, copy),
        )
        for job in jobs
        for copy in job.copies
    ]
    edits.sort(key=lambda edit: edit.start, reverse=True)
    text = source.text
    previous_start = len(text) + 1
    for edit in edits:
        if edit.end > previous_start:
            raise SourceReduceError("overlapping copies in one file")
        text = text[: edit.start] + edit.text + text[edit.end :]
        previous_start = edit.start
    if jobs:
        text = text.rstrip("\n") + "\n\n// Motifs extracted by loom-reduce.\n\n"
        text += "\n".join(_motif_text(job, source) for job in jobs)
    return text


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

Verifier = Callable[[Sequence[Path]], set[Path]]


def make_loom_analyze_verifier(executable: str) -> Verifier:
    """Returns a verifier reporting files that fail to parse or verify."""

    def verify(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        failing = _run_verifier(executable, paths)
        if failing is not None:
            return failing
        # Unattributed failure: check files one by one.
        return {path for path in paths if _run_verifier(executable, [path]) != set()}

    return verify


def _run_verifier(executable: str, paths: Sequence[Path]) -> set[Path] | None:
    """Returns failing paths, or None when a failure could not be attributed."""

    # A huge minimum size disables clone reporting: only loading is checked.
    command = [executable, "--min-ops=1000000000", *[str(path) for path in paths]]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode in (0, 1):
        return set()
    if completed.returncode != 2:
        raise SourceReduceError(f"{executable} failed: {completed.stderr.strip()}")
    by_name = {str(path): path for path in paths}
    failing = set()
    for line in completed.stderr.splitlines():
        match = _DIAGNOSTIC_FILE_PATTERN.match(line)
        if match and match.group("file") in by_name:
            failing.add(by_name[match.group("file")])
    return failing or None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceReduceError(f"cannot read source file {path}: {error}") from error


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def reduce_sources(
    plan: dict,
    *,
    root: Path,
    output_dir: Path,
    verify: Verifier | None,
) -> ReduceResult:
    """Applies |plan| to a copy of |root| written to |output_dir|."""

    result = ReduceResult()
    jobs = _plan_jobs(plan, result)
    sources = {}
    for job in jobs:
        if job.file not in sources:
            sources[job.file] = SourceText(_read(root / job.file))

    ready: dict[str, list[Job]] = {}
    for job in jobs:
        reason = _validate_job(job, sources)
        if reason:
            job.status, job.reason = "skipped", reason
        else:
            ready.setdefault(job.file, []).append(job)
    result.jobs = jobs

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SourceReduceError(f"output directory {output_dir} is not empty")
    shutil.copytree(root, output_dir, dirs_exist_ok=True)

    for file, file_jobs in sorted(ready.items()):
        _write(output_dir / file, _rewrite(sources[file], file_jobs))
    changed = {file: output_dir / file for file in ready}

    if verify is not None:
        failing = verify(list(changed.values()))
        for file in sorted(ready):
            if changed[file] not in failing:
                continue
            # Keep the largest subset of edits that verifies, adding one job
            # at a time in order of savings.
            kept: list[Job] = []
            for job in sorted(ready[file], key=lambda job: -job.removed_ops):
                _write(changed[file], _rewrite(sources[file], [*kept, job]))
                if verify([changed[file]]):
                    job.status, job.reason = (
                        "reverted",
                        "rewritten file failed verification",
                    )
                else:
                    kept.append(job)
            _write(changed[file], _rewrite(sources[file], kept))
            ready[file] = kept

    for file, file_jobs in ready.items():
        for job in file_jobs:
            job.status = "applied"
        if file_jobs:
            result.changed_files.append(file)
        else:
            _write(changed[file], sources[file].text)
    result.changed_files.sort()
    return result


def _report(result: ReduceResult, plan: dict) -> dict:
    applied = [job for job in result.jobs if job.status == "applied"]
    removed = sum(job.removed_ops for job in applied)
    total = plan.get("summary", {}).get("ops", 0)
    return {
        "tool": "loom-reduce",
        "scope": "file",
        "summary": {
            "ops_before": total,
            "removed_ops": removed,
            "motifs": len(applied),
            "replaced_copies": sum(len(job.copies) for job in applied),
            "changed_files": len(result.changed_files),
            "skipped": sum(1 for job in result.jobs if job.status == "skipped")
            + len(result.skipped),
            "reverted": sum(1 for job in result.jobs if job.status == "reverted"),
        },
        "jobs": [
            {
                "motif": job.name,
                "file": job.file,
                "fingerprint": job.fingerprint,
                "op_count": job.op_count,
                "copies": len(job.copies),
                "removed_ops": job.removed_ops if job.status == "applied" else 0,
                "status": job.status,
                "reason": job.reason,
            }
            for job in result.jobs
        ],
        "skipped_findings": result.skipped,
        "changed_files": result.changed_files,
    }


def run(arguments: argparse.Namespace, *, stdout: TextIO, stderr: TextIO) -> int:
    """Runs loom-reduce and returns the public process exit code."""

    try:
        if arguments.scope != "file":
            raise SourceReduceError("--scope=corpus is not implemented yet")
        plan = _load_plan(arguments.plan)
        root = arguments.root
        if not root.is_dir():
            raise SourceReduceError(f"source root {root} is not a directory")
        verify = None
        if not arguments.no_verify:
            executable = arguments.loom_analyze or shutil.which("loom-analyze")
            if not executable:
                raise SourceReduceError(
                    "loom-analyze not found; pass --loom-analyze or --no-verify"
                )
            verify = make_loom_analyze_verifier(executable)
        result = reduce_sources(
            plan, root=root, output_dir=arguments.output_dir, verify=verify
        )
    except SourceReduceError as error:
        stderr.write(f"loom-reduce: error: {error}\n")
        return 2

    report = _report(result, plan)
    if arguments.report:
        _write(arguments.report, json.dumps(report, indent=1, sort_keys=True) + "\n")
    summary = report["summary"]
    total = summary["ops_before"]
    percent = 100.0 * summary["removed_ops"] / total if total else 0.0
    stdout.write(
        f"loom-reduce: {summary['motifs']} motif(s) replace "
        f"{summary['replaced_copies']} copies in {summary['changed_files']} "
        f"file(s); removes {summary['removed_ops']} of {total} ops "
        f"({percent:.1f}%)\n"
        f"loom-reduce: {summary['skipped']} skipped, {summary['reverted']} "
        "reverted after verification\n"
    )
    return 0


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-reduce",
        description=(
            "Applies a loom-analyze --format=json refactor plan: each selected "
            "duplicate sequence becomes one func.def inline motif and its copies "
            "become calls. Writes a rewritten copy of the source tree."
        ),
    )
    parser.add_argument(
        "--plan", type=Path, required=True, help="loom-analyze --format=json report."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help=(
            "Directory the plan's file paths are relative to (where "
            "loom-analyze ran). Default: the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty directory receiving a rewritten copy of --root.",
    )
    parser.add_argument(
        "--scope",
        choices=("file", "corpus"),
        default="file",
        help="file: define each motif beside its copies (no link changes).",
    )
    parser.add_argument(
        "--loom-analyze", help="loom-analyze executable used for verification."
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip parse/verify of rewritten files."
    )
    parser.add_argument(
        "--report", type=Path, help="Write a JSON report of applied and skipped work."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _create_argument_parser().parse_args(argv)
    return run(arguments, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
