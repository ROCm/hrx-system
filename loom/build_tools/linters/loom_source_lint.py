#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks Loom source invariants that are easy to regress.

This is intentionally a small guardrail, not a general C parser. It catches the
module-value-cardinality storage pattern that is especially toxic for Loom's
function-local compiler phases: allocating or resetting scratch sized by
`module->values.count`/`capacity` instead of using function-local domains,
maintained facts, or reviewed module-owned structures.

It also protects the Loom execution mechanism boundary. Target-owned
qualification programs and manifests may invoke shared tools, but production
targets must not include execution internals or grow private runner/provider
stacks under `loom/src/loom/target/**`.

The core Loom compiler must stay independent from command-line flag machinery.
Only tools, tooling helpers, and tests may include `iree/base/tooling/flags.h`
or define IREE flags.

The SPIR-V backend has additional guardrails because its opcode, feature,
property, ABI, and resource cross-products are especially easy to turn into a
large hand-maintained emitter. Backend source files must stay below the reviewed
size ceiling and must not introduce broad `internal.h` umbrella headers as a
cosmetic split.

The authoring corpus additionally rejects stale boundary-proof boilerplate that
agents are likely to copy into generated kernels: redundant kernel-buffer
memory-space assumes, sentinel-sized views, late index-to-offset byte-address
casts, and ggml-style byte strides typed as logical indices.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
LOOM_SOURCE_ROOT = REPO_ROOT / "loom" / "src" / "loom"
AUTHORING_CORPUS_ROOT = LOOM_SOURCE_ROOT / "test" / "corpus" / "authoring"
SOURCE_SUFFIXES = {".c", ".cc", ".h"}
SPIRV_BACKEND_RELATIVE_ROOTS = (
    "loom/src/loom/target/arch/spirv",
    "loom/src/loom/target/emit/spirv",
)
SPIRV_BACKEND_SOURCE_LINE_LIMIT = 3000

# Matches common module pointers and nested owner paths ending in module, such
# as `module->values.count`, `source_module->values.capacity`, and
# `state->module->values.count`.
MODULE_VALUE_CARDINALITY_PATTERN = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*->)*"
    r"(?:source_|target_|input_|output_)?module->values\.(?:count|capacity)\b"
)

STORAGE_OR_RESET_PATTERNS = [
    re.compile(r"\biree_arena_allocate(?:_array)?\s*\("),
    re.compile(r"\biree_arena_grow_array\s*\("),
    re.compile(
        r"\biree_allocator_(?:"
        r"malloc(?:_array)?(?:_uninitialized)?|"
        r"realloc(?:_array)?|"
        r"(?:malloc|realloc)_aligned|"
        r"calloc|malloc_with_trailing|malloc_struct_array|grow_array"
        r")\s*\("
    ),
    re.compile(r"\bmemset\s*\("),
    re.compile(r"\biree_bitfield_[A-Za-z0-9_]*\s*\("),
]

# Local aliases are almost as dangerous as direct allocations: they make the
# storage pattern invisible to line-oriented review and usually precede dense
# tables sized by the whole module.
CARDINALITY_ALIAS_PATTERN = re.compile(
    r"\b(?:value_count|value_capacity|capacity|mask_count|"
    r"defined_bits_length|consuming_op_count|source_value_snapshot_count)\b"
    r"\s*(?:=|\+=)"
)

TARGET_EXECUTION_INCLUDE_PATTERN = re.compile(
    r'#\s*include\s+"loom/tooling/execution/hal/'
)
TARGET_EXECUTION_DEP_LABEL_PATTERN = re.compile(
    r"//loom/src/loom/tooling/execution/hal(?::|/)"
)
TARGET_PHYSICAL_TEST_SRC_PATTERN = re.compile(
    r'\bsrc\s*=\s*"(?P<label>//loom/src/loom/tools/'
    r'(?:iree-test-loom|iree-benchmark-loom)(?::[^"]*)?)"'
)
TARGET_EXECUTION_BAZEL_TOOLS_PATTERN = re.compile(
    r"\btools\s*=\s*\{(?P<body>.*?)\}", re.DOTALL
)
TARGET_PHYSICAL_TEST_TOOL_LABEL_PATTERN = re.compile(
    r"//loom/src/loom/tools/(?:iree-test-loom|iree-benchmark-loom)(?::|\")"
)
TARGET_EXECUTION_BAZEL_DEPS_PATTERN = re.compile(
    r"\b(?:deps|implementation_deps)\s*=\s*\[(?P<body>.*?)\]", re.DOTALL
)

TARGET_PRIVATE_EXECUTION_SYMBOL_PATTERN = re.compile(
    r"\b(?:"
    r"loom_run_hal_[A-Za-z0-9_]*"
    r"|loom_run_execution_(?:backend|provider)[A-Za-z0-9_]*"
    r")\b"
)

TOOLING_FLAGS_INCLUDE_PATTERN = re.compile(
    r"#\s*include\s+[<\"]iree/base/tooling/flags\.h[>\"]"
)
TOOLING_FLAGS_MACRO_PATTERN = re.compile(r"\bIREE_FLAG(?:_LIST)?(?:_NAMED)?\s*\(")

SPIRV_INTERNAL_HEADER_INCLUDE_PATTERN = re.compile(
    r'#\s*include\s+"loom/target/(?:arch|emit)/spirv/.+' r'(?:^|/|_)internal\.h"'
)

AUTHORING_MEMORY_SPACE_ASSUME_PATTERN = re.compile(
    r"\bbuffer\.assume\.memory_space<[^>]+>"
)
AUTHORING_SENTINEL_VIEW_EXTENT_PATTERN = re.compile(
    r"\bview<[^>\n]*\b(?:2147483647|4294967295|9223372036854775807)x"
)
AUTHORING_INDEX_TO_OFFSET_CAST_PATTERN = re.compile(
    r"\bindex\.cast\b.*:\s*index\s+to\s+offset\b"
)
AUTHORING_BYTE_STRIDE_INDEX_PATTERN = re.compile(
    r"%nb(?:\d+|_[A-Za-z0-9_]*)?\b\s*:\s*index\b"
)


@dataclass(frozen=True)
class ApprovedStatement:
    """A reviewed statement that intentionally mentions module value cardinality."""

    path: str
    pattern: re.Pattern[str]
    reason: str


APPROVED_STATEMENTS = [
    ApprovedStatement(
        "loom/src/loom/ir/module.c",
        re.compile(r"\bmodule->values\.capacity\b"),
        "core module value-table and module-owned scratch storage",
    ),
    ApprovedStatement(
        "loom/src/loom/format/text/printer/name_plan.c",
        re.compile(
            r"\biree_arena_allocate_array\s*\(\s*&out_plan->arena,\s*"
            r"module->values\.count,\s*sizeof\(\*out_plan->resolutions\)"
        ),
        "canonical per-print SSA name resolution indexed by value ID",
    ),
    ApprovedStatement(
        "loom/src/loom/format/text/printer/name_plan.c",
        re.compile(
            r"\bmemset\s*\(\s*out_plan->resolutions,\s*0,\s*"
            r"module->values\.count\s*\*\s*"
            r"sizeof\(\*out_plan->resolutions\)"
        ),
        "canonical per-print SSA name resolution initialization",
    ),
    ApprovedStatement(
        "loom/src/loom/pass/interpreter.c",
        re.compile(r"\.value_count\s*=\s*module->values\.count"),
        "pass interpreter epoch snapshot records IR mutation boundaries",
    ),
    ApprovedStatement(
        "loom/src/loom/pass/value_facts.c",
        re.compile(r"\bmodule->values\.capacity\b"),
        "maintained scoped value-fact owner storage",
    ),
    ApprovedStatement(
        "loom/src/loom/passes/refine_boundaries.c",
        re.compile(r"\bboundary_fact_value_capacity\s*=\s*module->values\.capacity"),
        "persistent boundary fact storage owned by the refine-boundaries pass",
    ),
    ApprovedStatement(
        "loom/src/loom/rewrite/rewriter.c",
        re.compile(r"\bvalue_count\s*=\s*rewriter->module->values\.count"),
        "rewriter checkpoint captures mutation boundaries, not scratch storage",
    ),
    ApprovedStatement(
        "loom/src/loom/rewrite/remap.c",
        re.compile(r"\.source_value_snapshot_count\s*=\s*source_module->values\.count"),
        "IR remap snapshot records source-module mutation boundaries",
    ),
    ApprovedStatement(
        "loom/src/loom/verify/verify.c",
        re.compile(r"\bmodule->values\.count\b"),
        "verifier masks over arbitrary user-provided IR",
    ),
    ApprovedStatement(
        "loom/src/loom/codegen/low/verify.c",
        re.compile(r"\bmodule->values\.count\b"),
        "low verifier masks over arbitrary user-provided IR",
    ),
]


@dataclass(frozen=True)
class Finding:
    """A lint finding with enough context for source review."""

    path: Path
    line: int
    message: str
    context: str


def _relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _strip_line_comments(statement: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in statement.splitlines())


def _iter_statements(path: Path) -> list[tuple[int, str]]:
    statements: list[tuple[int, str]] = []
    start_line = 0
    pending: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not pending:
            start_line = line_number
        pending.append(line.rstrip())
        stripped = line.strip()
        if ";" in stripped or stripped.endswith("{") or stripped.endswith("}"):
            statements.append((start_line, "\n".join(pending)))
            pending = []
    if pending:
        statements.append((start_line, "\n".join(pending)))
    return statements


def _is_approved(path: Path, statement: str) -> bool:
    relative_path = _relative_path(path)
    for approved in APPROVED_STATEMENTS:
        if relative_path == approved.path and approved.pattern.search(statement):
            return True
    return False


def _is_suspicious(statement: str) -> bool:
    uncommented = _strip_line_comments(statement)
    if not MODULE_VALUE_CARDINALITY_PATTERN.search(uncommented):
        return False
    if any(pattern.search(uncommented) for pattern in STORAGE_OR_RESET_PATTERNS):
        return True
    return CARDINALITY_ALIAS_PATTERN.search(uncommented) is not None


def _scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line, statement in _iter_statements(path):
        if _is_suspicious(statement) and not _is_approved(path, statement):
            findings.append(
                Finding(
                    path=path,
                    line=line,
                    message=(
                        "module-value-cardinality storage/reset requires "
                        "reviewed infrastructure"
                    ),
                    context=statement,
                )
            )
    return findings


def _iter_source_files() -> list[Path]:
    return sorted(
        path
        for path in LOOM_SOURCE_ROOT.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def _format_statement(statement: str) -> str:
    stripped_lines = [line.strip() for line in statement.splitlines()]
    return " ".join(line for line in stripped_lines if line)


def _is_spirv_backend_relative_path(relative_path: str) -> bool:
    return any(
        relative_path == root or relative_path.startswith(root + "/")
        for root in SPIRV_BACKEND_RELATIVE_ROOTS
    )


def _is_source_relative_path(relative_path: str) -> bool:
    return PurePosixPath(relative_path).suffix in SOURCE_SUFFIXES


def _is_internal_header_name(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name
    return name == "internal.h" or name.endswith("_internal.h")


def _is_test_source_name(name: str) -> bool:
    return (
        name.endswith("_test.c")
        or name.endswith("_test.cc")
        or name.endswith("_test.h")
    )


def _is_core_loom_source_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if path.suffix not in SOURCE_SUFFIXES:
        return False
    parts = path.parts
    if len(parts) < 4 or parts[:3] != ("loom", "src", "loom"):
        return False
    if parts[3] in ("tools", "tooling"):
        return False
    if "test" in parts[3:] or "testing" in parts[3:]:
        return False
    return not _is_test_source_name(path.name)


def _scan_core_tooling_flags_text(
    relative_path: str, text: str
) -> list[tuple[int, str, str]]:
    if not _is_core_loom_source_path(relative_path):
        return []

    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped_line = _strip_line_comments(line).strip()
        if TOOLING_FLAGS_INCLUDE_PATTERN.search(stripped_line):
            findings.append(
                (
                    line_number,
                    "core Loom compiler source must not include tooling flags",
                    stripped_line,
                )
            )
        if TOOLING_FLAGS_MACRO_PATTERN.search(stripped_line):
            findings.append(
                (
                    line_number,
                    "core Loom compiler source must not define command-line flags",
                    stripped_line,
                )
            )
    return findings


def _scan_spirv_backend_text(
    relative_path: str, text: str
) -> list[tuple[int, str, str]]:
    if not _is_spirv_backend_relative_path(relative_path):
        return []

    findings: list[tuple[int, str, str]] = []
    if _is_source_relative_path(relative_path):
        line_count = len(text.splitlines())
        if line_count > SPIRV_BACKEND_SOURCE_LINE_LIMIT:
            findings.append(
                (
                    SPIRV_BACKEND_SOURCE_LINE_LIMIT + 1,
                    "SPIR-V backend source exceeds the reviewed file-size ceiling",
                    (
                        f"{line_count} lines exceeds "
                        f"{SPIRV_BACKEND_SOURCE_LINE_LIMIT}; move the "
                        "cross-product into generated/source-of-truth tables "
                        "or a public invariant boundary"
                    ),
                )
            )

    if _is_internal_header_name(relative_path):
        findings.append(
            (
                1,
                "SPIR-V backend must not use internal.h-style umbrella headers",
                (
                    "split by public representation contract instead of "
                    "sharing private declarations through a broad internal header"
                ),
            )
        )

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped_line = _strip_line_comments(line).strip()
        if SPIRV_INTERNAL_HEADER_INCLUDE_PATTERN.search(stripped_line):
            findings.append(
                (
                    line_number,
                    "SPIR-V backend must not include internal.h-style umbrellas",
                    stripped_line,
                )
            )
    return findings


def _iter_lint_files() -> list[Path]:
    return sorted(
        path
        for path in LOOM_SOURCE_ROOT.rglob("*")
        if path.is_file()
        and (
            path.suffix in SOURCE_SUFFIXES
            or path.name == "BUILD.bazel"
            or path.name == "CMakeLists.txt"
        )
    )


def _iter_authoring_loom_files() -> list[Path]:
    return sorted(
        path for path in AUTHORING_CORPUS_ROOT.rglob("*.loom") if path.is_file()
    )


def _scan_authoring_text(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped_line = _strip_line_comments(line).strip()
        if not stripped_line:
            continue
        if AUTHORING_MEMORY_SPACE_ASSUME_PATTERN.search(stripped_line):
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    message=(
                        "authoring corpus must not reassert kernel ABI "
                        "memory-space facts"
                    ),
                    context=(
                        "kernel buffer arguments already publish global "
                        "memory-space facts; use the buffer directly and keep "
                        "buffer.assume.noalias only when the source contract "
                        "has restrict/noalias"
                    ),
                )
            )
        if AUTHORING_SENTINEL_VIEW_EXTENT_PATTERN.search(stripped_line):
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    message=(
                        "authoring corpus must not use sentinel-sized view extents"
                    ),
                    context=(
                        "views should carry real static or dynamic extents so "
                        "bounds checks, sanitizer paths, and footprint facts "
                        "see the accessible range"
                    ),
                )
            )
        if AUTHORING_INDEX_TO_OFFSET_CAST_PATTERN.search(stripped_line):
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    message=(
                        "authoring corpus must spell byte addresses in the "
                        "offset domain"
                    ),
                    context=(
                        "use offset constants/arguments for byte offsets and "
                        "index.scale for logical-index times byte-stride "
                        "conversion before buffer.view"
                    ),
                )
            )
        if AUTHORING_BYTE_STRIDE_INDEX_PATTERN.search(stripped_line):
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    message=(
                        "authoring corpus must type ggml-style byte strides as offset"
                    ),
                    context=(
                        "nb* names conventionally mean byte strides; use "
                        "offset or rename non-byte quantities before agents "
                        "copy the pattern"
                    ),
                )
            )
    return findings


def _scan_authoring_corpus(path: Path) -> list[Finding]:
    return _scan_authoring_text(path, path.read_text())


def _scan_target_execution_text(
    relative_path: str, text: str
) -> list[tuple[int, str, str]]:
    path = PurePosixPath(relative_path)
    if len(path.parts) < 4 or path.parts[:4] != (
        "loom",
        "src",
        "loom",
        "target",
    ):
        return []
    if path.suffix in SOURCE_SUFFIXES and _is_test_source_name(path.name):
        return []

    findings: list[tuple[int, str, str]] = []
    if path.suffix in SOURCE_SUFFIXES:
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped_line = _strip_line_comments(line).strip()
            if TARGET_EXECUTION_INCLUDE_PATTERN.search(stripped_line):
                findings.append(
                    (
                        line_number,
                        (
                            "target packages must not include or depend on "
                            "execution mechanism internals"
                        ),
                        stripped_line,
                    )
                )
            if TARGET_PRIVATE_EXECUTION_SYMBOL_PATTERN.search(stripped_line):
                findings.append(
                    (
                        line_number,
                        (
                            "target packages must not define private Loom "
                            "execution backend/provider stacks"
                        ),
                        stripped_line,
                    )
                )
        return findings

    if path.name != "BUILD.bazel":
        return findings
    lines = text.splitlines()
    bazel_text = "\n".join(line.split("#", 1)[0] for line in lines)
    for src_match in TARGET_PHYSICAL_TEST_SRC_PATTERN.finditer(bazel_text):
        line_number = bazel_text.count("\n", 0, src_match.start("label")) + 1
        findings.append(
            (
                line_number,
                "target packages must not own physical execution tests",
                lines[line_number - 1].strip(),
            )
        )
    for tools_match in TARGET_EXECUTION_BAZEL_TOOLS_PATTERN.finditer(bazel_text):
        body = tools_match.group("body")
        for label_match in TARGET_PHYSICAL_TEST_TOOL_LABEL_PATTERN.finditer(body):
            offset = tools_match.start("body") + label_match.start()
            line_number = bazel_text.count("\n", 0, offset) + 1
            findings.append(
                (
                    line_number,
                    "target packages must not own physical execution tests",
                    lines[line_number - 1].strip(),
                )
            )
    for deps_match in TARGET_EXECUTION_BAZEL_DEPS_PATTERN.finditer(bazel_text):
        body = deps_match.group("body")
        for label_match in TARGET_EXECUTION_DEP_LABEL_PATTERN.finditer(body):
            offset = deps_match.start("body") + label_match.start()
            line_number = bazel_text.count("\n", 0, offset) + 1
            findings.append(
                (
                    line_number,
                    (
                        "target packages must not include or depend on "
                        "execution mechanism internals"
                    ),
                    lines[line_number - 1].strip(),
                )
            )
    return findings


def _scan_target_execution_guardrails(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for line, message, context in _scan_target_execution_text(
        _relative_path(path), path.read_text()
    ):
        findings.append(
            Finding(
                path=path,
                line=line,
                message=message,
                context=context,
            )
        )
    return findings


def _scan_spirv_backend_guardrails(path: Path) -> list[Finding]:
    relative_path = _relative_path(path)
    findings: list[Finding] = []
    for line, message, context in _scan_spirv_backend_text(
        relative_path, path.read_text()
    ):
        findings.append(
            Finding(
                path=path,
                line=line,
                message=message,
                context=context,
            )
        )
    return findings


def _scan_core_tooling_flags_guardrails(path: Path) -> list[Finding]:
    relative_path = _relative_path(path)
    findings: list[Finding] = []
    for line, message, context in _scan_core_tooling_flags_text(
        relative_path, path.read_text()
    ):
        findings.append(
            Finding(
                path=path,
                line=line,
                message=message,
                context=context,
            )
        )
    return findings


def _expect_spirv_self_test(
    name: str, relative_path: str, text: str, expected_messages: tuple[str, ...]
) -> bool:
    findings = _scan_spirv_backend_text(relative_path, text)
    messages = tuple(finding[1] for finding in findings)
    if messages == expected_messages:
        print(f"  PASS  {name}")
        return True
    print(f"  FAIL  {name}")
    print(f"        expected: {expected_messages!r}")
    print(f"        actual:   {messages!r}")
    return False


def _expect_core_tooling_flags_self_test(
    name: str, relative_path: str, text: str, expected_messages: tuple[str, ...]
) -> bool:
    findings = _scan_core_tooling_flags_text(relative_path, text)
    messages = tuple(finding[1] for finding in findings)
    if messages == expected_messages:
        print(f"  PASS  {name}")
        return True
    print(f"  FAIL  {name}")
    print(f"        expected: {expected_messages!r}")
    print(f"        actual:   {messages!r}")
    return False


def _expect_target_execution_self_test(
    name: str, relative_path: str, text: str, expected_messages: tuple[str, ...]
) -> bool:
    findings = _scan_target_execution_text(relative_path, text)
    messages = tuple(finding[1] for finding in findings)
    if messages == expected_messages:
        print(f"  PASS  {name}")
        return True
    print(f"  FAIL  {name}")
    print(f"        expected: {expected_messages!r}")
    print(f"        actual:   {messages!r}")
    return False


def _expect_authoring_self_test(
    name: str, text: str, expected_messages: tuple[str, ...]
) -> bool:
    path = AUTHORING_CORPUS_ROOT / "self_test.loom"
    findings = _scan_authoring_text(path, text)
    messages = tuple(finding.message for finding in findings)
    if messages == expected_messages:
        print(f"  PASS  {name}")
        return True
    print(f"  FAIL  {name}")
    print(f"        expected: {expected_messages!r}")
    print(f"        actual:   {messages!r}")
    return False


def _run_self_tests() -> int:
    ok = True
    print("loom-source-lint: self-test")
    ok &= _expect_spirv_self_test(
        "SPIR-V small source passes",
        "loom/src/loom/target/emit/spirv/packet_rows.c",
        "void f(void) {}\n",
        (),
    )
    ok &= _expect_spirv_self_test(
        "SPIR-V oversized source fails",
        "loom/src/loom/target/emit/spirv/module_emitter.c",
        "\n".join("line" for _ in range(SPIRV_BACKEND_SOURCE_LINE_LIMIT + 1)),
        ("SPIR-V backend source exceeds the reviewed file-size ceiling",),
    )
    ok &= _expect_spirv_self_test(
        "SPIR-V internal header path fails",
        "loom/src/loom/target/emit/spirv/lower/internal.h",
        "",
        ("SPIR-V backend must not use internal.h-style umbrella headers",),
    )
    ok &= _expect_spirv_self_test(
        "SPIR-V internal header include fails",
        "loom/src/loom/target/emit/spirv/packet.c",
        '#include "loom/target/emit/spirv/lower/internal.h"\n',
        ("SPIR-V backend must not include internal.h-style umbrellas",),
    )
    ok &= _expect_spirv_self_test(
        "non-SPIR-V internal header is out of scope",
        "runtime/src/iree/tooling/profile/internal.h",
        "",
        (),
    )
    ok &= _expect_core_tooling_flags_self_test(
        "core tooling flags include fails",
        "loom/src/loom/sanitizer/options.c",
        '#include "iree/base/tooling/flags.h"\n',
        ("core Loom compiler source must not include tooling flags",),
    )
    ok &= _expect_core_tooling_flags_self_test(
        "core IREE flag macro fails",
        "loom/src/loom/sanitizer/options.c",
        'IREE_FLAG(string, sanitizer, "none", "Sanitizers");\n',
        ("core Loom compiler source must not define command-line flags",),
    )
    ok &= _expect_core_tooling_flags_self_test(
        "tooling flags include is in scope",
        "loom/src/loom/tooling/sanitizer/options_cli.h",
        '#include "iree/base/tooling/flags.h"\n',
        (),
    )
    ok &= _expect_core_tooling_flags_self_test(
        "test flags include is in scope",
        "loom/src/loom/sanitizer/options_test.cc",
        '#include "iree/base/tooling/flags.h"\n',
        (),
    )
    ok &= _expect_target_execution_self_test(
        "provider-owned qualification package passes",
        "loom/src/loom/tooling/target/amdgpu/test/corpus/BUILD.bazel",
        'src = "//loom/src/loom/tools/iree-test-loom"\n',
        (),
    )
    ok &= _expect_target_execution_self_test(
        "target-owned qualification package fails",
        "loom/src/loom/target/arch/amdgpu/test/BUILD.bazel",
        'src = "//loom/src/loom/tools/iree-test-loom"\n',
        ("target packages must not own physical execution tests",),
    )
    ok &= _expect_target_execution_self_test(
        "target-owned qualification tool fails",
        "loom/src/loom/target/arch/amdgpu/test/BUILD.bazel",
        ('tools = {"iree-test-loom": "//loom/src/loom/tools/iree-test-loom"}\n'),
        ("target packages must not own physical execution tests",),
    )
    ok &= _expect_target_execution_self_test(
        "physical tool visibility grant passes",
        "loom/src/loom/target/arch/amdgpu/BUILD.bazel",
        (
            "_PROVIDER_VISIBILITY = [\n    "
            '"//loom/src/loom/tools/iree-test-loom:__pkg__",\n]\n'
        ),
        (),
    )
    ok &= _expect_target_execution_self_test(
        "execution package visibility grant passes",
        "loom/src/loom/target/arch/amdgpu/BUILD.bazel",
        ('_EXECUTION_VISIBILITY = ["//loom/src/loom/tooling/execution/hal:__pkg__"]\n'),
        (),
    )
    ok &= _expect_target_execution_self_test(
        "commented execution dependency passes",
        "loom/src/loom/target/arch/amdgpu/BUILD.bazel",
        ('# deps = [\n#     "//loom/src/loom/tooling/execution/hal:runtime",\n# ]\n'),
        (),
    )
    ok &= _expect_target_execution_self_test(
        "target execution-internal dependency fails",
        "loom/src/loom/target/arch/amdgpu/BUILD.bazel",
        ('deps = [\n    "//loom/src/loom/tooling/execution/hal:runtime",\n]\n'),
        (
            "target packages must not include or depend on execution "
            "mechanism internals",
        ),
    )
    ok &= _expect_target_execution_self_test(
        "target execution-internal include fails",
        "loom/src/loom/target/arch/amdgpu/run.c",
        '#include "loom/tooling/execution/hal/runtime.h"\n',
        (
            "target packages must not include or depend on execution "
            "mechanism internals",
        ),
    )
    ok &= _expect_target_execution_self_test(
        "target private execution stack fails",
        "loom/src/loom/target/arch/amdgpu/run.c",
        "void loom_run_hal_backend(void) {}\n",
        (
            "target packages must not define private Loom execution "
            "backend/provider stacks",
        ),
    )
    checked_loom_happy_path = """
kernel.def @copy(%n: index) {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%n: index, %input: buffer, %output: buffer) {
  %base = index.constant 0 : offset
  %batch_size = index.constant 512 : index
  %c2 = index.constant 2 : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_point = scalar.constant 128 : i32
  %fourth_word_ordinal = scalar.constant 4 : i32
  %input_view = buffer.view %input[%base] : buffer -> view<[%n]xf32>
  test.use %batch_size, %c2, %c0_i32, %zero_point, %fourth_word_ordinal : index, index, i32, i32, i32
  kernel.return
}
"""
    ok &= _expect_authoring_self_test(
        "authoring happy-path source passes",
        checked_loom_happy_path,
        (),
    )
    ok &= _expect_authoring_self_test(
        "authoring memory-space assume fails",
        "%global = buffer.assume.memory_space<global> %input : buffer\n",
        ("authoring corpus must not reassert kernel ABI memory-space facts",),
    )
    ok &= _expect_authoring_self_test(
        "authoring sentinel extent fails",
        "%view = buffer.view %input[%zero] : buffer -> view<2147483647xi8>\n",
        ("authoring corpus must not use sentinel-sized view extents",),
    )
    ok &= _expect_authoring_self_test(
        "authoring index-to-offset cast fails",
        "%byte_offset = index.cast %byte_index : index to offset\n",
        ("authoring corpus must spell byte addresses in the offset domain",),
    )
    ok &= _expect_authoring_self_test(
        "authoring byte-stride index argument fails",
        "kernel.def @bad(%nb0: index) { kernel.return }\n",
        ("authoring corpus must type ggml-style byte strides as offset",),
    )
    return 0 if ok else 1


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _run_self_tests()
    if len(sys.argv) > 1:
        print("usage: loom_source_lint.py [--self-test]")
        return 2

    findings: list[Finding] = []
    for path in _iter_source_files():
        findings.extend(_scan_file(path))
    for path in _iter_lint_files():
        findings.extend(_scan_core_tooling_flags_guardrails(path))
        findings.extend(_scan_target_execution_guardrails(path))
        findings.extend(_scan_spirv_backend_guardrails(path))
    for path in _iter_authoring_loom_files():
        findings.extend(_scan_authoring_corpus(path))

    if not findings:
        print("loom-source-lint: PASS")
        return 0

    print("loom-source-lint: FAIL: Loom source invariant violation.")
    for finding in findings:
        print(
            f"{_relative_path(finding.path)}:{finding.line}: "
            f"{finding.message}: {_format_statement(finding.context)}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
