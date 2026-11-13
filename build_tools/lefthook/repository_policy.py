# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository-specific path and text policies.

These rules intentionally cover lexical contracts. Semantic C and C++
contracts belong in the clang-tidy plugin, where parsed source and macro
expansion information are available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyRule:
    # Stable identifier printed with each finding.
    rule_id: str
    # Explanation of the repository contract.
    message: str
    # Repository-relative paths owned by the rule.
    path_pattern: re.Pattern[str]
    # Forbidden source text within owned paths.
    content_pattern: re.Pattern[str]


@dataclass(frozen=True)
class PolicyFinding:
    # Repository-relative path containing the finding.
    path: str
    # One-based line containing the beginning of the match.
    line: int
    # One-based column containing the beginning of the match.
    column: int
    # Stable identifier of the violated rule.
    rule_id: str
    # Explanation of the repository contract.
    message: str


RULES = (
    PolicyRule(
        rule_id="iree.c.no-goto-error-handling",
        message=(
            "C runtime code uses scoped status handling and terminal cleanup "
            "blocks instead of goto-based error handling"
        ),
        path_pattern=re.compile(r"^(?:runtime/src/iree|loom/src/loom|libamdf)/.*\.c$"),
        content_pattern=re.compile(r"\bgoto\s+[A-Za-z_][A-Za-z0-9_]*\s*;"),
    ),
    PolicyRule(
        rule_id="iree.cpp.no-designated-initializer-suppression",
        message=(
            "C++ implementation sources must satisfy the designated-initializer "
            "policy without suppressing iree-cpp-designated-initializer"
        ),
        path_pattern=re.compile(
            r"^(?:runtime|loom|libamdf|libhrx)/.*\.(?:cc|cpp|cxx)$"
        ),
        content_pattern=re.compile(
            r"NOLINT(?:BEGIN|END|NEXTLINE)?\s*\([^)]*"
            r"\biree-cpp-designated-initializer\b[^)]*\)"
        ),
    ),
    PolicyRule(
        rule_id="iree.loom.target-diagnostics-are-target-generic",
        message=(
            "TARGET diagnostics are target-generic; architecture- or vendor-specific "
            "semantics belong to the owning error domain"
        ),
        path_pattern=re.compile(r"^loom/py/loom/error/target\.py$"),
        content_pattern=re.compile(
            r"\b(?:amd(?:gpu|gcn|64)?|rdna[0-9a-z._-]*|cdna[0-9a-z._-]*|gcn|"
            r"gfx[0-9][0-9a-z._-]*|rocm|aie[0-9a-z._-]*|xdna[0-9a-z._-]*|"
            r"x86[0-9a-z._-]*|sse[0-9a-z._-]*|avx[0-9a-z._-]*|wasm|"
            r"webassembly|spir-?v|vulkan|nvidia|cuda|ptx|risc-?v|aarch64|"
            r"arm(?:32|64)?|neon|intel|apple|metal)\b",
            re.IGNORECASE,
        ),
    ),
    PolicyRule(
        rule_id="iree.loom.arch-target-record-diagnostics-use-own-domain",
        message=(
            "architecture target-record diagnostics use the architecture's error "
            "domain instead of TARGET"
        ),
        path_pattern=re.compile(r"^loom/src/loom/target/arch/(?:.*/)?ops/target\.c$"),
        content_pattern=re.compile(r"\bLOOM_ERR_TARGET_[0-9]{3}(?:_REF)?\b"),
    ),
    PolicyRule(
        rule_id="amd-reference.no-library-vocabulary",
        message=("the AMD hardware reference is independent of libamdf APIs and CTS"),
        path_pattern=re.compile(r"^docs/reference/amd/.*$"),
        content_pattern=re.compile(
            r"\b(?:libamdf|amdf_[a-z0-9_]+|cts)\b", re.IGNORECASE
        ),
    ),
    PolicyRule(
        rule_id="amd-reference.no-local-implementation-links",
        message=(
            "the AMD hardware reference cites native primary sources instead of "
            "local implementation, header, test, or working-note paths"
        ),
        path_pattern=re.compile(r"^docs/reference/amd/.*$"),
        content_pattern=re.compile(
            r"(?:\]\(\s*<?|\]:[ \t]*<?)(?!(?:https?://|#))"
            r"(?:[^()\s<>]*/)?(?:src|include|cts|\.notes)/[^()\s<>]*",
            re.IGNORECASE,
        ),
    ),
    PolicyRule(
        rule_id="libamdf.docs.no-cts-dependencies",
        message=(
            "libamdf documentation describes the API independently of CTS; test "
            "documentation may consume library documentation"
        ),
        path_pattern=re.compile(r"^(?:libamdf/docs/.*|libamdf/README\.md)$"),
        content_pattern=re.compile(r"\bcts\b", re.IGNORECASE),
    ),
)


def is_candidate_path(path: str) -> bool:
    """Returns whether any repository policy owns the path."""
    return any(rule.path_pattern.fullmatch(path) for rule in RULES)


def findings_for_text(path: str, text: str) -> list[PolicyFinding]:
    """Returns all policy findings in repository-relative source text."""
    findings = []
    for rule in RULES:
        if not rule.path_pattern.fullmatch(path):
            continue
        for match in rule.content_pattern.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            findings.append(
                PolicyFinding(
                    path=path,
                    line=text.count("\n", 0, match.start()) + 1,
                    column=match.start() - line_start + 1,
                    rule_id=rule.rule_id,
                    message=rule.message,
                )
            )
    return findings
