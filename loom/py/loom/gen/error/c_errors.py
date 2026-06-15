# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Error definitions -> C error catalog tables and JSON catalog.

Reads ErrorDef instances from the Python error catalog and emits:

  error_catalog.c     - .rodata: param def arrays, error def structs,
                         lookup tables, name tables
  error_catalog.h     - canonical C names for direct defs and compact refs
  error_catalog.json  - JSON catalog for tooling/documentation

The generated files are build outputs. The Python error catalog is the source
of truth.

Usage:
    python3 loom/py/loom/gen/run.py c_errors --check
    bazel run //loom/py/loom/gen/error:c_errors -- --source=/tmp/error_catalog.c
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[3]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.errors import Emitter, ErrorDef, ErrorDomain, ParamKind, Severity  # noqa: E402
from loom.gen.support.files import write_text_file as _write_text  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402


def _c_symbol(error: ErrorDef) -> str:
    """Returns the public C symbol name for an error definition."""
    return f"loom_err_{error.domain.name.lower()}_{error.code:03d}"


def _error_def_macro(error: ErrorDef) -> str:
    """Returns the public C macro for an error definition pointer."""
    return f"LOOM_ERR_{error.domain.name}_{error.code:03d}"


def _error_ref_macro(error: ErrorDef) -> str:
    """Returns the public C macro for a compact error reference."""
    return f"LOOM_ERR_{error.domain.name}_{error.code:03d}_REF"


def _header_guard_from_public_header(public_header: str) -> str:
    guard = "".join(c.upper() if c.isalnum() else "_" for c in public_header)
    while "__" in guard:
        guard = guard.replace("__", "_")
    return guard.strip("_") + "_"


def _catalog_domain_symbol(catalog_symbol: str, domain: ErrorDomain) -> str:
    return f"{catalog_symbol}_{domain.name.lower()}"


def _catalog_domain_defs_symbol(catalog_symbol: str, domain: ErrorDomain) -> str:
    return f"{_catalog_domain_symbol(catalog_symbol, domain)}_defs"


def _param_kind_c_name(kind: ParamKind) -> str:
    return f"LOOM_PARAM_{kind.name}"


def _severity_c_name(severity: Severity) -> str:
    return f"LOOM_DIAGNOSTIC_{severity.name}"


def _domain_c_name(domain: ErrorDomain) -> str:
    return f"LOOM_ERROR_DOMAIN_{domain.name}"


def _emitter_c_name(emitter: Emitter) -> str:
    return f"LOOM_EMITTER_{emitter.name}"


def _escape_c_string(text: str) -> str:
    """Escapes a string for C string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _group_errors_by_domain(
    errors: Sequence[ErrorDef],
) -> dict[ErrorDomain, list[ErrorDef]]:
    domains: dict[ErrorDomain, list[ErrorDef]] = {}
    seen: set[tuple[ErrorDomain, int]] = set()
    for error in sorted(errors, key=lambda e: (e.domain.value, e.code)):
        key = (error.domain, error.code)
        if key in seen:
            raise ValueError(f"{error.error_id}: duplicate error code")
        seen.add(key)
        if error.code > ((1 << 10) - 1):
            raise ValueError(f"{error.error_id}: compact error ref code overflow")
        domains.setdefault(error.domain, []).append(error)
    return domains


def generate_error_catalog_c(
    errors: list[ErrorDef],
    *,
    catalog_symbol: str,
    public_header: str,
    fallback_catalog_symbol: str | None = None,
    fallback_public_header: str | None = None,
) -> str:
    """Generates error_catalog.c with .rodata error definitions."""
    lines = [
        *line_comment_header(
            "//",
            generator="loom.gen.error.c_errors",
            regenerate="build the owning loom_generated_cc_library target",
        ),
        "",
        f'#include "{public_header}"',
        "",
    ]
    if fallback_public_header is not None:
        lines.extend(
            [
                f'#include "{fallback_public_header}"',
                "",
            ]
        )

    # Per-error param def arrays and error def structs.
    for error in errors:
        symbol = _c_symbol(error)

        # Param defs array (only if params exist).
        if error.params:
            lines.append(f"static const loom_error_param_def_t {symbol}_params[] = {{")
            for param in error.params:
                name_escaped = _escape_c_string(param.name)
                lines.append(f'    {{"{name_escaped}", {_param_kind_c_name(param.kind)}}},')
            lines.append("};")
        lines.append(f"const loom_error_def_t {symbol} = {{")
        lines.append(f'    .error_id = "{_escape_c_string(error.error_id)}",')
        lines.append(f"    .domain = {_domain_c_name(error.domain)},")
        lines.append(f"    .severity = {_severity_c_name(error.severity)},")
        lines.append(f"    .code = {error.code},")
        lines.append(f'    .summary = "{_escape_c_string(error.summary)}",')
        lines.append(f'    .message_template = "{_escape_c_string(error.message)}",')
        if error.fix_hint:
            lines.append(f'    .fix_hint_template = "{_escape_c_string(error.fix_hint)}",')
        else:
            lines.append("    .fix_hint_template = NULL,")
        if error.params:
            lines.append(f"    .param_defs = {symbol}_params,")
        else:
            lines.append("    .param_defs = NULL,")
        lines.append(f"    .param_count = {len(error.params)},")
        lines.append("};")
        lines.append("")

    grouped_errors = _group_errors_by_domain(errors)
    for domain, domain_errors in grouped_errors.items():
        max_code = max(error.code for error in domain_errors)
        defs_symbol = _catalog_domain_defs_symbol(catalog_symbol, domain)
        lines.append(f"static const loom_error_def_t* const {defs_symbol}[{max_code + 1}] = {{")
        lines.extend(f"    [{error.code}] = &{_c_symbol(error)}," for error in domain_errors)
        lines.append("};")
        lines.append("")
        lines.append(f"const loom_error_domain_catalog_t {_catalog_domain_symbol(catalog_symbol, domain)} = {{")
        lines.append(f"    .domain = {_domain_c_name(domain)},")
        lines.append(f"    .code_count = {max_code + 1},")
        lines.append(f"    .errors_by_code = {defs_symbol},")
        lines.append("};")
        lines.append("")

    lines.append(f"const loom_error_catalog_t {catalog_symbol} = {{")
    lines.append("    .domains = {")
    lines.extend(f"        [{_domain_c_name(domain)}] = &{_catalog_domain_symbol(catalog_symbol, domain)}," for domain in grouped_errors)
    lines.append("    },")
    if fallback_catalog_symbol is not None:
        lines.append(f"    .fallback_catalog = &{fallback_catalog_symbol},")
    else:
        lines.append("    .fallback_catalog = NULL,")
    lines.append("};")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_error_catalog_h(errors: list[ErrorDef], *, catalog_symbol: str, public_header: str) -> str:
    """Generates error_catalog.h with canonical C error references."""

    seen_def_macros: set[str] = set()
    seen_ref_macros: set[str] = set()
    seen_symbols: set[str] = set()
    lines = [
        *line_comment_header(
            "//",
            generator="loom.gen.error.c_errors",
            regenerate="build the owning loom_generated_cc_library target",
        ),
        "",
        f"#ifndef {_header_guard_from_public_header(public_header)}",
        f"#define {_header_guard_from_public_header(public_header)}",
        "",
        '#include "loom/error/error_defs.h"',
        "",
        f"extern const loom_error_catalog_t {catalog_symbol};",
        "",
    ]
    lines.extend(f"extern const loom_error_domain_catalog_t {_catalog_domain_symbol(catalog_symbol, domain)};" for domain in _group_errors_by_domain(errors))
    if errors:
        lines.append("")
    for error in errors:
        symbol = _c_symbol(error)
        def_macro = _error_def_macro(error)
        ref_macro = _error_ref_macro(error)
        if symbol in seen_symbols:
            raise ValueError(f"{error.error_id}: duplicate error symbol {symbol}")
        if def_macro in seen_def_macros:
            raise ValueError(f"{error.error_id}: duplicate error def macro {def_macro}")
        if ref_macro in seen_ref_macros:
            raise ValueError(f"{error.error_id}: duplicate error ref macro {ref_macro}")
        seen_symbols.add(symbol)
        seen_def_macros.add(def_macro)
        seen_ref_macros.add(ref_macro)
        lines.append(f"// {error.error_id}: {error.summary}")
        lines.append(f"extern const loom_error_def_t {symbol};")
        lines.append(f"#define {def_macro} (&{symbol})")
        lines.append(f"#define {ref_macro} \\")
        lines.append(f"  LOOM_ERROR_REF({_domain_c_name(error.domain)}, {error.code})")
        lines.append("")
    lines.append(f"#endif  // {_header_guard_from_public_header(public_header)}")
    lines.append("")
    return "\n".join(lines)


def generate_error_runtime_tables_inl() -> str:
    """Generates private enum-name lookup tables for error_defs.c."""
    lines = [
        *line_comment_header(
            "//",
            generator="loom.gen.error.c_errors",
            regenerate="build the owning loom_generated_cc_library target",
        ),
        "",
        "// clang-format off",
        "static const char* const loom_diagnostic_severity_names[] = {",
    ]
    lines.extend(f'    [{_severity_c_name(severity)}] = "{severity.name.lower()}",' for severity in Severity)
    lines.extend(
        [
            "};",
            "static_assert(IREE_ARRAYSIZE(loom_diagnostic_severity_names) ==",
            "                  LOOM_DIAGNOSTIC_COUNT_,",
            '              "Python Severity enum must match loom_diagnostic_severity_t");',
            "",
            "static const char* const loom_error_domain_names[] = {",
        ]
    )
    lines.extend(f'    [{_domain_c_name(domain)}] = "{domain.name}",' for domain in ErrorDomain)
    lines.extend(
        [
            "};",
            "static_assert(IREE_ARRAYSIZE(loom_error_domain_names) ==",
            "                  LOOM_ERROR_DOMAIN_COUNT_,",
            '              "Python ErrorDomain enum must match loom_error_domain_t");',
            "",
            "static const char* const loom_emitter_names[] = {",
        ]
    )
    lines.extend(f'    [{_emitter_c_name(emitter)}] = "{emitter.name.lower()}",' for emitter in Emitter)
    lines.extend(
        [
            "};",
            "static_assert(IREE_ARRAYSIZE(loom_emitter_names) ==",
            "                  LOOM_EMITTER_COUNT_,",
            '              "Python Emitter enum must match loom_emitter_t");',
            "// clang-format on",
            "",
        ]
    )
    return "\n".join(lines)


def generate_error_catalog_json(errors: list[ErrorDef]) -> str:
    """Generates the JSON error catalog."""
    catalog = []
    for error in errors:
        entry = {
            "error_id": error.error_id,
            "domain": error.domain.value,
            "domain_name": error.domain.name,
            "code": error.code,
            "severity": error.severity.value,
            "severity_name": error.severity.name.lower(),
            "summary": error.summary,
            "message_template": error.message,
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind.value,
                    "kind_name": p.kind.name,
                }
                for p in error.params
            ],
        }
        if error.fix_hint:
            entry["fix_hint_template"] = error.fix_hint
        if error.description:
            entry["description"] = error.description
        if error.example:
            entry["example"] = error.example
        catalog.append(entry)

    return json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"


_OPTIONAL_TARGET_DOMAINS = frozenset(
    {
        ErrorDomain.AMDGPU,
        ErrorDomain.X86,
        ErrorDomain.WASM,
        ErrorDomain.SPIRV,
    }
)


def _catalog_shard_errors(shard: str) -> list[ErrorDef]:
    from loom.error import ALL_ERRORS

    errors = list(ALL_ERRORS)
    if shard == "all":
        return errors
    if shard == "core":
        return [error for error in errors if error.domain not in _OPTIONAL_TARGET_DOMAINS]
    domain_by_shard = {
        "amdgpu": ErrorDomain.AMDGPU,
        "x86": ErrorDomain.X86,
        "wasm": ErrorDomain.WASM,
        "spirv": ErrorDomain.SPIRV,
    }
    return [error for error in errors if error.domain == domain_by_shard[shard]]


def main(argv: Sequence[str] | None = None) -> int:
    """Generate C error catalog tables and JSON catalog."""
    parser = argparse.ArgumentParser(description="Generate Loom error catalogs from Python definitions.")
    parser.add_argument(
        "--shard",
        choices=("all", "core", "amdgpu", "x86", "wasm", "spirv"),
        default="all",
        help="Named error catalog shard to generate.",
    )
    parser.add_argument(
        "--catalog-symbol",
        help="C symbol name for the generated loom_error_catalog_t.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/error/error_catalog.h",
        help="Public include path for the generated catalog header.",
    )
    parser.add_argument(
        "--fallback-catalog-symbol",
        help="Catalog symbol searched when this shard misses a domain/code.",
    )
    parser.add_argument(
        "--fallback-public-header",
        help="Public include path declaring --fallback-catalog-symbol.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Generated C error catalog source path.",
    )
    parser.add_argument(
        "--header",
        type=Path,
        help="Generated C error catalog header path.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Generated JSON error catalog path.",
    )
    parser.add_argument(
        "--runtime-tables",
        type=Path,
        help="Generated private error_defs_tables.inl runtime table include.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation without writing files.",
    )
    args = parser.parse_args(argv)

    if not args.check and args.source is None and args.header is None and args.catalog is None and args.runtime_tables is None:
        parser.error("at least one of --source, --header, --catalog, --runtime-tables, or --check is required")

    errors = _catalog_shard_errors(args.shard)
    catalog_symbol = args.catalog_symbol or f"loom_error_catalog_{args.shard}"
    tables_c = generate_error_catalog_c(
        errors,
        catalog_symbol=catalog_symbol,
        public_header=args.public_header,
        fallback_catalog_symbol=args.fallback_catalog_symbol,
        fallback_public_header=args.fallback_public_header,
    )
    header_h = generate_error_catalog_h(
        errors,
        catalog_symbol=catalog_symbol,
        public_header=args.public_header,
    )
    catalog_json = generate_error_catalog_json(errors)

    if args.source is not None:
        _write_text(args.source, tables_c)
    if args.header is not None:
        _write_text(args.header, header_h)
    if args.catalog is not None:
        _write_text(args.catalog, catalog_json)
    if args.runtime_tables is not None:
        _write_text(args.runtime_tables, generate_error_runtime_tables_inl())

    if args.check:
        print(f"Validated {len(errors)} error definitions in {args.shard} shard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
