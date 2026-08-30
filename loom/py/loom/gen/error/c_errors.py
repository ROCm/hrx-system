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
from loom.gen.support.c import c_string_literal  # noqa: E402
from loom.gen.support.files import write_text_file as _write_text  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.support.string_pool import CStringPool  # noqa: E402


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


def _catalog_definitions_symbol(catalog_symbol: str) -> str:
    return f"{catalog_symbol}_definitions"


def _catalog_param_defs_symbol(catalog_symbol: str) -> str:
    return f"{catalog_symbol}_param_defs"


def _catalog_error_indices_symbol(catalog_symbol: str) -> str:
    return f"{catalog_symbol}_error_indices"


def _catalog_string_data_symbol(catalog_symbol: str) -> str:
    return f"{catalog_symbol}_string_data"


def _param_kind_c_name(kind: ParamKind) -> str:
    return f"LOOM_PARAM_{kind.name}"


def _severity_c_name(severity: Severity) -> str:
    return f"LOOM_DIAGNOSTIC_{severity.name}"


def _domain_c_name(domain: ErrorDomain) -> str:
    return f"LOOM_ERROR_DOMAIN_{domain.name}"


def _emitter_c_name(emitter: Emitter) -> str:
    return f"LOOM_EMITTER_{emitter.name}"


def _error_string_label(error: ErrorDef, field: str) -> str:
    return f"{_c_symbol(error)}_{field}"


def _error_param_string_label(error: ErrorDef, index: int) -> str:
    return f"{_c_symbol(error)}_param_{index}_{error.params[index].name}"


def _emit_cstring_table(pool: CStringPool, data_name: str) -> list[str]:
    """Emits one packed NUL-terminated C string table and offset constants."""
    if not pool.entries:
        return []
    lines = [
        "// clang-format off",
        f"static const char {data_name}[] =",
    ]
    for entry in pool.entries:
        if "\0" in entry.value:
            raise ValueError(f"error catalog string {entry.label!r} contains NUL")
        lines.append(f'    "{c_string_literal(entry.value)}\\0"')
    lines[-1] += ";"
    lines.extend(["// clang-format on", "", "enum {"])
    for index, entry in enumerate(pool.entries):
        enum_name = pool.enum_name(entry.label)
        if index == 0:
            lines.append(f"  {enum_name} = 0,")
        else:
            previous_entry = pool.entries[index - 1]
            previous_enum_name = pool.enum_name(previous_entry.label)
            previous_value = c_string_literal(previous_entry.value)
            lines.append(f'  {enum_name} = {previous_enum_name} + sizeof("{previous_value}"),')
    previous_entry = pool.entries[-1]
    previous_enum_name = pool.enum_name(previous_entry.label)
    previous_value = c_string_literal(previous_entry.value)
    lines.extend(
        [
            f'  {pool.c_enum_prefix}_STRING_END = {previous_enum_name} + sizeof("{previous_value}"),',
            "};",
            "",
        ]
    )
    return lines


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
    """Generates a compact error catalog source file."""
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

    ordered_errors = sorted(errors, key=lambda error: (error.domain.value, error.code))
    grouped_errors = _group_errors_by_domain(ordered_errors)
    if len(ordered_errors) >= (1 << 16) - 1:
        raise ValueError("error catalog has too many definitions for uint16 indices")

    string_prefix = catalog_symbol.upper()
    string_pool = CStringPool(string_prefix, max_payload_length=None)
    for error in ordered_errors:
        string_pool.intern(_error_string_label(error, "id"), error.error_id)
        string_pool.intern(_error_string_label(error, "summary"), error.summary)
        string_pool.intern(_error_string_label(error, "message"), error.message)
        if error.fix_hint:
            string_pool.intern(_error_string_label(error, "fix_hint"), error.fix_hint)
        if len(error.params) > 255:
            raise ValueError(f"{error.error_id}: too many diagnostic parameters")
        for param_index, param in enumerate(error.params):
            if param.kind.value >= (1 << 3):
                raise ValueError(f"{error.error_id}: parameter kind exceeds compact encoding")
            string_pool.intern(_error_param_string_label(error, param_index), param.name)
    if string_pool.next_offset >= (1 << 29):
        raise ValueError("error catalog string data exceeds compact parameter offset encoding")

    string_data_symbol = _catalog_string_data_symbol(catalog_symbol)
    lines.extend(_emit_cstring_table(string_pool, string_data_symbol))

    param_defs_symbol = _catalog_param_defs_symbol(catalog_symbol)
    param_starts: dict[tuple[ErrorDomain, int], int] = {}
    param_count = 0
    for error in ordered_errors:
        param_starts[(error.domain, error.code)] = param_count
        param_count += len(error.params)
    if param_count >= (1 << 16):
        raise ValueError("error catalog has too many parameter definitions")
    if param_count:
        lines.append(f"static const loom_error_param_def_t {param_defs_symbol}[] = {{")
        for error in ordered_errors:
            for param_index, param in enumerate(error.params):
                lines.append(f"    LOOM_ERROR_PARAM_DEF({string_pool.ref(_error_param_string_label(error, param_index))}, {_param_kind_c_name(param.kind)}),")
        lines.extend(["};", ""])

    definitions_symbol = _catalog_definitions_symbol(catalog_symbol)
    error_indices = {(error.domain, error.code): index for index, error in enumerate(ordered_errors)}
    if ordered_errors:
        lines.append(f"const loom_error_def_t {definitions_symbol}[] = {{")
        for error_index, error in enumerate(ordered_errors):
            lines.append(f"    [{error_index}] = {{")
            lines.append(f"        .catalog = &{catalog_symbol},")
            lines.append(f"        .error_id_offset = {string_pool.ref(_error_string_label(error, 'id'))},")
            lines.append(f"        .summary_offset = {string_pool.ref(_error_string_label(error, 'summary'))},")
            lines.append(f"        .message_template_offset = {string_pool.ref(_error_string_label(error, 'message'))},")
            if error.fix_hint:
                fix_hint_offset = string_pool.ref(_error_string_label(error, "fix_hint"))
            else:
                fix_hint_offset = "LOOM_ERROR_STRING_OFFSET_NONE"
            lines.append(f"        .fix_hint_template_offset = {fix_hint_offset},")
            lines.append(f"        .param_start = {param_starts[(error.domain, error.code)]},")
            lines.append(f"        .ref = LOOM_ERROR_REF({_domain_c_name(error.domain)}, {error.code}),")
            lines.append(f"        .severity = {_severity_c_name(error.severity)},")
            lines.append(f"        .param_count = {len(error.params)},")
            lines.append("    },")
        lines.extend(["};", ""])

    code_indices: list[str] = []
    domain_spans: dict[ErrorDomain, tuple[int, int]] = {}
    for domain, domain_errors in grouped_errors.items():
        max_code = max(error.code for error in domain_errors)
        code_index_start = len(code_indices)
        domain_indices = ["UINT16_MAX"] * (max_code + 1)
        for error in domain_errors:
            domain_indices[error.code] = str(error_indices[(error.domain, error.code)])
        code_indices.extend(domain_indices)
        domain_spans[domain] = (code_index_start, len(domain_indices))
    if len(code_indices) >= (1 << 16):
        raise ValueError("error catalog code index table exceeds uint16 spans")

    error_indices_symbol = _catalog_error_indices_symbol(catalog_symbol)
    if code_indices:
        lines.append(f"static const uint16_t {error_indices_symbol}[] = {{")
        for index, error_index in enumerate(code_indices):
            lines.append(f"    [{index}] = {error_index},")
        lines.extend(["};", ""])

    lines.append(f"const loom_error_catalog_t {catalog_symbol} = {{")
    lines.append(f"    .string_data = {string_data_symbol if string_pool.entries else 'NULL'},")
    lines.append(f"    .error_defs = {definitions_symbol if ordered_errors else 'NULL'},")
    lines.append(f"    .param_defs = {param_defs_symbol if param_count else 'NULL'},")
    lines.append(f"    .error_indices_by_code = {error_indices_symbol if code_indices else 'NULL'},")
    if fallback_catalog_symbol is not None:
        lines.append(f"    .fallback_catalog = &{fallback_catalog_symbol},")
    else:
        lines.append("    .fallback_catalog = NULL,")
    lines.append("    .domain_spans = {")
    for domain, (code_index_start, code_count) in domain_spans.items():
        lines.append(f"        [{_domain_c_name(domain)}] = {{")
        lines.append(f"            .code_index_start = {code_index_start},")
        lines.append(f"            .code_count = {code_count},")
        lines.append("        },")
    lines.append("    },")
    lines.append("};")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_error_catalog_h(errors: list[ErrorDef], *, catalog_symbol: str, public_header: str) -> str:
    """Generates error_catalog.h with canonical C error references."""

    seen_def_macros: set[str] = set()
    seen_ref_macros: set[str] = set()
    ordered_errors = sorted(errors, key=lambda error: (error.domain.value, error.code))
    definitions_symbol = _catalog_definitions_symbol(catalog_symbol)
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
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f"extern const loom_error_catalog_t {catalog_symbol};",
        f"extern const loom_error_def_t {definitions_symbol}[];",
        "",
    ]
    for error_index, error in enumerate(ordered_errors):
        def_macro = _error_def_macro(error)
        ref_macro = _error_ref_macro(error)
        if def_macro in seen_def_macros:
            raise ValueError(f"{error.error_id}: duplicate error def macro {def_macro}")
        if ref_macro in seen_ref_macros:
            raise ValueError(f"{error.error_id}: duplicate error ref macro {ref_macro}")
        seen_def_macros.add(def_macro)
        seen_ref_macros.add(ref_macro)
        lines.append(f"// {error.error_id}: {error.summary}")
        lines.append(f"#define {def_macro} (&{definitions_symbol}[{error_index}])")
        lines.append(f"#define {ref_macro} \\")
        lines.append(f"  LOOM_ERROR_REF({_domain_c_name(error.domain)}, {error.code})")
        lines.append("")
    lines.extend(
        [
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
        ]
    )
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
