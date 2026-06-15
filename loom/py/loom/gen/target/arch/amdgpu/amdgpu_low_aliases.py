# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU blocked low-alias metadata -> C diagnostic table."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx117x_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
    _gfx1250_core_overlays,
)
from loom.target.arch.amdgpu.low_aliases import (  # noqa: E402
    AmdgpuBlockedLowAlias,
    sorted_amdgpu_blocked_low_aliases,
    validate_amdgpu_blocked_low_aliases,
)


@dataclass(frozen=True, slots=True)
class _AliasLookupRow:
    lookup_name: str
    alias: AmdgpuBlockedLowAlias


def _padded_arg(value: str, width: int) -> str:
    return f"{value},{' ' * (width - len(value) + 1)}"


def _alias_lookup_rows(
    aliases: Sequence[AmdgpuBlockedLowAlias],
) -> tuple[_AliasLookupRow, ...]:
    rows = tuple(_AliasLookupRow(lookup_name=lookup_name, alias=alias) for alias in aliases for lookup_name in alias.lookup_names)
    return tuple(sorted(rows, key=lambda row: row.lookup_name))


def _amdgpu_core_overlay_sets():
    return (
        _gfx940_core_overlays(),
        _gfx950_core_overlays(),
        _gfx11_core_overlays(),
        _gfx117x_core_overlays(),
        _gfx12_core_overlays(),
        _gfx1250_core_overlays(),
    )


def _validate_alias_descriptor_contracts(
    aliases: Sequence[AmdgpuBlockedLowAlias],
) -> None:
    blocked_descriptor_keys = {alias.descriptor_key for alias in aliases}
    for descriptor_set in _amdgpu_core_overlay_sets():
        descriptors = {descriptor.descriptor_key: descriptor for descriptor in descriptor_set}
        authored_aliases = blocked_descriptor_keys & descriptors.keys()
        if authored_aliases:
            formatted_aliases = ", ".join(sorted(authored_aliases))
            raise ValueError(f"AMDGPU descriptor set {descriptor_set.key} authors blocked low aliases: {formatted_aliases}")
        for alias in aliases:
            descriptor = descriptors.get(alias.replacement_descriptor_key)
            if descriptor is None:
                raise ValueError(f"AMDGPU descriptor set {descriptor_set.key} lacks replacement descriptor {alias.replacement_descriptor_key}")
            if descriptor.mnemonic != alias.replacement_mnemonic:
                raise ValueError(
                    f"AMDGPU descriptor set {descriptor_set.key} replacement "
                    f"descriptor {alias.replacement_descriptor_key} has "
                    f"mnemonic {descriptor.mnemonic!r}, expected "
                    f"{alias.replacement_mnemonic!r}"
                )


def _emit_alias_rows(rows: Sequence[_AliasLookupRow]) -> list[str]:
    lookup_name_width = max(len(_c_string_arg(row.lookup_name)) for row in rows)
    mnemonic_width = max(len(_c_string_arg(row.alias.asm_mnemonic)) for row in rows)
    semantics_width = max(len(_c_string_arg(row.alias.alias_semantics)) for row in rows)
    replacement_key_width = max(len(_c_string_arg(row.alias.replacement_descriptor_key)) for row in rows)
    replacement_mnemonic_width = max(len(_c_string_arg(row.alias.replacement_mnemonic)) for row in rows)
    decision_width = max(len(_c_string_arg(row.alias.decision)) for row in rows)
    lines = [
        "#define LOOM_AMDGPU_LOW_BLOCKED_ALIAS(lookup_name_, alias_mnemonic_, alias_semantics_, replacement_descriptor_name_, replacement_mnemonic_, decision_, reason_) \\",
        "  { \\",
        "    .alias_name = IREE_SVL(lookup_name_), \\",
        "    .alias_mnemonic = IREE_SVL(alias_mnemonic_), \\",
        "    .alias_semantics = IREE_SVL(alias_semantics_), \\",
        "    .replacement_descriptor_name = IREE_SVL(replacement_descriptor_name_), \\",
        "    .replacement_mnemonic = IREE_SVL(replacement_mnemonic_), \\",
        "    .decision = IREE_SVL(decision_), \\",
        "    .reason = IREE_SVL(reason_), \\",
        "  }",
        "",
        "static const loom_amdgpu_low_blocked_alias_t",
        "    kLoomAmdgpuLowBlockedAliases[] = {",
        "  // lookup_name                      alias_mnemonic        semantics replacement_descriptor_name replacement_mnemonic decision reason",
    ]

    def alias_row_expr(row: _AliasLookupRow) -> str:
        lookup_name = _c_string_arg(row.lookup_name)
        alias_mnemonic = _c_string_arg(row.alias.asm_mnemonic)
        alias_semantics = _c_string_arg(row.alias.alias_semantics)
        replacement_descriptor_key = _c_string_arg(row.alias.replacement_descriptor_key)
        replacement_mnemonic = _c_string_arg(row.alias.replacement_mnemonic)
        decision = _c_string_arg(row.alias.decision)
        reason = _c_string_arg(row.alias.reason)
        return (
            "  LOOM_AMDGPU_LOW_BLOCKED_ALIAS("
            f"{_padded_arg(lookup_name, lookup_name_width)}"
            f"{_padded_arg(alias_mnemonic, mnemonic_width)}"
            f"{_padded_arg(alias_semantics, semantics_width)}"
            f"{_padded_arg(replacement_descriptor_key, replacement_key_width)}"
            f"{_padded_arg(replacement_mnemonic, replacement_mnemonic_width)}"
            f"{_padded_arg(decision, decision_width)}"
            f"{reason}),"
        )

    lines.extend(alias_row_expr(row) for row in rows)
    lines.extend(["};", "", "#undef LOOM_AMDGPU_LOW_BLOCKED_ALIAS", ""])
    return lines


def _emit_source(aliases: Sequence[AmdgpuBlockedLowAlias]) -> str:
    rows = _alias_lookup_rows(aliases)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_low_aliases"),
        "",
        '#include "loom/target/arch/amdgpu/low_aliases.h"',
        "",
        "// clang-format off",
    ]
    lines.extend(_emit_alias_rows(rows))
    lines.extend(
        [
            "// clang-format on",
            "",
            "const loom_amdgpu_low_blocked_alias_t*",
            "loom_amdgpu_low_blocked_alias_lookup(iree_string_view_t name) {",
            "  for (iree_host_size_t i = 0;",
            "       i < IREE_ARRAYSIZE(kLoomAmdgpuLowBlockedAliases); ++i) {",
            "    const loom_amdgpu_low_blocked_alias_t* alias =",
            "        &kLoomAmdgpuLowBlockedAliases[i];",
            "    if (iree_string_view_equal(name, alias->alias_name)) {",
            "      return alias;",
            "    }",
            "  }",
            "  return NULL;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_low_aliases_to_path(source_path: Path) -> None:
    aliases = sorted_amdgpu_blocked_low_aliases()
    validate_amdgpu_blocked_low_aliases(aliases)
    _validate_alias_descriptor_contracts(aliases)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(_emit_source(aliases), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU blocked low-alias C tables.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated blocked low-alias source path.",
    )
    args = parser.parse_args(argv)
    write_low_aliases_to_path(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
