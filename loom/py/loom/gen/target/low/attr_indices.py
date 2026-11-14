# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Named readers for fixed positions in canonical Low dictionaries."""

from __future__ import annotations

from collections.abc import Iterable

from loom.gen.target.low import c_spelling
from loom.target.low_descriptors import Immediate, ImmediateFlag, target_relative_name


def _immediate_attr_indices(immediates: tuple[Immediate, ...]) -> dict[tuple[str, bool], int]:
    indices: dict[tuple[str, bool], int] = {}
    for index, immediate in enumerate(sorted(immediates, key=lambda value: value.field_name)):
        optional = ImmediateFlag.DEFAULT_VALUE in immediate.flags
        if optional and index + 1 != len(immediates):
            # A later field makes presence at this index ambiguous. A trailing
            # optional field is present exactly when dictionary count > index.
            break
        indices[(immediate.field_name, optional)] = index
    return indices


def emit_attr_accessors(
    c_enum_prefix: str,
    descriptor_immediates: Iterable[tuple[str, tuple[Immediate, ...]]],
    *,
    target_key: str | None,
) -> list[str]:
    """Emits dictionary readers shared by descriptor variants.

    A target's common header can pass multiple schemas for the same descriptor,
    independently of the descriptor data selected for a particular build.
    Only fields with a fixed, identical position in every variant are exposed.
    A trailing optional field returns ABSENT when omitted. Storage positions
    remain implementation details of the named readers.
    """

    descriptor_indices: dict[str, dict[tuple[str, bool], int]] = {}
    for key, immediates in descriptor_immediates:
        indices = _immediate_attr_indices(immediates)
        if key in descriptor_indices:
            indices = {name: index for name, index in descriptor_indices[key].items() if indices.get(name) == index}
        descriptor_indices[key] = indices

    lines: list[str] = []
    accessor_fields: dict[str, tuple[str, str]] = {}
    prefix = c_enum_prefix.lower()
    if not prefix.startswith("loom_"):
        prefix = f"loom_{prefix}"
    for key, indices in descriptor_indices.items():
        if not indices:
            continue
        descriptor_name = c_spelling.c_identifier(target_relative_name(target_key, key)).lower()
        lines.append(f"// Canonical dictionary fields for {key}.")
        for (field_name, optional), index in indices.items():
            accessor = f"{prefix}_{descriptor_name}_{c_spelling.c_identifier(field_name)}"
            if accessor in accessor_fields:
                previous_key, previous_field = accessor_fields[accessor]
                raise ValueError(f"attribute accessor '{accessor}' collides between '{previous_key}.{previous_field}' and '{key}.{field_name}'")
            accessor_fields[accessor] = (key, field_name)
            lines.append(f"#define {accessor}(attributes) \\")
            reader = "loom_low_optional_immediate_attr" if optional else "loom_low_immediate_attr"
            lines.append(f"  {reader}((attributes), {index})")
        lines.append("")
    return lines
