# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: target-low descriptor inputs -> dense C tables.

The generator consumes a rich, explicit Python schema and emits compact
runtime tables under loom/src/loom. The C build only sees dense .rodata
arrays; Python owns source readability, validation, and allowlist closure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from loom.gen.target.low import c_emit, compiler, views
from loom.gen.target.low.compiled import DescriptorAllowlist, GeneratedDescriptorSet
from loom.target.low_descriptors import DescriptorSet


def generate_descriptor_set(
    spec: DescriptorSet,
    allowlist: DescriptorAllowlist | None = None,
) -> GeneratedDescriptorSet:
    compiled = compiler.compile_descriptor_set(spec, allowlist)
    return GeneratedDescriptorSet(
        header=c_emit.emit_header(compiled),
        source=c_emit.emit_source(compiled),
    )


def generate_descriptor_set_shared_source(
    storage_spec: DescriptorSet,
    view_specs: Sequence[DescriptorSet],
) -> str:
    """Generates one C source containing shared storage and multiple set views.

    Each view selects descriptors from |storage_spec| by stable key. Supporting
    tables are shared as a storage superset, while descriptor, operand-form, and
    asm-form tables are reused only when the selected view surface matches the
    storage rows. A view may provide its own asm forms for the same descriptor
    keys; those forms are compiled and validated against the shared storage
    vocabulary during generation.
    """

    compiled = compiler.compile_descriptor_set(
        storage_spec,
        allowlist=None,
        allow_ambiguous_asm_mnemonics=True,
    )
    descriptor_set_views = tuple(views.descriptor_set_view_for_spec(compiled, view_spec) for view_spec in view_specs)
    return c_emit.emit_source_for_views(compiled, views=descriptor_set_views)


def generate_descriptor_set_shared_header(
    storage_spec: DescriptorSet,
    view_spec: DescriptorSet,
) -> str:
    """Generates a public view header for a shared descriptor storage source."""

    compiled = compiler.compile_descriptor_set(
        storage_spec,
        allowlist=None,
        allow_ambiguous_asm_mnemonics=True,
    )
    views.descriptor_set_view_for_spec(compiled, view_spec)
    return c_emit.emit_header_for_spec(compiled, view_spec)


def write_descriptor_set(spec: DescriptorSet, allowlist: DescriptorAllowlist | None = None) -> None:
    from loom.gen import bootstrap as _bootstrap

    generated = generate_descriptor_set(spec, allowlist)
    (_bootstrap.REPO_ROOT / spec.c_header_path).write_text(generated.header, encoding="utf-8")
    (_bootstrap.REPO_ROOT / spec.c_source_path).write_text(generated.source, encoding="utf-8")


def write_descriptor_set_to_paths(
    spec: DescriptorSet,
    *,
    header_path: Path,
    source_path: Path,
    allowlist: DescriptorAllowlist | None = None,
) -> None:
    generated = generate_descriptor_set(spec, allowlist)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(generated.header, encoding="utf-8")
    source_path.write_text(generated.source, encoding="utf-8")


def main() -> None:
    from loom.gen.target.arch.x86.x86_packed_dot_contract import write_x86_packed_dot_contract_header
    from loom.target.descriptor_sets import iter_checked_in_c_descriptor_sets

    descriptor_sets = tuple(iter_checked_in_c_descriptor_sets())
    for descriptor_set in descriptor_sets:
        write_descriptor_set(descriptor_set)
    write_x86_packed_dot_contract_header()
    print(f"Generated {len(descriptor_sets)} low descriptor sets")


if __name__ == "__main__":
    main()
