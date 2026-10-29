# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C dialect artifact generation for Loom ops."""

from __future__ import annotations

from loom.gen.ops import c_builders
from loom.gen.ops.c_metadata_tables import (
    generate_sharded_tables_c,
    generate_tables_aggregator_c,
    generate_tables_c,
    generate_tables_h,
)
from loom.gen.ops.c_names import c_dialect_enum, c_dialect_path
from loom.gen.ops.c_names import c_dialect_include_path as _c_dialect_include_path
from loom.gen.ops.c_ops_header import generate_ops_h, generate_ops_inc
from loom.gen.ops.c_parameterized_types import parameterized_type_defs
from loom.gen.ops.model import DialectGeneration
from loom.gen.ops.type_registry import generate_dialect_type_registry, generate_type_builders

__all__ = [
    "generate_dialect_contents",
    "generate_ops_h",
    "generate_ops_inc",
    "generate_sharded_tables_c",
    "generate_tables_aggregator_c",
    "generate_tables_c",
    "generate_tables_h",
]


def generate_dialect_contents(generation: DialectGeneration) -> dict[str, str]:
    """Returns generated file contents keyed relative to the dialect C directory."""
    dialect = generation.dialect
    include_path = _c_dialect_include_path(dialect)
    table_files = (
        generate_sharded_tables_c(
            dialect.name,
            dialect.dialect_id,
            generation.ops,
            generation.table_shards,
            generation.parameterized_attrs,
            generation.encoding_families,
            include_path=include_path,
            referenced_ops=generation.referenced_ops,
        )
        if generation.table_shards is not None
        else {
            "tables.c": generate_tables_c(
                dialect.name,
                dialect.dialect_id,
                generation.ops,
                generation.parameterized_attrs,
                generation.encoding_families,
                include_path=include_path,
                referenced_ops=generation.referenced_ops,
            )
        }
    )
    contents = {
        "ops.h": generate_ops_h(
            dialect.name,
            dialect.dialect_id,
            generation.ops,
            generation.parameterized_attrs,
            generation.encoding_families,
            include_path=include_path,
        ),
        "ops.inc": generate_ops_inc(generation.ops),
        "builders.c": c_builders.generate_builders_c(
            dialect.name,
            generation.ops,
            generation.parameterized_attrs,
            encoding_families=generation.encoding_families,
            include_path=include_path,
        ),
        **table_files,
    }
    if generation.types:
        types_header, types_source = generate_dialect_type_registry(dialect, generation.types)
        contents["types.h"] = types_header
        contents["types.c"] = types_source
        if parameterized_type_defs(generation.types):
            builders_header, builders_source = generate_type_builders(generation.types, path=c_dialect_path(dialect), namespace=c_dialect_enum(dialect.name))
            contents["type_builders.h"] = builders_header
            contents["type_builders.c"] = builders_source
    return contents
