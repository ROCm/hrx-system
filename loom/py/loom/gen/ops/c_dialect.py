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
from loom.gen.ops.c_names import c_dialect_include_path as _c_dialect_include_path
from loom.gen.ops.c_ops_header import generate_ops_h
from loom.gen.ops.model import DialectGeneration

__all__ = [
    "generate_dialect_contents",
    "generate_ops_h",
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
            generation.table_shards,
            include_path=include_path,
        )
        if generation.table_shards is not None
        else {
            "tables.c": generate_tables_c(
                dialect.name,
                dialect.dialect_id,
                generation.ops,
                include_path=include_path,
            )
        }
    )
    return {
        "ops.h": generate_ops_h(dialect.name, dialect.dialect_id, generation.ops),
        "builders.c": c_builders.generate_builders_c(
            dialect.name,
            generation.ops,
            include_path=include_path,
        ),
        **table_files,
    }
