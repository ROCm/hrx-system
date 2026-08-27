// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode symbol metadata and dependency-facet serialization.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_SYMBOL_H_
#define LOOM_FORMAT_BYTECODE_WRITER_SYMBOL_H_

#include "loom/analysis/symbol_references.h"
#include "loom/format/bytecode/writer/body.h"
#include "loom/ir/module_record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepared reference analysis and counts for dependency-facet emission.
typedef struct loom_bytecode_symbol_reference_plan_t {
  // Canonical reference analysis whose linked rows are emitted directly.
  loom_symbol_reference_table_t table;
  // Number of dependency-role occurrences across all source rows.
  iree_host_size_t dependency_count;
  // Number of dependency-role occurrences in the module-root row.
  uint32_t module_dependency_count;
} loom_bytecode_symbol_reference_plan_t;

// Appends the indexed symbol metadata section to |builder|.
iree_status_t loom_bytecode_write_symbols_section(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_ir_region_list_t* ir_regions);

// Builds the canonical dependency-facet analysis and aggregate counts.
iree_status_t loom_bytecode_symbol_reference_plan_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_symbol_reference_plan_t* out_plan);

// Streams module and per-symbol dependency-facet rows.
iree_status_t loom_bytecode_write_symbol_references_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering,
    const loom_bytecode_symbol_reference_plan_t* plan);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_SYMBOL_H_
