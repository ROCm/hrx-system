// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated source workloads for source-to-low tests and fuzzers.
//
// This is synthetic target-low test infrastructure: it builds ordinary Loom
// source functions targeted to `test.target<low_core>` without linking or
// querying any concrete target descriptor registry. Consumers choose a target
// provider separately when they lower or execute the generated module.

#ifndef LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_
#define LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;
typedef struct loom_context_t loom_context_t;
typedef struct loom_module_t loom_module_t;

typedef struct loom_low_source_workload_config_t {
  // Module name assigned to the generated compilation unit. Empty uses a stable
  // default.
  iree_string_view_t module_name;
  // Symbol name for the generated target record. Empty uses a stable default.
  iree_string_view_t target_symbol_name;
  // Symbol name for the generated source function. Empty uses a stable default.
  iree_string_view_t function_symbol_name;
  // Number of non-structural source ops to emit in the generated function body.
  // CFG branches and the final return are emitted by the body shape itself.
  uint16_t op_count;
} loom_low_source_workload_config_t;

typedef struct loom_low_source_workload_counts_t {
  // Number of generated scalar integer arithmetic ops.
  uint32_t scalar_integer_op_count;
  // Number of generated scalar floating-point arithmetic ops.
  uint32_t scalar_float_op_count;
  // Number of generated scalar constants.
  uint32_t scalar_constant_count;
  // Number of generated lanewise vector integer arithmetic ops.
  uint32_t vector_integer_op_count;
  // Number of generated lanewise vector floating-point arithmetic ops.
  uint32_t vector_float_op_count;
  // Number of generated vector.reduce ops.
  uint32_t vector_reduce_op_count;
  // Number of generated f32 vector.reduce ops. This is a subset of
  // vector_reduce_op_count and is not included separately in the total count.
  uint32_t vector_float_reduce_op_count;
  // Number of generated vector dot ops.
  uint32_t vector_dot_op_count;
  // Number of generated vector.extract ops.
  uint32_t vector_extract_op_count;
  // Number of generated vector.shuffle ops.
  uint32_t vector_shuffle_op_count;
  // Number of generated vector.cmpi ops.
  uint32_t vector_cmpi_op_count;
  // Number of generated vector.select ops.
  uint32_t vector_select_op_count;
  // Number of generated vector.load ops.
  uint32_t vector_load_op_count;
  // Number of generated vector.load ops returning floating-point vectors. This
  // is a subset of vector_load_op_count and is not included separately in the
  // total count.
  uint32_t vector_float_load_op_count;
  // Number of generated vector.load ops with dynamic indices. This is a subset
  // of vector_load_op_count and is not included separately in the total count.
  uint32_t vector_dynamic_load_op_count;
  // Number of generated vector.store ops.
  uint32_t vector_store_op_count;
  // Number of generated vector.store ops storing floating-point vectors. This
  // is a subset of vector_store_op_count and is not included separately in the
  // total count.
  uint32_t vector_float_store_op_count;
  // Number of generated vector.store ops with dynamic indices. This is a subset
  // of vector_store_op_count and is not included separately in the total count.
  uint32_t vector_dynamic_store_op_count;
  // Number of generated index.madd ops.
  uint32_t index_madd_op_count;
  // Number of generated cfg.cond_br ops.
  uint32_t cfg_cond_branch_count;
  // Number of generated cfg.br ops.
  uint32_t cfg_branch_count;
} loom_low_source_workload_counts_t;

// Returns a generated source-to-low workload config.
// Scale 1 produces a compact body; larger scales increase only the op count.
loom_low_source_workload_config_t loom_low_source_workload_config_make(
    uint32_t scale);

// Registers the exact source and low dialects used by generated source-low
// workloads. Callers own context initialization and finalization so they can
// compose this with target-owned descriptor/policy registries without pulling
// the full production op registry into fuzzers and stress tests.
iree_status_t loom_low_source_workload_register_dialects(
    loom_context_t* context);

// Generates one targeted source function using a deterministic PRNG seed.
iree_status_t loom_low_source_workload_generate_seeded_module(
    uint64_t seed, const loom_low_source_workload_config_t* config,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module);

// Generates one targeted source function using fuzzer-provided bytes.
iree_status_t loom_low_source_workload_generate_fuzz_module(
    const uint8_t* data, iree_host_size_t data_length,
    const loom_low_source_workload_config_t* config, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, loom_module_t** out_module);

// Counts source ops emitted by generated source workloads.
void loom_low_source_workload_count_func_ops(
    const loom_module_t* module, const loom_op_t* func_op,
    loom_low_source_workload_counts_t* out_counts);

// Adds |source_counts| into |target_counts|.
void loom_low_source_workload_counts_accumulate(
    loom_low_source_workload_counts_t* target_counts,
    const loom_low_source_workload_counts_t* source_counts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_H_
