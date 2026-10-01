// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Root ownership and target specialization for resolved compiler requests.

#ifndef LOOM_TOOLING_COMPILE_PREPARATION_H_
#define LOOM_TOOLING_COMPILE_PREPARATION_H_

#include "loom/compile/request.h"
#include "loom/tooling/compile/pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes the selected roots and their dependency closure, then
// specializes the selected kernel-product roots for the explicit target. Root
// materialization establishes the deployment ABI independently of any check
// launches in the input module and ensures excluded roots cannot participate in
// specialization. Module products without explicit or excluded roots keep the
// whole module.
//
// The caller owns |*inout_module| on both success and failure. Successful
// transformations may replace it and free the previous module. Source storage
// referenced by |sources| must outlive this call and the module. Its table
// follows module replacements and its arena owns the projected entries.
// Specialization diagnostics are counted in |out_error_count|; status
// represents allocation, linking, or diagnostic-sink failures.
iree_status_t loom_compile_materialize_request(
    const loom_compile_request_t* request,
    const loom_compile_pipeline_options_t* options,
    loom_source_table_projection_t* sources,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count);

// Runs the selected pipeline for a materialized request. Module products carry
// explicit target specializations for public and retained functions through the
// pipeline to emission. The result owns those function versions and must remain
// alive until the final artifact consumer finishes, including on failure.
iree_status_t loom_compile_run_request_pipeline(
    const loom_compile_request_t* request, loom_module_t* module,
    const loom_compile_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_compile_pipeline_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_PREPARATION_H_
