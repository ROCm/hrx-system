// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly execution profile for the shared scenario testbench.

#ifndef LOOM_TOOLING_TARGET_WASM_TESTBENCH_H_
#define LOOM_TOOLING_TARGET_WASM_TESTBENCH_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/compiled_provider.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler inputs shared by prepared Wasm scenario products.
//
// Each unique product compiles and instantiates one ordinary function during
// scenario preparation. Trial domains calling the same function share that
// instance. Execution only transports raw scalar payloads and complete
// physical allocation roots through the synchronous host boundary.
typedef struct loom_wasm_testbench_t {
  // Borrowed public compiler inputs, live through the final product prepare.
  loom_testbench_compilation_t compilation;
  // Observer receiving each complete public compiler result.
  loom_testbench_compile_result_callback_t result_callback;
  // Owned Wasm SIMD128 profile applied to each scenario product.
  loomc_target_profile_t* target_profile;
  // Allocator for compiler scratch and runtime objects.
  iree_allocator_t host_allocator;
} loom_wasm_testbench_t;

// Initializes an execution profile and selects the Wasm SIMD128 target.
iree_status_t loom_wasm_testbench_initialize(
    loomc_target_environment_t* target_environment,
    iree_allocator_t host_allocator, loom_wasm_testbench_t* out_testbench);

// Releases the selected target profile. Safe for a zero-initialized testbench.
void loom_wasm_testbench_deinitialize(loom_wasm_testbench_t* testbench);

// Binds compilation inputs and returns an eager Wasm execution profile. Every
// unique prepared product owns a compiled and instantiated module. Product
// preparation finishes before trial-local values exist.
loom_testbench_execution_profile_t loom_wasm_testbench_execution_profile(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_WASM_TESTBENCH_H_
