// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for the structured IR generator.
//
// The generator's contract is: given any byte sequence as input, it
// produces valid loom IR. This fuzzer verifies that contract by
// feeding arbitrary bytes through the generator in fuzz mode and
// running the verifier on the result. A verification failure means
// the generator produced invalid IR — a generator bug.
//
// The first two bytes select the config preset and scale. Remaining
// bytes drive all structural decisions in the generator. When bytes
// are exhausted, the generator falls back to zeros (deterministic
// degenerate case).
//
// This fuzzer does NOT test downstream consumers (printer, parser,
// passes, bytecode). Those have their own fuzz targets where the
// generator serves as input source.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <cstddef>
#include <cstdint>

#include "loom/error/diagnostic.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/gen.h"
#include "loom/verify/verify.h"

//===----------------------------------------------------------------------===//
// Shared context (initialized once)
//===----------------------------------------------------------------------===//

static loom_context_t g_context;
static bool g_initialized = false;

static void trap_with_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static void ensure_context() {
  if (g_initialized) {
    return;
  }
  loom_context_initialize(iree_allocator_system(), &g_context);

  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables;

  vtables = loom_test_dialect_vtables(&count);
  trap_with_status(loom_context_register_dialect(&g_context, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));

  trap_with_status(loom_context_finalize(&g_context));
  g_initialized = true;
}

//===----------------------------------------------------------------------===//
// Main fuzzer entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) {
    return 0;
  }
  ensure_context();

  // First byte: preset selection. Second byte: scale (clamped to 1-5).
  uint8_t preset_index = data[0] % 5;
  uint8_t scale = (data[1] % 5) + 1;

  // Remaining bytes are the randomness source.
  const uint8_t* fuzz_data = data + 2;
  size_t fuzz_size = size - 2;

  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_fuzz_preset(preset_index, scale);

  // Per-iteration block pool. Freed after each iteration so the fuzzer
  // doesn't accumulate memory across runs.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  // Generate the module from fuzz bytes.
  loom_test_gen_t gen;
  loom_test_gen_initialize_fuzz(fuzz_data, fuzz_size, &gen);

  loom_module_t* module = nullptr;
  iree_status_t status =
      loom_test_gen_module(&gen, &config, &g_context, &block_pool, &module);

  if (iree_status_is_ok(status)) {
    // Verify the generated module. Verification errors mean the generator
    // produced invalid IR — that's a generator bug, so we trap.
    loom_verify_options_t options = {};
    options.sink = {loom_diagnostic_stderr_sink, NULL};
    loom_verify_result_t result = {};
    status = loom_verify_module(module, &options, &result);
    trap_with_status(status);
    if (iree_status_is_ok(status) && result.error_count > 0) {
      __builtin_trap();
    }
    loom_module_free(module);
  } else {
    // Generation itself failed — also a generator bug (it should always
    // succeed on any input).
    trap_with_status(status);
  }

  iree_arena_block_pool_deinitialize(&block_pool);
  return 0;
}
