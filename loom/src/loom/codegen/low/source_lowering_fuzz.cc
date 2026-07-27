// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzzes generated source-to-low lowering through the synthetic test-low
// target.
//
// The first byte selects a bounded workload scale. Remaining bytes drive the
// structured source generator. Every generated module is expected to verify,
// lower, low-verify, packetize, schedule, and allocate. A failure means either
// the workload generator produced IR outside its claimed contract or the low
// pipeline mishandled a valid source program.

#include <cstddef>
#include <cstdint>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/codegen/low/testing/source_workload_pipeline.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"

static loom_context_t g_context;
static loom_target_low_descriptor_registry_t g_descriptor_registry;
static loom_low_lower_policy_registry_t g_policy_registry;
static bool g_initialized = false;

static void trap_with_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static void ensure_context(void) {
  if (g_initialized) {
    return;
  }
  loom_context_initialize(iree_allocator_system(), &g_context);
  iree_status_t status = loom_low_source_workload_register_dialects(&g_context);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&g_context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(&g_context);
    trap_with_status(status);
  }
  loom_test_low_descriptor_registry_initialize(&g_descriptor_registry);
  loom_test_low_lower_policy_registry_initialize(&g_policy_registry);
  g_initialized = true;
}

static iree_status_t fuzz_one_input(const uint8_t* data, size_t size) {
  uint32_t scale = 1;
  if (size > 0) {
    scale = (uint32_t)(data[0] % 4) + 1;
    ++data;
    --size;
  }

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_low_source_workload_config_t workload_config =
      loom_low_source_workload_config_make(scale);

  loom_module_t* module = NULL;
  iree_status_t status = loom_low_source_workload_generate_fuzz_module(
      data, size, &workload_config, &g_context, &block_pool, &module);
  if (iree_status_is_ok(status)) {
    const loom_low_source_workload_pipeline_options_t pipeline_options = {
        /*.pass_registry=*/loom_pass_builtin_registry(),
        /*.descriptor_registry=*/&g_descriptor_registry.registry,
        /*.policy_registry=*/&g_policy_registry,
        /*.schedule_strategy=*/LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
    };
    loom_low_source_workload_pipeline_counters_t counters = {};
    status = loom_low_source_workload_run_pipeline(module, &pipeline_options,
                                                   &block_pool, &counters);
  }
  if (module) {
    loom_module_free(module);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  ensure_context();
  trap_with_status(fuzz_one_input(data, size));
  return 0;
}
