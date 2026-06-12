// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Loom text parsing and parser/printer round-trips.
//
// Strategy 1 feeds arbitrary bytes directly to the parser. Parse diagnostics
// are expected; infrastructure failures and crashes are not.
//
// Strategy 2 uses the structured IR generator to produce valid modules, prints
// them, parses the printed text, and requires print-parse-print identity. This
// reaches deep valid parser paths that raw byte fuzzing rarely discovers.

#include <cstddef>
#include <cstdint>

#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/testing/context.h"
#include "loom/testing/gen.h"

//===----------------------------------------------------------------------===//
// Shared context
//===----------------------------------------------------------------------===//

static loom_context_t g_context;
static bool g_context_initialized = false;

static void fuzz_ignore_status_or_trap(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static void fuzz_ensure_context(void) {
  if (g_context_initialized) {
    return;
  }
  // Raw parser fuzzing is a production text front door: broad dialect
  // registration lets arbitrary input reach every checked-in op format parser.
  // The generated valid-module strategy stays synthetic through the generator's
  // default test-dialect profile.
  loom_context_initialize(iree_allocator_system(), &g_context);
  fuzz_ignore_status_or_trap(
      loom_testing_context_register_all_dialects(&g_context));
  fuzz_ignore_status_or_trap(loom_context_finalize(&g_context));
  g_context_initialized = true;
}

//===----------------------------------------------------------------------===//
// Text helpers
//===----------------------------------------------------------------------===//

static loom_module_t* fuzz_parse_text(iree_string_view_t source,
                                      iree_string_view_t filename,
                                      iree_arena_block_pool_t* block_pool) {
  loom_text_parse_options_t options = {
      /*.diagnostic_sink=*/{},
      /*.max_errors=*/16,
  };
  loom_module_t* module = NULL;
  fuzz_ignore_status_or_trap(loom_text_parse(source, filename, &g_context,
                                             block_pool, &options, &module));
  return module;
}

static void fuzz_print_module(const loom_module_t* module,
                              iree_string_builder_t* builder) {
  fuzz_ignore_status_or_trap(loom_text_print_module_to_builder(
      module, builder, LOOM_TEXT_PRINT_DEFAULT));
}

static void fuzz_assert_round_trip(iree_string_view_t text,
                                   iree_string_view_t filename,
                                   iree_arena_block_pool_t* block_pool) {
  loom_module_t* module = fuzz_parse_text(text, filename, block_pool);
  if (!module) {
    __builtin_trap();
  }

  iree_string_builder_t printed_builder;
  iree_string_builder_initialize(iree_allocator_system(), &printed_builder);
  fuzz_print_module(module, &printed_builder);
  loom_module_free(module);

  iree_string_view_t printed_text =
      iree_make_string_view(iree_string_builder_buffer(&printed_builder),
                            iree_string_builder_size(&printed_builder));
  loom_module_t* reparsed = fuzz_parse_text(printed_text, filename, block_pool);
  if (!reparsed) {
    __builtin_trap();
  }

  iree_string_builder_t reprinted_builder;
  iree_string_builder_initialize(iree_allocator_system(), &reprinted_builder);
  fuzz_print_module(reparsed, &reprinted_builder);
  loom_module_free(reparsed);

  iree_string_view_t reprinted_text =
      iree_make_string_view(iree_string_builder_buffer(&reprinted_builder),
                            iree_string_builder_size(&reprinted_builder));
  if (!iree_string_view_equal(printed_text, reprinted_text)) {
    __builtin_trap();
  }

  iree_string_builder_deinitialize(&reprinted_builder);
  iree_string_builder_deinitialize(&printed_builder);
}

//===----------------------------------------------------------------------===//
// Raw parse strategy
//===----------------------------------------------------------------------===//

static void fuzz_strategy_raw_parse(const uint8_t* data, size_t size) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  iree_string_view_t source = {
      /*.data=*/(const char*)data,
      /*.size=*/(iree_host_size_t)size,
  };
  loom_module_t* module =
      fuzz_parse_text(source, IREE_SV("<raw-fuzz>"), &block_pool);
  if (module) {
    iree_string_builder_t printed_builder;
    iree_string_builder_initialize(iree_allocator_system(), &printed_builder);
    fuzz_print_module(module, &printed_builder);
    loom_module_free(module);

    iree_string_view_t printed_text =
        iree_make_string_view(iree_string_builder_buffer(&printed_builder),
                              iree_string_builder_size(&printed_builder));
    fuzz_assert_round_trip(printed_text, IREE_SV("<raw-fuzz-roundtrip>"),
                           &block_pool);
    iree_string_builder_deinitialize(&printed_builder);
  }

  iree_arena_block_pool_deinitialize(&block_pool);
}

static void fuzz_strategy_generated_roundtrip(const uint8_t* data,
                                              size_t size) {
  if (size < 2) return;

  uint8_t preset = data[0];
  uint8_t scale = (uint8_t)((data[1] % 5) + 1);
  const uint8_t* random_data = data + 2;
  size_t random_size = size - 2;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_fuzz_preset(preset, scale);
  loom_test_gen_t generator;
  loom_test_gen_initialize_fuzz(random_data, random_size, &generator);

  loom_module_t* generated_module = NULL;
  fuzz_ignore_status_or_trap(loom_test_gen_module(
      &generator, &config, &g_context, &block_pool, &generated_module));
  if (!generated_module) __builtin_trap();

  iree_string_builder_t printed_builder;
  iree_string_builder_initialize(iree_allocator_system(), &printed_builder);
  fuzz_print_module(generated_module, &printed_builder);
  loom_module_free(generated_module);

  iree_string_view_t printed_text =
      iree_make_string_view(iree_string_builder_buffer(&printed_builder),
                            iree_string_builder_size(&printed_builder));
  fuzz_assert_round_trip(printed_text, IREE_SV("<generated-fuzz>"),
                         &block_pool);

  iree_string_builder_deinitialize(&printed_builder);
  iree_arena_block_pool_deinitialize(&block_pool);
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  fuzz_ensure_context();
  fuzz_strategy_raw_parse(data, size);
  fuzz_strategy_generated_roundtrip(data, size);
  return 0;
}
