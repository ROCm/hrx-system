// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for the Loom bytecode reader.
//
// Arbitrary input must report malformed-bytecode diagnostics without crashing
// or growing unbounded scratch state. Generator-produced valid modules must
// survive C writer -> C reader -> C writer canonical stability.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
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
  // Arbitrary bytecode fuzzing intentionally registers every checked-in
  // dialect so malformed op payloads can reach all production bytecode
  // decoders. Generated valid modules still come from the synthetic test
  // generator profile.
  loom_context_initialize(iree_allocator_system(), &g_context);
  fuzz_ignore_status_or_trap(
      loom_testing_context_register_all_dialects(&g_context));
  fuzz_ignore_status_or_trap(loom_context_finalize(&g_context));
  g_context_initialized = true;
}

//===----------------------------------------------------------------------===//
// Bytecode read/write helpers
//===----------------------------------------------------------------------===//

static iree_status_t fuzz_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  (void)diagnostic;
  uint32_t* diagnostic_count = (uint32_t*)user_data;
  ++*diagnostic_count;
  return iree_ok_status();
}

static std::vector<uint8_t> fuzz_write_module(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool) {
  iree_io_stream_t* stream = NULL;
  fuzz_ignore_status_or_trap(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));

  fuzz_ignore_status_or_trap(
      loom_bytecode_write_module(module, stream, NULL, block_pool));

  iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  std::vector<uint8_t> bytes((size_t)stream_length);
  fuzz_ignore_status_or_trap(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  fuzz_ignore_status_or_trap(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  iree_io_stream_release(stream);
  return bytes;
}

static loom_bytecode_read_options_t fuzz_read_options(
    uint8_t control_byte, uint32_t* diagnostic_count) {
  return loom_bytecode_read_options_t{
      /*.diagnostic_sink=*/
      {
          /*.fn=*/fuzz_capture_diagnostic,
          /*.user_data=*/diagnostic_count,
      },
      /*.verify_module=*/(control_byte & 1) != 0,
      /*.verify_max_errors=*/16,
  };
}

static void fuzz_read_arbitrary_bytecode(const uint8_t* data, size_t size,
                                         uint8_t control_byte,
                                         iree_arena_block_pool_t* block_pool) {
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      fuzz_read_options(control_byte, &diagnostic_count);
  loom_bytecode_read_result_t metadata_result = {0};
  fuzz_ignore_status_or_trap(loom_bytecode_read_metadata(
      iree_make_const_byte_span(data, size), IREE_SV("fuzz.loombc"), &g_context,
      block_pool, &options, &metadata_result));

  loom_bytecode_read_result_t module_result = {0};
  loom_module_t* module = NULL;
  fuzz_ignore_status_or_trap(loom_bytecode_read_module(
      iree_make_const_byte_span(data, size), IREE_SV("fuzz.loombc"), &g_context,
      block_pool, &options, &module_result, &module, iree_allocator_system()));
  if (module) {
    loom_module_free(module);
  }
}

static void fuzz_read_generated_module(const uint8_t* data, size_t size,
                                       uint8_t control_byte,
                                       iree_arena_block_pool_t* block_pool) {
  uint8_t preset = size > 0 ? data[0] : 0;
  uint32_t scale = size > 1 ? (uint32_t)((data[1] % 5) + 1) : 1;
  const uint8_t* random_data = size > 2 ? data + 2 : NULL;
  size_t random_size = size > 2 ? size - 2 : 0;

  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_fuzz_preset(preset, scale);
  loom_test_gen_t generator;
  loom_test_gen_initialize_fuzz(random_data, random_size, &generator);

  loom_module_t* module = NULL;
  fuzz_ignore_status_or_trap(loom_test_gen_module(
      &generator, &config, &g_context, block_pool, &module));
  if (!module) __builtin_trap();

  std::vector<uint8_t> first = fuzz_write_module(module, block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      fuzz_read_options(control_byte, &diagnostic_count);
  loom_bytecode_read_result_t result = {0};
  loom_module_t* read_module = NULL;
  fuzz_ignore_status_or_trap(loom_bytecode_read_module(
      iree_make_const_byte_span(first.data(), first.size()),
      IREE_SV("generated.loombc"), &g_context, block_pool, &options, &result,
      &read_module, iree_allocator_system()));
  if (result.error_count != 0 || !read_module) __builtin_trap();
  std::vector<uint8_t> second = fuzz_write_module(read_module, block_pool);
  if (first != second) __builtin_trap();

  loom_module_free(read_module);
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;
  if (size > 1024 * 1024) return 0;
  fuzz_ensure_context();

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  uint8_t control_byte = data[0];
  fuzz_read_arbitrary_bytecode(data, size, control_byte, &block_pool);
  fuzz_read_generated_module(data + 1, size - 1, control_byte, &block_pool);

  iree_arena_block_pool_deinitialize(&block_pool);
  return 0;
}
