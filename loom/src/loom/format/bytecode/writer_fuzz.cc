// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for the Loom bytecode writer.
//
// Any valid generated module must serialize without crashing, and two writes
// of the same module with the same options must produce identical bytes.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
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
  loom_context_initialize(iree_allocator_system(), &g_context);
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  iree_status_t status = loom_context_register_dialect(
      &g_context, LOOM_DIALECT_TEST, vtables, (uint16_t)count);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&g_context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(&g_context);
    fuzz_ignore_status_or_trap(status);
  }
  g_context_initialized = true;
}

//===----------------------------------------------------------------------===//
// Bytecode write helpers
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) {
    return 0;
  }
  fuzz_ensure_context();

  uint8_t preset = data[0];
  uint32_t scale = (uint32_t)((data[1] % 5) + 1);
  const uint8_t* random_data = data + 2;
  size_t random_size = size - 2;

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_fuzz_preset(preset, scale);
  loom_test_gen_t generator;
  loom_test_gen_initialize_fuzz(random_data, random_size, &generator);

  loom_module_t* module = NULL;
  fuzz_ignore_status_or_trap(loom_test_gen_module(
      &generator, &config, &g_context, &block_pool, &module));
  if (!module) {
    __builtin_trap();
  }

  std::vector<uint8_t> first = fuzz_write_module(module, &block_pool);
  std::vector<uint8_t> second = fuzz_write_module(module, &block_pool);
  if (first != second) {
    __builtin_trap();
  }

  loom_module_free(module);
  iree_arena_block_pool_deinitialize(&block_pool);
  return 0;
}
