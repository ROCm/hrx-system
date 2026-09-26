// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzzes exclusion and single-entry proofs against concrete lane paths through
// verified IR. The low bits of the first byte select 1..8 blocks and its high
// bit permits cycles. The second byte supplies uniform-selector bits. Each
// block consumes a successor count (0..2) and target offsets: forward targets
// in acyclic mode, any nonentry block in cyclic mode. Missing bytes are zero.
// Incremental scopes are covered by fact_control_fuzz.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "loom/analysis/control_uniformity_test_util.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"

static void check(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

namespace {

class ControlContext {
 public:
  ControlContext() {
    loom_context_initialize(iree_allocator_system(), &context);
    iree_host_size_t count = 0;
    const auto* vtables = loom_cfg_dialect_vtables(&count);
    check(loom_context_register_dialect(&context, LOOM_DIALECT_CFG, vtables,
                                        static_cast<uint16_t>(count)));
    vtables = loom_test_dialect_vtables(&count);
    check(loom_context_register_dialect(&context, LOOM_DIALECT_TEST, vtables,
                                        static_cast<uint16_t>(count)));
    check(loom_context_finalize(&context));
  }
  ~ControlContext() { loom_context_deinitialize(&context); }

  // Shared dialect registry; each input owns its module and analysis storage.
  loom_context_t context;
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static ControlContext state;
  size_t position = 0;
  auto next = [&]() -> uint8_t {
    return position < size ? data[position++] : 0;
  };
  const uint8_t shape = next();
  const uint16_t count = 1 + shape % 8;
  const bool cyclic = shape & 0x80;
  const uint8_t uniform_selectors = next();
  std::vector<std::vector<uint16_t>> successors(count);
  for (uint16_t source = 0; source < count && count > 1; ++source) {
    if (!cyclic && source + 1 == count) {
      break;
    }
    const uint8_t edge_count = next() % 3;
    for (uint8_t i = 0; i < edge_count; ++i) {
      const uint16_t target = cyclic
                                  ? 1 + next() % (count - 1)
                                  : source + 1 + next() % (count - source - 1);
      successors[source].push_back(target);
    }
  }
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&pool, &arena);
  check(loom::testing::CheckControlExecution(&state.context, &pool, &arena,
                                             successors, uniform_selectors));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&pool);
  return 0;
}
