// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_CONTROL_UNIFORMITY_TEST_UTIL_H_
#define LOOM_ANALYSIS_CONTROL_UNIFORMITY_TEST_UTIL_H_

#include <vector>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"

namespace loom::testing {

// Builds and verifies a small CFG with one independent selector per block,
// then checks exclusion and single-entry claims against concrete lane paths.
// Inputs contain 1..8 blocks and at most two successors each; edges never
// target the function entry. A bit in uniform_selectors forces both lanes to
// share that choice. Choices are stable across iterations of a cyclic path.
// The context must have the test and cfg dialects registered. Analysis storage
// is borrowed for the call; the caller may reset it after the call returns.
iree_status_t CheckControlExecution(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* analysis_arena,
    const std::vector<std::vector<uint16_t>>& successors,
    uint32_t uniform_selectors);

}  // namespace loom::testing

#endif  // LOOM_ANALYSIS_CONTROL_UNIFORMITY_TEST_UTIL_H_
