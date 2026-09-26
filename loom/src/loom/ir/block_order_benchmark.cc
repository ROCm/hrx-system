// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace {

enum class InsertionPattern {
  kAppend,
  kRepeatedAfter,
  kRepeatedBefore,
  kAlternating,
};

class BenchmarkContext {
 public:
  BenchmarkContext() {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_context_finalize(&context_));
  }

  ~BenchmarkContext() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("block-order"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module));
    return module;
  }

 private:
  // Recycled backing storage for each benchmark module.
  iree_arena_block_pool_t block_pool_ = {};
  // Minimal context; block ordering does not require registered operations.
  loom_context_t context_ = {};
};

static loom_op_t* AllocateOp(loom_module_t* module) {
  loom_op_t* op = nullptr;
  IREE_CHECK_OK(iree_arena_allocate(&module->arena, sizeof(*op), (void**)&op));
  std::memset(op, 0, sizeof(*op));
  op->kind = 0x0100;
  return op;
}

static void VerifyOrder(const loom_block_t* block) {
  uint32_t count = 0;
  uint64_t previous_ordinal = 0;
  loom_op_t* previous_op = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(block, op) {
    if (op->prev_op != previous_op || op->block_ordinal <= previous_ordinal) {
      std::abort();
    }
    previous_op = op;
    previous_ordinal = op->block_ordinal;
    ++count;
  }
  if (previous_op != block->last_op || count != block->op_count) {
    std::abort();
  }
}

static void RunInsertionBenchmark(benchmark::State& state,
                                  InsertionPattern pattern) {
  const uint32_t op_count = static_cast<uint32_t>(state.range(0));
  BenchmarkContext context;
  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* module = context.AllocateModule();
    loom_block_t* block = loom_module_block(module);
    std::vector<loom_op_t*> ops;
    ops.reserve(op_count);
    for (uint32_t i = 0; i < op_count; ++i) {
      ops.push_back(AllocateOp(module));
    }
    state.ResumeTiming();

    switch (pattern) {
      case InsertionPattern::kAppend: {
        for (loom_op_t* op : ops) {
          IREE_CHECK_OK(loom_block_append_op(module, block, op));
        }
        break;
      }
      case InsertionPattern::kRepeatedAfter: {
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[0]));
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[1]));
        for (uint32_t i = 2; i < op_count; ++i) {
          IREE_CHECK_OK(loom_block_insert_before_op(module, block,
                                                    ops[0]->next_op, ops[i]));
        }
        break;
      }
      case InsertionPattern::kRepeatedBefore: {
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[0]));
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[1]));
        for (uint32_t i = 2; i < op_count; ++i) {
          IREE_CHECK_OK(
              loom_block_insert_before_op(module, block, ops[1], ops[i]));
        }
        break;
      }
      case InsertionPattern::kAlternating: {
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[0]));
        IREE_CHECK_OK(loom_block_append_op(module, block, ops[1]));
        for (uint32_t i = 2; i < op_count; ++i) {
          loom_op_t* before_op = i & 1 ? ops[0] : ops[0]->next_op;
          IREE_CHECK_OK(
              loom_block_insert_before_op(module, block, before_op, ops[i]));
        }
        break;
      }
    }

    benchmark::DoNotOptimize(block->last_op);
    state.PauseTiming();
    VerifyOrder(block);
    loom_module_free(module);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * op_count);
}

static void BM_Append(benchmark::State& state) {
  RunInsertionBenchmark(state, InsertionPattern::kAppend);
}
BENCHMARK(BM_Append)->Arg(256)->Arg(4096);

static void BM_RepeatedAfter(benchmark::State& state) {
  RunInsertionBenchmark(state, InsertionPattern::kRepeatedAfter);
}
BENCHMARK(BM_RepeatedAfter)->Arg(256)->Arg(4096);

static void BM_RepeatedBefore(benchmark::State& state) {
  RunInsertionBenchmark(state, InsertionPattern::kRepeatedBefore);
}
BENCHMARK(BM_RepeatedBefore)->Arg(256)->Arg(4096);

static void BM_Alternating(benchmark::State& state) {
  RunInsertionBenchmark(state, InsertionPattern::kAlternating);
}
BENCHMARK(BM_Alternating)->Arg(256)->Arg(4096);

}  // namespace
