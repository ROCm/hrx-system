// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace {

class TypeInternBenchmark {
 public:
  TypeInternBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("type_intern"),
                                       &block_pool_, nullptr,
                                       iree_allocator_system(), &module_));
  }

  ~TypeInternBenchmark() {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_type_t Intern(loom_type_t type) {
    loom_type_t interned_type = {};
    IREE_CHECK_OK(loom_module_intern_type(module_, type, &interned_type));
    return interned_type;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

static void BM_InternRepeatedScalar(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  fixture.Intern(type);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(type));
  }
}
BENCHMARK(BM_InternRepeatedScalar);

static void BM_InternAlternatingScalars(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F16),
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  fixture.Intern(types[0]);
  fixture.Intern(types[1]);
  iree_host_size_t type_index = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(types[type_index]));
    type_index ^= 1;
  }
}
BENCHMARK(BM_InternAlternatingScalars);

static void BM_InternRepeatedStaticVector(benchmark::State& state) {
  TypeInternBenchmark fixture;
  const loom_type_t type = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_static(16), /*encoding_id=*/0);
  fixture.Intern(type);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Intern(type));
  }
}
BENCHMARK(BM_InternRepeatedStaticVector);

}  // namespace
