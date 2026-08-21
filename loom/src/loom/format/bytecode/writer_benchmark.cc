// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks bytecode serialization of symbol catalogs whose stable symbol IDs
// disagree with their physical top-level presentation order.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace {

static void AbortOnError(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_abort(status);
  }
}

class WriterCatalogFixture {
 public:
  explicit WriterCatalogFixture(uint32_t symbol_count) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &module_block_pool_);
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &writer_block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    AbortOnError(loom_test_dialect_register(&context_));
    AbortOnError(loom_context_finalize(&context_));
    AbortOnError(loom_module_allocate(
        &context_, IREE_SV("writer_catalog_benchmark"), &module_block_pool_,
        /*archive=*/nullptr, iree_allocator_system(), &module_));

    std::vector<loom_symbol_ref_t> symbols;
    symbols.reserve(symbol_count);
    for (uint32_t symbol_ordinal = 0; symbol_ordinal < symbol_count;
         ++symbol_ordinal) {
      char name[32];
      std::snprintf(name, sizeof(name), "function_%08u", symbol_ordinal);
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      AbortOnError(loom_module_intern_string(
          module_, iree_make_cstring_view(name), &name_id));
      loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
      AbortOnError(loom_module_add_symbol(module_, name_id, &symbol_id));
      symbols.push_back({/*.module_id=*/0, /*.symbol_id=*/symbol_id});
    }

    // Define functions in reverse ID order so every write exercises the wire
    // permutation instead of its identity case.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    for (uint32_t symbol_ordinal = symbol_count; symbol_ordinal > 0;
         --symbol_ordinal) {
      loom_op_t* function_op = nullptr;
      AbortOnError(loom_test_func_build(
          &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
          symbols[symbol_ordinal - 1], /*arg_types=*/nullptr,
          /*arg_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
          /*result_dims=*/nullptr, /*result_dim_count=*/0,
          /*result_encodings=*/nullptr, /*result_encoding_count=*/0,
          LOOM_LOCATION_NONE, &function_op));
      loom_func_like_t function = loom_func_like_cast(module_, function_op);
      loom_builder_t body_builder;
      loom_builder_initialize(
          module_, &module_->arena,
          loom_region_entry_block(loom_func_like_body(function)),
          &body_builder);
      loom_op_t* yield_op = nullptr;
      AbortOnError(loom_test_yield_build(&body_builder, /*values=*/nullptr,
                                         /*value_count=*/0, LOOM_LOCATION_NONE,
                                         &yield_op));
    }

    AbortOnError(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_RESIZABLE,
        65536, iree_allocator_system(), &stream_));
    Write();
    bytecode_size_ = iree_io_stream_length(stream_);
    iree_arena_block_pool_query_statistics(&writer_block_pool_,
                                           &writer_pool_statistics_);
  }

  WriterCatalogFixture(const WriterCatalogFixture&) = delete;
  WriterCatalogFixture& operator=(const WriterCatalogFixture&) = delete;

  ~WriterCatalogFixture() {
    iree_io_stream_release(stream_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&writer_block_pool_);
    iree_arena_block_pool_deinitialize(&module_block_pool_);
  }

  void Rewind() {
    AbortOnError(iree_io_stream_seek(stream_, IREE_IO_STREAM_SEEK_SET, 0));
  }

  void Write() {
    AbortOnError(loom_bytecode_write_module(
        module_, stream_, /*options=*/nullptr, &writer_block_pool_));
  }

  iree_io_stream_pos_t bytecode_size() const { return bytecode_size_; }

  uint64_t writer_pool_bytes() const {
    return writer_pool_statistics_.block_system_allocation_bytes +
           writer_pool_statistics_.oversized_allocation_bytes;
  }

 private:
  // Pool owning the immutable source module storage.
  iree_arena_block_pool_t module_block_pool_;
  // Pool reused for temporary writer storage on every iteration.
  iree_arena_block_pool_t writer_block_pool_;
  // Context defining the source module's test dialect operations.
  loom_context_t context_;
  // Immutable source module serialized by every iteration.
  loom_module_t* module_ = nullptr;
  // Reusable output stream rewound before every iteration.
  iree_io_stream_t* stream_ = nullptr;
  // Stable serialized output size measured by the warm-up write.
  iree_io_stream_pos_t bytecode_size_ = 0;
  // Warm writer-pool allocation footprint after one complete write.
  iree_arena_block_pool_statistics_t writer_pool_statistics_ = {};
};

static void BM_WriteCatalogReverseSymbolOrder(benchmark::State& state) {
  const uint32_t symbol_count = static_cast<uint32_t>(state.range(0));
  WriterCatalogFixture fixture(symbol_count);
  for (auto _ : state) {
    state.PauseTiming();
    fixture.Rewind();
    state.ResumeTiming();
    fixture.Write();
    benchmark::ClobberMemory();
  }
  state.counters["bytecode_bytes"] =
      static_cast<double>(fixture.bytecode_size());
  state.counters["scratch_pool_bytes"] =
      static_cast<double>(fixture.writer_pool_bytes());
  state.counters["symbols"] = static_cast<double>(symbol_count);
  state.SetComplexityN(symbol_count);
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

BENCHMARK(BM_WriteCatalogReverseSymbolOrder)
    ->Apply(CatalogScales)
    ->Complexity();

}  // namespace

BENCHMARK_MAIN();
