// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks the public bytecode index across the independent provider-import
// scaling dimensions and verifies that indexing cost is independent of IR body
// size. Fixtures are fully serialized before timing starts.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/test/ops.h"

namespace {

enum class ImportShape {
  kManyProviders,
  kManyAnchors,
  kLargeBody,
};

static void AbortOnError(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

class ProviderImportBytecodeFixture {
 public:
  ProviderImportBytecodeFixture(ImportShape shape, int64_t scale) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_MODULE, loom_module_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    AbortOnError(loom_context_finalize(&context_));

    loom_module_t* module = nullptr;
    AbortOnError(loom_module_allocate(
        &context_, IREE_SV("provider_import_benchmark"), &block_pool_, nullptr,
        iree_allocator_system(), &module));
    BuildModule(module, shape, scale);
    bytes_ = WriteModule(module);
    loom_module_free(module);
  }

  ProviderImportBytecodeFixture(const ProviderImportBytecodeFixture&) = delete;
  ProviderImportBytecodeFixture& operator=(
      const ProviderImportBytecodeFixture&) = delete;

  ~ProviderImportBytecodeFixture() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }
  loom_context_t* context() { return &context_; }

 private:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    AbortOnError(loom_context_register_dialect(&context_, dialect_id, vtables,
                                               static_cast<uint16_t>(count)));
  }

  static loom_symbol_ref_t AddSymbol(loom_module_t* module,
                                     iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    AbortOnError(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    AbortOnError(loom_module_add_symbol(module, name_id, &symbol_id));
    return {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  static void AddImport(loom_module_t* module, loom_builder_t* builder,
                        iree_string_view_t provider,
                        const loom_symbol_ref_t* anchors,
                        iree_host_size_t anchor_count) {
    loom_string_id_t provider_id = LOOM_STRING_ID_INVALID;
    AbortOnError(loom_module_intern_string(module, provider, &provider_id));
    loom_op_t* import_op = nullptr;
    AbortOnError(loom_module_import_build(
        builder, provider_id, loom_make_symbol_ref_array(anchors, anchor_count),
        LOOM_LOCATION_NONE, &import_op));
  }

  static loom_symbol_ref_t AddFunction(loom_module_t* module,
                                       loom_builder_t* module_builder,
                                       int64_t body_op_count) {
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    AbortOnError(loom_module_intern_type(module, i32_type, &i32_type));
    loom_symbol_ref_t function = AddSymbol(module, IREE_SV("body"));
    loom_type_t arg_types[] = {i32_type};
    loom_type_t result_types[] = {i32_type};
    loom_op_t* function_op = nullptr;
    AbortOnError(loom_test_func_build(
        module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0, function,
        arg_types, IREE_ARRAYSIZE(arg_types), result_types,
        IREE_ARRAYSIZE(result_types), /*result_dims=*/nullptr,
        /*result_dim_count=*/0, /*result_encodings=*/nullptr,
        /*result_encoding_count=*/0, LOOM_LOCATION_NONE, &function_op));

    loom_func_like_t function_like = loom_func_like_cast(module, function_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args =
        loom_func_like_arg_ids(function_like, &arg_count);
    if (arg_count != 1) abort();
    loom_builder_t body_builder;
    loom_builder_initialize(
        module, &module->arena,
        loom_region_entry_block(loom_func_like_body(function_like)),
        &body_builder);
    loom_value_id_t current_value = args[0];
    for (int64_t i = 0; i < body_op_count; ++i) {
      loom_op_t* add_op = nullptr;
      AbortOnError(loom_test_addi_build(&body_builder, current_value, args[0],
                                        i32_type, LOOM_LOCATION_NONE, &add_op));
      current_value = loom_test_addi_result(add_op);
    }
    loom_op_t* yield_op = nullptr;
    AbortOnError(loom_test_yield_build(&body_builder, &current_value, 1,
                                       LOOM_LOCATION_NONE, &yield_op));
    return function;
  }

  static void BuildModule(loom_module_t* module, ImportShape shape,
                          int64_t scale) {
    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    if (shape == ImportShape::kManyProviders) {
      loom_symbol_ref_t anchor = AddSymbol(module, IREE_SV("shared_anchor"));
      for (int64_t i = 0; i < scale; ++i) {
        char provider[32];
        std::snprintf(provider, sizeof(provider), "provider/%08d.loom",
                      static_cast<int>(i));
        AddImport(module, &builder, iree_make_cstring_view(provider), &anchor,
                  1);
      }
      return;
    }

    if (shape == ImportShape::kManyAnchors) {
      std::vector<loom_symbol_ref_t> anchors;
      anchors.reserve(static_cast<size_t>(scale));
      for (int64_t i = 0; i < scale; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "anchor_%08d", static_cast<int>(i));
        anchors.push_back(AddSymbol(module, iree_make_cstring_view(name)));
      }
      AddImport(module, &builder, IREE_SV("provider.loom"), anchors.data(),
                anchors.size());
      return;
    }

    loom_symbol_ref_t anchors[] = {
        AddFunction(module, &builder, scale),
        AddSymbol(module, IREE_SV("missing")),
    };
    AddImport(module, &builder, IREE_SV("provider.loom"), anchors,
              IREE_ARRAYSIZE(anchors));
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    AbortOnError(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    AbortOnError(
        loom_bytecode_write_module(module, stream, nullptr, &block_pool_));
    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    AbortOnError(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    AbortOnError(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<uint8_t> bytes_;
};

static void RunIndexBenchmark(benchmark::State& state, ImportShape shape) {
  const int64_t scale = state.range(0);
  ProviderImportBytecodeFixture fixture(shape, scale);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool, &metadata_arena);

  loom_bytecode_read_result_t read_result = {};
  loom_bytecode_file_metadata_t metadata = {};
  AbortOnError(loom_bytecode_read_index(
      iree_make_const_byte_span(fixture.bytes().data(), fixture.bytes().size()),
      IREE_SV("provider_import_benchmark.loombc"), fixture.context(),
      &block_pool, &metadata_arena, /*options=*/nullptr, &read_result,
      &metadata));
  if (read_result.error_count != 0 || metadata.module_count != 1) abort();
  const iree_host_size_t provider_count =
      metadata.modules[0].provider_import_count;
  const iree_host_size_t anchor_count =
      metadata.modules[0].provider_import_anchor_count;
  const iree_host_size_t symbol_count = metadata.modules[0].symbol_count;
  const iree_host_size_t retained_used_bytes =
      metadata_arena.used_allocation_size;
  const iree_host_size_t retained_owned_bytes =
      metadata_arena.total_allocation_size;
  iree_arena_reset(&metadata_arena);

  for (auto _ : state) {
    read_result = {};
    metadata = {};
    AbortOnError(loom_bytecode_read_index(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("provider_import_benchmark.loombc"), fixture.context(),
        &block_pool, &metadata_arena, /*options=*/nullptr, &read_result,
        &metadata));
    benchmark::DoNotOptimize(metadata.modules[0].provider_imports);
    benchmark::DoNotOptimize(
        metadata.modules[0].provider_import_anchor_symbol_indices);

    state.PauseTiming();
    iree_arena_reset(&metadata_arena);
    state.ResumeTiming();
  }

  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  state.counters["anchors"] = static_cast<double>(anchor_count);
  state.counters["bytecode_bytes"] =
      static_cast<double>(fixture.bytes().size());
  state.counters["providers"] = static_cast<double>(provider_count);
  state.counters["retained_owned_bytes"] =
      static_cast<double>(retained_owned_bytes);
  state.counters["retained_used_bytes"] =
      static_cast<double>(retained_used_bytes);
  state.counters["symbols"] = static_cast<double>(symbol_count);
  if (shape != ImportShape::kLargeBody) {
    state.SetItemsProcessed(state.iterations() * scale);
    state.SetComplexityN(scale);
  }

  iree_arena_deinitialize(&metadata_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void ImportScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

static void BM_IndexManyProviders(benchmark::State& state) {
  RunIndexBenchmark(state, ImportShape::kManyProviders);
}
BENCHMARK(BM_IndexManyProviders)
    ->Apply(ImportScales)
    ->Complexity(benchmark::oN);

static void BM_IndexManyAnchors(benchmark::State& state) {
  RunIndexBenchmark(state, ImportShape::kManyAnchors);
}
BENCHMARK(BM_IndexManyAnchors)->Apply(ImportScales)->Complexity(benchmark::oN);

static void BM_IndexBodyIndependence(benchmark::State& state) {
  RunIndexBenchmark(state, ImportShape::kLargeBody);
}
BENCHMARK(BM_IndexBodyIndependence)->Apply(ImportScales);

}  // namespace

BENCHMARK_MAIN();
