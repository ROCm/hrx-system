// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for Loom bytecode reader throughput.
//
// Fixtures are generated and serialized before timing starts. The measured
// loops cover metadata-only validation, full IR materialization,
// materialization plus verifier handoff for representative generated modules,
// and catalog scaling for independently addressable symbol bodies.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/gen.h"

namespace {

static void IgnoreStatusOrAbort(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static iree_status_t IgnoreDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)diagnostic;
  auto* diagnostic_count = static_cast<uint32_t*>(user_data);
  ++*diagnostic_count;
  return iree_ok_status();
}

class SerializedBytecodeFixture {
 public:
  SerializedBytecodeFixture() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IgnoreStatusOrAbort(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)count));
    IgnoreStatusOrAbort(loom_context_finalize(&context_));
  }

  SerializedBytecodeFixture(const SerializedBytecodeFixture&) = delete;
  SerializedBytecodeFixture& operator=(const SerializedBytecodeFixture&) =
      delete;

  ~SerializedBytecodeFixture() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }
  loom_context_t* context() { return &context_; }

 protected:
  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

  void SerializeModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IgnoreStatusOrAbort(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IgnoreStatusOrAbort(
        loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    bytes_.resize((size_t)length);
    IgnoreStatusOrAbort(
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IgnoreStatusOrAbort(
        iree_io_stream_read(stream, bytes_.size(), bytes_.data(), nullptr));
    iree_io_stream_release(stream);
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<uint8_t> bytes_;
};

class GeneratedBytecodeFixture final : public SerializedBytecodeFixture {
 public:
  GeneratedBytecodeFixture(uint8_t preset, uint32_t scale) {
    loom_test_gen_module_config_t config =
        loom_test_gen_module_config_fuzz_preset(preset, scale);
    loom_test_gen_t generator;
    loom_test_gen_initialize_seeded(
        UINT64_C(0xBADC0DE000000000) | ((uint64_t)preset << 32) | scale,
        &generator);

    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_test_gen_module(&generator, &config, context(),
                                             block_pool(), &module));
    if (!module) {
      abort();
    }
    SerializeModule(module);
    loom_module_free(module);
  }
};

class CatalogBytecodeFixture final : public SerializedBytecodeFixture {
 public:
  CatalogBytecodeFixture(uint32_t symbol_count, uint32_t body_op_count) {
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_module_allocate(
        context(), IREE_SV("reader_catalog_benchmark"), block_pool(), nullptr,
        iree_allocator_system(), &module));
    BuildModule(module, symbol_count, body_op_count);
    SerializeModule(module);
    loom_module_free(module);
  }

 private:
  static void BuildModule(loom_module_t* module, uint32_t symbol_count,
                          uint32_t body_op_count) {
    loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IgnoreStatusOrAbort(loom_module_intern_type(module, i32_type, &i32_type));

    loom_builder_t module_builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &module_builder);
    for (uint32_t symbol_ordinal = 0; symbol_ordinal < symbol_count;
         ++symbol_ordinal) {
      char name[32];
      std::snprintf(name, sizeof(name), "function_%08u", symbol_ordinal);
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      IgnoreStatusOrAbort(loom_module_intern_string(
          module, iree_make_cstring_view(name), &name_id));
      loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
      IgnoreStatusOrAbort(loom_module_add_symbol(module, name_id, &symbol_id));

      loom_symbol_ref_t symbol = {/*.module_id=*/0,
                                  /*.symbol_id=*/symbol_id};
      loom_type_t argument_types[] = {i32_type};
      loom_type_t result_types[] = {i32_type};
      loom_op_t* function_op = nullptr;
      IgnoreStatusOrAbort(loom_test_func_build(
          &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
          symbol, argument_types, IREE_ARRAYSIZE(argument_types), result_types,
          IREE_ARRAYSIZE(result_types), /*result_dims=*/nullptr,
          /*result_dim_count=*/0, /*result_encodings=*/nullptr,
          /*result_encoding_count=*/0, LOOM_LOCATION_NONE, &function_op));

      loom_func_like_t function = loom_func_like_cast(module, function_op);
      uint16_t argument_count = 0;
      const loom_value_id_t* arguments =
          loom_func_like_arg_ids(function, &argument_count);
      if (argument_count != 1) abort();

      loom_builder_t body_builder;
      loom_builder_initialize(
          module, &module->arena,
          loom_region_entry_block(loom_func_like_body(function)),
          &body_builder);
      loom_value_id_t current_value = arguments[0];
      for (uint32_t op_ordinal = 0; op_ordinal < body_op_count; ++op_ordinal) {
        loom_op_t* add_op = nullptr;
        IgnoreStatusOrAbort(loom_test_addi_build(&body_builder, current_value,
                                                 arguments[0], i32_type,
                                                 LOOM_LOCATION_NONE, &add_op));
        current_value = loom_test_addi_result(add_op);
      }
      loom_op_t* yield_op = nullptr;
      IgnoreStatusOrAbort(loom_test_yield_build(
          &body_builder, &current_value, 1, LOOM_LOCATION_NONE, &yield_op));
    }
  }
};

static loom_bytecode_read_options_t ReadOptions(bool verify_module,
                                                uint32_t* diagnostic_count) {
  return loom_bytecode_read_options_t{
      /*.diagnostic_sink=*/
      {
          /*.fn=*/IgnoreDiagnostic,
          /*.user_data=*/diagnostic_count,
      },
      /*.verify_module=*/verify_module,
      /*.verify_max_errors=*/16,
  };
}

static void BenchmarkReadMetadata(benchmark::State& state, uint8_t preset) {
  GeneratedBytecodeFixture fixture(preset, (uint32_t)state.range(0));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_index_options_t options = {
      /*.diagnostic_sink=*/
      {
          /*.fn=*/IgnoreDiagnostic,
          /*.user_data=*/&diagnostic_count,
      },
  };

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {0};
    IgnoreStatusOrAbort(loom_bytecode_read_metadata(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("benchmark.loombc"), fixture.context(), &block_pool, &options,
        &result));
    benchmark::DoNotOptimize(result.first_module.op_count);
  }

  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BenchmarkReadModule(benchmark::State& state, uint8_t preset,
                                bool verify_module) {
  GeneratedBytecodeFixture fixture(preset, (uint32_t)state.range(0));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      ReadOptions(verify_module, &diagnostic_count);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {0};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_read_module(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("benchmark.loombc"), fixture.context(), &block_pool, &options,
        &result, &module, iree_allocator_system()));
    benchmark::DoNotOptimize(module);
    if (module) {
      loom_module_free(module);
    }
  }

  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  iree_arena_block_pool_deinitialize(&block_pool);
}

struct CatalogMetadataStats {
  iree_host_size_t body_bytes;
  iree_host_size_t retained_owned_bytes;
  iree_host_size_t retained_used_bytes;
  uint64_t body_op_count;
  iree_host_size_t symbol_count;
};

static CatalogMetadataStats InspectCatalogMetadata(
    CatalogBytecodeFixture& fixture, iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(block_pool, &metadata_arena);
  loom_bytecode_read_result_t result = {};
  loom_bytecode_file_metadata_t metadata = {};
  IgnoreStatusOrAbort(loom_bytecode_read_index(
      iree_make_const_byte_span(fixture.bytes().data(), fixture.bytes().size()),
      IREE_SV("catalog_benchmark.loombc"), fixture.context(), block_pool,
      &metadata_arena, /*options=*/nullptr, &result, &metadata));
  if (result.error_count != 0 || metadata.module_count != 1) abort();

  const loom_bytecode_module_metadata_t& module = metadata.modules[0];
  CatalogMetadataStats stats = {
      /*.body_bytes=*/0,
      /*.retained_owned_bytes=*/metadata_arena.total_allocation_size,
      /*.retained_used_bytes=*/metadata_arena.used_allocation_size,
      /*.body_op_count=*/module.summary.op_count,
      /*.symbol_count=*/module.symbol_count,
  };
  for (iree_host_size_t i = 0; i < module.symbol_count; ++i) {
    stats.body_bytes += module.symbols[i].body_length;
  }
  iree_arena_deinitialize(&metadata_arena);
  return stats;
}

static void SetCatalogCounters(benchmark::State& state,
                               const CatalogBytecodeFixture& fixture,
                               const CatalogMetadataStats& stats) {
  state.counters["body_bytes"] = static_cast<double>(stats.body_bytes);
  state.counters["body_ops"] = static_cast<double>(stats.body_op_count);
  state.counters["bytecode_bytes"] =
      static_cast<double>(fixture.bytes().size());
  state.counters["retained_owned_bytes"] =
      static_cast<double>(stats.retained_owned_bytes);
  state.counters["retained_used_bytes"] =
      static_cast<double>(stats.retained_used_bytes);
  state.counters["symbols"] = static_cast<double>(stats.symbol_count);
}

static void BM_ReadIndex_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  const uint32_t body_op_count = (uint32_t)state.range(1);
  CatalogBytecodeFixture fixture(symbol_count, body_op_count);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  CatalogMetadataStats stats = InspectCatalogMetadata(fixture, &block_pool);
  if (stats.symbol_count != symbol_count) abort();

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool, &metadata_arena);
  for (auto _ : state) {
    loom_bytecode_read_result_t result = {};
    loom_bytecode_file_metadata_t metadata = {};
    IgnoreStatusOrAbort(loom_bytecode_read_index(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("catalog_benchmark.loombc"), fixture.context(), &block_pool,
        &metadata_arena, /*options=*/nullptr, &result, &metadata));
    if (result.error_count != 0 || metadata.module_count != 1) abort();
    benchmark::DoNotOptimize(metadata.modules[0].symbols);

    state.PauseTiming();
    iree_arena_reset(&metadata_arena);
    state.ResumeTiming();
  }

  SetCatalogCounters(state, fixture, stats);
  state.counters["materialized_symbols"] = 0.0;
  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  state.SetItemsProcessed(state.iterations() * symbol_count);
  state.SetComplexityN(symbol_count);
  iree_arena_deinitialize(&metadata_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BM_ReadModule_Catalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  const uint32_t body_op_count = (uint32_t)state.range(1);
  CatalogBytecodeFixture fixture(symbol_count, body_op_count);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  CatalogMetadataStats stats = InspectCatalogMetadata(fixture, &block_pool);
  if (stats.symbol_count != symbol_count) abort();

  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      ReadOptions(/*verify_module=*/false, &diagnostic_count);
  for (auto _ : state) {
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_read_module(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("catalog_benchmark.loombc"), fixture.context(), &block_pool,
        &options, &result, &module, iree_allocator_system()));
    if (result.error_count != 0 || module == nullptr ||
        module->symbols.count != symbol_count) {
      abort();
    }
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }

  SetCatalogCounters(state, fixture, stats);
  state.counters["materialized_symbols"] = static_cast<double>(symbol_count);
  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  state.SetItemsProcessed(state.iterations() * symbol_count);
  state.SetComplexityN(symbol_count);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void CatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Args({1, 4})->Args({16, 4})->Args({64, 4})->Args({512, 4})->Args(
      {4096, 4});
}

BENCHMARK(BM_ReadIndex_Catalog)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_ReadModule_Catalog)
    ->Apply(CatalogScales)
    ->Complexity(benchmark::oN);

static void BM_ReadMetadata_Representative(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/0);
}
BENCHMARK(BM_ReadMetadata_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadMetadata_NestingStress(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/3);
}
BENCHMARK(BM_ReadMetadata_NestingStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadMetadata_FormatStress(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/4);
}
BENCHMARK(BM_ReadMetadata_FormatStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_Representative(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/0, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_NestingStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/3, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_NestingStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_FormatStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/4, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_FormatStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModuleVerify_Representative(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/0, /*verify_module=*/true);
}
BENCHMARK(BM_ReadModuleVerify_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModuleVerify_FormatStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/4, /*verify_module=*/true);
}
BENCHMARK(BM_ReadModuleVerify_FormatStress)->Arg(1)->Arg(5)->Arg(10);

}  // namespace

BENCHMARK_MAIN();
