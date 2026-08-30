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
#include "loom/format/bytecode/reader/selected_materializer.h"
#include "loom/format/bytecode/selected_reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/test/types.h"
#include "loom/testing/gen.h"
#include "loom/verify/verify.h"

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
    IgnoreStatusOrAbort(loom_test_dialect_register(&context_));
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

class TypePlanBytecodeFixture final : public SerializedBytecodeFixture {
 public:
  explicit TypePlanBytecodeFixture(uint32_t instance_count) {
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_module_allocate(
        context(), IREE_SV("reader_type_plan_benchmark"), block_pool(), nullptr,
        iree_allocator_system(), &module));
    BuildModule(module, instance_count);
    type_count_ = module->types.count;
    SerializeModule(module);
    loom_module_free(module);
  }

  iree_host_size_t type_count() const { return type_count_; }

 private:
  static void BuildModule(loom_module_t* module, uint32_t instance_count) {
    loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
    IgnoreStatusOrAbort(loom_module_intern_type_id(
        module, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));
    loom_string_id_t dialect_name_id = LOOM_STRING_ID_INVALID;
    IgnoreStatusOrAbort(loom_module_intern_string(
        module, IREE_SV("benchmark.type_pair"), &dialect_name_id));
    std::vector<loom_type_t> root_types;
    root_types.reserve(instance_count);

    for (uint32_t i = 0; i < instance_count; ++i) {
      loom_type_t parameterized_type = {};
      IgnoreStatusOrAbort(loom_test_array_type_make(
          module, LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_ALIGNMENT, bf16_type_id,
          /*alignment=*/(uint64_t)i + 1, loom_named_attr_slice_empty(),
          &parameterized_type));

      loom_type_t register_type = {};
      IgnoreStatusOrAbort(loom_module_intern_register_type(
          module, /*carrier_payload0=*/(uint64_t)i + 1,
          /*carrier_payload1=*/(uint64_t)4 << 16, parameterized_type,
          &register_type));

      loom_type_t dialect_parameters[] = {parameterized_type, register_type};
      loom_type_t dialect_type = {};
      IgnoreStatusOrAbort(loom_module_intern_type(
          module,
          loom_type_dialect(dialect_name_id, IREE_ARRAYSIZE(dialect_parameters),
                            dialect_parameters),
          &dialect_type));

      loom_type_t function_arguments[] = {parameterized_type, dialect_type};
      loom_type_t function_results[] = {register_type};
      loom_type_t function_type = {};
      IgnoreStatusOrAbort(loom_module_intern_function_type(
          module, function_arguments, IREE_ARRAYSIZE(function_arguments),
          function_results, IREE_ARRAYSIZE(function_results), &function_type));
      root_types.push_back(function_type);
    }

    loom_builder_t builder;
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
    IgnoreStatusOrAbort(loom_module_intern_string(
        module, IREE_SV("type_plan_root"), &symbol_name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IgnoreStatusOrAbort(
        loom_module_add_symbol(module, symbol_name_id, &symbol_id));
    loom_op_t* declaration_op = nullptr;
    IgnoreStatusOrAbort(loom_test_decl_build(
        &builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id},
        root_types.data(), root_types.size(), /*result_types=*/nullptr,
        /*result_count=*/0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &declaration_op));
  }

  iree_host_size_t type_count_ = 0;
};

static loom_bytecode_read_options_t ReadOptions(uint32_t* diagnostic_count) {
  return loom_bytecode_read_options_t{
      /*.diagnostic_sink=*/
      {
          /*.fn=*/IgnoreDiagnostic,
          /*.user_data=*/diagnostic_count,
      },
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
  loom_bytecode_read_options_t options = ReadOptions(&diagnostic_count);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {0};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_read_module(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("benchmark.loombc"), fixture.context(), &block_pool, &options,
        &result, &module, iree_allocator_system()));
    if (verify_module) {
      const loom_verify_options_t verify_options = {
          /*.sink=*/options.diagnostic_sink,
          /*.max_errors=*/16,
      };
      loom_verify_result_t verify_result = {0};
      IgnoreStatusOrAbort(
          loom_verify_module(module, &verify_options, &verify_result));
      benchmark::DoNotOptimize(verify_result.error_count);
    }
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
  for (iree_host_size_t i = 0; i < module.region_payload_count; ++i) {
    stats.body_bytes += module.region_payloads[i].length;
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
  loom_bytecode_read_options_t options = ReadOptions(&diagnostic_count);
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

struct SelectedMaterializationStats {
  iree_host_size_t body_bytes;
  iree_host_size_t body_count;
  iree_host_size_t output_owned_bytes;
  iree_host_size_t output_used_bytes;
  iree_host_size_t scratch_owned_bytes;
  iree_host_size_t scratch_used_bytes;
  iree_host_size_t type_count;
};

static SelectedMaterializationStats InspectSelectedMaterialization(
    CatalogBytecodeFixture& fixture,
    const loom_bytecode_module_metadata_t& metadata,
    const std::vector<iree_host_size_t>& ordinals,
    iree_arena_block_pool_t* block_pool) {
  uint32_t error_count = 0;
  loom_bytecode_reader_decoder_t decoder;
  loom_bytecode_reader_decoder_initialize(
      /*sink=*/{}, IREE_SV("catalog_benchmark.loombc"), &error_count, &decoder);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  const loom_bytecode_selected_module_materializer_t materializer = {
      /*.decoder=*/&decoder,
      /*.bytecode=*/
      iree_make_const_byte_span(fixture.bytes().data(), fixture.bytes().size()),
      /*.context=*/fixture.context(),
      /*.scratch_arena=*/&scratch_arena,
      /*.block_pool=*/block_pool,
      /*.metadata=*/&metadata,
      /*.low_repr_environment=*/{},
      /*.host_allocator=*/iree_allocator_system(),
  };
  loom_module_t* output_module = nullptr;
  IgnoreStatusOrAbort(loom_bytecode_selected_module_materialize(
      &materializer, ordinals.data(), ordinals.size(), &output_module));
  if (error_count != 0 || output_module == nullptr) abort();

  SelectedMaterializationStats stats = {
      /*.body_bytes=*/0,
      /*.body_count=*/0,
      /*.output_owned_bytes=*/output_module->arena.total_allocation_size,
      /*.output_used_bytes=*/output_module->arena.used_allocation_size,
      /*.scratch_owned_bytes=*/scratch_arena.total_allocation_size,
      /*.scratch_used_bytes=*/scratch_arena.used_allocation_size,
      /*.type_count=*/output_module->types.count,
  };
  for (iree_host_size_t ordinal : ordinals) {
    const loom_bytecode_symbol_metadata_t& symbol = metadata.symbols[ordinal];
    if (symbol.region_payload_count > 0) {
      for (uint8_t i = 0; i < symbol.region_payload_count; ++i) {
        stats.body_bytes +=
            metadata.region_payloads[symbol.first_region_payload_index + i]
                .length;
      }
      ++stats.body_count;
    }
  }
  loom_module_free(output_module);
  iree_arena_deinitialize(&scratch_arena);
  return stats;
}

static void BenchmarkMaterializeSelectedCatalog(benchmark::State& state) {
  const uint32_t symbol_count = (uint32_t)state.range(0);
  const uint32_t selected_symbol_count = (uint32_t)state.range(1);
  CatalogBytecodeFixture fixture(symbol_count, /*body_op_count=*/4);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  CatalogMetadataStats catalog_stats =
      InspectCatalogMetadata(fixture, &block_pool);

  iree_arena_allocator_t metadata_arena;
  iree_arena_initialize(&block_pool, &metadata_arena);
  loom_bytecode_read_result_t index_result = {};
  loom_bytecode_file_metadata_t file_metadata = {};
  IgnoreStatusOrAbort(loom_bytecode_read_index(
      iree_make_const_byte_span(fixture.bytes().data(), fixture.bytes().size()),
      IREE_SV("catalog_benchmark.loombc"), fixture.context(), &block_pool,
      &metadata_arena, /*options=*/nullptr, &index_result, &file_metadata));
  if (index_result.error_count != 0 || file_metadata.module_count != 1 ||
      selected_symbol_count == 0 || selected_symbol_count > symbol_count) {
    abort();
  }
  std::vector<iree_host_size_t> ordinals(selected_symbol_count);
  const iree_host_size_t first_ordinal = symbol_count - selected_symbol_count;
  for (iree_host_size_t i = 0; i < selected_symbol_count; ++i) {
    ordinals[i] = first_ordinal + i;
  }
  const SelectedMaterializationStats selected_stats =
      InspectSelectedMaterialization(fixture, file_metadata.modules[0],
                                     ordinals, &block_pool);

  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options = ReadOptions(&diagnostic_count);
  for (auto _ : state) {
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_materialize_module_symbols(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("catalog_benchmark.loombc"), fixture.context(), &block_pool,
        &file_metadata, /*module_ordinal=*/0,
        (loom_bytecode_symbol_ordinal_list_t){
            /*.count=*/ordinals.size(),
            /*.ordinals=*/ordinals.data(),
        },
        &options, &result, &module, iree_allocator_system()));
    if (result.error_count != 0 || module == nullptr ||
        module->symbols.count != selected_symbol_count) {
      abort();
    }
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }

  SetCatalogCounters(state, fixture, catalog_stats);
  state.counters["materialized_symbols"] =
      static_cast<double>(selected_symbol_count);
  state.counters["output_owned_bytes"] =
      static_cast<double>(selected_stats.output_owned_bytes);
  state.counters["output_used_bytes"] =
      static_cast<double>(selected_stats.output_used_bytes);
  state.counters["rejected_body_bytes"] =
      static_cast<double>(catalog_stats.body_bytes - selected_stats.body_bytes);
  state.counters["selected_bodies"] =
      static_cast<double>(selected_stats.body_count);
  state.counters["selected_body_bytes"] =
      static_cast<double>(selected_stats.body_bytes);
  state.counters["selected_types"] =
      static_cast<double>(selected_stats.type_count);
  state.counters["scratch_owned_bytes"] =
      static_cast<double>(selected_stats.scratch_owned_bytes);
  state.counters["scratch_used_bytes"] =
      static_cast<double>(selected_stats.scratch_used_bytes);
  state.SetItemsProcessed(state.iterations() * selected_symbol_count);
  iree_arena_deinitialize(&metadata_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BM_MaterializeSelectedLeaf_Catalog(benchmark::State& state) {
  BenchmarkMaterializeSelectedCatalog(state);
}

static void BM_MaterializeSelectedAll_Catalog(benchmark::State& state) {
  BenchmarkMaterializeSelectedCatalog(state);
}

static void SelectedLeafCatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Args({1, 1})->Args({16, 1})->Args({64, 1})->Args({512, 1})->Args(
      {4096, 1});
}

static void SelectedAllCatalogScales(benchmark::Benchmark* benchmark) {
  benchmark->Args({1, 1})
      ->Args({16, 16})
      ->Args({64, 64})
      ->Args({512, 512})
      ->Args({4096, 4096});
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
BENCHMARK(BM_MaterializeSelectedLeaf_Catalog)->Apply(SelectedLeafCatalogScales);
BENCHMARK(BM_MaterializeSelectedAll_Catalog)->Apply(SelectedAllCatalogScales);

static void SetTypePlanCounters(benchmark::State& state,
                                const TypePlanBytecodeFixture& fixture,
                                uint32_t instance_count) {
  state.counters["bytecode_bytes"] =
      static_cast<double>(fixture.bytes().size());
  state.counters["instances"] = static_cast<double>(instance_count);
  state.counters["types"] = static_cast<double>(fixture.type_count());
  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  state.SetItemsProcessed(state.iterations() * fixture.type_count());
  state.SetComplexityN(instance_count);
}

static void BM_ReadMetadata_TypePlan(benchmark::State& state) {
  const uint32_t instance_count = (uint32_t)state.range(0);
  TypePlanBytecodeFixture fixture(instance_count);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {};
    IgnoreStatusOrAbort(loom_bytecode_read_metadata(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("type_plan_benchmark.loombc"), fixture.context(), &block_pool,
        /*options=*/nullptr, &result));
    if (result.error_count != 0 ||
        result.first_module.type_count != fixture.type_count()) {
      abort();
    }
    benchmark::DoNotOptimize(result.first_module.type_count);
  }

  SetTypePlanCounters(state, fixture, instance_count);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BM_ReadModule_TypePlan(benchmark::State& state) {
  const uint32_t instance_count = (uint32_t)state.range(0);
  TypePlanBytecodeFixture fixture(instance_count);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options = ReadOptions(&diagnostic_count);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_read_module(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("type_plan_benchmark.loombc"), fixture.context(), &block_pool,
        &options, &result, &module, iree_allocator_system()));
    if (result.error_count != 0 || module == nullptr ||
        module->types.count != fixture.type_count()) {
      abort();
    }
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }

  SetTypePlanCounters(state, fixture, instance_count);
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void TypePlanScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

BENCHMARK(BM_ReadMetadata_TypePlan)
    ->Apply(TypePlanScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_ReadModule_TypePlan)
    ->Apply(TypePlanScales)
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
