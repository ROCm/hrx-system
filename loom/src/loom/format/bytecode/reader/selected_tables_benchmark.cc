// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/selected_tables.h"

namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

static void AppendUVarint(uint64_t value, std::vector<uint8_t>* bytes) {
  do {
    uint8_t byte = value & 0x7Fu;
    value >>= 7;
    if (value != 0) {
      byte |= 0x80u;
    }
    bytes->push_back(byte);
  } while (value != 0);
}

static void SelectedScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(64)->Arg(512)->Arg(4096);
}

static void SetStorageCounters(
    benchmark::State& state,
    const loom_bytecode_selected_table_materializer_t& materializer) {
  state.counters["projection_bytes"] = static_cast<double>(
      materializer.projection.slots.capacity * sizeof(uint64_t));
  state.counters["worklist_bytes"] =
      static_cast<double>(materializer.worklist.capacity *
                          sizeof(loom_bytecode_selected_table_frame_t));
}

static void BM_MaterializeTypeChain(benchmark::State& state) {
  const uint32_t type_count = static_cast<uint32_t>(state.range(0));
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(type_count);
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < type_count; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
    AppendUVarint(/*argument_count=*/1, &bytecode);
    AppendUVarint(/*result_count=*/0, &bytecode);
    AppendUVarint(/*prior_type_id=*/i - 1, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {};
  metadata.types = {entries.size(), entries.data()};

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_CHECK_OK(loom_context_finalize(&context));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  for (auto _ : state) {
    state.PauseTiming();
    uint32_t error_count = 0;
    loom_bytecode_reader_decoder_t decoder;
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_tables_benchmark.loombc"), &error_count, &decoder);
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool, &scratch_arena);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context, iree_string_view_empty(),
                                       &block_pool, nullptr,
                                       iree_allocator_system(), &module));
    loom_bytecode_selected_table_materializer_t materializer;
    loom_bytecode_selected_table_materializer_initialize(
        &decoder, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        &context, &metadata, &scratch_arena, module,
        loom_bytecode_selected_symbol_resolver_empty(), iree_allocator_system(),
        &materializer);
    state.ResumeTiming();

    loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_bytecode_selected_table_materialize_type(
        &materializer, type_count - 1, &target_type_id));
    benchmark::DoNotOptimize(target_type_id);

    state.PauseTiming();
    SetStorageCounters(state, materializer);
    loom_bytecode_selected_table_materializer_deinitialize(&materializer);
    loom_module_free(module);
    iree_arena_deinitialize(&scratch_arena);
    state.ResumeTiming();
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  loom_context_deinitialize(&context);
  state.SetItemsProcessed(state.iterations() * type_count);
  state.SetComplexityN(type_count);
  state.counters["bytecode_bytes"] = static_cast<double>(bytecode.size());
}
BENCHMARK(BM_MaterializeTypeChain)
    ->Apply(SelectedScales)
    ->Complexity(benchmark::oN);

static void BM_MaterializeWideFunction(benchmark::State& state) {
  const uint32_t argument_count = static_cast<uint32_t>(state.range(0));
  std::vector<uint8_t> bytecode;
  loom_bytecode_table_entry_metadata_t entries[2] = {};
  entries[0].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_NONE);
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  entries[1].entry_offset = bytecode.size();
  bytecode.push_back(LOOM_BYTECODE_TYPE_FUNCTION);
  AppendUVarint(argument_count, &bytecode);
  AppendUVarint(/*result_count=*/0, &bytecode);
  for (uint32_t i = 0; i < argument_count; ++i) {
    AppendUVarint(/*type_id=*/0, &bytecode);
  }
  entries[1].entry_length = bytecode.size() - entries[1].entry_offset;
  loom_bytecode_module_metadata_t metadata = {};
  metadata.types = {IREE_ARRAYSIZE(entries), entries};

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_CHECK_OK(loom_context_finalize(&context));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  for (auto _ : state) {
    state.PauseTiming();
    uint32_t error_count = 0;
    loom_bytecode_reader_decoder_t decoder;
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_tables_benchmark.loombc"), &error_count, &decoder);
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool, &scratch_arena);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context, iree_string_view_empty(),
                                       &block_pool, nullptr,
                                       iree_allocator_system(), &module));
    loom_bytecode_selected_table_materializer_t materializer;
    loom_bytecode_selected_table_materializer_initialize(
        &decoder, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        &context, &metadata, &scratch_arena, module,
        loom_bytecode_selected_symbol_resolver_empty(), iree_allocator_system(),
        &materializer);
    state.ResumeTiming();

    loom_type_id_t target_type_id = LOOM_TYPE_ID_INVALID;
    IREE_CHECK_OK(loom_bytecode_selected_table_materialize_type(
        &materializer, /*source_type_id=*/1, &target_type_id));
    benchmark::DoNotOptimize(target_type_id);

    state.PauseTiming();
    SetStorageCounters(state, materializer);
    loom_bytecode_selected_table_materializer_deinitialize(&materializer);
    loom_module_free(module);
    iree_arena_deinitialize(&scratch_arena);
    state.ResumeTiming();
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  loom_context_deinitialize(&context);
  state.SetItemsProcessed(state.iterations() * argument_count);
  state.SetComplexityN(argument_count);
  state.counters["bytecode_bytes"] = static_cast<double>(bytecode.size());
}
BENCHMARK(BM_MaterializeWideFunction)
    ->Apply(SelectedScales)
    ->Complexity(benchmark::oN);

static void BM_MaterializeLocationChain(benchmark::State& state) {
  const uint32_t location_count = static_cast<uint32_t>(state.range(0));
  std::vector<uint8_t> bytecode;
  std::vector<loom_bytecode_table_entry_metadata_t> entries(location_count);
  entries[0].entry_offset = bytecode.size();
  bytecode.insert(bytecode.end(), {LOOM_LOCATION_NONE, 0x00});
  entries[0].entry_length = bytecode.size() - entries[0].entry_offset;
  for (uint32_t i = 1; i < location_count; ++i) {
    entries[i].entry_offset = bytecode.size();
    bytecode.insert(bytecode.end(), {
                                        LOOM_LOCATION_TAGGED,
                                        0x00,  // Flags.
                                        0x01,  // Tag.
                                    });
    AppendUVarint(/*prior_location_id=*/i - 1, &bytecode);
    AppendUVarint(/*data_length=*/0, &bytecode);
    entries[i].entry_length = bytecode.size() - entries[i].entry_offset;
  }
  loom_bytecode_module_metadata_t metadata = {};
  metadata.locations = {entries.size(), entries.data()};

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_CHECK_OK(loom_context_finalize(&context));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  for (auto _ : state) {
    state.PauseTiming();
    uint32_t error_count = 0;
    loom_bytecode_reader_decoder_t decoder;
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("selected_tables_benchmark.loombc"), &error_count, &decoder);
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool, &scratch_arena);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context, iree_string_view_empty(),
                                       &block_pool, nullptr,
                                       iree_allocator_system(), &module));
    loom_bytecode_selected_table_materializer_t materializer;
    loom_bytecode_selected_table_materializer_initialize(
        &decoder, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        &context, &metadata, &scratch_arena, module,
        loom_bytecode_selected_symbol_resolver_empty(), iree_allocator_system(),
        &materializer);
    state.ResumeTiming();

    loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_bytecode_selected_table_materialize_location(
        &materializer, location_count - 1, &target_location_id));
    benchmark::DoNotOptimize(target_location_id);

    state.PauseTiming();
    SetStorageCounters(state, materializer);
    loom_bytecode_selected_table_materializer_deinitialize(&materializer);
    loom_module_free(module);
    iree_arena_deinitialize(&scratch_arena);
    state.ResumeTiming();
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  loom_context_deinitialize(&context);
  state.SetItemsProcessed(state.iterations() * location_count);
  state.SetComplexityN(location_count);
  state.counters["bytecode_bytes"] = static_cast<double>(bytecode.size());
}
BENCHMARK(BM_MaterializeLocationChain)
    ->Apply(SelectedScales)
    ->Complexity(benchmark::oN);

}  // namespace

BENCHMARK_MAIN();
