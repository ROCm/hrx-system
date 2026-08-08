// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks static encoding construction and queries at scales representative
// of modules carrying reusable schemas and many specialized schemas. The rows
// use production text parsing, canonical module storage, and family queries so
// representation changes remain accountable for both time and retained memory.

#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/util/fact_table.h"
#include "loom/util/stream.h"

namespace {

enum class SchemaPopulation {
  kRepeated,
  kDistinct,
};

class StaticEncodingBenchmark {
 public:
  StaticEncodingBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_encoding_dialect_vtables(&vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)vtable_count));
    IREE_CHECK_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));
  }

  StaticEncodingBenchmark(const StaticEncodingBenchmark&) = delete;
  StaticEncodingBenchmark& operator=(const StaticEncodingBenchmark&) = delete;

  ~StaticEncodingBenchmark() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Parse(iree_string_view_t source, loom_module_t** out_module) {
    loom_text_parse_options_t options = {};
    options.max_errors = 1;
    return loom_text_parse(source, IREE_SV("static_encoding_benchmark.loom"),
                           &context_, &block_pool_, &options, out_module);
  }

  void SetMemoryCounters(benchmark::State& state,
                         const loom_module_t* module) const {
    iree_arena_block_pool_statistics_t statistics;
    iree_arena_block_pool_query_statistics(&block_pool_, &statistics);
    state.counters["module_arena_used_bytes"] =
        (double)module->arena.used_allocation_size;
    state.counters["module_arena_owned_bytes"] =
        (double)module->arena.total_allocation_size;
    state.counters["block_pool_system_bytes"] =
        (double)statistics.block_system_allocation_bytes;
    state.counters["oversized_allocation_count"] =
        (double)statistics.oversized_allocation_count;
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static std::string BuildEncodingModule(SchemaPopulation population,
                                       int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 240);
  for (int64_t i = 0; i < definition_count; ++i) {
    const int64_t payload_element_count =
        population == SchemaPopulation::kDistinct ? i + 1 : 32;
    source.append("%schema");
    source.append(std::to_string(i));
    source.append(
        " = encoding.define #matrix_operand<element_format=f8e4m3, "
        "payload_elements=");
    source.append(std::to_string(payload_element_count));
    source.append(
        ", payload_packing=dense_lanes, payload_registers=0, "
        "rounding=finite_only> : encoding<schema>\n");
  }
  return source;
}

static loom_module_t* ParseModule(StaticEncodingBenchmark& fixture,
                                  const std::string& source) {
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(fixture.Parse(
      iree_make_string_view(source.data(), source.size()), &module));
  IREE_ASSERT(module != nullptr);
  return module;
}

static void BenchmarkParseStaticEncodings(benchmark::State& state,
                                          SchemaPopulation population) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildEncodingModule(population, definition_count);
  StaticEncodingBenchmark fixture;

  loom_module_t* warm_module = ParseModule(fixture, source);
  fixture.SetMemoryCounters(state, warm_module);
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ParseModule(fixture, source);
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}

static void BM_ParseRepeatedStaticEncoding(benchmark::State& state) {
  BenchmarkParseStaticEncodings(state, SchemaPopulation::kRepeated);
}
BENCHMARK(BM_ParseRepeatedStaticEncoding)->Arg(256);

static void BM_ParseDistinctStaticEncodings(benchmark::State& state) {
  BenchmarkParseStaticEncodings(state, SchemaPopulation::kDistinct);
}
BENCHMARK(BM_ParseDistinctStaticEncodings)->Arg(1)->Arg(32)->Arg(256);

static void BM_PrintDistinctStaticEncodings(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  StaticEncodingBenchmark fixture;
  loom_module_t* module = ParseModule(fixture, source);
  fixture.SetMemoryCounters(state, module);

  loom_output_stream_t preflight_stream;
  loom_output_stream_null(&preflight_stream);
  IREE_CHECK_OK(loom_text_print_module(module, &preflight_stream,
                                       LOOM_TEXT_PRINT_DEFAULT));
  const int64_t printed_byte_count = (int64_t)preflight_stream.offset;
  for (auto _ : state) {
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    IREE_CHECK_OK(
        loom_text_print_module(module, &stream, LOOM_TEXT_PRINT_DEFAULT));
    benchmark::DoNotOptimize(stream.offset);
  }
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * printed_byte_count);
  loom_module_free(module);
}
BENCHMARK(BM_PrintDistinctStaticEncodings)->Arg(1)->Arg(32)->Arg(256);

static void BM_QueryDistinctStaticEncodingRoles(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  StaticEncodingBenchmark fixture;
  loom_module_t* module = ParseModule(fixture, source);
  fixture.SetMemoryCounters(state, module);
  if (module->encodings.count != (iree_host_size_t)definition_count) {
    state.SkipWithError("distinct schemas were unexpectedly deduplicated");
    loom_module_free(module);
    return;
  }

  for (auto _ : state) {
    for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
      loom_encoding_role_t role =
          loom_encoding_static_role(module, &module->encodings.entries[i]);
      benchmark::DoNotOptimize(role);
    }
  }
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryDistinctStaticEncodingRoles)->Arg(1)->Arg(32)->Arg(256);

static void BM_QueryDistinctStaticStorageSchemas(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  StaticEncodingBenchmark fixture;
  loom_module_t* module = ParseModule(fixture, source);
  fixture.SetMemoryCounters(state, module);
  if (module->encodings.count != (iree_host_size_t)definition_count) {
    state.SkipWithError("distinct schemas were unexpectedly deduplicated");
    loom_module_free(module);
    return;
  }

  for (uint16_t encoding_id = 1; encoding_id <= module->encodings.count;
       ++encoding_id) {
    loom_value_fact_storage_schema_t schema = {};
    if (!loom_encoding_query_static_storage_schema(module, encoding_id,
                                                   &schema) ||
        schema.encoded_operand.payload_element_count != encoding_id) {
      state.SkipWithError("static storage schema query produced wrong facts");
      loom_module_free(module);
      return;
    }
  }

  for (auto _ : state) {
    for (uint16_t encoding_id = 1; encoding_id <= module->encodings.count;
         ++encoding_id) {
      loom_value_fact_storage_schema_t schema = {};
      bool resolved = loom_encoding_query_static_storage_schema(
          module, encoding_id, &schema);
      benchmark::DoNotOptimize(resolved);
      benchmark::DoNotOptimize(schema);
    }
  }
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryDistinctStaticStorageSchemas)->Arg(1)->Arg(32)->Arg(256);

}  // namespace
