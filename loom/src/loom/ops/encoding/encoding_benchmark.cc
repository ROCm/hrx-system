// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks static encoding construction, dynamic encoding definitions, and
// repeated vector encoding use at scales representative of large JIT modules.
// The rows use production text, bytecode, verifier, and fact paths so
// descriptor changes remain accountable for time and memory. Operation-local
// rows use fresh reusable block pools and report both live arena storage and
// pool high-water allocation.

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/hadamard.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/transforms/cleanup/canonicalize.h"
#include "loom/util/fact_table.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

namespace {

enum class SchemaPopulation {
  kRepeated,
  kDistinct,
};

static void ScaledEncodingCounts(::benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(32)->Arg(256)->Arg(4096);
}

class ScopedBlockPool {
 public:
  ScopedBlockPool() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(), &pool_);
  }

  ScopedBlockPool(const ScopedBlockPool&) = delete;
  ScopedBlockPool& operator=(const ScopedBlockPool&) = delete;

  ~ScopedBlockPool() { iree_arena_block_pool_deinitialize(&pool_); }

  iree_arena_block_pool_t* get() { return &pool_; }

 private:
  iree_arena_block_pool_t pool_;
};

static iree_arena_block_pool_statistics_t QueryBlockPoolStatistics(
    const iree_arena_block_pool_t* block_pool) {
  iree_arena_block_pool_statistics_t statistics;
  iree_arena_block_pool_query_statistics(block_pool, &statistics);
  return statistics;
}

// Tracks system allocations around one representative warm operation and its
// identical steady-state repetitions. Fixed blocks remain reusable in the
// pool, while oversized allocations expose per-operation allocator churn.
class OperationMemoryTracker {
 public:
  explicit OperationMemoryTracker(iree_arena_block_pool_t* block_pool)
      : block_pool_(block_pool),
        baseline_(QueryBlockPoolStatistics(block_pool)) {}

  OperationMemoryTracker(const OperationMemoryTracker&) = delete;
  OperationMemoryTracker& operator=(const OperationMemoryTracker&) = delete;

  void MarkWarmupComplete(const loom_module_t* module) {
    module_arena_used_bytes_ = module->arena.used_allocation_size;
    module_arena_owned_bytes_ = module->arena.total_allocation_size;
    warmup_ = QueryBlockPoolStatistics(block_pool_);
  }

  void SetCounters(benchmark::State& state) const {
    const iree_arena_block_pool_statistics_t current =
        QueryBlockPoolStatistics(block_pool_);
    const uint64_t steady_block_growth_count =
        current.block_system_allocation_count -
        warmup_.block_system_allocation_count;
    const uint64_t steady_block_growth_bytes =
        current.block_system_allocation_bytes -
        warmup_.block_system_allocation_bytes;
    const uint64_t steady_oversized_count =
        current.oversized_allocation_count - warmup_.oversized_allocation_count;
    const uint64_t steady_oversized_bytes =
        current.oversized_allocation_bytes - warmup_.oversized_allocation_bytes;

    state.counters["module_arena_used_bytes"] =
        (double)module_arena_used_bytes_;
    state.counters["module_arena_owned_bytes"] =
        (double)module_arena_owned_bytes_;
    state.counters["pool_block_system_bytes"] =
        (double)current.block_system_allocation_bytes;
    state.counters["pool_block_system_count"] =
        (double)current.block_system_allocation_count;
    state.counters["pool_block_warm_growth_bytes"] =
        (double)(warmup_.block_system_allocation_bytes -
                 baseline_.block_system_allocation_bytes);
    state.counters["pool_block_warm_growth_count"] =
        (double)(warmup_.block_system_allocation_count -
                 baseline_.block_system_allocation_count);
    state.counters["pool_block_steady_growth_bytes"] =
        (double)steady_block_growth_bytes;
    state.counters["pool_block_steady_growth_count"] =
        (double)steady_block_growth_count;
    state.counters["pool_oversized_warm_bytes"] =
        (double)(warmup_.oversized_allocation_bytes -
                 baseline_.oversized_allocation_bytes);
    state.counters["pool_oversized_warm_count"] =
        (double)(warmup_.oversized_allocation_count -
                 baseline_.oversized_allocation_count);
    state.counters["pool_oversized_bytes/iteration"] =
        (double)steady_oversized_bytes / (double)state.iterations();
    state.counters["pool_oversized_allocations/iteration"] =
        (double)steady_oversized_count / (double)state.iterations();

    if (steady_block_growth_count != 0) {
      state.SkipWithError("fixed block pool grew after representative warmup");
    }
  }

 private:
  iree_arena_block_pool_t* block_pool_;
  iree_arena_block_pool_statistics_t baseline_;
  iree_arena_block_pool_statistics_t warmup_ = {};
  iree_host_size_t module_arena_used_bytes_ = 0;
  iree_host_size_t module_arena_owned_bytes_ = 0;
};

class EncodingBenchmarkFixture {
 public:
  EncodingBenchmarkFixture() {
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables,
                    loom_encoding_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables,
                    loom_scalar_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables,
                    loom_scf_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables,
                    loom_vector_dialect_op_semantics);

    iree_host_size_t count = 0;
    const loom_condition_refinement_descriptor_t* condition_refinements =
        loom_encoding_dialect_condition_refinements(&count);
    IREE_CHECK_OK(loom_context_register_condition_refinements(
        &context_, LOOM_DIALECT_ENCODING, condition_refinements, count));
    const loom_parameterized_attr_descriptor_t* parameterized_attrs =
        loom_encoding_dialect_parameterized_attrs(&count);
    IREE_CHECK_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_ENCODING, parameterized_attrs, count));
    IREE_CHECK_OK(loom_test_dialect_register(&context_));
    IREE_CHECK_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));
  }

  EncodingBenchmarkFixture(const EncodingBenchmarkFixture&) = delete;
  EncodingBenchmarkFixture& operator=(const EncodingBenchmarkFixture&) = delete;

  ~EncodingBenchmarkFixture() { loom_context_deinitialize(&context_); }

  iree_status_t Parse(iree_string_view_t source, loom_module_t** out_module) {
    loom_text_parse_options_t options = {};
    options.max_errors = 1;
    return loom_text_parse(source, IREE_SV("encoding_benchmark.loom"),
                           &context_, block_pool_.get(), &options, out_module);
  }

  loom_context_t* context() { return &context_; }

  iree_arena_block_pool_t* block_pool() { return block_pool_.get(); }

 private:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&vtable_count);
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    IREE_ASSERT(vtable_count == semantics_count);
    IREE_CHECK_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                (uint16_t)vtable_count));
    IREE_CHECK_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  ScopedBlockPool block_pool_;
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
        " = encoding.define #encoding.operand<element_format=f8e4m3, "
        "payload_elements=");
    source.append(std::to_string(payload_element_count));
    source.append(
        ", payload_packing=dense_lanes, "
        "rounding=finite_only> : encoding<schema>\n");
  }
  return source;
}

static std::string BuildDynamicEncodingModule(int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 180);
  source.append(
      "func.def @encodings(%layout: encoding<layout>, %schema: "
      "encoding<schema>) {\n");
  for (int64_t i = 0; i < definition_count; ++i) {
    source.append("  %storage");
    source.append(std::to_string(i));
    source.append(
        " = encoding.define #encoding.storage {layout = %layout : "
        "encoding<layout>, schema = %schema : encoding<schema>} : "
        "encoding<storage>\n");
  }
  source.append("  func.return\n}\n");
  return source;
}

static std::string BuildHadamardTransformModule(int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 112);
  source.append("func.def @transforms() {\n");
  for (int64_t i = 0; i < definition_count; ++i) {
    source.append("  %transform");
    source.append(std::to_string(i));
    source.append(
        " = encoding.define "
        "#transform.hadamard<normalization=orthonormal> : "
        "encoding<transform>\n");
  }
  source.append("  func.return\n}\n");
  return source;
}

static std::string BuildVectorEncodingUseModule(int64_t pair_count) {
  std::string source;
  source.reserve((size_t)pair_count * 420);
  source.append(
      "func.def @encoding_uses(%source: vector<32xf32>, %scale: "
      "vector<1xf32>) {\n"
      "  %schema = encoding.define "
      "#encoding.operand<element_format=f8e4m3fn, "
      "payload_elements=32, payload_packing=dense_lanes, "
      "scale_format=f32, "
      "scale_group_elements=32, scale_operands=1, "
      "scale_topology=block_1d, affine=scale_only> : "
      "encoding<schema>\n");
  for (int64_t i = 0; i < pair_count; ++i) {
    source.append("  %encoded");
    source.append(std::to_string(i));
    source.append(
        " = vector.encode %source using %schema {scale = %scale : "
        "vector<1xf32>} : vector<32xf32>, encoding<schema> -> "
        "vector<32xf8E4M3>\n");
    source.append("  %decoded");
    source.append(std::to_string(i));
    source.append(" = vector.decode %encoded");
    source.append(std::to_string(i));
    source.append(
        " using %schema {scale = %scale : vector<1xf32>} : "
        "vector<32xf8E4M3>, encoding<schema> -> vector<32xf32>\n");
  }
  source.append("  func.return\n}\n");
  return source;
}

static std::string BuildDynamicEncodingQueryBranchModule(int64_t branch_count) {
  std::string source;
  source.reserve((size_t)branch_count * 480);
  source.append("func.def @query_branches(%schema: encoding<schema>) {\n");
  for (int64_t i = 0; i < branch_count; ++i) {
    const bool use_exact_query = (i & 1) == 0;
    const char* query =
        use_exact_query
            ? " = encoding.isa<#encoding.operand<affine=scale_plus_min, "
              "element_format=u4, payload_elements=256, "
              "payload_packing=multi_stream, payload_registers=8>> %schema "
              ": encoding<schema>\n"
            : " = encoding.matches<element_format = u4, affine = "
              "scale_plus_min> %schema : encoding<schema>\n";
    source.append("  %condition");
    source.append(std::to_string(i));
    source.append(query);
    source.append("  scf.if %condition");
    source.append(std::to_string(i));
    source.append(" {\n    %known");
    source.append(std::to_string(i));
    source.append(query);
    source.append("    test.use %known");
    source.append(std::to_string(i));
    source.append(" : i1\n  }\n");
  }
  source.append("  func.return\n}\n");
  return source;
}

static loom_module_t* ParseModule(EncodingBenchmarkFixture& fixture,
                                  const std::string& source) {
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(fixture.Parse(
      iree_make_string_view(source.data(), source.size()), &module));
  IREE_ASSERT(module != nullptr);
  return module;
}

static loom_func_like_t GetOnlyFunction(loom_module_t* module) {
  loom_block_t* module_block = loom_module_block(module);
  if (module_block->op_count != 1) return (loom_func_like_t){0};
  loom_op_t* op = loom_block_op(module_block, 0);
  if (!loom_func_def_isa(op)) return (loom_func_like_t){0};
  return loom_func_like_cast(module, op);
}

static bool IsExpectedVectorEncodingUseModule(loom_module_t* module,
                                              int64_t pair_count) {
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) return false;
  loom_block_t* body = loom_region_entry_block(loom_func_like_body(function));
  if (!body || body->op_count != (iree_host_size_t)(pair_count * 2 + 2)) {
    return false;
  }

  int64_t encode_count = 0;
  int64_t decode_count = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(body, op) {
    encode_count += loom_vector_encode_isa(op) ? 1 : 0;
    decode_count += loom_vector_decode_isa(op) ? 1 : 0;
  }
  return encode_count == pair_count && decode_count == pair_count;
}

static iree_io_stream_t* CreateBytecodeStream() {
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));
  return stream;
}

static std::vector<uint8_t> SerializeModule(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool) {
  iree_io_stream_t* stream = CreateBytecodeStream();
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, block_pool));
  const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  IREE_ASSERT(stream_length >= 0);
  std::vector<uint8_t> bytes((size_t)stream_length);
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  iree_io_stream_release(stream);
  return bytes;
}

static loom_module_t* ReadModule(EncodingBenchmarkFixture& fixture,
                                 const std::vector<uint8_t>& bytes,
                                 iree_arena_block_pool_t* block_pool) {
  loom_bytecode_read_options_t options = {};
  loom_bytecode_read_result_t result = {};
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("static_encoding_benchmark.loombc"), fixture.context(),
      block_pool, &options, &result, &module, iree_allocator_system()));
  IREE_ASSERT(module != nullptr);
  return module;
}

static bool IsExpectedDistinctModule(const loom_module_t* module,
                                     int64_t definition_count) {
  if (module->encodings.count != (iree_host_size_t)definition_count) {
    return false;
  }
  loom_value_fact_storage_schema_t schema = {};
  return loom_encoding_query_static_storage_schema(
             module, (uint16_t)definition_count, &schema) &&
         schema.encoded_operand.payload_element_count ==
             (uint32_t)definition_count;
}

static void BenchmarkParseStaticEncodings(benchmark::State& state,
                                          SchemaPopulation population) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildEncodingModule(population, definition_count);
  EncodingBenchmarkFixture fixture;
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  loom_module_t* warm_module = ParseModule(fixture, source);
  memory_tracker.MarkWarmupComplete(warm_module);
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ParseModule(fixture, source);
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}

static void BM_ParseRepeatedStaticEncoding(benchmark::State& state) {
  BenchmarkParseStaticEncodings(state, SchemaPopulation::kRepeated);
}
BENCHMARK(BM_ParseRepeatedStaticEncoding)->Arg(256)->Arg(4096);

static void BM_ParseDistinctStaticEncodings(benchmark::State& state) {
  BenchmarkParseStaticEncodings(state, SchemaPopulation::kDistinct);
}
BENCHMARK(BM_ParseDistinctStaticEncodings)->Apply(ScaledEncodingCounts);

static void BM_PrintDistinctStaticEncodings(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  loom_output_stream_t preflight_stream;
  loom_output_stream_null(&preflight_stream);
  IREE_CHECK_OK(loom_text_print_module(module, &preflight_stream,
                                       LOOM_TEXT_PRINT_DEFAULT));
  const int64_t printed_byte_count = (int64_t)preflight_stream.offset;
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    IREE_CHECK_OK(
        loom_text_print_module(module, &stream, LOOM_TEXT_PRINT_DEFAULT));
    benchmark::DoNotOptimize(stream.offset);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * printed_byte_count);
  loom_module_free(module);
}
BENCHMARK(BM_PrintDistinctStaticEncodings)->Apply(ScaledEncodingCounts);

static void BM_WriteDistinctStaticEncodings(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  if (!IsExpectedDistinctModule(module, definition_count)) {
    state.SkipWithError("distinct static encoding module was malformed");
    loom_module_free(module);
    return;
  }

  ScopedBlockPool write_pool;
  OperationMemoryTracker memory_tracker(write_pool.get());
  iree_io_stream_t* stream = CreateBytecodeStream();
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
  const int64_t serialized_byte_count = (int64_t)iree_io_stream_length(stream);
  memory_tracker.MarkWarmupComplete(module);
  state.counters["serialized_bytes"] = (double)serialized_byte_count;

  for (auto _ : state) {
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
    benchmark::DoNotOptimize(stream);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * serialized_byte_count);
  iree_io_stream_release(stream);
  loom_module_free(module);
}
BENCHMARK(BM_WriteDistinctStaticEncodings)->Apply(ScaledEncodingCounts);

static void BM_ReadDistinctStaticEncodings(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* source_module = ParseModule(fixture, source);
  const std::vector<uint8_t> bytes =
      SerializeModule(source_module, fixture.block_pool());
  loom_module_free(source_module);

  ScopedBlockPool read_pool;
  OperationMemoryTracker memory_tracker(read_pool.get());
  loom_module_t* warm_module = ReadModule(fixture, bytes, read_pool.get());
  if (!IsExpectedDistinctModule(warm_module, definition_count)) {
    state.SkipWithError("bytecode reader changed static encoding schemas");
    loom_module_free(warm_module);
    return;
  }
  memory_tracker.MarkWarmupComplete(warm_module);
  state.counters["serialized_bytes"] = (double)bytes.size();
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ReadModule(fixture, bytes, read_pool.get());
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)bytes.size());
}
BENCHMARK(BM_ReadDistinctStaticEncodings)->Apply(ScaledEncodingCounts);

static void BM_QueryDistinctStaticEncodingRoles(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());
  if (module->encodings.count != (iree_host_size_t)definition_count) {
    state.SkipWithError("distinct schemas were unexpectedly deduplicated");
    loom_module_free(module);
    return;
  }

  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    loom_encoding_role_t role =
        loom_encoding_static_role(module, &module->encodings.entries[i]);
    benchmark::DoNotOptimize(role);
  }
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
      loom_encoding_role_t role =
          loom_encoding_static_role(module, &module->encodings.entries[i]);
      benchmark::DoNotOptimize(role);
    }
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryDistinctStaticEncodingRoles)->Apply(ScaledEncodingCounts);

static void BM_QueryDistinctStaticStorageSchemas(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source =
      BuildEncodingModule(SchemaPopulation::kDistinct, definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());
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
  memory_tracker.MarkWarmupComplete(module);

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
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryDistinctStaticStorageSchemas)->Apply(ScaledEncodingCounts);

static void BM_QueryStaticFp8OperandSchema(benchmark::State& state) {
  const int64_t query_count = state.range(0);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(
      fixture,
      "%schema = encoding.define #encoding.operand<element_format=f8e4m3fn, "
      "payload_elements=1, payload_packing=dense_lanes, rounding=finite_only> "
      ": encoding<schema>\n");
  OperationMemoryTracker memory_tracker(fixture.block_pool());
  const loom_op_t* define_op =
      loom_block_const_op(loom_module_block(module), 0);
  const uint16_t encoding_id = loom_encoding_define_spec(define_op);

  loom_value_fact_storage_schema_t schema = {};
  if (!loom_encoding_query_static_storage_schema(module, encoding_id,
                                                 &schema) ||
      schema.encoded_operand.element_format !=
          LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN ||
      schema.encoded_operand.rounding_policy !=
          LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY) {
    state.SkipWithError("static FP8 operand schema query produced wrong facts");
    loom_module_free(module);
    return;
  }
  memory_tracker.MarkWarmupComplete(module);

  for (auto _ : state) {
    for (int64_t i = 0; i < query_count; ++i) {
      bool resolved = loom_encoding_query_static_storage_schema(
          module, encoding_id, &schema);
      benchmark::DoNotOptimize(resolved);
      benchmark::DoNotOptimize(schema);
    }
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * query_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryStaticFp8OperandSchema)->Apply(ScaledEncodingCounts);

static void BM_ParseDynamicEncodingDefinitions(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  loom_module_t* warm_module = ParseModule(fixture, source);
  memory_tracker.MarkWarmupComplete(warm_module);
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ParseModule(fixture, source);
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}
BENCHMARK(BM_ParseDynamicEncodingDefinitions)->Apply(ScaledEncodingCounts);

static void BM_PrintDynamicEncodingDefinitions(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  loom_output_stream_t preflight_stream;
  loom_output_stream_null(&preflight_stream);
  IREE_CHECK_OK(loom_text_print_module(module, &preflight_stream,
                                       LOOM_TEXT_PRINT_DEFAULT));
  const int64_t printed_byte_count = (int64_t)preflight_stream.offset;
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    IREE_CHECK_OK(
        loom_text_print_module(module, &stream, LOOM_TEXT_PRINT_DEFAULT));
    benchmark::DoNotOptimize(stream.offset);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * printed_byte_count);
  loom_module_free(module);
}
BENCHMARK(BM_PrintDynamicEncodingDefinitions)->Apply(ScaledEncodingCounts);

static void BM_WriteDynamicEncodingDefinitions(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);

  ScopedBlockPool write_pool;
  OperationMemoryTracker memory_tracker(write_pool.get());
  iree_io_stream_t* stream = CreateBytecodeStream();
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
  const int64_t serialized_byte_count = (int64_t)iree_io_stream_length(stream);
  memory_tracker.MarkWarmupComplete(module);
  state.counters["serialized_bytes"] = (double)serialized_byte_count;

  for (auto _ : state) {
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
    benchmark::DoNotOptimize(stream);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * serialized_byte_count);
  iree_io_stream_release(stream);
  loom_module_free(module);
}
BENCHMARK(BM_WriteDynamicEncodingDefinitions)->Apply(ScaledEncodingCounts);

static void BM_ReadDynamicEncodingDefinitions(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* source_module = ParseModule(fixture, source);
  const std::vector<uint8_t> bytes =
      SerializeModule(source_module, fixture.block_pool());
  loom_module_free(source_module);

  ScopedBlockPool read_pool;
  OperationMemoryTracker memory_tracker(read_pool.get());
  loom_module_t* warm_module = ReadModule(fixture, bytes, read_pool.get());
  if (!GetOnlyFunction(warm_module).op) {
    state.SkipWithError("bytecode reader changed dynamic encoding function");
    loom_module_free(warm_module);
    return;
  }
  memory_tracker.MarkWarmupComplete(warm_module);
  state.counters["serialized_bytes"] = (double)bytes.size();
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ReadModule(fixture, bytes, read_pool.get());
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)bytes.size());
}
BENCHMARK(BM_ReadDynamicEncodingDefinitions)->Apply(ScaledEncodingCounts);

static void VerifyEncodingModule(loom_module_t* module) {
  loom_verify_options_t options = {};
  loom_verify_result_t result = {};
  IREE_CHECK_OK(loom_verify_module(module, &options, &result));
  IREE_ASSERT(result.error_count == 0);
}

static void BM_ParseVectorEncodingUses(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  loom_module_t* warm_module = ParseModule(fixture, source);
  if (!IsExpectedVectorEncodingUseModule(warm_module, pair_count)) {
    state.SkipWithError("vector encoding use module was malformed");
    loom_module_free(warm_module);
    return;
  }
  memory_tracker.MarkWarmupComplete(warm_module);
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ParseModule(fixture, source);
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}
BENCHMARK(BM_ParseVectorEncodingUses)->Apply(ScaledEncodingCounts);

static void BM_PrintVectorEncodingUses(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());
  if (!IsExpectedVectorEncodingUseModule(module, pair_count)) {
    state.SkipWithError("vector encoding use module was malformed");
    loom_module_free(module);
    return;
  }

  loom_output_stream_t preflight_stream;
  loom_output_stream_null(&preflight_stream);
  IREE_CHECK_OK(loom_text_print_module(module, &preflight_stream,
                                       LOOM_TEXT_PRINT_DEFAULT));
  const int64_t printed_byte_count = (int64_t)preflight_stream.offset;
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    IREE_CHECK_OK(
        loom_text_print_module(module, &stream, LOOM_TEXT_PRINT_DEFAULT));
    benchmark::DoNotOptimize(stream.offset);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  state.SetBytesProcessed(state.iterations() * printed_byte_count);
  loom_module_free(module);
}
BENCHMARK(BM_PrintVectorEncodingUses)->Apply(ScaledEncodingCounts);

static void BM_WriteVectorEncodingUses(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  if (!IsExpectedVectorEncodingUseModule(module, pair_count)) {
    state.SkipWithError("vector encoding use module was malformed");
    loom_module_free(module);
    return;
  }

  ScopedBlockPool write_pool;
  OperationMemoryTracker memory_tracker(write_pool.get());
  iree_io_stream_t* stream = CreateBytecodeStream();
  IREE_CHECK_OK(
      loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
  const int64_t serialized_byte_count = (int64_t)iree_io_stream_length(stream);
  memory_tracker.MarkWarmupComplete(module);
  state.counters["serialized_bytes"] = (double)serialized_byte_count;

  for (auto _ : state) {
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, nullptr, write_pool.get()));
    benchmark::DoNotOptimize(stream);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  state.SetBytesProcessed(state.iterations() * serialized_byte_count);
  iree_io_stream_release(stream);
  loom_module_free(module);
}
BENCHMARK(BM_WriteVectorEncodingUses)->Apply(ScaledEncodingCounts);

static void BM_ReadVectorEncodingUses(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* source_module = ParseModule(fixture, source);
  const std::vector<uint8_t> bytes =
      SerializeModule(source_module, fixture.block_pool());
  loom_module_free(source_module);

  ScopedBlockPool read_pool;
  OperationMemoryTracker memory_tracker(read_pool.get());
  loom_module_t* warm_module = ReadModule(fixture, bytes, read_pool.get());
  if (!IsExpectedVectorEncodingUseModule(warm_module, pair_count)) {
    state.SkipWithError("bytecode reader changed vector encoding uses");
    loom_module_free(warm_module);
    return;
  }
  memory_tracker.MarkWarmupComplete(warm_module);
  state.counters["serialized_bytes"] = (double)bytes.size();
  loom_module_free(warm_module);

  for (auto _ : state) {
    loom_module_t* module = ReadModule(fixture, bytes, read_pool.get());
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  state.SetBytesProcessed(state.iterations() * (int64_t)bytes.size());
}
BENCHMARK(BM_ReadVectorEncodingUses)->Apply(ScaledEncodingCounts);

static void BM_VerifyVectorEncodingUses(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  VerifyEncodingModule(module);
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    VerifyEncodingModule(module);
    benchmark::DoNotOptimize(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  loom_module_free(module);
}
BENCHMARK(BM_VerifyVectorEncodingUses)->Apply(ScaledEncodingCounts);

static void BM_ComputeVectorEncodingFacts(benchmark::State& state) {
  const int64_t pair_count = state.range(0);
  const std::string source = BuildVectorEncodingUseModule(pair_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  VerifyEncodingModule(module);
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) {
    state.SkipWithError("vector encoding use function was malformed");
    loom_module_free(module);
    return;
  }

  ScopedBlockPool analysis_pool;
  OperationMemoryTracker memory_tracker(analysis_pool.get());
  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(analysis_pool.get(), &analysis_arena);
  loom_value_fact_table_t warm_table = {};
  IREE_CHECK_OK(loom_value_fact_table_initialize(&warm_table, &analysis_arena,
                                                 module->values.count));
  IREE_CHECK_OK(loom_value_fact_table_compute(&warm_table, module, function));
  state.counters["analysis_arena_used_bytes"] =
      (double)analysis_arena.used_allocation_size;
  state.counters["analysis_arena_owned_bytes"] =
      (double)analysis_arena.total_allocation_size;
  memory_tracker.MarkWarmupComplete(module);
  iree_arena_deinitialize(&analysis_arena);

  for (auto _ : state) {
    iree_arena_initialize(analysis_pool.get(), &analysis_arena);
    loom_value_fact_table_t table = {};
    IREE_CHECK_OK(loom_value_fact_table_initialize(&table, &analysis_arena,
                                                   module->values.count));
    IREE_CHECK_OK(loom_value_fact_table_compute(&table, module, function));
    benchmark::DoNotOptimize(table);
    iree_arena_deinitialize(&analysis_arena);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * pair_count * 2);
  loom_module_free(module);
}
BENCHMARK(BM_ComputeVectorEncodingFacts)->Apply(ScaledEncodingCounts);

static void BM_VerifyDynamicEncodingDefinitions(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  VerifyEncodingModule(module);
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    VerifyEncodingModule(module);
    benchmark::DoNotOptimize(module);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_VerifyDynamicEncodingDefinitions)->Apply(ScaledEncodingCounts);

static void BM_ComputeDynamicEncodingFacts(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildDynamicEncodingModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  VerifyEncodingModule(module);
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) {
    state.SkipWithError("dynamic encoding function was malformed");
    loom_module_free(module);
    return;
  }

  ScopedBlockPool analysis_pool;
  OperationMemoryTracker memory_tracker(analysis_pool.get());
  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(analysis_pool.get(), &analysis_arena);
  loom_value_fact_table_t warm_table = {};
  IREE_CHECK_OK(loom_value_fact_table_initialize(&warm_table, &analysis_arena,
                                                 module->values.count));
  IREE_CHECK_OK(loom_value_fact_table_compute(&warm_table, module, function));
  state.counters["analysis_arena_used_bytes"] =
      (double)analysis_arena.used_allocation_size;
  state.counters["analysis_arena_owned_bytes"] =
      (double)analysis_arena.total_allocation_size;
  memory_tracker.MarkWarmupComplete(module);
  iree_arena_deinitialize(&analysis_arena);

  for (auto _ : state) {
    iree_arena_initialize(analysis_pool.get(), &analysis_arena);
    loom_value_fact_table_t table = {};
    IREE_CHECK_OK(loom_value_fact_table_initialize(&table, &analysis_arena,
                                                   module->values.count));
    IREE_CHECK_OK(loom_value_fact_table_compute(&table, module, function));
    benchmark::DoNotOptimize(table);
    iree_arena_deinitialize(&analysis_arena);
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_ComputeDynamicEncodingFacts)->Apply(ScaledEncodingCounts);

static bool QueryVerifiedHadamardTransforms(loom_module_t* module,
                                            int64_t definition_count) {
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) return false;
  loom_block_t* body = loom_region_entry_block(loom_func_like_body(function));
  if (!body || body->op_count != (iree_host_size_t)definition_count + 1) {
    return false;
  }
  int64_t query_count = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(body, op) {
    if (loom_func_return_isa(op)) break;
    if (!loom_encoding_define_isa(op)) return false;
    loom_encoding_hadamard_descriptor_t descriptor;
    if (!loom_encoding_hadamard_try_read_verified_descriptor(
            module, loom_encoding_define_result(op), &descriptor) ||
        descriptor.normalization !=
            LOOM_ENCODING_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
      return false;
    }
    benchmark::DoNotOptimize(descriptor);
    ++query_count;
  }
  return query_count == definition_count;
}

static void BM_QueryVerifiedEncodingHadamardTransforms(
    benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  const std::string source = BuildHadamardTransformModule(definition_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* module = ParseModule(fixture, source);
  VerifyEncodingModule(module);
  OperationMemoryTracker memory_tracker(fixture.block_pool());

  if (!QueryVerifiedHadamardTransforms(module, definition_count)) {
    state.SkipWithError("Hadamard transform query produced wrong data");
    loom_module_free(module);
    return;
  }
  memory_tracker.MarkWarmupComplete(module);
  for (auto _ : state) {
    if (!QueryVerifiedHadamardTransforms(module, definition_count)) {
      state.SkipWithError("Hadamard transform query produced wrong data");
      break;
    }
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * definition_count);
  loom_module_free(module);
}
BENCHMARK(BM_QueryVerifiedEncodingHadamardTransforms)
    ->Apply(ScaledEncodingCounts);

static bool IsCanonicalDynamicEncodingQueryBranchModule(loom_module_t* module,
                                                        int64_t branch_count) {
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) return false;
  loom_block_t* body = loom_region_entry_block(loom_func_like_body(function));
  if (!body) return false;

  int64_t outer_query_count = 0;
  int64_t branch_count_seen = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(body, op) {
    if (loom_encoding_isa_isa(op) || loom_encoding_matches_isa(op)) {
      ++outer_query_count;
      continue;
    }
    if (!loom_scf_if_isa(op)) continue;
    ++branch_count_seen;
    loom_block_t* branch = loom_region_entry_block(loom_scf_if_then_region(op));
    if (!branch) return false;
    loom_op_t* branch_op = nullptr;
    loom_block_for_each_op(branch, branch_op) {
      if (loom_encoding_isa_isa(branch_op) ||
          loom_encoding_matches_isa(branch_op)) {
        return false;
      }
    }
  }
  return outer_query_count == branch_count && branch_count_seen == branch_count;
}

static iree_status_t CanonicalizeDynamicEncodingQueryBranches(
    loom_module_t* module, iree_arena_block_pool_t* block_pool,
    loom_canonicalizer_result_t* out_result) {
  loom_func_like_t function = GetOnlyFunction(module);
  if (!function.op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "query branch module has no function");
  }

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(block_pool, &value_facts);
  loom_canonicalizer_t canonicalizer;
  iree_status_t status = loom_canonicalizer_initialize(
      module, &pass_arena, &value_facts, &canonicalizer);
  if (iree_status_is_ok(status)) {
    status = loom_canonicalizer_run_function(&canonicalizer, function,
                                             /*options=*/nullptr, out_result);
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
  return status;
}

static void BM_CanonicalizeDynamicEncodingQueryBranches(
    benchmark::State& state) {
  const int64_t branch_count = state.range(0);
  const std::string source =
      BuildDynamicEncodingQueryBranchModule(branch_count);
  EncodingBenchmarkFixture fixture;
  loom_module_t* source_module = ParseModule(fixture, source);
  VerifyEncodingModule(source_module);
  const std::vector<uint8_t> bytes =
      SerializeModule(source_module, fixture.block_pool());
  loom_module_free(source_module);

  ScopedBlockPool module_pool;
  ScopedBlockPool pass_pool;
  OperationMemoryTracker memory_tracker(pass_pool.get());
  loom_module_t* warm_module = ReadModule(fixture, bytes, module_pool.get());
  loom_canonicalizer_result_t warm_result = {};
  IREE_CHECK_OK(CanonicalizeDynamicEncodingQueryBranches(
      warm_module, pass_pool.get(), &warm_result));
  if (!warm_result.changed ||
      !IsCanonicalDynamicEncodingQueryBranchModule(warm_module, branch_count)) {
    state.SkipWithError("dynamic encoding queries did not canonicalize");
    loom_module_free(warm_module);
    return;
  }
  memory_tracker.MarkWarmupComplete(warm_module);
  loom_module_free(warm_module);

  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* module = ReadModule(fixture, bytes, module_pool.get());
    loom_canonicalizer_result_t result = {};
    state.ResumeTiming();

    iree_status_t status = CanonicalizeDynamicEncodingQueryBranches(
        module, pass_pool.get(), &result);

    state.PauseTiming();
    IREE_CHECK_OK(status);
    if (!result.changed ||
        !IsCanonicalDynamicEncodingQueryBranchModule(module, branch_count)) {
      state.SkipWithError("dynamic encoding queries did not canonicalize");
    }
    loom_module_free(module);
    state.ResumeTiming();
  }
  memory_tracker.SetCounters(state);
  state.SetItemsProcessed(state.iterations() * branch_count * 2);
}
BENCHMARK(BM_CanonicalizeDynamicEncodingQueryBranches)
    ->Apply(ScaledEncodingCounts);

}  // namespace
