// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks built-in type and location syntax at scales that expose small
// per-occurrence costs in generated modules. Focused scalar classification
// rows isolate vocabulary lookup, while full-module rows include the actual
// parser and canonical printer paths used by the JIT.

#include <array>
#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"

namespace {

enum class SyntaxWorkload {
  kScalarTypes,
  kShapedTypes,
  kTaggedLocations,
};

static constexpr std::array<const char*, 13> kScalarTypeNames = {
    "index",  "offset", "i1",  "i8",   "i16", "i32", "i64",
    "f8E4M3", "f8E5M2", "f16", "bf16", "f32", "f64",
};

static constexpr std::array<const char*, 5> kShapedTypeNames = {
    "vector<16xbf16>",
    "tile<8x8xf32, %layout>",
    "tensor<1x64x64xf16, %layout>",
    "view<1024x512xf8E4M3, %layout>",
    "pool<4096>",
};

static constexpr std::array<const char*, 4> kLocationTagNames = {
    "sanitizer_site",
    "template_instantiation",
    "tile_lowering",
    "ukernel_selection",
};

class TypeSyntaxBenchmark {
 public:
  TypeSyntaxBenchmark() {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_CHECK_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_CHECK_OK(loom_context_finalize(&context_));
  }

  TypeSyntaxBenchmark(const TypeSyntaxBenchmark&) = delete;
  TypeSyntaxBenchmark& operator=(const TypeSyntaxBenchmark&) = delete;

  ~TypeSyntaxBenchmark() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Parse(iree_string_view_t source, loom_module_t** out_module) {
    loom_text_parse_options_t options = {};
    options.max_errors = 1;
    return loom_text_parse(source, IREE_SV("type_syntax_benchmark.loom"),
                           &context_, &block_pool_, &options, out_module);
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static std::string BuildSyntaxModule(SyntaxWorkload workload,
                                     int64_t operation_count) {
  // Keep pathological operation counts within the compact source line range.
  // Whitespace separates operations, so packing several on each source line
  // changes neither the parsed module nor its canonical printed form.
  static constexpr int64_t kOperationsPerSourceLine = 8;
  size_t estimated_bytes_per_operation = 0;
  switch (workload) {
    case SyntaxWorkload::kScalarTypes:
      estimated_bytes_per_operation = 40;
      break;
    case SyntaxWorkload::kShapedTypes:
      estimated_bytes_per_operation = 56;
      break;
    case SyntaxWorkload::kTaggedLocations:
      estimated_bytes_per_operation = 72;
      break;
  }
  std::string source;
  source.reserve((size_t)operation_count * estimated_bytes_per_operation);
  source.append("test.func @type_syntax(%layout: encoding<layout>) {\n");
  for (int64_t i = 0; i < operation_count; ++i) {
    source.append("  %v");
    source.append(std::to_string(i));
    source.append(" = test.constant 0 : ");
    switch (workload) {
      case SyntaxWorkload::kScalarTypes: {
        source.append(kScalarTypeNames[(size_t)i % kScalarTypeNames.size()]);
        break;
      }
      case SyntaxWorkload::kShapedTypes:
        source.append(kShapedTypeNames[(size_t)i % kShapedTypeNames.size()]);
        break;
      case SyntaxWorkload::kTaggedLocations:
        source.append("i32 loc(tagged<");
        source.append(kLocationTagNames[(size_t)i % kLocationTagNames.size()]);
        source.append(", \"00\">)");
        break;
    }
    source.push_back((i + 1) % kOperationsPerSourceLine == 0 ? '\n' : ' ');
  }
  if (operation_count % kOperationsPerSourceLine != 0) source.push_back('\n');
  source.append("  test.yield\n}\n");
  return source;
}

static void BenchmarkParseModule(benchmark::State& state,
                                 SyntaxWorkload workload) {
  const int64_t operation_count = state.range(0);
  std::string source = BuildSyntaxModule(workload, operation_count);
  TypeSyntaxBenchmark fixture;
  for (auto _ : state) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(fixture.Parse(
        iree_make_string_view(source.data(), source.size()), &module));
    if (!module) {
      state.SkipWithError("loom_text_parse produced diagnostics");
      break;
    }
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  state.SetItemsProcessed(state.iterations() * operation_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}

static void BenchmarkPrintModule(benchmark::State& state,
                                 SyntaxWorkload workload) {
  const int64_t operation_count = state.range(0);
  std::string source = BuildSyntaxModule(workload, operation_count);
  TypeSyntaxBenchmark fixture;
  loom_module_t* module = nullptr;
  IREE_CHECK_OK(fixture.Parse(
      iree_make_string_view(source.data(), source.size()), &module));
  if (!module) {
    state.SkipWithError("loom_text_parse produced diagnostics");
    return;
  }

  loom_text_print_flags_t flags = LOOM_TEXT_PRINT_DEFAULT;
  if (workload == SyntaxWorkload::kTaggedLocations) {
    flags |= LOOM_TEXT_PRINT_LOCATIONS;
  }
  loom_output_stream_t preflight_stream;
  loom_output_stream_null(&preflight_stream);
  IREE_CHECK_OK(loom_text_print_module(module, &preflight_stream, flags));
  const int64_t printed_byte_count = (int64_t)preflight_stream.offset;
  for (auto _ : state) {
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    IREE_CHECK_OK(loom_text_print_module(module, &stream, flags));
    benchmark::DoNotOptimize(stream.offset);
  }
  state.SetItemsProcessed(state.iterations() * operation_count);
  state.SetBytesProcessed(state.iterations() * printed_byte_count);
  loom_module_free(module);
}

static void BM_ScalarTypeClassifyKnown(benchmark::State& state) {
  for (auto _ : state) {
    for (const char* type_name : kScalarTypeNames) {
      loom_scalar_type_t type = 0;
      bool matched =
          loom_scalar_type_parse(iree_make_cstring_view(type_name), &type);
      benchmark::DoNotOptimize(matched);
      benchmark::DoNotOptimize(type);
    }
  }
  state.SetItemsProcessed(state.iterations() * kScalarTypeNames.size());
}
BENCHMARK(BM_ScalarTypeClassifyKnown);

static void BM_ScalarTypeClassifyMiss(benchmark::State& state) {
  for (auto _ : state) {
    loom_scalar_type_t type = 0;
    bool matched = loom_scalar_type_parse(IREE_SV("not_a_scalar_type"), &type);
    benchmark::DoNotOptimize(matched);
    benchmark::DoNotOptimize(type);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ScalarTypeClassifyMiss);

static void BM_ParseScalarTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kScalarTypes);
}
BENCHMARK(BM_ParseScalarTypes)->Arg(1000)->Arg(10000);

static void BM_PrintScalarTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kScalarTypes);
}
BENCHMARK(BM_PrintScalarTypes)->Arg(1000)->Arg(10000);

static void BM_ParseShapedTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kShapedTypes);
}
BENCHMARK(BM_ParseShapedTypes)->Arg(1000)->Arg(10000);

static void BM_PrintShapedTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kShapedTypes);
}
BENCHMARK(BM_PrintShapedTypes)->Arg(1000)->Arg(10000);

static void BM_ParseTaggedLocations(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kTaggedLocations);
}
BENCHMARK(BM_ParseTaggedLocations)->Arg(1000)->Arg(10000);

static void BM_PrintTaggedLocations(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kTaggedLocations);
}
BENCHMARK(BM_PrintTaggedLocations)->Arg(1000)->Arg(10000);

}  // namespace
