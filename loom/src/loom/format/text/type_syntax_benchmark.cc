// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks built-in type and location syntax at scales that expose small
// per-occurrence costs in generated modules. Focused vocabulary classification
// rows isolate name lookup, while full-module rows include the actual parser
// and canonical printer paths used by the JIT.

#include <array>
#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/location.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"

namespace {

enum class SyntaxWorkload {
  kScalarTypes,
  kShapedTypes,
  kNativeDenseShapedTypes,
  kExplicitDenseShapedTypes,
  kTaggedLocations,
};

static constexpr int64_t kClassificationBatchCount = 4096;

static std::array<iree_string_view_t, LOOM_SCALAR_TYPE_COUNT_>
BuildScalarTypeNameViews() {
  std::array<iree_string_view_t, LOOM_SCALAR_TYPE_COUNT_> names;
  for (int i = 0; i < LOOM_SCALAR_TYPE_COUNT_; ++i) {
    names[(size_t)i] =
        iree_make_cstring_view(loom_scalar_type_name((loom_scalar_type_t)i));
  }
  return names;
}

static constexpr std::array<const char*, 5> kShapedTypeNames = {
    "vector<16xbf16>",
    "tile<8x8xf32, %layout>",
    "tensor<1x64x64xf16, %layout>",
    "view<1024x512xf8E4M3, %layout>",
    "pool<4096>",
};

static constexpr std::array<const char*, 4> kNativeDenseShapedTypeNames = {
    "tile<8x8xf32>",
    "tensor<1x64x64xf16>",
    "view<1024x512xf8E4M3>",
    "view<1x32x4096xbf16>",
};

static constexpr std::array<const char*, 4> kExplicitDenseShapedTypeNames = {
    "tile<8x8xf32, #encoding.layout.dense>",
    "tensor<1x64x64xf16, #encoding.layout.dense>",
    "view<1024x512xf8E4M3, #encoding.layout.dense>",
    "view<1x32x4096xbf16, #encoding.layout.dense>",
};

static constexpr std::array<loom_location_tag_t, 4> kBuiltinLocationTags = {
    LOOM_LOCATION_TAG_SANITIZER_SITE,
    LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION,
    LOOM_LOCATION_TAG_TILE_LOWERING,
    LOOM_LOCATION_TAG_UKERNEL_SELECTION,
};

static std::array<iree_string_view_t, kBuiltinLocationTags.size()>
BuildLocationTagNameViews() {
  std::array<iree_string_view_t, kBuiltinLocationTags.size()> names;
  for (size_t i = 0; i < kBuiltinLocationTags.size(); ++i) {
    names[i] = loom_location_tag_name(kBuiltinLocationTags[i]);
  }
  return names;
}

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
    case SyntaxWorkload::kNativeDenseShapedTypes:
    case SyntaxWorkload::kExplicitDenseShapedTypes:
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
        const loom_scalar_type_t scalar_type =
            (loom_scalar_type_t)(i % LOOM_SCALAR_TYPE_COUNT_);
        source.append(loom_scalar_type_name(scalar_type));
        break;
      }
      case SyntaxWorkload::kShapedTypes:
        source.append(kShapedTypeNames[(size_t)i % kShapedTypeNames.size()]);
        break;
      case SyntaxWorkload::kNativeDenseShapedTypes:
        source.append(
            kNativeDenseShapedTypeNames[(size_t)i %
                                        kNativeDenseShapedTypeNames.size()]);
        break;
      case SyntaxWorkload::kExplicitDenseShapedTypes:
        source.append(kExplicitDenseShapedTypeNames
                          [(size_t)i % kExplicitDenseShapedTypeNames.size()]);
        break;
      case SyntaxWorkload::kTaggedLocations: {
        source.append("i32 loc(tagged<");
        const iree_string_view_t tag_name = loom_location_tag_name(
            kBuiltinLocationTags[(size_t)i % kBuiltinLocationTags.size()]);
        source.append(tag_name.data, tag_name.size);
        source.append(", \"00\">)");
        break;
      }
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
  const auto type_names = BuildScalarTypeNameViews();
  for (auto _ : state) {
    for (int64_t batch = 0; batch < kClassificationBatchCount; ++batch) {
      for (iree_string_view_t type_name : type_names) {
        benchmark::DoNotOptimize(type_name);
        loom_scalar_type_t type = 0;
        bool matched = loom_scalar_type_parse(type_name, &type);
        benchmark::DoNotOptimize(matched);
        benchmark::DoNotOptimize(type);
      }
    }
  }
  state.SetItemsProcessed(state.iterations() * LOOM_SCALAR_TYPE_COUNT_ *
                          kClassificationBatchCount);
}
BENCHMARK(BM_ScalarTypeClassifyKnown);

static void BM_ScalarTypeClassifyMiss(benchmark::State& state) {
  for (auto _ : state) {
    for (int64_t batch = 0; batch < kClassificationBatchCount; ++batch) {
      iree_string_view_t type_name = IREE_SV("not_a_scalar_type");
      benchmark::DoNotOptimize(type_name);
      loom_scalar_type_t type = 0;
      bool matched = loom_scalar_type_parse(type_name, &type);
      benchmark::DoNotOptimize(matched);
      benchmark::DoNotOptimize(type);
    }
  }
  state.SetItemsProcessed(state.iterations() * kClassificationBatchCount);
}
BENCHMARK(BM_ScalarTypeClassifyMiss);

static void BM_LocationTagClassifyKnown(benchmark::State& state) {
  const auto tag_names = BuildLocationTagNameViews();
  for (auto _ : state) {
    for (int64_t batch = 0; batch < kClassificationBatchCount; ++batch) {
      for (iree_string_view_t tag_name : tag_names) {
        benchmark::DoNotOptimize(tag_name);
        loom_location_tag_t tag = LOOM_LOCATION_TAG_INVALID;
        bool matched = loom_location_tag_parse(tag_name, &tag);
        benchmark::DoNotOptimize(matched);
        benchmark::DoNotOptimize(tag);
      }
    }
  }
  state.SetItemsProcessed(state.iterations() * kBuiltinLocationTags.size() *
                          kClassificationBatchCount);
}
BENCHMARK(BM_LocationTagClassifyKnown);

static void BM_LocationTagClassifyMiss(benchmark::State& state) {
  for (auto _ : state) {
    for (int64_t batch = 0; batch < kClassificationBatchCount; ++batch) {
      iree_string_view_t tag_name = IREE_SV("not_a_location_tag");
      benchmark::DoNotOptimize(tag_name);
      loom_location_tag_t tag = LOOM_LOCATION_TAG_INVALID;
      bool matched = loom_location_tag_parse(tag_name, &tag);
      benchmark::DoNotOptimize(matched);
      benchmark::DoNotOptimize(tag);
    }
  }
  state.SetItemsProcessed(state.iterations() * kClassificationBatchCount);
}
BENCHMARK(BM_LocationTagClassifyMiss);

static void BM_ParseScalarTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kScalarTypes);
}
BENCHMARK(BM_ParseScalarTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_PrintScalarTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kScalarTypes);
}
BENCHMARK(BM_PrintScalarTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_ParseShapedTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kShapedTypes);
}
BENCHMARK(BM_ParseShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_PrintShapedTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kShapedTypes);
}
BENCHMARK(BM_PrintShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_ParseNativeDenseShapedTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kNativeDenseShapedTypes);
}
BENCHMARK(BM_ParseNativeDenseShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_PrintNativeDenseShapedTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kNativeDenseShapedTypes);
}
BENCHMARK(BM_PrintNativeDenseShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_ParseExplicitDenseShapedTypes(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kExplicitDenseShapedTypes);
}
BENCHMARK(BM_ParseExplicitDenseShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_PrintExplicitDenseShapedTypes(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kExplicitDenseShapedTypes);
}
BENCHMARK(BM_PrintExplicitDenseShapedTypes)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_ParseTaggedLocations(benchmark::State& state) {
  BenchmarkParseModule(state, SyntaxWorkload::kTaggedLocations);
}
BENCHMARK(BM_ParseTaggedLocations)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

static void BM_PrintTaggedLocations(benchmark::State& state) {
  BenchmarkPrintModule(state, SyntaxWorkload::kTaggedLocations);
}
BENCHMARK(BM_PrintTaggedLocations)
    ->Arg(16)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

}  // namespace
