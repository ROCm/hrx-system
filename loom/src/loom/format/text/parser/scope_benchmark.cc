// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks parser SSA-name scope behavior for access patterns that show up in
// large generated functions:
//
//   * RecentChain: each op consumes the immediately preceding value.
//   * LongLivedStar: each op consumes an early function argument plus a recent
//     value.
//   * NestedCaptures: a nested region resolves both local values and parent
//     function-scope captures.
//
// These patterns are intentionally parser-scope focused. A reverse-scanned flat
// scope can look great on producer-consumer chains but become quadratic on
// repeated long-lived references; the star and nested-capture cases make that
// failure mode visible at 10^2, 10^3, and 10^4 SSA definitions.

#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

namespace {

class ParserScopeBenchmark {
 public:
  ParserScopeBenchmark() {
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

  ParserScopeBenchmark(const ParserScopeBenchmark&) = delete;
  ParserScopeBenchmark& operator=(const ParserScopeBenchmark&) = delete;

  ~ParserScopeBenchmark() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Parse(iree_string_view_t source, loom_module_t** out_module) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/1,
    };
    return loom_text_parse(source, IREE_SV("scope_benchmark.loom"), &context_,
                           &block_pool_, &options, out_module);
  }

 private:
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static void AppendFunctionHeader(std::string* source, const char* name) {
  source->append("test.func @");
  source->append(name);
  source->append("(%seed: i32, %cond: i1) -> (i32) {\n");
}

static void AppendFunctionReturn(std::string* source,
                                 const std::string& value_name) {
  source->append("  test.yield %");
  source->append(value_name);
  source->append(" : i32\n");
  source->append("}\n");
}

static std::string BuildRecentChainSource(int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 48);
  AppendFunctionHeader(&source, "recent_chain");
  for (int64_t i = 0; i < definition_count; ++i) {
    source.append("  %v");
    source.append(std::to_string(i));
    source.append(" = test.addi ");
    if (i == 0) {
      source.append("%seed, %seed");
    } else {
      source.append("%v");
      source.append(std::to_string(i - 1));
      source.append(", %v");
      source.append(std::to_string(i - 1));
    }
    source.append(" : i32\n");
  }
  AppendFunctionReturn(&source, "v" + std::to_string(definition_count - 1));
  return source;
}

static std::string BuildLongLivedStarSource(int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 48);
  AppendFunctionHeader(&source, "long_lived_star");
  for (int64_t i = 0; i < definition_count; ++i) {
    source.append("  %v");
    source.append(std::to_string(i));
    source.append(" = test.addi %seed, ");
    if (i == 0) {
      source.append("%seed");
    } else {
      source.append("%v");
      source.append(std::to_string(i - 1));
    }
    source.append(" : i32\n");
  }
  AppendFunctionReturn(&source, "v" + std::to_string(definition_count - 1));
  return source;
}

static void AppendNestedArm(std::string* source, const char* prefix,
                            int64_t definition_count) {
  for (int64_t i = 0; i < definition_count; ++i) {
    source->append("    %");
    source->append(prefix);
    source->append(std::to_string(i));
    source->append(" = test.addi ");
    if (i == 0) {
      source->append("%seed, %outer");
    } else {
      source->append("%");
      source->append(prefix);
      source->append(std::to_string(i - 1));
      source->append(", %outer");
    }
    source->append(" : i32\n");
  }
  source->append("    test.yield %");
  source->append(prefix);
  source->append(std::to_string(definition_count - 1));
  source->append(" : i32\n");
}

static std::string BuildNestedCapturesSource(int64_t definition_count) {
  std::string source;
  source.reserve((size_t)definition_count * 96);
  AppendFunctionHeader(&source, "nested_captures");
  source.append("  %outer = test.addi %seed, %seed : i32\n");
  source.append("  %r = test.branch %cond -> (i32) {\n");
  AppendNestedArm(&source, "then", definition_count);
  source.append("  } else {\n");
  AppendNestedArm(&source, "else", definition_count);
  source.append("  }\n");
  AppendFunctionReturn(&source, "r");
  return source;
}

static void ParseScopeSource(benchmark::State& state, const std::string& source,
                             int64_t definition_count) {
  ParserScopeBenchmark benchmark_fixture;
  for (auto _ : state) {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(benchmark_fixture.Parse(
        iree_make_string_view(source.data(), source.size()), &module));
    if (!module) {
      state.SkipWithError("loom_text_parse produced diagnostics");
      break;
    }
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  state.SetItemsProcessed(state.iterations() * definition_count);
  state.SetBytesProcessed(state.iterations() * (int64_t)source.size());
}

static void BM_ParseScope_RecentChain(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  std::string source = BuildRecentChainSource(definition_count);
  ParseScopeSource(state, source, definition_count);
}
BENCHMARK(BM_ParseScope_RecentChain)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_ParseScope_LongLivedStar(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  std::string source = BuildLongLivedStarSource(definition_count);
  ParseScopeSource(state, source, definition_count);
}
BENCHMARK(BM_ParseScope_LongLivedStar)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_ParseScope_NestedCaptures(benchmark::State& state) {
  const int64_t definition_count = state.range(0);
  std::string source = BuildNestedCapturesSource(definition_count);
  ParseScopeSource(state, source, definition_count * 2 + 1);
}
BENCHMARK(BM_ParseScope_NestedCaptures)->Arg(100)->Arg(1000)->Arg(10000);

}  // namespace
