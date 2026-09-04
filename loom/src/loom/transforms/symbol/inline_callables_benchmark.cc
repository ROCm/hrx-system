// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks complete required-inline callable composition over verified
// modules. Source construction, parsing, verification, and per-iteration
// module cloning happen outside the timed region; each measurement contains
// only the production inline-callables pass over a fresh module.

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/types.h"
#include "loom/rewrite/module_projection.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/transforms/symbol/inline_callables.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

std::string BuildSingleUseCfgChainSource(uint32_t helper_count) {
  if (helper_count == 0) std::abort();
  std::string source;
  source.reserve(static_cast<size_t>(helper_count) * 220u + 256u);

  const uint32_t leaf_index = helper_count - 1;
  source.append("func.def inline @helper_");
  source.append(std::to_string(leaf_index));
  source.append(
      "(%value: i32) -> (i32) {\n"
      "  cfg.br ^exit(%value: i32)\n"
      "^exit(%forwarded: i32):\n"
      "  func.return %forwarded : i32\n"
      "}\n\n");

  for (uint32_t callee_index = leaf_index; callee_index > 0; --callee_index) {
    const uint32_t caller_index = callee_index - 1;
    source.append("func.def inline @helper_");
    source.append(std::to_string(caller_index));
    source.append(
        "(%value: i32) -> (i32) {\n"
        "  %result = func.call @helper_");
    source.append(std::to_string(callee_index));
    source.append(
        "(%value) : (i32) -> (i32)\n"
        "  func.return %result : i32\n"
        "}\n\n");
  }

  source.append(
      "func.def public @entry(%value: i32) -> (i32) {\n"
      "  %result = func.call @helper_0(%value) : (i32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n");
  return source;
}

class InlineCallablesBenchmarkFixture {
 public:
  explicit InlineCallablesBenchmarkFixture(uint32_t helper_count)
      : expected_output_block_count_(4) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));

    const std::string source = BuildSingleUseCfgChainSource(helper_count);
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t parse_options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("inline_callables_benchmark.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    source_module_.reset(module);

    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(loom_verify_module(source_module_.get(), &verify_options,
                                     &verify_result));
    IREE_ASSERT_EQ(verify_result.error_count, 0u);
  }

  InlineCallablesBenchmarkFixture(const InlineCallablesBenchmarkFixture&) =
      delete;
  InlineCallablesBenchmarkFixture& operator=(
      const InlineCallablesBenchmarkFixture&) = delete;

  ~InlineCallablesBenchmarkFixture() {
    source_module_.reset();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* CloneModule() {
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool_, &scratch_arena);
    loom_ir_module_projection_t projection = {};
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_ir_module_clone(
        source_module_.get(), /*options=*/nullptr, &block_pool_, &scratch_arena,
        iree_allocator_system(), &projection, &module));
    iree_arena_deinitialize(&scratch_arena);
    return module;
  }

  iree_host_size_t ValidateOutput(const loom_module_t* module) const {
    if (module->symbols.count != 1) std::abort();
    const loom_string_id_t entry_name_id =
        loom_module_lookup_string(module, IREE_SV("entry"));
    if (entry_name_id == LOOM_STRING_ID_INVALID) std::abort();
    const loom_symbol_id_t entry_symbol_id =
        loom_module_find_symbol(module, entry_name_id);
    if (entry_symbol_id == LOOM_SYMBOL_ID_INVALID) std::abort();
    const loom_func_like_t entry = loom_func_like_cast(
        module, module->symbols.entries[entry_symbol_id].defining_op);
    if (!loom_func_like_isa(entry)) std::abort();

    const loom_region_t* body = loom_func_like_body(entry);
    if (body == nullptr || body->block_count != expected_output_block_count_) {
      std::abort();
    }
    iree_host_size_t output_op_count = 0;
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      output_op_count += loom_region_const_block(body, block_index)->op_count;
    }
    if (output_op_count != expected_output_block_count_) std::abort();
    return output_op_count;
  }

  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

  uint32_t expected_output_block_count() const {
    return expected_output_block_count_;
  }

 private:
  // Live blocks/ops after all single-use wrappers collapse into the CFG leaf.
  uint32_t expected_output_block_count_;
  // Shared arena block pool used by source and per-iteration module arenas.
  iree_arena_block_pool_t block_pool_ = {};
  // Dialect context shared by the immutable source and cloned modules.
  loom_context_t context_ = {};
  // Verified immutable module cloned outside each timed interval.
  ModulePtr source_module_;
};

void BM_InlineSingleUseCfgChain(benchmark::State& state) {
  const uint32_t helper_count = static_cast<uint32_t>(state.range(0));
  InlineCallablesBenchmarkFixture fixture(helper_count);
  iree_host_size_t output_op_count = 0;
  iree_host_size_t output_used_bytes = 0;
  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* module = fixture.CloneModule();
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(fixture.block_pool(), &pass_arena);
    const loom_pass_info_t* pass_info = loom_inline_callables_pass_info();
    std::vector<uint8_t> statistic_storage(
        pass_info->statistic_layout->storage_size, 0);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.module_run = loom_inline_callables_run;
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.statistic_storage = statistic_storage.data();
    IREE_CHECK_OK(loom_inline_callables_create(&pass, IREE_SV("")));
    state.ResumeTiming();

    IREE_CHECK_OK(loom_inline_callables_run(&pass, module));
    benchmark::DoNotOptimize(module);
    benchmark::ClobberMemory();

    state.PauseTiming();
    output_op_count = fixture.ValidateOutput(module);
    output_used_bytes = module->arena.used_allocation_size;
    iree_arena_deinitialize(&pass_arena);
    loom_module_free(module);
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * output_op_count);
  state.SetComplexityN(helper_count);
  state.counters["helpers"] = static_cast<double>(helper_count);
  state.counters["output_blocks"] =
      static_cast<double>(fixture.expected_output_block_count());
  state.counters["output_ops"] = static_cast<double>(output_op_count);
  state.counters["output_used_bytes"] = static_cast<double>(output_used_bytes);
}

static void RegisterCfgChainScales(benchmark::Benchmark* benchmark) {
  benchmark->ArgName("helpers");
  for (int64_t scale : {1, 4, 16, 64, 256, 1024}) {
    benchmark->Arg(scale);
  }
}

BENCHMARK(BM_InlineSingleUseCfgChain)
    ->Apply(RegisterCfgChainScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

}  // namespace
}  // namespace loom
