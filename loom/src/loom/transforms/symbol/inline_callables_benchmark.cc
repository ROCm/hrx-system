// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks complete required-inline callable composition over verified
// modules. Source construction, parsing, verification, per-iteration module
// cloning, and target-version setup happen outside the timed region; each
// measurement contains only the production inline-callables pass over a fresh
// module.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/rewrite/module_projection.h"
#include "loom/target/function_version.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/transforms/symbol/inline_callables.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

enum class CallableMode : uint8_t {
  kAuthoredFunc = 0,
  kTargetLow = 1,
};

enum class CompositionShape : uint8_t {
  // Linear wrappers around one two-block CFG leaf. Output size is constant.
  kLinearCfgLeaf = 0,
  // Every private helper has two blocks. Output grows with chain depth.
  kCfgChain = 1,
  // One two-block helper is called repeatedly by the public entry point.
  kSharedCfgFanout = 2,
  // A shared two-block helper first composes another two-block helper.
  kNestedSharedCfgFanout = 3,
};

enum class DefinitionRole : uint8_t {
  kPrivate = 0,
  kPublic = 1,
};

typedef struct CallableSyntax {
  const char* module_prefix;
  const char* private_definition;
  const char* public_definition;
  const char* body_introducer;
  const char* value_type;
  const char* call_op;
  const char* branch_op;
  const char* return_op;
} CallableSyntax;

static constexpr CallableSyntax kAuthoredFuncSyntax = {
    /*.module_prefix=*/"",
    /*.private_definition=*/"func.def inline ",
    /*.public_definition=*/"func.def public ",
    /*.body_introducer=*/"{\n",
    /*.value_type=*/"i32",
    /*.call_op=*/"func.call",
    /*.branch_op=*/"cfg.br",
    /*.return_op=*/"func.return",
};

static constexpr CallableSyntax kTargetLowSyntax = {
    /*.module_prefix=*/"test.target<low_core> @target\n\n",
    /*.private_definition=*/
    "low.func.def target<test.low.core>(@target) ",
    /*.public_definition=*/
    "low.func.def public target<test.low.core>(@target) ",
    /*.body_introducer=*/"{\n",
    /*.value_type=*/"reg<test.i32>",
    /*.call_op=*/"low.func.call",
    /*.branch_op=*/"low.br",
    /*.return_op=*/"low.return",
};

typedef struct CompositionWorkload {
  CallableMode mode;
  CompositionShape shape;
  uint32_t scale;
} CompositionWorkload;

static const CallableSyntax& SyntaxForMode(CallableMode mode) {
  switch (mode) {
    case CallableMode::kAuthoredFunc:
      return kAuthoredFuncSyntax;
    case CallableMode::kTargetLow:
      return kTargetLowSyntax;
  }
  std::abort();
}

static void AppendDefinitionHeader(std::string& source,
                                   const CallableSyntax& syntax,
                                   DefinitionRole role,
                                   const std::string& name) {
  source.append(role == DefinitionRole::kPrivate ? syntax.private_definition
                                                 : syntax.public_definition);
  source.push_back('@');
  source.append(name);
  source.append("(%value: ");
  source.append(syntax.value_type);
  source.append(") -> (");
  source.append(syntax.value_type);
  source.append(") ");
  source.append(syntax.body_introducer);
}

static void AppendCall(std::string& source, const CallableSyntax& syntax,
                       const std::string& result_name,
                       const std::string& callee_name,
                       const std::string& operand_name) {
  source.append("  ");
  source.append(result_name);
  source.append(" = ");
  source.append(syntax.call_op);
  source.append(" @");
  source.append(callee_name);
  source.push_back('(');
  source.append(operand_name);
  source.append(") : (");
  source.append(syntax.value_type);
  source.append(") -> (");
  source.append(syntax.value_type);
  source.append(")\n");
}

static void AppendReturn(std::string& source, const CallableSyntax& syntax,
                         const std::string& value_name) {
  source.append("  ");
  source.append(syntax.return_op);
  source.push_back(' ');
  source.append(value_name);
  source.append(" : ");
  source.append(syntax.value_type);
  source.push_back('\n');
}

static void AppendCfgIdentityDefinition(std::string& source,
                                        const CallableSyntax& syntax,
                                        DefinitionRole role,
                                        const std::string& name) {
  AppendDefinitionHeader(source, syntax, role, name);
  source.append("  ");
  source.append(syntax.branch_op);
  source.append(" ^exit(%value: ");
  source.append(syntax.value_type);
  source.append(")\n^exit(%forwarded: ");
  source.append(syntax.value_type);
  source.append("):\n");
  AppendReturn(source, syntax, "%forwarded");
  source.append("}\n\n");
}

static void AppendCallingDefinition(std::string& source,
                                    const CallableSyntax& syntax,
                                    DefinitionRole role,
                                    const std::string& name,
                                    const std::string& callee_name,
                                    CompositionShape shape) {
  AppendDefinitionHeader(source, syntax, role, name);
  AppendCall(source, syntax, "%result", callee_name, "%value");
  if (shape == CompositionShape::kCfgChain ||
      shape == CompositionShape::kNestedSharedCfgFanout) {
    source.append("  ");
    source.append(syntax.branch_op);
    source.append(" ^exit(%result: ");
    source.append(syntax.value_type);
    source.append(")\n^exit(%forwarded: ");
    source.append(syntax.value_type);
    source.append("):\n");
    AppendReturn(source, syntax, "%forwarded");
  } else {
    AppendReturn(source, syntax, "%result");
  }
  source.append("}\n\n");
}

static std::string BuildChainSource(const CompositionWorkload& workload) {
  const CallableSyntax& syntax = SyntaxForMode(workload.mode);
  std::string source(syntax.module_prefix);
  source.reserve(source.size() + static_cast<size_t>(workload.scale) * 240u +
                 256u);

  const uint32_t leaf_index = workload.scale - 1;
  AppendCfgIdentityDefinition(source, syntax, DefinitionRole::kPrivate,
                              "helper_" + std::to_string(leaf_index));
  for (uint32_t callee_index = leaf_index; callee_index > 0; --callee_index) {
    const uint32_t caller_index = callee_index - 1;
    AppendCallingDefinition(source, syntax, DefinitionRole::kPrivate,
                            "helper_" + std::to_string(caller_index),
                            "helper_" + std::to_string(callee_index),
                            workload.shape);
  }
  AppendCallingDefinition(source, syntax, DefinitionRole::kPublic, "entry",
                          "helper_0", CompositionShape::kLinearCfgLeaf);
  return source;
}

static std::string BuildSharedFanoutSource(
    const CompositionWorkload& workload) {
  const CallableSyntax& syntax = SyntaxForMode(workload.mode);
  std::string source(syntax.module_prefix);
  source.reserve(source.size() + static_cast<size_t>(workload.scale) * 120u +
                 512u);

  if (workload.shape == CompositionShape::kNestedSharedCfgFanout) {
    AppendCfgIdentityDefinition(source, syntax, DefinitionRole::kPrivate,
                                "leaf");
    AppendCallingDefinition(source, syntax, DefinitionRole::kPrivate, "shared",
                            "leaf", workload.shape);
  } else {
    AppendCfgIdentityDefinition(source, syntax, DefinitionRole::kPrivate,
                                "shared");
  }

  AppendDefinitionHeader(source, syntax, DefinitionRole::kPublic, "entry");
  std::string operand_name = "%value";
  for (uint32_t call_index = 0; call_index < workload.scale; ++call_index) {
    const std::string result_name = "%result_" + std::to_string(call_index);
    AppendCall(source, syntax, result_name, "shared", operand_name);
    operand_name = result_name;
  }
  AppendReturn(source, syntax, operand_name);
  source.append("}\n");
  return source;
}

static std::string BuildCompositionSource(const CompositionWorkload& workload) {
  if (workload.scale == 0) std::abort();
  switch (workload.shape) {
    case CompositionShape::kLinearCfgLeaf:
    case CompositionShape::kCfgChain:
      return BuildChainSource(workload);
    case CompositionShape::kSharedCfgFanout:
    case CompositionShape::kNestedSharedCfgFanout:
      return BuildSharedFanoutSource(workload);
  }
  std::abort();
}

typedef struct ModuleShape {
  iree_host_size_t symbol_count;
  iree_host_size_t block_count;
  iree_host_size_t op_count;
  iree_host_size_t call_count;
} ModuleShape;

static void AccumulateRegionShape(const loom_module_t* module,
                                  const loom_region_t* region,
                                  ModuleShape* shape) {
  if (!region) return;
  shape->block_count += region->block_count;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    for (uint16_t op_index = 0; op_index < block->op_count; ++op_index) {
      const loom_op_t* op = loom_block_const_op(block, op_index);
      ++shape->op_count;
      if (loom_call_like_isa(
              loom_call_like_cast(module, const_cast<loom_op_t*>(op)))) {
        ++shape->call_count;
      }
      loom_region_t* const* regions =
          loom_op_regions(const_cast<loom_op_t*>(op));
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        AccumulateRegionShape(module, regions[region_index], shape);
      }
    }
  }
}

static ModuleShape MeasureModuleShape(const loom_module_t* module) {
  ModuleShape shape = {
      /*.symbol_count=*/module->symbols.count,
  };
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    const loom_op_t* defining_op =
        module->symbols.entries[symbol_id].defining_op;
    if (!defining_op) continue;
    loom_region_t* const* regions =
        loom_op_regions(const_cast<loom_op_t*>(defining_op));
    for (uint8_t region_index = 0; region_index < defining_op->region_count;
         ++region_index) {
      AccumulateRegionShape(module, regions[region_index], &shape);
    }
  }
  return shape;
}

static int64_t ReadPassStatistic(const loom_pass_info_t* pass_info,
                                 const std::vector<uint8_t>& storage,
                                 iree_string_view_t name) {
  const loom_pass_statistic_layout_t* layout = pass_info->statistic_layout;
  if (!layout || storage.size() != layout->storage_size) std::abort();
  for (uint16_t field_index = 0; field_index < layout->field_count;
       ++field_index) {
    const loom_pass_statistic_field_t* field = &layout->fields[field_index];
    if (!iree_string_view_equal(field->name, name)) continue;
    int64_t value = 0;
    std::memcpy(&value, storage.data() + field->offset, sizeof(value));
    return value;
  }
  std::abort();
}

static loom_target_low_call_policy_t RequireInlineLowCalls(
    const loom_resolved_target_t* resolved_target) {
  (void)resolved_target;
  return LOOM_TARGET_LOW_CALL_POLICY_REQUIRE_INLINE;
}

static const loom_target_provider_t* RequireInlineProvider() {
  static const loom_target_provider_t provider = [] {
    loom_target_provider_t value = {};
    value.select_low_call_policy = RequireInlineLowCalls;
    return value;
  }();
  return &provider;
}

static iree_status_t PrepareTargetVersions(
    loom_module_t* module, iree_arena_allocator_t* arena,
    std::vector<loom_target_function_version_t>* versions,
    loom_function_version_owner_t* owner) {
  versions->clear();
  versions->resize(module->symbols.count);
  loom_function_version_owner_initialize(arena, owner);
  IREE_RETURN_IF_ERROR(
      loom_function_version_owner_reserve(owner, module->symbols.count));

  iree_host_size_t version_count = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
    if (!loom_low_func_def_isa(defining_op)) continue;
    loom_target_function_version_t* version = &(*versions)[version_count++];
    version->base.type = &loom_target_function_version_type;
    version->base.function = loom_func_like_cast(module, defining_op);
    version->resolved_target.provider = RequireInlineProvider();
    IREE_RETURN_IF_ERROR(
        loom_function_version_owner_append(owner, &version->base));
  }
  versions->resize(version_count);
  return iree_ok_status();
}

typedef struct RunMetrics {
  ModuleShape output_shape;
  iree_host_size_t module_arena_bytes;
  iree_host_size_t pass_arena_bytes;
  int64_t required_edge_count;
  int64_t cloned_call_count;
  int64_t transferred_call_count;
} RunMetrics;

class InlineCallablesBenchmarkFixture {
 public:
  explicit InlineCallablesBenchmarkFixture(CompositionWorkload workload)
      : workload_(workload) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));
    loom_target_core_test_low_descriptor_registry_initialize(
        &low_descriptor_registry_);

    const std::string source = BuildCompositionSource(workload_);
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/loom_diagnostic_stderr_sink,
        },
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_descriptor_registry_.registry, &parse_options.low_asm_environment);
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("inline_callables_benchmark.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    source_module_.reset(module);
    VerifyModule(source_module_.get());
    input_shape_ = MeasureModuleShape(source_module_.get());
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

  void ValidateOutput(const loom_module_t* module,
                      const RunMetrics& metrics) const {
    VerifyModule(module);
    const iree_host_size_t expected_symbol_count =
        workload_.mode == CallableMode::kTargetLow ? 2 : 1;
    const iree_host_size_t expected_block_count = ExpectedOutputBlockCount();
    const int64_t expected_required_edge_count = ExpectedRequiredEdgeCount();
    const int64_t expected_transferred_call_count =
        ExpectedTransferredCallCount();
    if (metrics.output_shape.symbol_count != expected_symbol_count ||
        metrics.output_shape.block_count != expected_block_count ||
        metrics.output_shape.op_count != expected_block_count ||
        metrics.output_shape.call_count != 0 ||
        metrics.required_edge_count != expected_required_edge_count ||
        metrics.transferred_call_count != expected_transferred_call_count ||
        metrics.cloned_call_count !=
            expected_required_edge_count - expected_transferred_call_count) {
      std::abort();
    }

    const loom_string_id_t entry_name_id =
        loom_module_lookup_string(module, IREE_SV("entry"));
    if (entry_name_id == LOOM_STRING_ID_INVALID) std::abort();
    const loom_symbol_id_t entry_symbol_id =
        loom_module_find_symbol(module, entry_name_id);
    if (entry_symbol_id == LOOM_SYMBOL_ID_INVALID) std::abort();
    const loom_func_like_t entry = loom_func_like_cast(
        module, module->symbols.entries[entry_symbol_id].defining_op);
    if (!loom_func_like_isa(entry)) std::abort();
  }

  const CompositionWorkload& workload() const { return workload_; }
  const ModuleShape& input_shape() const { return input_shape_; }
  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

 private:
  static void VerifyModule(const loom_module_t* module) {
    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &verify_result));
    if (verify_result.error_count != 0) std::abort();
  }

  iree_host_size_t ExpectedOutputBlockCount() const {
    switch (workload_.shape) {
      case CompositionShape::kLinearCfgLeaf:
        return 4;
      case CompositionShape::kCfgChain:
      case CompositionShape::kSharedCfgFanout:
        return 1 + 3 * workload_.scale;
      case CompositionShape::kNestedSharedCfgFanout:
        return 1 + 6 * workload_.scale;
    }
    std::abort();
  }

  int64_t ExpectedRequiredEdgeCount() const {
    switch (workload_.shape) {
      case CompositionShape::kLinearCfgLeaf:
      case CompositionShape::kCfgChain:
      case CompositionShape::kSharedCfgFanout:
        return workload_.scale;
      case CompositionShape::kNestedSharedCfgFanout:
        return workload_.scale + 1;
    }
    std::abort();
  }

  int64_t ExpectedTransferredCallCount() const {
    switch (workload_.shape) {
      case CompositionShape::kLinearCfgLeaf:
      case CompositionShape::kCfgChain:
        return workload_.scale;
      case CompositionShape::kSharedCfgFanout:
        return 1;
      case CompositionShape::kNestedSharedCfgFanout:
        return 2;
    }
    std::abort();
  }

  // Workload topology and policy exercised by this fixture.
  CompositionWorkload workload_;
  // Live source IR measured before per-iteration cloning.
  ModuleShape input_shape_ = {};
  // Shared arena block pool used by source and per-iteration module arenas.
  iree_arena_block_pool_t block_pool_ = {};
  // Dialect context shared by the immutable source and cloned modules.
  loom_context_t context_ = {};
  // Synthetic Low descriptor package used only to parse test register types.
  loom_target_low_descriptor_registry_t low_descriptor_registry_ = {};
  // Verified immutable module cloned outside each timed interval.
  ModulePtr source_module_;
};

static void BM_InlineComposition(benchmark::State& state, CallableMode mode,
                                 CompositionShape shape) {
  const CompositionWorkload workload = {
      /*.mode=*/mode,
      /*.shape=*/shape,
      /*.scale=*/static_cast<uint32_t>(state.range(0)),
  };
  InlineCallablesBenchmarkFixture fixture(workload);
  RunMetrics metrics = {};
  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* module = fixture.CloneModule();
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(fixture.block_pool(), &pass_arena);

    loom_function_version_owner_t version_owner = {};
    std::vector<loom_target_function_version_t> target_versions;
    loom_target_pass_capability_t target_capability = {};
    const loom_pass_environment_capability_t* capabilities[1] = {};
    loom_pass_environment_t pass_environment = {};
    if (mode == CallableMode::kTargetLow) {
      IREE_CHECK_OK(PrepareTargetVersions(module, &pass_arena, &target_versions,
                                          &version_owner));
      target_capability = loom_target_pass_capability_make_mutable(
          /*target_environment=*/nullptr, &version_owner);
      capabilities[0] = &target_capability.base;
      pass_environment = loom_pass_environment_make(
          capabilities, IREE_ARRAYSIZE(capabilities));
    }

    const loom_pass_info_t* pass_info = loom_inline_callables_pass_info();
    std::vector<uint8_t> statistic_storage(
        pass_info->statistic_layout->storage_size, 0);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.module_run = loom_inline_callables_run;
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.statistic_storage = statistic_storage.data();
    pass.environment =
        mode == CallableMode::kTargetLow ? &pass_environment : nullptr;
    const iree_string_view_t pass_options = mode == CallableMode::kTargetLow
                                                ? IREE_SV("policy=target")
                                                : IREE_SV("");
    IREE_CHECK_OK(loom_inline_callables_create(&pass, pass_options));
    state.ResumeTiming();

    IREE_CHECK_OK(loom_inline_callables_run(&pass, module));
    benchmark::DoNotOptimize(module);
    benchmark::ClobberMemory();

    state.PauseTiming();
    metrics = {
        /*.output_shape=*/MeasureModuleShape(module),
        /*.module_arena_bytes=*/module->arena.used_allocation_size,
        /*.pass_arena_bytes=*/pass_arena.used_allocation_size,
        /*.required_edge_count=*/
        ReadPassStatistic(pass_info, statistic_storage,
                          IREE_SV("required-edges")),
        /*.cloned_call_count=*/
        ReadPassStatistic(pass_info, statistic_storage,
                          IREE_SV("calls-cloned")),
        /*.transferred_call_count=*/
        ReadPassStatistic(pass_info, statistic_storage,
                          IREE_SV("calls-transferred")),
    };
    fixture.ValidateOutput(module, metrics);
    iree_arena_deinitialize(&pass_arena);
    loom_module_free(module);
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * metrics.output_shape.op_count);
  state.SetComplexityN(workload.scale);
  state.counters["input_symbols"] =
      static_cast<double>(fixture.input_shape().symbol_count);
  state.counters["input_blocks"] =
      static_cast<double>(fixture.input_shape().block_count);
  state.counters["input_ops"] =
      static_cast<double>(fixture.input_shape().op_count);
  state.counters["required_edges"] =
      static_cast<double>(metrics.required_edge_count);
  state.counters["calls_cloned"] =
      static_cast<double>(metrics.cloned_call_count);
  state.counters["calls_transferred"] =
      static_cast<double>(metrics.transferred_call_count);
  state.counters["output_symbols"] =
      static_cast<double>(metrics.output_shape.symbol_count);
  state.counters["output_blocks"] =
      static_cast<double>(metrics.output_shape.block_count);
  state.counters["output_ops"] =
      static_cast<double>(metrics.output_shape.op_count);
  state.counters["module_arena_bytes"] =
      static_cast<double>(metrics.module_arena_bytes);
  state.counters["pass_arena_bytes"] =
      static_cast<double>(metrics.pass_arena_bytes);
}

static void RegisterCompositionScales(benchmark::Benchmark* benchmark) {
  benchmark->ArgName("scale");
  // The pass is single-threaded. Wall time also avoids coarse process CPU
  // accounting on Windows obscuring the smaller matrix points.
  benchmark->UseRealTime();
  for (int64_t scale : {1, 4, 16, 64, 256, 1024}) {
    benchmark->Arg(scale);
  }
}

BENCHMARK_CAPTURE(BM_InlineComposition, Func_LinearCfgLeaf,
                  CallableMode::kAuthoredFunc, CompositionShape::kLinearCfgLeaf)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, Func_CfgChain,
                  CallableMode::kAuthoredFunc, CompositionShape::kCfgChain)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, Func_SharedCfgFanout,
                  CallableMode::kAuthoredFunc,
                  CompositionShape::kSharedCfgFanout)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, Func_NestedSharedCfgFanout,
                  CallableMode::kAuthoredFunc,
                  CompositionShape::kNestedSharedCfgFanout)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, LowTarget_LinearCfgLeaf,
                  CallableMode::kTargetLow, CompositionShape::kLinearCfgLeaf)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, LowTarget_CfgChain,
                  CallableMode::kTargetLow, CompositionShape::kCfgChain)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, LowTarget_SharedCfgFanout,
                  CallableMode::kTargetLow, CompositionShape::kSharedCfgFanout)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

BENCHMARK_CAPTURE(BM_InlineComposition, LowTarget_NestedSharedCfgFanout,
                  CallableMode::kTargetLow,
                  CompositionShape::kNestedSharedCfgFanout)
    ->Apply(RegisterCompositionScales)
    ->Unit(benchmark::kMicrosecond)
    ->Complexity();

}  // namespace
}  // namespace loom
