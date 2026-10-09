// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end template selection scaling over verified parsed modules. Parsing,
// verification, and provider catalog construction happen once in the fixture;
// each iteration performs the same reference analysis, immutable decision-model
// construction, fact propagation, ranked selection, and liveness work as one
// production query.

#include <algorithm>
#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/pass/builder.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/interpreter.h"
#include "loom/rewrite/module_projection.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/transforms/symbol/template_expansion_pipeline.h"
#include "loom/transforms/symbol/template_selection.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

enum class TemplateSelectionControlFlow {
  kFlat,
  kStructured,
  kCfg,
};

struct TemplateSelectionSourceOptions {
  // Number of providers implementing the demanded family.
  uint32_t provider_count = 1;

  // Number of application sites in the entry function.
  uint32_t site_count = 1;

  // Number of unrelated SSA values in other functions.
  uint32_t unrelated_value_count = 0;

  // True to add scalar constraints to the family and providers.
  bool constrained = false;

  // Control-flow shape containing the application sites.
  TemplateSelectionControlFlow control_flow =
      TemplateSelectionControlFlow::kFlat;

  // True to present providers in ascending rather than descending priority.
  bool reverse_priorities = false;

  // True to add a projectable target condition to every provider.
  bool target_condition = false;

  // True to place an exact call inside each provider whose requirement only
  // becomes provable after substituting the entry's more precise argument.
  bool deferred_exact_call = false;
};

std::string BuildSource(const TemplateSelectionSourceOptions& options) {
  std::string source;
  const bool structured =
      options.control_flow == TemplateSelectionControlFlow::kStructured;
  const bool cfg = options.control_flow == TemplateSelectionControlFlow::kCfg;
  if (options.target_condition) {
    source.append(
        "test.target<low_core> @benchmark_target {subgroup_size = 32}\n\n");
  }
  if (options.provider_count != 0) {
    source.append("template.decl @benchmark.choose(%value: index) -> (index)");
    if (options.constrained) {
      source.append(" where [ge(%value, -1)]");
    }
    source.append("\n\n");
  }
  if (options.deferred_exact_call) {
    source.append(
        "template.decl @leaf(%value: index) -> (index)\n"
        "template.def<@leaf> @exact(%value: index) -> (index) "
        "where [eq(%value, 128)] {\n"
        "  template.return %value : index\n}\n\n");
  }
  for (uint32_t i = 0; i < options.provider_count; ++i) {
    source.append("template.def<@benchmark.choose> ");
    if (options.target_condition) {
      source.append("requires [#target.subgroup.size<32>] ");
    }
    source.append("priority(");
    source.append(std::to_string(
        options.reverse_priorities ? i + 1 : options.provider_count - i));
    source.append(") @provider_");
    source.append(std::to_string(i));
    source.append("(%value: index) -> (index)");
    if (options.constrained) {
      source.append(" where [ge(%value, 0), multiple_of(%value, 16)]");
    }
    source.append(" {\n");
    if (options.deferred_exact_call) {
      source.append(
          "  %result = template.call @exact(%value) : (index) -> (index)\n"
          "  template.return %result : index\n");
    } else {
      source.append("  template.return %value : index\n");
    }
    source.append("}\n\n");
  }
  constexpr uint32_t kValuesPerFunction = 512;
  for (uint32_t begin = 0; begin < options.unrelated_value_count;
       begin += kValuesPerFunction) {
    source.append("func.def @unused_values_");
    source.append(std::to_string(begin / kValuesPerFunction));
    source.append("() {\n");
    const uint32_t end =
        std::min(options.unrelated_value_count, begin + kValuesPerFunction);
    for (uint32_t i = begin; i < end; ++i) {
      source.append("  %unused_value_");
      source.append(std::to_string(i));
      source.append(" = index.constant 0 : index\n");
    }
    source.append("  func.return\n}\n\n");
  }

  source.append("func.def public ");
  if (options.target_condition) {
    source.append("target(@benchmark_target) ");
  }
  source.append("@entry(");
  if (structured) {
    source.append("%condition: i1, ");
  }
  source.append("%value: index) -> (index)");
  if (options.constrained && !cfg) {
    source.append(" where [eq(%value, 128)]");
  }
  source.append(" {\n");
  if (structured) {
    source.append("  %nested_result = scf.if %condition -> (index) {\n");
  } else if (cfg) {
    source.append(
        "  %expected = index.constant 128 : index\n"
        "  %matches = index.cmp eq, %value, %expected : index\n"
        "  cfg.cond_br %matches, ^selected, ^other\n^selected:\n");
  }
  for (uint32_t i = 0; i < options.site_count; ++i) {
    source.append(structured ? "    %result_" : "  %result_");
    source.append(std::to_string(i));
    source.append(
        " = template.apply<@benchmark.choose>(%value) : (index) -> "
        "(index)\n");
  }
  if (structured) {
    source.append("    scf.yield %result_");
    source.append(std::to_string(options.site_count - 1));
    source.append(" : index\n  } else {\n    scf.yield %value : index\n  }\n");
  }
  source.append("  func.return %");
  if (options.site_count == 0) {
    source.append("value");
  } else {
    source.append(structured ? "nested_result" : "result_");
    if (!structured) {
      source.append(std::to_string(options.site_count - 1));
    }
  }
  source.append(" : index\n");
  if (cfg) {
    source.append("^other:\n  func.return %value : index\n");
  }
  source.append("}\n");
  return source;
}

class TemplateSelectionFixture {
 public:
  explicit TemplateSelectionFixture(
      const TemplateSelectionSourceOptions& options) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));

    const std::string source = BuildSource(options);
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t parse_options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("template_selection_benchmark.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    module_.reset(module);
    loom_verify_options_t verify_options = {.max_errors = 20};
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(
        loom_verify_module(module_.get(), &verify_options, &verify_result));
    IREE_ASSERT_EQ(verify_result.error_count, 0u);

    iree_arena_initialize(&block_pool_, &catalog_arena_);
    iree_arena_initialize(&block_pool_, &query_arena_);
    loom_symbol_fact_table_initialize(&symbol_facts_, &catalog_arena_);
    loom_template_provider_catalog_initialize(&catalog_, &catalog_arena_);
    IREE_CHECK_OK(loom_template_provider_catalog_build_local(
        &catalog_, module_.get(), &symbol_facts_));
    IREE_CHECK_OK(loom_symbol_reference_table_build(
        module_.get(), &catalog_arena_, &references_));
  }

  ~TemplateSelectionFixture() {
    loom_pass_program_deinitialize(&expansion_program_);
    expansion_pipeline_.reset();
    iree_arena_deinitialize(&query_arena_);
    iree_arena_deinitialize(&catalog_arena_);
    module_.reset();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint64_t Query() {
    loom_template_selection_query_options_t options = {
        .mode = LOOM_TEMPLATE_SELECTION_MODE_EARLY, .catalog = &catalog_};
    loom_template_selection_query_result_t result = {};
    IREE_CHECK_OK(loom_template_selection_query(
        module_.get(), &options, &block_pool_, &query_arena_, &result));
    IREE_ASSERT_EQ(result.unresolved_site_count, 0u);
    const uint64_t sink = result.required_origins.count + 1;
    iree_arena_reset(&query_arena_);
    return sink;
  }

  uint64_t BuildModels() {
    loom_template_decision_model_catalog_t models = {};
    IREE_CHECK_OK(loom_template_decision_model_catalog_build(
        module_.get(), &symbol_facts_, &references_, &catalog_, &query_arena_,
        &models));
    const uint64_t sink = models.model_count + models.maximum_choice_count + 1;
    iree_arena_reset(&query_arena_);
    return sink;
  }

  void PrepareModels() {
    IREE_CHECK_OK(loom_template_decision_model_catalog_build(
        module_.get(), &symbol_facts_, &references_, &catalog_, &catalog_arena_,
        &models_));
  }

  uint64_t FactRequirements() const {
    IREE_ASSERT_EQ(references_.template_demands.count, 1u);
    const loom_template_demand_t* demand =
        &references_.template_demands.values[0];
    loom_symbol_ref_t family = loom_symbol_ref_null();
    family.module_id = 0;
    family.symbol_id = demand->family_symbol_id;
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(&models_, family);
    IREE_ASSERT_NE(model, nullptr);
    return loom_template_decision_model_application_fact_requirements(model,
                                                                      demand);
  }

  void PrepareExpansion() {
    loom_module_t* pipeline_module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(
        &context_, IREE_SV("expansion"), &block_pool_, nullptr,
        iree_allocator_system(), &pipeline_module));
    expansion_pipeline_.reset(pipeline_module);
    loom_op_t* pipeline_op = nullptr;
    IREE_CHECK_OK(loom_pass_ir_build_pipeline(
        pipeline_module, IREE_SV("expansion"), LOOM_PASS_ANCHOR_MODULE,
        [](loom_builder_t* builder, void*) {
          return loom_template_expansion_pipeline_build(
              builder,
              [](loom_builder_t* builder, void*) {
                loom_op_t* for_op = nullptr;
                return loom_pass_ir_build_for(
                    builder, LOOM_PASS_ANCHOR_FUNC,
                    [](loom_builder_t* builder, void*) {
                      loom_op_t* run_op = nullptr;
                      return loom_pass_ir_build_run(
                          builder, 0, IREE_SV("canonicalize"),
                          loom_named_attr_slice_empty(), &run_op);
                    },
                    nullptr, &for_op);
              },
              nullptr);
        },
        nullptr, &pipeline_op));
    loom_pass_program_compile_options_t options = {
        .registry = loom_pass_builtin_registry()};
    IREE_CHECK_OK(loom_pass_program_compile_pipeline(
        pipeline_module, pipeline_op, &options, &block_pool_,
        &expansion_program_));
  }

  loom_module_t* CloneModule() {
    loom_ir_module_projection_t projection = {};
    loom_module_t* clone = nullptr;
    IREE_CHECK_OK(loom_ir_module_clone(module_.get(), nullptr, &block_pool_,
                                       &query_arena_, iree_allocator_system(),
                                       &projection, &clone));
    iree_arena_reset(&query_arena_);
    return clone;
  }

  void Expand(loom_module_t* module) {
    loom_pass_interpreter_options_t options = {.block_pool = &block_pool_};
    loom_pass_run_result_t result = {};
    IREE_CHECK_OK(loom_pass_interpreter_run_module(&expansion_program_, module,
                                                   &options, &result));
    if (result.error_count != 0) {
      IREE_CHECK_OK(iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                     "template expansion failed"));
    }
  }

 private:
  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  iree_arena_allocator_t catalog_arena_ = {};
  iree_arena_allocator_t query_arena_ = {};
  ModulePtr module_;
  loom_symbol_fact_table_t symbol_facts_ = {};
  loom_template_provider_catalog_t catalog_ = {};
  loom_symbol_reference_table_t references_ = {};
  loom_template_decision_model_catalog_t models_ = {};
  // Source ownership required by the compiled pipeline's diagnostic locations.
  ModulePtr expansion_pipeline_;
  // Immutable production expansion loop reused across fresh input modules.
  loom_pass_program_t expansion_program_ = {};
};

// Measures the complete production expansion loop with a warm block pool.
// Parsing, pipeline compilation, input cloning and output inspection are
// excluded; per-invocation pass state and all expansion iterations are
// included.
void BM_TemplateExpansion(benchmark::State& state) {
  TemplateSelectionSourceOptions options;
  options.provider_count = static_cast<uint32_t>(state.range(0));
  options.site_count = static_cast<uint32_t>(state.range(1));
  options.constrained = state.range(2) != 0;
  options.deferred_exact_call = state.range(3) != 0;
  TemplateSelectionFixture fixture(options);
  fixture.PrepareExpansion();
  iree_host_size_t module_bytes = 0;
  for (auto _ : state) {
    state.PauseTiming();
    loom_module_t* module = fixture.CloneModule();
    state.ResumeTiming();
    fixture.Expand(module);
    benchmark::DoNotOptimize(module);
    state.PauseTiming();
    module_bytes = module->arena.used_allocation_size;
    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      IREE_CHECK_OK(iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                     "expanded module failed verification"));
    }
    loom_module_free(module);
    state.ResumeTiming();
  }
  state.counters["module_arena_bytes"] = static_cast<double>(module_bytes);
  state.SetItemsProcessed(state.iterations() * options.site_count);
}
BENCHMARK(BM_TemplateExpansion)
    ->Args({0, 0, 0, 0})
    ->Args({1, 1, 0, 0})
    ->Args({1, 1, 1, 0})
    ->Args({1, 1, 1, 1})
    ->Args({8, 64, 1, 0})
    ->Args({8, 1024, 0, 0})
    ->Args({8, 1024, 1, 0})
    ->Args({8, 1024, 1, 1})
    ->UseRealTime();

void BM_TemplateSelection(benchmark::State& state) {
  const uint32_t provider_count = static_cast<uint32_t>(state.range(0));
  const uint32_t site_count = static_cast<uint32_t>(state.range(1));
  const bool constrained = state.range(2) != 0;
  TemplateSelectionSourceOptions options;
  options.provider_count = provider_count;
  options.site_count = site_count;
  options.constrained = constrained;
  options.control_flow =
      static_cast<TemplateSelectionControlFlow>(state.range(3));
  TemplateSelectionFixture fixture(options);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Query());
  }
  state.SetItemsProcessed(state.iterations() * site_count);
}
BENCHMARK(BM_TemplateSelection)
    ->Args({1, 1, 0, 0})
    ->Args({1, 1, 1, 0})
    ->Args({8, 1024, 0, 0})
    ->Args({8, 1024, 1, 0})
    ->Args({128, 1024, 0, 0})
    ->Args({128, 1024, 1, 0})
    ->Args({128, 1024, 1, 1})
    ->Args({1, 1, 1, 2})
    ->Args({8, 64, 1, 2})
    ->Args({8, 1024, 1, 2});

void BM_TemplateDecisionModelBuild(benchmark::State& state) {
  const uint32_t provider_count = static_cast<uint32_t>(state.range(0));
  const uint32_t unrelated_value_count = static_cast<uint32_t>(state.range(1));
  const bool reverse_priorities = state.range(2) != 0;
  TemplateSelectionSourceOptions options;
  options.provider_count = provider_count;
  options.unrelated_value_count = unrelated_value_count;
  options.constrained = true;
  options.reverse_priorities = reverse_priorities;
  TemplateSelectionFixture fixture(options);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.BuildModels());
  }
  state.SetItemsProcessed(state.iterations() * provider_count);
}
BENCHMARK(BM_TemplateDecisionModelBuild)
    ->Args({1, 0, 0})
    ->Args({1, 10000, 0})
    ->Args({1, 50000, 0})
    ->Args({128, 0, 0})
    ->Args({128, 0, 1})
    ->Args({4096, 0, 0})
    ->Args({4096, 0, 1});

void BM_TemplateDecisionFactRequirements(benchmark::State& state) {
  constexpr uint32_t kProviderCount = 4096;
  TemplateSelectionSourceOptions options;
  options.provider_count = kProviderCount;
  options.control_flow = TemplateSelectionControlFlow::kStructured;
  options.target_condition = true;
  TemplateSelectionFixture fixture(options);
  fixture.PrepareModels();
  IREE_ASSERT_EQ(fixture.FactRequirements(),
                 LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES |
                     LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_PATH);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.FactRequirements());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TemplateDecisionFactRequirements);

}  // namespace
}  // namespace loom
