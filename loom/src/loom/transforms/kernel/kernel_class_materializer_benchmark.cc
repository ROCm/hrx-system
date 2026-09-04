// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks publication of one independently owned kernel specialization
// class. Parsing, verification, analysis, classification, and launch-site
// collection happen once in the fixture; each iteration clones the source
// product and rewrites the selected applications exactly as the production
// command-product planner will.

#include <cstdint>
#include <string>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/transforms/kernel/kernel_class_classifier.h"
#include "loom/transforms/kernel/kernel_class_materializer.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/fact_table.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

std::string BuildSource(uint32_t decision_count) {
  std::string source;
  source.reserve(400u + decision_count * 560u);
  for (uint32_t i = 0; i < decision_count; ++i) {
    const std::string ordinal = std::to_string(i);
    source.append("template.decl @boundary_");
    source.append(ordinal);
    source.append(".family(%n: index)\n\n");

    source.append("template.def<@boundary_");
    source.append(ordinal);
    source.append(".family> priority(10) @large_");
    source.append(ordinal);
    source.append(
        "(%n: index) where [ge(%n, 128)] {\n"
        "  template.return\n"
        "}\n\n");

    source.append("template.def<@boundary_");
    source.append(ordinal);
    source.append(".family> priority(1) @small_");
    source.append(ordinal);
    source.append(
        "(%n: index) {\n"
        "  template.return\n"
        "}\n\n");
  }

  source.append(
      "kernel.def @classified() {\n"
      "  %one = index.constant 1 : index\n"
      "  kernel.launch.config workgroups(%one, %one, %one) "
      "workgroup_size(%one, %one, %one) : index\n"
      "} launch(%n: index) {\n");
  for (uint32_t i = 0; i < decision_count; ++i) {
    source.append("  template.apply<@boundary_");
    source.append(std::to_string(i));
    source.append(".family>(%n) : (index) -> ()\n");
  }
  source.append(
      "  kernel.return\n"
      "}\n");
  return source;
}

class KernelClassMaterializerBenchmarkFixture {
 public:
  explicit KernelClassMaterializerBenchmarkFixture(uint32_t decision_count)
      : decision_count_(decision_count) {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_CHECK_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);

    const std::string source = BuildSource(decision_count);
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("kernel_class_materializer_benchmark.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    module_.reset(module);

    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t verify_result = {};
    IREE_CHECK_OK(
        loom_verify_module(module_.get(), &verify_options, &verify_result));
    IREE_ASSERT_EQ(verify_result.error_count, 0u);

    IREE_CHECK_OK(loom_symbol_reference_table_build(
        module_.get(), &analysis_arena_, &references_));
    loom_symbol_fact_table_initialize(&symbol_facts_, &analysis_arena_);
    loom_template_provider_catalog_initialize(&providers_, &analysis_arena_);
    IREE_CHECK_OK(loom_template_provider_catalog_build_local(
        &providers_, module_.get(), &symbol_facts_));
    IREE_CHECK_OK(loom_template_decision_model_catalog_build(
        module_.get(), &symbol_facts_, &references_, &providers_,
        &analysis_arena_, &decision_models_));

    const loom_string_id_t kernel_name_id =
        loom_module_lookup_string(module_.get(), IREE_SV("classified"));
    IREE_ASSERT_NE(kernel_name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t kernel_symbol_id =
        loom_module_find_symbol(module_.get(), kernel_name_id);
    IREE_ASSERT_NE(kernel_symbol_id, LOOM_SYMBOL_ID_INVALID);
    const loom_func_like_t kernel = loom_func_like_cast(
        module_.get(), module_->symbols.entries[kernel_symbol_id].defining_op);
    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &kernel_facts_, &analysis_arena_, module_->values.count));
    IREE_CHECK_OK(
        loom_value_fact_table_compute(&kernel_facts_, module_.get(), kernel));
    loom_symbolic_expr_context_initialize(
        module_.get(), /*value_domain=*/nullptr, &kernel_facts_,
        &analysis_arena_, &expression_context_);
    const loom_template_applicability_target_t kernel_target = {};
    IREE_CHECK_OK(loom_kernel_class_classifier_build(
        module_.get(), kernel_symbol_id, &references_, &decision_models_,
        &kernel_facts_, &expression_context_, &kernel_target, &analysis_arena_,
        &classifier_));
    IREE_ASSERT_EQ(classifier_.decision_count, decision_count);

    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &site_facts_, &analysis_arena_, std::size(site_argument_values_)));
    site_facts_.count = std::size(site_argument_values_);
    for (iree_host_size_t i = 0; i < std::size(site_argument_values_); ++i) {
      site_argument_values_[i] = static_cast<loom_value_id_t>(i);
    }
    site_facts_.entries[0] = loom_value_facts_exact_i64(64);
    site_facts_.entries[1] = loom_value_facts_exact_i64(256);
    sites_[0] = {
        /*.facts=*/&site_facts_,
        /*.argument_values=*/&site_argument_values_[0],
    };
    sites_[1] = {
        /*.facts=*/&site_facts_,
        /*.argument_values=*/&site_argument_values_[1],
    };
    const loom_kernel_class_collection_options_t collection_options =
        loom_kernel_class_collection_options_default();
    IREE_CHECK_OK(loom_kernel_class_classifier_collect(
        &classifier_, sites_, std::size(sites_), &collection_options,
        &analysis_arena_, &collection_));
    IREE_ASSERT_EQ(collection_.accepted_decision_count, decision_count);
    IREE_ASSERT_EQ(collection_.class_count, decision_count == 0 ? 1u : 2u);

    class_ordinal_ = FindSpecializedClass();
    loom_module_t* validation_module = Materialize();
    IREE_ASSERT_NE(validation_module, nullptr);
    loom_module_free(validation_module);
  }

  KernelClassMaterializerBenchmarkFixture(
      const KernelClassMaterializerBenchmarkFixture&) = delete;
  KernelClassMaterializerBenchmarkFixture& operator=(
      const KernelClassMaterializerBenchmarkFixture&) = delete;

  ~KernelClassMaterializerBenchmarkFixture() {
    module_.reset();
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Materialize() {
    loom_kernel_class_product_t product = {};
    IREE_CHECK_OK(loom_kernel_class_materialize(
        &classifier_, &collection_, class_ordinal_, &block_pool_,
        iree_allocator_system(), &product));
    loom_module_t* materialized_module = product.module;
    product.module = nullptr;
    loom_kernel_class_product_deinitialize(&product);
    return materialized_module;
  }

  uint32_t decision_count() const { return decision_count_; }

 private:
  loom_decision_class_ordinal_t FindSpecializedClass() const {
    if (decision_count_ == 0) return 0;
    for (loom_decision_class_ordinal_t class_ordinal = 0;
         class_ordinal < collection_.class_count; ++class_ordinal) {
      const loom_kernel_class_trace_t* trace =
          &collection_.traces[collection_.classes[class_ordinal].trace_id];
      const loom_kernel_class_decision_t* decision =
          &classifier_.decisions[trace->decision_ordinal];
      if (decision->generic_result.action_ordinal != trace->action_ordinal) {
        return class_ordinal;
      }
    }
    IREE_ASSERT_UNREACHABLE("specialized kernel class not found");
    return 0;
  }

  uint32_t decision_count_;
  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  iree_arena_allocator_t analysis_arena_ = {};
  ModulePtr module_;
  loom_symbol_reference_table_t references_ = {};
  loom_symbol_fact_table_t symbol_facts_ = {};
  loom_template_provider_catalog_t providers_ = {};
  loom_template_decision_model_catalog_t decision_models_ = {};
  loom_value_fact_table_t kernel_facts_ = {};
  loom_symbolic_expr_context_t expression_context_ = {};
  loom_kernel_class_classifier_t classifier_ = {};
  loom_value_id_t site_argument_values_[2] = {};
  loom_value_fact_table_t site_facts_ = {};
  loom_kernel_class_site_t sites_[2] = {};
  loom_kernel_class_collection_t collection_ = {};
  loom_decision_class_ordinal_t class_ordinal_ = 0;
};

void BM_MaterializeSpecializedKernelClass(benchmark::State& state) {
  KernelClassMaterializerBenchmarkFixture fixture(
      static_cast<uint32_t>(state.range(0)));
  for (auto _ : state) {
    loom_module_t* module = fixture.Materialize();
    benchmark::DoNotOptimize(module);
    loom_module_free(module);
  }
  state.SetItemsProcessed(state.iterations() * fixture.decision_count());
}
BENCHMARK(BM_MaterializeSpecializedKernelClass)
    ->Arg(0)
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(128);

}  // namespace
}  // namespace loom
