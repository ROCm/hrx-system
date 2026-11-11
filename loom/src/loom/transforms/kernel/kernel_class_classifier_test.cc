// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_classifier.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class KernelClassClassifierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerify(iree_string_view_t source,
                           loom_symbol_reference_table_t* out_references) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse_with_symbol_references(
        source, IREE_SV("kernel_class_classifier_test.loom"), &context_,
        &block_pool_, &parse_options, &analysis_arena_, out_references,
        &module));
    if (module == nullptr) {
      return ModulePtr();
    }
    ModulePtr module_ptr(module);
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t verify_result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
    return module_ptr;
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    if (name_id == LOOM_STRING_ID_INVALID) {
      return LOOM_SYMBOL_ID_INVALID;
    }
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  ModulePtr BuildClassifier(iree_string_view_t source,
                            iree_string_view_t kernel_name,
                            loom_kernel_class_classifier_t* out_classifier) {
    loom_symbol_reference_table_t references = {};
    ModulePtr module = ParseAndVerify(source, &references);
    if (module == nullptr) {
      return ModulePtr();
    }

    loom_symbol_fact_table_t symbol_facts = {};
    loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
    loom_template_provider_catalog_t providers = {};
    loom_template_provider_catalog_initialize(&providers, &analysis_arena_);
    IREE_EXPECT_OK(loom_template_provider_catalog_build_local(
        &providers, module.get(), &symbol_facts));
    loom_template_decision_model_catalog_t decision_models = {};
    IREE_EXPECT_OK(loom_template_decision_model_catalog_build(
        module.get(), &symbol_facts, &references, &providers, &analysis_arena_,
        &decision_models));

    const loom_symbol_id_t kernel_symbol_id =
        FindSymbol(module.get(), kernel_name);
    if (kernel_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return ModulePtr();
    }
    const loom_func_like_t kernel = loom_func_like_cast(
        module.get(), module->symbols.entries[kernel_symbol_id].defining_op);
    loom_value_fact_table_t kernel_facts = {};
    IREE_EXPECT_OK(loom_value_fact_table_initialize(
        &kernel_facts, &analysis_arena_, module->values.count));
    IREE_EXPECT_OK(
        loom_value_fact_table_compute(&kernel_facts, module.get(), kernel));
    loom_symbolic_expr_context_t expression_context = {};
    loom_symbolic_expr_context_initialize(
        module.get(), /*value_domain=*/nullptr, &kernel_facts, &analysis_arena_,
        &expression_context);
    const loom_template_applicability_target_t kernel_target = {};
    IREE_EXPECT_OK(loom_kernel_class_classifier_build(
        module.get(), kernel_symbol_id, &references, &decision_models,
        &kernel_facts, &expression_context, &kernel_target, &analysis_arena_,
        out_classifier));
    // Construction scratch may retire before the classifier is queried.
    iree_arena_block_pool_trim(&block_pool_);
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
};

std::string BuildBoundarySource(iree_host_size_t site_count) {
  std::string source = R"(
template.decl @boundary.family(%effective_n: index, %irrelevant: index) -> (index)

template.def<@boundary.family> priority(10) @large(%effective_n: index, %irrelevant: index) -> (index) where [ge(%effective_n, 128)] {
  %two = index.constant 2 : index
  template.return %two : index
}

template.def<@boundary.family> priority(1) @small(%effective_n: index, %irrelevant: index) -> (index) {
  %one = index.constant 1 : index
  template.return %one : index
}

kernel.def @classified() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%n: index, %padding: index, %irrelevant: index) {
  %effective_n = index.add %n, %padding : index
  %selected = template.apply<@boundary.family>(%effective_n, %irrelevant) : (index, index) -> (index)
  kernel.return
}

func.def public @program() {
)";
  source.reserve(320 * 1024);
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    const int64_t n = static_cast<int64_t>(i);
    const int64_t padding = static_cast<int64_t>((i * 7) % 5);
    const int64_t irrelevant = static_cast<int64_t>(1000003 * i + 17);
    source.append("  %n_");
    source.append(std::to_string(i));
    source.append(" = index.constant ");
    source.append(std::to_string(n));
    source.append(" : index\n  %padding_");
    source.append(std::to_string(i));
    source.append(" = index.constant ");
    source.append(std::to_string(padding));
    source.append(" : index\n  %irrelevant_");
    source.append(std::to_string(i));
    source.append(" = index.constant ");
    source.append(std::to_string(irrelevant));
    source.append(" : index\n  kernel.launch @classified(%n_");
    source.append(std::to_string(i));
    source.append(", %padding_");
    source.append(std::to_string(i));
    source.append(", %irrelevant_");
    source.append(std::to_string(i));
    source.append(") : (index, index, index)\n");
  }
  source.append("  func.return\n}\n");
  return source;
}

// A compact compiler-owned fixture for collection-policy behavior. The parsed
// module test below owns source projection and model-construction coverage;
// this fixture isolates finite-quotient and conservative-widening contracts.
class ManualBinaryClassifier {
 public:
  explicit ManualBinaryClassifier(bool has_generic_residual) {
    predicate_count_ = has_generic_residual ? 1 : 2;
    predicates_[0] = {
        .kind = LOOM_PREDICATE_GE,
        .operand_count = 2,
        .reserved = {},
        .operands =
            {
                loom_decision_program_argument_ref(0),
                loom_decision_program_constant_ref(0),
            },
    };
    predicates_[1] = {
        .kind = LOOM_PREDICATE_LT,
        .operand_count = 2,
        .reserved = {},
        .operands =
            {
                loom_decision_program_argument_ref(0),
                loom_decision_program_constant_ref(0),
            },
    };
    choices_[0] = {
        .conjunction =
            {
                .first_predicate = 0,
                .first_feature = 0,
                .predicate_count = 1,
                .feature_count = 0,
            },
        .action_ordinal = 0,
    };
    choices_[1] = {
        .conjunction =
            {
                .first_predicate = 1,
                .first_feature = 0,
                .predicate_count =
                    static_cast<uint16_t>(has_generic_residual ? 0 : 1),
                .feature_count = 0,
            },
        .action_ordinal = 1,
    };
    for (uint32_t i = 0; i < 2; ++i) {
      actions_[0][i].choice_ordinal = i;
      actions_[1][i].choice_ordinal = i;
    }
    if (has_generic_residual) {
      priority_groups_[0].choice_count = 1;
      priority_groups_[1].choice_count = 1;
      priority_group_count_ = 2;
    } else {
      priority_groups_[0].choice_count = 2;
      priority_group_count_ = 1;
    }
    model_.program = {
        .predicates = predicates_,
        .choices = choices_,
        .priority_groups = priority_groups_,
        .constants = constants_,
        .hard_requirements = {},
        .predicate_count = predicate_count_,
        .feature_count = 0,
        .constant_count = 1,
        .choice_count = 2,
        .priority_group_count = priority_group_count_,
    };

    for (uint16_t i = 0; i < 2; ++i) {
      projection_terms_[i] = {
          .coefficient = 1,
          .argument_ordinal = i,
          .reserved = {},
      };
      projections_[i] = {
          .source_value_id = i,
          .kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE,
          .reserved = {},
          .constant = 0,
          .terms = &projection_terms_[i],
          .static_facts = {},
          .term_count = 1,
          .trailing_reserved = {},
      };
      decisions_[i] = {
          .application = nullptr,
          .model = &model_,
          .argument_values = decision_argument_values_[i],
          .result_values = nullptr,
          .projection_ordinals = &projection_ordinals_[i],
          .feature_outcomes = nullptr,
          .actions = actions_[i],
          .generic_result =
              {
                  .kind = static_cast<loom_decision_program_result_kind_t>(
                      has_generic_residual
                          ? LOOM_DECISION_PROGRAM_RESULT_SELECTED
                          : LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED),
                  .reserved = {},
                  .action_ordinal = has_generic_residual
                                        ? 1u
                                        : LOOM_DECISION_PROGRAM_ACTION_INVALID,
                  .unresolved_action_ordinal =
                      LOOM_DECISION_PROGRAM_ACTION_INVALID,
                  .unresolved_constraint =
                      LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
              },
          .argument_count = 2,
          .result_count = 0,
          .projection_count = 1,
          .unavailable_reason = LOOM_KERNEL_CLASS_DECISION_AVAILABLE,
          .hard_requirement_flags = 0,
          .reserved = {},
      };
    }
    classifier_ = {
        .module = nullptr,
        .kernel_symbol_id = 0,
        .projections = projections_,
        .decisions = decisions_,
        .projection_count = 2,
        .decision_count = 2,
        .maximum_provider_count = 2,
        .kernel_argument_count = 2,
        .specialization_count = 0,
    };
  }

  iree_status_t InitializeSites(iree_arena_allocator_t* arena) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_initialize(&fact_table_, arena, 8));
    for (iree_host_size_t i = 0; i < 4; ++i) {
      const loom_value_id_t first_value = static_cast<loom_value_id_t>(i * 2);
      argument_values_[i][0] = first_value;
      argument_values_[i][1] = first_value + 1;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          &fact_table_, first_value,
          loom_value_facts_exact_i64(static_cast<int64_t>(i / 2))));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          &fact_table_, first_value + 1,
          loom_value_facts_exact_i64(static_cast<int64_t>(i % 2))));
      sites_[i] = {
          .facts = &fact_table_,
          .argument_values = argument_values_[i],
      };
    }
    return iree_ok_status();
  }

  loom_kernel_class_classifier_t* classifier() { return &classifier_; }
  loom_kernel_class_decision_t* decisions() { return decisions_; }
  loom_kernel_class_action_t* actions(uint32_t decision_ordinal) {
    return actions_[decision_ordinal];
  }
  const loom_kernel_class_site_t* sites() const { return sites_; }

  iree_status_t SetSiteArgumentFacts(iree_host_size_t site, uint16_t argument,
                                     loom_value_facts_t facts) {
    return loom_value_fact_table_define(
        &fact_table_, argument_values_[site][argument], facts);
  }

 private:
  loom_decision_program_predicate_t predicates_[2] = {};
  loom_decision_program_choice_t choices_[2] = {};
  loom_decision_program_priority_group_t priority_groups_[2] = {};
  int64_t constants_[1] = {1};
  uint32_t predicate_count_ = 0;
  uint32_t priority_group_count_ = 0;
  loom_template_decision_model_t model_ = {};

  loom_kernel_class_projection_term_t projection_terms_[2] = {};
  loom_kernel_class_projection_t projections_[2] = {};
  // Both families receive both source arguments; only their first formal
  // participates in header selection. The other can supply body structure.
  loom_value_id_t decision_argument_values_[2][2] = {{0, 1}, {1, 0}};
  uint32_t projection_ordinals_[2] = {0, 1};
  loom_kernel_class_action_t actions_[2][2] = {};
  loom_kernel_class_decision_t decisions_[2] = {};
  loom_kernel_class_classifier_t classifier_ = {};

  loom_value_id_t argument_values_[4][2] = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_kernel_class_site_t sites_[4] = {};
};

TEST_F(KernelClassClassifierTest,
       AffineBoundaryProducesTwoClassesFromDistinctObservations) {
  constexpr iree_host_size_t kSiteCount = 1000;
  const std::string source = BuildBoundarySource(kSiteCount);
  loom_symbol_reference_table_t references = {};
  ModulePtr module = ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &references);
  ASSERT_NE(module, nullptr);

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena_);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &providers, module.get(), &symbol_facts));
  loom_template_decision_model_catalog_t decision_models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      module.get(), &symbol_facts, &references, &providers, &analysis_arena_,
      &decision_models));

  const loom_symbol_id_t kernel_symbol_id =
      FindSymbol(module.get(), IREE_SV("classified"));
  ASSERT_NE(kernel_symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* kernel_op = module->symbols.entries[kernel_symbol_id].defining_op;
  const loom_func_like_t kernel = loom_func_like_cast(module.get(), kernel_op);
  loom_value_fact_table_t kernel_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &kernel_facts, &analysis_arena_, module->values.count));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&kernel_facts, module.get(), kernel));
  loom_symbolic_expr_context_t expression_context = {};
  loom_symbolic_expr_context_initialize(module.get(), /*value_domain=*/nullptr,
                                        &kernel_facts, &analysis_arena_,
                                        &expression_context);
  const loom_template_applicability_target_t kernel_target = {};
  loom_kernel_class_classifier_t classifier = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_build(
      module.get(), kernel_symbol_id, &references, &decision_models,
      &kernel_facts, &expression_context, &kernel_target, &analysis_arena_,
      &classifier));
  ASSERT_EQ(classifier.decision_count, 1u);
  ASSERT_EQ(classifier.projection_count, 1u);
  ASSERT_EQ(classifier.projections[0].kind,
            LOOM_KERNEL_CLASS_PROJECTION_AFFINE);
  ASSERT_EQ(classifier.projections[0].term_count, 2u);

  const loom_symbol_id_t program_symbol_id =
      FindSymbol(module.get(), IREE_SV("program"));
  ASSERT_NE(program_symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* program_op =
      module->symbols.entries[program_symbol_id].defining_op;
  const loom_func_like_t program =
      loom_func_like_cast(module.get(), program_op);
  loom_value_fact_table_t program_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &program_facts, &analysis_arena_, module->values.count));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&program_facts, module.get(), program));

  std::vector<loom_kernel_class_site_t> sites;
  sites.reserve(kSiteCount);
  std::vector<const loom_op_t*> launches;
  launches.reserve(kSiteCount);
  loom_region_t* program_body = loom_func_like_body(program);
  ASSERT_NE(program_body, nullptr);
  ASSERT_EQ(program_body->block_count, 1u);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(loom_region_entry_block(program_body), op) {
    if (!loom_kernel_launch_isa(op)) {
      continue;
    }
    const loom_value_slice_t arguments = loom_kernel_launch_arguments(op);
    ASSERT_EQ(arguments.count, 3u);
    sites.push_back({
        .facts = &program_facts,
        .argument_values = arguments.values,
    });
    launches.push_back(op);
  }
  ASSERT_EQ(sites.size(), kSiteCount);

  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      &classifier, nullptr, sites.data(), sites.size(), &options,
      &analysis_arena_, &collection));
  ASSERT_EQ(collection.class_count, 2u);
  ASSERT_EQ(collection.accepted_decision_count, 1u);
  ASSERT_EQ(collection.skipped_decision_count, 0u);
  ASSERT_EQ(collection.trace_count, 2u);

  const loom_template_decision_model_t* model = classifier.decisions[0].model;
  uint32_t specialized_provider = UINT32_MAX;
  uint32_t generic_provider = UINT32_MAX;
  for (uint32_t i = 0; i < model->providers.count; ++i) {
    if (model->providers.providers[i].predicate_count == 0) {
      generic_provider = i;
    } else {
      specialized_provider = i;
    }
  }
  ASSERT_NE(specialized_provider, UINT32_MAX);
  ASSERT_NE(generic_provider, UINT32_MAX);
  ASSERT_EQ(classifier.decisions[0].generic_result.kind,
            LOOM_DECISION_PROGRAM_RESULT_SELECTED);
  EXPECT_EQ(classifier.decisions[0].generic_result.action_ordinal,
            generic_provider);

  iree_host_size_t generic_count = 0;
  iree_host_size_t specialized_count = 0;
  for (iree_host_size_t i = 0; i < kSiteCount; ++i) {
    const loom_value_slice_t arguments =
        loom_kernel_launch_arguments(launches[i]);
    int64_t n = 0;
    int64_t padding = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(&program_facts, arguments.values[0]), &n));
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(&program_facts, arguments.values[1]),
        &padding));
    const uint32_t expected_provider =
        n + padding >= 128 ? specialized_provider : generic_provider;
    const loom_decision_class_ordinal_t class_ordinal =
        collection.site_classes[i];
    ASSERT_LT(class_ordinal, collection.class_count);
    const loom_kernel_class_trace_id_t trace_id =
        collection.classes[class_ordinal].trace_id;
    ASSERT_LT(trace_id, collection.trace_count);
    EXPECT_EQ(collection.traces[trace_id].decision_ordinal, 0u);
    EXPECT_EQ(collection.traces[trace_id].action_ordinal, expected_provider);
    generic_count += expected_provider == generic_provider;
    specialized_count += expected_provider == specialized_provider;
  }
  EXPECT_EQ(generic_count, 126u);
  EXPECT_EQ(specialized_count, 874u);
  EXPECT_EQ(collection.classes[collection.site_classes[0]].member_count, 126u);
  EXPECT_EQ(
      collection.classes[collection.site_classes[kSiteCount - 1]].member_count,
      874u);
}

TEST_F(KernelClassClassifierTest, CorrelatedDecisionsRetainObservedQuotient) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.decisions()[1].argument_values =
      fixture.decisions()[0].argument_values;
  fixture.decisions()[1].projection_ordinals =
      fixture.decisions()[0].projection_ordinals;

  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
      &options, &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.accepted_decision_count, 2u);
  EXPECT_EQ(collection.skipped_decision_count, 0u);
  EXPECT_EQ(collection.trace_count, 4u);
}

TEST_F(KernelClassClassifierTest,
       ManyCorrelatedDecisionsRetainLinearTraceStorage) {
  constexpr uint32_t kDecisionCount = 128;
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  std::vector<loom_kernel_class_decision_t> decisions(kDecisionCount,
                                                      fixture.decisions()[0]);
  fixture.classifier()->decisions = decisions.data();
  fixture.classifier()->decision_count = decisions.size();

  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
      &options, &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.accepted_decision_count, kDecisionCount);
  EXPECT_EQ(collection.skipped_decision_count, 0u);
  EXPECT_EQ(collection.trace_count, 2u * kDecisionCount);
}

TEST_F(KernelClassClassifierTest,
       RepeatedShapesPreserveBindingsAndShareDecisionTraces) {
  constexpr uint32_t kDecisionCount = 32;
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = &width, .count = 1},
      {.projections = &width, .count = 1},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[0];
  std::vector<loom_kernel_class_decision_t> decisions(kDecisionCount,
                                                      fixture.decisions()[0]);
  fixture.classifier()->decisions = decisions.data();
  fixture.classifier()->decision_count = decisions.size();

  for (bool source_shape : {false, true}) {
    fixture.classifier()->shape =
        source_shape ? shapes[0] : loom_kernel_class_shape_t{};
    for (uint16_t site_count : {2, 4}) {
      for (uint16_t class_limit : {site_count, uint16_t{128}}) {
        auto options = loom_kernel_class_collection_options_default();
        options.class_limit = class_limit;
        loom_kernel_class_collection_t collection = {};
        IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
            fixture.classifier(), nullptr, fixture.sites(), site_count,
            &options, &analysis_arena_, &collection));
        ASSERT_EQ(collection.class_count, site_count);
        EXPECT_EQ(collection.accepted_decision_count, kDecisionCount);
        EXPECT_EQ(collection.trace_count, site_count / 2 * kDecisionCount);
        for (uint16_t site = 0; site < site_count; ++site) {
          const auto& entry = collection.classes[collection.site_classes[site]];
          ASSERT_EQ(entry.shape_value_count, 1u);
          EXPECT_EQ(entry.shape_values[0].projection, width);
          EXPECT_TRUE(loom_value_facts_is_exact(entry.shape_values[0].facts));
          EXPECT_EQ(entry.shape_values[0].facts.range_lo, site % 2);
          const auto& peer =
              collection.classes[collection.site_classes[site ^ 1]];
          EXPECT_EQ(entry.trace_id, peer.trace_id);
          auto trace_id = entry.trace_id;
          for (uint32_t ordinal = kDecisionCount; ordinal != 0; --ordinal) {
            ASSERT_NE(trace_id, LOOM_KERNEL_CLASS_TRACE_ID_INVALID);
            const auto& trace = collection.traces[trace_id];
            EXPECT_EQ(trace.decision_ordinal, ordinal - 1);
            EXPECT_EQ(trace.action_ordinal, site < 2 ? 1u : 0u);
            trace_id = trace.parent_trace_id;
          }
          EXPECT_EQ(trace_id, LOOM_KERNEL_CLASS_TRACE_ID_INVALID);
        }
      }
    }
  }
}

TEST_F(KernelClassClassifierTest,
       ClassLimitRetainsGenericResidualWithoutAliasingClasses) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  options.class_limit = 2;
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
      &options, &analysis_arena_, &collection));

  ASSERT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.accepted_decision_count, 1u);
  EXPECT_EQ(collection.skipped_decision_count, 1u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
  EXPECT_EQ(collection.decision_results[1].state,
            LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT);
  EXPECT_EQ(collection.site_classes[0], collection.site_classes[1]);
  EXPECT_EQ(collection.site_classes[2], collection.site_classes[3]);
  EXPECT_NE(collection.site_classes[0], collection.site_classes[2]);
}

TEST_F(KernelClassClassifierTest,
       ClassLimitFailsWhenDecisionHasNoGenericResidual) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/false);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  options.class_limit = 2;
  loom_kernel_class_collection_t collection = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_kernel_class_classifier_collect(
          fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
          &options, &analysis_arena_, &collection));
}

// The first provider reads argument 0 in its header and argument 1 only as
// body structure. These shapes are compiler-owned requirements, independent of
// provider applicability.
TEST_F(KernelClassClassifierTest, StructuralCostIsIncludedBeforeAdmission) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = &width, .count = 1},
      {},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[0];
  fixture.decisions()[0].can_reduce_structure = true;
  for (uint16_t limit : {1, 2, 3, 4}) {
    SCOPED_TRACE(limit);
    auto options = loom_kernel_class_collection_options_default();
    options.class_limit = limit;
    loom_kernel_class_collection_t collection = {};
    if (limit == 1) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                            loom_kernel_class_classifier_collect(
                                fixture.classifier(), nullptr, fixture.sites(),
                                4, &options, &analysis_arena_, &collection));
      continue;
    }
    IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
        fixture.classifier(), nullptr, fixture.sites(), 4, &options,
        &analysis_arena_, &collection));
    EXPECT_EQ(collection.class_count, limit < 3 ? 2 : 3);
    EXPECT_EQ(collection.accepted_decision_count, limit < 3 ? 0 : 1);
    if (limit >= 3) {
      EXPECT_EQ(collection.site_classes[0], collection.site_classes[1]);
      EXPECT_NE(collection.site_classes[2], collection.site_classes[3]);
      for (uint32_t site : {2, 3}) {
        const auto& entry = collection.classes[collection.site_classes[site]];
        ASSERT_EQ(entry.shape_value_count, 1u);
        EXPECT_EQ(entry.shape_values[0].projection, width);
        EXPECT_TRUE(loom_value_facts_is_exact(entry.shape_values[0].facts));
        EXPECT_EQ(entry.shape_values[0].facts.range_lo, site % 2);
      }
    }
  }
}

TEST_F(KernelClassClassifierTest, SelectedProvidersCanRemoveGenericStructure) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {},
      {.projections = &width, .count = 1},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[1];
  fixture.decisions()[0].can_reduce_structure = true;
  fixture.decisions()[1].action_shapes = shapes;
  fixture.decisions()[1].residual_shape = shapes[1];
  fixture.decisions()[1].can_reduce_structure = true;
  fixture.decisions()[1].argument_values =
      fixture.decisions()[0].argument_values;
  fixture.decisions()[1].projection_ordinals =
      fixture.decisions()[0].projection_ordinals;
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 1;
  for (uint32_t decision_count : {1, 2}) {
    SCOPED_TRACE(decision_count);
    fixture.classifier()->decision_count = decision_count;
    loom_kernel_class_collection_t collection = {};
    IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
        fixture.classifier(), nullptr, fixture.sites() + 2, 2, &options,
        &analysis_arena_, &collection));
    ASSERT_EQ(collection.class_count, 1u);
    EXPECT_EQ(collection.accepted_decision_count, decision_count);
    EXPECT_EQ(collection.classes[0].shape_value_count, 0u);
    EXPECT_EQ(collection.classes[0].member_count, 2u);
  }
}

TEST_F(KernelClassClassifierTest, RequiredDecisionsReserveBeforeOptionalOnes) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  ManualBinaryClassifier required(/*has_generic_residual=*/false);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.decisions()[1] = required.decisions()[1];
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 2;
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), 4, &options,
      &analysis_arena_, &collection));
  ASSERT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT);
  EXPECT_EQ(collection.decision_results[1].state,
            LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
  EXPECT_EQ(collection.site_classes[0], collection.site_classes[2]);
  EXPECT_EQ(collection.site_classes[1], collection.site_classes[3]);
  EXPECT_NE(collection.site_classes[0], collection.site_classes[1]);
}

TEST_F(KernelClassClassifierTest,
       LaterGenericStructureReservesRemainingBudget) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t first_shapes[2] = {};
  const loom_kernel_class_shape_t second_shapes[] = {
      {.projections = &width, .count = 1},
      {},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.decisions()[0].action_shapes = first_shapes;
  fixture.decisions()[1].action_shapes = second_shapes;
  fixture.decisions()[1].residual_shape = second_shapes[0];
  fixture.decisions()[1].can_reduce_structure = true;
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 2;
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), 4, &options,
      &analysis_arena_, &collection));
  ASSERT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT);
  EXPECT_EQ(collection.decision_results[1].state,
            LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
  EXPECT_EQ(collection.site_classes[0], collection.site_classes[2]);
  EXPECT_EQ(collection.site_classes[1], collection.site_classes[3]);
  EXPECT_NE(collection.site_classes[0], collection.site_classes[1]);
}

// A scalar-only choice cannot help an over-budget residual. Its extra classes
// must not block a later provider that removes structural requirements. The
// same holds when every action retains a shared, launch-invariant dimension.
TEST_F(KernelClassClassifierTest, OptionalGrowthLeavesRoomForStructuralRelief) {
  for (bool shared_shape : {false, true}) {
    for (bool reverse : {false, true}) {
      for (uint16_t limit : {2, 3, 4}) {
        SCOPED_TRACE(shared_shape);
        SCOPED_TRACE(reverse);
        SCOPED_TRACE(limit);
        ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
        const loom_kernel_class_projection_term_t shared_term = {
            .coefficient = 1,
            .argument_ordinal = 2,
        };
        const loom_kernel_class_projection_t projections[] = {
            fixture.classifier()->projections[0],
            fixture.classifier()->projections[1],
            {
                .source_value_id = 2,
                .kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE,
                .terms = &shared_term,
                .term_count = 1,
            },
        };
        fixture.classifier()->projections = projections;
        fixture.classifier()->projection_count = 3;
        fixture.classifier()->kernel_argument_count = 3;
        fixture.classifier()->has_shape_requirements = true;
        const uint32_t widths[] = {0, 1, 2};
        const loom_kernel_class_shape_t shared = {
            .projections = widths + 2,
            .count = shared_shape ? 1u : 0u,
        };
        const loom_kernel_class_shape_t first_shapes[] = {shared, shared};
        const loom_kernel_class_shape_t second_shapes[] = {
            {}, {.projections = widths, .count = 2}};
        fixture.decisions()[0].action_shapes = first_shapes;
        fixture.decisions()[0].residual_shape = shared;
        fixture.decisions()[1].action_shapes = second_shapes;
        fixture.decisions()[1].residual_shape = second_shapes[1];
        fixture.decisions()[1].can_reduce_structure = true;
        if (reverse) {
          std::swap(fixture.decisions()[0], fixture.decisions()[1]);
        }
        loom_value_fact_table_t facts = {};
        IREE_ASSERT_OK(
            loom_value_fact_table_initialize(&facts, &analysis_arena_, 12));
        loom_value_id_t arguments[4][3];
        loom_kernel_class_site_t sites[4];
        for (uint32_t site = 0; site < 4; ++site) {
          const int64_t values[] = {site / 2, site % 2, 16};
          for (uint32_t argument = 0; argument < 3; ++argument) {
            arguments[site][argument] = site * 3 + argument;
            IREE_ASSERT_OK(loom_value_fact_table_define(
                &facts, arguments[site][argument],
                loom_value_facts_exact_i64(values[argument])));
          }
          sites[site] = {.facts = &facts, .argument_values = arguments[site]};
        }
        auto options = loom_kernel_class_collection_options_default();
        options.class_limit = limit;
        loom_kernel_class_collection_t collection = {};
        auto status = loom_kernel_class_classifier_collect(
            fixture.classifier(), nullptr, sites, 4, &options, &analysis_arena_,
            &collection);
        if (limit == 2) {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
          continue;
        }
        IREE_ASSERT_OK(status);
        ASSERT_EQ(collection.class_count, limit);
        EXPECT_EQ(collection.accepted_decision_count, limit == 3 ? 1 : 2);
        if (limit == 3) {
          EXPECT_EQ(collection.decision_results[reverse ? 1 : 0].state,
                    LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT);
          EXPECT_EQ(collection.decision_results[reverse ? 0 : 1].state,
                    LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
          EXPECT_EQ(collection.site_classes[1], collection.site_classes[3]);
          EXPECT_NE(collection.site_classes[0], collection.site_classes[2]);
          EXPECT_NE(collection.site_classes[0], collection.site_classes[1]);
          EXPECT_NE(collection.site_classes[2], collection.site_classes[1]);
        }
      }
    }
  }
}

TEST_F(KernelClassClassifierTest, SourceStructureIsReservedBeforeDecisions) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[2] = {};
  fixture.classifier()->has_shape_requirements = true;
  fixture.classifier()->shape = {.projections = &width, .count = 1};
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].action_shapes = shapes;
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 2;
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), 4, &options,
      &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 2u);
  EXPECT_EQ(collection.accepted_decision_count, 0u);
  EXPECT_EQ(collection.site_classes[0], collection.site_classes[2]);
  EXPECT_EQ(collection.site_classes[1], collection.site_classes[3]);
  options.class_limit = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_kernel_class_classifier_collect(
                            fixture.classifier(), nullptr, fixture.sites(), 4,
                            &options, &analysis_arena_, &collection));
}

TEST_F(KernelClassClassifierTest, UnknownShapeValuesDoNotAliasExactShapes) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  IREE_ASSERT_OK(
      fixture.SetSiteArgumentFacts(2, 1, loom_value_facts_unknown()));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = &width, .count = 1},
      {},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[0];
  fixture.decisions()[0].can_reduce_structure = true;
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 3;
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), 4, &options,
      &analysis_arena_, &collection));
  ASSERT_EQ(collection.class_count, 3u);
  EXPECT_NE(collection.site_classes[2], collection.site_classes[3]);
  const auto& unknown = collection.classes[collection.site_classes[2]];
  ASSERT_EQ(unknown.shape_value_count, 1u);
  EXPECT_TRUE(loom_value_facts_is_unknown(unknown.shape_values[0].facts));
  options.class_limit = 2;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_kernel_class_classifier_collect(
                            fixture.classifier(), nullptr, fixture.sites(), 4,
                            &options, &analysis_arena_, &collection));
}

TEST_F(KernelClassClassifierTest, UnavailableDecisionsRetainTheirStructure) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = &width, .count = 1},
      {},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[0];
  fixture.decisions()[0].can_reduce_structure = true;
  for (auto reason : {LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION,
                      LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT,
                      LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET}) {
    fixture.decisions()[0].unavailable_reason = reason;
    auto options = loom_kernel_class_collection_options_default();
    options.class_limit = 2;
    loom_kernel_class_collection_t collection = {};
    IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
        fixture.classifier(), nullptr, fixture.sites(), 4, &options,
        &analysis_arena_, &collection));
    EXPECT_EQ(collection.class_count, 2u);
    EXPECT_EQ(collection.accepted_decision_count, 0u);
    EXPECT_EQ(collection.site_classes[0], collection.site_classes[2]);
    EXPECT_EQ(collection.site_classes[1], collection.site_classes[3]);
    options.class_limit = 1;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_kernel_class_classifier_collect(
                              fixture.classifier(), nullptr, fixture.sites(), 4,
                              &options, &analysis_arena_, &collection));
  }
}

TEST_F(KernelClassClassifierTest, AllocationFailureReleasesAllScratch) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  const uint32_t projections[] = {0, 1};
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = projections + 1, .count = 1},
      {.projections = projections, .count = 1},
  };
  fixture.classifier()->has_shape_requirements = true;
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = {
      .projections = projections,
      .count = 2,
  };
  fixture.decisions()[0].can_reduce_structure = true;
  fixture.decisions()[1].action_shapes = shapes;
  fixture.decisions()[1].residual_shape = {
      .projections = projections,
      .count = 2,
  };
  fixture.decisions()[1].can_reduce_structure = true;
  fixture.decisions()[1].argument_values =
      fixture.decisions()[0].argument_values;
  fixture.decisions()[1].projection_ordinals =
      fixture.decisions()[0].projection_ordinals;
  auto options = loom_kernel_class_collection_options_default();
  options.class_limit = 3;
  struct Allocations {
    // Index of the system allocation to fail in this invocation.
    size_t fail_at = 0;
    // System allocation attempts since the invocation began.
    size_t attempts = 0;
    // Successfully allocated blocks not yet returned to the system.
    size_t live = 0;
  };
  for (size_t fail_at = 0;; ++fail_at) {
    SCOPED_TRACE(fail_at);
    Allocations allocations = {.fail_at = fail_at};
    const iree_allocator_t allocator = {
        .self = &allocations,
        .ctl = [](void* self, iree_allocator_command_t command,
                  const void* params, void** inout_ptr) -> iree_status_t {
          auto& state = *static_cast<Allocations*>(self);
          if (command != IREE_ALLOCATOR_COMMAND_FREE &&
              state.attempts++ == state.fail_at) {
            return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
          }
          const bool had_allocation = *inout_ptr != nullptr;
          const auto system = iree_allocator_system();
          iree_status_t status =
              system.ctl(system.self, command, params, inout_ptr);
          if (iree_status_is_ok(status)) {
            if (command == IREE_ALLOCATOR_COMMAND_FREE && had_allocation) {
              --state.live;
            } else if (!had_allocation && *inout_ptr) {
              ++state.live;
            }
          }
          return status;
        },
    };
    iree_arena_block_pool_t pool;
    iree_arena_block_pool_initialize(4096, allocator, &pool);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&pool, &arena);
    loom_kernel_class_collection_t collection = {};
    iree_status_t status = loom_kernel_class_classifier_collect(
        fixture.classifier(), nullptr, fixture.sites(), 4, &options, &arena,
        &collection);
    const bool complete = iree_status_is_ok(status);
    if (complete) {
      EXPECT_EQ(collection.class_count, 3u);
      EXPECT_EQ(collection.accepted_decision_count, 2u);
    } else {
      EXPECT_TRUE(iree_status_is_resource_exhausted(status));
      EXPECT_EQ(allocations.attempts, fail_at + 1);
      EXPECT_EQ(collection.class_count, 0u);
      EXPECT_EQ(collection.classes, nullptr);
      iree_status_ignore(status);
    }
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool);
    EXPECT_EQ(allocations.live, 0u);
    if (complete) {
      break;
    }
    ASSERT_EQ(allocations.attempts, fail_at + 1);
  }
}

TEST_F(KernelClassClassifierTest, StagedTuplesShareOneStructuralBudget) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("launches"),
                                      &block_pool_, nullptr,
                                      iree_allocator_system(), &module));
  ModulePtr owner(module);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  const auto index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint32_t site = 0; site < 4; ++site) {
    for (int64_t value : {int64_t(site / 2), int64_t(site % 2 + 1)}) {
      loom_op_t* constant = nullptr;
      IREE_ASSERT_OK(
          loom_index_constant_build(&builder, loom_attr_i64(value), index_type,
                                    LOOM_LOCATION_UNKNOWN, &constant));
    }
    IREE_ASSERT_OK(fixture.SetSiteArgumentFacts(
        site, 1, loom_value_facts_exact_i64(site % 2 + 1)));
  }
  const uint32_t width = 1;
  const loom_kernel_class_shape_t shapes[] = {
      {.projections = &width, .count = 1},
      {},
  };
  fixture.decisions()[0] = fixture.decisions()[1];
  fixture.decisions()[0].action_shapes = shapes;
  fixture.decisions()[0].residual_shape = shapes[0];
  fixture.decisions()[0].can_reduce_structure = true;
  fixture.classifier()->decision_count = 1;
  fixture.classifier()->specialization_count = 1;
  fixture.classifier()->has_shape_requirements = true;
  for (uint16_t limit : {2, 3, 4}) {
    SCOPED_TRACE(limit);
    auto options = loom_kernel_class_collection_options_default();
    options.class_limit = limit;
    loom_kernel_class_collection_t collection = {};
    if (limit < 4) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                            loom_kernel_class_classifier_collect(
                                fixture.classifier(), module, fixture.sites(),
                                4, &options, &analysis_arena_, &collection));
      continue;
    }
    IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
        fixture.classifier(), module, fixture.sites(), 4, &options,
        &analysis_arena_, &collection));
    EXPECT_EQ(collection.class_count, 4u);
    EXPECT_EQ(collection.accepted_decision_count, 1u);
    for (uint32_t site = 0; site < 4; ++site) {
      const auto& entry = collection.classes[collection.site_classes[site]];
      ASSERT_NE(entry.specializations, nullptr);
      EXPECT_EQ(entry.specializations[0].facts.range_lo, site / 2);
    }
  }
}

TEST_F(KernelClassClassifierTest,
       UnprojectableSoftDecisionRetainsGenericResidual) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].unavailable_reason =
      LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT;
  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
      &options, &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 1u);
  EXPECT_EQ(collection.accepted_decision_count, 0u);
  EXPECT_EQ(collection.skipped_decision_count, 1u);
  EXPECT_EQ(collection.trace_count, 0u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNPROJECTABLE_INPUT);
}

TEST_F(KernelClassClassifierTest,
       UnprojectableHardRequirementFailsInsteadOfWidening) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.classifier()->decision_count = 1;
  fixture.decisions()[0].unavailable_reason =
      LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT;
  fixture.decisions()[0].generic_result.kind =
      LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT;
  fixture.decisions()[0].generic_result.action_ordinal =
      LOOM_DECISION_PROGRAM_ACTION_INVALID;
  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_kernel_class_classifier_collect(
          fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
          &options, &analysis_arena_, &collection));
}

TEST_F(KernelClassClassifierTest,
       SelectedResultDependentActionRetainsGenericResidual) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.classifier()->decision_count = 1;
  fixture.actions(0)[0].contract_flags =
      LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT;
  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/4,
      &options, &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 1u);
  EXPECT_EQ(collection.accepted_decision_count, 0u);
  EXPECT_EQ(collection.skipped_decision_count, 1u);
  EXPECT_EQ(collection.trace_count, 0u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_SKIPPED_RESULT_DEPENDENT);
}

TEST_F(KernelClassClassifierTest,
       UnselectedResultDependentActionDoesNotSuppressInputOnlyAction) {
  ManualBinaryClassifier fixture(/*has_generic_residual=*/true);
  IREE_ASSERT_OK(fixture.InitializeSites(&analysis_arena_));
  fixture.classifier()->decision_count = 1;
  fixture.actions(0)[0].contract_flags =
      LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT;
  const loom_kernel_class_collection_options_t options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      fixture.classifier(), nullptr, fixture.sites(), /*site_count=*/2,
      &options, &analysis_arena_, &collection));
  EXPECT_EQ(collection.class_count, 1u);
  EXPECT_EQ(collection.accepted_decision_count, 1u);
  EXPECT_EQ(collection.skipped_decision_count, 0u);
  EXPECT_EQ(collection.trace_count, 1u);
  EXPECT_EQ(collection.decision_results[0].state,
            LOOM_KERNEL_CLASS_DECISION_ACCEPTED);
  EXPECT_EQ(collection.traces[0].action_ordinal, 1u);
}

TEST_F(KernelClassClassifierTest, ResultDependencyIsScopedToItsProviderAction) {
  const iree_string_view_t source = IREE_SV(R"(
template.decl @result.family(%input: index) -> (%output: index)

template.def<@result.family> priority(10) @result_guard(%input: index) -> (%output: index) where [ge(%output, 128)] {
  template.return %input : index
}

template.def<@result.family> priority(1) @fallback(%input: index) -> (index) {
  template.return %input : index
}

kernel.def @classified() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%input: index) {
  %result = template.apply<@result.family>(%input) : (index) -> (index)
  kernel.return
}
)");
  loom_kernel_class_classifier_t classifier = {};
  ModulePtr module =
      BuildClassifier(source, IREE_SV("classified"), &classifier);
  ASSERT_NE(module, nullptr);
  ASSERT_EQ(classifier.decision_count, 1u);
  const loom_kernel_class_decision_t* decision = &classifier.decisions[0];
  EXPECT_EQ(decision->hard_requirement_flags, 0u);
  ASSERT_EQ(decision->model->providers.count, 2u);
  for (uint32_t i = 0; i < decision->model->providers.count; ++i) {
    const bool is_result_guard = iree_string_view_equal(
        decision->model->providers.providers[i].name, IREE_SV("result_guard"));
    EXPECT_EQ(
        iree_any_bit_set(decision->actions[i].contract_flags,
                         LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT),
        is_result_guard);
  }
}

}  // namespace
}  // namespace loom
