// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/template_decision_model.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct ReferenceResult {
  loom_decision_program_result_kind_t kind;
  uint32_t action_ordinal;
  uint32_t unresolved_action_ordinal;
  uint64_t live_mask;
};

uint64_t LiveMask(const uint32_t* actions, uint32_t action_count) {
  uint64_t mask = 0;
  for (uint32_t i = 0; i < action_count; ++i) {
    mask |= UINT64_C(1) << actions[i];
  }
  return mask;
}

loom_decision_truth_t DecisionTruth(
    loom_template_provider_feasibility_t feasibility) {
  switch (feasibility) {
    case LOOM_TEMPLATE_PROVIDER_REJECT:
      return LOOM_DECISION_TRUTH_FALSE;
    case LOOM_TEMPLATE_PROVIDER_MAYBE:
      return LOOM_DECISION_TRUTH_UNKNOWN;
    case LOOM_TEMPLATE_PROVIDER_MATCH:
      return LOOM_DECISION_TRUTH_TRUE;
  }
  IREE_ASSERT_UNREACHABLE("invalid template provider feasibility");
  IREE_BUILTIN_UNREACHABLE();
}

ReferenceResult EvaluateReference(
    const std::vector<loom_template_provider_classification_t>& classifications,
    loom_template_provider_slice_t providers,
    loom_decision_program_resolution_policy_t policy) {
  ReferenceResult result = {
      /*.kind=*/LOOM_DECISION_PROGRAM_RESULT_NO_MATCH,
      /*.action_ordinal=*/LOOM_DECISION_PROGRAM_ACTION_INVALID,
      /*.unresolved_action_ordinal=*/LOOM_DECISION_PROGRAM_ACTION_INVALID,
  };
  bool has_match = false;
  bool has_maybe = false;
  int64_t best_match_priority = INT64_MIN;
  int64_t highest_maybe_priority = INT64_MIN;
  uint32_t best_match_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  uint32_t highest_maybe_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  uint32_t best_match_count = 0;
  for (uint32_t i = 0; i < providers.count; ++i) {
    const auto feasibility = classifications[i].feasibility;
    const int64_t priority = providers.providers[i].priority;
    if (feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
      has_maybe = true;
      if (highest_maybe_action == LOOM_DECISION_PROGRAM_ACTION_INVALID ||
          priority > highest_maybe_priority) {
        highest_maybe_priority = priority;
        highest_maybe_action = i;
      }
      continue;
    }
    if (feasibility != LOOM_TEMPLATE_PROVIDER_MATCH) continue;
    if (!has_match || priority > best_match_priority) {
      has_match = true;
      best_match_priority = priority;
      best_match_count = 1;
      best_match_action = i;
    } else if (priority == best_match_priority) {
      ++best_match_count;
    }
  }

  const bool unresolved_blocks_match =
      policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED && has_maybe &&
      (!has_match || highest_maybe_priority >= best_match_priority);
  if (!has_match || unresolved_blocks_match) {
    if (has_maybe) {
      result.kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
      result.unresolved_action_ordinal = highest_maybe_action;
    }
    if (policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
      for (uint32_t i = 0; i < providers.count; ++i) {
        const auto feasibility = classifications[i].feasibility;
        if (feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) continue;
        if (has_match && feasibility == LOOM_TEMPLATE_PROVIDER_MATCH &&
            providers.providers[i].priority != best_match_priority) {
          continue;
        }
        if (has_match && feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE &&
            providers.providers[i].priority < best_match_priority) {
          continue;
        }
        result.live_mask |= UINT64_C(1) << i;
      }
      if (!has_maybe) {
        result.live_mask = providers.count == 64
                               ? UINT64_MAX
                               : (UINT64_C(1) << providers.count) - 1;
      }
    }
    return result;
  }
  if (best_match_count > 1) {
    result.kind = LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS;
    for (uint32_t i = 0; i < providers.count; ++i) {
      if (classifications[i].feasibility == LOOM_TEMPLATE_PROVIDER_MATCH &&
          providers.providers[i].priority == best_match_priority) {
        result.live_mask |= UINT64_C(1) << i;
      }
    }
    return result;
  }
  result.kind = LOOM_DECISION_PROGRAM_RESULT_SELECTED;
  result.action_ordinal = best_match_action;
  result.live_mask = UINT64_C(1) << best_match_action;
  return result;
}

class TemplateDecisionModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(iree_string_view_t source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(source,
                                  IREE_SV("template_decision_model_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr ParseModule(const char* source) {
    return ParseModule(iree_make_cstring_view(source));
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_template_applicability_contract_t FamilyContract(
      const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
      loom_symbol_ref_t family) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup_ref(symbol_facts, module,
                                                    family, &base_facts));
    const loom_func_symbol_facts_t* facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return {
        /*.module=*/module,
        /*.target_symbol=*/facts->target_symbol,
        /*.target_facts=*/{},
        /*.argument_ids=*/facts->argument_ids,
        /*.result_ids=*/facts->result_ids,
        /*.predicates=*/facts->predicates,
        /*.target_conditions=*/facts->target_conditions,
        /*.argument_count=*/facts->argument_count,
        /*.result_count=*/facts->result_count,
        /*.predicate_count=*/facts->predicate_count,
        /*.target_condition_count=*/facts->target_condition_count,
    };
  }

  void CompareSite(
      const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
      const loom_template_decision_model_t* model, const loom_op_t* apply_op,
      const loom_template_applicability_facts_t* application_facts) {
    const loom_template_applicability_target_t application_target = {};
    const loom_template_decision_site_t site = {
        /*.application_op=*/apply_op,
        /*.application_target=*/&application_target,
        /*.application_facts=*/application_facts,
    };
    const loom_template_applicability_contract_t family_contract =
        FamilyContract(module, symbol_facts, model->family);
    loom_template_provider_classification_t family_classification = {};
    loom_template_applicability_classify_contract(
        module, apply_op, &family_contract, &application_target,
        application_facts, &family_classification);

    std::vector<loom_template_provider_classification_t> classifications(
        model->providers.count);
    for (uint32_t i = 0; i < model->providers.count; ++i) {
      loom_template_applicability_classify_provider(
          module, apply_op, &model->providers.providers[i], &application_target,
          application_facts, &classifications[i]);
    }

    for (const auto policy : {LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED,
                              LOOM_DECISION_PROGRAM_SELECT_PROVEN}) {
      std::array<loom_decision_program_choice_evidence_t, 8> evidence = {};
      std::array<uint32_t, 8> live_actions = {};
      ASSERT_LE(model->providers.count, evidence.size());
      uint32_t live_action_count = 0;
      loom_decision_program_result_t actual = {};
      loom_template_decision_evidence_summary_t actual_summary = {};
      loom_template_decision_model_evaluate_all(
          model, &site, policy, evidence.data(), &actual_summary,
          live_actions.data(), &live_action_count, &actual);

      std::array<uint32_t, 8> minimal_live_actions = {};
      uint32_t minimal_live_action_count = 0;
      loom_decision_program_result_t minimal_result = {};
      loom_template_decision_evidence_summary_t minimal_summary = {};
      loom_template_decision_model_evaluate(
          model, &site, policy, &minimal_summary, minimal_live_actions.data(),
          &minimal_live_action_count, &minimal_result);
      EXPECT_EQ(minimal_result.kind, actual.kind);
      EXPECT_EQ(minimal_result.action_ordinal, actual.action_ordinal);
      EXPECT_EQ(minimal_result.unresolved_action_ordinal,
                actual.unresolved_action_ordinal);
      EXPECT_EQ(minimal_result.unresolved_constraint,
                actual.unresolved_constraint);
      EXPECT_EQ(
          LiveMask(minimal_live_actions.data(), minimal_live_action_count),
          LiveMask(live_actions.data(), live_action_count));

      if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_REJECT) {
        EXPECT_EQ(actual.kind, LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT);
        EXPECT_EQ(live_action_count, 0u);
        continue;
      }
      if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
        EXPECT_EQ(actual.kind, LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED);
        if (policy == LOOM_DECISION_PROGRAM_SELECT_PROVEN) {
          EXPECT_EQ(live_action_count, 0u);
          continue;
        }
      }

      for (uint32_t choice_ordinal = 0;
           choice_ordinal < model->program.choice_count; ++choice_ordinal) {
        const uint32_t provider_ordinal =
            model->program.choices[choice_ordinal].action_ordinal;
        EXPECT_EQ(evidence[choice_ordinal].feasibility,
                  DecisionTruth(classifications[provider_ordinal].feasibility));
        if (evidence[choice_ordinal].feasibility ==
            LOOM_DECISION_TRUTH_UNKNOWN) {
          const loom_template_decision_constraint_info_t info =
              loom_template_decision_model_constraint_info(
                  model, evidence[choice_ordinal].unresolved_constraint);
          EXPECT_EQ(info.reason,
                    classifications[provider_ordinal].unresolved_reason);
          EXPECT_EQ(
              info.target_condition,
              classifications[provider_ordinal].unresolved_target_condition);
        }
      }
      if (family_classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE) {
        continue;
      }

      const ReferenceResult expected =
          EvaluateReference(classifications, model->providers, policy);
      EXPECT_EQ(actual.kind, expected.kind);
      EXPECT_EQ(actual.action_ordinal, expected.action_ordinal);
      EXPECT_EQ(actual.unresolved_action_ordinal,
                expected.unresolved_action_ordinal);
      EXPECT_EQ(LiveMask(live_actions.data(), live_action_count),
                expected.live_mask);

      loom_template_decision_model_summarize_choice_evidence(
          model, evidence.data(), &actual_summary);
      uint32_t expected_possible_count = 0;
      uint32_t expected_target_match_count = 0;
      uint32_t expected_target_unresolved_count = 0;
      int64_t best_match_priority = INT64_MIN;
      int64_t highest_unresolved_priority = INT64_MIN;
      uint32_t expected_best_match_count = 0;
      uint32_t expected_highest_unresolved_provider =
          LOOM_DECISION_PROGRAM_ACTION_INVALID;
      for (uint32_t i = 0; i < model->providers.count; ++i) {
        const auto& classification = classifications[i];
        expected_possible_count +=
            classification.feasibility != LOOM_TEMPLATE_PROVIDER_REJECT;
        expected_target_match_count +=
            classification.target_feasibility == LOOM_TEMPLATE_PROVIDER_MATCH;
        expected_target_unresolved_count +=
            classification.target_feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE;
        if (classification.feasibility == LOOM_TEMPLATE_PROVIDER_MAYBE &&
            (expected_highest_unresolved_provider ==
                 LOOM_DECISION_PROGRAM_ACTION_INVALID ||
             model->providers.providers[i].priority >
                 highest_unresolved_priority)) {
          highest_unresolved_priority = model->providers.providers[i].priority;
          expected_highest_unresolved_provider = i;
        }
        if (classification.feasibility != LOOM_TEMPLATE_PROVIDER_MATCH) {
          continue;
        }
        const int64_t priority = model->providers.providers[i].priority;
        if (priority > best_match_priority) {
          best_match_priority = priority;
          expected_best_match_count = 1;
        } else if (priority == best_match_priority) {
          ++expected_best_match_count;
        }
      }
      EXPECT_EQ(actual_summary.possible_count, expected_possible_count);
      EXPECT_EQ(actual_summary.target_identity_match_count,
                expected_target_match_count);
      EXPECT_EQ(actual_summary.target_identity_unresolved_count,
                expected_target_unresolved_count);
      EXPECT_EQ(actual_summary.best_match_count, expected_best_match_count);
      EXPECT_EQ(actual_summary.highest_unresolved_provider_ordinal,
                expected_highest_unresolved_provider);
      if (expected_highest_unresolved_provider !=
          LOOM_DECISION_PROGRAM_ACTION_INVALID) {
        const loom_template_decision_constraint_info_t constraint_info =
            loom_template_decision_model_constraint_info(
                model, actual_summary.highest_unresolved_constraint);
        EXPECT_EQ(constraint_info.reason,
                  classifications[expected_highest_unresolved_provider]
                      .unresolved_reason);
        EXPECT_EQ(constraint_info.target_condition,
                  classifications[expected_highest_unresolved_provider]
                      .unresolved_target_condition);
      }
    }
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

loom_value_facts_t RandomFacts(std::mt19937_64& random) {
  switch (random() % 4) {
    case 0:
      return loom_value_facts_unknown();
    case 1:
      return loom_value_facts_exact_i64((int64_t)(random() % 513) - 128);
    default: {
      const int64_t lo = (int64_t)(random() % 257) - 128;
      const int64_t hi = lo + (int64_t)(random() % 129);
      const int64_t divisor = INT64_C(1) << (random() % 4);
      return loom_value_facts_make(lo, hi, divisor);
    }
  }
}

TEST_F(TemplateDecisionModelTest,
       SynthesizedModelsMatchApplicabilityAcrossFactDomains) {
  ModulePtr module = ParseModule(R"(
template.decl @family.a(%x: index, %y: index) -> (index) where [le(%x, %y)]

template.def<@family.a> priority(10) @a_blocked(%x: index, %y: index) -> (index) where [range(%x, 32, 64), mul(%y, 8)] {
  template.return %x : index
}

template.def<@family.a> priority(10) @a_equal(%x: index, %y: index) -> (index) where [eq(%x, %y)] {
  template.return %y : index
}

template.def<@family.a> priority(1) @a_fallback(%x: index, %y: index) -> (index) where [lt(%x, 128)] {
  template.return %x : index
}

template.decl @family.b(%v0: index, %v1: index, %v2: index, %v3: index, %v4: index, %v5: index, %v6: index, %v7: index, %v8: index) -> (index) where [ge(%v8, -32)]

template.def<@family.b> priority(20) @b_exact(%v0: index, %v1: index, %v2: index, %v3: index, %v4: index, %v5: index, %v6: index, %v7: index, %v8: index) -> (index) where [eq(%v8, 7), mul(%v0, 3)] {
  template.return %v8 : index
}

template.def<@family.b> priority(2) @b_fallback(%v0: index, %v1: index, %v2: index, %v3: index, %v4: index, %v5: index, %v6: index, %v7: index, %v8: index) -> (index) where [le(%v8, 256)] {
  template.return %v0 : index
}

func.def public @entry(%x: index, %y: index, %m: index) -> (index, index) {
  %a = template.apply<@family.a>(%x, %y) : (index, index) -> (index)
  %b = template.apply<@family.b>(%m, %m, %m, %m, %m, %m, %m, %m, %m) : (index, index, index, index, index, index, index, index, index) -> (index)
  func.return %a, %b : index, index
}
)");

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena_);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &providers, module.get(), &symbol_facts));
  loom_symbol_reference_table_t references = {};
  IREE_ASSERT_OK(loom_symbol_reference_table_build(
      module.get(), &analysis_arena_, &references));
  loom_template_decision_model_catalog_t models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      module.get(), &symbol_facts, &references, &providers, &analysis_arena_,
      &models));
  ASSERT_EQ(models.model_count, 2u);
  ASSERT_EQ(references.template_demands.count, 2u);

  loom_value_fact_table_t value_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &value_facts, &analysis_arena_, module->values.count));
  loom_template_applicability_facts_t application_facts = {
      /*.values=*/&value_facts,
  };
  loom_condition_fact_set_initialize(nullptr, 0, &application_facts.path);

  for (iree_host_size_t i = 0; i < references.template_demands.count; ++i) {
    const loom_template_demand_t& demand =
        references.template_demands.values[i];
    const loom_symbol_ref_t family = {
        /*.module_id=*/0,
        /*.symbol_id=*/demand.family_symbol_id,
    };
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(&models, family);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(loom_template_decision_model_application_fact_requirements(
                  model, &demand),
              LOOM_TEMPLATE_DECISION_FACT_REQUIREMENT_VALUES);
  }

  std::mt19937_64 random(0xdec1510ada77ULL);
  for (uint32_t iteration = 0; iteration < 5000; ++iteration) {
    loom_value_fact_table_clear_scope(&value_facts);
    std::array<loom_value_facts_t, 3> facts = {
        RandomFacts(random), RandomFacts(random), RandomFacts(random)};
    for (iree_host_size_t i = 0; i < references.template_demands.count; ++i) {
      const loom_template_demand_t& demand =
          references.template_demands.values[i];
      const loom_value_slice_t operands =
          loom_template_apply_operands(demand.apply_op);
      if (operands.count == 2) {
        IREE_ASSERT_OK(loom_value_fact_table_define(
            &value_facts, operands.values[0], facts[0]));
        IREE_ASSERT_OK(loom_value_fact_table_define(
            &value_facts, operands.values[1], facts[1]));
      } else {
        ASSERT_EQ(operands.count, 9u);
        for (uint16_t j = 0; j < operands.count; ++j) {
          IREE_ASSERT_OK(loom_value_fact_table_define(
              &value_facts, operands.values[j], facts[2]));
        }
      }
      const loom_symbol_ref_t family = {
          /*.module_id=*/0,
          /*.symbol_id=*/demand.family_symbol_id,
      };
      const loom_template_decision_model_t* model =
          loom_template_decision_model_lookup(&models, family);
      ASSERT_NE(model, nullptr);
      CompareSite(module.get(), &symbol_facts, model, demand.apply_op,
                  &application_facts);
    }
  }
}

TEST_F(TemplateDecisionModelTest, LargeCatalogLookupUsesOnlyDemandedFamilies) {
  constexpr uint32_t kFamilyCount = 129;
  std::string source;
  for (uint32_t i = 0; i < kFamilyCount; ++i) {
    source.append("template.decl @family_");
    source.append(std::to_string(i));
    source.append("(%value: index) -> (index)\n");
    source.append("template.def<@family_");
    source.append(std::to_string(i));
    source.append("> @provider_");
    source.append(std::to_string(i));
    source.append(
        "(%value: index) -> (index) {\n"
        "  template.return %value : index\n"
        "}\n");
  }
  source.append(
      "template.decl @undemanded(%value: index) -> (index)\n"
      "template.def<@undemanded> @undemanded_provider(%value: index) -> "
      "(index) {\n"
      "  template.return %value : index\n"
      "}\n"
      "func.def public @entry(%value: index) -> (index) {\n");
  for (uint32_t i = 0; i < kFamilyCount; ++i) {
    source.append("  %result_");
    source.append(std::to_string(i));
    source.append(" = template.apply<@family_");
    source.append(std::to_string(i));
    source.append(">(%value) : (index) -> (index)\n");
  }
  source.append("  func.return %result_128 : index\n}\n");

  ModulePtr module = ParseModule(
      iree_make_string_view(source.data(), (iree_host_size_t)source.size()));
  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena_);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &providers, module.get(), &symbol_facts));
  loom_symbol_reference_table_t references = {};
  IREE_ASSERT_OK(loom_symbol_reference_table_build(
      module.get(), &analysis_arena_, &references));
  loom_template_decision_model_catalog_t models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      module.get(), &symbol_facts, &references, &providers, &analysis_arena_,
      &models));

  ASSERT_EQ(references.template_demands.family_count, kFamilyCount);
  ASSERT_EQ(models.model_count, kFamilyCount);
  ASSERT_NE(models.symbol_pages, nullptr);
  for (iree_host_size_t i = 0; i < references.template_demands.family_count;
       ++i) {
    const loom_symbol_ref_t family = {
        /*.module_id=*/0,
        /*.symbol_id=*/references.template_demands.family_symbol_ids[i],
    };
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(&models, family);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->family.symbol_id, family.symbol_id);
  }

  ASSERT_GT(references.template_providers.count, 0u);
  const loom_symbol_ref_t provider_symbol = {
      /*.module_id=*/0,
      /*.symbol_id=*/references.template_providers.values[0].symbol_id,
  };
  EXPECT_EQ(loom_template_decision_model_lookup(&models, provider_symbol),
            nullptr);
}

TEST_F(TemplateDecisionModelTest,
       TargetEvidenceIsCapturedByTheSingleDecisionTraversal) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @application_target
test.target<quirky> @other_target

template.decl @select_target(%value: i32) -> (i32)

template.def<@select_target> target(@other_target) priority(20) @other(%value: i32) -> (i32) {
  template.return %value : i32
}

template.def<@select_target> target(@application_target) priority(10) @matching(%value: i32) -> (i32) {
  template.return %value : i32
}

template.def<@select_target> priority(1) @independent(%value: i32) -> (i32) {
  template.return %value : i32
}

template.decl target(@other_target) @rejected_family(%value: i32) -> (i32)

template.def<@rejected_family> @rejected_implementation(%value: i32) -> (i32) {
  template.return %value : i32
}

func.def public target(@application_target) @entry(%value: i32) -> (i32, i32) {
  %selected = template.apply<@select_target>(%value) : (i32) -> (i32)
  %rejected = template.apply<@rejected_family>(%value) : (i32) -> (i32)
  func.return %selected, %rejected : i32, i32
}
)");

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena_);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena_);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &providers, module.get(), &symbol_facts));
  loom_symbol_reference_table_t references = {};
  IREE_ASSERT_OK(loom_symbol_reference_table_build(
      module.get(), &analysis_arena_, &references));
  loom_template_decision_model_catalog_t models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      module.get(), &symbol_facts, &references, &providers, &analysis_arena_,
      &models));

  const loom_symbol_ref_t application_target_ref = {
      /*.module_id=*/0,
      /*.symbol_id=*/FindSymbol(module.get(), IREE_SV("application_target")),
  };
  const loom_symbol_facts_base_t* application_target_base = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(&symbol_facts, module.get(),
                                                   application_target_ref,
                                                   &application_target_base));
  const loom_target_symbol_facts_t* application_target_facts =
      loom_target_symbol_facts_cast(application_target_base);
  ASSERT_NE(application_target_facts, nullptr);
  const loom_template_applicability_target_t application_target = {
      /*.witness=*/application_target_ref,
      /*.facts=*/application_target_facts->projection,
  };
  loom_template_applicability_facts_t application_facts = {};
  loom_condition_fact_set_initialize(nullptr, 0, &application_facts.path);

  ASSERT_EQ(references.template_demands.count, 2u);
  for (iree_host_size_t i = 0; i < references.template_demands.count; ++i) {
    const loom_template_demand_t& demand =
        references.template_demands.values[i];
    const loom_template_decision_model_t* model =
        loom_template_decision_model_lookup(
            &models,
            (loom_symbol_ref_t){/*.module_id=*/0,
                                /*.symbol_id=*/demand.family_symbol_id});
    ASSERT_NE(model, nullptr);
    const loom_template_decision_site_t site = {
        /*.application_op=*/demand.apply_op,
        /*.application_target=*/&application_target,
        /*.application_facts=*/&application_facts,
    };
    std::array<loom_decision_program_choice_evidence_t, 3> evidence = {};
    std::array<uint32_t, 3> live_actions = {};
    uint32_t live_action_count = 0;
    loom_decision_program_result_t result = {};
    loom_template_decision_evidence_summary_t summary = {};
    loom_template_decision_model_evaluate_all(
        model, &site, LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED, evidence.data(),
        &summary, live_actions.data(), &live_action_count, &result);

    if (model->family.symbol_id ==
        FindSymbol(module.get(), IREE_SV("select_target"))) {
      EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_SELECTED);
      EXPECT_EQ(
          loom_template_decision_model_provider(model, result.action_ordinal)
              ->symbol.symbol_id,
          FindSymbol(module.get(), IREE_SV("matching")));
      EXPECT_EQ(summary.family_target_identity, LOOM_TEMPLATE_PROVIDER_MATCH);
      EXPECT_EQ(summary.target_identity_match_count, 2u);
      EXPECT_EQ(summary.target_identity_unresolved_count, 0u);
      loom_template_decision_model_summarize_choice_evidence(
          model, evidence.data(), &summary);
      EXPECT_EQ(summary.possible_count, 2u);
      EXPECT_EQ(summary.best_match_count, 1u);
    } else {
      EXPECT_EQ(model->family.symbol_id,
                FindSymbol(module.get(), IREE_SV("rejected_family")));
      EXPECT_EQ(result.kind, LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT);
      EXPECT_EQ(summary.family_target_identity, LOOM_TEMPLATE_PROVIDER_REJECT);
      EXPECT_EQ(summary.target_identity_match_count, 0u);
      EXPECT_EQ(summary.target_identity_unresolved_count, 0u);
    }
  }
}

}  // namespace
}  // namespace loom
