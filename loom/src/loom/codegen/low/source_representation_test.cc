// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_representation.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/source_representation_verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/lower/source_representation.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class LowSourceRepresentationTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    descriptor_set_ = loom_test_low_core_descriptor_set();
    configuration_.flags = LOOM_TEST_LOW_SOURCE_REPRESENTATION_ENABLE_ALTERNATE;
  }

  void TearDown() override {
    if (value_domain_acquired_) {
      loom_local_value_domain_release(&value_domain_);
    }
    module_.reset();
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_module_t* ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("source_representation_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    module_.reset(module);
    return module;
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_func_like_cast(module,
                               module->symbols.entries[symbol_id].defining_op);
  }

  loom_value_id_t FindValue(const loom_module_t* module,
                            iree_string_view_t name) const {
    for (loom_value_ordinal_t i = 0; i < value_domain_.value_count; ++i) {
      const loom_value_id_t value_id = value_domain_.value_ids[i];
      if (iree_string_view_equal(loom_module_value_name(module, value_id),
                                 name)) {
        return value_id;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  const loom_op_t* FindOperation(loom_op_kind_t kind) const {
    for (loom_source_program_node_ordinal_t i = 0; i < program_.node_count;
         ++i) {
      const loom_source_program_node_t* node = &program_.nodes[i];
      if (node->kind != LOOM_SOURCE_PROGRAM_NODE_OPERATION) continue;
      const loom_op_t* op = loom_source_program_node_operation(node);
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  void BuildProgram(loom_module_t* module, loom_func_like_t function) {
    const loom_region_t* body = loom_func_like_body(function);
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
        module, body, &arena_, &value_domain_));
    value_domain_acquired_ = true;
    IREE_ASSERT_OK(loom_source_program_build(
        module, function.op, body, &value_domain_, &arena_, &program_));
  }

  iree_status_t Plan(
      const loom_low_source_representation_provider_t* provider =
          &loom_test_low_source_representation_provider,
      const loom_low_descriptor_set_t* descriptor_set = nullptr) {
    if (provider == nullptr) {
      provider = &loom_test_low_source_representation_provider;
    }
    const loom_low_source_representation_environment_t environment = {
        .module = program_.module,
        .descriptor_set =
            descriptor_set == nullptr ? descriptor_set_ : descriptor_set,
        .configuration = &configuration_,
    };
    return loom_low_source_representation_plan(provider, &program_,
                                               &environment, &arena_, &plan_);
  }

  loom_low_source_representation_value_view_t Value(
      loom_value_id_t value_id) const {
    return loom_low_source_representation_plan_lookup_value(&plan_, value_id);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  ModulePtr module_;
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_test_low_source_representation_configuration_t configuration_ = {};
  loom_local_value_domain_t value_domain_ = {};
  bool value_domain_acquired_ = false;
  loom_source_program_t program_ = {};
  loom_low_source_representation_plan_t plan_ = {};
};

TEST_F(LowSourceRepresentationTest, VerifiesSyntheticTargetTable) {
  IREE_EXPECT_OK(loom_low_source_representation_provider_verify(
      &loom_test_low_source_representation_provider, descriptor_set_, &arena_));
}

TEST_F(LowSourceRepresentationTest, VerifierRejectsMalformedGeneratedSpans) {
  std::vector<loom_low_source_representation_group_t> groups(
      loom_test_low_source_representation_provider.groups,
      loom_test_low_source_representation_provider.groups +
          loom_test_low_source_representation_provider.group_count);
  groups[0].port_start =
      loom_test_low_source_representation_provider.port_count;
  loom_low_source_representation_provider_t provider =
      loom_test_low_source_representation_provider;
  provider.groups = groups.data();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_source_representation_provider_verify(
                            &provider, descriptor_set_, &arena_));
}

TEST_F(LowSourceRepresentationTest,
       VerifierRejectsARepresentationDomainWithoutCanonicalFallback) {
  std::vector<loom_low_source_representation_binding_t> bindings(
      loom_test_low_source_representation_provider.bindings,
      loom_test_low_source_representation_provider.bindings +
          loom_test_low_source_representation_provider.binding_count);
  bindings[0].flags = 0;
  loom_low_source_representation_provider_t provider =
      loom_test_low_source_representation_provider;
  provider.bindings = bindings.data();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_low_source_representation_provider_verify(
                            &provider, descriptor_set_, &arena_));
}

TEST_F(LowSourceRepresentationTest,
       SelectsCheaperRepresentationAcrossCfgAndRetainsRealization) {
  loom_module_t* module = ParseModule(R"(
func.def @cfg(%condition: i1, %lhs: i32, %rhs: i32) -> (i32) {
  %sum = test.addi %lhs, %rhs : i32
  cfg.cond_br %condition, ^then, ^else
^then:
  cfg.br ^join(%sum : i32)
^else:
  cfg.br ^join(%lhs : i32)
^join(%joined: i32):
  func.return %joined : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("cfg")));
  IREE_ASSERT_OK(Plan());
  ASSERT_EQ(plan_.problem.kind, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE);

  const loom_value_id_t lhs = FindValue(module, IREE_SV("lhs"));
  const loom_value_id_t rhs = FindValue(module, IREE_SV("rhs"));
  const loom_value_id_t sum = FindValue(module, IREE_SV("sum"));
  const loom_value_id_t joined = FindValue(module, IREE_SV("joined"));
  const auto lhs_view = Value(lhs);
  ASSERT_TRUE(lhs_view.selected);
  EXPECT_EQ(lhs_view.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_ALTERNATE_KEY);
  EXPECT_TRUE(iree_string_view_equal(lhs_view.representation_name,
                                     IREE_SV("test.alternate")));
  EXPECT_EQ(lhs_view.component_ordinal, Value(rhs).component_ordinal);
  EXPECT_EQ(lhs_view.component_ordinal, Value(sum).component_ordinal);
  EXPECT_EQ(lhs_view.component_ordinal, Value(joined).component_ordinal);
  ASSERT_NE(lhs_view.cost, nullptr);
  EXPECT_EQ(lhs_view.cost->instruction_count, 1u);

  const loom_op_t* addi = FindOperation(LOOM_OP_TEST_ADDI);
  ASSERT_NE(addi, nullptr);
  EXPECT_EQ(loom_low_source_representation_plan_candidate_count(&plan_, addi),
            1u);
  const auto candidate =
      loom_low_source_representation_plan_candidate_view(&plan_, addi, 0);
  ASSERT_TRUE(candidate.selected);
  EXPECT_TRUE(
      iree_string_view_equal(candidate.group_name, IREE_SV("test.addi.frame")));
  EXPECT_TRUE(iree_string_view_equal(candidate.candidate_name,
                                     IREE_SV("test.addi.alternate")));
  ASSERT_NE(candidate.target_data, nullptr);
  EXPECT_EQ(
      static_cast<const loom_test_low_source_representation_target_data_t*>(
          candidate.target_data)
          ->realization,
      LOOM_TEST_LOW_SOURCE_REPRESENTATION_REALIZATION_ADDI_ALTERNATE);
  const uint64_t predicate_invocation_count =
      plan_.statistics.predicate_invocation_count;
  loom_low_source_representation_candidate_view_t found = {};
  EXPECT_TRUE(loom_low_source_representation_plan_find_candidate(
      &plan_, addi, LOOM_TEST_LOW_SOURCE_REPRESENTATION_ADDI_GROUP_KEY,
      &found));
  EXPECT_EQ(found.candidate, candidate.candidate);
  EXPECT_EQ(plan_.statistics.predicate_invocation_count,
            predicate_invocation_count);
  EXPECT_EQ(plan_.statistics.value_seed_invocation_count,
            value_domain_.value_count);
  EXPECT_GE(plan_.statistics.preserving_flow_count, 2u);
}

TEST_F(LowSourceRepresentationTest,
       CapabilityPredicatePreservesCanonicalFallback) {
  loom_module_t* module = ParseModule(R"(
func.def @canonical(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = test.addi %lhs, %rhs : i32
  func.return %sum : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("canonical")));
  configuration_.flags = 0;
  IREE_ASSERT_OK(Plan());
  const auto sum = Value(FindValue(module, IREE_SV("sum")));
  ASSERT_TRUE(sum.selected);
  EXPECT_EQ(sum.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY);
  const auto candidate = loom_low_source_representation_plan_candidate_view(
      &plan_, FindOperation(LOOM_OP_TEST_ADDI), 0);
  ASSERT_TRUE(candidate.selected);
  EXPECT_TRUE(iree_string_view_equal(candidate.candidate_name,
                                     IREE_SV("test.addi.canonical")));
  EXPECT_EQ(plan_.statistics.predicate_invocation_count, 1u);
}

TEST_F(LowSourceRepresentationTest,
       UnionsRegionBranchAndLoopCarriedComponents) {
  loom_module_t* module = ParseModule(R"(
func.def @structured(%condition: i1, %seed: i32, %lower: index,
                     %upper: index, %step: index) -> (i32, i32) {
  %selected = scf.if %condition -> (i32) {
    scf.yield %seed : i32
  } else {
    scf.yield %seed : i32
  }
  %counted = scf.for %iv = [%lower to %upper step %step]
      (%carried = %selected : i32) -> (i32) {
    scf.yield %carried : i32
  }
  %conditional = scf.while(%before = %counted : i32) -> (i32) {
    scf.condition %condition, %before : i1, i32
  } do(%body: i32) {
    scf.yield %body : i32
  }
  func.return %counted, %conditional : i32, i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("structured")));
  IREE_ASSERT_OK(Plan());
  const loom_value_ordinal_t component =
      Value(FindValue(module, IREE_SV("seed"))).component_ordinal;
  const char* names[] = {"selected", "carried", "counted",
                         "before",   "body",    "conditional"};
  for (const char* name : names) {
    const auto view = Value(FindValue(module, iree_make_cstring_view(name)));
    ASSERT_TRUE(view.selected) << name;
    EXPECT_EQ(view.component_ordinal, component) << name;
    EXPECT_EQ(view.representation->stable_key,
              LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY)
        << name;
  }
}

TEST_F(LowSourceRepresentationTest, UnionsIdentityAndAliasFlows) {
  loom_module_t* module = ParseModule(R"(
func.def @alias(%seed: i32) -> (i32) {
  %assumed = scalar.assume %seed [range(%seed, 0, 31)] : i32
  func.return %assumed : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("alias")));
  IREE_ASSERT_OK(Plan());
  const auto seed = Value(FindValue(module, IREE_SV("seed")));
  const auto assumed = Value(FindValue(module, IREE_SV("assumed")));
  ASSERT_TRUE(seed.selected);
  ASSERT_TRUE(assumed.selected);
  EXPECT_EQ(seed.component_ordinal, assumed.component_ordinal);
}

TEST_F(LowSourceRepresentationTest, ExplicitTransitionKeepsComponentsSeparate) {
  loom_module_t* module = ParseModule(R"(
func.def @transition(%input: i32) -> (f32) {
  %result = test.cast %input : i32 to f32
  func.return %result : f32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("transition")));
  IREE_ASSERT_OK(Plan());
  const auto input = Value(FindValue(module, IREE_SV("input")));
  const auto result = Value(FindValue(module, IREE_SV("result")));
  ASSERT_TRUE(input.selected);
  ASSERT_TRUE(result.selected);
  EXPECT_NE(input.component_ordinal, result.component_ordinal);
  EXPECT_EQ(input.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY);
  EXPECT_EQ(result.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_ALTERNATE_KEY);
  const auto candidate = loom_low_source_representation_plan_candidate_view(
      &plan_, FindOperation(LOOM_OP_TEST_CAST), 0);
  ASSERT_TRUE(candidate.selected);
  EXPECT_TRUE(iree_string_view_equal(candidate.candidate_name,
                                     IREE_SV("test.cast.alternate")));
}

TEST_F(LowSourceRepresentationTest, ExactCostTieSelectsCanonical) {
  loom_module_t* module = ParseModule(R"(
func.def @tie(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = test.addi %lhs, %rhs : i32
  func.return %sum : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("tie")));
  std::vector<loom_low_source_representation_candidate_t> candidates(
      loom_test_low_source_representation_provider.candidates,
      loom_test_low_source_representation_provider.candidates +
          loom_test_low_source_representation_provider.candidate_count);
  candidates[1].recipe_entry_start = candidates[0].recipe_entry_start;
  loom_low_source_representation_provider_t provider =
      loom_test_low_source_representation_provider;
  provider.candidates = candidates.data();
  IREE_ASSERT_OK(Plan(&provider));
  const auto sum = Value(FindValue(module, IREE_SV("sum")));
  ASSERT_TRUE(sum.selected);
  EXPECT_EQ(sum.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY);
}

TEST_F(LowSourceRepresentationTest, UnknownCostsSelectCanonical) {
  loom_module_t* module = ParseModule(R"(
func.def @unknown(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = test.addi %lhs, %rhs : i32
  func.return %sum : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("unknown")));
  std::vector<loom_low_schedule_class_t> schedule_classes(
      descriptor_set_->schedule_classes,
      descriptor_set_->schedule_classes +
          descriptor_set_->schedule_class_count);
  for (auto& schedule_class : schedule_classes) {
    schedule_class.model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN;
  }
  loom_low_descriptor_set_t descriptor_set = *descriptor_set_;
  descriptor_set.schedule_classes = schedule_classes.data();
  IREE_ASSERT_OK(Plan(nullptr, &descriptor_set));
  const auto sum = Value(FindValue(module, IREE_SV("sum")));
  ASSERT_TRUE(sum.selected);
  EXPECT_EQ(sum.representation->stable_key,
            LOOM_TEST_LOW_SOURCE_REPRESENTATION_CANONICAL_KEY);
  EXPECT_EQ(sum.cost->model_quality, LOOM_LOW_MODEL_QUALITY_UNKNOWN);
}

TEST_F(LowSourceRepresentationTest, RetainsEmptyDomainProblem) {
  loom_module_t* module = ParseModule(R"(
func.def @empty(%input: i32) -> (f32) {
  %result = test.cast %input : i32 to f32
  func.return %result : f32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("empty")));
  configuration_.flags = LOOM_TEST_LOW_SOURCE_REPRESENTATION_ENABLE_ALTERNATE |
                         LOOM_TEST_LOW_SOURCE_REPRESENTATION_EMPTY_I32_DOMAIN;
  IREE_ASSERT_OK(Plan());
  EXPECT_EQ(plan_.problem.kind,
            LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_EMPTY_DOMAIN);
  EXPECT_EQ(plan_.problem.source_value_id, FindValue(module, IREE_SV("input")));
}

TEST_F(LowSourceRepresentationTest,
       RequiredAndOptionalUnavailableGroupsRemainDistinct) {
  loom_module_t* module = ParseModule(R"(
func.def @unavailable(%lhs: i32, %rhs: i32) -> (i32) {
  %sum = test.addi %lhs, %rhs : i32
  func.return %sum : i32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("unavailable")));
  configuration_.flags = 0;
  std::vector<loom_low_source_representation_group_t> groups(
      loom_test_low_source_representation_provider.groups,
      loom_test_low_source_representation_provider.groups +
          loom_test_low_source_representation_provider.group_count);
  groups[0].candidate_start = 1;
  groups[0].candidate_count = 1;
  loom_low_source_representation_provider_t provider =
      loom_test_low_source_representation_provider;
  provider.groups = groups.data();
  IREE_ASSERT_OK(Plan(&provider));
  EXPECT_EQ(plan_.problem.kind,
            LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_UNAVAILABLE_GROUP);

  groups[0].flags = LOOM_LOW_SOURCE_REPRESENTATION_GROUP_OPTIONAL;
  IREE_ASSERT_OK(Plan(&provider));
  EXPECT_EQ(plan_.problem.kind, LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE);
  const auto candidate = loom_low_source_representation_plan_candidate_view(
      &plan_, FindOperation(LOOM_OP_TEST_ADDI), 0);
  EXPECT_FALSE(candidate.selected);
  EXPECT_NE(candidate.group, nullptr);
}

TEST_F(LowSourceRepresentationTest, RejectsNonseparableCandidateGroup) {
  loom_module_t* module = ParseModule(R"(
func.def @coupled(%input: i32) -> (f32) {
  %result = test.cast %input : i32 to f32
  func.return %result : f32
}
)");
  BuildProgram(module, FindFunction(module, IREE_SV("coupled")));
  std::vector<loom_low_source_representation_binding_t> bindings(
      loom_test_low_source_representation_provider.bindings,
      loom_test_low_source_representation_provider.bindings +
          loom_test_low_source_representation_provider.binding_count);
  bindings[4].representation_index = 1;
  bindings[4].flags = 0;
  loom_low_source_representation_provider_t provider =
      loom_test_low_source_representation_provider;
  provider.bindings = bindings.data();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Plan(&provider));
}

}  // namespace
}  // namespace loom
