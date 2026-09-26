// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_uniformity.h"

#include <random>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/control_uniformity_test_util.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ControlUniformityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("control_uniformity_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT(function.op != nullptr);
    return function;
  }

  std::vector<const loom_op_t*> FindUses(loom_func_like_t function) {
    std::vector<const loom_op_t*> uses;
    FindUses(loom_func_like_body(function), &uses);
    return uses;
  }

  void FindUses(loom_region_t* region,
                std::vector<const loom_op_t*>* out_uses) {
    if (!region) {
      return;
    }
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_test_use_isa(op)) {
          out_uses->push_back(op);
        }
        loom_region_t* const* child_regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          FindUses(child_regions[i], out_uses);
        }
      }
    }
  }

  void DefineArgumentScope(loom_value_fact_table_t* fact_table,
                           loom_func_like_t function, uint16_t argument_ordinal,
                           loom_value_fact_uniform_scope_t scope) {
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function, &argument_count);
    ASSERT_LT(argument_ordinal, argument_count);
    loom_value_facts_t facts = loom_value_facts_make(0, 1, 1);
    loom_value_facts_mark_uniform_at_scope(&facts, scope);
    IREE_ASSERT_OK(loom_value_fact_table_define(
        fact_table, arguments[argument_ordinal], facts));
  }

  bool ProveMutuallyExclusive(loom_control_uniformity_info_t* info,
                              const std::vector<const loom_op_t*>& uses,
                              iree_host_size_t lhs_ordinal,
                              iree_host_size_t rhs_ordinal,
                              loom_value_fact_uniform_scope_t required_scope) {
    EXPECT_LT(lhs_ordinal, uses.size());
    EXPECT_LT(rhs_ordinal, uses.size());
    const loom_op_t* lhs_ops[] = {uses[lhs_ordinal]};
    const loom_op_t* rhs_ops[] = {uses[rhs_ordinal]};
    bool proven = false;
    IREE_EXPECT_OK(loom_control_uniformity_prove_mutually_exclusive_execution(
        info, IREE_ARRAYSIZE(lhs_ops), lhs_ops, IREE_ARRAYSIZE(rhs_ops),
        rhs_ops, required_scope, &proven));
    return proven;
  }

  bool ProveOperationSetsMutuallyExclusive(
      loom_control_uniformity_info_t* info,
      const std::vector<const loom_op_t*>& lhs_ops,
      const std::vector<const loom_op_t*>& rhs_ops,
      loom_value_fact_uniform_scope_t required_scope) {
    bool proven = false;
    IREE_EXPECT_OK(loom_control_uniformity_prove_mutually_exclusive_execution(
        info, lhs_ops.size(), lhs_ops.data(), rhs_ops.size(), rhs_ops.data(),
        required_scope, &proven));
    return proven;
  }

  void CheckExecution(const std::vector<std::vector<uint16_t>>& successors,
                      uint32_t uniform_selectors) {
    SCOPED_TRACE(::testing::PrintToString(successors));
    SCOPED_TRACE(uniform_selectors);
    IREE_ASSERT_OK(testing::CheckControlExecution(&context_, &block_pool_,
                                                  &analysis_arena_, successors,
                                                  uniform_selectors));
    iree_arena_reset(&analysis_arena_);
  }

  // Backing storage shared by module and analysis lifetimes.
  iree_arena_block_pool_t block_pool_;
  // Reset between independently generated CFGs.
  iree_arena_allocator_t analysis_arena_;
  // Registered dialect semantics used by fixtures.
  loom_context_t context_;
};

TEST_F(ControlUniformityTest, PartialReconvergenceIsNotAnExclusiveAlternative) {
  CheckExecution({{1, 2}, {3, 4}, {3, 4}, {4}, {}}, 0x1F);
}

TEST_F(ControlUniformityTest, PartialParticipationCannotProveExclusion) {
  CheckExecution({{1, 3}, {2, 3}, {4}, {4}, {}}, 0x1E);
}

TEST_F(ControlUniformityTest, BypassedControllerCannotProveExclusion) {
  CheckExecution({{2, 1}, {3, 4}, {4, 3}, {5, 5}, {5}, {}}, 0x0B);
}

TEST_F(ControlUniformityTest, ExhaustiveFourBlockPathsAndSelectorScopes) {
  // All forward CFGs with at most two alternatives, including duplicate edges,
  // and every choice of uniform versus lane-varying selectors.
  auto alternatives = [](uint16_t source) {
    std::vector<std::vector<uint16_t>> result{{}};
    for (uint16_t first = source + 1; first < 4; ++first) {
      result.push_back({first});
      for (uint16_t second = first; second < 4; ++second) {
        result.push_back({first, second});
      }
    }
    return result;
  };
  for (const auto& entry : alternatives(0)) {
    for (const auto& second : alternatives(1)) {
      for (const auto& third : alternatives(2)) {
        for (uint32_t uniform = 0; uniform < 8; ++uniform) {
          SCOPED_TRACE(uniform);
          CheckExecution({entry, second, third, {}}, uniform);
        }
      }
    }
  }
}

TEST_F(ControlUniformityTest, RandomAcyclicPathsAndMixedSelectorScopes) {
  std::mt19937 random(73864);
  for (uint32_t trial = 0; trial < 5000; ++trial) {
    SCOPED_TRACE(trial);
    const uint16_t count = 3 + random() % 5;
    std::vector<std::vector<uint16_t>> successors(count);
    for (uint16_t source = 0; source + 1 < count; ++source) {
      const uint32_t edge_count = random() % 3;
      for (uint32_t edge = 0; edge < edge_count; ++edge) {
        successors[source].push_back(source + 1 +
                                     random() % (count - source - 1));
      }
    }
    CheckExecution(successors, random() & ((1u << count) - 1));
  }
}

TEST_F(ControlUniformityTest, NonterminatingAlternativesCanFlowIntoEachOther) {
  CheckExecution({{1, 2}, {2}, {2}}, 0x7);
  CheckExecution({{2, 1}, {1}, {1}}, 0x7);
}

TEST_F(ControlUniformityTest, CyclicArmsAndPartialReconvergence) {
  CheckExecution({{1, 2}, {3, 5}, {4, 5}, {1, 5}, {2, 5}, {}}, 1);
  CheckExecution({{1, 2}, {3, 4}, {3, 4}, {1, 5}, {2, 5}, {}}, 1);
  CheckExecution({{1, 2}, {3}, {4}, {1, 4}, {2, 3}}, 0x1F);
}

TEST_F(ControlUniformityTest, RandomCyclicPathsAndMixedSelectorScopes) {
  std::mt19937 random(913756);
  for (uint32_t trial = 0; trial < 3000; ++trial) {
    SCOPED_TRACE(trial);
    const uint16_t count = 3 + random() % 5;
    std::vector<std::vector<uint16_t>> successors(count);
    for (auto& edges : successors) {
      const uint32_t edge_count = random() % 3;
      for (uint32_t i = 0; i < edge_count; ++i) {
        edges.push_back(1 + random() % (count - 1));
      }
    }
    CheckExecution(successors, random() & ((1u << count) - 1));
  }
}

TEST_F(ControlUniformityTest, ProvesDirectAlternativesAtSelectorScope) {
  ModulePtr module = ParseModule(R"(
func.def @direct(%condition: i1, %lhs: i32, %rhs: i32) {
  cfg.cond_br %condition, ^left, ^right
^left:
  test.use %lhs : i32
  cfg.br ^done
^right:
  test.use %rhs : i32
  cfg.br ^done
^done:
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("direct"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP));
  EXPECT_FALSE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));

  // Entry is uniform even when the selector partitions that execution scope.
  loom_region_t* body = loom_func_like_body(function);
  loom_block_t* entry = loom_region_entry_block(body);
  loom_condition_assumption_t condition;
  for (size_t i = 0; i < uses.size(); ++i) {
    ASSERT_TRUE(loom_control_uniformity_prove_single_entry(
        &info, uses[i]->parent_block, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP,
        &condition));
    EXPECT_EQ(condition.condition, loom_block_arg_id(entry, 0));
    EXPECT_EQ(condition.assumed_truth, i == 0);
  }
  EXPECT_FALSE(loom_control_uniformity_prove_single_entry(
      &info, entry, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, &condition));
  EXPECT_FALSE(loom_control_uniformity_prove_single_entry(
      &info, loom_region_block(body, 3),
      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, &condition));
}

TEST_F(ControlUniformityTest, ProvesInheritedNestedAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @nested(%outer: i1, %inner: i1, %a: i32, %b: i32, %c: i32) {
  cfg.cond_br %outer, ^first, ^not_first
^first:
  test.use %a : i32
  cfg.br ^done
^not_first:
  cfg.cond_br %inner, ^second, ^third
^second:
  test.use %b : i32
  cfg.br ^done
^third:
  test.use %c : i32
  cfg.br ^done
^done:
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("nested"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 3u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  DefineArgumentScope(&fact_table, function, 1,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 2,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 1, 2,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, ProvesCompleteFootprintsInsideAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @footprints(%outer: i1, %continue: i1, %lhs: i32, %rhs: i32, %joined: i32) {
  cfg.cond_br %outer, ^left, ^right
^left:
  test.use %lhs : i32
  cfg.br ^left_loop
^left_loop:
  test.use %lhs : i32
  cfg.cond_br %continue, ^left_loop, ^left_done
^left_done:
  cfg.br ^done
^right:
  test.use %rhs : i32
  cfg.br ^right_loop
^right_loop:
  test.use %rhs : i32
  cfg.cond_br %continue, ^right_loop, ^right_done
^right_done:
  cfg.br ^done
^done:
  test.use %joined : i32
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("footprints"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 5u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);

  const std::vector<const loom_op_t*> lhs_ops = {uses[0], uses[1]};
  const std::vector<const loom_op_t*> rhs_ops = {uses[2], uses[3]};
  EXPECT_TRUE(ProveOperationSetsMutuallyExclusive(
      &info, lhs_ops, rhs_ops, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));

  const std::vector<const loom_op_t*> lhs_join_live_ops = {uses[0], uses[1],
                                                           uses[4]};
  EXPECT_FALSE(ProveOperationSetsMutuallyExclusive(
      &info, lhs_join_live_ops, rhs_ops,
      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, ProvesNestedStructuredRegionAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @structured(%outer: i1, %inner: i1, %lhs: i32, %rhs: i32) {
  scf.if %outer {
    scf.if %inner {
      test.use %lhs : i32
    } else {
      test.use %lhs : i32
    }
  } else {
    test.use %rhs : i32
    test.use %rhs : i32
  }
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("structured"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 4u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  const std::vector<const loom_op_t*> lhs_ops = {uses[0], uses[1]};
  const std::vector<const loom_op_t*> rhs_ops = {uses[2], uses[3]};
  EXPECT_TRUE(ProveOperationSetsMutuallyExclusive(
      &info, lhs_ops, rhs_ops, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP));
  EXPECT_FALSE(ProveOperationSetsMutuallyExclusive(
      &info, lhs_ops, rhs_ops, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, ProvesConstantStructuredAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @constant_structured(%lhs: i32, %rhs: i32) {
  %condition = scalar.constant 1 : i1
  scf.if %condition {
    test.use %lhs : i32
  } else {
    test.use %rhs : i32
  }
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("constant_structured"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, ProvesMultiwayStructuredAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @structured_switch(%selector: index, %first: i32, %second: i32, %fallback: i32) {
  scf.switch %selector {
    case 0 {
      test.use %first : i32
      scf.yield
    }
    case 1 {
      test.use %second : i32
      scf.yield
    }
    default {
      test.use %fallback : i32
      scf.yield
    }
  }
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("structured_switch"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 3u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 0, 2,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
  EXPECT_TRUE(ProveMutuallyExclusive(&info, uses, 1, 2,
                                     LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, RejectsStructuredAlternativesInsideLoop) {
  ModulePtr module = ParseModule(R"(
func.def @structured_loop(%condition: i1, %lhs: i32, %rhs: i32) {
  %begin = index.constant 0 : index
  %end = index.constant 2 : index
  %step = index.constant 1 : index
  scf.for %iteration = [%begin to %end step %step] {
    scf.if %condition {
      test.use %lhs : i32
    } else {
      test.use %rhs : i32
    }
    scf.yield
  }
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("structured_loop"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_FALSE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, RejectsStructuredAlternativesInsideCfgCycle) {
  ModulePtr module = ParseModule(R"(
func.def @structured_cfg_loop(%continue: i1, %condition: i1, %lhs: i32, %rhs: i32) {
  cfg.br ^loop
^loop:
  scf.if %condition {
    test.use %lhs : i32
  } else {
    test.use %rhs : i32
  }
  cfg.cond_br %continue, ^loop, ^done
^done:
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("structured_cfg_loop"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  DefineArgumentScope(&fact_table, function, 1,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_FALSE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, RejectsAlternativesControlledInsideCycle) {
  ModulePtr module = ParseModule(R"(
func.def @cyclic(%continue: i1, %condition: i1, %lhs: i32, %rhs: i32) {
  cfg.br ^loop
^loop:
  cfg.cond_br %continue, ^body, ^done
^body:
  cfg.cond_br %condition, ^left, ^right
^left:
  test.use %lhs : i32
  cfg.br ^loop
^right:
  test.use %rhs : i32
  cfg.br ^loop
^done:
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("cyclic"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  DefineArgumentScope(&fact_table, function, 0,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  DefineArgumentScope(&fact_table, function, 1,
                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_FALSE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

TEST_F(ControlUniformityTest, SamePathOperationsAreNotAlternatives) {
  ModulePtr module = ParseModule(R"(
func.def @linear(%lhs: i32, %rhs: i32) {
  test.use %lhs : i32
  test.use %rhs : i32
  func.return
}
)");
  IREE_ASSERT_OK(loom_module_compute_uses(module.get()));
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("linear"));
  const std::vector<const loom_op_t*> uses = FindUses(function);
  ASSERT_EQ(uses.size(), 2u);

  loom_value_fact_table_t fact_table;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&fact_table, &analysis_arena_, 8));
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table, module.get(), function));
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &fact_table,
                                          &analysis_arena_, &info);
  EXPECT_FALSE(ProveMutuallyExclusive(&info, uses, 0, 1,
                                      LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP));
}

}  // namespace
}  // namespace loom
