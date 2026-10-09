// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_facts.h"

#include <array>
#include <initializer_list>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/condition_fact_scope.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CfgConditionFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t cfg_vtable_count = 0;
    const loom_op_vtable_t* const* cfg_vtables =
        loom_cfg_dialect_vtables(&cfg_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_CFG, cfg_vtables, (uint16_t)cfg_vtable_count));
    iree_host_size_t index_vtable_count = 0;
    const loom_op_vtable_t* const* index_vtables =
        loom_index_dialect_vtables(&index_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 index_vtables,
                                                 (uint16_t)index_vtable_count));
    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, callee, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op_));
    body_ = loom_func_like_body(loom_func_like_cast(module_, func_op_));
    body_->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;

    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&fact_table_, &analysis_arena_, 16));
  }

  void TearDown() override {
    if (loom_local_value_domain_is_acquired(&value_domain_)) {
      loom_local_value_domain_release(&value_domain_);
    }
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_block_t* AppendBlock() {
    loom_block_t* block = nullptr;
    IREE_CHECK_OK(loom_region_append_block(module_, body_, &block));
    return block;
  }

  void SetBlock(loom_block_t* block) {
    loom_builder_set_block(&builder_, block);
    builder_.ip.parent_op = func_op_;
  }

  loom_value_id_t AddBlockArg(
      loom_block_t* block, loom_scalar_type_t type = LOOM_SCALAR_TYPE_INDEX) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_define_value(module_, loom_type_scalar(type), &value_id));
    IREE_CHECK_OK(loom_block_add_arg(module_, block, value_id));
    return value_id;
  }

  loom_value_id_t BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    loom_value_id_t result = loom_index_constant_result(op);
    IREE_CHECK_OK(loom_value_fact_table_define(
        &fact_table_, result, loom_value_facts_exact_i64(value)));
    return result;
  }

  loom_value_id_t BuildIndexCompare(loom_index_cmp_predicate_t predicate,
                                    loom_value_id_t left,
                                    loom_value_id_t right) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_cmp_build(&builder_, predicate, left, right,
                                       LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_cmp_result(op);
  }

  void BuildBranch(loom_block_t* dest,
                   std::initializer_list<loom_value_id_t> arguments = {}) {
    BuildBranch(dest, arguments.begin(), (uint16_t)arguments.size());
  }

  void BuildBranch(loom_block_t* dest, const loom_value_id_t* arguments,
                   uint16_t argument_count) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder_, dest, arguments, argument_count,
                                     LOOM_LOCATION_UNKNOWN, &op));
  }

  void BuildConditionalBranch(loom_value_id_t condition,
                              loom_block_t* true_dest,
                              loom_block_t* false_dest) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_cfg_cond_br_build(&builder_, condition, true_dest,
                                          false_dest, LOOM_LOCATION_UNKNOWN,
                                          &op));
  }

  enum class IdentityMode { kEmpty, kCfg };

  loom_cfg_condition_relation_table_t ComputeRelationTable(
      const loom_cfg_graph_t* graph, const loom_dominance_info_t* dominance,
      IdentityMode identity_mode = IdentityMode::kCfg,
      const loom_cfg_condition_relation_anchor_provider_t* anchor_provider =
          nullptr) {
    EXPECT_FALSE(loom_local_value_domain_is_acquired(&value_domain_));
    IREE_CHECK_OK(loom_local_value_domain_acquire_for_region_tree(
        module_, body_, &analysis_arena_, &value_domain_));
    IREE_CHECK_OK(loom_cfg_value_identity_table_initialize(
        &value_domain_, &analysis_arena_, &identities_));
    if (identity_mode == IdentityMode::kCfg) {
      loom_value_fact_cfg_region_t retained_region = {};
      IREE_CHECK_OK(loom_value_fact_cfg_region_initialize(
          module_, body_, &analysis_arena_, &retained_region));
      IREE_CHECK_OK(loom_cfg_value_identity_table_update(
          &identities_, &retained_region, dominance, &analysis_arena_));
    }
    loom_cfg_condition_relation_table_t table = {};
    IREE_CHECK_OK(loom_cfg_condition_relation_table_compute(
        module_, graph, &fact_table_, dominance, &value_domain_, &identities_,
        anchor_provider, &analysis_arena_, &table));
    return table;
  }

  bool HasRelation(const loom_cfg_condition_relation_table_t* table,
                   const loom_cfg_condition_relation_view_t* view,
                   loom_symbolic_integer_relation_t relation,
                   loom_value_id_t left, loom_value_id_t right) {
    loom_condition_relation_outcome_bits_t required_exclusions = 0;
    switch (relation) {
      case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS |
                              LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
        break;
      case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL;
        break;
      case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL |
                              LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
        break;
      case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
        break;
      case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS |
                              LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL;
        break;
      case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
        required_exclusions = LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS;
        break;
      default:
        return false;
    }
    const loom_condition_relation_outcome_bits_t actual_exclusions =
        loom_cfg_condition_relation_view_query_excluded_outcomes(
            table, view, &fact_table_,
            loom_condition_integer_operand_t{
                .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = left,
            },
            loom_condition_integer_operand_t{
                .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = right,
            });
    return iree_all_bits_set(actual_exclusions, required_exclusions);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  loom_value_fact_table_t fact_table_;
  loom_local_value_domain_t value_domain_ = {};
  loom_cfg_value_identity_table_t identities_ = {};
};

TEST_F(CfgConditionFactsTest, OmitsViewsWhenNoPathFactsAreDerived) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  BuildBranch(exit);
  SetBlock(exit);
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));

  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);
  EXPECT_EQ(loom_cfg_condition_relation_table_block(&table, 0), nullptr);
  EXPECT_EQ(loom_cfg_condition_relation_table_block(&table, 1), nullptr);
  ASSERT_EQ(graph.edge_count, 1u);
  EXPECT_EQ(loom_cfg_condition_relation_table_edge(&table, 0), nullptr);
}

TEST_F(CfgConditionFactsTest, PropagatesNestedBranchRelationsToTailBlock) {
  loom_block_t* entry_block = loom_region_entry_block(body_);
  loom_block_t* in_bounds_block = AppendBlock();
  loom_block_t* done_block = AppendBlock();
  loom_block_t* rot_block = AppendBlock();
  loom_block_t* tail_block = AppendBlock();

  SetBlock(entry_block);
  loom_value_id_t lane = AddBlockArg(entry_block);
  loom_value_id_t pair = AddBlockArg(entry_block);
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &fact_table_, lane, loom_value_facts_make(0, 1023, 1)));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &fact_table_, pair, loom_value_facts_make(0, 1023, 1)));
  loom_value_id_t half_cols = BuildIndexConstant(64);
  loom_value_id_t in_bounds =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, lane, half_cols);
  BuildConditionalBranch(in_bounds, in_bounds_block, done_block);

  SetBlock(in_bounds_block);
  loom_value_id_t half_dims = BuildIndexConstant(48);
  loom_value_id_t pair_in_rot =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_ULT, pair, half_dims);
  BuildConditionalBranch(pair_in_rot, rot_block, tail_block);

  SetBlock(rot_block);
  BuildBranch(done_block);

  SetBlock(tail_block);
  BuildBranch(done_block);

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {0};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {0};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));

  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const loom_cfg_condition_relation_view_t* in_bounds_facts =
      loom_cfg_condition_relation_table_block(&table, 1);
  ASSERT_NE(in_bounds_facts, nullptr);
  EXPECT_TRUE(HasRelation(&table, in_bounds_facts,
                          LOOM_SYMBOLIC_INTEGER_RELATION_LT, lane, half_cols));

  const loom_cfg_condition_relation_view_t* tail_facts =
      loom_cfg_condition_relation_table_block(&table, 4);
  ASSERT_NE(tail_facts, nullptr);
  EXPECT_TRUE(HasRelation(&table, tail_facts, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                          lane, half_cols));
  EXPECT_TRUE(HasRelation(&table, tail_facts, LOOM_SYMBOLIC_INTEGER_RELATION_GE,
                          pair, half_dims));
}

TEST_F(CfgConditionFactsTest,
       ResolvesExactConstantsAndOppositeRelationOrientation) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* guarded = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t value = AddBlockArg(entry);
  const loom_value_id_t exact_bound = BuildIndexConstant(64);
  const loom_value_id_t condition =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, value, exact_bound);
  BuildConditionalBranch(condition, guarded, exit);
  loom_op_t* terminator = nullptr;
  SetBlock(guarded);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const uint16_t guarded_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, guarded);
  const auto* guarded_facts =
      loom_cfg_condition_relation_table_block(&table, guarded_index);
  ASSERT_NE(guarded_facts, nullptr);
  const loom_condition_integer_operand_t value_operand = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value,
  };
  const loom_condition_integer_operand_t bound_value_operand = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = exact_bound,
  };
  const loom_condition_integer_operand_t bound_constant_operand = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
      .value_id = {},
      .constant = 64,
  };
  const loom_condition_relation_outcome_bits_t expected_forward =
      LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL |
      LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
  EXPECT_EQ(loom_cfg_condition_relation_view_query_excluded_outcomes(
                &table, guarded_facts, &fact_table_, value_operand,
                bound_value_operand),
            expected_forward);
  EXPECT_EQ(loom_cfg_condition_relation_view_query_excluded_outcomes(
                &table, guarded_facts, &fact_table_, value_operand,
                bound_constant_operand),
            expected_forward);
  EXPECT_EQ(loom_cfg_condition_relation_view_query_excluded_outcomes(
                &table, guarded_facts, &fact_table_, bound_constant_operand,
                value_operand),
            LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS |
                LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL);

  loom_condition_fact_scope_t scope = {};
  loom_condition_fact_scope_initialize_indexed(nullptr, &table, guarded_facts,
                                               &scope);
  loom_condition_integer_relation_t query = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      .left = value_operand,
      .right = bound_constant_operand,
  };
  bool result = false;
  EXPECT_TRUE(loom_condition_fact_scope_proves_integer_relation(
      &scope, &fact_table_, &query, &result));
  EXPECT_TRUE(result);
  query.relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
  EXPECT_TRUE(loom_condition_fact_scope_proves_integer_relation(
      &scope, &fact_table_, &query, &result));
  EXPECT_FALSE(result);
}

TEST_F(CfgConditionFactsTest, DerivedAnchorVisitsRelationWithAuthoredOperands) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* guarded = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t relation_left = AddBlockArg(entry);
  const loom_value_id_t relation_right = AddBlockArg(entry);
  const loom_value_id_t derived_anchor = AddBlockArg(entry);
  const loom_value_id_t unrelated_anchor = AddBlockArg(entry);
  const loom_value_id_t condition = BuildIndexCompare(
      LOOM_INDEX_CMP_PREDICATE_SLT, relation_left, relation_right);
  BuildConditionalBranch(condition, guarded, exit);
  loom_op_t* terminator = nullptr;
  SetBlock(guarded);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  struct AnchorMapping {
    loom_value_id_t relation;
    loom_value_id_t anchor;
    iree_host_size_t query_count;
  } mapping = {
      .relation = relation_left,
      .anchor = derived_anchor,
      .query_count = 0,
  };
  const loom_cfg_condition_relation_anchor_provider_t anchor_provider = {
      .user_data = &mapping,
      .query =
          [](void* user_data, loom_value_id_t relation_value_id,
             const loom_cfg_condition_relation_anchor_sink_t* sink) {
            auto* mapping = static_cast<AnchorMapping*>(user_data);
            ++mapping->query_count;
            if (relation_value_id != mapping->relation) {
              return iree_ok_status();
            }
            IREE_RETURN_IF_ERROR(
                sink->emit(sink->user_data, relation_value_id));
            IREE_RETURN_IF_ERROR(sink->emit(sink->user_data, mapping->anchor));
            return sink->emit(sink->user_data, mapping->anchor);
          },
  };
  const loom_cfg_condition_relation_table_t table = ComputeRelationTable(
      &graph, &dominance, IdentityMode::kCfg, &anchor_provider);
  EXPECT_EQ(mapping.query_count, 2u);
  EXPECT_EQ(table.derived_anchor_count, 1u);

  const uint16_t guarded_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, guarded);
  const auto* guarded_facts =
      loom_cfg_condition_relation_table_block(&table, guarded_index);
  ASSERT_NE(guarded_facts, nullptr);
  std::vector<loom_condition_integer_relation_t> relations;
  auto collect = [](void* user_data,
                    const loom_condition_integer_relation_t* relation) {
    static_cast<std::vector<loom_condition_integer_relation_t>*>(user_data)
        ->push_back(*relation);
    return true;
  };
  EXPECT_TRUE(loom_cfg_condition_relation_view_for_each_while(
      &table, guarded_facts, &fact_table_,
      loom_condition_integer_operand_t{
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_VALUE,
          /*.value_id=*/derived_anchor,
      },
      collect, &relations));
  ASSERT_EQ(relations.size(), 1u);
  EXPECT_EQ(relations[0].relation, LOOM_SYMBOLIC_INTEGER_RELATION_LT);
  EXPECT_EQ(relations[0].left.value_id, relation_left);
  EXPECT_EQ(relations[0].right.value_id, relation_right);

  relations.clear();
  EXPECT_TRUE(loom_cfg_condition_relation_view_for_each_while(
      &table, guarded_facts, &fact_table_,
      loom_condition_integer_operand_t{
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_VALUE,
          /*.value_id=*/unrelated_anchor,
      },
      collect, &relations));
  EXPECT_TRUE(relations.empty());
}

TEST_F(CfgConditionFactsTest, IntersectsJoinPredecessors) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* left = AppendBlock();
  loom_block_t* right = AppendBlock();
  loom_block_t* merge = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t path_condition =
      AddBlockArg(entry, LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t relation_left = AddBlockArg(entry);
  const loom_value_id_t relation_right = AddBlockArg(entry);
  BuildConditionalBranch(path_condition, left, right);

  SetBlock(left);
  const loom_value_id_t left_condition = BuildIndexCompare(
      LOOM_INDEX_CMP_PREDICATE_SLT, relation_left, relation_right);
  BuildConditionalBranch(left_condition, merge, exit);

  SetBlock(right);
  const loom_value_id_t right_condition = BuildIndexCompare(
      LOOM_INDEX_CMP_PREDICATE_SLT, relation_left, relation_right);
  BuildConditionalBranch(right_condition, merge, exit);

  loom_op_t* terminator = nullptr;
  SetBlock(merge);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const uint16_t merge_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, merge);
  const auto* merge_facts =
      loom_cfg_condition_relation_table_block(&table, merge_index);
  ASSERT_NE(merge_facts, nullptr);
  EXPECT_TRUE(HasRelation(&table, merge_facts,
                          LOOM_SYMBOLIC_INTEGER_RELATION_LT, relation_left,
                          relation_right));
  bool value = false;
  EXPECT_FALSE(loom_cfg_condition_relation_view_query_boolean(
      &table, merge_facts, path_condition, &value));
  EXPECT_FALSE(loom_cfg_condition_relation_view_query_boolean(
      &table, merge_facts, left_condition, &value));
  EXPECT_FALSE(loom_cfg_condition_relation_view_query_boolean(
      &table, merge_facts, right_condition, &value));
}

TEST_F(CfgConditionFactsTest, PreservesPredecessorFactsAfterJoinIntersection) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* less_path = AppendBlock();
  loom_block_t* greater_path = AppendBlock();
  loom_block_t* less_predecessor = AppendBlock();
  loom_block_t* greater_predecessor = AppendBlock();
  loom_block_t* merge = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t path_condition =
      AddBlockArg(entry, LOOM_SCALAR_TYPE_I1);
  const loom_value_id_t left = AddBlockArg(entry);
  const loom_value_id_t right = AddBlockArg(entry);
  BuildConditionalBranch(path_condition, less_path, greater_path);

  SetBlock(less_path);
  BuildConditionalBranch(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, left, right),
      less_predecessor, exit);
  SetBlock(greater_path);
  BuildConditionalBranch(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SGT, left, right),
      greater_predecessor, exit);
  SetBlock(less_predecessor);
  BuildBranch(merge);
  SetBlock(greater_predecessor);
  BuildBranch(merge);

  loom_op_t* terminator = nullptr;
  SetBlock(merge);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const uint16_t less_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, less_predecessor);
  const uint16_t greater_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, greater_predecessor);
  const uint16_t merge_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, merge);
  const auto* less_facts =
      loom_cfg_condition_relation_table_block(&table, less_index);
  const auto* greater_facts =
      loom_cfg_condition_relation_table_block(&table, greater_index);
  const auto* merge_facts =
      loom_cfg_condition_relation_table_block(&table, merge_index);
  ASSERT_NE(less_facts, nullptr);
  ASSERT_NE(greater_facts, nullptr);
  ASSERT_NE(merge_facts, nullptr);
  EXPECT_TRUE(HasRelation(&table, less_facts, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                          left, right));
  EXPECT_TRUE(HasRelation(&table, greater_facts,
                          LOOM_SYMBOLIC_INTEGER_RELATION_GT, left, right));
  EXPECT_FALSE(HasRelation(&table, merge_facts,
                           LOOM_SYMBOLIC_INTEGER_RELATION_LT, left, right));
  EXPECT_FALSE(HasRelation(&table, merge_facts,
                           LOOM_SYMBOLIC_INTEGER_RELATION_GT, left, right));

  const loom_cfg_edge_index_span_t less_edges =
      loom_cfg_graph_successor_edges(&graph, less_index);
  const loom_cfg_edge_index_span_t greater_edges =
      loom_cfg_graph_successor_edges(&graph, greater_index);
  ASSERT_EQ(less_edges.count, 1u);
  ASSERT_EQ(greater_edges.count, 1u);
  const auto* less_edge =
      loom_cfg_condition_relation_table_edge(&table, less_edges.values[0]);
  const auto* greater_edge =
      loom_cfg_condition_relation_table_edge(&table, greater_edges.values[0]);
  ASSERT_NE(less_edge, nullptr);
  ASSERT_NE(greater_edge, nullptr);
  EXPECT_TRUE(HasRelation(&table, less_edge, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                          left, right));
  EXPECT_TRUE(HasRelation(&table, greater_edge,
                          LOOM_SYMBOLIC_INTEGER_RELATION_GT, left, right));
}

TEST_F(CfgConditionFactsTest, FactorizesRepeatedPayloadValues) {
  constexpr uint16_t kWidth = 16;
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* guarded = AppendBlock();
  loom_block_t* target = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t origin = AddBlockArg(entry);
  const loom_value_id_t bound = AddBlockArg(entry);
  const loom_value_id_t condition =
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, origin, bound);
  BuildConditionalBranch(condition, guarded, exit);

  std::vector<loom_value_id_t> payload;
  payload.reserve(kWidth * 2);
  payload.insert(payload.end(), kWidth, origin);
  payload.insert(payload.end(), kWidth, bound);
  SetBlock(guarded);
  BuildBranch(target, payload.data(), (uint16_t)payload.size());

  std::array<loom_value_id_t, kWidth> left_values;
  std::array<loom_value_id_t, kWidth> right_values;
  for (uint16_t i = 0; i < kWidth; ++i) {
    left_values[i] = AddBlockArg(target);
  }
  for (uint16_t i = 0; i < kWidth; ++i) {
    right_values[i] = AddBlockArg(target);
  }
  loom_op_t* terminator = nullptr;
  SetBlock(target);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance, IdentityMode::kEmpty);

  const uint16_t target_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, target);
  const auto* target_facts =
      loom_cfg_condition_relation_table_block(&table, target_index);
  ASSERT_NE(target_facts, nullptr);
  ASSERT_EQ(target_facts->integer_relations.encoding,
            LOOM_CONDITION_RELATION_MATRIX_VIEW_PAGES);
  bool has_range_page = false;
  for (uint32_t i = 0; i < target_facts->integer_relations.entry_count; ++i) {
    has_range_page |=
        target_facts->integer_relations.entries.pages[i]->contents.encoding ==
        LOOM_CONDITION_RELATION_MATRIX_VIEW_RANGES;
  }
  EXPECT_TRUE(has_range_page);
  for (uint16_t left = 0; left < kWidth; ++left) {
    for (uint16_t right = 0; right < kWidth; ++right) {
      EXPECT_TRUE(HasRelation(&table, target_facts,
                              LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                              left_values[left], right_values[right]));
    }
  }
  std::vector<loom_condition_integer_relation_t> incident_relations;
  const loom_condition_integer_operand_t anchor = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = left_values[0],
  };
  EXPECT_TRUE(loom_cfg_condition_relation_view_for_each_while(
      &table, target_facts, &fact_table_, anchor,
      [](void* user_data, const loom_condition_integer_relation_t* relation) {
        static_cast<std::vector<loom_condition_integer_relation_t>*>(user_data)
            ->push_back(*relation);
        return true;
      },
      &incident_relations));
  ASSERT_EQ(incident_relations.size(), kWidth + 1);
  bool saw_dominating_bound = false;
  for (const auto& relation : incident_relations) {
    EXPECT_EQ(relation.relation, LOOM_SYMBOLIC_INTEGER_RELATION_LT);
    EXPECT_EQ(relation.left.value_id, left_values[0]);
    saw_dominating_bound |=
        relation.right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
        relation.right.value_id == bound;
  }
  EXPECT_TRUE(saw_dominating_bound);
}

enum class LoopArgumentTransfer { kReplace, kSelfForward, kSwap, kDominating };

class CfgConditionFactsLoopTest
    : public CfgConditionFactsTest,
      public ::testing::WithParamInterface<LoopArgumentTransfer> {};

TEST_P(CfgConditionFactsLoopTest, TranslatesBackedgeValuesBeforeMeetingFacts) {
  const auto transfer = GetParam();
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* guarded = AppendBlock();
  loom_block_t* seed = AppendBlock();
  loom_block_t* header = AppendBlock();
  loom_block_t* checked = AppendBlock();
  loom_block_t* latch = AppendBlock();
  loom_block_t* exit = AppendBlock();

  SetBlock(entry);
  const auto initial = AddBlockArg(entry);
  const auto replacement = AddBlockArg(entry);
  const auto bound = AddBlockArg(entry);
  const auto initial_condition = AddBlockArg(entry, LOOM_SCALAR_TYPE_I1);
  const auto replacement_condition = AddBlockArg(entry, LOOM_SCALAR_TYPE_I1);
  BuildConditionalBranch(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, initial, bound), guarded,
      exit);
  SetBlock(guarded);
  BuildConditionalBranch(initial_condition, seed, exit);
  SetBlock(seed);
  BuildBranch(header,
              {initial, replacement, initial_condition, replacement_condition});

  SetBlock(header);
  const auto carried = AddBlockArg(header);
  const auto other = AddBlockArg(header);
  const auto condition = AddBlockArg(header, LOOM_SCALAR_TYPE_I1);
  const auto other_condition = AddBlockArg(header, LOOM_SCALAR_TYPE_I1);
  const auto tested_value =
      transfer == LoopArgumentTransfer::kDominating ? initial : carried;
  const auto tested_condition = transfer == LoopArgumentTransfer::kDominating
                                    ? initial_condition
                                    : condition;
  BuildConditionalBranch(
      BuildIndexCompare(LOOM_INDEX_CMP_PREDICATE_SLT, tested_value, bound),
      checked, exit);
  SetBlock(checked);
  BuildConditionalBranch(tested_condition, latch, exit);
  SetBlock(latch);
  switch (transfer) {
    case LoopArgumentTransfer::kReplace:
    case LoopArgumentTransfer::kDominating:
      BuildBranch(header,
                  {replacement, other, replacement_condition, other_condition});
      break;
    case LoopArgumentTransfer::kSelfForward:
      BuildBranch(header, {carried, other, condition, other_condition});
      break;
    case LoopArgumentTransfer::kSwap:
      BuildBranch(header, {other, carried, other_condition, condition});
      break;
  }
  SetBlock(exit);
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {0};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  const auto header_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, header);
  const auto latch_index = (uint16_t)loom_cfg_graph_block_index(&graph, latch);
  loom_dominance_info_t dominance = {0};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);
  ASSERT_EQ(table.block_count, graph.block_count);

  const loom_cfg_edge_index_span_t latch_edges =
      loom_cfg_graph_successor_edges(&graph, latch_index);
  ASSERT_EQ(latch_edges.count, 1u);
  const loom_cfg_condition_relation_view_t* edge =
      loom_cfg_condition_relation_table_edge(&table, latch_edges.values[0]);
  ASSERT_NE(edge, nullptr);

  // Each trip binds fresh header arguments. Only the outgoing payload can
  // transfer a fact about the previous trip's argument to the next one.
  if (transfer == LoopArgumentTransfer::kReplace) {
    bool condition_value = false;
    EXPECT_FALSE(loom_cfg_condition_relation_view_query_boolean(
        &table, edge, condition, &condition_value));
    EXPECT_FALSE(HasRelation(&table, edge, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                             carried, bound));
  } else {
    const auto expected_value =
        transfer == LoopArgumentTransfer::kSwap ? other : tested_value;
    const auto expected_condition = transfer == LoopArgumentTransfer::kSwap
                                        ? other_condition
                                        : tested_condition;
    bool condition_value = false;
    EXPECT_TRUE(loom_cfg_condition_relation_view_query_boolean(
        &table, edge, expected_condition, &condition_value));
    EXPECT_TRUE(condition_value);
    EXPECT_TRUE(HasRelation(&table, edge, LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                            expected_value, bound));
  }
  if (transfer != LoopArgumentTransfer::kSelfForward) {
    const auto* header_facts =
        loom_cfg_condition_relation_table_block(&table, header_index);
    ASSERT_NE(header_facts, nullptr);
    EXPECT_FALSE(HasRelation(&table, header_facts,
                             LOOM_SYMBOLIC_INTEGER_RELATION_LT, carried,
                             bound));
  }
}

TEST_F(CfgConditionFactsTest, DistinguishesSameDestinationSuccessorEdges) {
  loom_block_t* entry = loom_region_entry_block(body_);
  loom_block_t* merge = AppendBlock();

  SetBlock(entry);
  const loom_value_id_t condition = AddBlockArg(entry, LOOM_SCALAR_TYPE_I1);
  BuildConditionalBranch(condition, merge, merge);
  SetBlock(merge);
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const loom_cfg_edge_index_span_t edges =
      loom_cfg_graph_successor_edges(&graph, 0);
  ASSERT_EQ(edges.count, 2u);
  bool value = false;
  const auto* true_edge =
      loom_cfg_condition_relation_table_edge(&table, edges.values[0]);
  ASSERT_NE(true_edge, nullptr);
  EXPECT_TRUE(loom_cfg_condition_relation_view_query_boolean(
      &table, true_edge, condition, &value));
  EXPECT_TRUE(value);
  const auto* false_edge =
      loom_cfg_condition_relation_table_edge(&table, edges.values[1]);
  ASSERT_NE(false_edge, nullptr);
  EXPECT_TRUE(loom_cfg_condition_relation_view_query_boolean(
      &table, false_edge, condition, &value));
  EXPECT_FALSE(value);

  const auto* merge_facts = loom_cfg_condition_relation_table_block(&table, 1);
  ASSERT_NE(merge_facts, nullptr);
  EXPECT_FALSE(loom_cfg_condition_relation_view_query_boolean(
      &table, merge_facts, condition, &value));
}

TEST_F(CfgConditionFactsTest, PreservesMoreThanThirtyTwoRelations) {
  constexpr size_t kRelationCount = 40;
  loom_block_t* entry = loom_region_entry_block(body_);
  std::array<loom_value_id_t, kRelationCount> left_values;
  std::array<loom_value_id_t, kRelationCount> right_values;
  SetBlock(entry);
  for (size_t i = 0; i < kRelationCount; ++i) {
    left_values[i] = AddBlockArg(entry);
    right_values[i] = AddBlockArg(entry);
  }

  std::array<loom_block_t*, kRelationCount - 1> continuation_blocks;
  for (loom_block_t*& block : continuation_blocks) {
    block = AppendBlock();
  }
  loom_block_t* success = AppendBlock();
  loom_block_t* exit = AppendBlock();
  for (size_t i = 0; i < kRelationCount; ++i) {
    const loom_value_id_t condition = BuildIndexCompare(
        LOOM_INDEX_CMP_PREDICATE_SLT, left_values[i], right_values[i]);
    loom_block_t* next =
        i + 1 == kRelationCount ? success : continuation_blocks[i];
    BuildConditionalBranch(condition, next, exit);
    if (i + 1 != kRelationCount) {
      SetBlock(next);
    }
  }
  SetBlock(success);
  loom_op_t* terminator = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));
  SetBlock(exit);
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &terminator));

  IREE_ASSERT_OK(loom_module_compute_uses(module_));
  loom_cfg_graph_t graph = {};
  IREE_ASSERT_OK(
      loom_cfg_graph_build(module_, body_, &analysis_arena_, &graph));
  loom_dominance_info_t dominance = {};
  IREE_ASSERT_OK(
      loom_dominance_info_initialize(module_, &analysis_arena_, &dominance));
  const loom_cfg_condition_relation_table_t table =
      ComputeRelationTable(&graph, &dominance);

  const uint16_t success_index =
      (uint16_t)loom_cfg_graph_block_index(&graph, success);
  const auto* success_facts =
      loom_cfg_condition_relation_table_block(&table, success_index);
  ASSERT_NE(success_facts, nullptr);
  for (size_t i = 0; i < kRelationCount; ++i) {
    EXPECT_TRUE(HasRelation(&table, success_facts,
                            LOOM_SYMBOLIC_INTEGER_RELATION_LT, left_values[i],
                            right_values[i]))
        << "relation " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Backedge, CfgConditionFactsLoopTest,
    ::testing::Values(LoopArgumentTransfer::kReplace,
                      LoopArgumentTransfer::kSelfForward,
                      LoopArgumentTransfer::kSwap,
                      LoopArgumentTransfer::kDominating),
    [](const ::testing::TestParamInfo<LoopArgumentTransfer>& parameter) {
      switch (parameter.param) {
        case LoopArgumentTransfer::kReplace:
          return "Replacement";
        case LoopArgumentTransfer::kSelfForward:
          return "SelfForwarding";
        case LoopArgumentTransfer::kSwap:
          return "ArgumentExchange";
        case LoopArgumentTransfer::kDominating:
          return "DominatingValues";
      }
      return "Invalid";
    });

}  // namespace
}  // namespace loom
