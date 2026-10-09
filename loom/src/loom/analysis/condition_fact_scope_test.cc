// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_fact_scope.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static loom_condition_integer_operand_t ValueOperand(loom_value_id_t value) {
  return loom_condition_integer_operand_t{
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value,
  };
}

static loom_condition_derivation_t Derivation(
    loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_count) {
  return loom_condition_derivation_t{
      .integer_facts =
          {
              .integer_relations = relations,
              .integer_relation_count = relation_count,
              .integer_relation_capacity = relation_count,
          },
  };
}

TEST(ConditionFactScopeTest, ConjoinsRelationsAcrossFragments) {
  const loom_value_id_t left = 1;
  const loom_value_id_t right = 2;
  loom_condition_integer_relation_t parent_relations[] = {{
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  }};
  loom_condition_integer_relation_t child_relations[] = {{
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  }};
  const loom_condition_derivation_t parent_derivation =
      Derivation(parent_relations, IREE_ARRAYSIZE(parent_relations));
  const loom_condition_derivation_t child_derivation =
      Derivation(child_relations, IREE_ARRAYSIZE(child_relations));
  loom_condition_fact_scope_t parent_scope = {};
  loom_condition_fact_scope_initialize_local(nullptr, &parent_derivation,
                                             &parent_scope);
  loom_condition_fact_scope_t child_scope = {};
  loom_condition_fact_scope_initialize_local(&parent_scope, &child_derivation,
                                             &child_scope);

  const loom_condition_integer_relation_t query = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  };
  bool result = false;
  EXPECT_TRUE(loom_condition_fact_scope_proves_integer_relation(
      &child_scope, nullptr, &query, &result));
  EXPECT_TRUE(result);
}

TEST(ConditionFactScopeTest, ContradictoryFragmentsRemainUnknown) {
  const loom_value_id_t left = 1;
  const loom_value_id_t right = 2;
  loom_condition_integer_relation_t parent_relations[] = {{
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  }};
  loom_condition_integer_relation_t child_relations[] = {{
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  }};
  const loom_condition_derivation_t parent_derivation =
      Derivation(parent_relations, IREE_ARRAYSIZE(parent_relations));
  const loom_condition_derivation_t child_derivation =
      Derivation(child_relations, IREE_ARRAYSIZE(child_relations));
  loom_condition_fact_scope_t parent_scope = {};
  loom_condition_fact_scope_initialize_local(nullptr, &parent_derivation,
                                             &parent_scope);
  loom_condition_fact_scope_t child_scope = {};
  loom_condition_fact_scope_initialize_local(&parent_scope, &child_derivation,
                                             &child_scope);

  const loom_condition_integer_relation_t query = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      .left = ValueOperand(left),
      .right = ValueOperand(right),
  };
  bool result = false;
  EXPECT_FALSE(loom_condition_fact_scope_proves_integer_relation(
      &child_scope, nullptr, &query, &result));
}

TEST(ConditionFactScopeTest, ProjectedFragmentExposesRangeAndRelationFacts) {
  const loom_value_id_t value = 1;
  loom_condition_integer_relation_t relations[] = {
      {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE,
          .left = ValueOperand(value),
          .right = {
              /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
              /*.value_id=*/LOOM_VALUE_ID_INVALID,
              /*.constant=*/0,
          },
      },
      {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
          .left = ValueOperand(value),
          .right = {
              /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
              /*.value_id=*/LOOM_VALUE_ID_INVALID,
              /*.constant=*/16,
          },
      },
  };
  loom_condition_edge_projection_t projection = {};
  projection.source_derivation =
      Derivation(relations, IREE_ARRAYSIZE(relations));
  projection.visible_integer_relation_count = IREE_ARRAYSIZE(relations);
  loom_condition_fact_scope_t scope = {};
  loom_condition_fact_scope_initialize_projected(nullptr, &projection, &scope);

  EXPECT_TRUE(loom_condition_fact_scope_has_integer_relations(&scope));
  loom_value_facts_t value_facts = loom_value_facts_unknown();
  EXPECT_TRUE(loom_condition_fact_scope_apply_to_value_facts(
      &scope, nullptr, value, &value_facts));
  EXPECT_EQ(value_facts.range_lo, 0);
  EXPECT_EQ(value_facts.range_hi, 15);

  const loom_condition_integer_relation_t query = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      .left = ValueOperand(value),
      .right = {
          /*.kind=*/LOOM_CONDITION_INTEGER_OPERAND_CONSTANT,
          /*.value_id=*/LOOM_VALUE_ID_INVALID,
          /*.constant=*/16,
      },
  };
  bool result = false;
  EXPECT_TRUE(loom_condition_fact_scope_proves_integer_relation(
      &scope, nullptr, &query, &result));
  EXPECT_TRUE(result);
}

static bool CountVisitedRelation(
    void* user_data, const loom_condition_integer_relation_t* relation) {
  (void)relation;
  ++*(iree_host_size_t*)user_data;
  return true;
}

TEST(ConditionFactScopeTest, VisitsLocalRelationsOnceAcrossValueAnchors) {
  loom_condition_integer_relation_t relations[] = {
      {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
          .left = ValueOperand(1),
          .right = ValueOperand(2),
      },
      {
          .relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE,
          .left = ValueOperand(2),
          .right = ValueOperand(3),
      },
  };
  const loom_condition_derivation_t derivation =
      Derivation(relations, IREE_ARRAYSIZE(relations));
  loom_condition_fact_scope_t scope = {};
  loom_condition_fact_scope_initialize_local(nullptr, &derivation, &scope);
  const loom_value_id_t anchors[] = {1, 2, 3};

  iree_host_size_t visit_count = 0;
  EXPECT_TRUE(loom_condition_fact_scope_for_each_value_anchored_while(
      &scope, nullptr, anchors, IREE_ARRAYSIZE(anchors), CountVisitedRelation,
      &visit_count));
  EXPECT_EQ(visit_count, IREE_ARRAYSIZE(relations));
}

}  // namespace
}  // namespace loom
