// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class LowLowerSourcePlanTest : public ::testing::Test {
 protected:
  struct ObservedPlan {
    loom_op_kind_t source_op_kind = {};
    bool source_order_matches = false;
    bool elided = false;
  };

  struct SourcePlanObservation {
    loom_op_kind_t op_kinds[4] = {};
    iree_host_size_t op_count = 0;
    uint8_t phase = 0;
    bool invalid_lifecycle = false;
    bool selection_started = false;
    bool fail_at_end = false;
  };

  struct PlanObserver {
    const loom_op_t* expected_source_ops[3] = {nullptr, nullptr, nullptr};
    ObservedPlan plans[3];
    iree_host_size_t plan_count = 0;
    bool overflow = false;
    SourcePlanObservation source_plan;
  };

  static iree_status_t BeginSourcePlanObservation(
      void* user_data, loom_low_lower_context_t* context,
      void** out_observer_state) {
    auto* observer = static_cast<PlanObserver*>(user_data);
    SourcePlanObservation* observation = &observer->source_plan;
    observation->selection_started |=
        loom_low_lower_context_selected_plan_count(context) != 0;
    observation->invalid_lifecycle |= observation->phase != 0;
    observation->phase = 1;
    *out_observer_state = observation;
    return iree_ok_status();
  }

  static void ObserveSourcePlanOp(void* observer_state,
                                  loom_low_lower_context_t* context,
                                  const loom_op_t* source_op) {
    auto* observation = static_cast<SourcePlanObservation*>(observer_state);
    observation->selection_started |=
        loom_low_lower_context_selected_plan_count(context) != 0;
    observation->invalid_lifecycle |=
        observation->phase != 1 || source_op == nullptr;
    if (source_op != nullptr &&
        observation->op_count < IREE_ARRAYSIZE(observation->op_kinds)) {
      observation->op_kinds[observation->op_count] = source_op->kind;
    }
    ++observation->op_count;
  }

  static iree_status_t EndSourcePlanObservation(
      void* observer_state, loom_low_lower_context_t* context) {
    auto* observation = static_cast<SourcePlanObservation*>(observer_state);
    observation->selection_started |=
        loom_low_lower_context_selected_plan_count(context) != 0;
    observation->invalid_lifecycle |= observation->phase != 1;
    observation->phase = 2;
    if (observation->fail_at_end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source plan observation failed");
    }
    return iree_ok_status();
  }

  inline static const loom_low_lower_source_plan_observer_t
      kSourcePlanObserver = {
          .begin = BeginSourcePlanObservation,
          .observe = ObserveSourcePlanOp,
          .end = EndSourcePlanObservation,
          .user_data = nullptr,
      };

  static iree_status_t ObservePlan(void* user_data,
                                   loom_low_lower_context_t* context) {
    auto* observer = static_cast<PlanObserver*>(user_data);
    observer->plan_count = loom_low_lower_context_selected_plan_count(context);
    if (observer->plan_count > IREE_ARRAYSIZE(observer->plans)) {
      observer->overflow = true;
      return iree_ok_status();
    }
    for (iree_host_size_t i = 0; i < observer->plan_count; ++i) {
      const loom_low_lower_selected_plan_view_t plan =
          loom_low_lower_context_selected_plan_view(context, i);
      observer->plans[i].source_op_kind = plan.source_op->kind;
      observer->plans[i].source_order_matches =
          plan.source_op == observer->expected_source_ops[i];
      observer->plans[i].elided = plan.elided;
    }
    return iree_ok_status();
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source_plan_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];
    BuildFunction();
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));

    policy_ = *loom_test_low_lower_policy();
    source_plan_observer_ = kSourcePlanObserver;
    source_plan_observer_.user_data = &observer_;
    policy_.source_plan_observer = &source_plan_observer_;
    policy_.emit_preamble.fn = ObservePlan;
    policy_.emit_preamble.user_data = &observer_;
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = &policy_;
    options_.fact_table = &fact_table_;
  }

  void TearDown() override {
    loom_low_lower_result_deinitialize(&result_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&module_builder, IREE_SV("plan"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t argument_types[] = {i32_type, i32_type};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        argument_types, IREE_ARRAYSIZE(argument_types), &i32_type, 1, nullptr,
        0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);

    loom_region_t* body = loom_func_like_body(function_);
    loom_block_t* entry_block = loom_region_entry_block(body);
    const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
    const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena, entry_block,
                            &body_builder);
    body_builder.ip.parent_op = function_op;

    loom_op_t* dead_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, lhs, rhs, i32_type,
                                          LOOM_LOCATION_UNKNOWN, &dead_op));
    loom_op_t* dependency_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, lhs, rhs, i32_type,
                                          LOOM_LOCATION_UNKNOWN,
                                          &dependency_op));
    const loom_value_id_t dependency = loom_scalar_addi_result(dependency_op);
    loom_op_t* result_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, dependency, rhs,
                                          i32_type, LOOM_LOCATION_UNKNOWN,
                                          &result_op));
    const loom_value_id_t result = loom_scalar_addi_result(result_op);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));

    observer_.expected_source_ops[0] = dead_op;
    observer_.expected_source_ops[1] = dependency_op;
    observer_.expected_source_ops[2] = result_op;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_policy_t policy_ = {};
  loom_low_lower_options_t options_ = {};
  PlanObserver observer_;
  loom_low_lower_source_plan_observer_t source_plan_observer_ = {};
  loom_low_lower_result_t result_ = {};
};

TEST_F(LowLowerSourcePlanTest,
       ElidesDeadPlanAndRetainsReturnedDependencyChain) {
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_FALSE(observer_.overflow);
  ASSERT_EQ(observer_.plan_count, IREE_ARRAYSIZE(observer_.plans));
  for (const ObservedPlan& plan : observer_.plans) {
    EXPECT_EQ(plan.source_op_kind, LOOM_OP_SCALAR_ADDI);
    EXPECT_TRUE(plan.source_order_matches);
  }
  EXPECT_TRUE(observer_.plans[0].elided);
  EXPECT_FALSE(observer_.plans[1].elided);
  EXPECT_FALSE(observer_.plans[2].elided);

  EXPECT_EQ(observer_.source_plan.phase, 2u);
  EXPECT_FALSE(observer_.source_plan.invalid_lifecycle);
  EXPECT_FALSE(observer_.source_plan.selection_started);
  ASSERT_EQ(observer_.source_plan.op_count, 4u);
  EXPECT_EQ(observer_.source_plan.op_kinds[0], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[1], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[2], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[3], LOOM_OP_FUNC_RETURN);
}

TEST_F(LowLowerSourcePlanTest, PropagatesObserverEndFailureBeforeSelection) {
  observer_.source_plan.fail_at_end = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_lower_function(module_, function_, &options_, &result_));

  EXPECT_EQ(observer_.source_plan.phase, 2u);
  EXPECT_FALSE(observer_.source_plan.invalid_lifecycle);
  EXPECT_FALSE(observer_.source_plan.selection_started);
  EXPECT_EQ(observer_.plan_count, 0u);
}

TEST_F(LowLowerSourcePlanTest, FiltersObservedOperationKindRange) {
  source_plan_observer_.minimum_op_kind = LOOM_OP_FUNC_RETURN;
  source_plan_observer_.maximum_op_kind = LOOM_OP_FUNC_RETURN;

  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  EXPECT_EQ(observer_.source_plan.phase, 2u);
  EXPECT_FALSE(observer_.source_plan.invalid_lifecycle);
  EXPECT_FALSE(observer_.source_plan.selection_started);
  ASSERT_EQ(observer_.source_plan.op_count, 1u);
  EXPECT_EQ(observer_.source_plan.op_kinds[0], LOOM_OP_FUNC_RETURN);
}

}  // namespace
}  // namespace loom
