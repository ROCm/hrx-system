// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_plan.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/bindings.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/control_plan.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"
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
    loom_op_kind_t op_kinds[6] = {};
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
    // Entry resources are validated before any definition is published.
    struct {
      // Number of planning callbacks after storage demand has settled.
      unsigned planning_count = 0;
      // Number of executors reached after successful planning.
      unsigned emission_count = 0;
      // Injects a resource-budget rejection at the target planning boundary.
      bool reject = false;
      // First argument's native carrier before any Low SSA value exists.
      loom_type_t argument_type = {};
    } entry;
    // Optional source fact/CFG arena retired at the start of emission.
    iree_arena_allocator_t* retire_analysis_arena = nullptr;
    struct {
      // Planned blocks carrying the same tuple through two nested rewrites.
      loom_low_lower_block_ref_t first = 0;
      // Final effective source-edge destination.
      loom_low_lower_block_ref_t second = 0;
      // Authored destination reached after both interpositions.
      loom_low_lower_block_ref_t destination = 0;
      // Number of callbacks observed before any Low construction.
      unsigned planning_count = 0;
      // Injects a target constraint rejection after a complete topology plan.
      bool reject = false;
    } control;

    // Type mapping must finish before selected and structural plans execute.
    struct {
      // Original test-target mapper delegated to by the observer.
      loom_low_lower_map_type_callback_t callback = {};
      // Queries received before emission begins.
      uint32_t planning_queries = 0;
      // Queries received after emission begins.
      uint32_t emission_queries = 0;
      // True after the preamble observes the completed source plan.
      bool emission_started = false;
    } type_mapping;
  };

  static iree_status_t ObserveTypeMapping(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
    auto* observer = static_cast<PlanObserver*>(user_data);
    auto& mapping = observer->type_mapping;
    if (mapping.emission_started) {
      ++mapping.emission_queries;
    } else {
      ++mapping.planning_queries;
    }
    return mapping.callback.fn(mapping.callback.user_data, context, source_op,
                               source_type, out_low_type);
  }

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

  static iree_status_t PlanControl(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   iree_arena_allocator_t* scratch_arena) {
    auto& control = static_cast<PlanObserver*>(user_data)->control;
    ++control.planning_count;
    EXPECT_EQ(loom_low_lower_context_low_function(context), nullptr);
    control.destination =
        loom_low_lower_control_successor(context, source_op, 0);
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, control.destination, control.destination, &control.first));
    // Copying a synthetic signature must retain the canonical source tuple.
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, control.first, control.first, &control.second));
    loom_low_lower_block_ref_t previous = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, source_op, 0, control.first, &previous));
    EXPECT_EQ(previous, control.destination);
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, source_op, 0, control.second, &previous));
    EXPECT_EQ(previous, control.first);
    EXPECT_EQ(loom_low_lower_control_argument_count(context, control.second),
              1u);
    EXPECT_TRUE(loom_type_equal(
        loom_low_lower_control_argument_type(context, control.second, 0),
        loom_low_lower_control_argument_type(context, control.destination, 0)));
    if (control.reject) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("test_control_constraint"));
    }
    return iree_ok_status();
  }

  static iree_status_t ObservePlan(void* user_data,
                                   loom_low_lower_context_t* context) {
    auto* observer = static_cast<PlanObserver*>(user_data);
    observer->type_mapping.emission_started = true;
    if (observer->retire_analysis_arena) {
      iree_arena_reset(observer->retire_analysis_arena);
    }
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
    if (observer->control.first) {
      loom_builder_t* builder = loom_low_lower_context_builder(context);
      const loom_builder_ip_t saved_ip = loom_builder_save(builder);
      const loom_low_lower_block_ref_t blocks[] = {
          observer->control.second, observer->control.first,
          observer->control.destination};
      iree_status_t status = iree_ok_status();
      for (unsigned i = 0; i < 2 && iree_status_is_ok(status); ++i) {
        loom_block_t* block = loom_low_lower_control_block(context, blocks[i]);
        loom_builder_set_block(builder, block);
        loom_op_t* branch = nullptr;
        status = loom_low_br_build(
            builder, loom_low_lower_control_block(context, blocks[i + 1]),
            block->arg_ids, block->arg_count, LOOM_LOCATION_UNKNOWN, &branch);
      }
      loom_builder_restore(builder, saved_ip);
      return status;
    }
    return iree_ok_status();
  }

  static iree_status_t PlanEntry(void* user_data,
                                 loom_low_lower_context_t* context) {
    auto* observer = static_cast<PlanObserver*>(user_data);
    auto& entry = observer->entry;
    ++entry.planning_count;
    EXPECT_EQ(loom_low_lower_context_low_function(context), nullptr);
    EXPECT_GT(loom_low_lower_context_selected_plan_count(context), 0u);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments = loom_func_like_arg_ids(
        loom_low_lower_context_source_function(context), &argument_count);
    EXPECT_EQ(argument_count, 2u);
    entry.argument_type =
        loom_low_lower_value_binding_type(context, arguments[0]);
    EXPECT_TRUE(loom_type_is_register(entry.argument_type));
    EXPECT_FALSE(
        loom_low_lower_source_value_has_low_mapping(context, arguments[0]));
    const loom_value_id_t refined_dependency =
        loom_op_const_operands(observer->expected_source_ops[2])[0];
    EXPECT_TRUE(loom_type_equal(
        loom_low_lower_value_binding_type(context, refined_dependency),
        entry.argument_type));
    EXPECT_FALSE(loom_low_lower_source_value_has_low_mapping(
        context, refined_dependency));
    // The unused add is already elided, so the target can account only for
    // resources that the selected program will actually emit.
    EXPECT_TRUE(loom_low_lower_context_selected_plan_view(context, 0).elided);
    if (!entry.reject) {
      return iree_ok_status();
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("plan")),
        loom_param_string(IREE_SV("test")),
        loom_param_u64(192),
        loom_param_u64(128),
    };
    return loom_low_lower_emit_error_ref(
        context, loom_low_lower_context_source_function(context).op,
        LOOM_ERR_TARGET_051_REF, params, IREE_ARRAYSIZE(params));
  }

  static iree_status_t EmitEntry(void* user_data,
                                 loom_low_lower_context_t* context) {
    auto& entry = static_cast<PlanObserver*>(user_data)->entry;
    EXPECT_EQ(entry.planning_count, 1u);
    EXPECT_NE(loom_low_lower_context_low_function(context), nullptr);
    uint16_t argument_count = 0;
    const loom_value_id_t* arguments = loom_func_like_arg_ids(
        loom_low_lower_context_source_function(context), &argument_count);
    EXPECT_TRUE(
        loom_low_lower_source_value_has_low_mapping(context, arguments[0]));
    loom_value_id_t low_argument =
        loom_low_lower_lookup_value(context, arguments[0]);
    EXPECT_TRUE(loom_type_equal(
        entry.argument_type,
        loom_module_value_type(loom_low_lower_context_module(context),
                               low_argument)));
    ++entry.emission_count;
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
    observer_.type_mapping.callback = policy_.map_type;
    policy_.map_type = {ObserveTypeMapping, &observer_};
    source_plan_observer_ = kSourcePlanObserver;
    source_plan_observer_.user_data = &observer_;
    policy_.source_plan_observer = &source_plan_observer_;
    policy_.emit_preamble.fn = ObservePlan;
    policy_.emit_preamble.user_data = &observer_;
    options_.target_ref = loom_symbol_ref_null();
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

  void BuildFunction(iree_string_view_t name = IREE_SV("plan")) {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        .module_id = 0,
        .symbol_id = symbol_id,
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
    // Fresh identity prevents commoning but does not make an unused result
    // observable. The source plan must agree with canonical DCE on that
    // distinction.
    dead_op->traits |= LOOM_TRAIT_UNIQUE_IDENTITY;
    const loom_value_id_t dead = loom_scalar_addi_result(dead_op);
    loom_op_t* dead_identity_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_assume_build(
        &body_builder, &dead, 1, /*predicates=*/nullptr,
        /*predicates_count=*/0, &i32_type, 1, LOOM_LOCATION_UNKNOWN,
        &dead_identity_op));
    loom_op_t* dependency_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, lhs, rhs, i32_type,
                                          LOOM_LOCATION_UNKNOWN,
                                          &dependency_op));
    const loom_value_id_t dependency = loom_scalar_addi_result(dependency_op);
    loom_op_t* dependency_identity_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_assume_build(
        &body_builder, &dependency, 1, /*predicates=*/nullptr,
        /*predicates_count=*/0, &i32_type, 1, LOOM_LOCATION_UNKNOWN,
        &dependency_identity_op));
    const loom_value_id_t dependency_identity =
        loom_scalar_assume_results(dependency_identity_op).values[0];
    loom_op_t* result_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, dependency_identity,
                                          rhs, i32_type, LOOM_LOCATION_UNKNOWN,
                                          &result_op));
    const loom_value_id_t result = loom_scalar_addi_result(result_op);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));

    observer_.expected_source_ops[0] = dead_op;
    observer_.expected_source_ops[1] = dependency_op;
    observer_.expected_source_ops[2] = result_op;
  }

  void AddForwardingBlock() {
    policy_.source_plan_observer = nullptr;
    policy_.prepare_branch = {PlanControl, &observer_};
    loom_region_t* body = loom_func_like_body(function_);
    loom_block_t* entry = loom_region_entry_block(body);
    loom_op_t* original_return = entry->last_op;
    const loom_value_id_t result = loom_op_const_operands(original_return)[0];
    const loom_type_t type = loom_module_value_type(module_, result);
    loom_block_t* destination = nullptr;
    IREE_ASSERT_OK(loom_region_append_block(module_, body, &destination));
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena, entry, &builder);
    loom_builder_set_before(&builder, original_return);
    loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_define_block_arg(&builder, destination, type, &argument));
    loom_op_t* branch = nullptr;
    IREE_ASSERT_OK(loom_cfg_br_build(&builder, destination, &result, 1,
                                     LOOM_LOCATION_UNKNOWN, &branch));
    IREE_ASSERT_OK(loom_op_erase(module_, original_return));
    loom_builder_set_block(&builder, destination);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&builder, &argument, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));
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
       ElidesDeadUniqueIdentityChainAndRetainsReturnedIdentityChain) {
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
  EXPECT_GT(observer_.type_mapping.planning_queries, 0u);
  EXPECT_EQ(observer_.type_mapping.emission_queries, 0u);

  EXPECT_EQ(observer_.source_plan.phase, 2u);
  EXPECT_FALSE(observer_.source_plan.invalid_lifecycle);
  EXPECT_FALSE(observer_.source_plan.selection_started);
  ASSERT_EQ(observer_.source_plan.op_count, 6u);
  EXPECT_EQ(observer_.source_plan.op_kinds[0], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[1], LOOM_OP_SCALAR_ASSUME);
  EXPECT_EQ(observer_.source_plan.op_kinds[2], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[3], LOOM_OP_SCALAR_ASSUME);
  EXPECT_EQ(observer_.source_plan.op_kinds[4], LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(observer_.source_plan.op_kinds[5], LOOM_OP_FUNC_RETURN);
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

TEST_F(LowLowerSourcePlanTest, ExecutesExpandedControlAfterFactStorageRetires) {
  AddForwardingBlock();
  observer_.retire_analysis_arena = &analysis_arena_;
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_NE(result_.low_func_op, nullptr);
  EXPECT_EQ(observer_.control.planning_count, 1u);
  EXPECT_EQ(observer_.type_mapping.emission_queries, 0u);
  const loom_region_t* body =
      loom_func_like_body(loom_func_like_cast(module_, result_.low_func_op));
  ASSERT_EQ(body->block_count, 4u);
  // Before-placement and nested edge rewrites preserve the same linear chain.
  for (uint16_t i = 0; i < 3; ++i) {
    const loom_block_t* block = loom_region_const_block(body, i);
    const loom_block_t* next = loom_region_const_block(body, i + 1);
    const loom_op_t* branch = loom_block_const_last_op(block);
    ASSERT_TRUE(loom_low_br_isa(branch));
    EXPECT_EQ(loom_low_br_dest(branch), next);
    EXPECT_EQ(loom_low_br_args(branch).count, 1u);
    ASSERT_EQ(next->arg_count, 1u);
    EXPECT_TRUE(loom_type_equal(
        loom_module_value_type(module_, loom_low_br_args(branch).values[0]),
        loom_block_arg_type(module_, next, 0)));
  }
}

TEST_F(LowLowerSourcePlanTest, RejectsControlPlanWithoutPublishingBlocks) {
  AddForwardingBlock();
  observer_.control.reject = true;
  options_.max_errors = 1;
  const iree_host_size_t initial_value_count = module_->values.count;
  const loom_region_t* source_body = loom_func_like_body(function_);
  const loom_symbol_ref_t symbol = loom_func_like_callee(function_);
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);
  EXPECT_EQ(observer_.control.planning_count, 1u);
  EXPECT_FALSE(observer_.type_mapping.emission_started);
  EXPECT_EQ(module_->values.count, initial_value_count);
  EXPECT_EQ(source_body->block_count, 2u);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            function_.op);
}

enum class ControlBoundary { kBranch, kReturn };

class LowLowerControlOperandTest
    : public LowLowerSourcePlanTest,
      public ::testing::WithParamInterface<ControlBoundary> {};

TEST_P(LowLowerControlOperandTest,
       RejectsReceivingCarrierBeforePublishingValues) {
  if (GetParam() == ControlBoundary::kBranch) {
    AddForwardingBlock();
  }
  loom_value_id_t receiving_argument =
      GetParam() == ControlBoundary::kBranch
          ? loom_block_arg_id(
                loom_region_block(loom_func_like_body(function_), 1), 0)
          : LOOM_VALUE_ID_INVALID;
  // The test target's arithmetic produces one register, while this policy
  // requests pairs at receiving boundaries. This exercises conversion of an
  // actual producer carrier instead of remapping that producer's source type.
  policy_.map_value = {
      [](void* user_data, loom_low_lower_context_t* context,
         const loom_op_t* source_op, loom_value_id_t source_value,
         loom_type_t source_type, loom_type_t* out_type) -> iree_status_t {
        IREE_RETURN_IF_ERROR(loom_test_low_lower_map_type(
            nullptr, context, source_op, source_type, out_type));
        if (loom_func_return_isa(source_op) ||
            source_value == *static_cast<loom_value_id_t*>(user_data)) {
          *out_type =
              loom_low_register_carrier_type_with_unit_count(*out_type, 2);
        }
        return iree_ok_status();
      },
      &receiving_argument};
  policy_.control_operand = {
      [](void* user_data, loom_low_lower_context_t* context,
         const loom_op_t* terminator, loom_value_id_t source_value,
         loom_type_t required_type, const void** out_plan) -> iree_status_t {
        *out_plan = nullptr;
        EXPECT_EQ(loom_low_lower_context_low_function(context), nullptr);
        EXPECT_EQ(loom_low_register_type_unit_count(
                      loom_low_lower_value_binding_type(context, source_value)),
                  1u);
        EXPECT_EQ(loom_low_register_type_unit_count(required_type), 2u);
        return loom_low_lower_emit_branch_constraint(
            context, terminator, IREE_SV("test_control_operand_constraint"));
      },
      [](void* user_data, loom_low_lower_context_t* context,
         const loom_op_t* terminator, loom_value_id_t source_value,
         loom_value_id_t low_value, loom_type_t required_type, const void* plan,
         loom_value_id_t* out_value) -> iree_status_t {
        ADD_FAILURE() << "Rejected control operand reached emission";
        *out_value = low_value;
        return iree_ok_status();
      },
      nullptr};
  options_.max_errors = 1;
  const iree_host_size_t value_count = module_->values.count;
  const loom_region_t* body = loom_func_like_body(function_);
  const uint16_t block_count = body->block_count;
  const loom_symbol_ref_t symbol = loom_func_like_callee(function_);
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);
  EXPECT_FALSE(observer_.type_mapping.emission_started);
  EXPECT_EQ(module_->values.count, value_count);
  EXPECT_EQ(body->block_count, block_count);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            function_.op);
}

INSTANTIATE_TEST_SUITE_P(
    ControlBoundaries, LowLowerControlOperandTest,
    ::testing::Values(ControlBoundary::kBranch, ControlBoundary::kReturn),
    [](const ::testing::TestParamInfo<ControlBoundary>& info) {
      return info.param == ControlBoundary::kBranch ? "Branch" : "Return";
    });

TEST_F(LowLowerSourcePlanTest, InheritedCarrierTracksLaterProducerSelection) {
  const loom_op_t* dependency = observer_.expected_source_ops[1];
  const loom_value_id_t producer = loom_scalar_addi_result(dependency);
  const loom_op_t* consumer = observer_.expected_source_ops[2];
  const loom_value_id_t first = loom_op_const_operands(consumer)[0];
  const loom_type_t source_type = loom_module_value_type(module_, first);
  loom_builder_t builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_func_like_body(function_)), &builder);
  loom_builder_set_before(&builder, const_cast<loom_op_t*>(consumer));
  loom_op_t* second_op = nullptr;
  IREE_ASSERT_OK(loom_scalar_assume_build(&builder, &first, 1, nullptr, 0,
                                          &source_type, 1,
                                          LOOM_LOCATION_UNKNOWN, &second_op));
  const loom_value_id_t second = loom_op_const_results(second_op)[0];

  loom_low_lower_context_t lowering = {};
  lowering.module = module_;
  loom_low_lowering_frame_t frame = {};
  lowering.lowering = &frame;
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module_, loom_func_like_body(function_), &analysis_arena_,
      &frame.value_domain));
  std::vector<loom_low_lower_value_binding_t> bindings(
      frame.value_domain.value_count);
  std::vector<loom_low_lower_value_flags_t> flags(bindings.size(), 0);
  for (auto& binding : bindings) {
    binding.type = LOOM_TYPE_ID_INVALID;
  }
  frame.value_bindings = bindings.data();
  frame.source_plan.value_flags = flags.data();

  // Structural planning runs before the producing rule finalizes its carrier.
  loom_low_lower_inherit_value_type(&lowering, producer, first);
  loom_low_lower_inherit_value_type(&lowering, first, second);
  const loom_type_t native_type =
      loom_low_register_type(loom_test_low_core_descriptor_set()->stable_id,
                             TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  IREE_EXPECT_OK(
      loom_low_lower_plan_value_type(&lowering, producer, native_type));
  EXPECT_TRUE(loom_type_equal(
      loom_low_lower_value_binding_type(&lowering, second), native_type));
  const auto second_ordinal =
      loom_local_value_domain_ordinal(&frame.value_domain, second);
  EXPECT_EQ(bindings[second_ordinal].type_source,
            loom_local_value_domain_ordinal(&frame.value_domain, producer));
  EXPECT_FALSE(loom_low_lower_source_value_has_low_mapping(&lowering, second));
  loom_local_value_domain_release(&frame.value_domain);
}

TEST_F(LowLowerSourcePlanTest,
       PlansSurviveOtherFunctionsAndAnalysisRetirement) {
  // Two independent plans share the module's single ordinal scratch borrow.
  // Execute in reverse order after both fact scopes have been released.
  policy_.emit_preamble = {};
  const loom_func_like_t first_function = function_;
  const loom_symbol_ref_t first_symbol = loom_func_like_callee(first_function);
  iree_arena_allocator_t plan_arena;
  iree_arena_initialize(&block_pool_, &plan_arena);
  const auto observe_arena =
      +[](void* user_data, loom_low_lower_context_t* context) {
        EXPECT_EQ(loom_low_lower_context_function_arena(context), user_data);
        return iree_ok_status();
      };
  policy_.entry_setup = {observe_arena, observe_arena, &plan_arena};
  loom_low_lower_function_plan_t* first_plan = nullptr;
  IREE_ASSERT_OK(loom_low_lower_plan_function(
      module_, first_function, &options_, &plan_arena, &result_, &first_plan));
  ASSERT_NE(first_plan, nullptr);
  EXPECT_EQ(module_->symbols.entries[first_symbol.symbol_id].defining_op,
            first_function.op);

  BuildFunction(IREE_SV("second"));
  iree_arena_reset(&analysis_arena_);
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &fact_table_, &analysis_arena_, module_->values.count));
  fact_table_.context.target_facts = &target_facts_;
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table_, module_, function_));
  const loom_symbol_ref_t second_symbol = loom_func_like_callee(function_);
  const iree_host_size_t value_count = module_->values.count;
  loom_low_lower_result_t second_result = {};
  loom_low_lower_function_plan_t* second_plan = nullptr;
  IREE_ASSERT_OK(loom_low_lower_plan_function(module_, function_, &options_,
                                              &plan_arena, &second_result,
                                              &second_plan));
  ASSERT_NE(second_plan, nullptr);
  EXPECT_EQ(module_->values.count, value_count);
  EXPECT_EQ(module_->symbols.entries[second_symbol.symbol_id].defining_op,
            function_.op);
  iree_arena_reset(&analysis_arena_);

  IREE_ASSERT_OK(loom_low_lower_emit_function(second_plan, &second_result));
  IREE_ASSERT_OK(loom_low_lower_emit_function(first_plan, &result_));
  EXPECT_TRUE(loom_low_func_def_isa(
      module_->symbols.entries[first_symbol.symbol_id].defining_op));
  EXPECT_TRUE(loom_low_func_def_isa(
      module_->symbols.entries[second_symbol.symbol_id].defining_op));
  EXPECT_EQ(result_.error_count + second_result.error_count, 0u);
  loom_low_lower_result_deinitialize(&second_result);
  iree_arena_deinitialize(&plan_arena);
}

TEST_F(LowLowerSourcePlanTest, SkipsAccessGraphsWithoutLocalStorage) {
  bool queried = false;
  policy_.entry_setup = {
      +[](void* user_data, loom_low_lower_context_t* context) {
        *static_cast<bool*>(user_data) = true;
        EXPECT_EQ(context->storage_access->state, nullptr);
        loom_storage_interference_t* interference = nullptr;
        IREE_RETURN_IF_ERROR(loom_low_lower_context_storage_interference(
            context, &interference));
        EXPECT_NE(interference, nullptr);
        EXPECT_EQ(context->storage_access->state, nullptr);
        return iree_ok_status();
      },
      +[](void*, loom_low_lower_context_t*) { return iree_ok_status(); },
      &queried,
  };
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_TRUE(queried);
  EXPECT_EQ(result_.error_count, 0u);
}

TEST_F(LowLowerSourcePlanTest, PlansEntryResourcesAfterStorageDemand) {
  policy_.entry_setup = {PlanEntry, EmitEntry, &observer_};
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 0u);
  EXPECT_EQ(observer_.entry.planning_count, 1u);
  EXPECT_EQ(observer_.entry.emission_count, 1u);
}

TEST_F(LowLowerSourcePlanTest, RejectsEntryResourcesBeforeLowConstruction) {
  policy_.entry_setup = {PlanEntry, EmitEntry, &observer_};
  observer_.entry.reject = true;
  const iree_host_size_t initial_value_count = module_->values.count;
  const loom_region_t* source_body = loom_func_like_body(function_);
  const loom_symbol_ref_t symbol = loom_func_like_callee(function_);
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);
  EXPECT_EQ(observer_.entry.planning_count, 1u);
  EXPECT_EQ(observer_.entry.emission_count, 0u);
  EXPECT_FALSE(observer_.type_mapping.emission_started);
  EXPECT_EQ(module_->values.count, initial_value_count);
  EXPECT_EQ(loom_func_like_body(function_), source_body);
  EXPECT_EQ(module_->symbols.entries[symbol.symbol_id].defining_op,
            function_.op);
}

TEST_F(LowLowerSourcePlanTest, RejectsLaneWidthBeforeLowConstruction) {
  // A semantic aggregate can keep its width, but the selected per-lane recipe
  // has no relation for projecting that semantic type onto a scalar carrier.
  policy_.map_type.fn = [](void* user_data, loom_low_lower_context_t* context,
                           const loom_op_t* source_op, loom_type_t source_type,
                           loom_type_t* out_low_type) {
    IREE_RETURN_IF_ERROR(ObserveTypeMapping(user_data, context, source_op,
                                            source_type, out_low_type));
    if (!loom_type_is_vector(source_type)) {
      return iree_ok_status();
    }
    return loom_low_lower_make_typed_register_type(
        context, loom_low_register_type_class_id(*out_low_type),
        loom_low_register_type_unit_count(*out_low_type), source_type,
        out_low_type);
  };
  policy_.source_plan_observer = nullptr;
  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&builder, IREE_SV("lane"), &name));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol_id));
  const loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0, symbol,
      &vector_type, 1, &vector_type, 1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*predicates=*/nullptr,
      /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
  function_ = loom_func_like_cast(module_, function_op);
  loom_block_t* entry = loom_region_entry_block(loom_func_like_body(function_));
  loom_builder_set_block(&builder, entry);
  const loom_value_id_t input = loom_block_arg_id(entry, 0);
  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_vector_addi_build(&builder, 0, input, input, vector_type,
                                        LOOM_LOCATION_UNKNOWN, &add_op));
  const loom_value_id_t sum = loom_vector_addi_result(add_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &sum, 1, LOOM_LOCATION_UNKNOWN,
                                       &yield_op));
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &fact_table_, &analysis_arena_, module_->values.count));
  fact_table_.context.target_facts = &target_facts_;
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table_, module_, function_));

  const iree_host_size_t value_count = module_->values.count;
  const uint32_t op_count = entry->op_count;
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);
  EXPECT_FALSE(observer_.type_mapping.emission_started);
  EXPECT_EQ(module_->values.count, value_count);
  EXPECT_EQ(entry->op_count, op_count);
  EXPECT_EQ(module_->symbols.entries[symbol_id].defining_op, function_op);
}

enum class HelperConstraint { kPredicate, kOperandCarrier };

class LowLowerHelperPlanTest
    : public LowLowerSourcePlanTest,
      public ::testing::WithParamInterface<HelperConstraint> {};

TEST_P(LowLowerHelperPlanTest, RejectsConstraintBeforeLowConstruction) {
  // Produce the helper through the ordinary lowering interface. The new caller
  // violates either its first argument's predicate or its native carrier.
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  const loom_func_like_t helper =
      loom_func_like_cast(module_, result_.low_func_op);
  const loom_symbol_ref_t helper_ref = loom_func_like_callee(helper);
  const loom_value_id_t helper_argument =
      loom_region_entry_arg_id(loom_func_like_body(helper), 0);
  if (GetParam() == HelperConstraint::kPredicate) {
    loom_predicate_t* predicate = nullptr;
    IREE_ASSERT_OK(iree_arena_allocate(&module_->arena, sizeof(*predicate),
                                       reinterpret_cast<void**>(&predicate)));
    *predicate = {};
    predicate->kind = LOOM_PREDICATE_GE;
    predicate->arg_count = 2;
    predicate->arg_tags[0] = LOOM_PRED_ARG_VALUE;
    predicate->arg_tags[1] = LOOM_PRED_ARG_CONST;
    predicate->args[0] = helper_argument;
    IREE_ASSERT_OK(loom_op_set_attr(module_, helper.op,
                                    helper.vtable->predicates_attr_index,
                                    loom_attr_predicate_list(predicate, 1)));
  }
  loom_low_lower_result_deinitialize(&result_);
  result_ = {};
  observer_.type_mapping.emission_started = false;
  observer_.plan_count = 0;
  policy_.source_plan_observer = nullptr;
  options_.max_errors = 1;

  BuildFunction(IREE_SV("caller"));
  loom_block_t* entry = loom_region_entry_block(loom_func_like_body(function_));
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, entry, &builder);
  loom_builder_set_before(&builder, entry->last_op);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t operands[] = {entry->arg_ids[0], entry->arg_ids[1]};
  if (GetParam() == HelperConstraint::kOperandCarrier) {
    loom_op_t* bitcast = nullptr;
    IREE_ASSERT_OK(loom_scalar_bitcast_build(
        &builder, operands[0], i32_type, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
        LOOM_LOCATION_UNKNOWN, &bitcast));
    operands[0] = loom_scalar_bitcast_result(bitcast);
  }
  loom_op_t* invoke = nullptr;
  IREE_ASSERT_OK(loom_low_invoke_build(
      &builder, /*build_flags=*/0, /*purity=*/0, /*inline_policy=*/0,
      helper_ref, operands, IREE_ARRAYSIZE(operands), &i32_type, 1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &invoke));
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &fact_table_, &analysis_arena_, module_->values.count));
  fact_table_.context.target_facts = &target_facts_;
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table_, module_, function_));

  const iree_host_size_t value_count = module_->values.count;
  const uint32_t op_count = entry->op_count;
  const loom_symbol_ref_t caller_ref = loom_func_like_callee(function_);
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  EXPECT_EQ(result_.error_count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);
  EXPECT_FALSE(observer_.type_mapping.emission_started);
  EXPECT_EQ(module_->values.count, value_count);
  EXPECT_EQ(entry->op_count, op_count);
  EXPECT_EQ(module_->symbols.entries[caller_ref.symbol_id].defining_op,
            function_.op);
  EXPECT_EQ(module_->symbols.entries[helper_ref.symbol_id].defining_op,
            helper.op);
}

INSTANTIATE_TEST_SUITE_P(
    HelperConstraints, LowLowerHelperPlanTest,
    ::testing::Values(HelperConstraint::kPredicate,
                      HelperConstraint::kOperandCarrier),
    [](const ::testing::TestParamInfo<HelperConstraint>& info) {
      return info.param == HelperConstraint::kPredicate ? "Predicate"
                                                        : "OperandCarrier";
    });

TEST_F(LowLowerSourcePlanTest,
       LowersNonFuncDialectCallableBoundaryThroughInterfaces) {
  policy_.source_plan_observer = nullptr;

  loom_builder_t module_builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &module_builder);
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_string_id_t callee_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &module_builder, IREE_SV("test_callee"), &callee_name_id));
  uint16_t callee_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module_, callee_name_id, &callee_symbol_id));
  const loom_symbol_ref_t callee_ref = {
      .module_id = 0,
      .symbol_id = callee_symbol_id,
  };
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* declaration_op = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      callee_ref, &i32_type, 1, &i32_type, 1, &tied_result, 1,
      LOOM_LOCATION_UNKNOWN, &declaration_op));

  loom_string_id_t caller_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &module_builder, IREE_SV("test_caller"), &caller_name_id));
  uint16_t caller_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(module_, caller_name_id, &caller_symbol_id));
  const loom_symbol_ref_t caller_ref = {
      .module_id = 0,
      .symbol_id = caller_symbol_id,
  };
  loom_op_t* caller_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      caller_ref, &i32_type, 1, &i32_type, 1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*predicates=*/nullptr,
      /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &caller_op));
  function_ = loom_func_like_cast(module_, caller_op);
  loom_region_t* caller_body = loom_func_like_body(function_);
  loom_block_t* caller_entry = loom_region_entry_block(caller_body);
  loom_builder_t body_builder;
  loom_builder_initialize(module_, &module_->arena, caller_entry,
                          &body_builder);
  body_builder.ip.parent_op = caller_op;
  const loom_value_id_t argument = loom_block_arg_id(caller_entry, 0);
  loom_op_t* invoke_op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(&body_builder, callee_ref, &argument, 1,
                                        &i32_type, 1, &tied_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &invoke_op));
  const loom_value_id_t invoke_result = loom_op_const_results(invoke_op)[0];
  loom_op_t* exit_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&body_builder, &invoke_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &exit_op));

  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &fact_table_, &analysis_arena_, module_->values.count));
  fact_table_.context.target_facts = &target_facts_;
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&fact_table_, module_, function_));
  options_.fact_table = &fact_table_;

  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));
  ASSERT_EQ(result_.error_count, 0u);
  ASSERT_NE(result_.low_func_op, nullptr);
  EXPECT_GT(observer_.type_mapping.planning_queries, 0u);
  EXPECT_EQ(observer_.type_mapping.emission_queries, 0u);
  loom_region_t* low_body =
      loom_func_like_body(loom_func_like_cast(module_, result_.low_func_op));
  ASSERT_NE(low_body, nullptr);
  loom_block_t* low_entry = loom_region_entry_block(low_body);
  ASSERT_EQ(low_entry->op_count, 2u);
  const loom_op_t* low_call = loom_block_const_op(low_entry, 0);
  EXPECT_TRUE(loom_low_func_call_isa(low_call));
  ASSERT_EQ(low_call->tied_result_count, 1u);
  EXPECT_EQ(loom_op_tied_results(low_call)[0].result_index, 0u);
  EXPECT_EQ(loom_op_tied_results(low_call)[0].operand_index, 0u);
  EXPECT_TRUE(loom_low_return_isa(loom_block_const_op(low_entry, 1)));
}

}  // namespace
}  // namespace loom
