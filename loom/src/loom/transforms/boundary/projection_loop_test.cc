// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/transforms/boundary/projection_driver.h"
#include "loom/transforms/boundary/projection_plan.h"
#include "loom/util/walk.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

typedef struct TestSourcePlan {
  // Original semantic value used when no projected producer is available.
  loom_value_id_t value_id;
  // Projected producer candidate, or IREE_HOST_SIZE_MAX.
  iree_host_size_t dependency;
} TestSourcePlan;

static const bool kRejectThirteen = true;

static bool MatchesF32LoopState(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  (void)function;
  (void)block;
  return role == LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE &&
         loom_type_equal(loom_module_value_type(plan->module, value_id),
                         loom_type_scalar(LOOM_SCALAR_TYPE_F32));
}

static bool MatchesIndexLoopState(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  (void)function;
  (void)block;
  return role == LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE &&
         loom_type_equal(loom_module_value_type(plan->module, value_id),
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
}

static iree_status_t PlanDuplicateLoopState(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)function;
  (void)block;
  *out_schema = {};
  *out_claimed = false;
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE) {
    return iree_ok_status();
  }
  loom_type_t* component_types = nullptr;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, 2, sizeof(*component_types), (void**)&component_types));
  component_types[0] = loom_module_value_type(plan->module, value_id);
  component_types[1] = component_types[0];
  iree_string_view_t* suffixes = nullptr;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, 2, sizeof(*suffixes), (void**)&suffixes));
  suffixes[0] = IREE_SV("first");
  suffixes[1] = IREE_SV("second");
  *out_schema = {
      .rule = rule,
      .component_types = component_types,
      .component_name_suffixes = suffixes,
      .component_count = 2,
      .destination_mode = LOOM_BOUNDARY_PROJECTION_DESTINATION_RECONSTRUCT,
      .rule_plan = nullptr,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static iree_status_t PlanRejectingDuplicateLoopState(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  IREE_RETURN_IF_ERROR(PlanDuplicateLoopState(
      rule, plan, function, role, value_id, block, out_schema, out_claimed));
  if (*out_claimed) {
    out_schema->rule_plan = (void*)&kRejectThirteen;
  }
  return iree_ok_status();
}

static iree_status_t PlanEliminativeDuplicateLoopState(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  IREE_RETURN_IF_ERROR(PlanDuplicateLoopState(
      rule, plan, function, role, value_id, block, out_schema, out_claimed));
  if (*out_claimed) {
    out_schema->destination_mode =
        LOOM_BOUNDARY_PROJECTION_DESTINATION_ELIMINATE;
  }
  return iree_ok_status();
}

static iree_status_t PlanZeroComponentLoopState(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)plan;
  (void)function;
  (void)value_id;
  (void)block;
  *out_schema = {};
  *out_claimed = role == LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE;
  if (*out_claimed) {
    *out_schema = {
        .rule = rule,
        .component_types = nullptr,
        .component_name_suffixes = nullptr,
        .component_count = 0,
        .destination_mode = LOOM_BOUNDARY_PROJECTION_DESTINATION_RECONSTRUCT,
        .rule_plan = nullptr,
    };
  }
  return iree_ok_status();
}

static bool IsRejectedConstant(const loom_boundary_projection_plan_t* plan,
                               loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  return loom_index_constant_isa(op) &&
         loom_attr_as_i64(loom_index_constant_value(op)) == 13;
}

static iree_status_t PlanTestSource(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  *out_source = {};
  *out_planned = false;
  IREE_ASSERT(schema->rule == rule);
  if (schema->rule_plan == &kRejectThirteen &&
      IsRejectedConstant(plan, source_value_id)) {
    return iree_ok_status();
  }
  TestSourcePlan* source_plan = nullptr;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  *source_plan = {
      /*.value_id=*/source_value_id,
      /*.dependency=*/IREE_HOST_SIZE_MAX,
  };
  const loom_value_t* source_value =
      loom_module_value(plan->module, source_value_id);
  if (!loom_value_is_block_arg(source_value)) {
    const iree_host_size_t dependency =
        loom_boundary_projection_slot_index(function, source_value_id);
    if (dependency != IREE_HOST_SIZE_MAX &&
        function->candidates[dependency].selected &&
        function->candidates[dependency].schema.rule == rule &&
        function->candidates[dependency].schema.component_count ==
            schema->component_count) {
      source_plan->dependency = dependency;
      const iree_host_size_t destination_index =
          loom_boundary_projection_slot_index(function, destination->value_id);
      IREE_ASSERT_NE(destination_index, IREE_HOST_SIZE_MAX);
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, dependency, destination_index,
          /*orders_realization=*/false));
    }
  }
  *out_source = {
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t MaterializeTestSource(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(source->rule == rule);
  const auto* source_plan =
      static_cast<const TestSourcePlan*>(source->rule_plan);
  IREE_ASSERT(source_plan != nullptr);
  if (source_plan->dependency != IREE_HOST_SIZE_MAX) {
    const loom_boundary_projection_slot_t* producer =
        &function->candidates[source_plan->dependency];
    for (uint16_t i = 0; i < producer->schema.component_count; ++i) {
      IREE_ASSERT_NE(producer->component_value_ids[i], LOOM_VALUE_ID_INVALID);
      out_component_values[i] = producer->component_value_ids[i];
    }
    return iree_ok_status();
  }
  if (!out_component_values) {
    return iree_ok_status();
  }
  const loom_value_id_t source_value_id =
      loom_boundary_projection_resolve_value(function, source_plan->value_id);
  // Give each synthetic component its own SSA identity. This models real
  // projections whose physical components may carry separate ownership ties.
  const loom_type_t type =
      loom_module_value_type(plan->module, source_value_id);
  out_component_values[0] = source_value_id;
  loom_op_t* convert = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_test_convert_build(&plan->rewriter.builder, source_value_id, type,
                              source->boundary_op->location, &convert));
  out_component_values[1] = loom_test_convert_result(convert);
  return iree_ok_status();
}

static iree_status_t ReconstructFromFirstComponent(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  (void)logical_type;
  (void)location;
  IREE_ASSERT(slot->schema.rule == rule);
  IREE_ASSERT_EQ(slot->schema.component_count, 2);
  *out_logical_value = slot->component_value_ids[0];
  loom_boundary_projection_record(plan, rule, 1, 2);
  return iree_ok_status();
}

static iree_status_t ReconstructZeroComponent(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_ASSERT(slot->schema.rule == rule);
  IREE_ASSERT_EQ(slot->schema.component_count, 0);
  loom_op_t* constant = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_test_constant_build(&plan->rewriter.builder, loom_attr_f64(0.0),
                               logical_type, location, &constant));
  *out_logical_value = loom_test_constant_result(constant);
  loom_boundary_projection_record(plan, rule, 1, 0);
  return iree_ok_status();
}

static iree_status_t EliminateWithFirstComponent(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location) {
  (void)function;
  (void)logical_type;
  (void)location;
  IREE_ASSERT(slot->schema.rule == rule);
  IREE_ASSERT_EQ(slot->schema.component_count, 2);
  const int64_t use_count =
      (int64_t)loom_module_value(plan->module, slot->value_id)->use_count;
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      &plan->rewriter, slot->value_id, slot->component_value_ids[0]));
  loom_boundary_projection_record(plan, rule, 1, 2);
  loom_boundary_projection_record_destination_uses(plan, rule, use_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kDuplicateF32Rule = {
    /*.name=*/IREE_SVL("test-duplicate-f32-loop-state"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.slot_role_bits=*/
    LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    /*.function_applies=*/nullptr,
    /*.slot_matches=*/MatchesF32LoopState,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanDuplicateLoopState,
    /*.transport=*/
    {
        .plan_source = PlanTestSource,
        .materialize_source = MaterializeTestSource,
        .reconstruct = ReconstructFromFirstComponent,
        .eliminate = nullptr,
    },
};

static const loom_boundary_projection_rule_t kDuplicateIndexRule = {
    /*.name=*/IREE_SVL("test-duplicate-index-loop-state"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.slot_role_bits=*/
    LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    /*.function_applies=*/nullptr,
    /*.slot_matches=*/MatchesIndexLoopState,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanDuplicateLoopState,
    /*.transport=*/
    {
        .plan_source = PlanTestSource,
        .materialize_source = MaterializeTestSource,
        .reconstruct = ReconstructFromFirstComponent,
        .eliminate = nullptr,
    },
};

static const loom_boundary_projection_rule_t kEliminativeF32Rule = {
    /*.name=*/IREE_SVL("test-eliminate-f32-loop-state"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.slot_role_bits=*/
    LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    /*.function_applies=*/nullptr,
    /*.slot_matches=*/MatchesF32LoopState,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanEliminativeDuplicateLoopState,
    /*.transport=*/
    {
        .plan_source = PlanTestSource,
        .materialize_source = MaterializeTestSource,
        .reconstruct = nullptr,
        .eliminate = EliminateWithFirstComponent,
    },
};

static const loom_boundary_projection_rule_t kZeroF32Rule = {
    /*.name=*/IREE_SVL("test-zero-f32-loop-state"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.slot_role_bits=*/
    LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    /*.function_applies=*/nullptr,
    /*.slot_matches=*/MatchesF32LoopState,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanZeroComponentLoopState,
    /*.transport=*/
    {
        .plan_source = PlanTestSource,
        .materialize_source = MaterializeTestSource,
        .reconstruct = ReconstructZeroComponent,
        .eliminate = nullptr,
    },
};

static const loom_boundary_projection_rule_t kRejectingIndexRule = {
    /*.name=*/IREE_SVL("test-reject-index-loop-state"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.slot_role_bits=*/
    LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    /*.function_applies=*/nullptr,
    /*.slot_matches=*/MatchesIndexLoopState,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanRejectingDuplicateLoopState,
    /*.transport=*/
    {
        .plan_source = PlanTestSource,
        .materialize_source = MaterializeTestSource,
        .reconstruct = ReconstructFromFirstComponent,
        .eliminate = nullptr,
    },
};

static const loom_pass_info_t kProjectionPassInfo = {
    /*.name=*/IREE_SVL("test-loop-boundary-projection"),
    /*.description=*/IREE_SVL("Test LoopLike boundary projection."),
    /*.kind=*/LOOM_PASS_MODULE,
};

class LoopBoundaryProjectionTest : public ::testing::Test {
 protected:
  using VTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
    iree_arena_initialize(&block_pool_, &pass_arena_);
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts_);
    pass_.info = &kProjectionPassInfo;
    pass_.instance_arena = &pass_arena_;
    pass_.arena = &pass_arena_;
    pass_.value_facts = &value_facts_;
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&value_facts_);
    iree_arena_deinitialize(&pass_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(loom_dialect_id_t dialect_id, VTableFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t MakeSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&module_builder_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {.module_id = 0, .symbol_id = symbol_id};
  }

  loom_func_like_t BuildFunction(iree_string_view_t name,
                                 const loom_type_t* result_types,
                                 uint16_t result_count) {
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_test_func_build(
        &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        MakeSymbol(name), /*arg_types=*/nullptr, /*arg_types_count=*/0,
        result_types, result_count, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
    loom_func_like_t function = loom_func_like_cast(module_, function_op);
    EXPECT_TRUE(loom_func_like_isa(function));
    return function;
  }

  loom_builder_t FunctionBuilder(loom_func_like_t function) {
    loom_builder_t builder = {};
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function)), &builder);
    builder.ip.parent_op = function.op;
    return builder;
  }

  loom_value_id_t Constant(loom_builder_t* builder, loom_attribute_t value,
                           loom_type_t type) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(builder, value, type,
                                           LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  }

  loom_value_id_t Index(loom_builder_t* builder, int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  loom_op_t* TestYield(loom_builder_t* builder, const loom_value_id_t* values,
                       uint16_t value_count) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_yield_build(builder, values, value_count,
                                        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void ScfYield(loom_builder_t* builder, const loom_value_id_t* values,
                uint16_t value_count) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_scf_yield_build(builder, values, value_count,
                                        LOOM_LOCATION_UNKNOWN, &op));
  }

  void Project(const loom_boundary_projection_rule_t* rule,
               loom_boundary_projection_statistics_t* out_statistics) {
    const loom_boundary_projection_rule_t* rules[] = {rule};
    IREE_ASSERT_OK(loom_boundary_projection_run(
        &pass_, module_, /*version_list=*/nullptr,
        {/*.values=*/rules, /*.count=*/IREE_ARRAYSIZE(rules)},
        /*plan_sink=*/nullptr, out_statistics));
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
        .max_errors = 20,
    };
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  typedef struct LoopCollector {
    const loom_module_t* module;
    std::vector<loom_op_t*>* loops;
  } LoopCollector;

  static iree_status_t CollectLoopWithModule(void* user_data, loom_op_t* op,
                                             const loom_walk_context_t* context,
                                             loom_walk_result_t* out_result) {
    (void)context;
    *out_result = LOOM_WALK_CONTINUE;
    auto* collector = static_cast<LoopCollector*>(user_data);
    if (loom_loop_like_isa(loom_loop_like_cast(collector->module, op))) {
      collector->loops->push_back(op);
    }
    return iree_ok_status();
  }

  std::vector<loom_op_t*> CollectLoops(loom_func_like_t function) {
    std::vector<loom_op_t*> loops;
    LoopCollector collector = {.module = module_, .loops = &loops};
    loom_walk_result_t result = LOOM_WALK_CONTINUE;
    IREE_CHECK_OK(loom_walk_function(
        module_, function, LOOM_WALK_POST_ORDER,
        {/*.fn=*/CollectLoopWithModule, /*.user_data=*/&collector}, &result));
    return loops;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
  iree_arena_allocator_t pass_arena_ = {};
  loom_pass_value_fact_owner_t value_facts_ = {};
  loom_pass_t pass_ = {};
};

TEST_F(LoopBoundaryProjectionTest,
       RebuildsSyntheticCountedLoopThroughInterface) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_func_like_t function = BuildFunction(IREE_SV("counted"), &f32, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 8);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Constant(&builder, loom_attr_f64(1.0), f32);
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 3,
      .has_type_change = false,
  };
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder, lower, upper, step, &initial, 1,
                                      &f32, &tie, 1, LOOM_LOCATION_UNKNOWN,
                                      &source_loop));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_test_loop_body(source_loop));
  loom_op_t* neg = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder,
                                     loom_block_arg_id(source_body, 1), f32,
                                     LOOM_LOCATION_UNKNOWN, &neg));
  const loom_value_id_t negated = loom_test_neg_result(neg);
  loom_op_t* source_terminator = TestYield(&builder, &negated, 1);
  source_terminator->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  const iree_string_view_t terminator_comments[] = {
      IREE_SV("recurrence comment"),
  };
  IREE_ASSERT_OK(loom_module_attach_op_comments(
      module_, source_terminator, terminator_comments,
      IREE_ARRAYSIZE(terminator_comments)));
  loom_builder_restore(&builder, saved);
  loom_op_t* return_op = nullptr;
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &source_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateF32Rule, &statistics);

  EXPECT_TRUE(iree_any_bit_set(source_loop->flags, LOOM_OP_FLAG_DEAD));
  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 1u);
  loom_op_t* target_loop = loops[0];
  EXPECT_TRUE(loom_test_loop_isa(target_loop));
  EXPECT_EQ(target_loop->result_count, 2u);
  EXPECT_EQ(loom_test_loop_iter_args(target_loop).count, 2u);
  EXPECT_EQ(target_loop->tied_result_count, 2u);
  loom_block_t* target_body =
      loom_region_entry_block(loom_test_loop_body(target_loop));
  EXPECT_EQ(target_body->arg_count, 3u);
  EXPECT_EQ(loom_test_neg_input(neg), loom_block_arg_id(target_body, 1));
  ASSERT_EQ(target_body->last_op->operand_count, 2u);
  const loom_value_id_t first_yield =
      loom_op_const_operands(target_body->last_op)[0];
  const loom_value_id_t second_yield =
      loom_op_const_operands(target_body->last_op)[1];
  EXPECT_NE(first_yield, second_yield);
  EXPECT_EQ(first_yield, loom_test_neg_result(neg));
  EXPECT_EQ(loom_test_convert_input(
                loom_value_def_op(loom_module_value(module_, second_yield))),
            loom_test_neg_result(neg));
  EXPECT_TRUE(iree_any_bit_set(target_body->last_op->flags,
                               LOOM_OP_FLAG_LEADING_BLANK_LINE));
  iree_host_size_t comment_count = 0;
  EXPECT_NE(
      loom_module_op_comments(module_, target_body->last_op, &comment_count),
      nullptr);
  EXPECT_EQ(comment_count, IREE_ARRAYSIZE(terminator_comments));
  EXPECT_EQ(loom_op_const_operands(return_op)[0],
            loom_op_results(target_loop)[0]);
  EXPECT_EQ(statistics.loops_rewritten, 1);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, RebuildsConditionLoopEndpoints) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_func_like_t function = BuildFunction(IREE_SV("condition"), &index, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t initial[] = {
      Index(&builder, 0),
      Index(&builder, 1),
  };
  const loom_type_t result_types[] = {index, index};
  const loom_value_id_t condition =
      Constant(&builder, loom_attr_bool(true), i1);
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder, initial, IREE_ARRAYSIZE(initial), result_types, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* before =
      loom_region_entry_block(loom_scf_while_before(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_scf_while_before(source_loop));
  const loom_value_id_t condition_forwarded[] = {
      loom_block_arg_id(before, 1),
      loom_block_arg_id(before, 0),
  };
  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(
      loom_scf_condition_build(&builder, condition, condition_forwarded,
                               IREE_ARRAYSIZE(condition_forwarded),
                               LOOM_LOCATION_UNKNOWN, &condition_op));
  loom_builder_restore(&builder, saved);
  loom_block_t* after =
      loom_region_entry_block(loom_scf_while_after(source_loop));
  saved = loom_builder_enter_region(&builder, source_loop,
                                    loom_scf_while_after(source_loop));
  const loom_value_id_t backedge[] = {
      loom_block_arg_id(after, 1),
      loom_block_arg_id(after, 0),
  };
  ScfYield(&builder, backedge, IREE_ARRAYSIZE(backedge));
  loom_builder_restore(&builder, saved);
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &source_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateIndexRule, &statistics);

  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 1u);
  loom_op_t* target_loop = loops[0];
  ASSERT_TRUE(loom_scf_while_isa(target_loop));
  EXPECT_EQ(target_loop->result_count, 4u);
  loom_block_t* target_before =
      loom_region_entry_block(loom_scf_while_before(target_loop));
  loom_block_t* target_after =
      loom_region_entry_block(loom_scf_while_after(target_loop));
  ASSERT_EQ(target_before->arg_count, 4u);
  ASSERT_EQ(target_after->arg_count, 4u);
  ASSERT_TRUE(loom_scf_condition_isa(target_before->last_op));
  const loom_value_slice_t target_forwarded =
      loom_scf_condition_forwarded(target_before->last_op);
  ASSERT_EQ(target_forwarded.count, 4u);
  EXPECT_EQ(target_forwarded.values[0], loom_block_arg_id(target_before, 2));
  EXPECT_EQ(target_forwarded.values[2], loom_block_arg_id(target_before, 0));
  const loom_value_slice_t target_backedge =
      loom_scf_yield_values(target_after->last_op);
  ASSERT_EQ(target_backedge.count, 4u);
  EXPECT_EQ(target_backedge.values[0], loom_block_arg_id(target_after, 2));
  EXPECT_EQ(target_backedge.values[2], loom_block_arg_id(target_after, 0));
  EXPECT_EQ(loom_op_const_operands(return_op)[0],
            loom_op_results(target_loop)[0]);
  EXPECT_EQ(statistics.loops_rewritten, 1);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, SupportsEliminativeLoopState) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_func_like_t function = BuildFunction(IREE_SV("eliminate"), &f32, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 4);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Constant(&builder, loom_attr_f64(2.0), f32);
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(
      loom_test_loop_build(&builder, lower, upper, step, &initial, 1, &f32,
                           /*tied_results=*/nullptr, /*tied_result_count=*/0,
                           LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_test_loop_body(source_loop));
  loom_op_t* neg = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&builder,
                                     loom_block_arg_id(source_body, 1), f32,
                                     LOOM_LOCATION_UNKNOWN, &neg));
  const loom_value_id_t negated = loom_test_neg_result(neg);
  TestYield(&builder, &negated, 1);
  loom_builder_restore(&builder, saved);
  loom_op_t* return_op = nullptr;
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &source_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kEliminativeF32Rule, &statistics);

  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 1u);
  loom_op_t* target_loop = loops[0];
  ASSERT_EQ(target_loop->result_count, 2u);
  loom_block_t* target_body =
      loom_region_entry_block(loom_test_loop_body(target_loop));
  EXPECT_EQ(loom_test_neg_input(neg), loom_block_arg_id(target_body, 1));
  EXPECT_EQ(loom_op_const_operands(return_op)[0],
            loom_op_results(target_loop)[0]);
  ASSERT_EQ(statistics.rule_count, 1u);
  EXPECT_GT(statistics.rules[0].destination_uses_rewritten, 0);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, SupportsZeroComponentLoopState) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_func_like_t function = BuildFunction(IREE_SV("zero"), &f32, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 4);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Constant(&builder, loom_attr_f64(3.0), f32);
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(
      loom_test_loop_build(&builder, lower, upper, step, &initial, 1, &f32,
                           /*tied_results=*/nullptr, /*tied_result_count=*/0,
                           LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_test_loop_body(source_loop));
  TestYield(&builder, source_body->arg_ids + 1, 1);
  loom_builder_restore(&builder, saved);
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, &source_result, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kZeroF32Rule, &statistics);

  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 1u);
  loom_op_t* target_loop = loops[0];
  EXPECT_EQ(target_loop->result_count, 0u);
  EXPECT_EQ(loom_test_loop_iter_args(target_loop).count, 0u);
  loom_block_t* target_body =
      loom_region_entry_block(loom_test_loop_body(target_loop));
  EXPECT_EQ(target_body->arg_count, 1u);
  EXPECT_EQ(target_body->last_op->operand_count, 0u);
  const loom_value_id_t returned = loom_op_const_operands(return_op)[0];
  EXPECT_TRUE(loom_test_constant_isa(
      loom_value_def_op(loom_module_value(module_, returned))));
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, RejectsUnsupportedSourceAtomically) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_func_like_t function = BuildFunction(IREE_SV("reject"), &index, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 4);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Index(&builder, 13);
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(
      loom_test_loop_build(&builder, lower, upper, step, &initial, 1, &index,
                           /*tied_results=*/nullptr, /*tied_result_count=*/0,
                           LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_test_loop_body(source_loop));
  TestYield(&builder, source_body->arg_ids + 1, 1);
  loom_builder_restore(&builder, saved);
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  TestYield(&builder, &source_result, 1);
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kRejectingIndexRule, &statistics);

  EXPECT_FALSE(iree_any_bit_set(source_loop->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(source_loop->result_count, 1u);
  EXPECT_EQ(loom_test_loop_iter_args(source_loop).count, 1u);
  EXPECT_EQ(statistics.loops_rewritten, 0);
  ASSERT_EQ(statistics.rule_count, 1u);
  EXPECT_EQ(statistics.rules[0].projections, 0);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, RejectsMismatchedOwnershipTieAtomically) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_func_like_t function = BuildFunction(IREE_SV("mismatched_tie"), &f32, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 4);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Constant(&builder, loom_attr_f64(1.0), f32);
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = true,
  };
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder, lower, upper, step, &initial, 1,
                                      &f32, &tie, 1, LOOM_LOCATION_UNKNOWN,
                                      &source_loop));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_test_loop_body(source_loop));
  TestYield(&builder, source_body->arg_ids + 1, 1);
  loom_builder_restore(&builder, saved);
  const loom_value_id_t source_result = loom_op_results(source_loop)[0];
  TestYield(&builder, &source_result, 1);
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateF32Rule, &statistics);

  EXPECT_FALSE(iree_any_bit_set(source_loop->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(source_loop->result_count, 1u);
  EXPECT_EQ(statistics.loops_rewritten, 0);
  ASSERT_EQ(statistics.rule_count, 1u);
  EXPECT_EQ(statistics.rules[0].projections, 0);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, RebuildsNestedLoopsInDominanceOrder) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_func_like_t function = BuildFunction(IREE_SV("nested"), &f32, 1);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t lower = Index(&builder, 0);
  const loom_value_id_t upper = Index(&builder, 4);
  const loom_value_id_t step = Index(&builder, 1);
  const loom_value_id_t initial = Constant(&builder, loom_attr_f64(1.0), f32);
  loom_op_t* outer = nullptr;
  IREE_ASSERT_OK(
      loom_test_loop_build(&builder, lower, upper, step, &initial, 1, &f32,
                           /*tied_results=*/nullptr, /*tied_result_count=*/0,
                           LOOM_LOCATION_UNKNOWN, &outer));
  loom_block_t* outer_body =
      loom_region_entry_block(loom_test_loop_body(outer));
  loom_builder_ip_t outer_saved =
      loom_builder_enter_region(&builder, outer, loom_test_loop_body(outer));
  const loom_value_id_t inner_initial = loom_block_arg_id(outer_body, 1);
  loom_op_t* inner = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder, lower, upper, step, &inner_initial, 1, &f32,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &inner));
  loom_block_t* inner_body =
      loom_region_entry_block(loom_test_loop_body(inner));
  loom_builder_ip_t inner_saved =
      loom_builder_enter_region(&builder, inner, loom_test_loop_body(inner));
  TestYield(&builder, inner_body->arg_ids + 1, 1);
  loom_builder_restore(&builder, inner_saved);
  const loom_value_id_t inner_result = loom_op_results(inner)[0];
  TestYield(&builder, &inner_result, 1);
  loom_builder_restore(&builder, outer_saved);
  const loom_value_id_t outer_result = loom_op_results(outer)[0];
  TestYield(&builder, &outer_result, 1);
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateF32Rule, &statistics);

  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 2u);
  EXPECT_EQ(loops[0]->result_count, 2u);
  EXPECT_EQ(loops[1]->result_count, 2u);
  EXPECT_TRUE(loops[0]->parent_op == loops[1]);
  EXPECT_EQ(statistics.loops_rewritten, 2);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, PreservesDependentRecurringTypeScheme) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_func_like_t function =
      BuildFunction(IREE_SV("dependent"), /*result_types=*/nullptr, 0);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t rows = Index(&builder, 8);
  const loom_type_t dependent_initial_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(rows), 0);
  const loom_value_id_t scalar = Constant(&builder, loom_attr_f64(1.0), f32);
  const loom_value_id_t dependent =
      Constant(&builder, loom_attr_f64(0.0), dependent_initial_type);
  const loom_value_id_t condition =
      Constant(&builder, loom_attr_bool(true), i1);
  const loom_value_id_t initial[] = {scalar, dependent, rows};
  loom_value_id_t reserved[2 * IREE_ARRAYSIZE(initial)] = {};
  IREE_ASSERT_OK(loom_builder_reserve_values(&builder, IREE_ARRAYSIZE(reserved),
                                             reserved));
  const loom_type_t header_types[] = {
      f32,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved[2]), 0),
      index,
  };
  const loom_value_id_t* reserved_results = reserved + IREE_ARRAYSIZE(initial);
  const loom_type_t result_types[] = {
      f32,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[2]), 0),
      index,
  };
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder, initial, IREE_ARRAYSIZE(initial), header_types, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* before =
      loom_region_entry_block(loom_scf_while_before(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_scf_while_before(source_loop));
  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder, condition, before->arg_ids, before->arg_count,
      LOOM_LOCATION_UNKNOWN, &condition_op));
  loom_builder_restore(&builder, saved);
  loom_block_t* after =
      loom_region_entry_block(loom_scf_while_after(source_loop));
  saved = loom_builder_enter_region(&builder, source_loop,
                                    loom_scf_while_after(source_loop));
  ScfYield(&builder, after->arg_ids, after->arg_count);
  loom_builder_restore(&builder, saved);
  TestYield(&builder, /*values=*/nullptr, /*value_count=*/0);
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateF32Rule, &statistics);

  const std::vector<loom_op_t*> loops = CollectLoops(function);
  ASSERT_EQ(loops.size(), 1u);
  loom_op_t* target_loop = loops[0];
  ASSERT_EQ(target_loop->result_count, 4u);
  EXPECT_EQ(
      loom_type_dim_value_id_at(
          loom_module_value_type(module_, loom_op_results(target_loop)[2]), 0),
      loom_op_results(target_loop)[3]);
  loom_block_t* target_before =
      loom_region_entry_block(loom_scf_while_before(target_loop));
  loom_block_t* target_after =
      loom_region_entry_block(loom_scf_while_after(target_loop));
  EXPECT_EQ(loom_type_dim_value_id_at(
                loom_module_value_type(module_, target_before->arg_ids[2]), 0),
            target_before->arg_ids[3]);
  EXPECT_EQ(loom_type_dim_value_id_at(
                loom_module_value_type(module_, target_after->arg_ids[2]), 0),
            target_after->arg_ids[3]);
  Verify();
}

TEST_F(LoopBoundaryProjectionTest, RejectsProviderOfRecurringType) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_func_like_t function =
      BuildFunction(IREE_SV("dependent_provider"), /*result_types=*/nullptr, 0);
  loom_builder_t builder = FunctionBuilder(function);
  const loom_value_id_t rows = Index(&builder, 8);
  const loom_type_t dependent_initial_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(rows), 0);
  const loom_value_id_t dependent =
      Constant(&builder, loom_attr_f64(0.0), dependent_initial_type);
  const loom_value_id_t condition =
      Constant(&builder, loom_attr_bool(true), i1);
  const loom_value_id_t initial[] = {dependent, rows};
  loom_value_id_t reserved[2 * IREE_ARRAYSIZE(initial)] = {};
  IREE_ASSERT_OK(loom_builder_reserve_values(&builder, IREE_ARRAYSIZE(reserved),
                                             reserved));
  const loom_type_t header_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved[1]), 0),
      index,
  };
  const loom_value_id_t* reserved_results = reserved + IREE_ARRAYSIZE(initial);
  const loom_type_t result_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[1]), 0),
      index,
  };
  loom_op_t* source_loop = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder, initial, IREE_ARRAYSIZE(initial), header_types, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &source_loop));
  loom_block_t* before =
      loom_region_entry_block(loom_scf_while_before(source_loop));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder, source_loop, loom_scf_while_before(source_loop));
  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder, condition, before->arg_ids, before->arg_count,
      LOOM_LOCATION_UNKNOWN, &condition_op));
  loom_builder_restore(&builder, saved);
  loom_block_t* after =
      loom_region_entry_block(loom_scf_while_after(source_loop));
  saved = loom_builder_enter_region(&builder, source_loop,
                                    loom_scf_while_after(source_loop));
  ScfYield(&builder, after->arg_ids, after->arg_count);
  loom_builder_restore(&builder, saved);
  TestYield(&builder, /*values=*/nullptr, /*value_count=*/0);
  Verify();

  loom_boundary_projection_statistics_t statistics = {};
  Project(&kDuplicateIndexRule, &statistics);

  EXPECT_FALSE(iree_any_bit_set(source_loop->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(source_loop->result_count, IREE_ARRAYSIZE(initial));
  EXPECT_EQ(statistics.loops_rewritten, 0);
  ASSERT_EQ(statistics.rule_count, 1u);
  EXPECT_EQ(statistics.rules[0].projections, 0);
  Verify();
}

}  // namespace
}  // namespace loom
