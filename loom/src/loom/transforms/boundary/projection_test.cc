// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/transforms/boundary/projection_driver.h"
#include "loom/transforms/boundary/projection_plan.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

static iree_status_t PlanZeroComponentSlot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  *out_schema = {};
  *out_claimed = false;
  loom_region_t* body = loom_func_like_body(function->function);
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT || !body || !block ||
      block == loom_region_entry_block(body) ||
      !loom_type_equal(loom_module_value_type(plan->module, value_id),
                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return iree_ok_status();
  }
  *out_schema = {
      /*.rule=*/rule,
      /*.component_types=*/nullptr,
      /*.component_name_suffixes=*/nullptr,
      /*.component_count=*/0,
      /*.rule_plan=*/nullptr,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static iree_status_t PlanZeroComponentSource(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  (void)plan;
  (void)function;
  (void)destination;
  (void)source_value_id;
  IREE_ASSERT(schema->rule == rule);
  *out_source = {
      /*.rule=*/rule,
      /*.rule_plan=*/nullptr,
      /*.boundary_op=*/boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t MaterializeZeroComponentSource(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  (void)plan;
  (void)function;
  IREE_ASSERT(source->rule == rule);
  IREE_ASSERT(out_component_values == nullptr);
  return iree_ok_status();
}

static iree_status_t ReconstructZeroComponentSlot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_ASSERT(slot->schema.rule == rule);
  IREE_ASSERT_EQ(slot->schema.component_count, 0);
  loom_op_t* constant = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(&plan->rewriter.builder,
                                                 loom_attr_i64(0), logical_type,
                                                 location, &constant));
  *out_logical_value = loom_index_constant_result(constant);
  loom_boundary_projection_record(plan, rule, 1, 0);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kZeroComponentRule = {
    /*.name=*/IREE_SVL("test-zero-component"),
    /*.type_kind_bits=*/
    LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR),
    /*.function_applies=*/nullptr,
    /*.initialize=*/nullptr,
    /*.prepare_function=*/nullptr,
    /*.plan_slot=*/PlanZeroComponentSlot,
    /*.transport=*/
    {
        /*.plan_source=*/PlanZeroComponentSource,
        /*.materialize_source=*/MaterializeZeroComponentSource,
        /*.reconstruct=*/ReconstructZeroComponentSlot,
    },
};

class BoundaryProjectionTest : public ::testing::Test {
 protected:
  using VTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
  }

  void TearDown() override {
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
    return {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_builder_t BlockBuilder(loom_op_t* function_op, loom_block_t* block) {
    loom_builder_t builder = {};
    loom_builder_initialize(module_, &module_->arena, block, &builder);
    builder.ip.parent_op = function_op;
    return builder;
  }

  void Verify(loom_module_t* module) {
    const loom_verify_options_t options = {
        /*.sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
};

TEST_F(BoundaryProjectionTest, SupportsZeroComponentCfgProjection) {
  const loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* function_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      MakeSymbol(IREE_SV("project_zero")), &i1_type, /*arg_types_count=*/1,
      /*result_types=*/&index_type, /*result_count=*/1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0,
      /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
      &function_op));
  loom_func_like_t function = loom_func_like_cast(module_, function_op);
  ASSERT_TRUE(loom_func_like_isa(function));
  loom_region_t* body = loom_func_like_body(function);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_block_t* left_block = nullptr;
  loom_block_t* right_block = nullptr;
  loom_block_t* projected_block = nullptr;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &left_block));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &right_block));
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &projected_block));

  loom_value_id_t projected_argument = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_define_value(module_, index_type, &projected_argument));
  IREE_ASSERT_OK(
      loom_block_add_arg(module_, projected_block, projected_argument));

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 1);
  loom_builder_t entry_builder = BlockBuilder(function_op, entry_block);
  loom_op_t* condition_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_cond_br_build(
      &entry_builder, arguments[0], left_block, right_block,
      LOOM_LOCATION_UNKNOWN, &condition_branch));

  loom_builder_t left_builder = BlockBuilder(function_op, left_block);
  loom_op_t* left_constant = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&left_builder, loom_attr_i64(1),
                                           index_type, LOOM_LOCATION_UNKNOWN,
                                           &left_constant));
  const loom_value_id_t left_value = loom_index_constant_result(left_constant);
  loom_op_t* left_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(&left_builder, projected_block, &left_value,
                                   /*args_count=*/1, LOOM_LOCATION_UNKNOWN,
                                   &left_branch));

  loom_builder_t right_builder = BlockBuilder(function_op, right_block);
  loom_op_t* right_constant = nullptr;
  IREE_ASSERT_OK(loom_index_constant_build(&right_builder, loom_attr_i64(2),
                                           index_type, LOOM_LOCATION_UNKNOWN,
                                           &right_constant));
  const loom_value_id_t right_value =
      loom_index_constant_result(right_constant);
  loom_op_t* right_branch = nullptr;
  IREE_ASSERT_OK(loom_cfg_br_build(
      &right_builder, projected_block, &right_value,
      /*args_count=*/1, LOOM_LOCATION_UNKNOWN, &right_branch));

  loom_builder_t join_builder = BlockBuilder(function_op, projected_block);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&join_builder, &projected_argument,
                                       /*values_count=*/1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));
  Verify(module_);

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
  const loom_pass_info_t pass_info = {
      /*.name=*/IREE_SVL("test-boundary-projection"),
      /*.description=*/IREE_SVL("Test boundary projection."),
      /*.kind=*/LOOM_PASS_MODULE,
  };
  loom_pass_t pass = {};
  pass.info = &pass_info;
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  pass.value_facts = &value_facts;
  const loom_boundary_projection_rule_t* rules[] = {&kZeroComponentRule};
  loom_boundary_projection_statistics_t statistics = {};
  IREE_ASSERT_OK(loom_boundary_projection_run(
      &pass, module_, /*version_list=*/nullptr,
      {/*.values=*/rules, /*.count=*/IREE_ARRAYSIZE(rules)}, &statistics));

  EXPECT_EQ(projected_block->arg_count, 0);
  ASSERT_EQ(return_op->operand_count, 1);
  const loom_value_id_t returned_value = loom_op_const_operands(return_op)[0];
  EXPECT_NE(returned_value, projected_argument);
  EXPECT_TRUE(loom_index_constant_isa(
      loom_value_def_op(loom_module_value(module_, returned_value))));
  ASSERT_EQ(statistics.rule_count, 1u);
  EXPECT_EQ(statistics.rules[0].projections, 1);
  EXPECT_EQ(statistics.rules[0].components, 0);
  Verify(module_);

  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&pass_arena);
}

}  // namespace
}  // namespace loom
