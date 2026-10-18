// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/resolve_launches.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_references.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class ResolveLaunchesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
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
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {.module_id = 0, .symbol_id = symbol_id};
  }

  loom_symbol_ref_t BuildConfiguration(
      iree_string_view_t name, loom_callable_effect_contract_t effect_contract,
      loom_type_t workload_type) {
    const loom_symbol_ref_t symbol = AddSymbol(name);
    const loom_type_t result_types[] = {
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
    };
    loom_op_t* function = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, LOOM_FUNC_DEF_BUILD_FLAG_HAS_EFFECT_CONTRACT,
        /*visibility=*/0, /*linkage=*/0, /*retain=*/0, /*cc=*/0,
        effect_contract,
        /*temperature=*/0, /*inline_policy=*/0, loom_symbol_ref_null(),
        /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), symbol, &workload_type, 1, result_types,
        IREE_ARRAYSIZE(result_types), /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function));

    loom_builder_t body_builder = module_builder_;
    loom_builder_enter_region(&body_builder, function,
                              loom_func_def_body(function));
    const loom_block_t* entry_block =
        loom_region_const_entry_block(loom_func_def_body(function));
    loom_op_t* one = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(&body_builder, loom_attr_i64(1),
                                            result_types[0],
                                            LOOM_LOCATION_UNKNOWN, &one));
    const loom_value_id_t results[] = {
        loom_type_equal(workload_type, result_types[0])
            ? entry_block->arg_ids[0]
            : loom_index_constant_result(one),
        loom_index_constant_result(one),
        loom_index_constant_result(one),
    };
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, results,
                                         IREE_ARRAYSIZE(results),
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return symbol;
  }

  loom_symbol_ref_t BuildKernel(iree_string_view_t name,
                                loom_type_t workload_type) {
    const loom_symbol_ref_t symbol = AddSymbol(name);
    loom_op_t* declaration = nullptr;
    IREE_CHECK_OK(loom_kernel_decl_build(
        &module_builder_, /*build_flags=*/0, /*retain=*/0,
        loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, /*export_linkage=*/0,
        symbol, &workload_type, 1, /*args_types=*/nullptr,
        /*args_types_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &declaration));
    return symbol;
  }

  loom_symbol_ref_t BuildProgram(loom_symbol_ref_t hoisted_pure_kernel,
                                 loom_symbol_ref_t local_pure_kernel,
                                 loom_symbol_ref_t readonly_kernel) {
    const loom_symbol_ref_t symbol = AddSymbol(IREE_SV("root"));
    const loom_type_t specialization_type =
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    const loom_type_t binding_type = loom_type_buffer();
    loom_op_t* program = nullptr;
    IREE_CHECK_OK(loom_command_program_def_build(
        &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        loom_symbol_ref_null(), symbol, &specialization_type,
        /*specializations_types_count=*/1, &binding_type,
        /*bindings_types_count=*/1, /*predicates=*/nullptr,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &program));

    loom_builder_t program_builder = module_builder_;
    loom_builder_enter_region(&program_builder, program,
                              loom_command_program_def_body(program));
    const loom_block_t* entry_block =
        loom_region_const_entry_block(loom_command_program_def_body(program));
    loom_op_t* schedule = nullptr;
    IREE_CHECK_OK(loom_kernel_launch_serial_build(
        &program_builder, LOOM_LOCATION_UNKNOWN, &schedule));
    loom_builder_t schedule_builder = program_builder;
    loom_builder_enter_region(&schedule_builder, schedule,
                              loom_kernel_launch_serial_body(schedule));
    loom_op_t* launch = nullptr;
    IREE_CHECK_OK(loom_kernel_launch_build(
        &schedule_builder, hoisted_pure_kernel, &entry_block->arg_ids[0], 1,
        /*arguments=*/nullptr, /*arguments_count=*/0, LOOM_LOCATION_UNKNOWN,
        &launch));
    loom_op_t* local_workload = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &schedule_builder, loom_attr_i64(2), specialization_type,
        LOOM_LOCATION_UNKNOWN, &local_workload));
    const loom_value_id_t local_workload_value =
        loom_index_constant_result(local_workload);
    IREE_CHECK_OK(loom_kernel_launch_build(
        &schedule_builder, local_pure_kernel, &local_workload_value, 1,
        /*arguments=*/nullptr, /*arguments_count=*/0, LOOM_LOCATION_UNKNOWN,
        &launch));
    if (loom_symbol_ref_is_valid(readonly_kernel)) {
      IREE_CHECK_OK(loom_kernel_launch_build(
          &schedule_builder, readonly_kernel, &entry_block->arg_ids[1], 1,
          /*arguments=*/nullptr, /*arguments_count=*/0, LOOM_LOCATION_UNKNOWN,
          &launch));
    }
    loom_op_t* yield_op = nullptr;
    IREE_CHECK_OK(loom_kernel_launch_yield_build(
        &schedule_builder, LOOM_LOCATION_UNKNOWN, &yield_op));
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_command_return_build(&program_builder,
                                            LOOM_LOCATION_UNKNOWN, &return_op));
    return symbol;
  }

  void Verify() {
    loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
        .max_errors = 20,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
};

TEST_F(ResolveLaunchesTest, PreservesConfigurationPlacement) {
  const loom_symbol_ref_t pure_config = BuildConfiguration(
      IREE_SV("pure_config"), LOOM_CALLABLE_EFFECT_CONTRACT_PURE,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const loom_symbol_ref_t readonly_config = BuildConfiguration(
      IREE_SV("readonly_config"), LOOM_CALLABLE_EFFECT_CONTRACT_READONLY,
      loom_type_buffer());
  const loom_symbol_ref_t pure_kernel = BuildKernel(
      IREE_SV("pure_kernel"), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const loom_symbol_ref_t local_pure_kernel = BuildKernel(
      IREE_SV("local_pure_kernel"), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const loom_symbol_ref_t readonly_kernel =
      BuildKernel(IREE_SV("readonly_kernel"), loom_type_buffer());
  const loom_symbol_ref_t root =
      BuildProgram(pure_kernel, local_pure_kernel, readonly_kernel);
  Verify();

  std::vector<loom_symbol_ref_t> configurations(module_->symbols.count,
                                                loom_symbol_ref_null());
  configurations[pure_kernel.symbol_id] = pure_config;
  configurations[local_pure_kernel.symbol_id] = pure_config;
  configurations[readonly_kernel.symbol_id] = readonly_config;
  loom_symbol_reference_table_t references = {};
  IREE_ASSERT_OK(
      loom_symbol_reference_table_build(module_, &scratch_arena_, &references));
  loom_kernel_launch_resolution_t resolution = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_kernel_resolve_launches(
      module_, &references, configurations.data(), configurations.size(),
      /*diagnostic_emitter=*/{}, &scratch_arena_, &resolution, &valid));
  ASSERT_TRUE(valid);
  EXPECT_TRUE(resolution.has_readonly_launches);

  const loom_op_t* root_op =
      module_->symbols.entries[root.symbol_id].defining_op;
  const loom_block_t* root_block =
      loom_region_const_entry_block(loom_command_program_def_body(root_op));
  const loom_op_t* pure_call = root_block->first_op;
  ASSERT_TRUE(loom_func_call_isa(pure_call));
  EXPECT_EQ(loom_func_call_effect_contract(pure_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_PURE);
  EXPECT_EQ(loom_func_call_callee(pure_call).symbol_id, pure_config.symbol_id);
  const loom_op_t* schedule = pure_call->next_op;
  ASSERT_TRUE(loom_kernel_launch_serial_isa(schedule));

  const loom_block_t* schedule_block =
      loom_region_const_entry_block(loom_kernel_launch_serial_body(schedule));
  const loom_op_t* pure_dispatch = schedule_block->first_op;
  ASSERT_TRUE(loom_kernel_dispatch_isa(pure_dispatch));
  const loom_op_t* local_workload = pure_dispatch->next_op;
  ASSERT_TRUE(loom_index_constant_isa(local_workload));
  const loom_op_t* local_pure_call = local_workload->next_op;
  ASSERT_TRUE(loom_func_call_isa(local_pure_call));
  EXPECT_EQ(loom_func_call_effect_contract(local_pure_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_PURE);
  EXPECT_EQ(loom_func_call_callee(local_pure_call).symbol_id,
            pure_config.symbol_id);
  const loom_value_slice_t local_pure_operands =
      loom_func_call_operands(local_pure_call);
  ASSERT_EQ(local_pure_operands.count, 1u);
  EXPECT_EQ(local_pure_operands.values[0],
            loom_index_constant_result(local_workload));
  const loom_op_t* local_pure_dispatch = local_pure_call->next_op;
  ASSERT_TRUE(loom_kernel_dispatch_isa(local_pure_dispatch));
  const loom_op_t* readonly_call = local_pure_dispatch->next_op;
  ASSERT_TRUE(loom_func_call_isa(readonly_call));
  EXPECT_EQ(loom_func_call_effect_contract(readonly_call),
            LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
  EXPECT_EQ(loom_func_call_callee(readonly_call).symbol_id,
            readonly_config.symbol_id);
  EXPECT_TRUE(loom_kernel_dispatch_isa(readonly_call->next_op));

  Verify();
}

TEST_F(ResolveLaunchesTest, ReportsNoReadonlyLaunches) {
  const loom_symbol_ref_t pure_config = BuildConfiguration(
      IREE_SV("pure_config"), LOOM_CALLABLE_EFFECT_CONTRACT_PURE,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const loom_symbol_ref_t pure_kernel = BuildKernel(
      IREE_SV("pure_kernel"), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  BuildProgram(pure_kernel, pure_kernel, loom_symbol_ref_null());
  Verify();

  std::vector<loom_symbol_ref_t> configurations(module_->symbols.count,
                                                loom_symbol_ref_null());
  configurations[pure_kernel.symbol_id] = pure_config;
  loom_symbol_reference_table_t references = {};
  IREE_ASSERT_OK(
      loom_symbol_reference_table_build(module_, &scratch_arena_, &references));
  loom_kernel_launch_resolution_t resolution = {};
  bool valid = false;
  IREE_ASSERT_OK(loom_kernel_resolve_launches(
      module_, &references, configurations.data(), configurations.size(),
      /*diagnostic_emitter=*/{}, &scratch_arena_, &resolution, &valid));
  ASSERT_TRUE(valid);
  EXPECT_FALSE(resolution.has_readonly_launches);

  Verify();
}

}  // namespace
}  // namespace loom
