// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/case_command_program.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class CaseCommandProgramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_COMMAND, loom_command_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("case_command_program_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return {.module_id = 0, .symbol_id = symbol_id};
  }

  loom_symbol_ref_t BuildKernel(iree_string_view_t name,
                                const loom_type_t* workload_types,
                                iree_host_size_t workload_count,
                                const loom_type_t* argument_types,
                                iree_host_size_t argument_count) {
    const loom_symbol_ref_t symbol = AddSymbol(name);
    loom_op_t* declaration = nullptr;
    IREE_CHECK_OK(loom_kernel_decl_build(
        &module_builder_, /*build_flags=*/0, /*retain=*/0,
        loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, /*export_linkage=*/0,
        symbol, workload_types, workload_count, argument_types, argument_count,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &declaration));
    return symbol;
  }

  loom_value_id_t BuildLiteral(loom_builder_t* builder, loom_attribute_t value,
                               loom_scalar_type_t scalar_type) {
    loom_op_t* literal = nullptr;
    IREE_CHECK_OK(loom_check_literal_build(builder, value,
                                           loom_type_scalar(scalar_type),
                                           LOOM_LOCATION_UNKNOWN, &literal));
    return loom_check_literal_result(literal);
  }

  loom_testbench_module_plan_t PlanModule() {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module_, nullptr, &plan_arena_, &plan));
    return plan;
  }

  void ExpectModuleValid() {
    loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
        .max_errors = 20,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0);
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t plan_arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
};

TEST_F(CaseCommandProgramTest,
       BuildsConcreteScalarsAliasesAndAuthoredSchedule) {
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_scalar_type_t scalar_types[] = {
      LOOM_SCALAR_TYPE_I1,     LOOM_SCALAR_TYPE_I8,     LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32,    LOOM_SCALAR_TYPE_I64,    LOOM_SCALAR_TYPE_OFFSET,
      LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16,
      LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  loom_type_t argument_types[2 + IREE_ARRAYSIZE(scalar_types)] = {
      loom_type_buffer(),
      loom_type_buffer(),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(scalar_types); ++i) {
    argument_types[i + 2] = loom_type_scalar(scalar_types[i]);
  }
  const loom_symbol_ref_t kernel =
      BuildKernel(IREE_SV("step"), &index_type, 1, argument_types,
                  IREE_ARRAYSIZE(argument_types));
  const loom_type_t tail_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(2), 0);
  const loom_tied_result_t tied_tail = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  const loom_symbol_ref_t identity_symbol =
      AddSymbol(IREE_SV("identity_tensor"));
  loom_op_t* identity = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      identity_symbol, &tail_type, 1, &tail_type, 1, &tied_tail, 1,
      /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
      &identity));
  const loom_builder_ip_t identity_ip = loom_builder_enter_region(
      &module_builder_, identity, loom_test_func_body(identity));
  const loom_block_t* identity_block =
      loom_region_const_entry_block(loom_test_func_body(identity));
  loom_op_t* identity_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&module_builder_,
                                       identity_block->arg_ids, 1,
                                       LOOM_LOCATION_UNKNOWN, &identity_yield));
  loom_builder_restore(&module_builder_, identity_ip);

  const loom_symbol_ref_t case_symbol = AddSymbol(IREE_SV("sample"));
  loom_op_t* case_op = nullptr;
  IREE_ASSERT_OK(loom_check_case_build(&module_builder_, /*build_flags=*/0,
                                       /*visibility=*/0, case_symbol,
                                       LOOM_LOCATION_UNKNOWN, &case_op));
  const loom_builder_ip_t module_ip = loom_builder_enter_region(
      &module_builder_, case_op, loom_check_case_body(case_op));

  // The selected size is beyond a host-sized Cartesian index. These prefix
  // parameters are not launch operands and need no generated constants.
  constexpr int64_t seed_count = int64_t{1} << (sizeof(iree_host_size_t) * 4);
  for (int i = 0; i < 2; ++i) {
    loom_op_t* seed = nullptr;
    IREE_ASSERT_OK(loom_check_param_seed_build(
        &module_builder_, /*build_flags=*/0, /*base=*/0, seed_count,
        LOOM_STRING_ID_INVALID, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        LOOM_LOCATION_UNKNOWN, &seed));
  }
  const int64_t size_choices[] = {4, 8};
  loom_op_t* size_parameter = nullptr;
  IREE_ASSERT_OK(loom_check_param_choice_build(
      &module_builder_, /*build_flags=*/0, size_choices,
      IREE_ARRAYSIZE(size_choices), LOOM_STRING_ID_INVALID, index_type,
      LOOM_LOCATION_UNKNOWN, &size_parameter));
  const loom_value_id_t size = loom_check_param_choice_result(size_parameter);
  const loom_value_id_t shape_extent =
      BuildLiteral(&module_builder_, loom_attr_i64(8), LOOM_SCALAR_TYPE_INDEX);
  const loom_attribute_t scalar_values[] = {
      loom_attr_bool(true), loom_attr_i64(-8),  loom_attr_i64(-16),
      loom_attr_i64(-32),   loom_attr_i64(-64), loom_attr_i64(24),
      loom_attr_f64(1.0),   loom_attr_f64(1.0), loom_attr_f64(1.0),
      loom_attr_f64(1.0),   loom_attr_f64(1.0), loom_attr_f64(1.0),
  };
  loom_value_id_t launch_arguments[IREE_ARRAYSIZE(argument_types)] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(scalar_types); ++i) {
    launch_arguments[i + 2] =
        BuildLiteral(&module_builder_, scalar_values[i], scalar_types[i]);
  }

  const loom_type_t root_type =
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_dynamic(shape_extent), 0);
  loom_op_t* root_fill = nullptr;
  IREE_ASSERT_OK(loom_check_generate_fill_build(
      &module_builder_, loom_attr_i64(0), root_type, LOOM_LOCATION_UNKNOWN,
      &root_fill));
  launch_arguments[0] = loom_check_generate_fill_result(root_fill);
  loom_op_t* tail_view = nullptr;
  IREE_ASSERT_OK(loom_check_tensor_view_build(
      &module_builder_, launch_arguments[0], /*byte_offset=*/8, tail_type,
      LOOM_LOCATION_UNKNOWN, &tail_view));
  const loom_value_id_t tail = loom_check_tensor_view_result(tail_view);
  loom_op_t* tail_identity = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(&module_builder_, identity_symbol,
                                        &tail, 1, &tail_type, 1, &tied_tail, 1,
                                        LOOM_LOCATION_UNKNOWN, &tail_identity));
  launch_arguments[1] = loom_test_invoke_results(tail_identity).values[0];

  loom_op_t* concurrent = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_concurrent_build(
      &module_builder_, LOOM_LOCATION_UNKNOWN, &concurrent));
  const loom_builder_ip_t concurrent_ip =
      loom_builder_enter_region(&module_builder_, concurrent,
                                loom_kernel_launch_concurrent_body(concurrent));
  loom_op_t* serial = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_serial_build(
      &module_builder_, LOOM_LOCATION_UNKNOWN, &serial));
  const loom_builder_ip_t serial_ip = loom_builder_enter_region(
      &module_builder_, serial, loom_kernel_launch_serial_body(serial));
  for (int i = 0; i < 2; ++i) {
    loom_op_t* launch = nullptr;
    IREE_ASSERT_OK(loom_kernel_launch_build(
        &module_builder_, kernel, &size, 1, launch_arguments,
        IREE_ARRAYSIZE(launch_arguments), LOOM_LOCATION_UNKNOWN, &launch));
  }
  loom_op_t* serial_yield = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_yield_build(
      &module_builder_, LOOM_LOCATION_UNKNOWN, &serial_yield));
  loom_builder_restore(&module_builder_, serial_ip);
  loom_op_t* concurrent_launch = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_build(
      &module_builder_, kernel, &size, 1, launch_arguments,
      IREE_ARRAYSIZE(launch_arguments), LOOM_LOCATION_UNKNOWN,
      &concurrent_launch));
  loom_op_t* yield = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_yield_build(&module_builder_,
                                                LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(&module_builder_, concurrent_ip);
  loom_op_t* serial_launch = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_build(
      &module_builder_, kernel, &size, 1, launch_arguments,
      IREE_ARRAYSIZE(launch_arguments), LOOM_LOCATION_UNKNOWN, &serial_launch));
  loom_op_t* case_return = nullptr;
  IREE_ASSERT_OK(loom_check_return_build(&module_builder_,
                                         LOOM_LOCATION_UNKNOWN, &case_return));
  loom_builder_restore(&module_builder_, module_ip);

  ExpectModuleValid();
  loom_testbench_module_plan_t module_plan = PlanModule();
  ASSERT_EQ(module_plan.case_count, 1u);
  ASSERT_EQ(module_plan.issue_count, 0u);
  const loom_testbench_case_plan_t& case_plan = module_plan.cases[0];
  ASSERT_EQ(case_plan.cartesian_sample_count, IREE_HOST_SIZE_MAX);
  ASSERT_EQ(module_plan.value_storage_identity_count, module_->values.count);
  const loom_testbench_storage_identity_t root_storage =
      module_plan.value_storage_identities[launch_arguments[0]];
  EXPECT_EQ(root_storage.root_value_id, launch_arguments[0]);
  EXPECT_EQ(root_storage.byte_offset, 0u);
  const loom_testbench_storage_identity_t view_storage =
      module_plan.value_storage_identities[tail];
  EXPECT_EQ(view_storage.root_value_id, launch_arguments[0]);
  EXPECT_EQ(view_storage.byte_offset, 8u);
  const loom_testbench_storage_identity_t tied_storage =
      module_plan.value_storage_identities[launch_arguments[1]];
  EXPECT_EQ(tied_storage.root_value_id, launch_arguments[0]);
  EXPECT_EQ(tied_storage.byte_offset, 8u);

  loom_testbench_case_command_program_t program = {};
  const iree_host_size_t assignments[] = {0, 0, 1};
  IREE_ASSERT_OK(loom_testbench_case_command_program_build(
      module_, &module_plan, &case_plan, {.assignments = assignments},
      &program));
  ASSERT_TRUE(loom_symbol_ref_is_valid(program.root_ref));
  ASSERT_EQ(program.binding_count, 1u);
  ASSERT_NE(program.binding_value_ids, nullptr);
  EXPECT_EQ(program.binding_value_ids[0], launch_arguments[0]);

  const loom_symbol_t& program_symbol =
      module_->symbols.entries[program.root_ref.symbol_id];
  ASSERT_TRUE(loom_command_def_isa(program_symbol.defining_op));
  const loom_func_like_t program_function =
      loom_func_like_cast(module_, program_symbol.defining_op);
  uint16_t program_argument_count = 0;
  const loom_value_id_t* program_arguments =
      loom_func_like_arg_ids(program_function, &program_argument_count);
  ASSERT_EQ(program_argument_count, 1u);
  EXPECT_TRUE(loom_type_is_buffer(
      loom_module_value_type(module_, program_arguments[0])));

  const loom_block_t* program_block = loom_region_const_entry_block(
      loom_command_def_body(program_symbol.defining_op));
  const loom_op_t* concurrent_op = nullptr;
  const loom_op_t* direct_launch = nullptr;
  const loom_op_t* body_op = nullptr;
  loom_block_for_each_op(program_block, body_op) {
    if (loom_command_concurrent_isa(body_op)) {
      ASSERT_EQ(concurrent_op, nullptr);
      concurrent_op = body_op;
    } else if (loom_kernel_launch_isa(body_op)) {
      ASSERT_EQ(direct_launch, nullptr);
      direct_launch = body_op;
    }
  }
  ASSERT_NE(concurrent_op, nullptr);
  ASSERT_NE(direct_launch, nullptr);
  const loom_op_t* serial_op = nullptr;
  iree_host_size_t concurrent_launch_count = 0;
  const loom_block_t* concurrent_block = loom_region_const_entry_block(
      loom_command_concurrent_body(concurrent_op));
  loom_block_for_each_op(concurrent_block, body_op) {
    if (loom_command_serial_isa(body_op)) {
      ASSERT_EQ(serial_op, nullptr);
      serial_op = body_op;
    } else if (loom_kernel_launch_isa(body_op)) {
      ++concurrent_launch_count;
    }
  }
  ASSERT_NE(serial_op, nullptr);
  ASSERT_EQ(concurrent_launch_count, 1u);

  iree_host_size_t serial_launch_count = 0;
  const loom_op_t* first_launch = nullptr;
  const loom_block_t* serial_block =
      loom_region_const_entry_block(loom_command_serial_body(serial_op));
  loom_block_for_each_op(serial_block, body_op) {
    if (loom_kernel_launch_isa(body_op)) {
      if (first_launch == nullptr) {
        first_launch = body_op;
      }
      ++serial_launch_count;
    }
  }
  ASSERT_EQ(serial_launch_count, 2u);
  ASSERT_NE(first_launch, nullptr);

  const loom_value_slice_t workloads =
      loom_kernel_launch_workloads(first_launch);
  ASSERT_EQ(workloads.count, 1u);
  const loom_op_t* workload_constant =
      loom_value_def_op(loom_module_value(module_, workloads.values[0]));
  ASSERT_TRUE(loom_index_constant_isa(workload_constant));
  EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(workload_constant)), 8);

  const loom_value_slice_t arguments =
      loom_kernel_launch_arguments(first_launch);
  ASSERT_EQ(arguments.count, IREE_ARRAYSIZE(argument_types));
  const loom_op_t* root_view =
      loom_value_def_op(loom_module_value(module_, arguments.values[0]));
  const loom_op_t* alias_view =
      loom_value_def_op(loom_module_value(module_, arguments.values[1]));
  ASSERT_TRUE(loom_buffer_view_isa(root_view));
  ASSERT_TRUE(loom_buffer_view_isa(alias_view));
  EXPECT_EQ(loom_buffer_view_buffer(root_view), program_arguments[0]);
  EXPECT_EQ(loom_buffer_view_buffer(alias_view), program_arguments[0]);
  const loom_type_t generated_root_type =
      loom_module_value_type(module_, arguments.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(generated_root_type, 0));
  const loom_op_t* generated_extent = loom_value_def_op(loom_module_value(
      module_, loom_type_dim_value_id_at(generated_root_type, 0)));
  ASSERT_TRUE(loom_index_constant_isa(generated_extent));
  EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(generated_extent)), 8);
  const loom_op_t* root_offset = loom_value_def_op(
      loom_module_value(module_, loom_buffer_view_byte_offset(root_view)));
  const loom_op_t* alias_offset = loom_value_def_op(
      loom_module_value(module_, loom_buffer_view_byte_offset(alias_view)));
  ASSERT_TRUE(loom_index_constant_isa(root_offset));
  ASSERT_TRUE(loom_index_constant_isa(alias_offset));
  EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(root_offset)), 0);
  EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(alias_offset)), 8);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(scalar_types); ++i) {
    const loom_type_t type =
        loom_module_value_type(module_, arguments.values[i + 2]);
    EXPECT_TRUE(loom_type_is_scalar(type));
    EXPECT_EQ(loom_type_element_type(type), scalar_types[i]);
  }
  ExpectModuleValid();
}

TEST_F(CaseCommandProgramTest, RejectsRuntimeScalarWithoutMutatingTheModule) {
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_symbol_ref_t identity_symbol = AddSymbol(IREE_SV("identity"));
  loom_op_t* identity = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &module_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      identity_symbol, &i32_type, 1, &i32_type, 1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0,
      /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
      &identity));
  const loom_builder_ip_t module_ip = loom_builder_enter_region(
      &module_builder_, identity, loom_test_func_body(identity));
  const loom_block_t* identity_block =
      loom_region_const_entry_block(loom_test_func_body(identity));
  loom_op_t* identity_yield = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&module_builder_,
                                       identity_block->arg_ids, 1,
                                       LOOM_LOCATION_UNKNOWN, &identity_yield));
  loom_builder_restore(&module_builder_, module_ip);

  const loom_symbol_ref_t kernel =
      BuildKernel(IREE_SV("consume"), nullptr, 0, &i32_type, 1);
  const loom_symbol_ref_t case_symbol = AddSymbol(IREE_SV("generated_scalar"));
  loom_op_t* case_op = nullptr;
  IREE_ASSERT_OK(loom_check_case_build(&module_builder_, /*build_flags=*/0,
                                       /*visibility=*/0, case_symbol,
                                       LOOM_LOCATION_UNKNOWN, &case_op));
  const loom_builder_ip_t case_ip = loom_builder_enter_region(
      &module_builder_, case_op, loom_check_case_body(case_op));
  const loom_value_id_t input =
      BuildLiteral(&module_builder_, loom_attr_i64(7), LOOM_SCALAR_TYPE_I32);
  loom_op_t* invoke = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &module_builder_, identity_symbol, &input, 1, &i32_type, 1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN,
      &invoke));
  const loom_value_id_t generated = loom_test_invoke_results(invoke).values[0];
  loom_op_t* launch = nullptr;
  IREE_ASSERT_OK(loom_kernel_launch_build(&module_builder_, kernel, nullptr, 0,
                                          &generated, 1, LOOM_LOCATION_UNKNOWN,
                                          &launch));
  loom_op_t* case_return = nullptr;
  IREE_ASSERT_OK(loom_check_return_build(&module_builder_,
                                         LOOM_LOCATION_UNKNOWN, &case_return));
  loom_builder_restore(&module_builder_, case_ip);

  ExpectModuleValid();
  loom_testbench_module_plan_t module_plan = PlanModule();
  ASSERT_EQ(module_plan.case_count, 1u);
  ASSERT_EQ(module_plan.issue_count, 0u);
  const iree_host_size_t symbol_count = module_->symbols.count;
  const iree_host_size_t value_count = module_->values.count;
  const loom_op_t* last_module_op = loom_module_block(module_)->last_op;

  loom_testbench_case_command_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_testbench_case_command_program_build(
                            module_, &module_plan, &module_plan.cases[0],
                            {.ordinal = 0}, &program));
  EXPECT_FALSE(loom_symbol_ref_is_valid(program.root_ref));
  EXPECT_EQ(module_->symbols.count, symbol_count);
  EXPECT_EQ(module_->values.count, value_count);
  EXPECT_EQ(loom_module_block(module_)->last_op, last_module_op);
}

}  // namespace
}  // namespace loom
