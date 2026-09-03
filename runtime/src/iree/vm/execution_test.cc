// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/execution.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"

namespace {

TEST(VMExecutionTest, AddressesDirectAndOverflowValueBanks) {
  std::array<uint64_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT> direct_arguments =
      {};
  std::array<uint64_t, 2> overflow_arguments = {};
  std::array<uint64_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT> direct_results = {};
  std::array<uint64_t, 2> overflow_results = {};
  direct_arguments.back() = 15;
  overflow_arguments[0] = 16;

  iree_vm_call_packet_t call = {};
  call.value_arguments.direct = direct_arguments.data();
  call.value_arguments.overflow = overflow_arguments.data();
  call.value_results.direct = direct_results.data();
  call.value_results.overflow = overflow_results.data();
  EXPECT_EQ(iree_vm_call_value_argument_load(&call, 15), 15u);
  EXPECT_EQ(iree_vm_call_value_argument_load(&call, 16), 16u);

  iree_vm_call_value_result_store(&call, 15, 115);
  iree_vm_call_value_result_store(&call, 16, 116);
  EXPECT_EQ(direct_results.back(), 115u);
  EXPECT_EQ(overflow_results[0], 116u);
}

TEST(VMExecutionTest, CopiesDirectAndOverflowFunctionBanks) {
  std::array<iree_vm_function_ref_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT>
      direct_arguments = {};
  std::array<iree_vm_function_ref_t, 1> overflow_arguments = {};
  std::array<iree_vm_function_ref_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT>
      direct_results = {};
  std::array<iree_vm_function_ref_t, 1> overflow_results = {};
  direct_arguments.back() = {15, 150};
  overflow_arguments[0] = {16, 160};

  iree_vm_call_packet_t call = {};
  call.function_arguments.direct = direct_arguments.data();
  call.function_arguments.overflow = overflow_arguments.data();
  call.function_results.direct = direct_results.data();
  call.function_results.overflow = overflow_results.data();
  EXPECT_EQ(iree_vm_call_function_argument_load(&call, 15).target_bits, 150u);
  EXPECT_EQ(iree_vm_call_function_argument_load(&call, 16).target_bits, 160u);

  iree_vm_call_function_result_store(&call, 15, {115, 1150});
  iree_vm_call_function_result_store(&call, 16, {116, 1160});
  EXPECT_EQ(direct_results.back().program_bits, 115u);
  EXPECT_EQ(overflow_results[0].program_bits, 116u);
}

TEST(VMExecutionTest, TransfersRefBanksWithExplicitOwnership) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  const iree_vm_ref_type_table_t* table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm"));
  ASSERT_NE(table, nullptr);
  iree_vm_ref_types_t types = {};
  IREE_ASSERT_OK(iree_vm_ref_types_resolve(table, &types));

  iree_vm_buffer_t* direct_buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(4, 0, iree_allocator_system(), &direct_buffer));
  iree_vm_buffer_t* overflow_buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(4, 0, iree_allocator_system(), &overflow_buffer));
  std::array<iree_vm_ref_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT>
      direct_arguments = {};
  std::array<iree_vm_ref_t, 1> overflow_arguments = {};
  direct_arguments.back() =
      iree_vm_buffer_ref_from_ptr_borrowed(&types, direct_buffer);
  overflow_arguments[0] =
      iree_vm_buffer_ref_from_ptr_borrowed(&types, overflow_buffer);
  std::array<iree_vm_ref_t, IREE_VM_CALL_DIRECT_REGISTER_COUNT> direct_results =
      {};
  std::array<iree_vm_ref_t, 1> overflow_results = {};
  iree_vm_call_packet_t call = {};
  call.ref_arguments.direct = direct_arguments.data();
  call.ref_arguments.overflow = overflow_arguments.data();
  call.ref_results.direct = direct_results.data();
  call.ref_results.overflow = overflow_results.data();

  for (uint16_t ordinal : {uint16_t{15}, uint16_t{16}}) {
    iree_vm_ref_t* argument = ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
                                  ? &direct_arguments[ordinal]
                                  : &overflow_arguments[0];
    iree_vm_ref_t* result = ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
                                ? &direct_results[ordinal]
                                : &overflow_results[0];
    void* expected_object = argument->object;
    iree_vm_ref_t local = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&call, ordinal, &local);
    EXPECT_EQ(local.object, expected_object);
    EXPECT_TRUE(iree_vm_buffer_ref_isa(&types, local));
    EXPECT_EQ(local.type_and_state & IREE_VM_REF_STATE_MASK,
              IREE_VM_REF_STATE_BORROWED);
    iree_vm_call_ref_result_store_move(&call, ordinal, &local);
    EXPECT_TRUE(iree_vm_ref_is_null(local));
    EXPECT_EQ(result->object, expected_object);
    EXPECT_EQ(result->type_and_state & IREE_VM_REF_STATE_MASK,
              IREE_VM_REF_STATE_OWNED);

    iree_vm_call_ref_argument_load_move(&call, ordinal, &local);
    EXPECT_TRUE(iree_vm_ref_is_null(*argument));
    EXPECT_TRUE(iree_vm_buffer_ref_isa(&types, local));
    EXPECT_EQ(local.type_and_state & IREE_VM_REF_STATE_MASK,
              IREE_VM_REF_STATE_BORROWED);
    iree_vm_ref_reset(&local);
    iree_vm_ref_reset(result);
  }
  iree_vm_buffer_release(direct_buffer);
  iree_vm_buffer_release(overflow_buffer);
  iree_vm_environment_free(environment);
}

TEST(VMExecutionTest, UnreachableResumeLeavesOutcomeUntouched) {
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_vm_module_function_resume_unreachable(nullptr, nullptr, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
}

}  // namespace
