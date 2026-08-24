// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/process.h"

namespace iree::vm::bytecode::testing {
namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;
constexpr iree_host_size_t kDeepInvocationStorageSize = 1024 * 1024;

struct CallExecutionHarness {
  // Immutable storage backing the loaded bytecode module.
  std::vector<uint8_t> image;
  // Owned bytecode module.
  iree_vm_module_t* module = nullptr;
  // Owned immutable linked program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation storage.
  iree_vm_invocation_t* invocation = nullptr;

  explicit CallExecutionHarness(std::vector<uint8_t> image)
      : image(std::move(image)) {}
  CallExecutionHarness(const CallExecutionHarness&) = delete;
  CallExecutionHarness& operator=(const CallExecutionHarness&) = delete;

  ~CallExecutionHarness() {
    iree_vm_process_release(process);
    iree_vm_invocation_free(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(module);
  }

  iree_status_t Initialize(
      iree_string_view_t module_name,
      iree_host_size_t invocation_storage_size = kInvocationStorageSize) {
    iree_vm_environment_t* environment = nullptr;
    iree_status_t status =
        iree_vm_environment_allocate(iree_allocator_system(), &environment);
    if (iree_status_is_ok(status)) {
      status = iree_vm_bytecode_module_create(
          environment, module_name,
          {iree_make_const_byte_span(image.data(), image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &module);
    }
    iree_vm_environment_free(environment);
    if (iree_status_is_ok(status)) {
      status = iree_vm_program_create({module, iree_vm_module_span_empty()},
                                      iree_allocator_system(), &program);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_invocation_allocate(
          invocation_storage_size, iree_allocator_system(), &invocation);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_process_create(program, invocation,
                                      iree_vm_variant_span_empty(),
                                      iree_allocator_system(), &process);
    }
    return status;
  }

  iree_status_t LookupFunction(iree_string_view_t module_name,
                               iree_string_view_t function_name,
                               iree_vm_function_t* out_function) {
    return iree_vm_process_lookup_function(process, module_name, function_name,
                                           out_function);
  }
};

void ExpectVariantEqual(iree_vm_variant_t actual, iree_vm_variant_t expected) {
  EXPECT_EQ(actual.payload, expected.payload);
  EXPECT_EQ(actual.metadata, expected.metadata);
}

TEST(VMBytecodeModuleCallTest, RejectsMalformedCallInstructions) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));

  std::vector<uint8_t> valid_image = BuildCallModuleImage();
  iree_vm_module_t* valid_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("call"),
      {iree_make_const_byte_span(valid_image.data(), valid_image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &valid_module));
  iree_vm_module_release(valid_module);

  const auto expect_rejected = [&](uint32_t function_ordinal,
                                   const auto& mutate) {
    std::vector<uint8_t> image = BuildCallModuleImage();
    const MutableFunctionImage function =
        FindFunctionImage(&image, function_ordinal);
    ASSERT_NE(function.row, nullptr);
    mutate(function);
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_call"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };

  constexpr uint32_t kDirectCallOffset = 4;
  constexpr uint32_t kIndirectCallOffset = 12;
  expect_rejected(1, [](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_call_record_t*>(
        function.bytecode + kDirectCallOffset);
    record->target_kind_u8 = UINT8_MAX;
  });
  expect_rejected(1, [](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_call_record_t*>(
        function.bytecode + kDirectCallOffset);
    record->target_ordinal_u16 = 6;
  });
  expect_rejected(1, [](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_call_record_t*>(
        function.bytecode + kDirectCallOffset);
    record->direct_ref_move_mask_u16 = 1;
  });
  expect_rejected(1, [](MutableFunctionImage function) {
    auto* record = reinterpret_cast<iree_vm_isa_control_call_record_t*>(
        function.bytecode + kDirectCallOffset);
    record->zero_padding_u16 = 1;
  });
  expect_rejected(1, [](MutableFunctionImage function) {
    function.row->value_register_count_u16 = 0;
  });
  expect_rejected(1, [](MutableFunctionImage function) {
    function.row->flags_u16 &= ~IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
  });
  expect_rejected(0, [](MutableFunctionImage function) {
    function.row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
  });
  expect_rejected(2, [](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_control_call_indirect_record_t*>(
            function.bytecode + kIndirectCallOffset);
    record->target_f8 = 1;
  });
  expect_rejected(2, [](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_control_call_indirect_record_t*>(
            function.bytecode + kIndirectCallOffset);
    record->callable_type_ordinal_u16 = 2;
  });
  expect_rejected(2, [](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_control_call_indirect_record_t*>(
            function.bytecode + kIndirectCallOffset);
    record->direct_ref_move_mask_u16 = 1;
  });
  expect_rejected(2, [](MutableFunctionImage function) {
    auto* record =
        reinterpret_cast<iree_vm_isa_control_call_indirect_record_t*>(
            function.bytecode + kIndirectCallOffset);
    record->zero_padding_u16 = 1;
  });
  expect_rejected(4, [](MutableFunctionImage function) {
    function.row->callable_type_ordinal_u16 = 0;
    function.row->flags_u16 = 0;
  });

  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleCallTest, ExecutesDirectIndirectAndSuspendingCalls) {
  CallExecutionHarness harness(BuildCallModuleImage());
  IREE_ASSERT_OK(harness.Initialize(IREE_SV("call")));

  const auto expect_sync_result = [&](iree_string_view_t name, int32_t input,
                                      int32_t expected) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_ASSERT_OK(harness.LookupFunction(IREE_SV("call"), name, &function));
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(harness.invocation, function,
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    int32_t result = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &result));
    EXPECT_EQ(result, expected);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  };
  expect_sync_result(IREE_SV("call_direct"), 41, 42);
  expect_sync_result(IREE_SV("call_indirect"), 42, 43);

  const auto expect_suspending_result = [&](iree_string_view_t name,
                                            int32_t input, int32_t expected) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_ASSERT_OK(harness.LookupFunction(IREE_SV("call"), name, &function));
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
    const iree_vm_variant_t untouched_result = results[0];
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    IREE_ASSERT_OK(iree_vm_invocation_start(
        harness.invocation, function,
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        iree_vm_invocation_wake_callback_t{}, &outcome));
    EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
    ExpectVariantEqual(results[0], untouched_result);
    IREE_ASSERT_OK(iree_vm_invocation_resume(
        harness.invocation, iree_vm_variant_span_from_array(results),
        &outcome));
    EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
    int32_t result = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &result));
    EXPECT_EQ(result, expected);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  };
  expect_suspending_result(IREE_SV("call_yield"), 43, 44);
  expect_suspending_result(IREE_SV("call_yield_indirect"), 44, 45);
}

TEST(VMBytecodeModuleCallTest,
     TrampolinesRecursiveCallsUntilInvocationStorageExhausts) {
  std::vector<uint8_t> image = BuildCallModuleImage();
  const MutableFunctionImage function = FindFunctionImage(&image, 0);
  ASSERT_NE(function.row, nullptr);
  function.row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
  const iree_vm_isa_control_call_record_t call = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_CALL,
      IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 0, 0, 0};
  std::memcpy(function.bytecode + 4, &call, sizeof(call));

  CallExecutionHarness harness(std::move(image));
  IREE_ASSERT_OK(
      harness.Initialize(IREE_SV("call"), kDeepInvocationStorageSize));
  iree_vm_function_t call_direct = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupFunction(IREE_SV("call"), IREE_SV("call_direct"),
                                        &call_direct));
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(41)};
  const iree_vm_variant_t sentinel = iree_vm_variant_from_i64(0x1234);
  iree_vm_variant_t results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_vm_invoke(harness.invocation, call_direct,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  ExpectVariantEqual(results[0], sentinel);
}

TEST(VMBytecodeModuleCallTest, RejectsMismatchedIndirectTargetContract) {
  std::vector<uint8_t> image = BuildCallModuleImage();
  const MutableFunctionImage function = FindFunctionImage(&image, 2);
  ASSERT_NE(function.row, nullptr);
  auto* address = reinterpret_cast<iree_vm_isa_func_address_record_t*>(
      function.bytecode + 4);
  address->target_ordinal_u16 = 3;
  address->callable_type_ordinal_u16 = 1;

  CallExecutionHarness harness(std::move(image));
  IREE_ASSERT_OK(harness.Initialize(IREE_SV("call")));
  iree_vm_function_t call_indirect = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupFunction(
      IREE_SV("call"), IREE_SV("call_indirect"), &call_indirect));
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(41)};
  const iree_vm_variant_t sentinel = iree_vm_variant_from_i64(0x1234);
  iree_vm_variant_t results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invoke(harness.invocation, call_indirect,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectVariantEqual(results[0], sentinel);
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
