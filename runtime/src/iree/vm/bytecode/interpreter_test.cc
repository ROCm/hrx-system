// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/execution_testdata.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/invocation_test_module.h"
#include "iree/vm/sync.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

void CountWake(void* user_data) { ++*static_cast<int*>(user_data); }

void RecordBufferRelease(void* user_data, iree_byte_span_t storage) {
  (void)storage;
  ++*static_cast<int*>(user_data);
}

void ExpectVariantEqual(iree_vm_variant_t actual, iree_vm_variant_t expected) {
  EXPECT_EQ(actual.payload, expected.payload);
  EXPECT_EQ(actual.metadata, expected.metadata);
}

class BytecodeInterpreterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    const iree_file_toc_t* files = iree_vm_bytecode_execution_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_execution_testdata_size(), 1u);
    const iree_vm_bytecode_module_storage_t storage = {
        iree_make_const_byte_span(files[0].data, files[0].size),
        iree_allocator_null(),
    };
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("execution.test"), storage,
        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(iree_vm_invocation_test_module_initialize(&native_counters_,
                                                             &native_module_));
    iree_vm_module_t* libraries[] = {&native_module_.base};
    IREE_ASSERT_OK(iree_vm_program_create(
        {module_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(&native_module_.base);
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
    EXPECT_EQ(native_counters_.destroy_count, 1);
  }

  iree_vm_function_t LookupFunction(iree_string_view_t name) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_EXPECT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("execution.test"), name, &function));
    return function;
  }

  void ExpectI32(iree_vm_variant_t variant, int32_t expected_value) {
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(variant, &value));
    EXPECT_EQ(value, expected_value);
  }

  void ExpectI64(iree_vm_variant_t variant, int64_t expected_value) {
    int64_t value = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(variant, &value));
    EXPECT_EQ(value, expected_value);
  }

  iree_vm_environment_t* environment_ = nullptr;
  iree_vm_ref_types_t types_ = {};
  iree_vm_module_t* module_ = nullptr;
  iree_vm_invocation_test_counters_t native_counters_ = {};
  iree_vm_invocation_test_module_t native_module_ = {};
  iree_vm_program_t* program_ = nullptr;
  iree_vm_process_t* process_ = nullptr;
  alignas(iree_max_align_t)
      std::array<uint8_t, kInvocationStorageSize> invocation_storage_ = {};
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(BytecodeInterpreterTest, InitializesAndInvokesGeneratedModule) {
  const iree_vm_function_t function = LookupFunction(IREE_SV("run"));

  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(5)};
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, function,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));

  ExpectI32(results[0], 12);

  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_ptr_from_variant_borrowed(&types_, results[1], &buffer));
  ASSERT_NE(buffer, nullptr);
  iree_const_byte_span_t contents = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &contents));
  ASSERT_EQ(contents.data_length, 10u);
  EXPECT_EQ(std::memcmp(contents.data, "loom-vm-v1", contents.data_length), 0);

  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
}

TEST_F(BytecodeInterpreterTest, ExecutesScalarLeaf) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(7)};
  iree_vm_variant_t results[1] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, LookupFunction(IREE_SV("leaf")),
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));

  ExpectI32(results[0], -7);
}

TEST_F(BytecodeInterpreterTest, DrivesLocalAndIndirectCalls) {
  for (iree_string_view_t name : {IREE_SV("call_local"), IREE_SV("call_import"),
                                  IREE_SV("call_indirect")}) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(19),
        iree_vm_variant_from_i32(23),
    };
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_, LookupFunction(name),
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    ExpectI32(results[0], 42);
  }
}

TEST_F(BytecodeInterpreterTest, ExecutesAllDirectBranchForms) {
  for (const auto& [input, expected] :
       std::array<std::pair<int32_t, int32_t>, 2>{{{0, 20}, {1, 10}}}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_,
                                  LookupFunction(IREE_SV("control_flow")),
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    ExpectI32(results[0], expected);
  }
}

TEST_F(BytecodeInterpreterTest, ExecutesSwitchCasesHolesAndDefault) {
  for (const auto& [input, expected] :
       std::array<std::pair<int32_t, int32_t>, 4>{
           {{0, 10}, {1, 99}, {2, 12}, {3, 99}}}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_,
                                  LookupFunction(IREE_SV("switch")),
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    ExpectI32(results[0], expected);
  }
}

TEST_F(BytecodeInterpreterTest, PersistsTypedProcessGlobals) {
  const iree_vm_function_t function = LookupFunction(IREE_SV("global_state"));
  for (const auto& [input, expected] :
       std::array<std::pair<int32_t, int32_t>, 2>{{{34, -42}, {5, -13}}}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[1] = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_, function,
                                  iree_vm_variant_span_from_array(arguments),
                                  iree_vm_variant_span_from_array(results)));
    ExpectI32(results[0], expected);
  }
}

TEST_F(BytecodeInterpreterTest, ExecutesBufferAndLocalMemoryRoundTrip) {
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(8, 0, iree_allocator_system(), &buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&types_, buffer),
  };
  iree_vm_variant_t results[2] = {};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_,
                                LookupFunction(IREE_SV("memory_roundtrip")),
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));

  ExpectI64(results[0], INT64_C(0x2211221122112211));
  ExpectI32(results[1], 0);
  const auto* bytes = static_cast<const uint8_t*>(iree_vm_buffer_data(buffer));
  ASSERT_NE(bytes, nullptr);
  for (iree_host_size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(bytes[i], i % 2 == 0 ? 0x11 : 0x22);
  }

  iree_vm_buffer_release(buffer);
}

TEST_F(BytecodeInterpreterTest, TransfersValueOverflowAcrossNestedCall) {
  std::array<iree_vm_variant_t, 17> arguments = {};
  std::array<iree_vm_variant_t, 17> results = {};
  for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
    arguments[i] = iree_vm_variant_from_i32((int32_t)(i * 3));
  }

  IREE_ASSERT_OK(iree_vm_invoke(
      invocation_, LookupFunction(IREE_SV("call_overflow_values")),
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      iree_vm_variant_span_from_ptr(results.data(), results.size())));

  for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
    EXPECT_TRUE(iree_vm_variant_is_empty(arguments[i]));
    ExpectI32(results[i], (int32_t)(i * 3));
  }
}

TEST_F(BytecodeInterpreterTest, TransfersRefOverflowAcrossNestedCall) {
  std::array<uint8_t, 1> storage = {0};
  int release_count = 0;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          release_callback, iree_allocator_system(), &buffer));
  std::array<iree_vm_variant_t, 17> arguments = {};
  std::array<iree_vm_variant_t, 17> results = {};
  for (iree_vm_variant_t& argument : arguments) {
    argument = iree_vm_buffer_variant_from_ptr_retained(&types_, buffer);
  }

  IREE_ASSERT_OK(iree_vm_invoke(
      invocation_, LookupFunction(IREE_SV("call_overflow_refs")),
      iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
      iree_vm_variant_span_from_ptr(results.data(), results.size())));

  for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
    EXPECT_TRUE(iree_vm_variant_is_empty(arguments[i]));
    iree_vm_buffer_t* returned_buffer = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(&types_, results[i],
                                                            &returned_buffer));
    EXPECT_EQ(returned_buffer, buffer);
  }
  iree_vm_variant_span_reset(
      iree_vm_variant_span_from_ptr(results.data(), results.size()));
  EXPECT_EQ(release_count, 0);
  iree_vm_buffer_release(buffer);
  EXPECT_EQ(release_count, 1);
}

TEST_F(BytecodeInterpreterTest, FailsUnresolvedOptionalCallTransactionally) {
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  const iree_vm_variant_t untouched_result =
      iree_vm_variant_from_i64(INT64_C(0x12345678));
  iree_vm_variant_t results[] = {untouched_result};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("call_optional")),
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  ExpectVariantEqual(results[0], untouched_result);
}

TEST_F(BytecodeInterpreterTest, YieldsToExplicitContinuation) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(41)};
  const iree_vm_variant_t untouched_result =
      iree_vm_variant_from_i64(INT64_C(0x12345678));
  iree_vm_variant_t results[] = {untouched_result};
  int wake_count = 0;
  const iree_vm_invocation_wake_callback_t wake_callback = {
      CountWake,
      &wake_count,
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("yield_once")),
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results), wake_callback, &outcome));

  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(wake_count, 1);
  ExpectVariantEqual(results[0], untouched_result);

  IREE_ASSERT_OK(iree_vm_invocation_resume(
      invocation_, iree_vm_variant_span_from_array(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ExpectI32(results[0], 42);
}

TEST_F(BytecodeInterpreterTest, TransfersRefsThroughBorrowAndMoveCalls) {
  std::array<uint8_t, 4> borrowed_storage = {1, 2, 3, 4};
  int borrowed_release_count = 0;
  const iree_vm_buffer_release_callback_t borrowed_release_callback = {
      RecordBufferRelease,
      &borrowed_release_count,
  };
  iree_vm_buffer_t* borrowed_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(borrowed_storage.data(), borrowed_storage.size()),
      borrowed_release_callback, iree_allocator_system(), &borrowed_buffer));
  iree_vm_variant_t borrowed_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&types_, borrowed_buffer),
  };
  iree_vm_variant_t borrowed_results[1] = {};
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("call_ref_borrow")),
                     iree_vm_variant_span_from_array(borrowed_arguments),
                     iree_vm_variant_span_from_array(borrowed_results)));
  iree_vm_buffer_t* returned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &types_, borrowed_results[0], &returned_buffer));
  EXPECT_EQ(returned_buffer, borrowed_buffer);
  iree_vm_variant_reset(&borrowed_results[0]);
  EXPECT_EQ(borrowed_release_count, 0);
  iree_vm_buffer_release(borrowed_buffer);
  EXPECT_EQ(borrowed_release_count, 1);

  std::array<uint8_t, 4> moved_storage = {5, 6, 7, 8};
  int moved_release_count = 0;
  const iree_vm_buffer_release_callback_t moved_release_callback = {
      RecordBufferRelease,
      &moved_release_count,
  };
  iree_vm_buffer_t* moved_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(moved_storage.data(), moved_storage.size()),
      moved_release_callback, iree_allocator_system(), &moved_buffer));
  iree_vm_buffer_t* expected_buffer = moved_buffer;
  iree_vm_variant_t moved_arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &moved_buffer),
  };
  iree_vm_variant_t moved_results[1] = {};
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("call_ref_move")),
                     iree_vm_variant_span_from_array(moved_arguments),
                     iree_vm_variant_span_from_array(moved_results)));
  EXPECT_EQ(moved_buffer, nullptr);
  returned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
      &types_, moved_results[0], &returned_buffer));
  EXPECT_EQ(returned_buffer, expected_buffer);
  iree_vm_variant_reset(&moved_results[0]);
  EXPECT_EQ(moved_release_count, 1);
}

TEST_F(BytecodeInterpreterTest, ReleasesMovedRefDiscardedByChild) {
  std::array<uint8_t, 1> storage = {0};
  int release_count = 0;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          release_callback, iree_allocator_system(), &buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
  };
  IREE_ASSERT_OK(iree_vm_invoke(invocation_,
                                LookupFunction(IREE_SV("call_ref_drop")),
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_empty()));

  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(release_count, 1);
}

TEST_F(BytecodeInterpreterTest, RejectsBorrowedRefBeforeYieldableEntry) {
  std::array<uint8_t, 1> storage = {0};
  int release_count = 0;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          release_callback, iree_allocator_system(), &buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&types_, buffer),
  };
  const iree_vm_variant_t untouched_result = iree_vm_variant_from_i32(123);
  iree_vm_variant_t results[] = {untouched_result};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          invocation_, LookupFunction(IREE_SV("yield_ref")),
          iree_vm_variant_span_from_array(arguments),
          iree_vm_variant_span_from_array(results), {}, &outcome));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectVariantEqual(results[0], untouched_result);
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_EQ(release_count, 0);
  iree_vm_buffer_release(buffer);
  EXPECT_EQ(release_count, 1);
}

TEST_F(BytecodeInterpreterTest, PreservesOwnedRefAcrossYield) {
  std::array<uint8_t, 1> storage = {0};
  int release_count = 0;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          release_callback, iree_allocator_system(), &buffer));
  iree_vm_buffer_t* expected_buffer = buffer;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
  };
  const iree_vm_variant_t untouched_result = iree_vm_variant_from_i32(123);
  iree_vm_variant_t results[] = {untouched_result};
  int wake_count = 0;
  const iree_vm_invocation_wake_callback_t wake_callback = {
      CountWake,
      &wake_count,
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("yield_ref")),
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results), wake_callback, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(wake_count, 1);
  ExpectVariantEqual(results[0], untouched_result);
  EXPECT_EQ(release_count, 0);

  IREE_ASSERT_OK(iree_vm_invocation_resume(
      invocation_, iree_vm_variant_span_from_array(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  iree_vm_buffer_t* returned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(&types_, results[0],
                                                          &returned_buffer));
  EXPECT_EQ(returned_buffer, expected_buffer);
  iree_vm_variant_reset(&results[0]);
  EXPECT_EQ(release_count, 1);
}

TEST_F(BytecodeInterpreterTest, CancellationUnwindsSuspendedFrame) {
  std::array<uint8_t, 1> storage = {0};
  int release_count = 0;
  const iree_vm_buffer_release_callback_t release_callback = {
      RecordBufferRelease,
      &release_count,
  };
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                          iree_make_byte_span(storage.data(), storage.size()),
                          release_callback, iree_allocator_system(), &buffer));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(&types_, &buffer),
  };
  const iree_vm_variant_t untouched_result = iree_vm_variant_from_i32(123);
  iree_vm_variant_t results[] = {untouched_result};
  int wake_count = 0;
  const iree_vm_invocation_wake_callback_t wake_callback = {
      CountWake,
      &wake_count,
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("yield_ref")),
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results), wake_callback, &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(release_count, 0);
  EXPECT_EQ(wake_count, 1);

  EXPECT_TRUE(iree_vm_invocation_request_cancel(
      invocation_, IREE_VM_CANCEL_REASON_CANCELLED));
  EXPECT_EQ(wake_count, 2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_vm_invocation_resume(
          invocation_, iree_vm_variant_span_from_array(results), &outcome));

  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  ExpectVariantEqual(results[0], untouched_result);
  EXPECT_EQ(release_count, 1);
}

TEST_F(BytecodeInterpreterTest, FailureLeavesResultsUntouched) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(1)};
  const iree_vm_variant_t untouched_result =
      iree_vm_variant_from_i64(INT64_C(0x12345678));
  iree_vm_variant_t results[] = {untouched_result};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("fail")),
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectVariantEqual(results[0], untouched_result);
}

}  // namespace
