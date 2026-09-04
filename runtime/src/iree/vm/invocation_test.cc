// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/invocation.h"

#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/invocation_test_module.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

void CountWake(void* user_data) { ++*static_cast<int*>(user_data); }

void CountAtomicWake(void* user_data) {
  static_cast<std::atomic<int>*>(user_data)->fetch_add(
      1, std::memory_order_relaxed);
}

void ExpectVariantEqual(iree_vm_variant_t actual, iree_vm_variant_t expected) {
  EXPECT_EQ(actual.payload, expected.payload);
  EXPECT_EQ(actual.metadata, expected.metadata);
}

iree_vm_invocation_wake_callback_t MakeWakeCallback(int* wake_count) {
  return iree_vm_invocation_wake_callback_t{CountWake, wake_count};
}

template <size_t N>
iree_vm_variant_span_t MakeVariantSpan(
    std::array<iree_vm_variant_t, N>& variants) {
  return iree_vm_variant_span_from_ptr(variants.data(), variants.size());
}

struct ScopedRoundingModeRestore {
  int rounding_mode;
  ~ScopedRoundingModeRestore() { std::fesetround(rounding_mode); }
};

class VMInvocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_invocation_test_module_initialize(&counters_, &module_));
    iree_vm_program_modules_t modules = {&module_.base,
                                         iree_vm_module_span_empty()};
    IREE_ASSERT_OK(
        iree_vm_program_create(modules, iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    iree_vm_process_create_outcome_t outcome = {};
    IREE_ASSERT_OK(iree_vm_process_create_start(
        program_, invocation_, iree_vm_variant_span_empty(), {},
        iree_allocator_system(), &outcome));
    ASSERT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
    ASSERT_NE(outcome.process, nullptr);
    process_ = outcome.process;
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(&module_.base);
    EXPECT_EQ(counters_.destroy_count, 1);
  }

  iree_vm_function_t LookupFunction(iree_string_view_t name) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_EXPECT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("invocation.test"), name, &function));
    return function;
  }

  void ExpectI32(iree_vm_variant_t variant, int32_t expected_value) {
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(variant, &value));
    EXPECT_EQ(value, expected_value);
  }

  iree_vm_invocation_test_counters_t counters_ = {};
  iree_vm_invocation_test_module_t module_ = {};
  iree_vm_program_t* program_ = nullptr;
  iree_vm_process_t* process_ = nullptr;
  alignas(iree_max_align_t)
      std::array<uint8_t, kInvocationStorageSize> invocation_storage_ = {};
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST(VMInvocationStorageTest, SupportsPlacementAndAllocatedStorage) {
  std::array<uint8_t, 1> insufficient_storage = {};
  iree_vm_invocation_t* invocation =
      reinterpret_cast<iree_vm_invocation_t*>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_invocation_initialize(
                            iree_make_byte_span(insufficient_storage.data(),
                                                insufficient_storage.size()),
                            &invocation));
  EXPECT_EQ(invocation, nullptr);

  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize + 1>
      unaligned_storage = {};
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(unaligned_storage.data() + 1,
                          unaligned_storage.size() - 1),
      &invocation));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(invocation) % iree_max_align_t, 0u);
  iree_vm_invocation_deinitialize(invocation);

  invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, iree_allocator_system(), &invocation));
  ASSERT_NE(invocation, nullptr);
  iree_vm_invocation_free(invocation);
}

TEST_F(VMInvocationTest, ImmediateCallConsumesArgumentsAndPublishesResults) {
  iree_vm_function_t function = LookupFunction(IREE_SV("add"));
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), {}, &outcome));

  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  ExpectI32(results[0], 42);
  EXPECT_EQ(counters_.start_count, 1);
  EXPECT_EQ(counters_.resume_count, 0);
}

TEST_F(VMInvocationTest, RejectsSemanticInputsAfterConsumingArguments) {
  struct Case {
    // Diagnostic case name.
    const char* name;
    // Resolved or deliberately null function under test.
    iree_vm_function_t function;
    // Argument storage consumed by every semantic rejection.
    std::array<iree_vm_variant_t, 2> arguments;
    // Number of arguments presented from |arguments|.
    iree_host_size_t argument_count;
    // Number of results presented from the shared result storage.
    iree_host_size_t result_count;
  };
  const iree_vm_function_t add_function = LookupFunction(IREE_SV("add"));
  Case cases[] = {
      {"wrong argument count",
       add_function,
       {iree_vm_variant_from_i32(1), iree_vm_variant_empty()},
       1,
       1},
      {"wrong argument type",
       add_function,
       {iree_vm_variant_from_i64(1), iree_vm_variant_from_i32(2)},
       2,
       1},
      {"wrong result count",
       add_function,
       {iree_vm_variant_from_i32(1), iree_vm_variant_from_i32(2)},
       2,
       0},
      {"null function",
       iree_vm_function_null(),
       {iree_vm_variant_from_i32(1), iree_vm_variant_from_i32(2)},
       2,
       1},
  };
  const iree_vm_variant_t untouched_result =
      iree_vm_variant_from_i64(INT64_C(0x12345678));
  for (Case& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    std::array<iree_vm_variant_t, 1> results = {untouched_result};
    iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_invocation_start(
            invocation_, test_case.function,
            {test_case.arguments.data(), test_case.argument_count},
            {results.data(), test_case.result_count}, {}, &outcome));
    for (iree_host_size_t i = 0; i < test_case.argument_count; ++i) {
      EXPECT_TRUE(iree_vm_variant_is_empty(test_case.arguments[i]));
    }
    ExpectVariantEqual(results[0], untouched_result);
    EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  }

  EXPECT_EQ(counters_.start_count, 0);
}

TEST_F(VMInvocationTest, RejectsOverlappingBoundaryWithoutTouchingStorage) {
  iree_vm_function_t function = LookupFunction(IREE_SV("add"));
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  const auto original_arguments = arguments;
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  const iree_vm_variant_t original_result = results[0];
  auto* outcome =
      reinterpret_cast<iree_vm_execution_outcome_t*>(arguments.data());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_invocation_start(
                            invocation_, function, MakeVariantSpan(arguments),
                            MakeVariantSpan(results), {}, outcome));

  ExpectVariantEqual(arguments[0], original_arguments[0]);
  ExpectVariantEqual(arguments[1], original_arguments[1]);
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(counters_.start_count, 0);
}

TEST_F(VMInvocationTest, DrivesLocalImportAndIndirectCallsIteratively) {
  for (iree_string_view_t name :
       {IREE_SV("call_local"), IREE_SV("call_import")}) {
    iree_vm_function_t function = LookupFunction(name);
    std::array<iree_vm_variant_t, 2> arguments = {
        iree_vm_variant_from_i32(19),
        iree_vm_variant_from_i32(23),
    };
    std::array<iree_vm_variant_t, 1> results = {
        iree_vm_variant_empty(),
    };
    iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
    IREE_ASSERT_OK(iree_vm_invocation_start(
        invocation_, function, MakeVariantSpan(arguments),
        MakeVariantSpan(results), {}, &outcome));
    EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
    ExpectI32(results[0], 42);
  }

  iree_vm_function_t return_function = LookupFunction(IREE_SV("return_local"));
  std::array<iree_vm_variant_t, 1> function_results = {
      iree_vm_variant_empty(),
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, return_function, iree_vm_variant_span_empty(),
      MakeVariantSpan(function_results), {}, &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ASSERT_TRUE(iree_vm_variant_is_function_ref(function_results[0]));

  iree_vm_function_t call_function = LookupFunction(IREE_SV("call_function"));
  std::array<iree_vm_variant_t, 3> arguments = {
      function_results[0],
      iree_vm_variant_from_i32(20),
      iree_vm_variant_from_i32(22),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, call_function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), {}, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ExpectI32(results[0], 42);
  EXPECT_EQ(counters_.start_count, 7);
  EXPECT_EQ(counters_.resume_count, 3);
  EXPECT_EQ(counters_.cleanup_count, 3);
}

TEST_F(VMInvocationTest, PreservesOptionalImportSemantics) {
  iree_vm_function_t call_function = LookupFunction(IREE_SV("call_optional"));
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(1),
      iree_vm_variant_from_i32(2),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  const iree_vm_variant_t original_result = results[0];
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_vm_invocation_start(invocation_, call_function,
                               MakeVariantSpan(arguments),
                               MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(counters_.cleanup_count, 1);

  iree_vm_function_t return_function =
      LookupFunction(IREE_SV("return_optional"));
  results[0] = iree_vm_variant_empty();
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, return_function, iree_vm_variant_span_empty(),
      MakeVariantSpan(results), {}, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_TRUE(iree_vm_variant_is_function_ref(results[0]));
  EXPECT_TRUE(iree_vm_variant_is_null(results[0]));
}

TEST_F(VMInvocationTest, ResumesTransactionallyAndReusesStorage) {
  iree_vm_function_t function = LookupFunction(IREE_SV("yield_twice"));
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(41),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  const iree_vm_variant_t original_result = results[0];
  int wake_count = 0;
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), MakeWakeCallback(&wake_count), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(wake_count, 1);

  outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_resume(invocation_, iree_vm_variant_span_empty(),
                                &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_EQ(counters_.resume_count, 0);

  auto* overlapping_outcome =
      reinterpret_cast<iree_vm_execution_outcome_t*>(results.data());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_resume(invocation_, MakeVariantSpan(results),
                                overlapping_outcome));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(counters_.resume_count, 0);

  IREE_ASSERT_OK(iree_vm_invocation_resume(invocation_,
                                           MakeVariantSpan(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(wake_count, 2);
  IREE_ASSERT_OK(iree_vm_invocation_resume(invocation_,
                                           MakeVariantSpan(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ExpectI32(results[0], 42);
  EXPECT_EQ(counters_.resume_count, 2);
  EXPECT_EQ(counters_.cleanup_count, 1);

  function = LookupFunction(IREE_SV("add"));
  std::array<iree_vm_variant_t, 2> reuse_arguments = {
      iree_vm_variant_from_i32(20),
      iree_vm_variant_from_i32(22),
  };
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, function, MakeVariantSpan(reuse_arguments),
      MakeVariantSpan(results), {}, &outcome));
  ExpectI32(results[0], 42);
}

TEST_F(VMInvocationTest, CancellationWinsTerminalCompletion) {
  int destroy_count = 0;
  iree_vm_invocation_test_object_t object = {};
  iree_vm_invocation_test_object_initialize(&destroy_count, &object);
  void* object_ptr = &object;
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_ptr_move(&object_ptr,
                                    iree_vm_invocation_test_object_type()),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i32(123),
  };
  const iree_vm_variant_t original_result = results[0];
  int wake_count = 0;
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("yield_ref")),
      MakeVariantSpan(arguments), MakeVariantSpan(results),
      MakeWakeCallback(&wake_count), &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(wake_count, 1);
  EXPECT_FALSE(iree_vm_invocation_request_cancel(invocation_,
                                                 IREE_VM_CANCEL_REASON_NONE));
  EXPECT_TRUE(iree_vm_invocation_request_cancel(
      invocation_, IREE_VM_CANCEL_REASON_CANCELLED));
  EXPECT_FALSE(iree_vm_invocation_request_cancel(
      invocation_, IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED));
  EXPECT_EQ(wake_count, 2);

  outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_vm_invocation_resume(
                            invocation_, MakeVariantSpan(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(counters_.cleanup_count, 1);
  EXPECT_EQ(destroy_count, 1);
}

TEST_F(VMInvocationTest, CancellationRemainsStickyAcrossRepeatedSuspension) {
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(41),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  const iree_vm_variant_t original_result = results[0];
  std::atomic<int> wake_count{0};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("yield_twice")),
      MakeVariantSpan(arguments), MakeVariantSpan(results),
      {CountAtomicWake, &wake_count}, &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(wake_count.load(std::memory_order_relaxed), 1);

  bool cancellation_accepted = false;
  std::thread cancellation_thread([&]() {
    cancellation_accepted = iree_vm_invocation_request_cancel(
        invocation_, IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED);
  });
  cancellation_thread.join();
  EXPECT_TRUE(cancellation_accepted);
  EXPECT_EQ(wake_count.load(std::memory_order_relaxed), 2);

  IREE_ASSERT_OK(iree_vm_invocation_resume(invocation_,
                                           MakeVariantSpan(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(wake_count.load(std::memory_order_relaxed), 3);
  ExpectVariantEqual(results[0], original_result);

  outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEADLINE_EXCEEDED,
                        iree_vm_invocation_resume(
                            invocation_, MakeVariantSpan(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(counters_.cleanup_count, 1);
}

TEST_F(VMInvocationTest, TransfersOwnedAndBorrowedRefs) {
  iree_vm_function_t function = LookupFunction(IREE_SV("echo_ref"));

  int owned_destroy_count = 0;
  iree_vm_invocation_test_object_t owned_object = {};
  iree_vm_invocation_test_object_initialize(&owned_destroy_count,
                                            &owned_object);
  void* owned_ptr = &owned_object;
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_ptr_move(&owned_ptr,
                                    iree_vm_invocation_test_object_type()),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_ref_isa(results[0],
                                      iree_vm_invocation_test_object_type()));
  EXPECT_EQ(owned_destroy_count, 0);
  iree_vm_variant_reset(&results[0]);
  EXPECT_EQ(owned_destroy_count, 1);

  int borrowed_destroy_count = 0;
  iree_vm_invocation_test_object_t borrowed_object = {};
  iree_vm_invocation_test_object_initialize(&borrowed_destroy_count,
                                            &borrowed_object);
  arguments[0] = iree_vm_variant_from_ptr_borrowed(
      &borrowed_object, iree_vm_invocation_test_object_type());
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_ref_isa(results[0],
                                      iree_vm_invocation_test_object_type()));
  iree_vm_variant_reset(&results[0]);
  EXPECT_EQ(borrowed_destroy_count, 0);
  iree_vm_ref_object_release(&borrowed_object,
                             iree_vm_invocation_test_object_type());
  EXPECT_EQ(borrowed_destroy_count, 1);
}

TEST_F(VMInvocationTest, RejectsBorrowedRefsAcrossDeclaredOrActualYield) {
  int first_destroy_count = 0;
  iree_vm_invocation_test_object_t first_object = {};
  iree_vm_invocation_test_object_initialize(&first_destroy_count,
                                            &first_object);
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_ptr_borrowed(&first_object,
                                        iree_vm_invocation_test_object_type()),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i32(123),
  };
  const iree_vm_variant_t original_result = results[0];
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          invocation_, LookupFunction(IREE_SV("yield_ref")),
          MakeVariantSpan(arguments), MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(counters_.start_count, 0);
  iree_vm_ref_object_release(&first_object,
                             iree_vm_invocation_test_object_type());

  int second_destroy_count = 0;
  iree_vm_invocation_test_object_t second_object = {};
  iree_vm_invocation_test_object_initialize(&second_destroy_count,
                                            &second_object);
  arguments[0] = iree_vm_variant_from_ptr_borrowed(
      &second_object, iree_vm_invocation_test_object_type());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_invocation_start(
          invocation_, LookupFunction(IREE_SV("bad_yield_ref")),
          MakeVariantSpan(arguments), MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(counters_.start_count, 1);
  EXPECT_EQ(counters_.cleanup_count, 1);
  iree_vm_ref_object_release(&second_object,
                             iree_vm_invocation_test_object_type());
  EXPECT_EQ(first_destroy_count, 1);
  EXPECT_EQ(second_destroy_count, 1);
}

TEST_F(VMInvocationTest, DiscardsInvalidProviderResultsAndFailures) {
  int destroy_count = 0;
  iree_vm_invocation_test_object_t object = {};
  iree_vm_invocation_test_object_initialize(&destroy_count, &object);
  void* object_ptr = &object;
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_ptr_move(&object_ptr,
                                    iree_vm_invocation_test_object_type()),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i32(123),
  };
  const iree_vm_variant_t original_result = results[0];
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_vm_invocation_start(
          invocation_, LookupFunction(IREE_SV("bad_ref_result")),
          MakeVariantSpan(arguments), MakeVariantSpan(results), {}, &outcome));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(destroy_count, 1);

  std::array<iree_vm_variant_t, 2> fail_arguments = {
      iree_vm_variant_from_i32(1),
      iree_vm_variant_from_i32(2),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_vm_invocation_start(invocation_, LookupFunction(IREE_SV("fail")),
                               MakeVariantSpan(fail_arguments),
                               MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(fail_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(fail_arguments[1]));
  ExpectVariantEqual(results[0], original_result);
}

TEST_F(VMInvocationTest, ReportsFixedFrameStorageExhaustion) {
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  std::array<iree_vm_variant_t, 1> results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  const iree_vm_variant_t original_result = results[0];
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_vm_invocation_start(invocation_, LookupFunction(IREE_SV("exhaust")),
                               MakeVariantSpan(arguments),
                               MakeVariantSpan(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  ExpectVariantEqual(results[0], original_result);
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(counters_.start_count, 1);
}

TEST_F(VMInvocationTest, ScopesCanonicalFloatingPointStateToProviderCalls) {
  const int previous_rounding_mode = std::fegetround();
  ASSERT_NE(previous_rounding_mode, -1);
  const ScopedRoundingModeRestore rounding_mode_restore = {
      previous_rounding_mode};
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);

  const float input = std::nextafter(1.0f, 2.0f);
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_f32(input),
      iree_vm_variant_from_f32(input),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_ASSERT_OK(iree_vm_invocation_start(
      invocation_, LookupFunction(IREE_SV("multiply_f32")),
      MakeVariantSpan(arguments), MakeVariantSpan(results), {}, &outcome));
  EXPECT_EQ(std::fegetround(), FE_UPWARD);

  float result = 0.0f;
  IREE_ASSERT_OK(iree_vm_f32_from_variant(results[0], &result));
  const float first_step = std::nextafter(1.0f, 2.0f);
  const float nearest_product = std::nextafter(first_step, 2.0f);
  EXPECT_EQ(result, nearest_product);
}

}  // namespace
