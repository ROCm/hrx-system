// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/execution_test_provider.h"
#include "iree/vm/process.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

struct CountingAllocator {
  // Allocator receiving every command.
  iree_allocator_t delegate = iree_allocator_system();
  // Number of allocation or reallocation commands.
  iree_host_size_t allocation_count = 0;
  // Number of free commands.
  iree_host_size_t free_count = 0;
};

iree_status_t CountingAllocatorControl(void* self,
                                       iree_allocator_command_t command,
                                       const void* params, void** inout_ptr) {
  auto* allocator = static_cast<CountingAllocator*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC ||
      command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    ++allocator->allocation_count;
  } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    ++allocator->free_count;
  }
  return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                 inout_ptr);
}

iree_allocator_t MakeCountingAllocator(CountingAllocator* allocator) {
  return iree_allocator_t{allocator, CountingAllocatorControl};
}

struct ExecutionHarness {
  // Application module observations.
  iree_vm_execution_test_counters_t application_counters = {};
  // Math module observations.
  iree_vm_execution_test_counters_t math_counters = {};
  // Owned application module.
  iree_vm_module_t* application_module = nullptr;
  // Owned math module.
  iree_vm_module_t* math_module = nullptr;
  // Owned immutable linked program.
  iree_vm_program_t* program = nullptr;
  // Owned published process when created.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation.
  iree_vm_invocation_t* invocation = nullptr;

  ExecutionHarness() = default;
  ExecutionHarness(const ExecutionHarness&) = delete;
  ExecutionHarness& operator=(const ExecutionHarness&) = delete;

  ~ExecutionHarness() {
    iree_vm_process_release(process);
    iree_vm_invocation_free(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(application_module);
    iree_vm_module_release(math_module);
  }

  void Initialize(iree_vm_execution_test_options_t application_options = {},
                  iree_vm_execution_test_options_t math_options = {}) {
    IREE_ASSERT_OK(iree_vm_execution_test_module_create(
        IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION, application_options,
        &application_counters, iree_allocator_system(), &application_module));
    IREE_ASSERT_OK(iree_vm_execution_test_module_create(
        IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH, math_options, &math_counters,
        iree_allocator_system(), &math_module));
    iree_vm_module_t* libraries[] = {math_module};
    IREE_ASSERT_OK(iree_vm_program_create(
        {application_module, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program));
    IREE_ASSERT_OK(iree_vm_invocation_allocate(
        kInvocationStorageSize, iree_allocator_system(), &invocation));
  }

  void CreateProcess(int32_t initialization_value = 42) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(initialization_value),
    };
    IREE_ASSERT_OK(iree_vm_process_create(
        program, invocation, iree_vm_variant_span_from_array(arguments),
        iree_allocator_system(), &process));
    ASSERT_NE(process, nullptr);
  }

  iree_status_t Lookup(iree_string_view_t module_name,
                       iree_string_view_t export_name,
                       iree_vm_function_t* out_function) {
    return iree_vm_process_lookup_function(process, module_name, export_name,
                                           out_function);
  }

  iree_status_t LookupApplication(iree_string_view_t export_name,
                                  iree_vm_function_t* out_function) {
    return Lookup(IREE_SV("execution.app"), export_name, out_function);
  }

  iree_status_t InvokeBinary(iree_string_view_t export_name, int32_t lhs,
                             int32_t rhs, int32_t* out_value) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_RETURN_IF_ERROR(LookupApplication(export_name, &function));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(lhs),
        iree_vm_variant_from_i32(rhs),
    };
    iree_vm_variant_t results[1] = {};
    iree_status_t status = iree_vm_invoke(
        invocation, function, iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results));
    int32_t value = 0;
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(results[0], &value);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    if (iree_status_is_ok(status)) {
      *out_value = value;
    }
    return status;
  }
};

struct WakeCounter {
  // Number of callback invocations.
  int count = 0;
};

void CountWake(void* user_data) {
  auto* counter = static_cast<WakeCounter*>(user_data);
  ++counter->count;
}

iree_vm_invocation_wake_callback_t MakeWakeCallback(WakeCounter* counter) {
  return iree_vm_invocation_wake_callback_t{CountWake, counter};
}

struct BlockingCancellationWake {
  // Number of provider and cancellation wake calls.
  std::atomic<int> call_count{0};
  // Protects the deterministic callback-retirement rendezvous.
  std::mutex mutex;
  // Signals callback entry and host-authorized retirement.
  std::condition_variable condition;
  // True once the cancellation callback is blocked in user code.
  bool cancellation_callback_entered = false;
  // True once the host permits the cancellation callback to return.
  bool release_cancellation_callback = false;
};

void BlockCancellationWake(void* user_data) {
  auto* wake = static_cast<BlockingCancellationWake*>(user_data);
  const int call_ordinal =
      wake->call_count.fetch_add(1, std::memory_order_relaxed);
  if (call_ordinal != 1) return;
  std::unique_lock<std::mutex> lock(wake->mutex);
  wake->cancellation_callback_entered = true;
  wake->condition.notify_all();
  wake->condition.wait(lock,
                       [&] { return wake->release_cancellation_callback; });
}

TEST(VMExecutionTest, ReusesPlacementAndAllocatedInvocationStorage) {
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  ASSERT_NE(invocation, nullptr);
  iree_vm_invocation_deinitialize(invocation);

  invocation = nullptr;
  IREE_ASSERT_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(storage.data(), storage.size()), &invocation));
  iree_vm_invocation_deinitialize(invocation);

  invocation = nullptr;
  CountingAllocator allocator;
  IREE_ASSERT_OK(iree_vm_invocation_allocate(
      kInvocationStorageSize, MakeCountingAllocator(&allocator), &invocation));
  ASSERT_NE(invocation, nullptr);
  EXPECT_EQ(allocator.allocation_count, 1u);
  EXPECT_EQ(allocator.free_count, 0u);
  iree_vm_invocation_free(invocation);
  EXPECT_EQ(allocator.free_count, 1u);

  alignas(iree_max_align_t) std::array<uint8_t, 1> too_small;
  std::memset(too_small.data(), 0xA5, too_small.size());
  const auto expected = too_small;
  invocation = reinterpret_cast<iree_vm_invocation_t*>(1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_initialize(
          iree_make_byte_span(too_small.data(), too_small.size()),
          &invocation));
  EXPECT_EQ(invocation, nullptr);
  EXPECT_EQ(too_small, expected);
}

TEST(VMExecutionTest, RejectsOverlappingCallerStorageBeforeTransaction) {
  ExecutionHarness harness;
  harness.Initialize();

  iree_vm_variant_t initialization_arguments[] = {
      iree_vm_variant_from_i32(42),
  };
  const iree_vm_variant_t untouched_initialization_argument =
      initialization_arguments[0];
  auto* aliased_process_outcome =
      reinterpret_cast<iree_vm_process_create_outcome_t*>(
          initialization_arguments);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_process_create_start(
          harness.program, harness.invocation,
          iree_vm_variant_span_from_array(initialization_arguments), {},
          iree_allocator_system(), aliased_process_outcome));
  EXPECT_EQ(initialization_arguments[0].payload,
            untouched_initialization_argument.payload);
  EXPECT_EQ(initialization_arguments[0].metadata,
            untouched_initialization_argument.metadata);
  EXPECT_EQ(harness.application_counters.attach_count, 0);
  EXPECT_EQ(harness.math_counters.attach_count, 0);

  harness.CreateProcess();
  iree_vm_function_t add = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("add"), &add));

  iree_vm_variant_t overlapping_values[] = {
      iree_vm_variant_from_i32(7),
      iree_vm_variant_from_i32(11),
  };
  const iree_vm_variant_t untouched_overlapping_values[] = {
      overlapping_values[0],
      overlapping_values[1],
  };
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  const int start_count = harness.application_counters.function_start_count;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          harness.invocation, add,
          iree_vm_variant_span_from_array(overlapping_values),
          iree_vm_variant_span_from_ptr(&overlapping_values[1], 1), {},
          &outcome));
  EXPECT_EQ(overlapping_values[0].payload,
            untouched_overlapping_values[0].payload);
  EXPECT_EQ(overlapping_values[0].metadata,
            untouched_overlapping_values[0].metadata);
  EXPECT_EQ(overlapping_values[1].payload,
            untouched_overlapping_values[1].payload);
  EXPECT_EQ(overlapping_values[1].metadata,
            untouched_overlapping_values[1].metadata);
  EXPECT_EQ(outcome, UINT32_MAX);

  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(7),
      iree_vm_variant_from_i32(11),
  };
  const iree_vm_variant_t untouched_arguments[] = {
      arguments[0],
      arguments[1],
  };
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  auto* aliased_invocation_outcome =
      reinterpret_cast<iree_vm_execution_outcome_t*>(results);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(harness.invocation, add,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results), {},
                               aliased_invocation_outcome));
  EXPECT_EQ(arguments[0].payload, untouched_arguments[0].payload);
  EXPECT_EQ(arguments[0].metadata, untouched_arguments[0].metadata);
  EXPECT_EQ(arguments[1].payload, untouched_arguments[1].payload);
  EXPECT_EQ(arguments[1].metadata, untouched_arguments[1].metadata);
  EXPECT_EQ(harness.application_counters.function_start_count, start_count);

  outcome = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          harness.invocation, add, iree_vm_variant_span_from_array(arguments),
          iree_vm_variant_span_from_ptr(
              reinterpret_cast<iree_vm_variant_t*>(harness.invocation), 1),
          {}, &outcome));
  EXPECT_EQ(arguments[0].payload, untouched_arguments[0].payload);
  EXPECT_EQ(arguments[0].metadata, untouched_arguments[0].metadata);
  EXPECT_EQ(arguments[1].payload, untouched_arguments[1].payload);
  EXPECT_EQ(arguments[1].metadata, untouched_arguments[1].metadata);
  EXPECT_EQ(outcome, UINT32_MAX);
  EXPECT_EQ(harness.application_counters.function_start_count, start_count);

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);
}

TEST(VMExecutionTest, SemanticRootRejectionConsumesArgumentsBeforeEntry) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  iree_vm_function_t add = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("add"), &add));
  const int start_count = harness.application_counters.function_start_count;
  iree_vm_execution_outcome_t outcome = UINT32_MAX;

  iree_vm_variant_t wrong_type_arguments[] = {
      iree_vm_variant_from_i64(7),
      iree_vm_variant_from_i32(11),
  };
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          harness.invocation, add,
          iree_vm_variant_span_from_array(wrong_type_arguments),
          iree_vm_variant_span_from_array(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_type_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_type_arguments[1]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_EQ(outcome, UINT32_MAX);
  EXPECT_EQ(harness.application_counters.function_start_count, start_count);

  iree_vm_variant_t wrong_count_arguments[] = {
      iree_vm_variant_from_i32(7),
      iree_vm_variant_from_i32(11),
  };
  iree_vm_variant_t wrong_count_results[] = {
      iree_vm_variant_from_i64(0x1234),
      iree_vm_variant_from_i64(0x5678),
  };
  const iree_vm_variant_t untouched_wrong_count_results[] = {
      wrong_count_results[0],
      wrong_count_results[1],
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          harness.invocation, add,
          iree_vm_variant_span_from_array(wrong_count_arguments),
          iree_vm_variant_span_from_array(wrong_count_results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_count_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_count_arguments[1]));
  EXPECT_EQ(wrong_count_results[0].payload,
            untouched_wrong_count_results[0].payload);
  EXPECT_EQ(wrong_count_results[0].metadata,
            untouched_wrong_count_results[0].metadata);
  EXPECT_EQ(wrong_count_results[1].payload,
            untouched_wrong_count_results[1].payload);
  EXPECT_EQ(wrong_count_results[1].metadata,
            untouched_wrong_count_results[1].metadata);
  EXPECT_EQ(harness.application_counters.function_start_count, start_count);

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);
}

TEST(VMExecutionTest, RejectsInvalidInitializerBeforeProcessAllocation) {
  ExecutionHarness harness;
  harness.Initialize();
  CountingAllocator allocator;
  iree_vm_process_create_outcome_t outcome = {
      UINT32_MAX,
      reinterpret_cast<iree_vm_process_t*>(1),
  };

  iree_vm_variant_t wrong_type_arguments[] = {
      iree_vm_variant_from_i64(42),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_process_create_start(
          harness.program, harness.invocation,
          iree_vm_variant_span_from_array(wrong_type_arguments), {},
          MakeCountingAllocator(&allocator), &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_type_arguments[0]));
  EXPECT_EQ(outcome.execution_outcome, UINT32_MAX);
  EXPECT_EQ(outcome.process, reinterpret_cast<iree_vm_process_t*>(1));

  iree_vm_variant_t wrong_count_arguments[] = {
      iree_vm_variant_from_i32(42),
      iree_vm_variant_from_i32(43),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_process_create_start(
          harness.program, harness.invocation,
          iree_vm_variant_span_from_array(wrong_count_arguments), {},
          MakeCountingAllocator(&allocator), &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_count_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(wrong_count_arguments[1]));
  EXPECT_EQ(outcome.execution_outcome, UINT32_MAX);
  EXPECT_EQ(outcome.process, reinterpret_cast<iree_vm_process_t*>(1));

  EXPECT_EQ(allocator.allocation_count, 0u);
  EXPECT_EQ(allocator.free_count, 0u);
  EXPECT_EQ(harness.application_counters.attach_count, 0);
  EXPECT_EQ(harness.math_counters.attach_count, 0);
  EXPECT_EQ(harness.application_counters.function_start_count, 0);
  EXPECT_EQ(harness.math_counters.function_start_count, 0);
}

TEST(VMExecutionTest, ConstructsAndPublishesAProcessAsynchronously) {
  ExecutionHarness harness;
  harness.Initialize({IREE_VM_EXECUTION_TEST_FLAG_NONE, 2});

  WakeCounter wake_counter;
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(42)};
  iree_vm_process_create_outcome_t outcome = {
      UINT32_MAX,
      reinterpret_cast<iree_vm_process_t*>(1),
  };
  IREE_ASSERT_OK(iree_vm_process_create_start(
      harness.program, harness.invocation,
      iree_vm_variant_span_from_array(arguments),
      MakeWakeCallback(&wake_counter), iree_allocator_system(), &outcome));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(outcome.process, nullptr);
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(wake_counter.count, 1);
  EXPECT_EQ(harness.application_counters.attach_count, 1);
  EXPECT_EQ(harness.math_counters.attach_count, 1);
  EXPECT_EQ(harness.application_counters.seal_count, 0);

  IREE_ASSERT_OK(iree_vm_process_create_resume(harness.invocation, &outcome));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(outcome.process, nullptr);
  EXPECT_EQ(wake_counter.count, 2);

  IREE_ASSERT_OK(iree_vm_process_create_resume(harness.invocation, &outcome));
  EXPECT_EQ(outcome.execution_outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  ASSERT_NE(outcome.process, nullptr);
  harness.process = outcome.process;
  EXPECT_EQ(harness.application_counters.seal_count, 1);
  EXPECT_EQ(harness.math_counters.seal_count, 1);
  EXPECT_EQ(harness.application_counters.frame_cleanup_count, 1);

  iree_vm_process_release(harness.process);
  harness.process = nullptr;
  EXPECT_EQ(harness.application_counters.detach_count, 1);
  EXPECT_EQ(harness.math_counters.detach_count, 1);
}

TEST(VMExecutionTest, CallsDirectLocalImportAndFunctionRefTargets) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 7, 11, &value));
  EXPECT_EQ(value, 18);
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("call_local"), 7, 11, &value));
  EXPECT_EQ(value, 18);
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("call_import"), 7, 11, &value));
  EXPECT_EQ(value, 18);

  iree_vm_export_t math_add_export = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(
      harness.math_module, IREE_SV("add"), &math_add_export));
  iree_vm_function_ref_t math_add = iree_vm_function_ref_null();
  IREE_ASSERT_OK(iree_vm_function_ref_from_export(harness.program,
                                                  math_add_export, &math_add));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_function_ref(math_add),
      iree_vm_variant_from_i32(13),
      iree_vm_variant_from_i32(29),
  };
  iree_vm_variant_t results[1];
  iree_vm_function_t call_function = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("call_function"), &call_function));
  IREE_ASSERT_OK(iree_vm_invoke(harness.invocation, call_function,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, 42);

  iree_vm_function_t bound_math_add = iree_vm_function_null();
  IREE_ASSERT_OK(iree_vm_function_from_function_ref(harness.process, math_add,
                                                    &bound_math_add));
  EXPECT_FALSE(iree_vm_function_is_null(bound_math_add));
}

TEST(VMExecutionTest, PublishesCompleteLaunchConfiguration) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  iree_vm_function_t launch_config = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("launch_config"), &launch_config));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(64),
      iree_vm_variant_from_bf16_bits(0x4000),
  };
  iree_vm_variant_t results[11];
  IREE_ASSERT_OK(iree_vm_invoke(harness.invocation, launch_config,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));

  const int64_t expected[] = {128, 1, 1, 1, 1, 1, 1, 1, 1, 32, 256};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    int64_t value = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[i], &value));
    EXPECT_EQ(value, expected[i]);
  }
}

TEST(VMExecutionTest, RepeatedAndNestedSuspensionUseOneFrameStack) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  WakeCounter wake_counter;
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(9)};
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t yield_twice = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("yield_twice"), &yield_twice));
  IREE_ASSERT_OK(
      iree_vm_invocation_start(harness.invocation, yield_twice,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results),
                               MakeWakeCallback(&wake_counter), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_EQ(wake_counter.count, 1);

  iree_vm_variant_t wrong_results[] = {
      iree_vm_variant_from_i64(0x5678),
      iree_vm_variant_from_i64(0x9ABC),
  };
  const iree_vm_variant_t untouched_wrong_results[] = {
      wrong_results[0],
      wrong_results[1],
  };
  const int application_resume_count =
      harness.application_counters.function_resume_count;
  const int math_resume_count = harness.math_counters.function_resume_count;
  const int wake_count = wake_counter.count;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_resume(harness.invocation,
                                iree_vm_variant_span_from_array(wrong_results),
                                &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(harness.application_counters.function_resume_count,
            application_resume_count);
  EXPECT_EQ(harness.math_counters.function_resume_count, math_resume_count);
  EXPECT_EQ(wake_counter.count, wake_count);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(wrong_results); ++i) {
    EXPECT_EQ(wrong_results[i].payload, untouched_wrong_results[i].payload);
    EXPECT_EQ(wrong_results[i].metadata, untouched_wrong_results[i].metadata);
  }

  IREE_ASSERT_OK(iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_EQ(wake_counter.count, 2);

  IREE_ASSERT_OK(iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, 10);

  const int cleanup_count = harness.application_counters.frame_cleanup_count +
                            harness.math_counters.frame_cleanup_count;
  arguments[0] = iree_vm_variant_from_i32(20);
  results[0] = iree_vm_variant_from_i64(0x5678);
  outcome = UINT32_MAX;
  iree_vm_function_t nested_yield = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("nested_yield"), &nested_yield));
  IREE_ASSERT_OK(
      iree_vm_invocation_start(harness.invocation, nested_yield,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results),
                               MakeWakeCallback(&wake_counter), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  IREE_ASSERT_OK(iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, 21);
  EXPECT_EQ(harness.application_counters.frame_cleanup_count +
                harness.math_counters.frame_cleanup_count,
            cleanup_count + 2);
}

TEST(VMExecutionTest, RefArgumentsPreserveAndPromoteOwnershipExactly) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  int destruction_count = 0;
  iree_vm_execution_test_object_t object;
  iree_vm_execution_test_object_initialize(&destruction_count, &object);
  iree_vm_variant_t borrowed_arguments[] = {
      iree_vm_variant_from_ptr_borrowed(&object,
                                        iree_vm_execution_test_object_type()),
  };
  iree_vm_variant_t results[1];
  iree_vm_function_t echo_ref = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("echo_ref"), &echo_ref));
  IREE_ASSERT_OK(
      iree_vm_invoke(harness.invocation, echo_ref,
                     iree_vm_variant_span_from_array(borrowed_arguments),
                     iree_vm_variant_span_from_array(results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(borrowed_arguments[0]));
  EXPECT_TRUE(iree_vm_variant_ref_isa(results[0],
                                      iree_vm_execution_test_object_type()));
  iree_vm_variant_reset(&results[0]);
  EXPECT_EQ(destruction_count, 0);

  borrowed_arguments[0] = iree_vm_variant_from_ptr_borrowed(
      &object, iree_vm_execution_test_object_type());
  results[0] = iree_vm_variant_from_i64(0x1234);
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t yield_ref = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("yield_ref"), &yield_ref));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_invocation_start(
          harness.invocation, yield_ref,
          iree_vm_variant_span_from_array(borrowed_arguments),
          iree_vm_variant_span_from_array(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(borrowed_arguments[0]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);

  borrowed_arguments[0] = iree_vm_variant_from_ptr_borrowed(
      &object, iree_vm_execution_test_object_type());
  iree_vm_function_t bad_yield_ref = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("bad_yield_ref"), &bad_yield_ref));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_vm_invocation_start(
          harness.invocation, bad_yield_ref,
          iree_vm_variant_span_from_array(borrowed_arguments),
          iree_vm_variant_span_from_array(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(borrowed_arguments[0]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);

  iree_vm_ref_object_release(&object, iree_vm_execution_test_object_type());
  EXPECT_EQ(destruction_count, 1);
}

TEST(VMExecutionTest, InvalidDynamicResultUnwindsBeforePublication) {
  ExecutionHarness harness;
  harness.Initialize({IREE_VM_EXECUTION_TEST_FLAG_RETURN_WRONG_REF_TYPE, 0},
                     {});
  harness.CreateProcess();

  int destruction_count = 0;
  iree_vm_execution_test_object_t object;
  iree_vm_execution_test_object_initialize(&destruction_count, &object);
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_ptr_retained(&object,
                                        iree_vm_execution_test_object_type()),
  };
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t echo_ref = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("echo_ref"), &echo_ref));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_vm_invocation_start(harness.invocation, echo_ref,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results), {},
                               &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_EQ(outcome, UINT32_MAX);
  EXPECT_EQ(destruction_count, 0);

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);

  iree_vm_ref_object_release(&object, iree_vm_execution_test_object_type());
  EXPECT_EQ(destruction_count, 1);
}

TEST(VMExecutionTest, CancellationWinsBeforeTransactionalPublication) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  WakeCounter wake_counter;
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(9)};
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t yield_twice = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("yield_twice"), &yield_twice));
  IREE_ASSERT_OK(
      iree_vm_invocation_start(harness.invocation, yield_twice,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results),
                               MakeWakeCallback(&wake_counter), &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  EXPECT_TRUE(iree_vm_invocation_request_cancel(
      harness.invocation, IREE_VM_CANCEL_REASON_CANCELLED));
  EXPECT_FALSE(iree_vm_invocation_request_cancel(
      harness.invocation, IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED));
  EXPECT_EQ(iree_vm_invocation_cancel_reason(harness.invocation),
            IREE_VM_CANCEL_REASON_CANCELLED);

  IREE_ASSERT_OK(iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_vm_invocation_resume(harness.invocation,
                                iree_vm_variant_span_from_array(results),
                                &outcome));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_FALSE(iree_vm_invocation_request_cancel(
      harness.invocation, IREE_VM_CANCEL_REASON_CANCELLED));

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);
}

TEST(VMExecutionTest, CancellationCallbackMayRetireAfterTerminalReturn) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  BlockingCancellationWake wake;
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(9)};
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t yield_twice = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("yield_twice"), &yield_twice));
  const iree_vm_invocation_wake_callback_t wake_callback = {
      BlockCancellationWake,
      &wake,
  };
  IREE_ASSERT_OK(iree_vm_invocation_start(
      harness.invocation, yield_twice,
      iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results), wake_callback, &outcome));
  ASSERT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_SUSPENDED);

  bool cancellation_accepted = false;
  std::thread cancellation_thread([&] {
    cancellation_accepted = iree_vm_invocation_request_cancel(
        harness.invocation, IREE_VM_CANCEL_REASON_CANCELLED);
  });
  {
    std::unique_lock<std::mutex> lock(wake.mutex);
    wake.condition.wait(lock,
                        [&] { return wake.cancellation_callback_entered; });
  }

  iree_status_t first_resume_status = iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome);
  iree_status_t terminal_status = iree_vm_invocation_resume(
      harness.invocation, iree_vm_variant_span_from_array(results), &outcome);

  {
    std::lock_guard<std::mutex> lock(wake.mutex);
    wake.release_cancellation_callback = true;
  }
  wake.condition.notify_all();
  cancellation_thread.join();

  EXPECT_TRUE(cancellation_accepted);
  IREE_EXPECT_OK(first_resume_status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, terminal_status);
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);
}

TEST(VMExecutionTest, CapacityFailureIsTerminalAndReusable) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  iree_vm_invocation_t* invocation = nullptr;
  IREE_ASSERT_OK(
      iree_vm_invocation_allocate(1024, iree_allocator_system(), &invocation));
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(9)};
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  iree_vm_execution_outcome_t outcome = UINT32_MAX;
  iree_vm_function_t yield_twice = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("yield_twice"), &yield_twice));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_vm_invocation_start(
          invocation, yield_twice, iree_vm_variant_span_from_array(arguments),
          iree_vm_variant_span_from_array(results), {}, &outcome));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);

  iree_vm_function_t add = iree_vm_function_null();
  IREE_ASSERT_OK(harness.LookupApplication(IREE_SV("add"), &add));
  iree_vm_variant_t add_arguments[] = {
      iree_vm_variant_from_i32(2),
      iree_vm_variant_from_i32(3),
  };
  IREE_ASSERT_OK(iree_vm_invoke(invocation, add,
                                iree_vm_variant_span_from_array(add_arguments),
                                iree_vm_variant_span_from_array(results)));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, 5);
  iree_vm_invocation_free(invocation);
}

TEST(VMExecutionTest, ProviderFailureIsTerminalAndLeavesResultsUntouched) {
  ExecutionHarness harness;
  harness.Initialize({}, {IREE_VM_EXECUTION_TEST_FLAG_FAIL_FUNCTION, 0});
  harness.CreateProcess();

  iree_vm_function_t call_import = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("call_import"), &call_import));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(7),
      iree_vm_variant_from_i32(11),
  };
  iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
  const iree_vm_variant_t untouched_result = results[0];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_vm_invoke(harness.invocation, call_import,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);

  int32_t value = 0;
  IREE_ASSERT_OK(harness.InvokeBinary(IREE_SV("add"), 2, 3, &value));
  EXPECT_EQ(value, 5);
}

TEST(VMExecutionTest, ProcessConstructionUnwindsAttachAndSealFailures) {
  {
    ExecutionHarness harness;
    harness.Initialize({IREE_VM_EXECUTION_TEST_FLAG_FAIL_ATTACH, 0}, {});
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(42)};
    iree_vm_process_t* process = reinterpret_cast<iree_vm_process_t*>(1);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_vm_process_create(harness.program, harness.invocation,
                               iree_vm_variant_span_from_array(arguments),
                               iree_allocator_system(), &process));
    EXPECT_EQ(process, nullptr);
    EXPECT_EQ(harness.application_counters.attach_self_cleanup_count, 1);
    EXPECT_EQ(harness.application_counters.detach_count, 0);
    EXPECT_EQ(harness.math_counters.attach_count, 0);
  }
  {
    ExecutionHarness harness;
    harness.Initialize({}, {IREE_VM_EXECUTION_TEST_FLAG_FAIL_ATTACH, 0});
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(42)};
    iree_vm_process_t* process = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_vm_process_create(harness.program, harness.invocation,
                               iree_vm_variant_span_from_array(arguments),
                               iree_allocator_system(), &process));
    EXPECT_EQ(process, nullptr);
    EXPECT_EQ(harness.application_counters.detach_count, 1);
    EXPECT_EQ(harness.math_counters.attach_self_cleanup_count, 1);
    EXPECT_EQ(harness.math_counters.detach_count, 0);
  }
  {
    ExecutionHarness harness;
    harness.Initialize({IREE_VM_EXECUTION_TEST_FLAG_FAIL_SEAL, 0}, {});
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(42)};
    iree_vm_process_t* process = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_vm_process_create(harness.program, harness.invocation,
                               iree_vm_variant_span_from_array(arguments),
                               iree_allocator_system(), &process));
    EXPECT_EQ(process, nullptr);
    EXPECT_EQ(harness.application_counters.detach_count, 1);
    EXPECT_EQ(harness.math_counters.detach_count, 1);
    EXPECT_EQ(harness.math_counters.seal_count, 0);
  }
}

TEST(VMExecutionTest, SynchronousAdapterDrivesARepeatedlyYieldingFunction) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(31)};
  iree_vm_variant_t results[1];
  iree_vm_function_t yield_twice = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("yield_twice"), &yield_twice));
  IREE_ASSERT_OK(iree_vm_invoke(harness.invocation, yield_twice,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
  int32_t value = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &value));
  EXPECT_EQ(value, 32);
}

TEST(VMExecutionTest, InvocationScopesCanonicalFpuState) {
  ExecutionHarness harness;
  harness.Initialize();
  harness.CreateProcess();

  iree_vm_function_t multiply_f32 = iree_vm_function_null();
  IREE_ASSERT_OK(
      harness.LookupApplication(IREE_SV("multiply_f32"), &multiply_f32));
  iree_vm_variant_t arguments[2] = {};
  IREE_ASSERT_OK(iree_vm_variant_from_scalar_bits(
      IREE_VM_SCALAR_TYPE_F32, UINT32_C(0x00800000), &arguments[0]));
  IREE_ASSERT_OK(iree_vm_variant_from_scalar_bits(
      IREE_VM_SCALAR_TYPE_F32, UINT32_C(0x3F000000), &arguments[1]));
  iree_vm_variant_t results[1] = {};

  const iree_fpu_state_t caller_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  iree_status_t status =
      iree_vm_invoke(harness.invocation, multiply_f32,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  const iree_fpu_state_t observed_caller_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  const bool caller_state_was_restored = observed_caller_state.previous_value ==
                                         observed_caller_state.current_value;
  iree_fpu_state_pop(observed_caller_state);
  iree_fpu_state_pop(caller_state);

  IREE_ASSERT_OK(status);
  uint64_t result_bits = 0;
  IREE_ASSERT_OK(iree_vm_scalar_bits_from_variant(
      results[0], IREE_VM_SCALAR_TYPE_F32, &result_bits));
  EXPECT_EQ(result_bits, UINT32_C(0x00400000));
  EXPECT_TRUE(caller_state_was_restored);
}

}  // namespace
