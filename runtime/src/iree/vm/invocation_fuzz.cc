// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stateful fuzzing for the public asynchronous invocation driver.
//
// A deliberately small external model tracks only caller-visible state. Each
// operation compares status, argument consumption, result publication,
// outcome publication, provider entry, cleanup, wake delivery, and cancellation
// state without reading the private invocation representation.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/vm/execution_test_provider.h"
#include "iree/vm/process.h"

namespace {

constexpr iree_host_size_t kFullInvocationStorageSize = 16 * 1024;
constexpr iree_host_size_t kConstrainedInvocationStorageSize = 1024;
constexpr size_t kMaximumOperationCount = 128;

void Require(bool condition) {
  if (!condition) std::abort();
}

void RequireOk(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

iree_status_code_t ConsumeStatusCode(iree_status_t status) {
  iree::Status owned_status(std::move(status));
  return static_cast<iree_status_code_t>(owned_status.code());
}

bool VariantsEqual(iree_vm_variant_t lhs, iree_vm_variant_t rhs) {
  return lhs.payload == rhs.payload && lhs.metadata == rhs.metadata;
}

struct WakeCounter {
  // Number of synchronously delivered wake callbacks.
  int count = 0;
};

void CountWake(void* user_data) {
  auto* counter = static_cast<WakeCounter*>(user_data);
  ++counter->count;
}

iree_vm_invocation_wake_callback_t MakeWakeCallback(WakeCounter* counter) {
  return {CountWake, counter};
}

struct Harness {
  // Application-provider observations.
  iree_vm_execution_test_counters_t application_counters = {};
  // Math-provider observations.
  iree_vm_execution_test_counters_t math_counters = {};
  // Owned application module.
  iree_vm_module_t* application_module = nullptr;
  // Owned math module.
  iree_vm_module_t* math_module = nullptr;
  // Owned linked program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation storage.
  iree_vm_invocation_t* invocation = nullptr;
  // Bound immediate-completion function.
  iree_vm_function_t add = iree_vm_function_null();
  // Bound imported-call function.
  iree_vm_function_t call_import = iree_vm_function_null();
  // Bound repeatedly-suspending function.
  iree_vm_function_t yield_twice = iree_vm_function_null();

  Harness(bool constrained_capacity, bool fail_math_provider) {
    const iree_vm_execution_test_options_t application_options = {};
    const iree_vm_execution_test_options_t math_options = {
        fail_math_provider ? IREE_VM_EXECUTION_TEST_FLAG_FAIL_FUNCTION
                           : IREE_VM_EXECUTION_TEST_FLAG_NONE,
        0,
    };
    RequireOk(iree_vm_execution_test_module_create(
        IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION, application_options,
        &application_counters, iree_allocator_system(), &application_module));
    RequireOk(iree_vm_execution_test_module_create(
        IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH, math_options, &math_counters,
        iree_allocator_system(), &math_module));
    iree_vm_module_t* libraries[] = {math_module};
    RequireOk(iree_vm_program_create(
        {application_module, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program));
    const iree_host_size_t invocation_storage_size =
        constrained_capacity ? kConstrainedInvocationStorageSize
                             : kFullInvocationStorageSize;
    RequireOk(iree_vm_invocation_allocate(
        invocation_storage_size, iree_allocator_system(), &invocation));
    iree_vm_variant_t initialization_arguments[] = {
        iree_vm_variant_from_i32(42),
    };
    RequireOk(iree_vm_process_create(
        program, invocation,
        iree_vm_variant_span_from_array(initialization_arguments),
        iree_allocator_system(), &process));
    Require(iree_vm_variant_is_empty(initialization_arguments[0]));
    RequireOk(iree_vm_process_lookup_function(process, IREE_SV("execution.app"),
                                              IREE_SV("add"), &add));
    RequireOk(iree_vm_process_lookup_function(process, IREE_SV("execution.app"),
                                              IREE_SV("call_import"),
                                              &call_import));
    RequireOk(iree_vm_process_lookup_function(process, IREE_SV("execution.app"),
                                              IREE_SV("yield_twice"),
                                              &yield_twice));
  }

  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

  ~Harness() {
    iree_vm_process_release(process);
    iree_vm_invocation_free(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(application_module);
    iree_vm_module_release(math_module);
  }
};

struct Observations {
  // Application function-start callback count.
  int application_start_count;
  // Math function-start callback count.
  int math_start_count;
  // Application function-resume callback count.
  int application_resume_count;
  // Math function-resume callback count.
  int math_resume_count;
  // Total provider frame-cleanup count.
  int frame_cleanup_count;
  // Caller wake-callback count.
  int wake_count;
};

Observations CaptureObservations(const Harness& harness,
                                 const WakeCounter& wake_counter) {
  return {
      harness.application_counters.function_start_count,
      harness.math_counters.function_start_count,
      harness.application_counters.function_resume_count,
      harness.math_counters.function_resume_count,
      harness.application_counters.frame_cleanup_count +
          harness.math_counters.frame_cleanup_count,
      wake_counter.count,
  };
}

void RequireObservationDelta(const Observations& before,
                             const Observations& after,
                             const Observations& expected_delta) {
  Require(after.application_start_count - before.application_start_count ==
          expected_delta.application_start_count);
  Require(after.math_start_count - before.math_start_count ==
          expected_delta.math_start_count);
  Require(after.application_resume_count - before.application_resume_count ==
          expected_delta.application_resume_count);
  Require(after.math_resume_count - before.math_resume_count ==
          expected_delta.math_resume_count);
  Require(after.frame_cleanup_count - before.frame_cleanup_count ==
          expected_delta.frame_cleanup_count);
  Require(after.wake_count - before.wake_count == expected_delta.wake_count);
}

enum class DriverState {
  kIdle,
  kSuspended,
};

struct DriverModel {
  // Caller-visible invocation driver state.
  DriverState state = DriverState::kIdle;
  // Resumes remaining before the suspended operation terminates.
  int remaining_resume_count = 0;
  // Input whose increment is published by the yielding function.
  int32_t yield_input = 0;
  // First accepted cancellation reason for the active operation.
  iree_vm_cancel_reason_t cancel_reason = IREE_VM_CANCEL_REASON_NONE;
};

class InvocationStateMachine {
 public:
  InvocationStateMachine(bool constrained_capacity, bool fail_math_provider)
      : harness_(constrained_capacity, fail_math_provider),
        constrained_capacity_(constrained_capacity),
        fail_math_provider_(fail_math_provider) {}

  void Apply(uint8_t operation) {
    switch (operation % 13) {
      case 0:
        StartAdd(operation);
        break;
      case 1:
        StartYield(operation);
        break;
      case 2:
        StartImport(operation);
        break;
      case 3:
        StartWithWrongArgumentCount();
        break;
      case 4:
        StartWithWrongArgumentType();
        break;
      case 5:
        StartWithWrongResultCount();
        break;
      case 6:
        StartWithNullFunction();
        break;
      case 7:
        Resume();
        break;
      case 8:
        ResumeWithWrongResultCount();
        break;
      case 9:
        RequestCancellation((operation & 0x80u) != 0
                                ? IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED
                                : IREE_VM_CANCEL_REASON_CANCELLED);
        break;
      case 10:
        RequestCancellation(IREE_VM_CANCEL_REASON_NONE);
        break;
      case 11:
        ResumeWithNullOutcome();
        break;
      case 12:
        StartWithOverlappingSpans();
        break;
    }
    VerifyCancellationState();
  }

  void DrainAndProveReuse() {
    while (model_.state == DriverState::kSuspended) {
      Resume();
      VerifyCancellationState();
    }
    StartAdd(19);
    VerifyCancellationState();
    Require(model_.state == DriverState::kIdle);
  }

 private:
  void StartAdd(uint8_t bits) {
    const int32_t lhs = bits;
    const int32_t rhs = static_cast<int32_t>(bits ^ 0x5Au);
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(lhs),
        iree_vm_variant_from_i32(rhs),
    };
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x1234)};
    const iree_vm_variant_t untouched_result = results[0];
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code = ConsumeStatusCode(
        iree_vm_invocation_start(harness_.invocation, harness_.add,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results),
                                 MakeWakeCallback(&wake_counter_), &outcome));

    Require(iree_vm_variant_is_empty(arguments[0]));
    Require(iree_vm_variant_is_empty(arguments[1]));
    if (model_.state == DriverState::kSuspended) {
      Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
      Require(VariantsEqual(results[0], untouched_result));
      Require(outcome == UINT32_MAX);
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_), {});
      return;
    }
    Require(status_code == IREE_STATUS_OK);
    Require(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
    int32_t result = 0;
    RequireOk(iree_vm_i32_from_variant(results[0], &result));
    Require(result == lhs + rhs);
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_),
                            {1, 0, 0, 0, 0, 0});
  }

  void StartYield(uint8_t bits) {
    const int32_t value = bits;
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(value)};
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x2345)};
    const iree_vm_variant_t untouched_result = results[0];
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code = ConsumeStatusCode(
        iree_vm_invocation_start(harness_.invocation, harness_.yield_twice,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results),
                                 MakeWakeCallback(&wake_counter_), &outcome));

    Require(iree_vm_variant_is_empty(arguments[0]));
    Require(VariantsEqual(results[0], untouched_result));
    if (model_.state == DriverState::kSuspended) {
      Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
      Require(outcome == UINT32_MAX);
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_), {});
      return;
    }
    if (constrained_capacity_) {
      Require(status_code == IREE_STATUS_RESOURCE_EXHAUSTED);
      Require(outcome == UINT32_MAX);
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_),
                              {1, 0, 0, 0, 0, 0});
      return;
    }
    Require(status_code == IREE_STATUS_OK);
    Require(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_),
                            {1, 0, 0, 0, 0, 1});
    model_.state = DriverState::kSuspended;
    model_.remaining_resume_count = 2;
    model_.yield_input = value;
    model_.cancel_reason = IREE_VM_CANCEL_REASON_NONE;
  }

  void StartImport(uint8_t bits) {
    const int32_t lhs = bits;
    const int32_t rhs = static_cast<int32_t>(bits ^ 0xA5u);
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(lhs),
        iree_vm_variant_from_i32(rhs),
    };
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x3456)};
    const iree_vm_variant_t untouched_result = results[0];
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code = ConsumeStatusCode(
        iree_vm_invocation_start(harness_.invocation, harness_.call_import,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results),
                                 MakeWakeCallback(&wake_counter_), &outcome));

    Require(iree_vm_variant_is_empty(arguments[0]));
    Require(iree_vm_variant_is_empty(arguments[1]));
    if (model_.state == DriverState::kSuspended) {
      Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
      Require(VariantsEqual(results[0], untouched_result));
      Require(outcome == UINT32_MAX);
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_), {});
      return;
    }
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_),
                            {1, 1, 0, 0, 0, 0});
    if (fail_math_provider_) {
      Require(status_code == IREE_STATUS_ABORTED);
      Require(VariantsEqual(results[0], untouched_result));
      Require(outcome == UINT32_MAX);
      return;
    }
    Require(status_code == IREE_STATUS_OK);
    Require(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
    int32_t result = 0;
    RequireOk(iree_vm_i32_from_variant(results[0], &result));
    Require(result == lhs + rhs);
  }

  template <size_t ArgumentCount, size_t ResultCount>
  void RequireRejectedStart(iree_vm_function_t function,
                            iree_vm_variant_t (&arguments)[ArgumentCount],
                            iree_vm_variant_t (&results)[ResultCount]) {
    iree_vm_variant_t untouched_results[ResultCount];
    for (size_t i = 0; i < ResultCount; ++i) {
      untouched_results[i] = results[i];
    }
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code = ConsumeStatusCode(
        iree_vm_invocation_start(harness_.invocation, function,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results),
                                 MakeWakeCallback(&wake_counter_), &outcome));
    Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
    for (size_t i = 0; i < ArgumentCount; ++i) {
      Require(iree_vm_variant_is_empty(arguments[i]));
    }
    for (size_t i = 0; i < ResultCount; ++i) {
      Require(VariantsEqual(results[i], untouched_results[i]));
    }
    Require(outcome == UINT32_MAX);
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_), {});
  }

  void StartWithWrongArgumentCount() {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(7)};
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x4567)};
    RequireRejectedStart(harness_.add, arguments, results);
  }

  void StartWithWrongArgumentType() {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i64(7),
        iree_vm_variant_from_i32(11),
    };
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x5678)};
    RequireRejectedStart(harness_.add, arguments, results);
  }

  void StartWithWrongResultCount() {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(7),
        iree_vm_variant_from_i32(11),
    };
    iree_vm_variant_t results[] = {
        iree_vm_variant_from_i64(0x6789),
        iree_vm_variant_from_i64(0x789A),
    };
    RequireRejectedStart(harness_.add, arguments, results);
  }

  void StartWithNullFunction() {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(7),
        iree_vm_variant_from_i32(11),
    };
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x89AB)};
    RequireRejectedStart(iree_vm_function_null(), arguments, results);
  }

  void StartWithOverlappingSpans() {
    iree_vm_variant_t storage[] = {
        iree_vm_variant_from_i32(7),
        iree_vm_variant_from_i32(11),
    };
    const iree_vm_variant_t untouched_storage[] = {storage[0], storage[1]};
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code =
        ConsumeStatusCode(iree_vm_invocation_start(
            harness_.invocation, harness_.add,
            iree_vm_variant_span_from_array(storage), {storage, 1},
            MakeWakeCallback(&wake_counter_), &outcome));
    Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
    Require(VariantsEqual(storage[0], untouched_storage[0]));
    Require(VariantsEqual(storage[1], untouched_storage[1]));
    Require(outcome == UINT32_MAX);
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_), {});
  }

  void Resume() {
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0x9ABC)};
    const iree_vm_variant_t untouched_result = results[0];
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code =
        ConsumeStatusCode(iree_vm_invocation_resume(
            harness_.invocation, iree_vm_variant_span_from_array(results),
            &outcome));
    if (model_.state == DriverState::kIdle) {
      Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
      Require(VariantsEqual(results[0], untouched_result));
      Require(outcome == UINT32_MAX);
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_), {});
      return;
    }

    --model_.remaining_resume_count;
    if (model_.remaining_resume_count != 0) {
      Require(status_code == IREE_STATUS_OK);
      Require(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
      Require(VariantsEqual(results[0], untouched_result));
      RequireObservationDelta(before,
                              CaptureObservations(harness_, wake_counter_),
                              {0, 0, 1, 0, 0, 1});
      return;
    }

    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_),
                            {0, 0, 1, 0, 1, 0});
    if (model_.cancel_reason == IREE_VM_CANCEL_REASON_NONE) {
      Require(status_code == IREE_STATUS_OK);
      Require(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
      int32_t result = 0;
      RequireOk(iree_vm_i32_from_variant(results[0], &result));
      Require(result == model_.yield_input + 1);
    } else {
      const iree_status_code_t expected_code =
          model_.cancel_reason == IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED
              ? IREE_STATUS_DEADLINE_EXCEEDED
              : IREE_STATUS_CANCELLED;
      Require(status_code == expected_code);
      Require(VariantsEqual(results[0], untouched_result));
      Require(outcome == UINT32_MAX);
    }
    model_ = {};
  }

  void ResumeWithWrongResultCount() {
    iree_vm_variant_t results[] = {
        iree_vm_variant_from_i64(0xABCD),
        iree_vm_variant_from_i64(0xBCDE),
    };
    const iree_vm_variant_t untouched_results[] = {results[0], results[1]};
    iree_vm_execution_outcome_t outcome = UINT32_MAX;
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code =
        ConsumeStatusCode(iree_vm_invocation_resume(
            harness_.invocation, iree_vm_variant_span_from_array(results),
            &outcome));
    Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
    Require(VariantsEqual(results[0], untouched_results[0]));
    Require(VariantsEqual(results[1], untouched_results[1]));
    Require(outcome == UINT32_MAX);
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_), {});
  }

  void ResumeWithNullOutcome() {
    iree_vm_variant_t results[] = {iree_vm_variant_from_i64(0xCDEF)};
    const iree_vm_variant_t untouched_result = results[0];
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const iree_status_code_t status_code =
        ConsumeStatusCode(iree_vm_invocation_resume(
            harness_.invocation, iree_vm_variant_span_from_array(results),
            nullptr));
    Require(status_code == IREE_STATUS_INVALID_ARGUMENT);
    Require(VariantsEqual(results[0], untouched_result));
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_), {});
  }

  void RequestCancellation(iree_vm_cancel_reason_t reason) {
    const Observations before = CaptureObservations(harness_, wake_counter_);
    const bool accepted =
        iree_vm_invocation_request_cancel(harness_.invocation, reason);
    const bool expected_accepted =
        model_.state == DriverState::kSuspended &&
        model_.cancel_reason == IREE_VM_CANCEL_REASON_NONE &&
        (reason == IREE_VM_CANCEL_REASON_CANCELLED ||
         reason == IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED);
    Require(accepted == expected_accepted);
    if (expected_accepted) model_.cancel_reason = reason;
    RequireObservationDelta(before,
                            CaptureObservations(harness_, wake_counter_),
                            {0, 0, 0, 0, 0, expected_accepted ? 1 : 0});
  }

  void VerifyCancellationState() {
    const iree_vm_cancel_reason_t expected_reason =
        model_.state == DriverState::kSuspended ? model_.cancel_reason
                                                : IREE_VM_CANCEL_REASON_NONE;
    Require(iree_vm_invocation_cancel_reason(harness_.invocation) ==
            expected_reason);
  }

  // Native modules, process, and invocation driven by the fuzzer.
  Harness harness_;
  // Counts every provider and cancellation wake delivery.
  WakeCounter wake_counter_;
  // Independent externally observable state model.
  DriverModel model_;
  // Whether provider continuation frames exceed invocation capacity.
  bool constrained_capacity_;
  // Whether the imported math provider rejects every function entry.
  bool fail_math_provider_;
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;
  const bool constrained_capacity = (data[0] & 1u) != 0;
  const bool fail_math_provider = (data[0] & 2u) != 0;
  InvocationStateMachine machine(constrained_capacity, fail_math_provider);
  const size_t operation_count =
      size - 1 < kMaximumOperationCount ? size - 1 : kMaximumOperationCount;
  for (size_t i = 0; i < operation_count; ++i) {
    machine.Apply(data[i + 1]);
  }
  machine.DrainAndProveReuse();
  return 0;
}
