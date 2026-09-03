// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/vm/invocation.h"
#include "iree/vm/invocation_test_module.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"

#define IREE_VM_FUZZ_CHECK(condition)   \
  do {                                  \
    if (!(condition)) __builtin_trap(); \
  } while (0)

#define IREE_VM_FUZZ_CHECK_OK(expression) \
  do {                                    \
    iree_status_t status = (expression);  \
    if (!iree_status_is_ok(status)) {     \
      __builtin_trap();                   \
    }                                     \
  } while (0)

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

class InputCursor {
 public:
  InputCursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  bool has_remaining() const { return position_ < size_; }

  uint8_t TakeU8() { return has_remaining() ? data_[position_++] : uint8_t{0}; }

  uint32_t TakeU32() {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      value |= uint32_t{TakeU8()} << (i * 8);
    }
    return value;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t position_ = 0;
};

template <size_t N>
iree_vm_variant_span_t MakeVariantSpan(
    std::array<iree_vm_variant_t, N>& variants) {
  return iree_vm_variant_span_from_ptr(variants.data(), variants.size());
}

iree_vm_function_t LookupFunction(iree_vm_process_t* process,
                                  iree_string_view_t name) {
  iree_vm_function_t function = iree_vm_function_null();
  IREE_VM_FUZZ_CHECK_OK(iree_vm_process_lookup_function(
      process, IREE_SV("invocation.test"), name, &function));
  return function;
}

int32_t ReadI32(iree_vm_variant_t variant) {
  int32_t value = 0;
  IREE_VM_FUZZ_CHECK_OK(iree_vm_i32_from_variant(variant, &value));
  return value;
}

void CheckStatusCode(iree_status_t status, iree_status_code_t expected_code) {
  const iree_status_code_t actual_code = iree_status_code(status);
  iree_status_free(status);
  IREE_VM_FUZZ_CHECK(actual_code == expected_code);
}

void InvokeAddLike(iree_vm_invocation_t* invocation,
                   iree_vm_function_t function, uint32_t lhs, uint32_t rhs) {
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(static_cast<int32_t>(lhs)),
      iree_vm_variant_from_i32(static_cast<int32_t>(rhs)),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_start(invocation, function, MakeVariantSpan(arguments),
                               MakeVariantSpan(results), {}, &outcome));
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  IREE_VM_FUZZ_CHECK(iree_vm_variant_is_empty(arguments[0]));
  IREE_VM_FUZZ_CHECK(iree_vm_variant_is_empty(arguments[1]));
  IREE_VM_FUZZ_CHECK(static_cast<uint32_t>(ReadI32(results[0])) == lhs + rhs);
}

void InvokeYieldingI32(iree_vm_invocation_t* invocation,
                       iree_vm_function_t function, uint32_t value,
                       bool change_result_storage) {
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(static_cast<int32_t>(value)),
  };
  std::array<iree_vm_variant_t, 1> first_results = {
      iree_vm_variant_from_i64(INT64_C(0x12345678)),
  };
  std::array<iree_vm_variant_t, 1> second_results = first_results;
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_start(invocation, function, MakeVariantSpan(arguments),
                               MakeVariantSpan(first_results), {}, &outcome));
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  IREE_VM_FUZZ_CHECK(iree_vm_variant_is_empty(arguments[0]));
  IREE_VM_FUZZ_CHECK_OK(iree_vm_invocation_resume(
      invocation, MakeVariantSpan(first_results), &outcome));
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  iree_vm_variant_span_t terminal_results =
      change_result_storage ? MakeVariantSpan(second_results)
                            : MakeVariantSpan(first_results);
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_resume(invocation, terminal_results, &outcome));
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  IREE_VM_FUZZ_CHECK(static_cast<uint32_t>(ReadI32(terminal_results.data[0])) ==
                     value + 1);
}

void InvokeFunctionRoundTrip(iree_vm_invocation_t* invocation,
                             iree_vm_process_t* process,
                             iree_vm_function_t return_function,
                             iree_vm_function_t call_function, uint32_t lhs,
                             uint32_t rhs) {
  std::array<iree_vm_variant_t, 1> function_results = {
      iree_vm_variant_empty(),
  };
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_VM_FUZZ_CHECK_OK(iree_vm_invocation_start(
      invocation, return_function, iree_vm_variant_span_empty(),
      MakeVariantSpan(function_results), {}, &outcome));
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_function_ref_from_variant(function_results[0], &function_ref));
  iree_vm_function_t rebound_function = iree_vm_function_null();
  IREE_VM_FUZZ_CHECK_OK(iree_vm_function_from_function_ref(
      process, function_ref, &rebound_function));

  std::array<iree_vm_variant_t, 3> arguments = {
      function_results[0],
      iree_vm_variant_from_i32(static_cast<int32_t>(lhs)),
      iree_vm_variant_from_i32(static_cast<int32_t>(rhs)),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  IREE_VM_FUZZ_CHECK_OK(iree_vm_invocation_start(
      invocation, call_function, MakeVariantSpan(arguments),
      MakeVariantSpan(results), {}, &outcome));
  IREE_VM_FUZZ_CHECK(static_cast<uint32_t>(ReadI32(results[0])) == lhs + rhs);
  InvokeAddLike(invocation, rebound_function, lhs, rhs);
}

void InvokeRefRoundTrip(iree_vm_invocation_t* invocation,
                        iree_vm_function_t function, bool may_yield) {
  int destroy_count = 0;
  iree_vm_invocation_test_object_t object = {};
  iree_vm_invocation_test_object_initialize(&destroy_count, &object);
  void* object_ptr = &object;
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_ptr_move(&object_ptr,
                                    iree_vm_invocation_test_object_type()),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_SUSPENDED;
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_start(invocation, function, MakeVariantSpan(arguments),
                               MakeVariantSpan(results), {}, &outcome));
  if (may_yield) {
    IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
    IREE_VM_FUZZ_CHECK_OK(iree_vm_invocation_resume(
        invocation, MakeVariantSpan(results), &outcome));
  }
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  IREE_VM_FUZZ_CHECK(iree_vm_variant_ref_isa(
      results[0], iree_vm_invocation_test_object_type()));
  iree_vm_variant_reset(&results[0]);
  IREE_VM_FUZZ_CHECK(destroy_count == 1);
}

void InvokeCancelledRef(iree_vm_invocation_t* invocation,
                        iree_vm_function_t function,
                        iree_vm_cancel_reason_t reason) {
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
  iree_vm_execution_outcome_t outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_start(invocation, function, MakeVariantSpan(arguments),
                               MakeVariantSpan(results), {}, &outcome));
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  IREE_VM_FUZZ_CHECK(iree_vm_invocation_request_cancel(invocation, reason));
  iree_status_t status =
      iree_vm_invocation_resume(invocation, MakeVariantSpan(results), &outcome);
  CheckStatusCode(status, reason == IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED
                              ? IREE_STATUS_DEADLINE_EXCEEDED
                              : IREE_STATUS_CANCELLED);
  IREE_VM_FUZZ_CHECK(outcome == IREE_VM_EXECUTION_OUTCOME_SUSPENDED);
  IREE_VM_FUZZ_CHECK(results[0].payload == original_result.payload);
  IREE_VM_FUZZ_CHECK(results[0].metadata == original_result.metadata);
  IREE_VM_FUZZ_CHECK(destroy_count == 1);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  InputCursor input(data, size);
  iree_vm_invocation_test_counters_t counters = {};
  iree_vm_invocation_test_module_t module = {};
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_invocation_test_module_initialize(&counters, &module));
  iree_vm_program_t* program = nullptr;
  IREE_VM_FUZZ_CHECK_OK(
      iree_vm_program_create({&module.base, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  alignas(iree_max_align_t) std::array<uint8_t, kInvocationStorageSize>
      invocation_storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  IREE_VM_FUZZ_CHECK_OK(iree_vm_invocation_initialize(
      iree_make_byte_span(invocation_storage.data(), invocation_storage.size()),
      &invocation));
  iree_vm_process_create_outcome_t process_outcome = {};
  IREE_VM_FUZZ_CHECK_OK(iree_vm_process_create_start(
      program, invocation, iree_vm_variant_span_empty(), {},
      iree_allocator_system(), &process_outcome));
  IREE_VM_FUZZ_CHECK(process_outcome.execution_outcome ==
                     IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  iree_vm_process_t* process = process_outcome.process;

  const iree_vm_function_t add = LookupFunction(process, IREE_SV("add"));
  const iree_vm_function_t call_function =
      LookupFunction(process, IREE_SV("call_function"));
  const iree_vm_function_t call_import =
      LookupFunction(process, IREE_SV("call_import"));
  const iree_vm_function_t call_local =
      LookupFunction(process, IREE_SV("call_local"));
  const iree_vm_function_t echo_ref =
      LookupFunction(process, IREE_SV("echo_ref"));
  const iree_vm_function_t return_local =
      LookupFunction(process, IREE_SV("return_local"));
  const iree_vm_function_t yield_ref =
      LookupFunction(process, IREE_SV("yield_ref"));
  const iree_vm_function_t yield_twice =
      LookupFunction(process, IREE_SV("yield_twice"));

  while (input.has_remaining()) {
    const uint8_t operation = input.TakeU8() % 8;
    const uint32_t lhs = input.TakeU32();
    const uint32_t rhs = input.TakeU32();
    switch (operation) {
      case 0:
        InvokeAddLike(invocation, add, lhs, rhs);
        break;
      case 1:
        InvokeAddLike(invocation, call_local, lhs, rhs);
        break;
      case 2:
        InvokeAddLike(invocation, call_import, lhs, rhs);
        break;
      case 3:
        InvokeYieldingI32(invocation, yield_twice, lhs, (rhs & 1) != 0);
        break;
      case 4:
        InvokeFunctionRoundTrip(invocation, process, return_local,
                                call_function, lhs, rhs);
        break;
      case 5:
        InvokeRefRoundTrip(invocation, echo_ref, false);
        break;
      case 6:
        InvokeRefRoundTrip(invocation, yield_ref, true);
        break;
      case 7:
        InvokeCancelledRef(invocation, yield_ref,
                           (rhs & 1) ? IREE_VM_CANCEL_REASON_DEADLINE_EXCEEDED
                                     : IREE_VM_CANCEL_REASON_CANCELLED);
        break;
    }
  }

  iree_vm_process_release(process);
  iree_vm_invocation_deinitialize(invocation);
  iree_vm_program_release(program);
  iree_vm_module_release(&module.base);
  IREE_VM_FUZZ_CHECK(counters.destroy_count == 1);
  return 0;
}
