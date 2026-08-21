// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/testing/benchmark.h"
#include "iree/vm/execution_test_provider.h"
#include "iree/vm/process.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

struct BenchmarkContext {
  // Application module observations required by the native provider.
  iree_vm_execution_test_counters_t application_counters = {};
  // Math module observations required by the native provider.
  iree_vm_execution_test_counters_t math_counters = {};
  // Owned application module.
  iree_vm_module_t* application_module = nullptr;
  // Owned math module.
  iree_vm_module_t* math_module = nullptr;
  // Owned immutable program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation.
  iree_vm_invocation_t* invocation = nullptr;
  // Borrowed process-bound launch configuration function.
  iree_vm_function_t function = iree_vm_function_null();
};

void DeinitializeBenchmarkContext(BenchmarkContext* context) {
  iree_vm_invocation_free(context->invocation);
  iree_vm_process_release(context->process);
  iree_vm_program_release(context->program);
  iree_vm_module_release(context->application_module);
  iree_vm_module_release(context->math_module);
  std::memset(context, 0, sizeof(*context));
}

iree_status_t InitializeBenchmarkContext(BenchmarkContext* context) {
  iree_status_t status = iree_vm_execution_test_module_create(
      IREE_VM_EXECUTION_TEST_MODULE_KIND_APPLICATION, {},
      &context->application_counters, iree_allocator_system(),
      &context->application_module);
  if (iree_status_is_ok(status)) {
    status = iree_vm_execution_test_module_create(
        IREE_VM_EXECUTION_TEST_MODULE_KIND_MATH, {}, &context->math_counters,
        iree_allocator_system(), &context->math_module);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_module_t* libraries[] = {context->math_module};
    status = iree_vm_program_create({context->application_module,
                                     iree_vm_module_span_from_array(libraries)},
                                    iree_allocator_system(), &context->program);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invocation_allocate(
        kInvocationStorageSize, iree_allocator_system(), &context->invocation);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(42)};
    status = iree_vm_process_create(context->program, context->invocation,
                                    iree_vm_variant_span_from_array(arguments),
                                    iree_allocator_system(), &context->process);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        context->process, IREE_SV("execution.app"), IREE_SV("launch_config"),
        &context->function);
  }
  if (!iree_status_is_ok(status)) DeinitializeBenchmarkContext(context);
  return status;
}

iree_status_t InvokeLaunchConfig(iree_vm_invocation_t* invocation,
                                 iree_vm_function_t function,
                                 uint32_t row_count,
                                 iree_vm_variant_span_t results) {
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32((int32_t)row_count),
      iree_vm_variant_from_bf16_bits(0x4000),
  };
  return iree_vm_invoke(invocation, function,
                        iree_vm_variant_span_from_array(arguments), results);
}

IREE_ATTRIBUTE_NOINLINE iree_status_t
InvokeLaunchConfigOnStack(iree_vm_function_t function, uint32_t row_count,
                          iree_vm_variant_span_t results) {
  alignas(max_align_t) uint8_t storage[kInvocationStorageSize];
  iree_vm_invocation_t* invocation = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
      iree_make_byte_span(storage, sizeof(storage)), &invocation));
  iree_status_t status =
      InvokeLaunchConfig(invocation, function, row_count, results);
  iree_vm_invocation_deinitialize(invocation);
  return status;
}

enum class BenchmarkMode {
  kReuseRaw,
  kReuseChecked,
  kStack16KChecked,
};

iree_status_t RunLaunchConfigBenchmark(iree_benchmark_state_t* benchmark_state,
                                       BenchmarkMode benchmark_mode) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  uint64_t accumulator = 0;
  uint32_t row_count = 1;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_variant_t results[11];
    status = benchmark_mode != BenchmarkMode::kStack16KChecked
                 ? InvokeLaunchConfig(context.invocation, context.function,
                                      row_count,
                                      iree_vm_variant_span_from_array(results))
                 : InvokeLaunchConfigOnStack(
                       context.function, row_count,
                       iree_vm_variant_span_from_array(results));
    if (benchmark_mode == BenchmarkMode::kReuseRaw) {
      for (iree_host_size_t i = 0;
           i < IREE_ARRAYSIZE(results) && iree_status_is_ok(status); ++i) {
        accumulator += results[i].payload;
      }
    } else {
      for (iree_host_size_t i = 0;
           i < IREE_ARRAYSIZE(results) && iree_status_is_ok(status); ++i) {
        int64_t value = 0;
        status = iree_vm_i64_from_variant(results[i], &value);
        accumulator += (uint64_t)value;
      }
    }
    row_count = row_count == 64 ? 1 : row_count + 1;
    iree_optimization_barrier(accumulator);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

IREE_BENCHMARK_FN(BM_LaunchConfigReuseRaw) {
  return RunLaunchConfigBenchmark(benchmark_state, BenchmarkMode::kReuseRaw);
}
IREE_BENCHMARK_REGISTER(BM_LaunchConfigReuseRaw);

IREE_BENCHMARK_FN(BM_LaunchConfigReuseChecked) {
  return RunLaunchConfigBenchmark(benchmark_state,
                                  BenchmarkMode::kReuseChecked);
}
IREE_BENCHMARK_REGISTER(BM_LaunchConfigReuseChecked);

IREE_BENCHMARK_FN(BM_LaunchConfigStack16KChecked) {
  return RunLaunchConfigBenchmark(benchmark_state,
                                  BenchmarkMode::kStack16KChecked);
}
IREE_BENCHMARK_REGISTER(BM_LaunchConfigStack16KChecked);

}  // namespace
