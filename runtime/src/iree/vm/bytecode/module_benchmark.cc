// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/benchmark.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/process.h"

namespace iree::vm::bytecode::testing {
namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

// Amortizes timer suspension around construction-object teardown.
constexpr uint64_t kConstructionBatchSize = 64;

struct BenchmarkContext {
  // Immutable borrowed bytecode image storage.
  std::vector<uint8_t> image;
  // Owned type environment used by repeated module construction.
  iree_vm_environment_t* environment = nullptr;
  // Owned bytecode module.
  iree_vm_module_t* module = nullptr;
  // Owned immutable linked program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation.
  iree_vm_invocation_t* invocation = nullptr;
  // Borrowed process-bound launch configuration function.
  iree_vm_function_t function = iree_vm_function_null();
  // Borrowed process-bound empty decomposition function.
  iree_vm_function_t empty_function = iree_vm_function_null();
  // Borrowed process-bound full-signature decomposition function.
  iree_vm_function_t noop_function = iree_vm_function_null();
};

void DeinitializeBenchmarkContext(BenchmarkContext* context) {
  iree_vm_invocation_free(context->invocation);
  iree_vm_process_release(context->process);
  iree_vm_program_release(context->program);
  iree_vm_module_release(context->module);
  iree_vm_environment_free(context->environment);
  context->invocation = nullptr;
  context->process = nullptr;
  context->program = nullptr;
  context->module = nullptr;
  context->environment = nullptr;
  context->function = iree_vm_function_null();
  context->empty_function = iree_vm_function_null();
  context->noop_function = iree_vm_function_null();
}

iree_status_t InvokeLaunchConfig(iree_vm_invocation_t* invocation,
                                 iree_vm_function_t function,
                                 uint32_t row_count,
                                 iree_vm_variant_span_t results) {
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(static_cast<int32_t>(row_count)),
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

iree_status_t InitializeBenchmarkContext(BenchmarkContext* context) {
  context->image = BuildLaunchConfigModuleImage();
  iree_status_t status = iree_vm_environment_allocate(iree_allocator_system(),
                                                      &context->environment);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        context->environment, IREE_SV("launch"),
        {iree_make_const_byte_span(context->image.data(),
                                   context->image.size()),
         iree_allocator_null()},
        iree_allocator_system(), &context->module);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_program_create({context->module, iree_vm_module_span_empty()},
                               iree_allocator_system(), &context->program);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invocation_allocate(
        kInvocationStorageSize, iree_allocator_system(), &context->invocation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(context->program, context->invocation,
                                    iree_vm_variant_span_empty(),
                                    iree_allocator_system(), &context->process);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_process_lookup_function(context->process, IREE_SV("launch"),
                                        IREE_SV("decode"), &context->function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        context->process, IREE_SV("launch"), IREE_SV("empty"),
        &context->empty_function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(context->process,
                                             IREE_SV("launch"), IREE_SV("noop"),
                                             &context->noop_function);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t results[11] = {};
    status = InvokeLaunchConfig(context->invocation, context->function, 64,
                                iree_vm_variant_span_from_array(results));
    const int64_t expected[] = {128, 1, 1, 1, 1, 1, 1, 1, 1, 32, 256};
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(results) && iree_status_is_ok(status); ++i) {
      int64_t value = 0;
      status = iree_vm_i64_from_variant(results[i], &value);
      if (iree_status_is_ok(status) && value != expected[i]) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "launch result does not match the oracle");
      }
    }
  }
  if (!iree_status_is_ok(status)) DeinitializeBenchmarkContext(context);
  return status;
}

iree_status_t RunModuleCreateBenchmark(
    iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kConstructionBatchSize)) {
    iree_vm_module_t* modules[kConstructionBatchSize];
    iree_host_size_t module_count = 0;
    while (module_count < IREE_ARRAYSIZE(modules) &&
           iree_status_is_ok(status)) {
      status = iree_vm_bytecode_module_create(
          context.environment, IREE_SV("launch"),
          {iree_make_const_byte_span(context.image.data(),
                                     context.image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &modules[module_count]);
      if (iree_status_is_ok(status)) ++module_count;
    }
    iree_benchmark_pause_timing(benchmark_state);
    for (iree_host_size_t i = 0; i < module_count; ++i) {
      iree_vm_module_release(modules[i]);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

iree_status_t RunProgramCreateBenchmark(
    iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kConstructionBatchSize)) {
    iree_vm_program_t* programs[kConstructionBatchSize];
    iree_host_size_t program_count = 0;
    while (program_count < IREE_ARRAYSIZE(programs) &&
           iree_status_is_ok(status)) {
      status = iree_vm_program_create(
          {context.module, iree_vm_module_span_empty()},
          iree_allocator_system(), &programs[program_count]);
      if (iree_status_is_ok(status)) ++program_count;
    }
    iree_benchmark_pause_timing(benchmark_state);
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      iree_vm_program_release(programs[i]);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

iree_status_t RunProcessCreateBenchmark(
    iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kConstructionBatchSize)) {
    iree_vm_process_t* processes[kConstructionBatchSize];
    iree_host_size_t process_count = 0;
    while (process_count < IREE_ARRAYSIZE(processes) &&
           iree_status_is_ok(status)) {
      status = iree_vm_process_create(
          context.program, context.invocation, iree_vm_variant_span_empty(),
          iree_allocator_system(), &processes[process_count]);
      if (iree_status_is_ok(status)) ++process_count;
    }
    iree_benchmark_pause_timing(benchmark_state);
    for (iree_host_size_t i = 0; i < process_count; ++i) {
      iree_vm_process_release(processes[i]);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

enum class BenchmarkMode {
  kRaw,
  kChecked,
  kStack16KChecked,
};

iree_status_t RunEmptyBenchmark(iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    status = iree_vm_invoke(context.invocation, context.empty_function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_empty());
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

iree_status_t RunNoopBenchmark(iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  uint32_t row_count = 1;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_variant_t results[11];
    status =
        InvokeLaunchConfig(context.invocation, context.noop_function, row_count,
                           iree_vm_variant_span_from_array(results));
    row_count = row_count == 64 ? 1 : row_count + 1;
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

iree_status_t RunLaunchConfigBenchmark(iree_benchmark_state_t* benchmark_state,
                                       BenchmarkMode mode) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  uint64_t accumulator = 0;
  uint32_t row_count = 1;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_variant_t results[11];
    status = mode != BenchmarkMode::kStack16KChecked
                 ? InvokeLaunchConfig(context.invocation, context.function,
                                      row_count,
                                      iree_vm_variant_span_from_array(results))
                 : InvokeLaunchConfigOnStack(
                       context.function, row_count,
                       iree_vm_variant_span_from_array(results));
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(results) && iree_status_is_ok(status); ++i) {
      if (mode == BenchmarkMode::kRaw) {
        accumulator += results[i].payload;
      } else {
        int64_t value = 0;
        status = iree_vm_i64_from_variant(results[i], &value);
        accumulator += static_cast<uint64_t>(value);
      }
    }
    row_count = row_count == 64 ? 1 : row_count + 1;
    iree_optimization_barrier(accumulator);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

IREE_BENCHMARK_FN(BM_BytecodeLaunchConfigReuseRaw) {
  return RunLaunchConfigBenchmark(benchmark_state, BenchmarkMode::kRaw);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeLaunchConfigReuseRaw);

IREE_BENCHMARK_FN(BM_BytecodeModuleCreate) {
  return RunModuleCreateBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeModuleCreate);

IREE_BENCHMARK_FN(BM_BytecodeProgramCreate) {
  return RunProgramCreateBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeProgramCreate);

IREE_BENCHMARK_FN(BM_BytecodeProcessCreate) {
  return RunProcessCreateBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeProcessCreate);

IREE_BENCHMARK_FN(BM_BytecodeLaunchConfigReuseChecked) {
  return RunLaunchConfigBenchmark(benchmark_state, BenchmarkMode::kChecked);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeLaunchConfigReuseChecked);

IREE_BENCHMARK_FN(BM_BytecodeLaunchConfigStack16KChecked) {
  return RunLaunchConfigBenchmark(benchmark_state,
                                  BenchmarkMode::kStack16KChecked);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeLaunchConfigStack16KChecked);

IREE_BENCHMARK_FN(BM_BytecodeEmptyReuse) {
  return RunEmptyBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeEmptyReuse);

IREE_BENCHMARK_FN(BM_BytecodeNoopFullSignatureReuse) {
  return RunNoopBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeNoopFullSignatureReuse);

}  // namespace
}  // namespace iree::vm::bytecode::testing
