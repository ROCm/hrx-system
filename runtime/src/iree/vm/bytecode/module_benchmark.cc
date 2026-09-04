// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/testing/benchmark.h"
#include "iree/vm/bytecode/launch_config_testdata.h"
#include "iree/vm/bytecode/layout.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;
constexpr uint64_t kConstructionBatchSize = 64;

struct BenchmarkContext {
  // Owned ref-type environment.
  iree_vm_environment_t* environment = nullptr;
  // Owned bytecode module.
  iree_vm_module_t* module = nullptr;
  // Owned linked program.
  iree_vm_program_t* program = nullptr;
  // Owned reusable invocation.
  iree_vm_invocation_t* invocation = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Borrowed process-bound scalar function.
  iree_vm_function_t function = iree_vm_function_null();
};

struct LoadedProgram {
  // Owned bytecode module.
  iree_vm_module_t* module = nullptr;
  // Owned linked program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
};

using BytecodeModuleCreateFn = decltype(&iree_vm_bytecode_module_create);

iree_vm_bytecode_module_storage_t MakeModuleStorage() {
  const iree_file_toc_t* files =
      iree_vm_bytecode_launch_config_testdata_create();
  return {
      iree_make_const_byte_span(files[0].data, files[0].size),
      iree_allocator_null(),
  };
}

void DeinitializeBenchmarkContext(BenchmarkContext* context) {
  iree_vm_process_release(context->process);
  iree_vm_invocation_free(context->invocation);
  iree_vm_program_release(context->program);
  iree_vm_module_release(context->module);
  iree_vm_environment_free(context->environment);
  *context = {};
}

iree_status_t InitializeBenchmarkContext(BenchmarkContext* context) {
  iree_status_t status = iree_vm_environment_allocate(iree_allocator_system(),
                                                      &context->environment);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(
        context->environment, IREE_SV("launch"), MakeModuleStorage(),
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
    status = iree_vm_process_lookup_function(
        context->process, IREE_SV("launch"), IREE_SV("launch_config"),
        &context->function);
  }
  if (!iree_status_is_ok(status)) DeinitializeBenchmarkContext(context);
  return status;
}

template <BytecodeModuleCreateFn create_module>
iree_status_t RunModuleCreateBenchmark(
    iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kConstructionBatchSize)) {
    iree_vm_module_t* modules[kConstructionBatchSize] = {};
    iree_host_size_t module_count = 0;
    while (module_count < IREE_ARRAYSIZE(modules) &&
           iree_status_is_ok(status)) {
      status = create_module(context.environment, IREE_SV("launch"),
                             MakeModuleStorage(), iree_allocator_system(),
                             &modules[module_count]);
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

iree_status_t RunModuleMapBenchmark(iree_benchmark_state_t* benchmark_state) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_bytecode_module_plan_t plan;
    status =
        iree_vm_bytecode_module_plan_build(MakeModuleStorage().contents, &plan);
    iree_optimization_barrier(plan.layout.image.header);
  }
  return status;
}

template <BytecodeModuleCreateFn create_module>
iree_status_t RunLoadAndBindBenchmark(iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, kConstructionBatchSize)) {
    LoadedProgram programs[kConstructionBatchSize] = {};
    iree_host_size_t program_count = 0;
    while (program_count < IREE_ARRAYSIZE(programs) &&
           iree_status_is_ok(status)) {
      LoadedProgram* program = &programs[program_count++];
      status = create_module(context.environment, IREE_SV("launch"),
                             MakeModuleStorage(), iree_allocator_system(),
                             &program->module);
      if (iree_status_is_ok(status)) {
        status = iree_vm_program_create(
            {program->module, iree_vm_module_span_empty()},
            iree_allocator_system(), &program->program);
      }
      if (iree_status_is_ok(status)) {
        status = iree_vm_process_create(
            program->program, context.invocation, iree_vm_variant_span_empty(),
            iree_allocator_system(), &program->process);
      }
      if (iree_status_is_ok(status)) {
        iree_vm_function_t function = iree_vm_function_null();
        status = iree_vm_process_lookup_function(
            program->process, IREE_SV("launch"), IREE_SV("launch_config"),
            &function);
        iree_optimization_barrier(function);
      }
    }
    iree_benchmark_pause_timing(benchmark_state);
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      iree_vm_process_release(programs[i].process);
      iree_vm_program_release(programs[i].program);
      iree_vm_module_release(programs[i].module);
    }
    iree_benchmark_resume_timing(benchmark_state);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

iree_status_t RunScalarLeafBenchmark(iree_benchmark_state_t* benchmark_state) {
  BenchmarkContext context;
  IREE_RETURN_IF_ERROR(InitializeBenchmarkContext(&context));

  uint64_t accumulator = 0;
  int32_t input = 1;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(input)};
    iree_vm_variant_t results[1] = {};
    status = iree_vm_invoke(context.invocation, context.function,
                            iree_vm_variant_span_from_array(arguments),
                            iree_vm_variant_span_from_array(results));
    int32_t result = 0;
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(results[0], &result);
    }
    accumulator += (uint32_t)result;
    input = input == 64 ? 1 : input + 1;
    iree_optimization_barrier(accumulator);
  }

  DeinitializeBenchmarkContext(&context);
  return status;
}

IREE_BENCHMARK_FN(BM_BytecodeModuleCreate) {
  return RunModuleCreateBenchmark<iree_vm_bytecode_module_create>(
      benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeModuleCreate);

IREE_BENCHMARK_FN(BM_BytecodeModuleCreateTrusted) {
  return RunModuleCreateBenchmark<iree_vm_bytecode_module_create_trusted>(
      benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeModuleCreateTrusted);

IREE_BENCHMARK_FN(BM_BytecodeModuleMap) {
  return RunModuleMapBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeModuleMap);

IREE_BENCHMARK_FN(BM_BytecodeLoadAndBind) {
  return RunLoadAndBindBenchmark<iree_vm_bytecode_module_create>(
      benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeLoadAndBind);

IREE_BENCHMARK_FN(BM_BytecodeLoadAndBindTrusted) {
  return RunLoadAndBindBenchmark<iree_vm_bytecode_module_create_trusted>(
      benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeLoadAndBindTrusted);

IREE_BENCHMARK_FN(BM_BytecodeScalarLeaf) {
  return RunScalarLeafBenchmark(benchmark_state);
}
IREE_BENCHMARK_REGISTER(BM_BytecodeScalarLeaf);

}  // namespace
