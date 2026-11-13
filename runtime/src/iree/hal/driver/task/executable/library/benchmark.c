// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/benchmark.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/task/device.h"
#include "iree/hal/driver/task/executable/executable.h"
#include "iree/hal/driver/task/executable/library/abi.h"
#include "iree/hal/driver/task/executable/loader.h"
#include "iree/hal/driver/task/executable/loaders/registration/init.h"
#include "iree/hal/memory/heap_pool.h"
#include "iree/io/file_contents.h"
#include "iree/task/topology.h"

IREE_FLAG(string, executable_loader, "",
          "Name of the executable loader to use.");
IREE_FLAG(string, executable_file, "",
          "Path to the executable library file to load.");

IREE_FLAG(int32_t, export_ordinal, 0, "Export ordinal to run.");

IREE_FLAG(int32_t, workgroup_count_x, 1,
          "X dimension of the workgroup count defining the number of\n"
          "workgroup invocations that will be run per benchmark iteration.\n"
          "This is the fastest-changing dimension.");
IREE_FLAG(int32_t, workgroup_count_y, 1,
          "Y dimension of the workgroup count defining the number of\n"
          "workgroup invocations that will be run per benchmark iteration.");
IREE_FLAG(int32_t, workgroup_count_z, 1,
          "Z dimension of the workgroup count defining the number of\n"
          "workgroup invocations that will be run per benchmark iteration.\n"
          "This is the slowest-changing dimension.");
IREE_FLAG(int32_t, workgroup_size_x, 1,
          "X dimension of the workgroup size passed to the executable.");
IREE_FLAG(int32_t, workgroup_size_y, 1,
          "Y dimension of the workgroup size passed to the executable.");
IREE_FLAG(int32_t, workgroup_size_z, 1,
          "Z dimension of the workgroup size passed to the executable.");

IREE_FLAG(int32_t, max_concurrency, 1,
          "Maximum available concurrency exposed to the dispatch.");

// Parsed parameters from flags.
// Used to construct the dispatch parameters for the benchmark invocation.
struct {
  int32_t constant_count;
  union {
    uint32_t ui32;
  } constants[IREE_HAL_EXECUTABLE_MAX_CONSTANT_COUNT];

  int32_t binding_count;
  iree_string_view_t bindings[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
} dispatch_params = {
    .constant_count = 0,
    .binding_count = 0,
};

static iree_status_t parse_constant(iree_string_view_t flag_name, void* storage,
                                    iree_string_view_t value) {
  IREE_ASSERT_LE(dispatch_params.constant_count + 1,
                 IREE_ARRAYSIZE(dispatch_params.constants),
                 "too many push constants");
  dispatch_params.constants[dispatch_params.constant_count++].ui32 =
      atoi(value.data);
  return iree_ok_status();
}
static void print_constant(iree_string_view_t flag_name, void* storage,
                           FILE* file) {
  if (dispatch_params.constant_count == 0) {
    fprintf(file, "# --%.*s=[integer value]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < dispatch_params.constant_count; ++i) {
    fprintf(file, "--%.*s=%u", (int)flag_name.size, flag_name.data,
            dispatch_params.constants[i].ui32);
    if (i < dispatch_params.constant_count - 1) {
      fprintf(file, "\n");
    }
  }
}
IREE_FLAG_CALLBACK(parse_constant, print_constant, &dispatch_params,
                   constant_callback,
                   "Appends a uint32_t push constant value.\n");

static iree_status_t parse_binding(iree_string_view_t flag_name, void* storage,
                                   iree_string_view_t value) {
  IREE_ASSERT_LE(dispatch_params.binding_count + 1,
                 IREE_ARRAYSIZE(dispatch_params.bindings), "too many bindings");
  dispatch_params.bindings[dispatch_params.binding_count++] = value;
  return iree_ok_status();
}
static void print_binding(iree_string_view_t flag_name, void* storage,
                          FILE* file) {
  if (dispatch_params.binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype[=values]\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < dispatch_params.binding_count; ++i) {
    const iree_string_view_t binding_str = dispatch_params.bindings[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_str.size, binding_str.data);
  }
}
IREE_FLAG_CALLBACK(
    parse_binding, print_binding, &dispatch_params, binding,
    "Appends a binding to the dispatch parameters.\n"
    "Bindings are defined by their shape, element type, and their data.\n"
    "Examples:\n"
    "  # 16 4-byte elements zero-initialized:\n"
    "  --binding=2x8xi32\n"
    "  # 10000 bytes all initialized to 123:\n"
    "  --binding=10000xi8=123\n"
    "  # 2 4-byte floating-point values with contents [[1.4], [2.1]]:\n"
    "  --binding=2x1xf32=1.4,2.1");

// Creates the owner of the executable's canonical family. Dispatches execute
// inline; the executor stays idle and the proactor needs no polling runner.
// Keep the topology setup stack frame out of the timed dispatch function.
static IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_hal_executable_library_create_device(
    iree_hal_executable_loader_t* executable_loader,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(1, &topology);
  iree_task_executor_options_t executor_options;
  iree_task_executor_options_initialize(&executor_options);
  iree_task_executor_t* executor = NULL;
  iree_status_t status = iree_task_executor_create(executor_options, &topology,
                                                   host_allocator, &executor);
  iree_task_topology_deinitialize(&topology);
  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    const iree_async_proactor_pool_options_t proactor_options = {0};
    status = iree_async_proactor_pool_create(1, NULL, proactor_options,
                                             host_allocator, &proactor_pool);
  }
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(host_allocator, &device_allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_task_device_params_t device_params;
    iree_hal_task_device_params_initialize(&device_params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_task_device_create(
        IREE_SV("executable-library-benchmark"), &device_params, 1, &executor,
        1, &executable_loader, device_allocator, &create_params, host_allocator,
        out_device);
  }
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  iree_task_executor_release(executor);
  return status;
}

// Measures only inline dispatch. Setup and teardown stay outside the timer.
static iree_status_t iree_hal_executable_library_dispatch(
    iree_benchmark_state_t* benchmark_state,
    iree_hal_task_executable_t* task_executable,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    iree_byte_span_t local_memory) {
  // Execute benchmark the workgroup invocation.
  // Note that each iteration runs through the whole grid as it's important that
  // we are testing the memory access patterns: if we just ran the same single
  // tile processing the same exact region of memory over and over we are not
  // testing cache effects.
  int64_t dispatch_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, /*batch_count=*/1)) {
    IREE_RETURN_IF_ERROR(iree_hal_task_executable_issue_dispatch_inline(
        task_executable, FLAG_export_ordinal, dispatch_state, 0, local_memory));
    ++dispatch_count;
  }

  // To get a total time per invocation we set the item count to the total
  // invocations dispatched. That gives us both total dispatch and single
  // invocation times in the reporter output.
  int64_t total_invocations =
      dispatch_count * dispatch_state->workgroup_count_x *
      dispatch_state->workgroup_count_y * dispatch_state->workgroup_count_z;
  iree_benchmark_set_items_processed(benchmark_state, total_invocations);

  return iree_ok_status();
}

static iree_status_t iree_hal_executable_library_run(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  (void)benchmark_def;
  if (!FLAG_executable_loader[0] || !FLAG_executable_file[0]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "benchmark requires --executable_loader and --executable_file");
  }
  iree_allocator_t host_allocator = benchmark_state->host_allocator;

  iree_hal_executable_loader_t* executable_loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_executable_loader_by_name(
      iree_make_cstring_view(FLAG_executable_loader), host_allocator,
      &executable_loader));
  iree_hal_device_t* device = NULL;
  iree_status_t status = iree_hal_executable_library_create_device(
      executable_loader, host_allocator, &device);
  iree_hal_executable_loader_release(executable_loader);

  iree_hal_pool_t* binding_pool = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_host_access_t host_access = {
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .modes =
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
    };
    status = iree_hal_heap_pool_create(host_access, host_allocator,
                                       host_allocator, &binding_pool);
  }

  // Select the highest-priority target advertised by the chosen loader.
  const iree_hal_executable_target_t* target = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_executable_target_selection_t selection = {0};
    const iree_hal_executable_target_selection_result_t result =
        iree_hal_device_spec_select_executable_target(
            iree_hal_device_spec(device), &selection);
    if (result.outcome !=
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "selected executable loader did not advertise a unique target");
    } else {
      target = result.target;
    }
  }

  iree_io_file_contents_t* file_contents = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_io_file_contents_read(iree_make_cstring_view(FLAG_executable_file),
                                   host_allocator, &file_contents);
  }
  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t executable_params;
    iree_hal_executable_load_params_initialize(&executable_params);
    executable_params.flags |=
        IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION;
    executable_params.executable_data = file_contents->const_buffer;
    status = iree_hal_executable_load(iree_hal_device_queue_family(device, 0),
                                      target, &executable_params, &executable);
  }
  iree_hal_task_executable_t* task_executable = NULL;
  if (iree_status_is_ok(status)) {
    task_executable = iree_hal_task_executable_cast(executable);
    if (FLAG_export_ordinal < 0 ||
        FLAG_export_ordinal >= task_executable->export_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE, "export ordinal %d out of range [0, %zu)",
          FLAG_export_ordinal, task_executable->export_count);
    }
  }

  // Allocate workgroup-local memory that each invocation can use.
  iree_byte_span_t local_memory = iree_byte_span_empty();
  if (iree_status_is_ok(status) && task_executable->dispatch_attrs) {
    iree_host_size_t local_memory_size =
        task_executable->dispatch_attrs[FLAG_export_ordinal]
            .local_memory_pages *
        IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;
    if (local_memory_size > 0) {
      status = iree_allocator_malloc(host_allocator, local_memory_size,
                                     (void**)&local_memory.data);
      local_memory.data_length = local_memory_size;
    }
  }

  // Allocate storage for buffers and populate them.
  // They only need to remain valid for the duration of the invocation and all
  // memory accessed by the invocation will come from here.
  iree_host_size_t buffer_view_count = 0;
  iree_hal_buffer_view_t* buffer_views[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  void* binding_ptrs[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  size_t binding_lengths[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < dispatch_params.binding_count; ++i) {
    status =
        iree_hal_buffer_view_parse(dispatch_params.bindings[i], binding_pool,
                                   /*min_alignment=*/0, &buffer_views[i]);
    if (iree_status_is_ok(status)) {
      ++buffer_view_count;
      iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_views[i]);
      iree_device_size_t buffer_length =
          iree_hal_buffer_view_byte_length(buffer_views[i]);
      iree_hal_buffer_mapping_t buffer_mapping = {{0}};
      status = iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, buffer_length, &buffer_mapping);
      binding_ptrs[i] = buffer_mapping.contents.data;
      binding_lengths[i] = (size_t)buffer_mapping.contents.data_length;
    }
  }

  // Setup dispatch state.
  const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_count_x = FLAG_workgroup_count_x,
      .workgroup_count_y = FLAG_workgroup_count_y,
      .workgroup_count_z = FLAG_workgroup_count_z,
      .workgroup_size_x = FLAG_workgroup_size_x,
      .workgroup_size_y = FLAG_workgroup_size_y,
      .workgroup_size_z = FLAG_workgroup_size_z,
      .max_concurrency = FLAG_max_concurrency,
      .constants = {(const uint8_t*)&dispatch_params.constants[0].ui32,
                    (size_t)dispatch_params.constant_count * sizeof(uint32_t)},
      .binding_count = dispatch_params.binding_count,
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
  };

  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_dispatch(
        benchmark_state, task_executable, &dispatch_state, local_memory);
  }

  // Deallocate buffers.
  for (iree_host_size_t i = 0; i < buffer_view_count; ++i) {
    iree_hal_buffer_view_release(buffer_views[i]);
  }
  iree_allocator_free(host_allocator, local_memory.data);

  // Unload.
  iree_hal_executable_release(executable);
  iree_hal_pool_release(binding_pool);
  iree_hal_device_release(device);
  iree_io_file_contents_free(file_contents);

  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "executable_library_benchmark",
      "Benchmarks a single entry point within an executable library.\n"
      "The parameters used can be inferred from the entry point\n"
      "`hal.interface` and dispatches to it.\n"
      "\n"
      "Executable artifacts can be emitted directly by producer toolchains or\n"
      "generated by vendor tools.\n"
      "\n"
      "Note that this tool is intentionally low level: you must specify all\n"
      "of the push constant/binding parameters precisely as they are expected\n"
      "by the executable. This tool invokes executable exports inline,\n"
      "bypassing HAL queue submission and task scheduling.\n"
      "\n"
      "Example --flagfile:\n"
      "  --executable_loader=embedded-elf\n"
      "  --executable_file=iree/hal/driver/task/executable/elf/testdata/"
      "elementwise_mul_x86_64.so\n"
      "  --export_ordinal=0\n"
      "  --workgroup_count_x=1\n"
      "  --workgroup_count_y=1\n"
      "  --workgroup_count_z=1\n"
      "  --workgroup_size_x=1\n"
      "  --workgroup_size_y=1\n"
      "  --workgroup_size_z=1\n"
      "  --binding=4xf32=1,2,3,4\n"
      "  --binding=4xf32=100,200,300,400\n"
      "  --binding=4xf32=0,0,0,0\n"
      "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);

  // TODO(benvanik): override these with our own flags.
  iree_benchmark_def_t benchmark_def = {
      .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
               IREE_BENCHMARK_FLAG_USE_REAL_TIME,
      .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
      .minimum_duration_ns = 0,
      .iteration_count = 0,
      .run = iree_hal_executable_library_run,
  };
  iree_benchmark_register(iree_make_cstring_view("dispatch"), &benchmark_def);

  return iree_benchmark_run_specified() ? 0 : 1;
}
