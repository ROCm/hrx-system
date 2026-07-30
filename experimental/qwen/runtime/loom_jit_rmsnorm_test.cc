// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "experimental/qwen/runtime/loom_jit.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/init.h"
#include "iree/io/memory_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/numpy_io.h"

namespace {

template <typename T, void (*Release)(T*)>
struct HandleDeleter {
  void operator()(T* value) const { Release(value); }
};

using ProactorPoolPtr =
    std::unique_ptr<iree_async_proactor_pool_t,
                    HandleDeleter<iree_async_proactor_pool_t,
                                  iree_async_proactor_pool_release>>;
using FrontierTrackerPtr =
    std::unique_ptr<iree_async_frontier_tracker_t,
                    HandleDeleter<iree_async_frontier_tracker_t,
                                  iree_async_frontier_tracker_release>>;
using DevicePtr =
    std::unique_ptr<iree_hal_device_t,
                    HandleDeleter<iree_hal_device_t, iree_hal_device_release>>;
using DeviceGroupPtr = std::unique_ptr<
    iree_hal_device_group_t,
    HandleDeleter<iree_hal_device_group_t, iree_hal_device_group_release>>;
using BufferPtr =
    std::unique_ptr<iree_hal_buffer_t,
                    HandleDeleter<iree_hal_buffer_t, iree_hal_buffer_release>>;
using BufferViewPtr = std::unique_ptr<
    iree_hal_buffer_view_t,
    HandleDeleter<iree_hal_buffer_view_t, iree_hal_buffer_view_release>>;
using CommandBufferPtr = std::unique_ptr<
    iree_hal_command_buffer_t,
    HandleDeleter<iree_hal_command_buffer_t, iree_hal_command_buffer_release>>;
using SemaphorePtr = std::unique_ptr<
    iree_hal_semaphore_t,
    HandleDeleter<iree_hal_semaphore_t, iree_hal_semaphore_release>>;
using StreamPtr =
    std::unique_ptr<iree_io_stream_t,
                    HandleDeleter<iree_io_stream_t, iree_io_stream_release>>;
using JitPtr =
    std::unique_ptr<qwen_loom_jit_t,
                    HandleDeleter<qwen_loom_jit_t, qwen_loom_jit_release>>;
using ExecutablePtr = std::unique_ptr<
    qwen_loom_executable_t,
    HandleDeleter<qwen_loom_executable_t, qwen_loom_executable_release>>;

static iree_status_t ReadFile(const std::string& path,
                              std::vector<uint8_t>* out_contents) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open `%s`",
                            path.c_str());
  }
  const std::streamsize length = file.tellg();
  if (length < 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to query length of `%s`", path.c_str());
  }
  out_contents->resize(static_cast<size_t>(length));
  file.seekg(0, std::ios::beg);
  if (length != 0 &&
      !file.read(reinterpret_cast<char*>(out_contents->data()), length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "failed to read `%s`",
                            path.c_str());
  }
  return iree_ok_status();
}

static iree_status_t LoadNpy(iree_hal_device_t* device, const std::string& path,
                             iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = nullptr;
  std::vector<uint8_t> contents;
  IREE_RETURN_IF_ERROR(ReadFile(path, &contents));

  iree_io_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_memory_stream_wrap(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_SEEKABLE,
      iree_make_byte_span(contents.data(), contents.size()),
      iree_io_stream_release_callback_null(), iree_allocator_system(),
      &stream));
  StreamPtr stream_ptr(stream);

  iree_hal_buffer_params_t buffer_params = {};
  buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_numpy_npy_load_ndarray(
      stream_ptr.get(), IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params,
      device, iree_hal_device_allocator(device), out_buffer_view);
}

static iree_status_t RequireF32Shape(iree_hal_buffer_view_t* buffer_view,
                                     iree_host_size_t expected_rank,
                                     const iree_hal_dim_t* expected_shape,
                                     iree_string_view_t label) {
  if (iree_hal_buffer_view_element_type(buffer_view) !=
      IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s fixture must contain f32 values",
                            (int)label.size, label.data);
  }
  if (iree_hal_buffer_view_shape_rank(buffer_view) != expected_rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s fixture rank %" PRIhsz " does not match expected %" PRIhsz,
        (int)label.size, label.data,
        iree_hal_buffer_view_shape_rank(buffer_view), expected_rank);
  }
  for (iree_host_size_t i = 0; i < expected_rank; ++i) {
    if (iree_hal_buffer_view_shape_dim(buffer_view, i) != expected_shape[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%.*s fixture dimension %" PRIhsz " does not match expected %" PRIu64,
          (int)label.size, label.data, i, expected_shape[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t CreateLiveDevice(iree_string_view_t device_uri,
                                      ProactorPoolPtr* out_proactor_pool,
                                      FrontierTrackerPtr* out_frontier_tracker,
                                      DevicePtr* out_device,
                                      DeviceGroupPtr* out_device_group) {
  iree_allocator_t host_allocator = iree_allocator_system();
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), host_allocator,
      &proactor_pool));
  out_proactor_pool->reset(proactor_pool);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = out_proactor_pool->get();
  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_hal_create_device(iree_hal_driver_registry_default(), device_uri,
                             &create_params, host_allocator, &device));
  out_device->reset(device);

  iree_async_frontier_tracker_options_t tracker_options =
      iree_async_frontier_tracker_options_default();
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      tracker_options, host_allocator, &frontier_tracker));
  out_frontier_tracker->reset(frontier_tracker);

  iree_hal_device_group_t* device_group = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      out_device->get(), out_frontier_tracker->get(), host_allocator,
      &device_group));
  out_device_group->reset(device_group);
  return iree_ok_status();
}

static iree_status_t SubmitAndWait(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_semaphore_t* semaphore, uint64_t signal_value) {
  const iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_semaphores, command_buffer, binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE));
  return iree_hal_semaphore_wait(semaphore, signal_value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t CompareF32Buffers(iree_hal_device_t* device,
                                       iree_hal_buffer_t* actual_buffer,
                                       iree_hal_buffer_t* expected_buffer) {
  const iree_device_size_t byte_length =
      iree_hal_buffer_byte_length(expected_buffer);
  if (iree_hal_buffer_byte_length(actual_buffer) != byte_length ||
      byte_length % sizeof(float) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "actual and expected RMSNorm buffers have incompatible lengths");
  }

  std::vector<float> actual(byte_length / sizeof(float));
  std::vector<float> expected(byte_length / sizeof(float));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, actual_buffer, /*source_offset=*/0, actual.data(), byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, expected_buffer, /*source_offset=*/0, expected.data(),
      byte_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  constexpr float kAbsoluteTolerance = 0.0001f;
  constexpr float kRelativeTolerance = 0.0001f;
  for (iree_host_size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(actual[i]) != std::isnan(expected[i])) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "RMSNorm value %" PRIhsz " has mismatched NaN state", i);
    }
    if (std::isnan(actual[i])) continue;
    const float error = std::abs(actual[i] - expected[i]);
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * std::abs(expected[i]);
    if (error > tolerance) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "RMSNorm value %" PRIhsz
          " differs: actual=%g expected=%g error=%g tolerance=%g",
          i, actual[i], expected[i], error, tolerance);
    }
  }
  return iree_ok_status();
}

static iree_status_t RunRmsnormFixture(const std::string& fixture_directory) {
  const char* device_uri_environment = std::getenv("QWEN_DEVICE_URI");
  const iree_string_view_t device_uri =
      device_uri_environment ? iree_make_cstring_view(device_uri_environment)
                             : IREE_SV("amdgpu://0");

  ProactorPoolPtr proactor_pool;
  FrontierTrackerPtr frontier_tracker;
  DeviceGroupPtr device_group;
  DevicePtr device;
  IREE_RETURN_IF_ERROR(CreateLiveDevice(
      device_uri, &proactor_pool, &frontier_tracker, &device, &device_group));

  qwen_loom_jit_t* jit = nullptr;
  const qwen_loom_jit_options_t jit_options = {
      .structure_size = sizeof(jit_options),
      .next = nullptr,
      .device = device.get(),
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .entry_limit = 4,
      .sanitizer_checks = LOOMC_SANITIZER_CHECK_ACCESS,
  };
  IREE_RETURN_IF_ERROR(
      qwen_loom_jit_create(&jit_options, iree_allocator_system(), &jit));
  JitPtr jit_ptr(jit);

  qwen_loom_source_module_t source_module;
  IREE_RETURN_IF_ERROR(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED), &source_module));
  const qwen_loom_config_binding_t config_bindings[] = {
      {
          .key = IREE_SV("qwen3_moe.model.hidden_size"),
          .value = IREE_SV("2048"),
      },
      {
          .key = IREE_SV("qwen3_moe.model.rms_epsilon"),
          .value = IREE_SV("0.000001"),
      },
  };
  const int64_t workload_arguments[] = {512};
  qwen_loom_jit_prepare_options_t prepare_options = {
      .structure_size = sizeof(prepare_options),
      .next = nullptr,
      .source_module = &source_module,
      .function_name = IREE_SV("qwen3_moe_rmsnorm_f32"),
      .config_binding_count = IREE_ARRAYSIZE(config_bindings),
      .config_bindings = config_bindings,
      .workload_argument_count = IREE_ARRAYSIZE(workload_arguments),
      .workload_arguments = workload_arguments,
  };

  qwen_loom_executable_t* executable = nullptr;
  IREE_RETURN_IF_ERROR(
      qwen_loom_jit_prepare(jit_ptr.get(), &prepare_options, &executable));
  ExecutablePtr executable_ptr(executable);
  const iree_hal_dispatch_config_t dispatch_config =
      qwen_loom_executable_dispatch_config(executable_ptr.get());
  if (dispatch_config.workgroup_count[0] != 512 ||
      dispatch_config.workgroup_count[1] != 1 ||
      dispatch_config.workgroup_count[2] != 1 ||
      dispatch_config.workgroup_size[0] != 256 ||
      dispatch_config.workgroup_size[1] != 1 ||
      dispatch_config.workgroup_size[2] != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "workload 512 resolved unexpected RMSNorm launch geometry");
  }

  qwen_loom_executable_t* cache_hit = nullptr;
  IREE_RETURN_IF_ERROR(
      qwen_loom_jit_prepare(jit_ptr.get(), &prepare_options, &cache_hit));
  ExecutablePtr cache_hit_ptr(cache_hit);
  if (cache_hit != executable_ptr.get() ||
      qwen_loom_jit_entry_count(jit_ptr.get()) != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "identical Qwen Loom specialization did not reuse its cache entry");
  }

  const int64_t distinct_workload_arguments[] = {511};
  prepare_options.workload_arguments = distinct_workload_arguments;
  qwen_loom_executable_t* distinct_executable = nullptr;
  IREE_RETURN_IF_ERROR(qwen_loom_jit_prepare(jit_ptr.get(), &prepare_options,
                                             &distinct_executable));
  ExecutablePtr distinct_executable_ptr(distinct_executable);
  const iree_hal_dispatch_config_t distinct_dispatch_config =
      qwen_loom_executable_dispatch_config(distinct_executable_ptr.get());
  if (distinct_executable == executable_ptr.get() ||
      qwen_loom_jit_entry_count(jit_ptr.get()) != 2 ||
      distinct_dispatch_config.workgroup_count[0] != 511) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "workload values do not participate in exact Qwen JIT identity");
  }

  iree_hal_buffer_view_t* input_view = nullptr;
  IREE_RETURN_IF_ERROR(
      LoadNpy(device.get(), fixture_directory + "/ffn_input.npy", &input_view));
  BufferViewPtr input_view_ptr(input_view);
  iree_hal_buffer_view_t* weight_view = nullptr;
  IREE_RETURN_IF_ERROR(LoadNpy(
      device.get(), fixture_directory + "/ffn_norm_weight.npy", &weight_view));
  BufferViewPtr weight_view_ptr(weight_view);
  iree_hal_buffer_view_t* expected_view = nullptr;
  IREE_RETURN_IF_ERROR(LoadNpy(device.get(),
                               fixture_directory + "/expected_ffn_norm.npy",
                               &expected_view));
  BufferViewPtr expected_view_ptr(expected_view);

  const iree_hal_dim_t matrix_shape[] = {512, 2048};
  const iree_hal_dim_t weight_shape[] = {2048};
  IREE_RETURN_IF_ERROR(RequireF32Shape(input_view_ptr.get(),
                                       IREE_ARRAYSIZE(matrix_shape),
                                       matrix_shape, IREE_SV("input")));
  IREE_RETURN_IF_ERROR(RequireF32Shape(weight_view_ptr.get(),
                                       IREE_ARRAYSIZE(weight_shape),
                                       weight_shape, IREE_SV("weight")));
  IREE_RETURN_IF_ERROR(RequireF32Shape(expected_view_ptr.get(),
                                       IREE_ARRAYSIZE(matrix_shape),
                                       matrix_shape, IREE_SV("expected")));

  iree_hal_buffer_t* input_buffer =
      iree_hal_buffer_view_buffer(input_view_ptr.get());
  iree_hal_buffer_t* weight_buffer =
      iree_hal_buffer_view_buffer(weight_view_ptr.get());
  iree_hal_buffer_t* expected_buffer =
      iree_hal_buffer_view_buffer(expected_view_ptr.get());
  iree_hal_buffer_params_t output_params = {};
  output_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  output_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  output_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device.get()), output_params,
      iree_hal_buffer_byte_length(expected_buffer), &output_buffer));
  BufferPtr output_buffer_ptr(output_buffer);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device.get(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/3, &command_buffer));
  CommandBufferPtr command_buffer_ptr(command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer_ptr.get()));

  const iree_hal_buffer_ref_t binding_refs[] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0,
          iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, /*offset=*/0,
          iree_hal_buffer_byte_length(weight_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/2, /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer_ptr.get())),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };
  const uint32_t token_count_constant = 512;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
      command_buffer_ptr.get(),
      qwen_loom_executable_hal_executable(executable_ptr.get()),
      qwen_loom_executable_function(executable_ptr.get()), dispatch_config,
      iree_make_const_byte_span(&token_count_constant,
                                sizeof(token_count_constant)),
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer_ptr.get()));

  const iree_hal_buffer_binding_t bindings[] = {
      {
          .buffer = input_buffer,
          .offset = 0,
          .length = iree_hal_buffer_byte_length(input_buffer),
      },
      {
          .buffer = weight_buffer,
          .offset = 0,
          .length = iree_hal_buffer_byte_length(weight_buffer),
      },
      {
          .buffer = output_buffer_ptr.get(),
          .offset = 0,
          .length = iree_hal_buffer_byte_length(output_buffer_ptr.get()),
      },
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };

  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  SemaphorePtr semaphore_ptr(semaphore);
  IREE_RETURN_IF_ERROR(SubmitAndWait(device.get(), command_buffer_ptr.get(),
                                     binding_table, semaphore_ptr.get(),
                                     /*signal_value=*/1));
  IREE_RETURN_IF_ERROR(SubmitAndWait(device.get(), command_buffer_ptr.get(),
                                     binding_table, semaphore_ptr.get(),
                                     /*signal_value=*/2));
  return CompareF32Buffers(device.get(), output_buffer_ptr.get(),
                           expected_buffer);
}

TEST(QwenLoomJitRmsnormTest,
     CompilesCachesAndExecutesPrefill512FixtureWithAccessSanitizer) {
  const char* fixture_directory = std::getenv("QWEN_RMSNORM_FIXTURE_DIR");
  if (!fixture_directory) {
    GTEST_SKIP()
        << "QWEN_RMSNORM_FIXTURE_DIR must name the layer0 fixture directory";
  }
  IREE_ASSERT_OK(RunRmsnormFixture(fixture_directory));
}

}  // namespace
