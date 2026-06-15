// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_TEST_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu::test {

using iree::hal::cts::Ref;

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(base_device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(
      const iree_hal_amdgpu_logical_device_options_t* options,
      const iree_hal_amdgpu_libhsa_t* libhsa,
      const iree_hal_amdgpu_topology_t* topology,
      iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), options, libhsa, topology, create_context_.params(),
        host_allocator, &base_device_));
    return iree_hal_device_group_create_from_device(
        base_device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* base_device() const { return base_device_; }

  iree_hal_allocator_t* allocator() const {
    return iree_hal_device_allocator(base_device_);
  }

  iree_hal_amdgpu_logical_device_t* logical_device() const {
    return (iree_hal_amdgpu_logical_device_t*)base_device_;
  }

  iree_hal_amdgpu_host_queue_t* first_host_queue() const {
    iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
    if (logical_device->physical_device_count == 0) return NULL;
    iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[0];
    if (physical_device->host_queue_count == 0) return NULL;
    return &physical_device->host_queues[0];
  }

 private:
  // Creation context supplying the proactor pool and frontier tracker.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the topology-owning group.
  iree_hal_device_t* base_device_ = NULL;

  // Device group that owns the topology assigned to |base_device_|.
  iree_hal_device_group_t* device_group_ = NULL;
};

static iree_status_t QueueAffinityForPhysicalDevice(
    const TestLogicalDevice& test_device,
    iree_host_size_t physical_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      /*.supported_affinity=*/logical_device->queue_affinity_mask,
      /*.physical_device_count=*/logical_device->physical_device_count,
      /*.queue_count_per_physical_device=*/
      logical_device->system->topology.gpu_agent_queue_count,
  };
  return iree_hal_amdgpu_queue_affinity_for_physical_device(
      domain, physical_device_ordinal, out_queue_affinity);
}

static iree_status_t CreateHostVisibleTransferBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL |
                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static iree_status_t CreateHostVisibleDispatchBuffer(
    iree_hal_allocator_t* allocator, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL |
                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  return iree_hal_allocator_allocate_buffer(allocator, params, buffer_size,
                                            out_buffer);
}

static bool IsAmdgpuCtsExecutableFormat(
    const iree::hal::cts::ExecutableFormat& format) {
  if (format.format == nullptr || format.data_fn == nullptr) return false;
  return iree_string_view_starts_with(
      iree_make_string_view(format.name.data(), format.name.size()),
      IREE_SV("amdgpu_"));
}

static iree_status_t LoadCtsExecutable(
    iree_hal_device_t* device, iree_string_view_t file_name,
    iree_hal_executable_cache_t** out_executable_cache,
    iree_hal_executable_t** out_executable) {
  *out_executable_cache = NULL;
  *out_executable = NULL;

  const auto formats =
      iree::hal::cts::CtsRegistry::ListExecutableFormats("amdgpu");
  iree_status_t candidate_status = iree_ok_status();
  bool found_format = false;
  bool found_executable_data = false;
  for (const auto& format : formats) {
    if (!IsAmdgpuCtsExecutableFormat(format)) continue;
    found_format = true;
    iree_const_byte_span_t executable_data = format.data_fn(file_name);
    if (executable_data.data_length == 0) continue;
    found_executable_data = true;

    iree_hal_executable_cache_t* executable_cache = NULL;
    iree_hal_executable_t* executable = NULL;
    iree_status_t status = iree_hal_executable_cache_create(
        device, iree_make_cstring_view("default"), &executable_cache);
    if (iree_status_is_ok(status)) {
      iree_hal_executable_params_t executable_params;
      iree_hal_executable_params_initialize(&executable_params);
      executable_params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      executable_params.executable_format =
          iree_make_cstring_view(format.format);
      executable_params.executable_data = executable_data;
      status = iree_hal_executable_cache_prepare_executable(
          executable_cache, &executable_params, &executable);
    }

    if (iree_status_is_ok(status)) {
      *out_executable_cache = executable_cache;
      *out_executable = executable;
      return iree_ok_status();
    }
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
    candidate_status = iree_status_join(candidate_status, status);
  }

  if (!iree_status_is_ok(candidate_status)) {
    return candidate_status;
  }
  if (!found_format) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no registered AMDGPU CTS executable formats");
  }
  if (!found_executable_data) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU CTS executable not found");
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "registered AMDGPU CTS executable data was not accepted by the device");
}

static iree_status_t QueueTransientTransferBuffer(
    iree_hal_device_t* device, const iree_hal_semaphore_list_t signal_list,
    iree_device_size_t buffer_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_device_queue_alloca(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      iree_hal_semaphore_list_empty(),
                                      signal_list,
                                      /*pool=*/NULL, params, buffer_size,
                                      IREE_HAL_ALLOCA_FLAG_NONE, out_buffer);
}

static iree_status_t EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
  return iree_ok_status();
}

static bool HostQueueHasPostDrainAction(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.post_drain_mutex);
  const bool has_action = queue->post_drain.head != NULL;
  iree_slim_mutex_unlock(&queue->locks.post_drain_mutex);
  return has_action;
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_status_t AppendConstantsBindingsDispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, iree_hal_buffer_ref_list_t bindings) {
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));
  return iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE);
}

struct TwoDispatchCommandBuffer {
  ~TwoDispatchCommandBuffer() {
    iree_hal_executable_release(executable);
    iree_hal_executable_cache_release(executable_cache);
  }

  // Executable cache owning |executable|.
  iree_hal_executable_cache_t* executable_cache = NULL;

  // CTS executable containing the constants+bindings dispatch entry point.
  iree_hal_executable_t* executable = NULL;

  // Host-visible input buffer shared by both dispatches.
  Ref<iree_hal_buffer_t> input_buffer;

  // Host-visible output buffer written by command index 0.
  Ref<iree_hal_buffer_t> output_buffer0;

  // Host-visible output buffer written by command index 1.
  Ref<iree_hal_buffer_t> output_buffer1;

  // Command buffer containing two equivalent dispatch operations.
  Ref<iree_hal_command_buffer_t> command_buffer;
};

static iree_status_t CreateTwoDispatchCommandBuffer(
    TestLogicalDevice* test_device, TwoDispatchCommandBuffer* out_fixture,
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA) {
  IREE_RETURN_IF_ERROR(LoadCtsExecutable(
      test_device->base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &out_fixture->executable_cache, &out_fixture->executable));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_write(out_fixture->input_buffer, /*target_offset=*/0,
                                input_values, sizeof(input_values)));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->output_buffer0.out()));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_zero(
      out_fixture->output_buffer0, /*offset=*/0, IREE_HAL_WHOLE_BUFFER));

  IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
      test_device->allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      out_fixture->output_buffer1.out()));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_zero(
      out_fixture->output_buffer1, /*offset=*/0, IREE_HAL_WHOLE_BUFFER));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      test_device->base_device(), mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0,
      out_fixture->command_buffer.out()));
  IREE_RETURN_IF_ERROR(
      iree_hal_command_buffer_begin(out_fixture->command_buffer));
  iree_hal_buffer_ref_t binding_refs0[2] = {
      iree_hal_make_buffer_ref(
          out_fixture->input_buffer, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->input_buffer)),
      iree_hal_make_buffer_ref(
          out_fixture->output_buffer0, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->output_buffer0)),
  };
  const iree_hal_buffer_ref_list_t bindings0 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs0),
      /*values=*/binding_refs0,
  };
  IREE_RETURN_IF_ERROR(AppendConstantsBindingsDispatch(
      out_fixture->command_buffer, out_fixture->executable, bindings0));
  iree_hal_buffer_ref_t binding_refs1[2] = {
      iree_hal_make_buffer_ref(
          out_fixture->input_buffer, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->input_buffer)),
      iree_hal_make_buffer_ref(
          out_fixture->output_buffer1, /*offset=*/0,
          iree_hal_buffer_byte_length(out_fixture->output_buffer1)),
  };
  const iree_hal_buffer_ref_list_t bindings1 = {
      /*count=*/IREE_ARRAYSIZE(binding_refs1),
      /*values=*/binding_refs1,
  };
  IREE_RETURN_IF_ERROR(AppendConstantsBindingsDispatch(
      out_fixture->command_buffer, out_fixture->executable, bindings1));
  return iree_hal_command_buffer_end(out_fixture->command_buffer);
}

static void ExpectTwoDispatchOutputs(const TwoDispatchCommandBuffer& fixture) {
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t output_values0[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(fixture.output_buffer0, /*offset=*/0,
                                          output_values0,
                                          sizeof(output_values0)));
  EXPECT_EQ(0,
            memcmp(output_values0, expected_values, sizeof(expected_values)));
  uint32_t output_values1[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(fixture.output_buffer1, /*offset=*/0,
                                          output_values1,
                                          sizeof(output_values1)));
  EXPECT_EQ(0,
            memcmp(output_values1, expected_values, sizeof(expected_values)));
}

}  // namespace iree::hal::amdgpu::test

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_TEST_UTIL_H_
