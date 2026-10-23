// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_TEST_FIXTURE_H_
#define AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_TEST_FIXTURE_H_

#include <linux/kfd_ioctl.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/atomics.h"
#include "libamdf/src/gpu/umd/kfd/aql.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"
#include "libamdf/src/gpu/umd/kfd/user_queue_native.h"
#include "libamdf/src/gpu/umd/user_queue.h"

namespace {

struct FakeBuffer {
  std::vector<uint64_t> storage;
  amdf_gpu_kfd_buffer_create_info_t create_info = {};
  uint64_t device_address = 0;
  size_t byte_length = 0;
  bool live = false;
};

struct FakeNativeState {
  static constexpr size_t kBufferCount = 5;

  bool FailCreationOperation() {
    ++creation_operation_count;
    return creation_operation_count == failed_creation_operation;
  }

  static amdf_status_t BufferCreate(
      void* user_data, amdf_gpu_umd_device_t*,
      const amdf_gpu_kfd_buffer_create_info_t* create_info,
      amdf_gpu_kfd_buffer_t** out_buffer,
      amdf_gpu_kfd_buffer_result_t* out_result) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->buffer_create_count;
    FakeBuffer& buffer = self->buffers[self->buffer_create_count - 1];
    buffer.create_info = *create_info;
    if (self->FailCreationOperation()) {
      return self->creation_failure;
    }
    buffer.storage.resize((create_info->byte_length + sizeof(uint64_t) - 1) /
                          sizeof(uint64_t));
    buffer.device_address =
        UINT64_C(0x10000000) +
        (self->buffer_create_count - 1) * UINT64_C(0x01000000);
    buffer.byte_length = create_info->byte_length;
    buffer.live = true;
    *out_buffer = reinterpret_cast<amdf_gpu_kfd_buffer_t*>(&buffer);
    *out_result = {
        .device_address = buffer.device_address,
        .host_pointer =
            create_info->host_access == AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE
                ? nullptr
                : buffer.storage.data(),
    };
    return AMDF_STATUS_OK;
  }

  static amdf_status_t BufferDestroy(void* user_data,
                                     amdf_gpu_kfd_buffer_t* native_buffer) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->buffer_destroy_count;
    if (self->buffer_destroy_count == self->failed_buffer_destroy_call) {
      return self->buffer_destroy_failure;
    }
    auto* buffer = reinterpret_cast<FakeBuffer*>(native_buffer);
    self->destroyed_buffer_indices.push_back(
        static_cast<size_t>(buffer - self->buffers.data()));
    buffer->live = false;
    return AMDF_STATUS_OK;
  }

  static void BufferAbandon(void* user_data,
                            amdf_gpu_kfd_buffer_t* native_buffer) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    auto* buffer = reinterpret_cast<FakeBuffer*>(native_buffer);
    EXPECT_TRUE(buffer->live);
    self->abandoned_buffer_indices.push_back(
        static_cast<size_t>(buffer - self->buffers.data()));
    // Native backing remains live independently of abandoned host metadata.
  }

  static amdf_status_t FlushHostWrites(void* user_data,
                                       amdf_gpu_umd_device_t*) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->host_publication.call_count;
    return self->host_publication.status;
  }

  static amdf_status_t QueueCreate(
      void* user_data, amdf_gpu_umd_device_t* device,
      struct kfd_ioctl_create_queue_args* inout_arguments) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->queue_create_count;
    self->host_publication.calls_before_activation =
        self->host_publication.call_count;
    self->observed_create = *inout_arguments;
    if (inout_arguments->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
      std::memcpy(&self->observed_aql_descriptor,
                  self->buffers[1].storage.data(),
                  sizeof(self->observed_aql_descriptor));
      const auto* words =
          reinterpret_cast<const uint32_t*>(self->buffers[0].storage.data());
      for (size_t i = 0; i < inout_arguments->ring_size / sizeof(uint32_t);
           i += 16) {
        EXPECT_EQ(words[i], 1u) << "native create requires invalid AQL slots";
      }
    }
    if (inout_arguments->ctx_save_restore_size != 0) {
      const size_t context_index = inout_arguments->eop_buffer_size ? 3 : 2;
      const auto* context = reinterpret_cast<const uint8_t*>(
          self->buffers[context_index].storage.data());
      for (uint32_t i = 0; i < device->topology.properties.topology.xcc_count;
           ++i) {
        self->observed_context_headers.push_back(
            *reinterpret_cast<const kfd_context_save_area_header*>(
                context + i * inout_arguments->ctx_save_restore_size));
      }
    }
    if (self->FailCreationOperation()) {
      return self->creation_failure;
    }
    inout_arguments->queue_id = self->created_queue_identifier;
    inout_arguments->doorbell_offset = self->doorbell_offset;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t QueueDestroy(void* user_data, amdf_gpu_umd_device_t*,
                                    uint32_t queue_identifier) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->queue_destroy_count;
    self->destroyed_queue_identifiers.push_back(queue_identifier);
    return self->queue_destroy_status;
  }

  static amdf_status_t QueueUpdate(
      void* user_data, amdf_gpu_umd_device_t*,
      const struct kfd_ioctl_update_queue_args* arguments) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    self->queue_update.arguments.push_back(*arguments);
    self->queue_update.descriptors.push_back(
        *reinterpret_cast<const amdf_gpu_kfd_aql_descriptor_t*>(
            self->buffers[1].storage.data()));
    return self->queue_update.arguments.size() == self->queue_update.fail_call
               ? amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO)
               : AMDF_STATUS_OK;
  }

  static amdf_status_t DoorbellMap(void* user_data, amdf_gpu_umd_device_t*,
                                   uint64_t native_byte_offset,
                                   size_t byte_length, void** out_mapping) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->doorbell_map_count;
    self->observed_doorbell_mapping_offset = native_byte_offset;
    self->observed_doorbell_mapping_length = byte_length;
    if (self->FailCreationOperation()) {
      return self->creation_failure;
    }
    *out_mapping = self->doorbell_mapping.data();
    return AMDF_STATUS_OK;
  }

  static amdf_status_t DoorbellUnmap(void* user_data, void* mapping,
                                     size_t byte_length) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->doorbell_unmap_count;
    self->observed_doorbell_unmapping = mapping;
    self->observed_doorbell_unmapping_length = byte_length;
    return self->doorbell_unmap_status;
  }

  static amdf_status_t VmFaultQuery(
      void* user_data, amdf_gpu_umd_device_t*,
      struct drm_amdgpu_info_gpuvm_fault* out_fault) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->vm_fault.query_count;
    if (!amdf_status_is_ok(self->vm_fault.query_status)) {
      return self->vm_fault.query_status;
    }
    *out_fault = self->vm_fault.info;
    return AMDF_STATUS_OK;
  }

  static amdf_status_t DeviceDoorbellCreate(
      void* user_data, amdf_gpu_umd_device_t*, size_t byte_length,
      amdf_gpu_kfd_doorbell_t** out_doorbell, uint64_t* out_device_address) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->device_doorbell.create_count;
    EXPECT_EQ(byte_length, 8192u);
    if (self->FailCreationOperation()) {
      return self->creation_failure;
    }
    self->device_doorbell.live = true;
    *out_doorbell =
        reinterpret_cast<amdf_gpu_kfd_doorbell_t*>(&self->device_doorbell);
    *out_device_address = UINT64_C(0x50000000);
    return AMDF_STATUS_OK;
  }

  static amdf_status_t DeviceDoorbellDestroy(
      void* user_data, amdf_gpu_kfd_doorbell_t* doorbell) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    EXPECT_EQ(doorbell, reinterpret_cast<amdf_gpu_kfd_doorbell_t*>(
                            &self->device_doorbell));
    EXPECT_EQ(self->queue_destroy_count, 1);
    // The retirement-flush buffer is created last and released before the
    // device doorbell; ring and control storage remain live until afterward.
    EXPECT_EQ(self->destroyed_buffer_indices,
              (std::vector<size_t>{
                  static_cast<size_t>(self->buffer_create_count - 1)}));
    ++self->device_doorbell.destroy_count;
    if (amdf_status_is_ok(self->device_doorbell.destroy_status)) {
      self->device_doorbell.live = false;
    }
    return self->device_doorbell.destroy_status;
  }

  static void DeviceDoorbellAbandon(void* user_data, amdf_gpu_kfd_doorbell_t*) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    EXPECT_TRUE(self->device_doorbell.live);
    ++self->device_doorbell.abandon_count;
  }

  void Reset() {
    for (FakeBuffer& buffer : buffers) {
      buffer = {};
    }
    std::memset(doorbell_mapping.data(), 0, doorbell_mapping.size());
    creation_operation_count = 0;
    failed_creation_operation = 0;
    creation_failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM);
    buffer_create_count = 0;
    buffer_destroy_count = 0;
    failed_buffer_destroy_call = 0;
    buffer_destroy_failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
    destroyed_buffer_indices.clear();
    abandoned_buffer_indices.clear();
    queue_create_count = 0;
    queue_destroy_count = 0;
    queue_destroy_status = AMDF_STATUS_OK;
    queue_update = {};
    host_publication = {};
    destroyed_queue_identifiers.clear();
    doorbell_map_count = 0;
    doorbell_unmap_count = 0;
    doorbell_unmap_status = AMDF_STATUS_OK;
    device_doorbell = {};
    vm_fault = {};
    observed_create = {};
    observed_aql_descriptor = {};
    observed_context_headers.clear();
    observed_doorbell_mapping_offset = 0;
    observed_doorbell_mapping_length = 0;
    observed_doorbell_unmapping = nullptr;
    observed_doorbell_unmapping_length = 0;
  }

  size_t LiveBufferCount() const {
    size_t count = 0;
    for (const FakeBuffer& buffer : buffers) {
      count += buffer.live ? 1 : 0;
    }
    return count;
  }

  std::array<FakeBuffer, kBufferCount> buffers;
  alignas(uint64_t) std::array<uint8_t, 8192> doorbell_mapping = {};
  int creation_operation_count = 0;
  int failed_creation_operation = 0;
  amdf_status_t creation_failure = AMDF_STATUS_OK;
  int buffer_create_count = 0;
  int buffer_destroy_count = 0;
  int failed_buffer_destroy_call = 0;
  amdf_status_t buffer_destroy_failure = AMDF_STATUS_OK;
  std::vector<size_t> destroyed_buffer_indices;
  // Buffers whose metadata was abandoned without releasing reachable backing.
  std::vector<size_t> abandoned_buffer_indices;
  int queue_create_count = 0;
  int queue_destroy_count = 0;
  amdf_status_t queue_destroy_status = AMDF_STATUS_OK;
  // Native host publication below the queue ownership boundary.
  struct {
    // Completed native release attempts.
    uint32_t call_count = 0;
    // Release attempts observed before firmware could consume the ring.
    uint32_t calls_before_activation = 0;
    // Result supplied by the native release dependency.
    amdf_status_t status = AMDF_STATUS_OK;
  } host_publication;
  // Native detach/reconnect observations and dependency failure selection.
  struct {
    // Exact ring identity and activation percentage passed to each call.
    std::vector<kfd_ioctl_update_queue_args> arguments;
    // Firmware descriptor as each native operation observes it.
    std::vector<amdf_gpu_kfd_aql_descriptor_t> descriptors;
    // One-based failed native operation, or zero for success.
    size_t fail_call = 0;
  } queue_update;
  std::vector<uint32_t> destroyed_queue_identifiers;
  int doorbell_map_count = 0;
  int doorbell_unmap_count = 0;
  amdf_status_t doorbell_unmap_status = AMDF_STATUS_OK;
  // Queue-owned GPU view, independent of the host notification view.
  struct {
    // Optional native construction attempts.
    uint32_t create_count = 0;
    // Consuming native release attempts.
    uint32_t destroy_count = 0;
    // Metadata-only abandonment after failed queue retirement.
    uint32_t abandon_count = 0;
    // Terminal native cleanup result.
    amdf_status_t destroy_status = AMDF_STATUS_OK;
    // Whether the modeled native allocation remains mapped.
    bool live = false;
  } device_doorbell;
  // Per-render-VM observation supplied by the native dependency.
  struct {
    // Number of native fault queries performed.
    uint32_t query_count = 0;
    // Native ioctl result, independent of whether the VM faulted.
    amdf_status_t query_status = AMDF_STATUS_OK;
    // Cached fault record returned on successful native observation.
    drm_amdgpu_info_gpuvm_fault info = {};
  } vm_fault;
  uint32_t created_queue_identifier = 47;
  uint64_t doorbell_offset = UINT64_C(0x20000080);
  struct kfd_ioctl_create_queue_args observed_create = {};
  // Firmware-visible AQL descriptor captured at the activation boundary.
  amdf_gpu_kfd_aql_descriptor_t observed_aql_descriptor = {};
  // Per-XCC save headers captured before native creation can consume them.
  std::vector<kfd_context_save_area_header> observed_context_headers;
  uint64_t observed_doorbell_mapping_offset = 0;
  size_t observed_doorbell_mapping_length = 0;
  void* observed_doorbell_unmapping = nullptr;
  size_t observed_doorbell_unmapping_length = 0;
};

class KfdUserQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state_.Reset();
    device_.host_allocator = amdf_allocator_system();
    device_.page_size = 4096;
    device_.cache_line_size = 64;
    device_.topology.properties.gfx_ip = {11, 5, 1};
    device_.topology.gc_ip = {11, 5, 1, true};
    device_.topology.properties.compute.wavefront_size = 32;
    device_.topology.properties.compute.compute_unit_count = 2;
    device_.topology.properties.compute.maximum_wave_count_per_compute_unit =
        32;
    device_.topology.properties.compute
        .maximum_scratch_wave_count_per_compute_unit = 32;
    device_.topology.properties.compute.local_data_share_byte_length = 65536;
    device_.topology.properties.topology.xcc_count = 1;
    device_.topology.properties.topology.shader_engine_count_per_xcc = 1;
    device_.topology.gpu_id = 73;
    device_.topology.compute_queue_count = 8;
    device_.topology.sdma.engine_count = 1;
    device_.topology.sdma.queue_count_per_engine = 6;
    device_.topology.sdma.ip = {6, 1, 1, true};
    device_.topology.context_save_restore_byte_length = 4096;
    device_.topology.control_stack_byte_length = 4096;
    device_.topology.virtual_address.alignment = 4096;
    device_.user_queue_native_api = &native_api_;
  }

  void TearDown() override {
    if (mapping_ != nullptr) {
      EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_),
                AMDF_STATUS_OK);
      mapping_ = nullptr;
    }
    if (queue_ != nullptr) {
      if (native_state_.buffers[1].live) {
        const auto& control = native_state_.buffers[1];
        auto* base =
            reinterpret_cast<uint8_t*>(native_state_.buffers[1].storage.data());
        auto* read_index = reinterpret_cast<amdf_atomic_uint64_t*>(
            base + (native_state_.observed_create.read_pointer_address -
                    control.device_address));
        auto* write_index = reinterpret_cast<amdf_atomic_uint64_t*>(
            base + (native_state_.observed_create.write_pointer_address -
                    control.device_address));
        const uint64_t published_index =
            amdf_atomic_uint64_load_acquire(write_index);
        const uint64_t read_index_mask =
            native_state_.observed_create.queue_type ==
                    KFD_IOC_QUEUE_TYPE_COMPUTE
                ? native_state_.observed_create.ring_size / sizeof(uint32_t) - 1
                : UINT64_MAX;
        amdf_atomic_uint64_store_release(read_index,
                                         published_index & read_index_mask);
      }
      native_state_.queue_destroy_status = AMDF_STATUS_OK;
      native_state_.doorbell_unmap_status = AMDF_STATUS_OK;
      native_state_.failed_buffer_destroy_call = 0;
      EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
      queue_ = nullptr;
    }
  }

  amdf_gpu_umd_user_queue_create_info_t MakeCreateInfo(
      amdf_queue_command_type_t command_type =
          AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) const {
    amdf_gpu_kfd_user_queue_plans_t plans;
    amdf_gpu_kfd_target_user_queue_plans_initialize(
        &device_.topology, device_.page_size, device_.cache_line_size, &plans);
    for (uint32_t i = 0; i < plans.count; ++i) {
      const amdf_gpu_queue_family_properties_t& family = plans.values[i].family;
      if (family.command_type != command_type) {
        continue;
      }
      return {
          .command_type = family.command_type,
          .format_version = family.format_version,
          .priority = AMDF_QUEUE_PRIORITY_NORMAL,
          .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
          .roles = family.roles,
      };
    }
    ADD_FAILURE() << "no target queue plan for command type " << command_type;
    return {};
  }

  void CreateQueue(amdf_queue_command_type_t command_type =
                       AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
    const amdf_gpu_umd_user_queue_create_info_t create_info =
        MakeCreateInfo(command_type);
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    ASSERT_NE(queue_, nullptr);
  }

  void MapQueue() {
    ASSERT_EQ(amdf_gpu_umd_user_queue_map(queue_, nullptr, &mapping_,
                                          &mapping_result_),
              AMDF_STATUS_OK);
    ASSERT_NE(mapping_, nullptr);
  }

  void SelectAqlTopology() {
    device_.topology.properties.gfx_ip = {9, 4, 2};
    device_.topology.properties.compute.wavefront_size = 64;
    device_.topology.properties.compute.compute_unit_count = 16;
    device_.topology.properties.compute.maximum_wave_count_per_compute_unit =
        32;
    device_.topology.properties.compute
        .maximum_scratch_wave_count_per_compute_unit = 32;
    device_.topology.properties.compute.local_data_share_byte_length = 65536;
    device_.topology.properties.topology.xcc_count = 8;
    device_.topology.properties.topology.shader_engine_count_per_xcc = 2;
  }

  amdf_atomic_uint64_t* ReadIndex() {
    return reinterpret_cast<amdf_atomic_uint64_t*>(
        mapping_result_.read_index_address);
  }

  amdf_atomic_uint64_t* WriteIndex() {
    return reinterpret_cast<amdf_atomic_uint64_t*>(
        mapping_result_.write_index_address);
  }

  amdf_atomic_uint64_t* ErrorPayload() {
    return reinterpret_cast<amdf_atomic_uint64_t*>(
        reinterpret_cast<uint8_t*>(native_state_.buffers[1].storage.data()) +
        128);
  }

  FakeNativeState native_state_;
  amdf_gpu_kfd_user_queue_native_api_t native_api_ = {
      .user_data = &native_state_,
      .buffer_create = FakeNativeState::BufferCreate,
      .buffer_destroy = FakeNativeState::BufferDestroy,
      .buffer_abandon = FakeNativeState::BufferAbandon,
      .queue_create = FakeNativeState::QueueCreate,
      .queue_update = FakeNativeState::QueueUpdate,
      .queue_destroy = FakeNativeState::QueueDestroy,
      .doorbell_map = FakeNativeState::DoorbellMap,
      .doorbell_unmap = FakeNativeState::DoorbellUnmap,
      .device_doorbell_create = FakeNativeState::DeviceDoorbellCreate,
      .device_doorbell_destroy = FakeNativeState::DeviceDoorbellDestroy,
      .device_doorbell_abandon = FakeNativeState::DeviceDoorbellAbandon,
      .vm_fault_query = FakeNativeState::VmFaultQuery,
      .flush_host_writes = FakeNativeState::FlushHostWrites,
  };
  amdf_gpu_umd_device_t device_ = {};
  amdf_gpu_umd_user_queue_t* queue_ = nullptr;
  amdf_gpu_umd_user_queue_mapping_t* mapping_ = nullptr;
  amdf_gpu_umd_user_queue_result_t queue_result_ = {};
  amdf_gpu_umd_user_queue_mapping_result_t mapping_result_ = {};
};

}  // namespace

#endif  // AMDF_SRC_GPU_UMD_KFD_USER_QUEUE_TEST_FIXTURE_H_
