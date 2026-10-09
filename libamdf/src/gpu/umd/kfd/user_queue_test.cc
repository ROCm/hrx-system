// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/user_queue.h"

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

  static amdf_status_t QueueCreate(
      void* user_data, amdf_gpu_umd_device_t* device,
      struct kfd_ioctl_create_queue_args* inout_arguments) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->queue_create_count;
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
    EXPECT_EQ(self->destroyed_buffer_indices, (std::vector<size_t>{2}));
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
      .queue_destroy = FakeNativeState::QueueDestroy,
      .doorbell_map = FakeNativeState::DoorbellMap,
      .doorbell_unmap = FakeNativeState::DoorbellUnmap,
      .device_doorbell_create = FakeNativeState::DeviceDoorbellCreate,
      .device_doorbell_destroy = FakeNativeState::DeviceDoorbellDestroy,
      .device_doorbell_abandon = FakeNativeState::DeviceDoorbellAbandon,
      .vm_fault_query = FakeNativeState::VmFaultQuery,
  };
  amdf_gpu_umd_device_t device_ = {};
  amdf_gpu_umd_user_queue_t* queue_ = nullptr;
  amdf_gpu_umd_user_queue_mapping_t* mapping_ = nullptr;
  amdf_gpu_umd_user_queue_result_t queue_result_ = {};
  amdf_gpu_umd_user_queue_mapping_result_t mapping_result_ = {};
};

TEST_F(KfdUserQueueTest, ConstructionDependenciesPublishOnlyOnSuccess) {
  const amdf_gpu_umd_user_queue_create_info_t create_info = MakeCreateInfo();
  for (int failed_operation = 1; failed_operation <= 7; ++failed_operation) {
    native_state_.Reset();
    native_state_.failed_creation_operation = failed_operation;
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    amdf_gpu_umd_user_queue_t* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_gpu_umd_user_queue_result_t original_result = result;

    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result),
        native_state_.creation_failure);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    EXPECT_EQ(native_state_.queue_destroy_count, failed_operation == 7 ? 1 : 0);
  }
}

TEST_F(KfdUserQueueTest, PublishesExactNativeQueueAndHostMapping) {
  CreateQueue();
  EXPECT_TRUE(amdf_queue_id_is_valid(&queue_result_.queue_id));
  EXPECT_EQ(queue_result_.capabilities,
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER);
  EXPECT_EQ(queue_result_.ring_byte_length, 4096u);
  EXPECT_EQ(queue_result_.metadata_ring_byte_length, 0u);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);

  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  EXPECT_EQ(native_state_.buffers[0].create_info.native_flags,
            host_storage_flags);
  EXPECT_EQ(native_state_.buffers[0].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[0].create_info.alignment, 4096u);
  EXPECT_EQ(native_state_.buffers[0].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED);
  EXPECT_EQ(native_state_.buffers[1].create_info.native_flags,
            host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED);
  EXPECT_EQ(native_state_.buffers[1].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[1].create_info.alignment, 4096u);
  EXPECT_EQ(native_state_.buffers[1].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED);
  EXPECT_EQ(native_state_.buffers[2].create_info.native_flags,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE);
  EXPECT_EQ(native_state_.buffers[2].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[2].create_info.alignment, 4096u);
  EXPECT_EQ(native_state_.buffers[2].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE);
  EXPECT_EQ(native_state_.buffers[3].create_info.native_flags,
            host_storage_flags);
  EXPECT_EQ(native_state_.buffers[3].create_info.byte_length, 8192u);
  EXPECT_EQ(native_state_.buffers[3].create_info.alignment, 4096u);
  EXPECT_EQ(native_state_.buffers[3].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED);
  EXPECT_EQ(native_state_.buffers[4].create_info.native_flags,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);
  EXPECT_EQ(native_state_.buffers[4].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[4].create_info.alignment, 4096u);
  EXPECT_EQ(native_state_.buffers[4].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE);

  const auto& create = native_state_.observed_create;
  EXPECT_EQ(create.ring_base_address, native_state_.buffers[0].device_address);
  EXPECT_EQ(create.write_pointer_address,
            native_state_.buffers[1].device_address + 64);
  EXPECT_EQ(create.read_pointer_address,
            native_state_.buffers[1].device_address);
  EXPECT_EQ(create.ring_size, 4096u);
  EXPECT_EQ(create.gpu_id, device_.topology.gpu_id);
  EXPECT_EQ(create.queue_type, KFD_IOC_QUEUE_TYPE_COMPUTE);
  EXPECT_EQ(create.queue_percentage, KFD_MAX_QUEUE_PERCENTAGE);
  EXPECT_EQ(create.queue_priority, 7u);
  EXPECT_EQ(create.eop_buffer_address, native_state_.buffers[2].device_address);
  EXPECT_EQ(create.eop_buffer_size, 4096u);
  EXPECT_EQ(create.ctx_save_restore_address,
            native_state_.buffers[3].device_address);
  EXPECT_EQ(create.ctx_save_restore_size, 4096u);
  EXPECT_EQ(create.ctl_stack_size, 4096u);
  EXPECT_EQ(native_state_.observed_doorbell_mapping_offset,
            UINT64_C(0x20000000));
  EXPECT_EQ(native_state_.observed_doorbell_mapping_length, 8192u);

  const auto* context_header =
      reinterpret_cast<const struct kfd_context_save_area_header*>(
          native_state_.buffers[3].storage.data());
  EXPECT_EQ(context_header->debug_offset, 4096u);
  EXPECT_EQ(context_header->debug_size, 2048u);
  EXPECT_EQ(context_header->err_payload_addr,
            native_state_.buffers[1].device_address + 128);

  MapQueue();
  EXPECT_EQ(
      mapping_result_.ring_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[0].storage.data()));
  EXPECT_EQ(
      mapping_result_.read_index_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[1].storage.data()));
  EXPECT_EQ(mapping_result_.write_index_address,
            mapping_result_.read_index_address + 64);
  EXPECT_EQ(
      mapping_result_.doorbell_address,
      reinterpret_cast<uintptr_t>(native_state_.doorbell_mapping.data()) + 128);
  EXPECT_EQ(mapping_result_.index_bits, 64u);
  EXPECT_EQ(mapping_result_.doorbell_bits, 64u);
  EXPECT_EQ(mapping_result_.metadata_ring_address, 0u);

  auto* const mapping_sentinel =
      reinterpret_cast<amdf_gpu_umd_user_queue_mapping_t*>(uintptr_t{1});
  amdf_gpu_umd_user_queue_mapping_t* rejected_mapping = mapping_sentinel;
  amdf_gpu_umd_user_queue_mapping_result_t rejected_result;
  std::memset(&rejected_result, 0xA5, sizeof(rejected_result));
  const amdf_gpu_umd_user_queue_mapping_result_t original_result =
      rejected_result;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_map(
                queue_, &device_, &rejected_mapping, &rejected_result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(rejected_mapping, mapping_sentinel);
  EXPECT_EQ(
      std::memcmp(&rejected_result, &original_result, sizeof(rejected_result)),
      0);
}

TEST_F(KfdUserQueueTest, BorrowsExistingHostMappingsWithoutAllocating) {
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  device_.host_allocator.allocate = [](void*, uint64_t, uint64_t) -> void* {
    ADD_FAILURE() << "queue-owned host mappings require no new allocation";
    return nullptr;
  };
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  amdf_gpu_umd_user_queue_mapping_t* second_mapping = nullptr;
  amdf_gpu_umd_user_queue_mapping_result_t second_result = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_map(queue_, nullptr, &second_mapping,
                                        &second_result),
            AMDF_STATUS_OK);
  EXPECT_EQ(second_result.ring_address, mapping_result_.ring_address);
  EXPECT_EQ(second_result.read_index_address,
            mapping_result_.read_index_address);
  EXPECT_EQ(second_result.write_index_address,
            mapping_result_.write_index_address);
  EXPECT_EQ(second_result.doorbell_address, mapping_result_.doorbell_address);
  EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(second_mapping),
            AMDF_STATUS_OK);
  // Releasing either borrow must leave the queue's control page and doorbell
  // live for the other. The fake native dependency consumes no commands.
  amdf_atomic_uint64_store_release(WriteIndex(), 7);
  amdf_atomic_uint64_store_release(ReadIndex(), 3);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.producer_index, 7u);
  EXPECT_EQ(status.consumed_index, 3u);
  EXPECT_EQ(native_state_.buffer_destroy_count, 0);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 0);
}

TEST_F(KfdUserQueueTest, ComputeStorageUsesReportedTopologyAcrossGfx11) {
  device_.topology.properties.gfx_ip = {11, 0, 0};
  device_.topology.properties.compute.compute_unit_count = 3;
  device_.topology.properties.compute.maximum_wave_count_per_compute_unit = 17;
  device_.topology.context_save_restore_byte_length = 16384;
  device_.topology.control_stack_byte_length = 8192;
  CreateQueue();
  ASSERT_NE(queue_, nullptr);
  EXPECT_EQ(native_state_.observed_create.ctx_save_restore_size, 16384u);
  EXPECT_EQ(native_state_.observed_create.ctl_stack_size, 8192u);
  const auto* header =
      reinterpret_cast<const struct kfd_context_save_area_header*>(
          native_state_.buffers[3].storage.data());
  EXPECT_EQ(header->debug_offset, 16384u);
  // The GFX11 save protocol reserves 32 waves per CU independently of the
  // reported resident limit. The whole allocation remains page aligned.
  EXPECT_EQ(header->debug_size, 3072u);
  EXPECT_EQ(native_state_.buffers[3].byte_length, 20480u);
}

TEST_F(KfdUserQueueTest, SdmaConstructionPublishesOnlyAfterSuccess) {
  const amdf_gpu_umd_user_queue_create_info_t create_info =
      MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  for (int failed_operation = 1; failed_operation <= 5; ++failed_operation) {
    native_state_.Reset();
    native_state_.failed_creation_operation = failed_operation;
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    amdf_gpu_umd_user_queue_t* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_gpu_umd_user_queue_result_t original_result = result;

    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result),
        native_state_.creation_failure);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    EXPECT_EQ(native_state_.queue_destroy_count, failed_operation == 5 ? 1 : 0);
  }
}

TEST_F(KfdUserQueueTest, PublishesExactSdmaQueueAndHostMapping) {
  CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  EXPECT_TRUE(amdf_queue_id_is_valid(&queue_result_.queue_id));
  EXPECT_EQ(queue_result_.capabilities,
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER);
  EXPECT_EQ(native_state_.device_doorbell.create_count, 0u);
  EXPECT_EQ(queue_result_.ring_byte_length, 4096u);
  EXPECT_EQ(queue_result_.metadata_ring_byte_length, 0u);
  ASSERT_EQ(native_state_.LiveBufferCount(), 3u);
  ASSERT_EQ(native_state_.buffer_create_count, 3);

  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  EXPECT_EQ(native_state_.buffers[0].create_info.native_flags,
            host_storage_flags);
  EXPECT_EQ(native_state_.buffers[0].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[1].create_info.native_flags,
            host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED);
  EXPECT_EQ(native_state_.buffers[1].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[2].create_info.native_flags,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);
  EXPECT_EQ(native_state_.buffers[2].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[2].create_info.host_access,
            AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE);

  const auto& create = native_state_.observed_create;
  EXPECT_EQ(create.ring_base_address, native_state_.buffers[0].device_address);
  EXPECT_EQ(create.write_pointer_address,
            native_state_.buffers[1].device_address + 64);
  EXPECT_EQ(create.read_pointer_address,
            native_state_.buffers[1].device_address);
  EXPECT_EQ(create.ring_size, 4096u);
  EXPECT_EQ(create.queue_type, KFD_IOC_QUEUE_TYPE_SDMA);
  EXPECT_EQ(create.eop_buffer_address, 0u);
  EXPECT_EQ(create.eop_buffer_size, 0u);
  EXPECT_EQ(create.ctx_save_restore_address, 0u);
  EXPECT_EQ(create.ctx_save_restore_size, 0u);
  EXPECT_EQ(create.ctl_stack_size, 0u);

  MapQueue();
  EXPECT_EQ(
      mapping_result_.ring_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[0].storage.data()));
  EXPECT_EQ(
      mapping_result_.read_index_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[1].storage.data()));
  EXPECT_EQ(mapping_result_.write_index_address,
            mapping_result_.read_index_address + 64);
  EXPECT_EQ(mapping_result_.index_bits, 64u);
  EXPECT_EQ(mapping_result_.doorbell_bits, 64u);
  EXPECT_EQ(mapping_result_.metadata_ring_address, 0u);

  auto* const absent_error_payload = reinterpret_cast<amdf_atomic_uint64_t*>(
      reinterpret_cast<uint8_t*>(native_state_.buffers[1].storage.data()) +
      128);
  amdf_atomic_uint64_store_release(absent_error_payload, UINT64_MAX);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);

  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.destroyed_buffer_indices,
            (std::vector<size_t>{2, 1, 0}));
}

TEST_F(KfdUserQueueTest, DevicePublicationRequiresCreationRequest) {
  CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  auto* const sentinel =
      reinterpret_cast<amdf_gpu_umd_user_queue_mapping_t*>(uintptr_t{1});
  auto* mapping = sentinel;
  amdf_gpu_umd_user_queue_mapping_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original = result;
  EXPECT_EQ(amdf_gpu_umd_user_queue_map(queue_, &device_, &mapping, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(mapping, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(native_state_.device_doorbell.create_count, 0u);
}

TEST_F(KfdUserQueueTest, BorrowsExactOwnerDeviceAddressesUntilQueueRelease) {
  auto create = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  create.required_capabilities |= AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.capabilities, create.required_capabilities);
  ASSERT_EQ(amdf_gpu_umd_user_queue_map(queue_, &device_, &mapping_,
                                        &mapping_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_result_.ring_address,
            native_state_.buffers[0].device_address);
  EXPECT_EQ(mapping_result_.read_index_address,
            native_state_.buffers[1].device_address);
  EXPECT_EQ(mapping_result_.write_index_address,
            native_state_.buffers[1].device_address + 64);
  EXPECT_EQ(mapping_result_.doorbell_address, UINT64_C(0x50000080));
  EXPECT_EQ(mapping_result_.index_bits, 64u);
  EXPECT_EQ(mapping_result_.doorbell_bits, 64u);
  EXPECT_EQ(mapping_result_.metadata_ring_address, 0u);

  amdf_gpu_umd_device_t other_device = device_;
  auto* rejected_mapping = mapping_;
  auto rejected_result = mapping_result_;
  EXPECT_EQ(amdf_gpu_umd_user_queue_map(queue_, &other_device,
                                        &rejected_mapping, &rejected_result),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(rejected_mapping, mapping_);
  EXPECT_EQ(
      std::memcmp(&rejected_result, &mapping_result_, sizeof(rejected_result)),
      0);

  EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  EXPECT_EQ(native_state_.device_doorbell.destroy_count, 0u);
  EXPECT_TRUE(native_state_.device_doorbell.live);
  MapQueue();
  EXPECT_EQ(
      mapping_result_.ring_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[0].storage.data()));
  EXPECT_EQ(native_state_.device_doorbell.create_count, 1u);
  EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.device_doorbell.destroy_count, 1u);
  EXPECT_FALSE(native_state_.device_doorbell.live);
  EXPECT_EQ(native_state_.device_doorbell.abandon_count, 0u);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_F(KfdUserQueueTest, DeviceConstructionHasNoPartialCapabilityResult) {
  auto create = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  create.required_capabilities |= AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  for (int failed_operation = 1; failed_operation <= 6; ++failed_operation) {
    SCOPED_TRACE(failed_operation);
    native_state_.Reset();
    native_state_.failed_creation_operation = failed_operation;
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    auto* output = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original = result;
    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create, &output, &result),
        native_state_.creation_failure);
    EXPECT_EQ(output, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    EXPECT_FALSE(native_state_.device_doorbell.live);
    EXPECT_EQ(native_state_.queue_destroy_count, failed_operation >= 5 ? 1 : 0);
  }
}

TEST_F(KfdUserQueueTest, DeviceViewFollowsTerminalQueueOwnership) {
  for (int failure_stage : {0, 1, 2}) {
    SCOPED_TRACE(failure_stage);
    native_state_.Reset();
    auto create = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    create.required_capabilities |= AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
    if (failure_stage == 0) {
      native_state_.queue_destroy_status = failure;
    } else if (failure_stage == 1) {
      native_state_.failed_buffer_destroy_call = 1;
      native_state_.buffer_destroy_failure = failure;
    } else {
      native_state_.device_doorbell.destroy_status = failure;
    }
    EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), failure);
    queue_ = nullptr;
    EXPECT_EQ(native_state_.device_doorbell.destroy_count,
              failure_stage == 2 ? 1u : 0u);
    EXPECT_EQ(native_state_.device_doorbell.abandon_count,
              failure_stage == 2 ? 0u : 1u);
    EXPECT_TRUE(native_state_.device_doorbell.live);
    EXPECT_EQ(native_state_.doorbell_unmap_count, 0);
  }
}

TEST_F(KfdUserQueueTest, RejectsUnsupportedQueueWithoutPublishingOutputs) {
  amdf_gpu_umd_user_queue_create_info_t create_info = MakeCreateInfo();
  auto* const sentinel =
      reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
  amdf_gpu_umd_user_queue_t* queue = sentinel;
  amdf_gpu_umd_user_queue_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_user_queue_result_t original_result = result;

  device_.topology.properties.topology.xcc_count = 2;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  device_.topology.properties.topology.xcc_count = 1;
  create_info.priority = AMDF_QUEUE_PRIORITY_HIGH;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  create_info = MakeCreateInfo();
  create_info.ring_byte_length = UINT64_C(1) << 32;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  create_info = MakeCreateInfo();
  create_info.roles = AMDF_QUEUE_ROLE_COMPUTE;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);

  EXPECT_EQ(queue, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(native_state_.creation_operation_count, 0);
}

TEST_F(KfdUserQueueTest, SamplesProgressAndLatchesTerminalFailures) {
  CreateQueue();
  MapQueue();
  amdf_atomic_uint64_store_release(WriteIndex(), 16);
  amdf_atomic_uint64_store_release(ReadIndex(), 8);

  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(status.producer_index, 16u);
  EXPECT_EQ(status.consumed_index, 8u);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);

  const uint64_t queue_error = KFD_EC_MASK(EC_QUEUE_WAVE_TRAP);
  amdf_atomic_uint64_store_release(ErrorPayload(), queue_error);
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
  EXPECT_EQ(status.terminal_status,
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE,
                             static_cast<uint32_t>(queue_error)));

  amdf_atomic_uint64_store_release(ErrorPayload(), 0);
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
  EXPECT_EQ(status.terminal_status,
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE,
                             static_cast<uint32_t>(queue_error)));
}

class KfdSizedUserQueueTest : public KfdUserQueueTest,
                              public ::testing::WithParamInterface<uint64_t> {};

TEST_P(KfdSizedUserQueueTest, CarriesCapacityThroughNativeCreationAndMapping) {
  for (auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL}) {
    SCOPED_TRACE(command_type);
    native_state_.Reset();
    auto create_info = MakeCreateInfo(command_type);
    create_info.ring_byte_length = GetParam();
    const uint64_t capacity = GetParam() == 0 ? 4096 : GetParam();
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    EXPECT_EQ(native_state_.buffers[0].byte_length, capacity);
    EXPECT_EQ(native_state_.observed_create.ring_size, capacity);
    EXPECT_EQ(native_state_.observed_create.ring_base_address,
              native_state_.buffers[0].device_address);
    EXPECT_EQ(queue_result_.ring_byte_length, capacity);
    if (command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      EXPECT_EQ(native_state_.observed_aql_descriptor.packet_count,
                capacity / 64);
    }
    ASSERT_NO_FATAL_FAILURE(MapQueue());
    EXPECT_EQ(
        mapping_result_.ring_address,
        reinterpret_cast<uint64_t>(native_state_.buffers[0].storage.data()));
    ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_),
              AMDF_STATUS_OK);
    mapping_ = nullptr;
    ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    EXPECT_EQ(native_state_.queue_destroy_count, 1);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
  }
}

TEST_F(KfdUserQueueTest, MaximumRingRequestReachesAllocationWithoutNarrowing) {
  for (auto command_type :
       {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL}) {
    SCOPED_TRACE(command_type);
    native_state_.Reset();
    native_state_.failed_creation_operation = 1;
    auto create_info = MakeCreateInfo(command_type);
    create_info.ring_byte_length = UINT64_C(1) << 31;
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    auto* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original = result;
    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result),
        native_state_.creation_failure);
    EXPECT_EQ(native_state_.buffers[0].create_info.byte_length,
              create_info.ring_byte_length);
    EXPECT_EQ(native_state_.buffer_create_count, 1);
    EXPECT_EQ(native_state_.queue_create_count, 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  }
}

TEST_P(KfdSizedUserQueueTest, ExpandsPm4ProgressAcrossUnobservedRingWraps) {
  auto create_info = MakeCreateInfo();
  create_info.ring_byte_length = GetParam();
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  const uint64_t capacity = queue_result_.ring_byte_length / sizeof(uint32_t);
  const amdf_wait_deadline_t deadline = {0, 0};
  // Native RPTR is ring-relative even if no status query observed earlier
  // laps. The caller's publication window always leaves at least one slot.
  for (uint64_t published : {capacity + 184, capacity * 1024 + 184}) {
    SCOPED_TRACE(published);
    amdf_atomic_uint64_store_release(WriteIndex(), published);
    for (uint64_t pending : {capacity - 1, capacity / 2, UINT64_C(184),
                             UINT64_C(8), UINT64_C(0)}) {
      SCOPED_TRACE(pending);
      const uint64_t consumed = published - pending;
      amdf_atomic_uint64_store_release(ReadIndex(), consumed & (capacity - 1));
      amdf_user_queue_status_t status = {};
      ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.producer_index, published);
      EXPECT_EQ(status.consumed_index, consumed);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      EXPECT_EQ(
          amdf_gpu_umd_user_queue_wait_consumed(queue_, consumed, &deadline),
          AMDF_STATUS_OK);
      if (pending != 0) {
        EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_wait_consumed(
                      queue_, published, &deadline)),
                  AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
        EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_destroy(queue_)),
                  AMDF_STATUS_CODE_BUSY);
      }
    }
  }
  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_P(KfdSizedUserQueueTest,
       PreservesSdmaMonotonicProgressBeyondRingCapacity) {
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  create_info.ring_byte_length = GetParam();
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  const uint64_t capacity = queue_result_.ring_byte_length;
  amdf_atomic_uint64_store_release(WriteIndex(), 3 * capacity + 40);
  amdf_atomic_uint64_store_release(ReadIndex(), 2 * capacity + 48);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.producer_index, 3 * capacity + 40);
  EXPECT_EQ(status.consumed_index, 2 * capacity + 48);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
}

INSTANTIATE_TEST_SUITE_P(Capacities, KfdSizedUserQueueTest,
                         ::testing::Values(UINT64_C(0), UINT64_C(4096),
                                           UINT64_C(8192), UINT64_C(16384),
                                           UINT64_C(65536)));

TEST_F(KfdUserQueueTest, ClassifiesDeviceFailure) {
  CreateQueue();
  MapQueue();
  native_state_.vm_fault.info = {
      .addr = native_state_.buffers[1].device_address,
      .status = 0x00800830,
      .vmhub = AMDGPU_VMHUB_TYPE_GFX,
  };
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  EXPECT_EQ(status.terminal_status,
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(amdf_atomic_uint64_load_acquire(ErrorPayload()), 0u);

  // A latched failure remains observable even when a subsequent ioctl fails.
  native_state_.vm_fault.query_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_DEVICE_LOST);
  EXPECT_EQ(native_state_.vm_fault.query_count, 1u);
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  const amdf_wait_deadline_t deadline = {UINT64_MAX, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(native_state_.queue_destroy_count, 0);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);
}

TEST_F(KfdUserQueueTest, PropagatesNativeFaultObservationFailure) {
  CreateQueue();
  MapQueue();
  const amdf_status_t failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  native_state_.vm_fault.query_status = failure;
  amdf_user_queue_status_t status;
  std::memset(&status, 0xA5, sizeof(status));
  const amdf_user_queue_status_t original_status = status;
  EXPECT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status), failure);
  EXPECT_EQ(std::memcmp(&status, &original_status, sizeof(status)), 0);
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  const amdf_wait_deadline_t deadline = {UINT64_MAX, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            failure);

  // Failure to observe is not itself a terminal device failure.
  native_state_.vm_fault.query_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
}

TEST_F(KfdUserQueueTest, ObservesVmFaultWithoutComputeContextStorage) {
  CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  MapQueue();
  native_state_.vm_fault.info.status = 0x00800830;
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  const amdf_wait_deadline_t deadline = {UINT64_MAX, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
}

TEST_F(KfdUserQueueTest, MalformedProgressLatchesInternalFailure) {
  CreateQueue();
  MapQueue();
  amdf_atomic_uint64_store_release(ReadIndex(), 17);
  amdf_atomic_uint64_store_release(WriteIndex(), 16);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.terminal_status,
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  amdf_atomic_uint64_store_release(ReadIndex(), 16);
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.terminal_status,
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
}

TEST_F(KfdUserQueueTest, WaitAndDestroyRequireConsumedPublication) {
  native_state_.created_queue_identifier = 0;
  CreateQueue();
  MapQueue();
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  amdf_atomic_uint64_store_release(ReadIndex(), 4);

  const amdf_wait_deadline_t expired_deadline = {0, 0};
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_wait_consumed(
                queue_, 9, &expired_deadline)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_wait_consumed(
                queue_, 8, &expired_deadline)),
            AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(native_state_.queue_destroy_count, 0);

  amdf_atomic_uint64_store_release(ReadIndex(), 8);
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &expired_deadline),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  ASSERT_EQ(native_state_.destroyed_queue_identifiers.size(), 1u);
  EXPECT_EQ(native_state_.destroyed_queue_identifiers[0], 0u);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
  EXPECT_EQ(native_state_.destroyed_buffer_indices,
            (std::vector<size_t>{4, 3, 2, 1, 0}));
}

TEST_F(KfdUserQueueTest, NativeRemovalFailureConsumesOwnerAndPreservesBacking) {
  // These errors include both retained and consumed native identifiers. None
  // supplies proof that queue-reachable storage is safe to release.
  for (int error : {EBUSY, ETIME, EIO}) {
    SCOPED_TRACE(error);
    native_state_.Reset();
    CreateQueue();
    native_state_.queue_destroy_status =
        amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, error);
    EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
              native_state_.queue_destroy_status);
    queue_ = nullptr;
    EXPECT_EQ(native_state_.destroyed_queue_identifiers,
              (std::vector<uint32_t>{47}));
    EXPECT_EQ(native_state_.queue_destroy_count, 1);
    EXPECT_EQ(native_state_.buffer_destroy_count, 0);
    EXPECT_EQ(native_state_.doorbell_unmap_count, 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 5u);
    EXPECT_EQ(native_state_.abandoned_buffer_indices,
              (std::vector<size_t>{0, 1, 2, 3, 4}));
  }
}

TEST_F(KfdUserQueueTest, StorageReleaseFailureConsumesOwnerWithoutRetry) {
  CreateQueue();
  native_state_.doorbell_unmap_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.doorbell_unmap_status);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 4u);

  EXPECT_EQ(native_state_.destroyed_buffer_indices, (std::vector<size_t>{4}));
  EXPECT_EQ(native_state_.abandoned_buffer_indices,
            (std::vector<size_t>{0, 1, 2, 3}));
}

TEST_F(KfdUserQueueTest, FailedRetirementFlushPreservesAllReachableBacking) {
  CreateQueue();
  native_state_.failed_buffer_destroy_call = 1;
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.buffer_destroy_failure);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 0);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);

  EXPECT_EQ(native_state_.buffer_destroy_count, 1);
  EXPECT_TRUE(native_state_.destroyed_buffer_indices.empty());
  EXPECT_EQ(native_state_.abandoned_buffer_indices,
            (std::vector<size_t>{0, 1, 2, 3, 4}));
}

TEST_F(KfdUserQueueTest,
       FailedRollbackAbandonsMetadataWithoutReleasingBacking) {
  struct Scenario {
    // Native queue-destroy error; zero selects successful native destruction.
    int destroy_error;
    // One-based buffer release call to fail; zero leaves every release working.
    int failed_buffer_destroy_call;
    // Backing whose native release completed before the terminal error.
    std::vector<size_t> released;
    // Remaining backing whose metadata alone must be abandoned.
    std::vector<size_t> abandoned;
  };
  const Scenario scenarios[] = {
      {EBUSY, 0, {}, {0, 1, 2, 3, 4}},
      {ETIME, 0, {}, {0, 1, 2, 3, 4}},
      {0, 1, {}, {0, 1, 2, 3, 4}},
      {0, 3, {4, 3}, {0, 1, 2}},
  };
  for (const auto& scenario : scenarios) {
    SCOPED_TRACE(scenario.destroy_error);
    SCOPED_TRACE(scenario.failed_buffer_destroy_call);
    native_state_.Reset();
    native_state_.failed_creation_operation = 7;
    native_state_.queue_destroy_status =
        scenario.destroy_error == 0 ? AMDF_STATUS_OK
                                    : amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO,
                                                       scenario.destroy_error);
    native_state_.failed_buffer_destroy_call =
        scenario.failed_buffer_destroy_call;
    const amdf_gpu_umd_user_queue_create_info_t create_info = MakeCreateInfo();
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    amdf_gpu_umd_user_queue_t* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original_result = result;

    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result),
        scenario.destroy_error != 0 ? native_state_.queue_destroy_status
                                    : native_state_.buffer_destroy_failure);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), scenario.abandoned.size());
    EXPECT_EQ(native_state_.queue_destroy_count, 1);
    EXPECT_EQ(native_state_.buffer_destroy_count,
              scenario.failed_buffer_destroy_call);
    EXPECT_EQ(native_state_.destroyed_buffer_indices, scenario.released);
    EXPECT_EQ(native_state_.abandoned_buffer_indices, scenario.abandoned);
  }
}

TEST_F(KfdUserQueueTest,
       AqlInitializesAllNativeReachableStorageBeforeActivation) {
  SelectAqlTopology();
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL));
  const auto& descriptor = native_state_.observed_aql_descriptor;
  EXPECT_EQ(descriptor.queue_type, 1u);
  EXPECT_EQ(descriptor.packet_count, 64u);
  EXPECT_EQ(descriptor.ring_address, native_state_.buffers[0].device_address);
  EXPECT_EQ(descriptor.read_dispatch_id_byte_offset, 128u);
  EXPECT_EQ(descriptor.maximum_compute_unit_id, 15u);
  EXPECT_EQ(descriptor.maximum_wave_id, 31u);
  EXPECT_EQ(descriptor.group_segment_aperture_base_hi, 0x10000u);
  EXPECT_EQ(descriptor.private_segment_aperture_base_hi, 0x20000u);
  EXPECT_EQ(descriptor.compute_temporary_ring_size, 0u);
  EXPECT_EQ(descriptor.scratch_backing_address, 0u);
  EXPECT_EQ(descriptor.inactive_signal_address,
            native_state_.buffers[1].device_address + 256);
  const auto* signal = reinterpret_cast<const amdf_gpu_kfd_aql_signal_t*>(
      reinterpret_cast<const uint8_t*>(
          native_state_.buffers[1].storage.data()) +
      256);
  EXPECT_EQ(signal->kind, 1);
  EXPECT_EQ(signal->value, 0u);
  EXPECT_EQ(signal->event_mailbox_address, 0u);
  EXPECT_EQ(native_state_.observed_create.eop_buffer_size, 0u);
  EXPECT_EQ(native_state_.buffer_create_count, 4);
  ASSERT_EQ(native_state_.observed_context_headers.size(), 8u);
  for (size_t i = 0; i < 8; ++i) {
    const auto& header = native_state_.observed_context_headers[i];
    EXPECT_EQ(header.debug_offset, (8 - i) * 4096);
    EXPECT_EQ(header.debug_size, 20480u);
    EXPECT_EQ(header.err_payload_addr,
              native_state_.buffers[1].device_address + 320);
  }
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  EXPECT_EQ(
      mapping_result_.read_index_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[1].storage.data()) +
          128);
  EXPECT_EQ(
      mapping_result_.write_index_address,
      reinterpret_cast<uintptr_t>(native_state_.buffers[1].storage.data()) +
          56);
}

TEST_F(KfdUserQueueTest, AqlEncodesCallerOwnedRetainedScratchPerXcc) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  create_info.scratch = {
      .device_address = UINT64_C(0x123456780000),
      .byte_length = 512 * 3072,
      .maximum_private_segment_byte_length = 33,
      .maximum_wave_count = 512,
  };
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  const auto& descriptor = native_state_.observed_aql_descriptor;
  EXPECT_EQ(descriptor.compute_temporary_ring_size, (3u << 12) | 64u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[0], 0x56780000u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[1], 0x80001234u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[2], 64u * 3072);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[3], 0x00ea4facu);
  EXPECT_EQ(descriptor.scratch_backing_address,
            create_info.scratch.device_address);
  EXPECT_EQ(descriptor.scratch_wave64_lane_byte_length, 48u);
  EXPECT_EQ(descriptor.queue_properties, 2u);
  EXPECT_EQ(native_state_.buffer_create_count, 4);
}

TEST_F(KfdUserQueueTest,
       RdnaAqlDescriptorUsesNativeAperturesAndScratchEncoding) {
  for (uint32_t target : {110000u, 110501u, 110700u, 120000u, 120500u}) {
    SCOPED_TRACE(target);
    native_state_.Reset();
    device_.topology.properties.gfx_ip = {target / 10000, (target / 100) % 100,
                                          target % 100};
    const bool multiple_xccs = target == 120500;
    device_.topology.gc_ip = multiple_xccs
                                 ? amdf_gpu_kfd_ip_version_t{12, 1, 0, true}
                                 : amdf_gpu_kfd_ip_version_t{};
    device_.topology.properties.compute.compute_unit_count =
        multiple_xccs ? 76 : 38;
    device_.topology.properties.topology.xcc_count = multiple_xccs ? 2 : 1;
    device_.topology.properties.topology.shader_engine_count_per_xcc = 4;
    auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    create_info.scratch = {
        .device_address = UINT64_C(0x123456780000),
        .byte_length = multiple_xccs ? 7864320u : 3932160u,
        .maximum_private_segment_byte_length = 33,
        .maximum_wave_count = multiple_xccs ? 2560u : 1280u,
    };
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    const auto& descriptor = native_state_.observed_aql_descriptor;
    EXPECT_EQ(descriptor.group_segment_aperture_base_hi,
              multiple_xccs ? 0x20000000u : 0x10000u);
    EXPECT_EQ(descriptor.private_segment_aperture_base_hi,
              multiple_xccs ? 0x10000000u : 0x20000u);
    EXPECT_EQ(descriptor.scratch_resource_descriptor[0], 0x56780000u);
    EXPECT_EQ(descriptor.scratch_resource_descriptor[1], 0x40001234u);
    EXPECT_EQ(descriptor.scratch_resource_descriptor[2], 3932160u);
    EXPECT_EQ(descriptor.scratch_resource_descriptor[3], 0x20814facu);
    EXPECT_EQ(descriptor.scratch_wave64_lane_byte_length, 48u);
    EXPECT_EQ(descriptor.compute_temporary_ring_size, 0xc140u);
    EXPECT_EQ(descriptor.maximum_compute_unit_id, multiple_xccs ? 75u : 37u);
    EXPECT_EQ(descriptor.maximum_wave_id, 31u);
    EXPECT_EQ(descriptor.capabilities, 0u);
    EXPECT_EQ(descriptor.queue_properties, 2u);
    EXPECT_EQ(native_state_.observed_create.eop_buffer_size, 4096u);
    EXPECT_EQ(native_state_.observed_context_headers.size(),
              multiple_xccs ? 2u : 1u);
    for (size_t i = 0; i < native_state_.observed_context_headers.size(); ++i) {
      const auto& header = native_state_.observed_context_headers[i];
      EXPECT_EQ(header.debug_offset, (multiple_xccs ? 2u - i : 1u) * 4096u);
      EXPECT_EQ(header.debug_size, multiple_xccs ? 155648u : 38912u);
    }
    ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
  }
}

TEST_F(KfdUserQueueTest, AqlScratchUsesTargetWaveSizeUnitsAndLimits) {
  for (uint32_t major : {9u, 11u, 12u}) {
    SCOPED_TRACE(major);
    native_state_.Reset();
    device_.topology.properties.gfx_ip = {major, major == 9 ? 4u : 0u, 0};
    device_.topology.properties.compute.wavefront_size = major == 9 ? 64 : 32;
    device_.topology.properties.compute.compute_unit_count = 1;
    auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    create_info.scratch = {
        .device_address = UINT64_C(0x123400000000),
        .byte_length = major == 12 ? UINT64_C(2147418112) : UINT64_C(268402688),
        .maximum_private_segment_byte_length = major == 12 ? 1048544u : 131056u,
        .maximum_wave_count = 32,
    };
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    EXPECT_EQ(native_state_.observed_aql_descriptor.compute_temporary_ring_size,
              major == 9    ? 0x01fff020u
              : major == 11 ? 0x07ffc020u
                            : 0x3fff8020u);
    ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    native_state_.Reset();
    ++create_info.scratch.maximum_private_segment_byte_length;
    create_info.scratch.byte_length += 32 * 1024;
    EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                  &device_, &create_info, &queue_, &queue_result_)),
              AMDF_STATUS_CODE_OUT_OF_RANGE);
    EXPECT_EQ(native_state_.creation_operation_count, 0);
    EXPECT_EQ(queue_, nullptr);
  }
}

TEST_F(KfdUserQueueTest, RdnaAqlScratchFreeQueueStillInitializesEopAndSrd) {
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL));
  const auto& descriptor = native_state_.observed_aql_descriptor;
  EXPECT_EQ(descriptor.scratch_resource_descriptor[0], 0u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[1], 0x40000000u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[2], 0u);
  EXPECT_EQ(descriptor.scratch_resource_descriptor[3], 0x20814facu);
  EXPECT_EQ(descriptor.compute_temporary_ring_size, 0u);
  EXPECT_EQ(descriptor.scratch_backing_address, 0u);
  EXPECT_EQ(native_state_.observed_create.eop_buffer_size, 4096u);
  EXPECT_EQ(native_state_.buffer_create_count, 5);
  ASSERT_EQ(native_state_.observed_context_headers.size(), 1u);
  EXPECT_EQ(native_state_.observed_context_headers[0].debug_size, 2048u);
}

TEST_F(KfdUserQueueTest, MultiXccPm4InitializesEveryNativeContextHeader) {
  device_.topology.properties.gfx_ip = {12, 5, 0};
  device_.topology.gc_ip = {12, 1, 0, true};
  device_.topology.properties.topology.xcc_count = 2;
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  EXPECT_EQ(native_state_.observed_create.queue_percentage,
            KFD_MAX_QUEUE_PERCENTAGE);
  EXPECT_EQ(native_state_.observed_create.eop_buffer_size, 4096u);
  ASSERT_EQ(native_state_.observed_context_headers.size(), 2u);
  EXPECT_EQ(native_state_.observed_context_headers[0].debug_offset, 8192u);
  EXPECT_EQ(native_state_.observed_context_headers[1].debug_offset, 4096u);
  EXPECT_EQ(native_state_.observed_context_headers[0].debug_size, 4096u);
  EXPECT_EQ(native_state_.observed_context_headers[1].debug_size, 4096u);
}

TEST_F(KfdUserQueueTest, AqlReservationsMayLeadPublicationAndWrapTheRing) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  create_info.producer_mode = AMDF_QUEUE_PRODUCER_MODE_MULTI;
  create_info.ring_byte_length = 16384;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.ring_byte_length, 16384u);
  EXPECT_EQ(native_state_.observed_aql_descriptor.queue_type, 0u);
  EXPECT_EQ(native_state_.observed_aql_descriptor.packet_count, 256u);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  amdf_atomic_uint64_store_release(WriteIndex(), 1025);
  amdf_atomic_uint64_store_release(ReadIndex(), 257);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.producer_index, 1025u);
  EXPECT_EQ(status.consumed_index, 257u);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
}

TEST_F(KfdUserQueueTest,
       AqlScratchRejectsUnrepresentableRequestsBeforeAllocation) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  const amdf_gpu_umd_queue_scratch_t valid = {
      .device_address = UINT64_C(0x123456780000),
      .byte_length = 512 * 3072,
      .maximum_private_segment_byte_length = 33,
      .maximum_wave_count = 512,
  };
  for (uint32_t scenario = 0; scenario < 5; ++scenario) {
    SCOPED_TRACE(scenario);
    create_info.scratch = valid;
    switch (scenario) {
      case 0:
        --create_info.scratch.maximum_wave_count;
        break;
      case 1:
        --create_info.scratch.byte_length;
        break;
      case 2:
        ++create_info.scratch.device_address;
        break;
      case 3:
        create_info.scratch.device_address = UINT64_C(1) << 48;
        break;
      case 4:
        create_info.scratch.maximum_private_segment_byte_length = UINT32_MAX;
    }
    auto* sentinel = reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    auto* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original_result = result;
    const amdf_status_t status =
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result);
    EXPECT_EQ(amdf_status_code(status), scenario == 0
                                            ? AMDF_STATUS_CODE_UNSUPPORTED
                                            : AMDF_STATUS_CODE_OUT_OF_RANGE);
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.buffer_create_count, 0);
  }
}

TEST_F(KfdUserQueueTest, AqlConstructionRollbackPreservesCallerOwnership) {
  SelectAqlTopology();
  const auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  for (int operation = 1; operation <= 6; ++operation) {
    SCOPED_TRACE(operation);
    native_state_.Reset();
    native_state_.failed_creation_operation = operation;
    amdf_gpu_umd_user_queue_t* queue = nullptr;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original_result = result;
    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue, &result),
        native_state_.creation_failure);
    EXPECT_EQ(queue, nullptr);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    EXPECT_EQ(native_state_.queue_destroy_count, operation == 6 ? 1 : 0);
  }
}

TEST_F(KfdUserQueueTest, AqlStatusObservesItsNativeInactiveSignal) {
  SelectAqlTopology();
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL));
  auto* value = reinterpret_cast<amdf_atomic_uint64_t*>(
      reinterpret_cast<uint8_t*>(native_state_.buffers[1].storage.data()) +
      264);
  amdf_atomic_uint64_store_release(value, 1);
  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
  EXPECT_EQ(status.terminal_status,
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 1));
  amdf_atomic_uint64_store_release(value, 0);
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
}

TEST_F(KfdUserQueueTest, ComputeFamilyCannotSilentlyIgnoreRequestedScratch) {
  auto create_info = MakeCreateInfo();
  create_info.scratch = {.device_address = 0x10000000,
                         .byte_length = 4096,
                         .maximum_private_segment_byte_length = 16,
                         .maximum_wave_count = 4};
  EXPECT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(queue_, nullptr);
  EXPECT_EQ(native_state_.buffer_create_count, 0);
}

}  // namespace
