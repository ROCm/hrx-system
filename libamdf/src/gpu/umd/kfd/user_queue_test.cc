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
    if (self->FailCreationOperation()) return self->creation_failure;
    FakeBuffer& buffer = self->buffers[self->buffer_create_count - 1];
    buffer.storage.resize((create_info->byte_length + sizeof(uint64_t) - 1) /
                          sizeof(uint64_t));
    buffer.create_info = *create_info;
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
      void* user_data, amdf_gpu_umd_device_t*,
      struct kfd_ioctl_create_queue_args* inout_arguments) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->queue_create_count;
    self->observed_create = *inout_arguments;
    if (self->FailCreationOperation()) return self->creation_failure;
    inout_arguments->queue_id = self->created_queue_identifier;
    inout_arguments->doorbell_offset = self->doorbell_offset;
    return AMDF_STATUS_OK;
  }

  static amdf_gpu_kfd_user_queue_destroy_result_t QueueDestroy(
      void* user_data, amdf_gpu_umd_device_t*, uint32_t queue_identifier) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->queue_destroy_count;
    self->destroyed_queue_identifiers.push_back(queue_identifier);
    return {
        .status = self->queue_destroy_status,
        .identifier_consumed = self->queue_identifier_consumed,
    };
  }

  static amdf_status_t DoorbellMap(void* user_data, amdf_gpu_umd_device_t*,
                                   uint64_t native_byte_offset,
                                   size_t byte_length, void** out_mapping) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->doorbell_map_count;
    self->observed_doorbell_mapping_offset = native_byte_offset;
    self->observed_doorbell_mapping_length = byte_length;
    if (self->FailCreationOperation()) return self->creation_failure;
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

  static amdf_status_t ResetQuery(void* user_data, amdf_gpu_umd_device_t*,
                                  amdf_gpu_kfd_reset_state_t* out_state) {
    auto* self = static_cast<FakeNativeState*>(user_data);
    ++self->reset_query_count;
    if (!amdf_status_is_ok(self->reset_query_status)) {
      return self->reset_query_status;
    }
    *out_state = self->reset_state;
    return AMDF_STATUS_OK;
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

  void Reset() {
    for (FakeBuffer& buffer : buffers) buffer = {};
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
    queue_identifier_consumed = true;
    destroyed_queue_identifiers.clear();
    doorbell_map_count = 0;
    doorbell_unmap_count = 0;
    doorbell_unmap_status = AMDF_STATUS_OK;
    reset_query_count = 0;
    reset_query_status = AMDF_STATUS_OK;
    reset_state = {};
    vm_fault = {};
    observed_create = {};
    observed_doorbell_mapping_offset = 0;
    observed_doorbell_mapping_length = 0;
    observed_doorbell_unmapping = nullptr;
    observed_doorbell_unmapping_length = 0;
  }

  size_t LiveBufferCount() const {
    size_t count = 0;
    for (const FakeBuffer& buffer : buffers) count += buffer.live ? 1 : 0;
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
  bool queue_identifier_consumed = true;
  std::vector<uint32_t> destroyed_queue_identifiers;
  int doorbell_map_count = 0;
  int doorbell_unmap_count = 0;
  amdf_status_t doorbell_unmap_status = AMDF_STATUS_OK;
  int reset_query_count = 0;
  amdf_status_t reset_query_status = AMDF_STATUS_OK;
  amdf_gpu_kfd_reset_state_t reset_state = {};
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
    device_.topology.properties.topology.xcc_count = 1;
    device_.topology.gpu_id = 73;
    device_.topology.compute_queue_count = 8;
    device_.topology.sdma.engine_count = 1;
    device_.topology.sdma.queue_count_per_engine = 6;
    device_.topology.sdma.ip = {6, 1, 1, true};
    device_.topology.context_save_restore_byte_length = 4096;
    device_.topology.control_stack_byte_length = 4096;
    device_.topology.virtual_address.alignment = 4096;
    device_.reset_monitor.context_owned = true;
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
        auto* read_index = reinterpret_cast<amdf_atomic_uint64_t*>(
            native_state_.buffers[1].storage.data());
        auto* write_index = reinterpret_cast<amdf_atomic_uint64_t*>(
            reinterpret_cast<uint8_t*>(
                native_state_.buffers[1].storage.data()) +
            64);
        const uint64_t published_index =
            amdf_atomic_uint64_load_acquire(write_index);
        amdf_atomic_uint64_store_release(read_index, published_index);
      }
      native_state_.queue_destroy_status = AMDF_STATUS_OK;
      native_state_.queue_identifier_consumed = true;
      native_state_.doorbell_unmap_status = AMDF_STATUS_OK;
      native_state_.failed_buffer_destroy_call = 0;
      native_state_.reset_query_status = AMDF_STATUS_OK;
      native_state_.reset_state = {.reset_observed = true};
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
      if (family.command_type != command_type) continue;
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
      .vm_fault_query = FakeNativeState::VmFaultQuery,
      .reset_query = FakeNativeState::ResetQuery,
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
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
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
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
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
            KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
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
  EXPECT_EQ(queue_result_.ring_byte_length, 4096u);
  EXPECT_EQ(queue_result_.metadata_ring_byte_length, 0u);
  ASSERT_EQ(native_state_.LiveBufferCount(), 3u);
  ASSERT_EQ(native_state_.buffer_create_count, 3);

  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  EXPECT_EQ(native_state_.buffers[0].create_info.native_flags,
            host_storage_flags);
  EXPECT_EQ(native_state_.buffers[0].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[1].create_info.native_flags,
            host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED);
  EXPECT_EQ(native_state_.buffers[1].create_info.byte_length, 4096u);
  EXPECT_EQ(native_state_.buffers[2].create_info.native_flags,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
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

TEST_F(KfdUserQueueTest, RejectsUnqualifiedQueueWithoutPublishingOutputs) {
  amdf_gpu_umd_user_queue_create_info_t create_info = MakeCreateInfo();
  auto* const sentinel =
      reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
  amdf_gpu_umd_user_queue_t* queue = sentinel;
  amdf_gpu_umd_user_queue_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const amdf_gpu_umd_user_queue_result_t original_result = result;

  device_.topology.properties.gfx_ip.stepping = 0;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  device_.topology.properties.gfx_ip.stepping = 1;
  create_info.priority = AMDF_QUEUE_PRIORITY_HIGH;
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_create(
                &device_, &create_info, &queue, &result)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  create_info = MakeCreateInfo();
  create_info.ring_byte_length = 8192;
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
  EXPECT_EQ(native_state_.destroyed_queue_identifiers[0], 47u);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
  EXPECT_EQ(native_state_.destroyed_buffer_indices,
            (std::vector<size_t>{4, 3, 2, 1, 0}));
}

TEST_F(KfdUserQueueTest, RetainedIdentifierRetriesTheSameNativeQueue) {
  CreateQueue();
  native_state_.queue_destroy_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
  native_state_.queue_identifier_consumed = false;
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.queue_destroy_status);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 0);

  native_state_.queue_destroy_status = AMDF_STATUS_OK;
  native_state_.queue_identifier_consumed = true;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  ASSERT_EQ(native_state_.destroyed_queue_identifiers.size(), 2u);
  EXPECT_EQ(native_state_.destroyed_queue_identifiers[0], 47u);
  EXPECT_EQ(native_state_.destroyed_queue_identifiers[1], 47u);
}

TEST_F(KfdUserQueueTest, ConsumedIdentifierWaitsForCompletedReset) {
  CreateQueue();
  native_state_.queue_destroy_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ETIME);
  native_state_.queue_identifier_consumed = true;
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.queue_destroy_status);
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);

  native_state_.queue_destroy_status = AMDF_STATUS_OK;
  native_state_.reset_state = {};
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  native_state_.reset_state = {
      .reset_observed = true,
      .reset_in_progress = true,
  };
  EXPECT_EQ(amdf_status_code(amdf_gpu_umd_user_queue_destroy(queue_)),
            AMDF_STATUS_CODE_BUSY);
  native_state_.reset_state = {.reset_observed = true};
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.reset_query_count, 3);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_F(KfdUserQueueTest, StorageReleaseRetriesWithoutNativeQueueMutation) {
  CreateQueue();
  native_state_.doorbell_unmap_status =
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.doorbell_unmap_status);
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 1);
  EXPECT_EQ(native_state_.LiveBufferCount(), 4u);

  native_state_.doorbell_unmap_status = AMDF_STATUS_OK;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 2);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_F(KfdUserQueueTest, RetirementFlushRetriesWithoutNativeQueueMutation) {
  CreateQueue();
  native_state_.failed_buffer_destroy_call = 1;
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            native_state_.buffer_destroy_failure);
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.doorbell_unmap_count, 0);
  EXPECT_EQ(native_state_.LiveBufferCount(), 5u);

  native_state_.failed_buffer_destroy_call = 0;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.queue_destroy_count, 1);
  EXPECT_EQ(native_state_.destroyed_buffer_indices,
            (std::vector<size_t>{4, 3, 2, 1, 0}));
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
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
    native_state_.queue_identifier_consumed = scenario.destroy_error != EBUSY;
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
    EXPECT_EQ(native_state_.reset_query_count, 0);
    EXPECT_EQ(native_state_.destroyed_buffer_indices, scenario.released);
    EXPECT_EQ(native_state_.abandoned_buffer_indices, scenario.abandoned);
  }
}

}  // namespace
