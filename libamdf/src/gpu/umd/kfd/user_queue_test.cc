// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/user_queue_test_fixture.h"

namespace {

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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
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

  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  device_.topology.gc_ip = {11, 0, 0, true};
  device_.topology.properties.compute.compute_unit_count = 3;
  device_.topology.properties.compute.maximum_wave_count_per_compute_unit = 17;
  device_.topology.context_save_restore_byte_length = 16384;
  device_.topology.control_stack_byte_length = 8192;
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));
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

  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));
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
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  native_state_.vm_fault.info.status = 0x00800830;
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  const amdf_wait_deadline_t deadline = {UINT64_MAX, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
}

TEST_F(KfdUserQueueTest, MalformedProgressLatchesInternalFailure) {
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
  ASSERT_NO_FATAL_FAILURE(MapQueue());
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
    ASSERT_NO_FATAL_FAILURE(CreateQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue());
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

TEST_F(KfdUserQueueTest, AqlSelectsDispatchTimestampsBeforeNativeActivation) {
  for (const uint32_t target :
       {90400u, 90401u, 90402u, 90500u, 110000u, 110001u, 110002u, 110003u,
        110500u, 110501u, 110502u, 110503u, 110700u, 110701u, 110702u, 120000u,
        120001u, 120500u, 120501u}) {
    SCOPED_TRACE(target);
    SelectAqlTopology();
    const bool cdna = target < 100000;
    device_.topology.properties.gfx_ip = {target / 10000, (target / 100) % 100,
                                          target % 100};
    device_.topology.properties.compute.wavefront_size = cdna ? 64 : 32;
    device_.topology.properties.topology.xcc_count = cdna ? 8 : 1;
    device_.topology.gc_ip = target >= 120500
                                 ? amdf_gpu_kfd_ip_version_t{12, 1, 0, true}
                                 : amdf_gpu_kfd_ip_version_t{};
    for (const bool enabled : {false, true}) {
      SCOPED_TRACE(enabled);
      native_state_.Reset();
      auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
      create_info.dispatch_timestamps = enabled;
      ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                               &queue_result_),
                AMDF_STATUS_OK);
      const auto& descriptor = native_state_.observed_aql_descriptor;
      // The fake native create snapshots the actual descriptor before the
      // native queue can fetch it. No later descriptor update is required.
      EXPECT_EQ(descriptor.queue_properties, enabled ? 10u : 2u);
      EXPECT_EQ(descriptor.write_dispatch_id, 0u);
      EXPECT_EQ(descriptor.read_dispatch_id, 0u);
      EXPECT_EQ(native_state_.queue_create_count, 1);
      ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
      queue_ = nullptr;
      EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
    }
  }
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

TEST_F(KfdUserQueueTest, AqlScratchServicePreservesOwnershipAcrossGrowth) {
  for (const uint32_t wavefront_size : {32u, 64u}) {
    SCOPED_TRACE(wavefront_size);
    SelectAqlTopology();
    if (wavefront_size == 32) {
      device_.topology.properties.gfx_ip = {11, 0, 0};
      device_.topology.properties.compute.wavefront_size = 32;
      device_.topology.properties.topology.xcc_count = 1;
    }
    native_state_.Reset();
    auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    create_info.scratch_service = true;
    create_info.inactive_event = {
        .format = AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL,
        .format_version = 1,
        .mailbox_address = 0x76540000,
        .interrupt_context = 23,
    };
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapQueue());
    auto* descriptor = reinterpret_cast<amdf_gpu_kfd_aql_descriptor_t*>(
        native_state_.buffers[1].storage.data());
    auto* signal = reinterpret_cast<amdf_gpu_kfd_aql_signal_t*>(
        reinterpret_cast<uint8_t*>(descriptor) + 256);
    EXPECT_EQ(signal->event_mailbox_address, 0x76540000u);
    EXPECT_EQ(signal->event_id, 23u);
    amdf_gpu_aql_scratch_request_t request;
    ASSERT_EQ(amdf_gpu_umd_user_queue_query_scratch(queue_, &request),
              AMDF_STATUS_OK);
    EXPECT_EQ(request.byte_length, 0u);
    const uint32_t alignment = wavefront_size == 32 ? 256 : 1024;
    for (const uint32_t private_bytes : {17u, 33u, 65u}) {
      // A preceding barrier and invalid slot exercise EOP lag and ring wrap.
      auto* packets = reinterpret_cast<uint32_t*>(mapping_result_.ring_address);
      packets[62 * 16] = 3;
      packets[63 * 16] = 1;
      packets[0] = 2;
      packets[6] = private_bytes;
      amdf_atomic_uint64_store_release(ReadIndex(), 62);
      amdf_atomic_uint64_store_release(WriteIndex(), 65);
      signal->value = wavefront_size == 32 ? 0x401 : 1;
      ASSERT_EQ(amdf_gpu_umd_user_queue_query_scratch(queue_, &request),
                AMDF_STATUS_OK);
      const uint64_t wave_bytes =
          (private_bytes * wavefront_size + alignment - 1) & ~(alignment - 1);
      EXPECT_EQ(request.dispatch_index, 64u);
      EXPECT_EQ(request.wavefront_size, wavefront_size);
      EXPECT_EQ(request.private_segment_byte_length, private_bytes);
      EXPECT_EQ(request.wave_count, 512u);
      EXPECT_EQ(request.byte_length, wave_bytes * 512);
      amdf_user_queue_status_t status;
      ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      amdf_gpu_umd_queue_scratch_t scratch = {
          .device_address =
              UINT64_C(0x120000000000) + uint64_t{private_bytes} * 0x100000000,
          .byte_length = request.byte_length,
          .maximum_private_segment_byte_length = private_bytes,
          .maximum_wave_count = request.wave_count,
      };
      const auto original_descriptor = *descriptor;
      --scratch.byte_length;
      EXPECT_EQ(amdf_gpu_umd_user_queue_resume_scratch(queue_, &scratch),
                amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
      EXPECT_EQ(std::memcmp(descriptor, &original_descriptor,
                            sizeof(original_descriptor)),
                0);
      EXPECT_NE(signal->value, 0u);
      ++scratch.byte_length;
      ASSERT_EQ(amdf_gpu_umd_user_queue_resume_scratch(queue_, &scratch),
                AMDF_STATUS_OK);
      EXPECT_EQ(signal->value, 0u);
      EXPECT_EQ(descriptor->scratch_backing_address, scratch.device_address);
      EXPECT_EQ(descriptor->scratch_resource_descriptor[1] & 0xffff,
                scratch.device_address >> 32);
      EXPECT_EQ(descriptor->scratch_wave64_lane_byte_length, wave_bytes / 64);
      EXPECT_EQ(
          descriptor->scratch_resource_descriptor[2],
          request.byte_length / device_.topology.properties.topology.xcc_count);
      EXPECT_EQ(descriptor->compute_temporary_ring_size >> 12,
                wave_bytes / alignment);
      EXPECT_EQ(descriptor->read_dispatch_id, 62u);
      EXPECT_EQ(descriptor->write_dispatch_id, 65u);
      ASSERT_EQ(amdf_gpu_umd_user_queue_query_scratch(queue_, &request),
                AMDF_STATUS_OK);
      EXPECT_EQ(request.byte_length, 0u);
      EXPECT_EQ(amdf_gpu_umd_user_queue_resume_scratch(queue_, &scratch),
                amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
    }
    // Service does not recreate or allocate native queue storage.
    EXPECT_EQ(native_state_.queue_create_count, 1);
    EXPECT_EQ(native_state_.buffer_create_count, wavefront_size == 32 ? 5 : 4);
    ASSERT_NO_FATAL_FAILURE(TearDown());
  }
}

TEST_F(KfdUserQueueTest, ScratchServiceDoesNotHideOtherFirmwareErrors) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  create_info.scratch_service = true;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  auto* signal = reinterpret_cast<amdf_gpu_kfd_aql_signal_t*>(
      reinterpret_cast<uint8_t*>(native_state_.buffers[1].storage.data()) +
      256);
  signal->value = 0x401 | 2;
  amdf_gpu_aql_scratch_request_t request = {.byte_length = 123};
  EXPECT_EQ(amdf_gpu_umd_user_queue_query_scratch(queue_, &request),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, signal->value));
  EXPECT_EQ(request.byte_length, 123u);
  amdf_user_queue_status_t status;
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
}

TEST_F(KfdUserQueueTest, IdleScratchTrimRebindsOnlyTheScratchDescriptor) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  create_info.scratch_service = true;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  auto* descriptor = reinterpret_cast<amdf_gpu_kfd_aql_descriptor_t*>(
      native_state_.buffers[1].storage.data());
  // Model the state left by a previously serviced and completed dispatch.
  descriptor->read_dispatch_id = descriptor->write_dispatch_id = 37;
  descriptor->scratch_backing_address = 0x123456789000;
  descriptor->scratch_wave64_lane_byte_length = 32;
  descriptor->compute_temporary_ring_size = 0x20040;
  descriptor->scratch_resource_descriptor[0] = 0x56789000;
  descriptor->scratch_resource_descriptor[1] |= 0x1234;
  descriptor->scratch_resource_descriptor[2] = 0x80000;
  const auto original = *descriptor;
  ASSERT_EQ(amdf_gpu_umd_user_queue_trim_scratch(queue_), AMDF_STATUS_OK);
  const auto& updates = native_state_.queue_update;
  ASSERT_EQ(updates.arguments.size(), 2u);
  EXPECT_EQ(updates.arguments[0].queue_percentage, 0u);
  EXPECT_EQ(updates.arguments[1].queue_percentage, KFD_MAX_QUEUE_PERCENTAGE);
  for (const auto& arguments : updates.arguments) {
    EXPECT_EQ(arguments.queue_id, native_state_.created_queue_identifier);
    EXPECT_EQ(arguments.ring_base_address,
              native_state_.observed_create.ring_base_address);
    EXPECT_EQ(arguments.ring_size, native_state_.observed_create.ring_size);
    EXPECT_EQ(arguments.queue_priority,
              native_state_.observed_create.queue_priority);
  }
  EXPECT_EQ(std::memcmp(&updates.descriptors[0], &original, sizeof(original)),
            0);
  auto expected = original;
  expected.compute_temporary_ring_size = 0;
  std::memcpy(expected.scratch_resource_descriptor,
              native_state_.observed_aql_descriptor.scratch_resource_descriptor,
              sizeof(expected.scratch_resource_descriptor));
  expected.scratch_backing_address = 0;
  expected.scratch_wave64_lane_byte_length = 0;
  EXPECT_EQ(std::memcmp(&updates.descriptors[1], &expected, sizeof(expected)),
            0);
  EXPECT_EQ(std::memcmp(descriptor, &expected, sizeof(expected)), 0);
  EXPECT_EQ(amdf_gpu_umd_user_queue_trim_scratch(queue_), AMDF_STATUS_OK);
  EXPECT_EQ(updates.arguments.size(), 2u);
  EXPECT_EQ(native_state_.queue_create_count, 1);
  EXPECT_EQ(native_state_.queue_destroy_count, 0);
  EXPECT_EQ(native_state_.buffer_destroy_count, 0);
}

TEST_F(KfdUserQueueTest, ScratchTrimNativeFailureIsTerminal) {
  for (size_t fail_call : {1u, 2u}) {
    ASSERT_NO_FATAL_FAILURE(SetUp());
    SelectAqlTopology();
    auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    create_info.scratch_service = true;
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    auto* descriptor = reinterpret_cast<amdf_gpu_kfd_aql_descriptor_t*>(
        native_state_.buffers[1].storage.data());
    descriptor->scratch_wave64_lane_byte_length = 32;
    native_state_.queue_update.fail_call = fail_call;
    const amdf_status_t failure =
        amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
    EXPECT_EQ(amdf_gpu_umd_user_queue_trim_scratch(queue_), failure);
    EXPECT_EQ(native_state_.queue_update.arguments.size(), fail_call);
    EXPECT_EQ(descriptor->scratch_wave64_lane_byte_length,
              fail_call == 1 ? 32u : 0u);
    // A retry cannot turn cleared host fields into a false surrender proof.
    EXPECT_EQ(amdf_gpu_umd_user_queue_trim_scratch(queue_), failure);
    EXPECT_EQ(native_state_.queue_update.arguments.size(), fail_call);
    amdf_user_queue_status_t status;
    ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
    EXPECT_EQ(status.terminal_status, failure);
    EXPECT_EQ(native_state_.buffer_destroy_count, 0);
    ASSERT_NO_FATAL_FAILURE(TearDown());
  }
}

TEST_F(KfdUserQueueTest, DispatchTimestampSelectionPreservesQueueAndScratch) {
  SelectAqlTopology();
  auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  auto* descriptor = reinterpret_cast<amdf_gpu_kfd_aql_descriptor_t*>(
      native_state_.buffers[1].storage.data());
  auto expected = *descriptor;
  for (bool enabled : {true, true, false, false}) {
    ASSERT_EQ(amdf_gpu_umd_user_queue_set_dispatch_timestamps(queue_, enabled),
              AMDF_STATUS_OK);
    expected.queue_properties =
        (expected.queue_properties & ~(1u << 3)) | (enabled ? 1u << 3 : 0);
    EXPECT_EQ(std::memcmp(descriptor, &expected, sizeof(expected)), 0);
    EXPECT_TRUE(native_state_.queue_update.arguments.empty());
  }
  // Execution, including all scratch uses, has completed on this queue.
  descriptor->read_dispatch_id = descriptor->write_dispatch_id = 37;
  descriptor->scratch_backing_address = 0x123456789000;
  descriptor->scratch_wave64_lane_byte_length = 32;
  expected = *descriptor;
  for (bool enabled : {true, false, true}) {
    const size_t before = native_state_.queue_update.arguments.size();
    ASSERT_EQ(amdf_gpu_umd_user_queue_set_dispatch_timestamps(queue_, enabled),
              AMDF_STATUS_OK);
    const auto& updates = native_state_.queue_update;
    ASSERT_EQ(updates.arguments.size(), before + 2);
    EXPECT_EQ(updates.arguments[before].queue_percentage, 0u);
    EXPECT_EQ(updates.arguments[before + 1].queue_percentage,
              KFD_MAX_QUEUE_PERCENTAGE);
    EXPECT_EQ(
        std::memcmp(&updates.descriptors[before], &expected, sizeof(expected)),
        0);
    expected.queue_properties =
        (expected.queue_properties & ~(1u << 3)) | (enabled ? 1u << 3 : 0);
    EXPECT_EQ(std::memcmp(&updates.descriptors[before + 1], &expected,
                          sizeof(expected)),
              0);
    EXPECT_EQ(std::memcmp(descriptor, &expected, sizeof(expected)), 0);
    ASSERT_EQ(amdf_gpu_umd_user_queue_set_dispatch_timestamps(queue_, enabled),
              AMDF_STATUS_OK);
    EXPECT_EQ(updates.arguments.size(), before + 2);
  }
  for (const auto& update : native_state_.queue_update.arguments) {
    EXPECT_EQ(update.queue_id, native_state_.created_queue_identifier);
    EXPECT_EQ(update.ring_base_address,
              native_state_.observed_create.ring_base_address);
    EXPECT_EQ(update.ring_size, native_state_.observed_create.ring_size);
    EXPECT_EQ(update.queue_priority,
              native_state_.observed_create.queue_priority);
  }
  EXPECT_EQ(native_state_.queue_create_count, 1);
  EXPECT_EQ(native_state_.queue_destroy_count, 0);
  EXPECT_EQ(native_state_.buffer_destroy_count, 0);
}

TEST_F(KfdUserQueueTest, DispatchTimestampNativeFailureRemainsTerminal) {
  for (size_t fail_call : {1u, 2u}) {
    ASSERT_NO_FATAL_FAILURE(SetUp());
    SelectAqlTopology();
    auto create_info = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
    ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                             &queue_result_),
              AMDF_STATUS_OK);
    auto* descriptor = reinterpret_cast<amdf_gpu_kfd_aql_descriptor_t*>(
        native_state_.buffers[1].storage.data());
    descriptor->read_dispatch_id = descriptor->write_dispatch_id = 37;
    native_state_.queue_update.fail_call = fail_call;
    const amdf_status_t failure =
        amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
    EXPECT_EQ(amdf_gpu_umd_user_queue_set_dispatch_timestamps(queue_, true),
              failure);
    EXPECT_EQ(descriptor->queue_properties, fail_call == 1 ? 2u : 10u);
    // Neither a same-state selection nor a retry can hide native failure.
    for (bool enabled : {false, true}) {
      EXPECT_EQ(
          amdf_gpu_umd_user_queue_set_dispatch_timestamps(queue_, enabled),
          failure);
    }
    EXPECT_EQ(native_state_.queue_update.arguments.size(), fail_call);
    amdf_user_queue_status_t status;
    ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
    EXPECT_EQ(status.terminal_status, failure);
    EXPECT_EQ(native_state_.buffer_destroy_count, 0);
    ASSERT_NO_FATAL_FAILURE(TearDown());
  }
}

TEST_F(KfdUserQueueTest, ComputeFamilyCannotSilentlyIgnoreRequestedScratch) {
  auto create_info = MakeCreateInfo();
  create_info.scratch = {
      .device_address = 0x10000000,
      .byte_length = 4096,
      .maximum_private_segment_byte_length = 16,
      .maximum_wave_count = 4,
  };
  EXPECT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create_info, &queue_,
                                           &queue_result_),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(queue_, nullptr);
  EXPECT_EQ(native_state_.buffer_create_count, 0);
}

}  // namespace
