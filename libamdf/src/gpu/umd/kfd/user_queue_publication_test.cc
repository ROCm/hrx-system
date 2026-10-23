// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/user_queue_test_fixture.h"

namespace {

class KfdQueuePlacementTest
    : public KfdUserQueueTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  void SelectLocalTopology() {
    if (GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      SelectAqlTopology();
    }
    device_.topology.vram.total_byte_length = UINT64_C(1) << 30;
    device_.topology.vram.visible_byte_length = UINT64_C(1) << 30;
    device_.hdp = reinterpret_cast<amdf_gpu_kfd_hdp_t*>(uintptr_t{1});
  }
};

TEST_P(KfdQueuePlacementTest, LocalRingKeepsCoherentControlAndProducerAccess) {
  SelectLocalTopology();
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL;
  const bool device_producer = GetParam() != AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  if (device_producer) {
    create.required_capabilities |= AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER;
  }
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.placement, AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL);
  const auto ring_flags = native_state_.buffers[0].create_info.native_flags;
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0u);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0u);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED, 0u);
  EXPECT_EQ(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0u);
  EXPECT_NE(native_state_.buffers[1].create_info.native_flags &
                KFD_IOC_ALLOC_MEM_FLAGS_GTT,
            0u);
  ASSERT_EQ(
      amdf_gpu_umd_user_queue_map(queue_, nullptr, &mapping_, &mapping_result_),
      AMDF_STATUS_OK);
  EXPECT_EQ(mapping_result_.host_publish_executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(mapping_result_.host_publish_fence, AMDF_HOST_CACHE_FENCE_NONE);
  EXPECT_EQ(native_state_.host_publication.calls_before_activation, 1u);
  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_flush(mapping_), AMDF_STATUS_OK);
  EXPECT_EQ(native_state_.host_publication.call_count, 2u);
  if (device_producer) {
    ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_),
              AMDF_STATUS_OK);
    mapping_ = nullptr;
    ASSERT_EQ(amdf_gpu_umd_user_queue_map(queue_, &device_, &mapping_,
                                          &mapping_result_),
              AMDF_STATUS_OK);
    EXPECT_EQ(mapping_result_.host_publish_executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
    EXPECT_EQ(mapping_result_.host_publish_fence, AMDF_HOST_CACHE_FENCE_NONE);
    EXPECT_EQ(mapping_result_.ring_address,
              native_state_.buffers[0].device_address);
    EXPECT_EQ(mapping_result_.read_index_address,
              native_state_.observed_create.read_pointer_address);
  }
}

TEST_P(KfdQueuePlacementTest, UnifiedLocalRingUsesSystemBacking) {
  if (GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
    SelectAqlTopology();
  }
  device_.topology.integrated = true;
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.placement, AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL);
  EXPECT_NE(native_state_.buffers[0].create_info.native_flags &
                KFD_IOC_ALLOC_MEM_FLAGS_GTT,
            0u);
  ASSERT_EQ(
      amdf_gpu_umd_user_queue_map(queue_, nullptr, &mapping_, &mapping_result_),
      AMDF_STATUS_OK);
  EXPECT_EQ(mapping_result_.host_publish_executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(mapping_result_.host_publish_fence, AMDF_HOST_CACHE_FENCE_NONE);
  EXPECT_EQ(native_state_.host_publication.call_count, 0u);
}

TEST_P(KfdQueuePlacementTest, FailedInitialReleaseCannotActivateNativeQueue) {
  SelectLocalTopology();
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  native_state_.host_publication.status = failure;
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL;
  auto* const sentinel =
      reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
  auto* queue = sentinel;
  amdf_gpu_umd_user_queue_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original_result = result;
  EXPECT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue, &result),
            failure);
  EXPECT_EQ(queue, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  EXPECT_EQ(native_state_.host_publication.call_count, 1u);
  EXPECT_EQ(native_state_.queue_create_count, 0);
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_P(KfdQueuePlacementTest, FailedReleaseIsTerminalWithoutProvingRetirement) {
  SelectLocalTopology();
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  amdf_atomic_uint64_store_release(WriteIndex(), 8);
  amdf_atomic_uint64_store_release(ReadIndex(), 4);
  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_flush(mapping_), AMDF_STATUS_OK);
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);
  native_state_.host_publication.status = failure;
  EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_flush(mapping_), failure);
  EXPECT_EQ(native_state_.host_publication.call_count, 3u);
  native_state_.host_publication.status = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_gpu_umd_user_queue_mapping_flush(mapping_), failure);
  EXPECT_EQ(native_state_.host_publication.call_count, 3u);

  amdf_user_queue_status_t status = {};
  ASSERT_EQ(amdf_gpu_umd_user_queue_query_status(queue_, &status),
            AMDF_STATUS_OK);
  EXPECT_EQ(status.state, AMDF_QUEUE_STATE_FAILED);
  EXPECT_EQ(status.terminal_status, failure);
  EXPECT_EQ(status.producer_index, 8u);
  EXPECT_EQ(status.consumed_index, 4u);
  const amdf_wait_deadline_t deadline = {UINT64_MAX, UINT64_MAX};
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            failure);
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 9, &deadline),
            failure);
  EXPECT_EQ(amdf_gpu_umd_user_queue_destroy(queue_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(native_state_.queue_destroy_count, 0);
  EXPECT_EQ(native_state_.buffer_destroy_count, 0);

  // Only independent consumption permits ordinary teardown. The flush error
  // remains terminal even after the queue's earlier publication retires.
  amdf_atomic_uint64_store_release(ReadIndex(), 8);
  EXPECT_EQ(amdf_gpu_umd_user_queue_wait_consumed(queue_, 8, &deadline),
            failure);
  ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_destroy(mapping_), AMDF_STATUS_OK);
  mapping_ = nullptr;
  ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
}

TEST_P(KfdQueuePlacementTest, HostLocalRingNeedsNoNativeRelease) {
  SelectLocalTopology();
  ASSERT_NO_FATAL_FAILURE(CreateQueue(GetParam()));
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  EXPECT_EQ(queue_result_.placement, AMDF_QUEUE_PLACEMENT_HOST_LOCAL);
  EXPECT_EQ(mapping_result_.host_publish_executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(mapping_result_.host_publish_fence, AMDF_HOST_CACHE_FENCE_NONE);
  EXPECT_EQ(native_state_.host_publication.call_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(Formats, KfdQueuePlacementTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

}  // namespace
