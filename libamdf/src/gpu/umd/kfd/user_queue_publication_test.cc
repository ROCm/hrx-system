// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>

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
    device_.topology.memory_features = amdf_gpu_kfd_topology_memory_features(
        &device_.topology, device_.topology.integrated);
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
  EXPECT_EQ((queue_result_.capabilities &
             AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION) != 0,
            GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  const auto ring_flags = native_state_.buffers[0].create_info.native_flags;
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0u);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0u);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED, 0u);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0u);
  EXPECT_EQ(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0u);
  EXPECT_NE(native_state_.buffers[1].create_info.native_flags &
                KFD_IOC_ALLOC_MEM_FLAGS_GTT,
            0u);
  ASSERT_EQ(
      amdf_gpu_umd_user_queue_map(queue_, nullptr, &mapping_, &mapping_result_),
      AMDF_STATUS_OK);
  const bool header_commit = GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  EXPECT_EQ(mapping_result_.host_publish_executor,
            header_commit ? AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT
                          : AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  EXPECT_EQ(mapping_result_.host_publish_fence,
            header_commit ? AMDF_HOST_CACHE_FENCE_X86_SFENCE
                          : AMDF_HOST_CACHE_FENCE_NONE);
  EXPECT_EQ(native_state_.host_publication.calls_before_activation, 1u);
  if (!header_commit) {
    ASSERT_EQ(amdf_gpu_umd_user_queue_mapping_flush(mapping_), AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_state_.host_publication.call_count, header_commit ? 1u : 2u);
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
  SelectLocalTopology();
  device_.topology.integrated = true;
  device_.topology.gtt_byte_length = UINT64_C(2) << 30;
  device_.topology.memory_features = amdf_gpu_kfd_topology_memory_features(
      &device_.topology, device_.topology.integrated);
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.placement, AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL);
  EXPECT_EQ(queue_result_.capabilities &
                AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION,
            0u);
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

class KfdQueueNativePublicationTest : public KfdQueuePlacementTest {};

TEST_P(KfdQueueNativePublicationTest,
       FailedReleaseIsTerminalWithoutProvingRetirement) {
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
  auto create = MakeCreateInfo(GetParam());
  create.placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  EXPECT_EQ(queue_result_.placement, AMDF_QUEUE_PLACEMENT_HOST_LOCAL);
  EXPECT_EQ(queue_result_.capabilities &
                AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION,
            0u);
  EXPECT_EQ(mapping_result_.host_publish_executor,
            AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(mapping_result_.host_publish_fence, AMDF_HOST_CACHE_FENCE_NONE);
  EXPECT_EQ(native_state_.host_publication.call_count, 0u);
}

TEST_P(KfdQueuePlacementTest,
       NonTemporalPayloadPublicationRequiresQualifiedPlacement) {
  SelectLocalTopology();
  for (const auto placement :
       {AMDF_QUEUE_PLACEMENT_DEFAULT, AMDF_QUEUE_PLACEMENT_HOST_LOCAL,
        AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL}) {
    SCOPED_TRACE(placement);
    auto create = MakeCreateInfo(GetParam());
    create.placement = placement;
    create.required_capabilities |=
        AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION;
    if (GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
        placement != AMDF_QUEUE_PLACEMENT_HOST_LOCAL) {
      ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                               &queue_result_),
                AMDF_STATUS_OK);
      EXPECT_NE(queue_result_.capabilities &
                    AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION,
                0u);
      ASSERT_EQ(amdf_gpu_umd_user_queue_destroy(queue_), AMDF_STATUS_OK);
      queue_ = nullptr;
      EXPECT_EQ(native_state_.LiveBufferCount(), 0u);
      native_state_.Reset();
    } else {
      auto* const sentinel =
          reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
      auto* queue = sentinel;
      amdf_gpu_umd_user_queue_result_t result;
      std::memset(&result, 0xA5, sizeof(result));
      const auto original_result = result;
      EXPECT_EQ(
          amdf_gpu_umd_user_queue_create(&device_, &create, &queue, &result),
          amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
      EXPECT_EQ(queue, sentinel);
      EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
      EXPECT_EQ(native_state_.creation_operation_count, 0);
      EXPECT_EQ(native_state_.buffer_create_count, 0);
      EXPECT_EQ(native_state_.queue_create_count, 0);
    }
  }
}

enum class AqlMemoryRoute {
  kVisibleLocal,
  kPartialBar,
  kUnified,
  kNoLocalProfile,
  kNoHostRelease,
  kVirtualFunction,
};

class KfdAqlPlacementTest
    : public KfdUserQueueTest,
      public ::testing::WithParamInterface<
          std::tuple<uint32_t, AqlMemoryRoute, amdf_queue_placement_t>> {};

TEST_P(KfdAqlPlacementTest, ResolvesPlacementBeforeAllocationAndPublication) {
  const auto [target, route, requested] = GetParam();
  const uint32_t major = target / 10000;
  const uint32_t minor = target / 100 % 100;
  const uint32_t stepping = target % 100;
  if (major == 9) {
    SelectAqlTopology();
  }
  device_.topology.properties.gfx_ip = {major, minor, stepping};
  device_.topology.gc_ip = {major, minor, stepping, true};
  if (target == 120500) {
    device_.topology.gc_ip = {12, 1, 0, true};
  }
  device_.topology.integrated = route == AqlMemoryRoute::kUnified;
  device_.topology.virtual_function = route == AqlMemoryRoute::kVirtualFunction;
  device_.topology.vram.total_byte_length = UINT64_C(1) << 30;
  device_.topology.vram.visible_byte_length =
      UINT64_C(1) << (route == AqlMemoryRoute::kPartialBar ? 28 : 30);
  device_.topology.gtt_byte_length = UINT64_C(2) << 30;
  device_.topology.memory_features =
      route == AqlMemoryRoute::kNoLocalProfile
          ? 0
          : amdf_gpu_kfd_topology_memory_features(&device_.topology,
                                                  device_.topology.integrated);
  device_.hdp = route == AqlMemoryRoute::kNoHostRelease
                    ? nullptr
                    : reinterpret_cast<amdf_gpu_kfd_hdp_t*>(uintptr_t{1});
  auto create = MakeCreateInfo(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL);
  create.placement = requested;
  const bool supported = requested != AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL ||
                         (route != AqlMemoryRoute::kPartialBar &&
                          route != AqlMemoryRoute::kNoHostRelease &&
                          route != AqlMemoryRoute::kVirtualFunction);
  if (!supported) {
    auto* const sentinel =
        reinterpret_cast<amdf_gpu_umd_user_queue_t*>(uintptr_t{1});
    auto* queue = sentinel;
    amdf_gpu_umd_user_queue_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original_result = result;
    EXPECT_EQ(
        amdf_gpu_umd_user_queue_create(&device_, &create, &queue, &result),
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(queue, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
    EXPECT_EQ(native_state_.creation_operation_count, 0);
    return;
  }
  const bool local = requested == AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL ||
                     (requested == AMDF_QUEUE_PLACEMENT_DEFAULT && major == 9 &&
                      route == AqlMemoryRoute::kVisibleLocal);
  const bool local_backing = local && route != AqlMemoryRoute::kUnified;
  const bool local_payload =
      local_backing && route == AqlMemoryRoute::kVisibleLocal;
  ASSERT_EQ(amdf_gpu_umd_user_queue_create(&device_, &create, &queue_,
                                           &queue_result_),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_result_.placement, local ? AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL
                                           : AMDF_QUEUE_PLACEMENT_HOST_LOCAL);
  EXPECT_EQ((queue_result_.capabilities &
             AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION) != 0,
            local_payload);
  const auto ring_flags = native_state_.buffers[0].create_info.native_flags;
  EXPECT_EQ((ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) != 0, local_backing);
  EXPECT_EQ((ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC) != 0, local_backing);
  EXPECT_EQ((ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED) != 0,
            local_backing);
  EXPECT_EQ((ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) != 0, !local_backing);
  EXPECT_NE(ring_flags & KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0u);
  EXPECT_NE(native_state_.buffers[1].create_info.native_flags &
                KFD_IOC_ALLOC_MEM_FLAGS_GTT,
            0u);
  EXPECT_EQ(native_state_.host_publication.calls_before_activation,
            local_backing ? 1u : 0u);
  ASSERT_NO_FATAL_FAILURE(MapQueue());
  EXPECT_EQ(mapping_result_.host_publish_executor,
            local_backing ? AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT
                          : AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  EXPECT_EQ(mapping_result_.host_publish_fence,
            local_backing ? AMDF_HOST_CACHE_FENCE_X86_SFENCE
                          : AMDF_HOST_CACHE_FENCE_NONE);
}

INSTANTIATE_TEST_SUITE_P(
    MemoryContracts, KfdAqlPlacementTest,
    ::testing::Combine(::testing::Values(90010u, 90400u, 90401u, 90402u, 90500u,
                                         110000u, 110001u, 110002u, 110003u,
                                         110500u, 110501u, 120000u, 120001u,
                                         120500u),
                       ::testing::Values(AqlMemoryRoute::kVisibleLocal,
                                         AqlMemoryRoute::kPartialBar,
                                         AqlMemoryRoute::kUnified,
                                         AqlMemoryRoute::kNoLocalProfile,
                                         AqlMemoryRoute::kNoHostRelease,
                                         AqlMemoryRoute::kVirtualFunction),
                       ::testing::Values(AMDF_QUEUE_PLACEMENT_DEFAULT,
                                         AMDF_QUEUE_PLACEMENT_HOST_LOCAL,
                                         AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL)));

INSTANTIATE_TEST_SUITE_P(Formats, KfdQueuePlacementTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

INSTANTIATE_TEST_SUITE_P(Formats, KfdQueueNativePublicationTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA));

}  // namespace
