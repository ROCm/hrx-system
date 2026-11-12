// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/transfer_memory.h"

#include <array>

#include "iree/hal/driver/amd/fabric/fabric.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only passive native discovery is substituted. The production selector must
// admit the exact construction request and both reader visibility boundaries.
class TransferMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (uint32_t i = 0; i < profiles_.size(); ++i) {
      profiles_[i] = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(amdf_memory_profile_t),
          .ordinal = i,
          .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
          .roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                   AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          .supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      };
      profiles_[i].host_mapping.supported_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
      profiles_[i].host_mapping.properties =
          AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
          AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
    }
    for (uint32_t direction = 0; direction < pairs_.size(); ++direction) {
      auto& pair = pairs_[direction];
      pair.flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
      auto& host = direction == 0 ? pair.release : pair.acquire;
      auto& queue = direction == 0 ? pair.acquire : pair.release;
      host.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
      queue = {
          .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
          .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
          .operation = direction == 0 ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                                      : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
      };
    }
    families_[2].cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                                    AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM;
    families_[2].cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL;
    fabric_.system_scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    context_.fabric = &fabric_;
    context_.device = reinterpret_cast<amdf_device_t*>(&context_);
    context_.api = &api_;
    context_.families = families_.data();
    api_.memory_scope_query_info = [](amdf_memory_scope_t* scope,
                                      amdf_memory_scope_info_t* info) {
      auto* self = reinterpret_cast<TransferMemoryTest*>(scope);
      info->memory_profile_count = self->profile_count_;
      return self->scope_status_;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t* scope, uint32_t ordinal, uint32_t access_count,
           const amdf_memory_device_access_t* accesses,
           amdf_memory_profile_t* profile,
           amdf_memory_access_capabilities_t* capabilities) {
          auto* self = reinterpret_cast<TransferMemoryTest*>(scope);
          EXPECT_EQ(access_count, 1u);
          self->CheckAccess(accesses[0]);
          *profile = self->profiles_[ordinal];
          // Device cache policy is an attachment fact, not a backing flag.
          capabilities->guaranteed_flags = self->attachment_flags_;
          return self->profile_status_[ordinal];
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t* scope,
           const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t* info) {
          auto* self = reinterpret_cast<TransferMemoryTest*>(scope);
          EXPECT_EQ(query->access_count, 1u);
          self->CheckAccess(query->accesses[0]);
          EXPECT_EQ(query->required_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE);
          const bool upload =
              query->producer.kind == AMDF_MEMORY_SITE_KIND_HOST;
          const auto& host = upload ? query->producer : query->consumer;
          const auto& device = upload ? query->consumer : query->producer;
          EXPECT_EQ(host.kind, AMDF_MEMORY_SITE_KIND_HOST);
          EXPECT_EQ(host.value.host_access, upload ? AMDF_MEMORY_MAP_FLAG_WRITE
                                                   : AMDF_MEMORY_MAP_FLAG_READ);
          EXPECT_EQ(device.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
          EXPECT_EQ(device.value.device.access_ordinal, 0u);
          EXPECT_EQ(device.value.device.queue_family_ordinal, 2u);
          const uint32_t direction = upload ? 0 : 1;
          ++self->pair_queries_[direction];
          *info = self->pairs_[direction];
          return self->pair_status_[direction];
        };
  }

  void CheckAccess(const amdf_memory_device_access_t& access) {
    EXPECT_EQ(access.device, context_.device);
    EXPECT_EQ(access.requirements.access,
              AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
    EXPECT_EQ(access.requirements.flags,
              AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT);
    EXPECT_EQ(access.requirements.address_kinds,
              UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU);
  }

  iree_status_t Initialize() {
    return iree_hal_amd_gpu_transfer_memory_initialize(&context_, 2, &memory_);
  }

  // Real selector inputs and output, kept stable for borrowed policy pointers.
  iree_hal_amd_gpu_context_t context_ = {};
  // System scope supplies the fake native discovery dependency.
  iree_hal_amd_fabric_t fabric_ = {};
  // Discovery entry points; no allocation or host maintenance API is installed.
  amdf_api_t api_ = {};
  // Only the exact third family supplies queue visibility operations.
  std::array<amdf_queue_family_info_t, 3> families_ = {};
  // Backing profiles, independent of device attachment capabilities.
  std::array<amdf_memory_profile_t, 2> profiles_ = {};
  // Number of profiles exposed in the selected scope.
  uint32_t profile_count_ = 1;
  // Discovery failure before profile enumeration.
  amdf_status_t scope_status_ = AMDF_STATUS_OK;
  // Independent device-admission results for each backing profile.
  std::array<amdf_status_t, 2> profile_status_ = {};
  // Attachment properties returned separately from backing profiles.
  amdf_memory_flags_t attachment_flags_ = AMDF_MEMORY_FLAG_HOST_COHERENT;
  // Upload and download visibility recipes from native discovery.
  std::array<amdf_memory_pair_info_t, 2> pairs_ = {};
  // Native discovery failures for each direction.
  std::array<amdf_status_t, 2> pair_status_ = {};
  // Observed upload and download query counts.
  std::array<uint32_t, 2> pair_queries_ = {};
  // Stable selected access descriptor and allocation options.
  iree_hal_amd_gpu_transfer_memory_t memory_ = {};
};

TEST_F(TransferMemoryTest, SelectsReadWritePayloadAndBothVisibilityDirections) {
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.scope, fabric_.system_scope);
  EXPECT_EQ(memory_.options.accesses, &memory_.access);
  EXPECT_EQ(memory_.options.access_count, 1u);
  EXPECT_EQ(memory_.options.required_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE);
  EXPECT_EQ(memory_.options.host_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
  CheckAccess(memory_.access);
  EXPECT_EQ(pair_queries_, (std::array<uint32_t, 2>{1, 1}));
}

TEST_F(TransferMemoryTest, AcceptsCoherentQueueWithoutCacheOperations) {
  pairs_[0].acquire = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE};
  pairs_[1].release = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE};
  families_[2].cache_operations = 0;
  families_[2].cache_transition_kinds = 0;
  IREE_ASSERT_OK(Initialize());
}

TEST_F(TransferMemoryTest, DoesNotInventPositiveDeviceCacheGuarantee) {
  attachment_flags_ |= AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.profile.supported_flags,
            AMDF_MEMORY_FLAG_HOST_VISIBLE);
}

TEST_F(TransferMemoryTest, SkipsProfilesMissingActualHostContract) {
  profile_count_ = 2;
  const auto valid = profiles_[0];
  for (uint32_t missing = 0; missing < 7; ++missing) {
    SCOPED_TRACE(missing);
    profiles_[0] = valid;
    switch (missing) {
      case 0:
        profiles_[0].roles &= ~AMDF_MEMORY_PROFILE_ROLE_CREATE;
        break;
      case 1:
        profiles_[0].roles &= ~AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
        break;
      case 2:
        profiles_[0].supported_flags = 0;
        break;
      case 3:
        profiles_[0].host_mapping.supported_access &=
            ~AMDF_MEMORY_MAP_FLAG_READ;
        break;
      case 4:
        profiles_[0].host_mapping.supported_access &=
            ~AMDF_MEMORY_MAP_FLAG_WRITE;
        break;
      case 5:
        profiles_[0].host_mapping.properties =
            AMDF_HOST_MEMORY_PROPERTY_READ_CACHED;
        break;
      case 6:
        profiles_[0].availability =
            amdf_make_api_status(AMDF_STATUS_CODE_PERMISSION_DENIED);
        break;
    }
    IREE_ASSERT_OK(Initialize());
    EXPECT_EQ(memory_.options.profile.ordinal, 1u);
  }
}

TEST_F(TransferMemoryTest, SkipsUnsupportedDeviceAttachment) {
  profile_count_ = 2;
  profile_status_[0] = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.profile.ordinal, 1u);
}

TEST_F(TransferMemoryTest, RejectsUnavailableVisibilityInEitherDirection) {
  const auto valid = pairs_;
  for (uint32_t direction = 0; direction < 2; ++direction) {
    for (uint32_t missing = 0; missing < 7; ++missing) {
      SCOPED_TRACE(direction);
      SCOPED_TRACE(missing);
      pairs_ = valid;
      auto& pair = pairs_[direction];
      auto& host = direction == 0 ? pair.release : pair.acquire;
      auto& queue = direction == 0 ? pair.acquire : pair.release;
      switch (missing) {
        case 0:
          pair.flags = 0;
          break;
        case 1:
          host = {
              .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
              .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
          };
          break;
        case 2:
          host.kind = AMDF_CACHE_TRANSITION_KIND_UNKNOWN;
          break;
        case 3:
          queue.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
          break;
        case 4:
          queue.executor = AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM;
          break;
        case 5:
          queue.operation = AMDF_CACHE_OPERATION_NONE;
          break;
        case 6:
          queue.kind = AMDF_CACHE_TRANSITION_KIND_UNKNOWN;
          break;
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
    }
  }
}

TEST_F(TransferMemoryTest, RequiresExactQueueOperations) {
  for (auto operation : {AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
                         AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM}) {
    families_[2].cache_operations = operation;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
  }
  families_[2].cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
  families_[2].cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_RANGE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
}

TEST_F(TransferMemoryTest, PreservesDiscoveryErrors) {
  scope_status_ = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  scope_status_ = AMDF_STATUS_OK;
  profile_status_[0] =
      amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  profile_status_[0] = AMDF_STATUS_OK;
  for (uint32_t direction = 0; direction < 2; ++direction) {
    pair_status_ = {};
    pair_status_[direction] =
        amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
    pair_status_[direction] =
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
  }
}

}  // namespace
