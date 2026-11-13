// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/input_memory.h"

#include <array>

#include "iree/hal/driver/amd/fabric/fabric.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Substitute only native discovery. No allocator or maintenance executor is
// installed: selection must qualify the real consuming family's existing edge.
class InputMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fabric_.system_scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    context_.fabric = &fabric_;
    context_.api = &api_;
    context_.families = families_.data();
    families_[1].cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
    families_[1].cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL;
    for (uint32_t i = 0; i < profiles_.size(); ++i) {
      profiles_[i] = {
          .ordinal = i,
          .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
          .roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                   AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          .supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      };
      profiles_[i].host_mapping.supported_access = AMDF_MEMORY_MAP_FLAG_WRITE;
      profiles_[i].host_mapping.properties =
          AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
    }
    pair_.flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    pair_.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    pair_.acquire = {
        .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
        .operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
    };
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* info) {
      info->memory_profile_count = 2;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t* scope, uint32_t ordinal, uint32_t count,
           const amdf_memory_device_access_t* access,
           amdf_memory_profile_t* profile,
           amdf_memory_access_capabilities_t* capabilities) {
          auto* self = reinterpret_cast<InputMemoryTest*>(scope);
          EXPECT_EQ(count, 1u);
          EXPECT_EQ(access->requirements.access, self->device_access_);
          EXPECT_EQ(access->requirements.flags, self->access_flags_);
          *profile = self->profiles_[ordinal];
          capabilities->guaranteed_flags = self->flags_[ordinal];
          return AMDF_STATUS_OK;
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t* scope,
           const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t* pair) {
          auto* self = reinterpret_cast<InputMemoryTest*>(scope);
          EXPECT_EQ(query->producer.kind, AMDF_MEMORY_SITE_KIND_HOST);
          EXPECT_EQ(query->producer.value.host_access,
                    AMDF_MEMORY_MAP_FLAG_WRITE);
          EXPECT_EQ(query->consumer.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
          EXPECT_EQ(query->consumer.value.device.queue_family_ordinal, 1u);
          *pair = self->pair_;
          return self->pair_status_;
        };
  }

  iree_status_t Initialize() {
    return iree_hal_amd_gpu_input_memory_initialize(&context_, 1,
                                                    device_access_, &memory_);
  }

  // Real selector input, with native discovery as its only substituted service.
  iree_hal_amd_gpu_context_t context_ = {};
  // System scope whose identity carries the discovery fixture.
  iree_hal_amd_fabric_t fabric_ = {};
  // Passive native operations; no allocation API is installed.
  amdf_api_t api_ = {};
  // The exact second family is the consuming queue.
  std::array<amdf_queue_family_info_t, 2> families_ = {};
  // Candidate backing profiles with independent host mapping contracts.
  std::array<amdf_memory_profile_t, 2> profiles_ = {};
  // Device attachment cache facts, independent of the backing profile.
  std::array<amdf_memory_flags_t, 2> flags_ = {};
  // Host-write to shader-read visibility recipe.
  amdf_memory_pair_info_t pair_ = {};
  // Native visibility discovery result.
  amdf_status_t pair_status_ = AMDF_STATUS_OK;
  // Device writes are needed only for ABI fields patched by indirect dispatch.
  amdf_memory_access_t device_access_ = AMDF_MEMORY_ACCESS_READ;
  // Attachment requirements for coherent SYSTEM or qualified LOCAL authoring.
  amdf_memory_flags_t access_flags_ =
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  // Selected persistent policy, published only after full qualification.
  iree_hal_amd_gpu_input_memory_t memory_ = {};
};

TEST_F(InputMemoryTest, SelectsReadOnlyInputsWithIndependentCachePolicy) {
  flags_[0] = AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.profile.ordinal, 1u);
  EXPECT_EQ(memory_.options.scope, fabric_.system_scope);
  EXPECT_EQ(memory_.options.host_access, AMDF_MEMORY_MAP_FLAG_WRITE);
  EXPECT_EQ(memory_.options.accesses, &memory_.access);
  EXPECT_EQ(memory_.access.requirements.access, AMDF_MEMORY_ACCESS_READ);
}

TEST_F(InputMemoryTest, SkipsNonWriteBackHostMapping) {
  profiles_[0].host_mapping.properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED;
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.profile.ordinal, 1u);
}

TEST_F(InputMemoryTest, WritableAbiKeepsTheSameHostPublicationContract) {
  device_access_ |= AMDF_MEMORY_ACCESS_WRITE;
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.access.requirements.access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(memory_.options.host_access, AMDF_MEMORY_MAP_FLAG_WRITE);
}

TEST_F(InputMemoryTest, RejectsHostReadOnlyProfiles) {
  profiles_[0].host_mapping.supported_access = AMDF_MEMORY_MAP_FLAG_READ;
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(memory_.options.profile.ordinal, 1u);
  profiles_[1].host_mapping.supported_access = AMDF_MEMORY_MAP_FLAG_READ;
  memory_ = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
  EXPECT_EQ(memory_.options.scope, nullptr);
}

TEST_F(InputMemoryTest, CoherentShadersNeedNoQueueOperation) {
  pair_.acquire = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE};
  families_[1].cache_operations = 0;
  families_[1].cache_transition_kinds = 0;
  IREE_ASSERT_OK(Initialize());
}

TEST_F(InputMemoryTest, RejectsUnimplementedVisibilityWithoutPublishingPolicy) {
  const auto valid = pair_;
  for (uint32_t missing = 0; missing < 7; ++missing) {
    SCOPED_TRACE(missing);
    pair_ = valid;
    switch (missing) {
      case 0:
        pair_.flags = 0;
        break;
      case 1:
        pair_.release = {
            .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
            .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
        };
        break;
      case 2:
        pair_.acquire.kind = AMDF_CACHE_TRANSITION_KIND_UNKNOWN;
        break;
      case 3:
        pair_.acquire.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
        break;
      case 4:
        pair_.acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_PROGRAM;
        break;
      case 5:
        pair_.acquire.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
        break;
      case 6:
        families_[1].cache_operations = 0;
        break;
    }
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
    EXPECT_EQ(memory_.options.scope, nullptr);
  }
}

TEST_F(InputMemoryTest, PreservesNativeDiscoveryFailure) {
  pair_status_ = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  pair_status_ = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
}

TEST_F(InputMemoryTest, LocalPublicationAcceptsMigrationStableHostProperties) {
  amdf_memory_scope_t* scope = reinterpret_cast<amdf_memory_scope_t*>(this);
  context_.memory_scopes = &scope;
  context_.memory_scope_count = 1;
  access_flags_ = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  device_access_ |= AMDF_MEMORY_ACCESS_WRITE;
  api_.memory_scope_query_pair_info = nullptr;
  for (auto& profile : profiles_) {
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.supported_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    // A migratable LOCAL allocation promises no fixed CPU cache policy.
    profile.host_mapping.properties = 0;
  }
  flags_[0] = AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
  IREE_ASSERT_OK(iree_hal_amd_gpu_input_memory_initialize_local(
      &context_, device_access_, &memory_));
  EXPECT_EQ(memory_.options.scope, scope);
  EXPECT_EQ(memory_.options.profile.ordinal, 1u);
  EXPECT_EQ(memory_.options.profile.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(memory_.options.accesses, &memory_.access);
  EXPECT_EQ(memory_.access.requirements.access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(memory_.options.host_access, AMDF_MEMORY_MAP_FLAG_WRITE);
}

TEST_F(InputMemoryTest, LocalPublicationRejectsSystemAndUnmappedProfiles) {
  amdf_memory_scope_t* scope = reinterpret_cast<amdf_memory_scope_t*>(this);
  context_.memory_scopes = &scope;
  context_.memory_scope_count = 1;
  access_flags_ = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  profiles_[0].supported_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  profiles_[1].memory_class = AMDF_MEMORY_CLASS_LOCAL;
  profiles_[1].supported_flags |= AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  profiles_[1].host_mapping.supported_access = AMDF_MEMORY_MAP_FLAG_READ;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_amd_gpu_input_memory_initialize_local(
                            &context_, device_access_, &memory_));
  EXPECT_EQ(memory_.options.scope, nullptr);
}

}  // namespace
