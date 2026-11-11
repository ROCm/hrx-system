// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/backing.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ArgumentMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.api = &api_;
    control_.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    control_.access_count = 1;
    control_.accesses = &control_access_;
    control_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    control_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    control_access_.device = reinterpret_cast<amdf_device_t*>(this);
    control_access_.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                          AMDF_MEMORY_ACCESS_WRITE |
                                          AMDF_MEMORY_ACCESS_EXECUTE;
    control_access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                         AMDF_MEMORY_FLAG_HOST_COHERENT |
                                         AMDF_MEMORY_FLAG_QUEUE_STORAGE;
    for (uint32_t i = 0; i < profiles_.size(); ++i) {
      profiles_[i] = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(amdf_memory_profile_t),
          .ordinal = i,
          .roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                   AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          .supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .host_mapping = {.properties = AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK},
      };
    }
    profiles_[0].guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    profiles_[1].roles = AMDF_MEMORY_PROFILE_ROLE_REGISTER;
    control_.profile = profiles_[0];
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* info) {
      info->memory_profile_count = 3;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t* scope, uint32_t ordinal, uint32_t count,
           const amdf_memory_device_access_t* accesses,
           amdf_memory_profile_t* profile, amdf_memory_access_capabilities_t*) {
          auto* self = reinterpret_cast<ArgumentMemoryTest*>(scope);
          EXPECT_EQ(count, 1u);
          EXPECT_EQ(accesses[0].device, self->control_access_.device);
          EXPECT_EQ(
              accesses[0].requirements.flags,
              AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT);
          EXPECT_EQ(accesses[0].requirements.access,
                    AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
          if (ordinal == self->fail_ordinal_) {
            return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
          }
          if (ordinal == self->unsupported_ordinal_) {
            return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
          }
          *profile = self->profiles_[ordinal];
          return AMDF_STATUS_OK;
        };
  }

  iree_status_t Select() {
    return iree_hal_amd_gpu_argument_memory_select(&context_, &control_,
                                                   &access_, &selected_);
  }

  amdf_api_t api_ = {};
  iree_hal_amd_gpu_context_t context_ = {};
  amdf_memory_device_access_t control_access_ = {}, access_ = {};
  iree_hal_amd_fabric_memory_options_t control_ = {}, selected_ = {};
  std::array<amdf_memory_profile_t, 3> profiles_;
  uint32_t fail_ordinal_ = UINT32_MAX;
  uint32_t unsupported_ordinal_ = UINT32_MAX;
};

TEST_F(ArgumentMemoryTest, SelectsCachedOwnedBackingWithoutChangingControl) {
  IREE_ASSERT_OK(Select());
  EXPECT_EQ(selected_.profile.ordinal, 2u);
  EXPECT_EQ(selected_.accesses, &access_);
  EXPECT_EQ(selected_.required_flags, control_.required_flags);
  EXPECT_EQ(selected_.host_access, control_.host_access);
  EXPECT_EQ(control_.profile.ordinal, 0u);
  EXPECT_NE(control_access_.requirements.flags & AMDF_MEMORY_FLAG_QUEUE_STORAGE,
            0u);
}

TEST_F(ArgumentMemoryTest, RetainsControlPolicyWhenCachedBackingIsUnavailable) {
  unsupported_ordinal_ = 2;
  IREE_ASSERT_OK(Select());
  EXPECT_EQ(selected_.profile.ordinal, control_.profile.ordinal);
  EXPECT_EQ(selected_.accesses, control_.accesses);
}

TEST_F(ArgumentMemoryTest, RejectsUnmappedOrWriteCombinedCandidates) {
  profiles_[2].host_mapping.properties = 0;
  IREE_ASSERT_OK(Select());
  EXPECT_EQ(selected_.profile.ordinal, control_.profile.ordinal);
  profiles_[2].host_mapping.properties = AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
  profiles_[2].supported_flags = 0;
  IREE_ASSERT_OK(Select());
  EXPECT_EQ(selected_.profile.ordinal, control_.profile.ordinal);
}

TEST_F(ArgumentMemoryTest, PropagatesDiscoveryFailureWithoutPublishingOutputs) {
  fail_ordinal_ = 1;
  selected_.profile.ordinal = 91;
  access_.requirements.flags = 17;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Select());
  EXPECT_EQ(selected_.profile.ordinal, 91u);
  EXPECT_EQ(access_.requirements.flags, 17u);
}

// Only native capability discovery and backing ownership are dependencies.
// Production construction owns admission, acquisition, address capture, and
// rollback. No host mapping or submission service is available to this owner.
class GridStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.api = &api_;
    options_.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    options_.profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(options_.profile),
        .ordinal = 7,
        .allocation =
            {
                .maximum_byte_length = 4096,
                .byte_length_granularity = 64,
                .minimum_alignment = 64,
            },
    };
    options_.access_count = 1;
    options_.accesses = &access_;
    options_.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t* scope, uint32_t ordinal, uint32_t access_count,
           const amdf_memory_device_access_t* accesses,
           amdf_memory_profile_t* out_profile,
           amdf_memory_access_capabilities_t* out_access) {
          auto* self = reinterpret_cast<GridStorageTest*>(scope);
          EXPECT_EQ(ordinal, 7u);
          EXPECT_EQ(access_count, 1u);
          EXPECT_EQ(accesses, &self->access_);
          EXPECT_EQ(self->allocations_, 0u);
          *out_profile = self->options_.profile;
          auto* program =
              static_cast<amdf_memory_program_atomics_t*>(out_access->next);
          EXPECT_EQ(program->type, AMDF_STRUCTURE_TYPE_MEMORY_PROGRAM_ATOMICS);
          program->operations.device_scope_32 = self->operations_;
          return AMDF_STATUS_OK;
        };
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      auto* self = reinterpret_cast<GridStorageTest*>(scope);
      EXPECT_EQ(info->memory_profile_ordinal, 7u);
      EXPECT_EQ(info->access_count, 1u);
      EXPECT_EQ(info->accesses, &self->access_);
      EXPECT_EQ(info->required_flags, AMDF_MEMORY_FLAG_DEVICE_LOCAL);
      EXPECT_EQ(info->byte_length, 64u);
      EXPECT_EQ(info->minimum_alignment, 64u);
      EXPECT_EQ(info->registered_host_pointer, nullptr);
      *out_release_info = {};
      if (self->fail_allocation_) {
        return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
      ++self->allocations_;
      *out_memory = reinterpret_cast<amdf_memory_t*>(self);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* memory, uint32_t ordinal,
                                   amdf_memory_address_kind_t kind,
                                   uint64_t* out_address) {
      auto* self = reinterpret_cast<GridStorageTest*>(memory);
      EXPECT_EQ(ordinal, 0u);
      EXPECT_EQ(kind, AMDF_MEMORY_ADDRESS_GPU);
      if (self->fail_address_) {
        return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
      *out_address = UINT64_C(0x123400000000);
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* memory,
                             amdf_memory_release_info_t* out_release_info) {
      auto* self = reinterpret_cast<GridStorageTest*>(memory);
      ++self->releases_;
      *out_release_info = {};
      return AMDF_STATUS_OK;
    };
  }

  void TearDown() override {
    iree_hal_amd_gpu_grid_storage_deinitialize(&context_, &storage_);
    EXPECT_EQ(releases_, allocations_);
  }

  iree_status_t Initialize() {
    return iree_hal_amd_gpu_grid_storage_initialize(&context_, &options_,
                                                    &storage_);
  }

  // Native dependency entries used by the production backing owner.
  amdf_api_t api_ = {};
  // Borrowed context, with no execution or host-mapping service.
  iree_hal_amd_gpu_context_t context_ = {};
  // Device-local resident policy supplied by queue construction.
  iree_hal_amd_fabric_memory_options_t options_ = {};
  // Single native consumer captured by the resident policy.
  amdf_memory_device_access_t access_ = {};
  // Production owner, empty after failed construction and final release.
  iree_hal_amd_gpu_grid_storage_t storage_ = {};
  // Native shader guarantees, independent of command-processor atomics.
  amdf_atomic_operations_t operations_ =
      AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_ADD;
  // Number of successfully acquired native backings.
  uint32_t allocations_ = 0;
  // Number of consumed native backings, including constructor rollback.
  uint32_t releases_ = 0;
  // Dependency rejects acquisition before returning native ownership.
  bool fail_allocation_ = false;
  // Dependency rejects address discovery after successful acquisition.
  bool fail_address_ = false;
};

TEST_F(GridStorageTest, RetainsOnlyDeviceBackingAndAddress) {
  static_assert(sizeof(void*) != 8 || sizeof(storage_) == 16);
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(storage_.memory, reinterpret_cast<amdf_memory_t*>(this));
  EXPECT_EQ(storage_.address, UINT64_C(0x123400000000));
  EXPECT_EQ(allocations_, 1u);
  EXPECT_EQ(releases_, 0u);
  iree_hal_amd_gpu_grid_storage_deinitialize(&context_, &storage_);
  EXPECT_EQ(storage_.memory, nullptr);
  EXPECT_EQ(storage_.address, 0u);
  EXPECT_EQ(releases_, 1u);
}

TEST_F(GridStorageTest, MissingShaderOperationsRejectBeforeAllocation) {
  for (auto operations : {AMDF_ATOMIC_OPERATION_LOAD, AMDF_ATOMIC_OPERATION_ADD,
                          AMDF_ATOMIC_OPERATION_STORE}) {
    operations_ = operations;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Initialize());
    EXPECT_EQ(allocations_, 0u);
    EXPECT_EQ(storage_.memory, nullptr);
    EXPECT_EQ(storage_.address, 0u);
  }
}

TEST_F(GridStorageTest, AllocationFailureLeavesNoOwner) {
  fail_allocation_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  EXPECT_EQ(storage_.memory, nullptr);
  EXPECT_EQ(storage_.address, 0u);
  EXPECT_EQ(allocations_, 0u);
}

TEST_F(GridStorageTest, AddressFailureReturnsUnpublishedBacking) {
  fail_address_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Initialize());
  EXPECT_EQ(storage_.memory, nullptr);
  EXPECT_EQ(storage_.address, 0u);
  EXPECT_EQ(allocations_, 1u);
  EXPECT_EQ(releases_, 1u);
}

}  // namespace
