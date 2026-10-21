// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/tsan.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

TEST(TsanTest, DefaultArenaPreservesShaderAbi) {
  iree_hal_amd_gpu_tsan_options_t options;
  iree_hal_amd_gpu_tsan_options_initialize(&options);
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.record_length, 32u);
  EXPECT_EQ(config.abi_version, 0u);
  EXPECT_EQ(config.flags, 3u);
  EXPECT_EQ(config.reserved0, 0u);
  EXPECT_EQ(config.shadow_size, UINT64_C(512) * 1024 * 1024);
  EXPECT_EQ(config.shadow_base, 0u);
}

TEST(TsanTest, ExplicitByteBudgetIsIndependentOfWorkgroupGeometry) {
  const iree_hal_amd_gpu_tsan_options_t options = {.shadow_size = 2176};
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.shadow_size, 2176u);
}

TEST(TsanTest, ArenaMustFitCompilerByteOffsets) {
  iree_hal_amd_gpu_tsan_options_t options = {};
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.record_length, 0u);
  options.shadow_size = UINT64_C(1) << 32;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.shadow_size, UINT64_C(1) << 32);
  for (uint64_t size : {(UINT64_C(1) << 32) + 1, UINT64_MAX}) {
    options.shadow_size = size;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
    EXPECT_EQ(config.record_length, 0u);
  }
}

// The production initializer selects, allocates, zeros, and publishes shadow
// storage. Only its native memory dependency is substituted here.
class TsanPlacementTest : public ::testing::TestWithParam<bool> {
 protected:
  struct Scope {
    // Availability of the native CPU-to-device publication path.
    amdf_status_t publication_status = AMDF_STATUS_OK;
    // Backing owned from native creation until destruction.
    std::vector<uint8_t> bytes;
    // Total physical allocation count.
    uint32_t allocation_count = 0;
  };

  void SetUp() override {
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* out) {
      out->memory_profile_count = 1;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t*, uint32_t, uint32_t,
           const amdf_memory_device_access_t*, amdf_memory_profile_t* out,
           amdf_memory_access_capabilities_t*) {
          out->roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                       AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
          out->supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
          out->host_mapping.supported_access =
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
          out->allocation.byte_length_granularity = 64;
          out->allocation.minimum_alignment = 64;
          out->allocation.maximum_alignment = 64;
          out->allocation.maximum_byte_length = UINT64_MAX;
          return AMDF_STATUS_OK;
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t* scope,
           const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t*) {
          EXPECT_EQ(query->producer.kind, AMDF_MEMORY_SITE_KIND_HOST);
          EXPECT_EQ(query->producer.value.host_access,
                    AMDF_MEMORY_MAP_FLAG_WRITE);
          EXPECT_EQ(query->consumer.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
          EXPECT_EQ(query->consumer.value.device.access_ordinal, 0u);
          EXPECT_EQ(query->consumer.value.device.queue_family_ordinal, 5u);
          return reinterpret_cast<Scope*>(scope)->publication_status;
        };
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out, amdf_memory_release_info_t*) {
      auto* storage = reinterpret_cast<Scope*>(scope);
      EXPECT_EQ(storage->publication_status, AMDF_STATUS_OK);
      storage->bytes.resize(info->byte_length, 0xcc);
      ++storage->allocation_count;
      *out = reinterpret_cast<amdf_memory_t*>(scope);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_info = [](amdf_memory_t* memory,
                                amdf_memory_info_t* out) {
      out->byte_length = reinterpret_cast<Scope*>(memory)->bytes.size();
      out->alignment = 64;
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* memory, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** out) {
      *out = reinterpret_cast<amdf_host_mapping_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* mapping,
                                      amdf_host_mapping_info_t* out) {
      auto* storage = reinterpret_cast<Scope*>(mapping);
      out->pointer = storage->bytes.data();
      out->byte_length = storage->bytes.size();
      out->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t*, uint32_t,
                                   amdf_memory_address_kind_t, uint64_t* out) {
      *out = 0x10000;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t* mapping,
                                         amdf_host_cache_operation_t,
                                         uint64_t offset, uint64_t length) {
      const auto& bytes = reinterpret_cast<Scope*>(mapping)->bytes;
      EXPECT_EQ(offset, 0u);
      EXPECT_EQ(length, bytes.size());
      EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), 0));
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* memory,
                             amdf_memory_release_info_t*) {
      reinterpret_cast<Scope*>(memory)->bytes.clear();
      return AMDF_STATUS_OK;
    };
    fabric_.system_scope = reinterpret_cast<amdf_memory_scope_t*>(&system_);
    scopes_[0] = reinterpret_cast<amdf_memory_scope_t*>(&local_);
    context_.api = &api_;
    context_.fabric = &fabric_;
    context_.memory_scope_count = IREE_ARRAYSIZE(scopes_);
    context_.memory_scopes = scopes_;
    context_.event_sink = iree_hal_device_event_sink_stderr();
  }

  void TearDown() override {
    iree_hal_amd_gpu_tsan_deinitialize(&context_, &tsan_);
    EXPECT_TRUE(local_.bytes.empty());
    EXPECT_TRUE(system_.bytes.empty());
  }

  // Native memory API substituted at the production dependency boundary.
  amdf_api_t api_ = {};
  // Preferred local storage with an independently qualified host aperture.
  Scope local_;
  // Host storage supporting the same device access.
  Scope system_;
  // Native local scopes borrowed by the context.
  amdf_memory_scope_t* scopes_[1] = {};
  // Shared fabric supplying the system scope.
  iree_hal_amd_fabric_t fabric_ = {};
  // Production context selecting between native placements.
  iree_hal_amd_gpu_context_t context_ = {};
  // Production shadow owner, released even after partial construction.
  iree_hal_amd_gpu_tsan_t tsan_ = {};
};

TEST_P(TsanPlacementTest, QualifiesPublicationBeforeAllocatingShadow) {
  if (!GetParam()) {
    local_.publication_status =
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const iree_hal_amd_gpu_tsan_options_t options = {.shadow_size = 256};
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_tsan_initialize(&context_, 5, &options, &tsan_));
  EXPECT_EQ(local_.allocation_count, GetParam() ? 1u : 0u);
  EXPECT_EQ(system_.allocation_count, GetParam() ? 0u : 1u);
  EXPECT_EQ(tsan_.config.shadow_base, 0x10000u);
  EXPECT_EQ(tsan_.config.shadow_size, 256u);
}

TEST_P(TsanPlacementTest, FailedImagePreparationRollsBackOnlyNewMembership) {
  if (!GetParam()) {
    local_.publication_status =
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const iree_hal_amd_gpu_tsan_options_t options = {.shadow_size = 256};
  iree_hal_amd_gpu_tsan_registry_t registry;
  iree_hal_amd_gpu_tsan_registry_initialize(&context_, 5, 0, &options,
                                            iree_allocator_system(), &registry);
  struct Image {
    // Dependency whose native placement can fail independently of shadow setup.
    bool fail = false;
    // Current placement ownership, removed during either rollback or release.
    bool placed = false;
    // Embedded registration follows the same ownership as a native executable.
    iree_hal_amd_gpu_tsan_client_t client = {};
    Image() {
      client.user_data = this;
      client.prepare = [](void* user_data, uint32_t ordinal,
                          const iree_hal_amd_gpu_tsan_config_t* config) {
        auto& image = *static_cast<Image*>(user_data);
        EXPECT_EQ(ordinal, 3u);
        EXPECT_EQ(config->shadow_base, 0x10000u);
        image.placed = true;
        return image.fail
                   ? iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED)
                   : iree_ok_status();
      };
      client.release = [](void* user_data, uint32_t ordinal) {
        if (ordinal == 3) {
          static_cast<Image*>(user_data)->placed = false;
        }
      };
    }
  } first, second, rejected;
  IREE_EXPECT_OK(
      iree_hal_amd_gpu_tsan_registry_register(&registry, &first.client));
  IREE_EXPECT_OK(
      iree_hal_amd_gpu_tsan_registry_register(&registry, &second.client));
  first.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_gpu_tsan_registry_add_queue(&registry, 3));
  EXPECT_FALSE(first.placed);
  EXPECT_FALSE(second.placed);
  EXPECT_EQ(registry.queues[3], nullptr);
  EXPECT_TRUE(local_.bytes.empty());
  EXPECT_TRUE(system_.bytes.empty());
  first.fail = false;
  IREE_EXPECT_OK(iree_hal_amd_gpu_tsan_registry_add_queue(&registry, 3));
  EXPECT_TRUE(first.placed);
  EXPECT_TRUE(second.placed);
  rejected.fail = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amd_gpu_tsan_registry_register(&registry, &rejected.client));
  EXPECT_EQ(rejected.client.registry, nullptr);
  EXPECT_FALSE(rejected.placed);
  EXPECT_TRUE(first.placed);
  EXPECT_TRUE(second.placed);
  iree_hal_amd_gpu_tsan_registry_remove_queue(&registry, 3);
  EXPECT_FALSE(first.placed);
  EXPECT_FALSE(second.placed);
  iree_hal_amd_gpu_tsan_registry_unregister(&first.client);
  iree_hal_amd_gpu_tsan_registry_unregister(&second.client);
  iree_hal_amd_gpu_tsan_registry_deinitialize(&registry);
}

INSTANTIATE_TEST_SUITE_P(LocalPublication, TsanPlacementTest,
                         ::testing::Bool());

}  // namespace
}  // namespace iree::hal::amd::gpu
