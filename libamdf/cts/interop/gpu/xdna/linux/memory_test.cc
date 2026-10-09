// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "amdf/xdna.h"
#include "gtest/gtest.h"
#include "libamdf/cts/util/device_cache.h"
#include "libamdf/cts/util/provider.h"

namespace {

void ExpectMemoryPairEqual(const amdf_memory_pair_info_t& actual,
                           const amdf_memory_pair_info_t& expected) {
  EXPECT_EQ(actual.type, expected.type);
  EXPECT_EQ(actual.structure_size, expected.structure_size);
  EXPECT_EQ(actual.next, expected.next);
  EXPECT_EQ(actual.flags, expected.flags);
  EXPECT_EQ(actual.atomic_reach.scope_32, expected.atomic_reach.scope_32);
  EXPECT_EQ(actual.atomic_reach.scope_64, expected.atomic_reach.scope_64);
  EXPECT_EQ(actual.estimated_fixed_cost_nanoseconds,
            expected.estimated_fixed_cost_nanoseconds);
  for (auto member :
       {&amdf_memory_pair_info_t::release, &amdf_memory_pair_info_t::acquire}) {
    const auto& actual_transition = actual.*member;
    const auto& expected_transition = expected.*member;
    SCOPED_TRACE(member == &amdf_memory_pair_info_t::release ? "release"
                                                             : "acquire");
    EXPECT_EQ(actual_transition.kind, expected_transition.kind);
    EXPECT_EQ(actual_transition.executor, expected_transition.executor);
    EXPECT_EQ(actual_transition.operation, expected_transition.operation);
    EXPECT_EQ(actual_transition.host_operation,
              expected_transition.host_operation);
    EXPECT_EQ(actual_transition.host_instruction,
              expected_transition.host_instruction);
    EXPECT_EQ(actual_transition.host_fence_before,
              expected_transition.host_fence_before);
    EXPECT_EQ(actual_transition.host_fence_after,
              expected_transition.host_fence_after);
    EXPECT_EQ(actual_transition.range_granularity,
              expected_transition.range_granularity);
  }
}

const amdf_external_memory_support_t* FindDmaBufSupport(
    const amdf_memory_profile_t& profile,
    amdf_external_memory_support_flags_t required_flags) {
  for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
    const amdf_external_memory_support_t& support =
        profile.external_memory_support[i];
    if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
        (support.flags & required_flags) == required_flags) {
      return &support;
    }
  }
  return nullptr;
}

amdf_status_t FindDmaBufProfile(
    const amdf_api_t* api, amdf_memory_scope_t* scope, amdf_device_t* device,
    amdf_memory_profile_roles_t required_roles,
    amdf_memory_flags_t required_memory_flags,
    amdf_memory_access_requirements_t requirements,
    amdf_external_memory_support_flags_t required_external_flags,
    uint32_t* out_ordinal, amdf_memory_profile_t* out_profile) {
  for (uint32_t ordinal = 0; ordinal != UINT32_MAX; ++ordinal) {
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    const amdf_memory_device_access_t access = {device, requirements};
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities)};
    const amdf_status_t status = api->memory_scope_query_device_profile(
        scope, ordinal, 1, &access, &profile, &capabilities);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
      *out_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
      return AMDF_STATUS_OK;
    }
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (profile.memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
        (profile.roles & required_roles) == required_roles &&
        (required_memory_flags & ~profile.supported_flags) == 0 &&
        FindDmaBufSupport(profile, required_external_flags) != nullptr) {
      *out_ordinal = ordinal;
      *out_profile = profile;
      return AMDF_STATUS_OK;
    }
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
}

amdf_status_t FindQueueFamilyOrdinal(const amdf_api_t* api,
                                     amdf_endpoint_t* endpoint,
                                     amdf_queue_command_type_t command_type,
                                     uint32_t* out_ordinal) {
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info)};
  amdf_status_t status = api->endpoint_query_info(endpoint, &endpoint_info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
       ++ordinal) {
    amdf_queue_family_info_t family_info = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family_info)};
    status =
        api->endpoint_query_queue_family_info(endpoint, ordinal, &family_info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (family_info.command_type == command_type) {
      *out_ordinal = ordinal;
      return AMDF_STATUS_OK;
    }
  }
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

class GpuXdnaMemoryInteropTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    ASSERT_NE(api_, nullptr);

    const void* extension_api = nullptr;
    ASSERT_EQ(api_->query_extension(
                  AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                  AMDF_GPU_EXTENSION_VERSION_LATEST, &extension_api),
              AMDF_STATUS_OK);
    gpu_api_ = static_cast<const amdf_gpu_api_t*>(extension_api);
    ASSERT_NE(gpu_api_, nullptr);
    extension_api = nullptr;
    ASSERT_EQ(api_->query_extension(
                  AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                  AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension_api),
              AMDF_STATUS_OK);
    xdna_api_ = static_cast<const amdf_xdna_api_t*>(extension_api);
    ASSERT_NE(xdna_api_, nullptr);

    amdf_status_t status = GetCtsDeviceCache().GetInstance(&instance_);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "platform provider is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);

    uint32_t scope_count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  instance_, 0, nullptr, &scope_count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    std::vector<amdf_memory_scope_t*> scopes(scope_count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(
                  instance_, scope_count, scopes.data(), &scope_count),
              AMDF_STATUS_OK);
    for (amdf_memory_scope_t* scope : scopes) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info)};
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        system_scope_ = scope;
      }
    }
    ASSERT_NE(system_scope_, nullptr);

    uint32_t endpoint_count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &endpoint_count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> summaries(endpoint_count);
    if (endpoint_count != 0) {
      ASSERT_EQ(api_->endpoint_enumerate(instance_, endpoint_count,
                                         summaries.data(), &endpoint_count),
                AMDF_STATUS_OK);
    }
    for (const amdf_endpoint_summary_t& summary : summaries) {
      amdf_endpoint_t** endpoint = nullptr;
      if (summary.engine_kind == AMDF_ENGINE_KIND_GPU &&
          gpu_endpoint_ == nullptr) {
        endpoint = &gpu_endpoint_;
      } else if (summary.engine_kind == AMDF_ENGINE_KIND_XDNA &&
                 xdna_endpoint_ == nullptr) {
        endpoint = &xdna_endpoint_;
      }
      if (endpoint != nullptr) {
        ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, endpoint),
                  AMDF_STATUS_OK);
      }
    }
    if (gpu_endpoint_ == nullptr || xdna_endpoint_ == nullptr) {
      GTEST_SKIP() << "a qualified GPU and XDNA endpoint pair is required";
    }

    status = AcquireGpuDevice();
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native GPU activation is unavailable for this lifetime";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    amdf_gpu_device_info_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
        .structure_size = sizeof(capabilities)};
    ASSERT_EQ(gpu_api_->device_query_info(gpu_device_, &capabilities),
              AMDF_STATUS_OK);
    if ((capabilities.features & RequiredGpuFeatures()) !=
        RequiredGpuFeatures()) {
      GTEST_SKIP()
          << "source teardown requires reclaimable native VM acquisition";
    }

    status = GetCtsDeviceCache().GetXdnaDevice(xdna_endpoint_, &xdna_device_);
    if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
      GTEST_SKIP() << "XDNA ordinary-address-domain creation is unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    gpu_access_.device = gpu_device_;
    gpu_access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    gpu_access_.requirements.flags =
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    xdna_access_.device = xdna_device_;
    xdna_access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    xdna_access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  }

  virtual amdf_gpu_device_features_t RequiredGpuFeatures() const { return 0; }

  void ImportGpuSubrangeAndReleaseAllocation();
  void CheckSurvivingImport();

  void FindJointProfile(amdf_memory_profile_roles_t role,
                        amdf_memory_profile_t* out_profile,
                        amdf_memory_access_capabilities_t* capabilities) {
    const amdf_memory_device_access_t devices[] = {xdna_access_, gpu_access_};
    out_profile->ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0;; ++ordinal) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile)};
      for (uint32_t i = 0; i < 2; ++i) {
        capabilities[i] = {};
        capabilities[i].type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capabilities[i].structure_size = sizeof(capabilities[i]);
      }
      const amdf_status_t status = api_->memory_scope_query_device_profile(
          system_scope_, ordinal, 2, devices, &profile, capabilities);
      if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) {
        break;
      }
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & role) != 0 &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
        *out_profile = profile;
        break;
      }
    }
  }

  virtual amdf_status_t AcquireGpuDevice() {
    return GetCtsDeviceCache().GetGpuDevice(gpu_endpoint_, &gpu_device_);
  }

  void TearDown() override {
    if (api_ != nullptr) {
      api_->external_memory_release(&external_memory_);
    }
    if (xdna_mapping_ != nullptr) {
      EXPECT_EQ(
          api_->host_mapping_destroy(std::exchange(xdna_mapping_, nullptr)),
          AMDF_STATUS_OK);
    }
    if (gpu_mapping_ != nullptr) {
      EXPECT_EQ(
          api_->host_mapping_destroy(std::exchange(gpu_mapping_, nullptr)),
          AMDF_STATUS_OK);
    }
    if (xdna_memory_ != nullptr) {
      EXPECT_EQ(api_->memory_destroy(std::exchange(xdna_memory_, nullptr)),
                AMDF_STATUS_OK);
    }
    if (gpu_memory_ != nullptr) {
      EXPECT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
                AMDF_STATUS_OK);
    }
    if (additional_gpu_device_ != nullptr) {
      EXPECT_EQ(api_->device_destroy(additional_gpu_device_), AMDF_STATUS_OK);
    }
    if (!HasFailure() && caller_pages_.pointer != nullptr) {
      EXPECT_EQ(munmap(caller_pages_.pointer, caller_pages_.byte_length), 0);
    }
  }

  amdf_status_t Map(amdf_memory_t* memory, uint64_t byte_length,
                    amdf_host_mapping_t** out_mapping,
                    amdf_host_mapping_info_t* out_info) {
    amdf_memory_map_info_t map_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map_info),
        .byte_length = byte_length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE};
    amdf_status_t status = api_->memory_map(memory, &map_info, out_mapping);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    out_info->type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    out_info->structure_size = sizeof(*out_info);
    return api_->host_mapping_query_info(*out_mapping, out_info);
  }

  // Core table borrowed from the CTS provider.
  const amdf_api_t* api_ = nullptr;
  // GPU table borrowed from the CTS provider.
  const amdf_gpu_api_t* gpu_api_ = nullptr;
  // XDNA table borrowed from the CTS provider.
  const amdf_xdna_api_t* xdna_api_ = nullptr;
  // Shared instance defining the identity scope for both devices.
  amdf_instance_t* instance_ = nullptr;
  // Instance system scope shared by explicit construction and import.
  amdf_memory_scope_t* system_scope_ = nullptr;
  // Source GPU requirements and its explicit live owner.
  amdf_memory_device_access_t gpu_access_ = {};
  // Importing XDNA requirements and its explicit live owner.
  amdf_memory_device_access_t xdna_access_ = {};
  // Case-owned second GPU owner used to query a complete access set.
  amdf_device_t* additional_gpu_device_ = nullptr;
  // Shared GPU endpoint.
  amdf_endpoint_t* gpu_endpoint_ = nullptr;
  // Shared XDNA endpoint.
  amdf_endpoint_t* xdna_endpoint_ = nullptr;
  // Borrowed GPU owner, except in the explicit source-lifetime scenario.
  amdf_device_t* gpu_device_ = nullptr;
  // Shared XDNA device surviving each case's imported attachment.
  amdf_device_t* xdna_device_ = nullptr;
  // Case-owned GPU source backing.
  amdf_memory_t* gpu_memory_ = nullptr;
  // Caller-owned pages kept alive through release of their joint registration.
  struct {
    // Complete page-aligned reservation, or null when not used by this case.
    uint8_t* pointer = nullptr;
    // Complete reservation length in bytes, including unregistered guards.
    size_t byte_length = 0;
  } caller_pages_;
  // Case-owned imported attachment to the same physical backing.
  amdf_memory_t* xdna_memory_ = nullptr;
  // Case-owned host view of the source attachment.
  amdf_host_mapping_t* gpu_mapping_ = nullptr;
  // Case-owned host view of the imported attachment.
  amdf_host_mapping_t* xdna_mapping_ = nullptr;
  // Export transport owned until consumed or explicitly released.
  amdf_external_memory_t external_memory_ = {};
};

// This scenario specifically destroys the exporting native device while the
// imported attachment remains live. Only that source owner is case-local.
class GpuXdnaMemoryLifetimeTest : public GpuXdnaMemoryInteropTest {
 protected:
  amdf_gpu_device_features_t RequiredGpuFeatures() const override {
    return AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION;
  }
  amdf_status_t AcquireGpuDevice() override {
    amdf_gpu_device_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create_info)};
    return gpu_api_->device_create(gpu_endpoint_, &create_info, &gpu_device_);
  }

  void TearDown() override {
    GpuXdnaMemoryInteropTest::TearDown();
    if (gpu_device_ != nullptr) {
      ASSERT_EQ(api_->device_destroy(gpu_device_), AMDF_STATUS_OK);
      gpu_device_ = nullptr;
    }
  }
};

void GpuXdnaMemoryInteropTest::ImportGpuSubrangeAndReleaseAllocation() {
  const long system_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(system_page_size, 0);
  const uint64_t page_size = static_cast<uint64_t>(system_page_size);
  const uint64_t backing_byte_length = page_size * 3;

  constexpr amdf_memory_flags_t kGpuRequiredFlags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  uint32_t gpu_profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_profile_t gpu_profile = {};
  ASSERT_EQ(
      FindDmaBufProfile(api_, system_scope_, gpu_device_,
                        AMDF_MEMORY_PROFILE_ROLE_CREATE |
                            AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
                        kGpuRequiredFlags, gpu_access_.requirements,
                        AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET,
                        &gpu_profile_ordinal, &gpu_profile),
      AMDF_STATUS_OK);
  ASSERT_NE(gpu_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_external_memory_support_t* gpu_dma_buf_support = FindDmaBufSupport(
      gpu_profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                       AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
  ASSERT_NE(gpu_dma_buf_support, nullptr);
  ASSERT_NE(gpu_dma_buf_support->source_offset_alignment, 0u);
  ASSERT_EQ(page_size % gpu_dma_buf_support->source_offset_alignment, 0u);
  ASSERT_NE(gpu_dma_buf_support->byte_length_alignment, 0u);
  ASSERT_EQ(page_size % gpu_dma_buf_support->byte_length_alignment, 0u);

  constexpr amdf_memory_flags_t kXdnaRequiredFlags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE;
  uint32_t xdna_profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_profile_t xdna_profile = {};
  ASSERT_EQ(
      FindDmaBufProfile(
          api_, system_scope_, xdna_device_,
          AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          kXdnaRequiredFlags, xdna_access_.requirements,
          AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
          &xdna_profile_ordinal, &xdna_profile),
      AMDF_STATUS_OK);
  ASSERT_NE(xdna_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  amdf_memory_profile_pair_query_t pool_query = {};
  pool_query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
  pool_query.structure_size = sizeof(pool_query);
  pool_query.memory_profile_ordinal = xdna_profile_ordinal;
  pool_query.access_count = 1;
  pool_query.accesses = &xdna_access_;
  pool_query.required_flags = kXdnaRequiredFlags;
  pool_query.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
  for (uint32_t i = 0; i < gpu_profile.external_memory_support_count; ++i) {
    if (gpu_profile.external_memory_support[i].type ==
        pool_query.external_memory_type) {
      pool_query.external_memory_provenance =
          gpu_profile.external_memory_support[i].provenance;
    }
  }
  ASSERT_TRUE(amdf_external_memory_provenance_is_valid(
      &pool_query.external_memory_provenance));
  pool_query.producer.kind = AMDF_MEMORY_SITE_KIND_HOST;
  pool_query.producer.value.host_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  pool_query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  ASSERT_EQ(FindQueueFamilyOrdinal(
                api_, xdna_endpoint_, AMDF_QUEUE_COMMAND_TYPE_XDNA,
                &pool_query.consumer.value.device.queue_family_ordinal),
            AMDF_STATUS_OK);
  amdf_memory_pair_info_t qualified = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(qualified)};
  ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &pool_query,
                                               &qualified),
            AMDF_STATUS_OK);
  EXPECT_EQ(qualified.release.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  const amdf_memory_pair_info_t original_qualified = qualified;
  pool_query.external_memory_provenance.words[0] ^= 1;
  EXPECT_EQ(amdf_status_code(api_->memory_scope_query_pair_info(
                system_scope_, &pool_query, &qualified)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&qualified, &original_qualified, sizeof(qualified)), 0);
  pool_query.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
  pool_query.external_memory_provenance = {};
  EXPECT_EQ(amdf_status_code(api_->memory_scope_query_pair_info(
                system_scope_, &pool_query, &qualified)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&qualified, &original_qualified, sizeof(qualified)), 0);
  pool_query.producer = pool_query.consumer;
  amdf_memory_pair_info_t device_only = qualified;
  ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &pool_query,
                                               &device_only),
            AMDF_STATUS_OK);
  EXPECT_EQ(device_only.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(device_only.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);

  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = gpu_profile_ordinal;
  create_info.access_count = 1;
  create_info.accesses = &gpu_access_;
  create_info.required_flags = kGpuRequiredFlags;
  create_info.byte_length = backing_byte_length;
  create_info.minimum_alignment = page_size;
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &gpu_memory_),
            AMDF_STATUS_OK);

  amdf_memory_info_t gpu_memory_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(gpu_memory_info)};
  ASSERT_EQ(api_->memory_query_info(gpu_memory_, &gpu_memory_info),
            AMDF_STATUS_OK);
  amdf_memory_access_info_t gpu_access_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(gpu_access_info)};
  ASSERT_EQ(api_->memory_query_access_info(gpu_memory_, 0, &gpu_access_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_memory_info.memory_profile_ordinal, gpu_profile_ordinal);
  EXPECT_EQ((gpu_memory_info.flags | gpu_access_info.flags) & kGpuRequiredFlags,
            kGpuRequiredFlags);
  ASSERT_GE(gpu_memory_info.byte_length, backing_byte_length);
  ASSERT_TRUE(
      amdf_physical_memory_id_is_valid(&gpu_memory_info.physical_backing_id));

  amdf_host_mapping_info_t gpu_mapping_info = {};
  ASSERT_EQ(Map(gpu_memory_, gpu_memory_info.byte_length, &gpu_mapping_,
                &gpu_mapping_info),
            AMDF_STATUS_OK);
  // GPU coherence does not turn CPU cache control into a universal no-op for
  // the XDNA consumer importing these same pages.
  EXPECT_EQ(gpu_mapping_info.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  EXPECT_EQ(gpu_mapping_info.flush.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  EXPECT_EQ(gpu_mapping_info.flush.host_instruction,
            AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(gpu_mapping_info.invalidate.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
  auto* gpu_bytes = static_cast<uint8_t*>(gpu_mapping_info.pointer);
  ASSERT_NE(gpu_bytes, nullptr);
  std::memset(gpu_bytes, 0x11, static_cast<size_t>(page_size));
  std::memset(gpu_bytes + page_size, 0x22, static_cast<size_t>(page_size));
  std::memset(gpu_bytes + 2 * page_size, 0x33, static_cast<size_t>(page_size));
  ASSERT_EQ(api_->host_mapping_cache_control(gpu_mapping_,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             backing_byte_length),
            AMDF_STATUS_OK);

  amdf_memory_export_info_t export_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      .structure_size = sizeof(export_info),
      .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD,
      .byte_offset = page_size,
      .byte_length = page_size};
  ASSERT_EQ(api_->memory_export(gpu_memory_, &export_info, &external_memory_),
            AMDF_STATUS_OK);
  ASSERT_EQ(external_memory_.type, AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD);
  ASSERT_GE(external_memory_.payload.file_descriptor, 0);
  ASSERT_LE(external_memory_.payload.file_descriptor, INT32_MAX);
  const int exported_descriptor =
      static_cast<int>(external_memory_.payload.file_descriptor);
  const int descriptor_flags = fcntl(exported_descriptor, F_GETFD);
  ASSERT_NE(descriptor_flags, -1);
  EXPECT_NE(descriptor_flags & FD_CLOEXEC, 0);
  EXPECT_EQ(external_memory_.source_byte_offset, page_size);
  EXPECT_EQ(external_memory_.byte_length, page_size);
  EXPECT_TRUE(
      amdf_physical_memory_id_is_equal(&external_memory_.physical_backing_id,
                                       &gpu_memory_info.physical_backing_id));

  amdf_memory_import_info_t import_info = {};
  import_info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
  import_info.structure_size = sizeof(import_info);
  import_info.memory_profile_ordinal = xdna_profile_ordinal;
  import_info.access_count = 1;
  import_info.accesses = &xdna_access_;
  import_info.required_flags = kXdnaRequiredFlags;
  import_info.minimum_alignment = page_size;
  external_memory_.provenance.words[0] ^= 1;
  const auto mismatched_transport = external_memory_;
  auto* sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  auto* rejected = sentinel;
  EXPECT_EQ(amdf_status_code(api_->memory_import(system_scope_, &import_info,
                                                 &external_memory_, &rejected)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(rejected, sentinel);
  EXPECT_EQ(std::memcmp(&external_memory_, &mismatched_transport,
                        sizeof(external_memory_)),
            0);
  external_memory_.provenance.words[0] ^= 1;
  ASSERT_EQ(api_->memory_import(system_scope_, &import_info, &external_memory_,
                                &xdna_memory_),
            AMDF_STATUS_OK);
  const amdf_external_memory_t empty_external_memory = {};
  EXPECT_EQ(std::memcmp(&external_memory_, &empty_external_memory,
                        sizeof(external_memory_)),
            0);
  errno = 0;
  EXPECT_EQ(fcntl(exported_descriptor, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);

  amdf_memory_info_t xdna_memory_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(xdna_memory_info)};
  ASSERT_EQ(api_->memory_query_info(xdna_memory_, &xdna_memory_info),
            AMDF_STATUS_OK);
  amdf_memory_access_info_t xdna_access_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(xdna_access_info)};
  ASSERT_EQ(api_->memory_query_access_info(xdna_memory_, 0, &xdna_access_info),
            AMDF_STATUS_OK);
  EXPECT_EQ(xdna_memory_info.memory_profile_ordinal, xdna_profile_ordinal);
  EXPECT_EQ(
      (xdna_memory_info.flags | xdna_access_info.flags) & kXdnaRequiredFlags,
      kXdnaRequiredFlags);
  EXPECT_EQ(xdna_memory_info.source_byte_offset, page_size);
  EXPECT_EQ(xdna_memory_info.byte_length, page_size);
  EXPECT_GE(xdna_memory_info.alignment, page_size);
  uint64_t xdna_address = 0;
  ASSERT_EQ(
      api_->memory_query_address(
          xdna_memory_, 0, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE, &xdna_address),
      AMDF_STATUS_OK);
  EXPECT_EQ(xdna_address & (page_size - 1), 0u);
  EXPECT_TRUE(
      amdf_physical_memory_id_is_equal(&xdna_memory_info.physical_backing_id,
                                       &gpu_memory_info.physical_backing_id));

  uint32_t gpu_queue_family_ordinal = UINT32_MAX;
  ASSERT_EQ(FindQueueFamilyOrdinal(api_, gpu_endpoint_,
                                   AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                   &gpu_queue_family_ordinal),
            AMDF_STATUS_OK);
  uint32_t xdna_queue_family_ordinal = UINT32_MAX;
  ASSERT_EQ(
      FindQueueFamilyOrdinal(api_, xdna_endpoint_, AMDF_QUEUE_COMMAND_TYPE_XDNA,
                             &xdna_queue_family_ordinal),
      AMDF_STATUS_OK);

  amdf_memory_site_t gpu_site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                 .structure_size = sizeof(gpu_site)};
  gpu_site.value.device.memory = gpu_memory_;
  gpu_site.value.device.queue_family_ordinal = gpu_queue_family_ordinal;
  amdf_memory_site_t xdna_site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                  .structure_size = sizeof(xdna_site)};
  xdna_site.value.device.memory = xdna_memory_;
  xdna_site.value.device.queue_family_ordinal = xdna_queue_family_ordinal;

  amdf_memory_pair_info_t gpu_to_xdna = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(gpu_to_xdna)};
  ASSERT_EQ(api_->memory_query_pair_info(&gpu_site, &xdna_site, &gpu_to_xdna),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_to_xdna.flags, AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE);
  EXPECT_EQ(gpu_to_xdna.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  EXPECT_EQ(gpu_to_xdna.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  EXPECT_EQ(gpu_to_xdna.release.operation,
            AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
  EXPECT_EQ(gpu_to_xdna.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(gpu_to_xdna.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(gpu_to_xdna.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);

  amdf_memory_pair_info_t xdna_to_gpu = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(xdna_to_gpu)};
  ASSERT_EQ(api_->memory_query_pair_info(&xdna_site, &gpu_site, &xdna_to_gpu),
            AMDF_STATUS_OK);
  EXPECT_EQ(xdna_to_gpu.flags, AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE);
  EXPECT_EQ(xdna_to_gpu.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  EXPECT_EQ(xdna_to_gpu.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  EXPECT_EQ(xdna_to_gpu.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  EXPECT_EQ(xdna_to_gpu.acquire.operation,
            AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
  EXPECT_EQ(xdna_to_gpu.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
  EXPECT_EQ(xdna_to_gpu.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);

  amdf_host_mapping_info_t xdna_mapping_info = {};
  ASSERT_EQ(Map(xdna_memory_, page_size, &xdna_mapping_, &xdna_mapping_info),
            AMDF_STATUS_OK);
  amdf_memory_site_t host_site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                  .structure_size = sizeof(host_site),
                                  .kind = AMDF_MEMORY_SITE_KIND_HOST};
  host_site.value.host_mapping = xdna_mapping_;
  amdf_memory_pair_info_t actual = qualified;
  ASSERT_EQ(api_->memory_query_pair_info(&host_site, &xdna_site, &actual),
            AMDF_STATUS_OK);
  ExpectMemoryPairEqual(actual, qualified);
  auto* xdna_bytes = static_cast<uint8_t*>(xdna_mapping_info.pointer);
  ASSERT_NE(xdna_bytes, nullptr);
  ASSERT_EQ(
      api_->host_mapping_cache_control(
          xdna_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, page_size),
      AMDF_STATUS_OK);
  EXPECT_EQ(xdna_bytes[0], 0x22);
  EXPECT_EQ(xdna_bytes[page_size - 1], 0x22);

  std::memset(xdna_bytes, 0xC3, static_cast<size_t>(page_size));
  ASSERT_EQ(api_->host_mapping_cache_control(
                xdna_mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, page_size),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->host_mapping_cache_control(
                gpu_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, page_size,
                page_size),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_bytes[0], 0x11);
  EXPECT_EQ(gpu_bytes[page_size], 0xC3);
  EXPECT_EQ(gpu_bytes[2 * page_size - 1], 0xC3);
  EXPECT_EQ(gpu_bytes[2 * page_size], 0x33);

  ASSERT_EQ(api_->host_mapping_destroy(gpu_mapping_), AMDF_STATUS_OK);
  gpu_mapping_ = nullptr;
  ASSERT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
            AMDF_STATUS_OK);
}

void GpuXdnaMemoryInteropTest::CheckSurvivingImport() {
  amdf_host_mapping_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(info)};
  ASSERT_EQ(api_->host_mapping_query_info(xdna_mapping_, &info),
            AMDF_STATUS_OK);
  auto* xdna_bytes = static_cast<uint8_t*>(info.pointer);
  const uint64_t page_size = info.byte_length;

  ASSERT_EQ(
      api_->host_mapping_cache_control(
          xdna_mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, page_size),
      AMDF_STATUS_OK);
  EXPECT_EQ(xdna_bytes[0], 0xC3);
  EXPECT_EQ(xdna_bytes[page_size - 1], 0xC3);
  xdna_bytes[page_size / 2] = 0x7D;
  EXPECT_EQ(xdna_bytes[page_size / 2], 0x7D);
}

TEST_F(GpuXdnaMemoryInteropTest,
       ImportsGpuSubrangeAndSurvivesAllocationTeardown) {
  ASSERT_NO_FATAL_FAILURE(ImportGpuSubrangeAndReleaseAllocation());
  ASSERT_NO_FATAL_FAILURE(CheckSurvivingImport());
}

TEST_F(GpuXdnaMemoryLifetimeTest, ImportsGpuSubrangeAndSurvivesSourceTeardown) {
  ASSERT_NO_FATAL_FAILURE(ImportGpuSubrangeAndReleaseAllocation());
  ASSERT_EQ(api_->device_destroy(gpu_device_), AMDF_STATUS_OK);
  gpu_device_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(CheckSurvivingImport());
}

TEST_F(GpuXdnaMemoryInteropTest,
       ConstructsOneBackingWithCallerOrderedGpuAndXdnaAccess) {
  // XDNA cannot export an ordinary allocation to KFD. Its first-place ordinal
  // therefore checks that backing selection does not reorder public accesses.
  amdf_endpoint_t* endpoints[] = {xdna_endpoint_, gpu_endpoint_};
  const amdf_memory_device_access_t devices[] = {xdna_access_, gpu_access_};
  amdf_memory_profile_t profile = {};
  amdf_memory_access_capabilities_t capabilities[2] = {};
  ASSERT_NO_FATAL_FAILURE(FindJointProfile(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                                           &profile, capabilities));
  ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 2,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = 8195,
      .minimum_alignment = 4096,
      .accesses = devices,
  };
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &gpu_memory_),
            AMDF_STATUS_OK);
  amdf_memory_info_t info = {.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
                             .structure_size = sizeof(info)};
  ASSERT_EQ(api_->memory_query_info(gpu_memory_, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.access_count, 2u);
  EXPECT_EQ(info.byte_length, create_info.byte_length);
  EXPECT_GT(info.native_allocation_byte_length, info.byte_length);
  EXPECT_EQ(info.memory_profile_ordinal, profile.ordinal);
  for (uint32_t i = 0; i < 2; ++i) {
    amdf_memory_access_info_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
        .structure_size = sizeof(access)};
    ASSERT_EQ(api_->memory_query_access_info(gpu_memory_, i, &access),
              AMDF_STATUS_OK);
    EXPECT_EQ(access.ordinal, i);
    EXPECT_EQ(access.access, devices[i].requirements.access);
    EXPECT_EQ(access.flags & devices[i].requirements.flags,
              devices[i].requirements.flags);
    EXPECT_EQ(access.address_kinds & ~capabilities[i].address_kinds, 0u);
  }
  uint64_t gpu_address = 0;
  uint64_t xdna_address = 0;
  ASSERT_EQ(api_->memory_query_address(gpu_memory_, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       &gpu_address),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_query_address(
                gpu_memory_, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &xdna_address),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_address & 4095, 0u);
  EXPECT_EQ(xdna_address & 4095, 0u);
  amdf_host_mapping_info_t first = {};
  amdf_host_mapping_info_t second = {};
  ASSERT_EQ(Map(gpu_memory_, create_info.byte_length, &gpu_mapping_, &first),
            AMDF_STATUS_OK);
  ASSERT_EQ(Map(gpu_memory_, create_info.byte_length, &xdna_mapping_, &second),
            AMDF_STATUS_OK);
  EXPECT_EQ(first.pointer, second.pointer);
  std::memset(first.pointer, 0x69, create_info.byte_length);
  EXPECT_EQ(
      static_cast<const uint8_t*>(second.pointer)[create_info.byte_length - 1],
      0x69);
  uint64_t address_after = 0;
  ASSERT_EQ(api_->memory_query_address(gpu_memory_, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       &address_after),
            AMDF_STATUS_OK);
  EXPECT_EQ(address_after, gpu_address);

  amdf_memory_site_t host_site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                  .structure_size = sizeof(host_site),
                                  .kind = AMDF_MEMORY_SITE_KIND_HOST};
  host_site.value.host_mapping = gpu_mapping_;
  amdf_memory_site_t device_site = {.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                                    .structure_size = sizeof(device_site)};
  device_site.value.device.memory = gpu_memory_;
  for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
    device_site.value.device.access_ordinal = ordinal;
    ASSERT_EQ(
        FindQueueFamilyOrdinal(api_, endpoints[ordinal],
                               ordinal == 0 ? AMDF_QUEUE_COMMAND_TYPE_XDNA
                                            : AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                               &device_site.value.device.queue_family_ordinal),
        AMDF_STATUS_OK);
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair)};
    ASSERT_EQ(api_->memory_query_pair_info(&host_site, &device_site, &pair),
              AMDF_STATUS_OK);
    EXPECT_EQ(pair.release.kind, ordinal == 0
                                     ? AMDF_CACHE_TRANSITION_KIND_RANGE
                                     : AMDF_CACHE_TRANSITION_KIND_NONE);
    if (ordinal == 0) {
      EXPECT_EQ(pair.release.executor,
                AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
      EXPECT_EQ(pair.release.host_instruction,
                AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
      EXPECT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
    }
    ASSERT_EQ(api_->memory_query_pair_info(&device_site, &host_site, &pair),
              AMDF_STATUS_OK);
    EXPECT_EQ(pair.acquire.kind, ordinal == 0
                                     ? AMDF_CACHE_TRANSITION_KIND_RANGE
                                     : AMDF_CACHE_TRANSITION_KIND_NONE);
    if (ordinal == 0) {
      EXPECT_EQ(pair.acquire.executor,
                AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
      EXPECT_EQ(pair.acquire.host_operation,
                AMDF_HOST_CACHE_OPERATION_INVALIDATE);
    }
  }
}

TEST_F(GpuXdnaMemoryInteropTest,
       QualifiesAllDirectionalPairsBeforeRepeatedConstruction) {
  const amdf_memory_device_access_t accesses[] = {xdna_access_, gpu_access_};
  std::array<amdf_memory_profile_site_t, 3> sites = {};
  for (uint32_t i = 0; i < 2; ++i) {
    sites[i].kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    sites[i].value.device.access_ordinal = i;
    ASSERT_EQ(
        FindQueueFamilyOrdinal(api_, i == 0 ? xdna_endpoint_ : gpu_endpoint_,
                               i == 0 ? AMDF_QUEUE_COMMAND_TYPE_XDNA
                                      : AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                               &sites[i].value.device.queue_family_ordinal),
        AMDF_STATUS_OK);
  }
  sites[2].kind = AMDF_MEMORY_SITE_KIND_HOST;
  sites[2].value.host_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;

  for (auto role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_REGISTER}) {
    amdf_memory_profile_t profile = {};
    amdf_memory_access_capabilities_t capabilities[2] = {};
    ASSERT_NO_FATAL_FAILURE(FindJointProfile(role, &profile, capabilities));
    if (profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      // Registration is absent when the native VM has an independent lease.
      ASSERT_EQ(role, AMDF_MEMORY_PROFILE_ROLE_REGISTER);
      continue;
    }
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = profile.ordinal;
    query.access_count = 2;
    query.accesses = accesses;
    query.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    query.registered_host_cacheability =
        profile.registration.registered_host_cacheability;
    std::array<amdf_memory_pair_info_t, 9> qualified = {};
    for (uint32_t producer = 0; producer < 3; ++producer) {
      for (uint32_t consumer = 0; consumer < 3; ++consumer) {
        query.producer = sites[producer];
        query.consumer = sites[consumer];
        auto& pair = qualified[producer * 3 + consumer];
        pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
        pair.structure_size = sizeof(pair);
        ASSERT_EQ(
            api_->memory_scope_query_pair_info(system_scope_, &query, &pair),
            AMDF_STATUS_OK);
      }
    }
    // Caller order is XDNA, GPU, host. Each peer retains its own requirements.
    EXPECT_EQ(qualified[3].release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(qualified[3].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(qualified[1].release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(qualified[1].acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(qualified[6].release.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    EXPECT_EQ(qualified[7].release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    EXPECT_EQ(qualified[2].acquire.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    EXPECT_EQ(qualified[5].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);

    if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
      const size_t length = 16384;
      void* pages = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      ASSERT_NE(pages, MAP_FAILED);
      caller_pages_.pointer = static_cast<uint8_t*>(pages);
      caller_pages_.byte_length = length;
    }
    for (uint32_t iteration = 0; iteration < 3; ++iteration) {
      amdf_memory_create_info_t create = {};
      create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create.structure_size = sizeof(create);
      create.memory_profile_ordinal = profile.ordinal;
      create.access_count = 2;
      create.accesses = accesses;
      create.required_flags = query.required_flags;
      create.byte_length = 4099 + iteration * 127;
      create.registered_host_pointer = caller_pages_.pointer == nullptr
                                           ? nullptr
                                           : caller_pages_.pointer + 3;
      create.registered_host_cacheability = query.registered_host_cacheability;
      if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        for (auto cacheability : {AMDF_HOST_CACHEABILITY_UNKNOWN,
                                  AMDF_HOST_CACHEABILITY_WRITE_COMBINED,
                                  AMDF_HOST_CACHEABILITY_UNCACHED}) {
          create.registered_host_cacheability = cacheability;
          auto* sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
          amdf_memory_t* rejected = sentinel;
          EXPECT_EQ(amdf_status_code(
                        api_->memory_create(system_scope_, &create, &rejected)),
                    AMDF_STATUS_CODE_UNSUPPORTED);
          EXPECT_EQ(rejected, sentinel);
        }
        create.registered_host_cacheability =
            query.registered_host_cacheability;
      }
      ASSERT_EQ(api_->memory_create(system_scope_, &create, &gpu_memory_),
                AMDF_STATUS_OK);
      amdf_host_mapping_info_t mapping = {};
      ASSERT_EQ(Map(gpu_memory_, create.byte_length, &gpu_mapping_, &mapping),
                AMDF_STATUS_OK);
      std::array<amdf_memory_site_t, 3> actual_sites = {};
      for (uint32_t i = 0; i < 3; ++i) {
        auto& site = actual_sites[i];
        site.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
        site.structure_size = sizeof(site);
        site.kind = sites[i].kind;
        if (i == 2) {
          site.value.host_mapping = gpu_mapping_;
        } else {
          site.value.device.memory = gpu_memory_;
          site.value.device.access_ordinal = i;
          site.value.device.queue_family_ordinal =
              sites[i].value.device.queue_family_ordinal;
        }
      }
      for (uint32_t producer = 0; producer < 3; ++producer) {
        for (uint32_t consumer = 0; consumer < 3; ++consumer) {
          amdf_memory_pair_info_t pair = {
              .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
              .structure_size = sizeof(pair)};
          ASSERT_EQ(
              api_->memory_query_pair_info(&actual_sites[producer],
                                           &actual_sites[consumer], &pair),
              AMDF_STATUS_OK);
          SCOPED_TRACE(::testing::Message()
                       << role << ": " << producer << " -> " << consumer);
          ExpectMemoryPairEqual(pair, qualified[producer * 3 + consumer]);
        }
      }
      ASSERT_EQ(api_->host_mapping_destroy(gpu_mapping_), AMDF_STATUS_OK);
      gpu_mapping_ = nullptr;
      ASSERT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
                AMDF_STATUS_OK);
    }
  }
}

TEST_F(GpuXdnaMemoryInteropTest,
       RegistersCallerPagesWithJointAccessAndPreservesCallerOwnership) {
  amdf_memory_profile_t profile = {};
  amdf_memory_access_capabilities_t capabilities[2] = {};
  ASSERT_NO_FATAL_FAILURE(FindJointProfile(AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                                           &profile, capabilities));
  amdf_gpu_device_info_t gpu_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(gpu_capabilities)};
  ASSERT_EQ(gpu_api_->device_query_info(gpu_device_, &gpu_capabilities),
            AMDF_STATUS_OK);
  if ((gpu_capabilities.features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) ==
      0) {
    EXPECT_EQ(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    return;
  }
  ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  // Separate device owners may share one native VM. Registration still
  // requires equal exact permissions for the shared backing.
  const amdf_gpu_device_create_info_t create_device = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create_device),
  };
  ASSERT_EQ(gpu_api_->device_create(gpu_endpoint_, &create_device,
                                    &additional_gpu_device_),
            AMDF_STATUS_OK);
  amdf_memory_device_access_t gpu_devices[] = {
      {gpu_device_, gpu_access_.requirements},
      {additional_gpu_device_, gpu_access_.requirements},
  };
  amdf_memory_profile_t shared = profile;
  ASSERT_EQ(api_->memory_scope_query_device_profile(
                system_scope_, profile.ordinal, 2, gpu_devices, &shared,
                capabilities),
            AMDF_STATUS_OK);
  gpu_devices[1].requirements.access = AMDF_MEMORY_ACCESS_READ;
  amdf_memory_profile_t rejected = profile;
  EXPECT_EQ(amdf_status_code(api_->memory_scope_query_device_profile(
                system_scope_, profile.ordinal, 2, gpu_devices, &rejected,
                capabilities)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(std::memcmp(&rejected, &profile, sizeof(profile)), 0);

  const long native_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(native_page_size, 0);
  const size_t page_size = static_cast<size_t>(native_page_size);
  const size_t reservation_length = page_size * 3;
  void* pages = mmap(nullptr, reservation_length, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(pages, MAP_FAILED);
  std::memset(pages, 0x7B, reservation_length);
  const amdf_memory_device_access_t devices[] = {xdna_access_, gpu_access_};
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 2,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = page_size + 17,
      .minimum_alignment = 1,
      .registered_host_pointer = static_cast<uint8_t*>(pages) + 3,
      .accesses = devices,
      .registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
  };
  // An exceptional failed construction may leave native pins behind. Preserve
  // caller pages until process exit unless a published owner can release them.
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &gpu_memory_),
            AMDF_STATUS_OK);
  caller_pages_.pointer = static_cast<uint8_t*>(pages);
  caller_pages_.byte_length = reservation_length;
  amdf_memory_info_t info = {.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
                             .structure_size = sizeof(info)};
  ASSERT_EQ(api_->memory_query_info(gpu_memory_, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.access_count, 2u);
  EXPECT_EQ(info.byte_length, create_info.byte_length);
  EXPECT_EQ(info.source_byte_offset, 3u);
  uint64_t gpu_address = 0;
  uint64_t xdna_address = 0;
  ASSERT_EQ(api_->memory_query_address(gpu_memory_, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       &gpu_address),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_query_address(
                gpu_memory_, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &xdna_address),
            AMDF_STATUS_OK);
  EXPECT_EQ(gpu_address % page_size, 3u);
  EXPECT_EQ(xdna_address % page_size, 3u);
  amdf_host_mapping_info_t mapping = {};
  ASSERT_EQ(Map(gpu_memory_, info.byte_length, &gpu_mapping_, &mapping),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping.pointer, create_info.registered_host_pointer);
  std::memset(mapping.pointer, 0x39, mapping.byte_length);
  ASSERT_EQ(api_->host_mapping_destroy(gpu_mapping_), AMDF_STATUS_OK);
  gpu_mapping_ = nullptr;
  uint64_t stable_address = 0;
  ASSERT_EQ(api_->memory_query_address(gpu_memory_, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       &stable_address),
            AMDF_STATUS_OK);
  EXPECT_EQ(stable_address, gpu_address);
  ASSERT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
            AMDF_STATUS_OK);
  EXPECT_EQ(caller_pages_.pointer[2], 0x7B);
  EXPECT_EQ(caller_pages_.pointer[3], 0x39);
  EXPECT_EQ(caller_pages_.pointer[2 + create_info.byte_length], 0x39);
  EXPECT_EQ(caller_pages_.pointer[3 + create_info.byte_length], 0x7B);
  std::memset(caller_pages_.pointer, 0x45, reservation_length);
}

TEST_F(GpuXdnaMemoryInteropTest,
       RollsBackRejectedRegistrationAndReusesCallerPages) {
  amdf_memory_profile_t profile = {};
  amdf_memory_access_capabilities_t capabilities[2] = {};
  ASSERT_NO_FATAL_FAILURE(FindJointProfile(AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                                           &profile, capabilities));
  amdf_gpu_device_info_t gpu_capabilities = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(gpu_capabilities)};
  ASSERT_EQ(gpu_api_->device_query_info(gpu_device_, &gpu_capabilities),
            AMDF_STATUS_OK);
  if ((gpu_capabilities.features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION) ==
      0) {
    EXPECT_EQ(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    GTEST_SKIP() << "GPU host registration is unavailable for this lifetime";
  }
  ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  ASSERT_EQ(profile.registration.registered_host_cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_BACK);

  const long native_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(native_page_size, 0);
  const size_t page_size = static_cast<size_t>(native_page_size);
  const std::array<size_t, 2> byte_lengths = {page_size, 16 * page_size};
  const size_t reservation_length = byte_lengths.back() + 2 * page_size;
  void* pages = mmap(nullptr, reservation_length, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(pages, MAP_FAILED);
  caller_pages_.pointer = static_cast<uint8_t*>(pages);
  caller_pages_.byte_length = reservation_length;
  std::vector<uint8_t> expected(reservation_length, 0xA5);
  std::memcpy(pages, expected.data(), expected.size());

  struct rlimit saved_limit = {};
  ASSERT_EQ(getrlimit(RLIMIT_MEMLOCK, &saved_limit), 0);
  struct rlimit restricted_limit = saved_limit;
  // Keep mlock permission while admitting zero complete pages. Only this
  // process's soft limit changes; restore it before evaluating each result.
  restricted_limit.rlim_cur = 1;
  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &restricted_limit), 0);
  // Sanitizer interceptors make libc mlock/munlock no-ops. Query the actual
  // kernel quota so sanitized runs do not appear to have privileged bypass.
  const long lock_result = syscall(SYS_mlock, pages, page_size);
  const int lock_error = errno;
  const int restore_result = setrlimit(RLIMIT_MEMLOCK, &saved_limit);
  if (lock_result == 0) {
    ASSERT_EQ(syscall(SYS_munlock, pages, page_size), 0);
    ASSERT_EQ(restore_result, 0);
    GTEST_SKIP() << "memlock quota is bypassed by this process's privileges";
  }
  ASSERT_EQ(restore_result, 0);
  ASSERT_EQ(lock_result, -1);
  ASSERT_EQ(lock_error, ENOMEM);

  amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = byte_lengths.back(),
      .minimum_alignment = profile.registration.minimum_alignment,
      .registered_host_pointer = caller_pages_.pointer + page_size,
      .accesses = &gpu_access_,
      .registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
  };
  // GPU USERPTR registration succeeds under this quota. With GPU first in
  // the joint request, an XDNA quota rejection must unwind that preparation.
  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &restricted_limit), 0);
  const amdf_status_t control_status =
      api_->memory_create(system_scope_, &create_info, &gpu_memory_);
  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &saved_limit), 0);
  ASSERT_EQ(control_status, AMDF_STATUS_OK);
  ASSERT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
            AMDF_STATUS_OK);
  ASSERT_EQ(std::memcmp(pages, expected.data(), expected.size()), 0);

  for (uint32_t gpu_ordinal = 0; gpu_ordinal < 2; ++gpu_ordinal) {
    SCOPED_TRACE(gpu_ordinal);
    amdf_memory_device_access_t devices[2] = {};
    devices[gpu_ordinal] = gpu_access_;
    devices[1 - gpu_ordinal] = xdna_access_;
    ASSERT_EQ(
        api_->memory_scope_query_device_profile(
            system_scope_, profile.ordinal, 2, devices, &profile, capabilities),
        AMDF_STATUS_OK);
    create_info.access_count = 2;
    create_info.accesses = devices;
    for (size_t byte_length : byte_lengths) {
      SCOPED_TRACE(byte_length);
      create_info.byte_length = byte_length;
      for (uint32_t generation = 0; generation < 3; ++generation) {
        SCOPED_TRACE(generation);
        auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
        amdf_memory_t* output = sentinel;
        ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &restricted_limit), 0);
        const amdf_status_t status =
            api_->memory_create(system_scope_, &create_info, &output);
        const int restored = setrlimit(RLIMIT_MEMLOCK, &saved_limit);
        if (amdf_status_is_ok(status)) {
          gpu_memory_ = output;
        }
        ASSERT_EQ(restored, 0);
        ASSERT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_ERRNO);
        ASSERT_EQ(amdf_status_code(status), ENOMEM);
        ASSERT_EQ(output, sentinel);
        ASSERT_EQ(std::memcmp(pages, expected.data(), expected.size()), 0);

        // Reuse the same caller bytes immediately after the failed transaction.
        // Registration and its host view must expose only the new generation.
        std::memset(expected.data() + page_size, 0x30 + generation,
                    byte_length);
        std::memcpy(pages, expected.data(), expected.size());
        ASSERT_EQ(
            api_->memory_create(system_scope_, &create_info, &gpu_memory_),
            AMDF_STATUS_OK);
        uint64_t gpu_address = 0;
        uint64_t xdna_address = 0;
        ASSERT_EQ(
            api_->memory_query_address(gpu_memory_, gpu_ordinal,
                                       AMDF_MEMORY_ADDRESS_GPU, &gpu_address),
            AMDF_STATUS_OK);
        ASSERT_EQ(api_->memory_query_address(gpu_memory_, 1 - gpu_ordinal,
                                             AMDF_MEMORY_ADDRESS_XDNA_DMA,
                                             &xdna_address),
                  AMDF_STATUS_OK);
        EXPECT_NE(gpu_address, 0u);
        EXPECT_NE(xdna_address, 0u);
        amdf_host_mapping_info_t mapping = {};
        ASSERT_EQ(Map(gpu_memory_, byte_length, &gpu_mapping_, &mapping),
                  AMDF_STATUS_OK);
        ASSERT_EQ(mapping.pointer, create_info.registered_host_pointer);
        ASSERT_EQ(mapping.byte_length, byte_length);
        ASSERT_EQ(std::memcmp(pages, expected.data(), expected.size()), 0);
        std::memset(mapping.pointer, 0x60 + generation, byte_length);
        std::memset(expected.data() + page_size, 0x60 + generation,
                    byte_length);
        ASSERT_EQ(
            api_->host_mapping_destroy(std::exchange(gpu_mapping_, nullptr)),
            AMDF_STATUS_OK);
        ASSERT_EQ(api_->memory_destroy(std::exchange(gpu_memory_, nullptr)),
                  AMDF_STATUS_OK);
        ASSERT_EQ(std::memcmp(pages, expected.data(), expected.size()), 0);
      }
    }
  }
}

TEST_F(GpuXdnaMemoryInteropTest,
       RejectsMismatchedIdentityWithoutConsumingTransport) {
  const long system_page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(system_page_size, 0);
  const uint64_t page_size = static_cast<uint64_t>(system_page_size);

  uint32_t xdna_profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_profile_t xdna_profile = {};
  ASSERT_EQ(
      FindDmaBufProfile(
          api_, system_scope_, xdna_device_,
          AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          AMDF_MEMORY_FLAG_HOST_VISIBLE, xdna_access_.requirements,
          AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
          &xdna_profile_ordinal, &xdna_profile),
      AMDF_STATUS_OK);
  ASSERT_NE(xdna_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  uint32_t gpu_profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  amdf_memory_profile_t gpu_profile = {};
  ASSERT_EQ(
      FindDmaBufProfile(
          api_, system_scope_, gpu_device_,
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT,
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE,
          gpu_access_.requirements, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT,
          &gpu_profile_ordinal, &gpu_profile),
      AMDF_STATUS_OK);
  ASSERT_NE(gpu_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);

  amdf_memory_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.memory_profile_ordinal = gpu_profile_ordinal;
  create_info.access_count = 1;
  create_info.accesses = &gpu_access_;
  create_info.required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
  create_info.byte_length = page_size;
  ASSERT_EQ(api_->memory_create(system_scope_, &create_info, &gpu_memory_),
            AMDF_STATUS_OK);

  amdf_memory_export_info_t export_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      .structure_size = sizeof(export_info),
      .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .byte_length = page_size};
  ASSERT_EQ(api_->memory_export(gpu_memory_, &export_info, &external_memory_),
            AMDF_STATUS_OK);
  ASSERT_TRUE(
      amdf_physical_memory_id_is_valid(&external_memory_.physical_backing_id));
  external_memory_.physical_backing_id.words[0] ^= UINT64_C(1);
  const amdf_external_memory_t mismatched_external_memory = external_memory_;
  ASSERT_GE(external_memory_.payload.file_descriptor, 0);
  ASSERT_LE(external_memory_.payload.file_descriptor, INT32_MAX);
  const int descriptor =
      static_cast<int>(external_memory_.payload.file_descriptor);

  amdf_memory_import_info_t import_info = {};
  import_info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
  import_info.structure_size = sizeof(import_info);
  import_info.memory_profile_ordinal = xdna_profile_ordinal;
  import_info.access_count = 1;
  import_info.accesses = &xdna_access_;
  import_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* output = sentinel;
  EXPECT_EQ(amdf_status_code(api_->memory_import(system_scope_, &import_info,
                                                 &external_memory_, &output)),
            AMDF_STATUS_CODE_FAILED_PRECONDITION);
  EXPECT_EQ(output, sentinel);
  EXPECT_EQ(std::memcmp(&external_memory_, &mismatched_external_memory,
                        sizeof(external_memory_)),
            0);
  EXPECT_NE(fcntl(descriptor, F_GETFD), -1);
}

}  // namespace
