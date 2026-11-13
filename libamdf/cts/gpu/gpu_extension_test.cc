// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

const amdf_api_t* QueryApi() {
  const amdf_api_t* api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api)));
  return api;
}

const amdf_gpu_api_t* QueryGpuApi(const amdf_api_t* api) {
  const void* extension_api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                           AMDF_GPU_EXTENSION_VERSION_LATEST, &extension_api)));
  return static_cast<const amdf_gpu_api_t*>(extension_api);
}

TEST_F(GpuDeviceFixture, InactivatesAdvertisedQueuesBeforeReleasingOwners) {
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
       ++ordinal) {
    SCOPED_TRACE(ordinal);
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    ASSERT_EQ(
        api_->endpoint_query_queue_family_info(endpoint_, ordinal, &family),
        AMDF_STATUS_OK);
    amdf_queue_family_placement_info_t placement = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO,
        .structure_size = sizeof(placement),
    };
    ASSERT_EQ(api_->device_query_queue_family_placement_info(device_, ordinal,
                                                             &placement),
              AMDF_STATUS_OK);
    if (family.user_queue_capabilities &
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) {
      const amdf_queue_placement_t requested_placement =
          (placement.placement_capabilities &
           AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL)
              ? AMDF_QUEUE_PLACEMENT_HOST_LOCAL
              : AMDF_QUEUE_PLACEMENT_DEFAULT;
      const amdf_gpu_user_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = ordinal,
          .priority = AMDF_QUEUE_PRIORITY_NORMAL,
          .placement = requested_placement,
          .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      };
      amdf_user_queue_t* queue = nullptr;
      ASSERT_EQ(gpu_api_->user_queue_create(device_, &create, &queue),
                AMDF_STATUS_OK);
      const auto release = [this](amdf_user_queue_t* value) {
        EXPECT_EQ(api_->user_queue_destroy(value), AMDF_STATUS_OK);
      };
      const std::unique_ptr<amdf_user_queue_t, decltype(release)> owner(
          queue, release);
      amdf_user_queue_mapping_t* mapping = nullptr;
      ASSERT_EQ(api_->user_queue_map(queue, nullptr, &mapping), AMDF_STATUS_OK);
      const auto unmap = [this](amdf_user_queue_mapping_t* value) {
        EXPECT_EQ(api_->user_queue_mapping_destroy(value), AMDF_STATUS_OK);
      };
      const std::unique_ptr<amdf_user_queue_mapping_t, decltype(unmap)> view(
          mapping, unmap);
      ASSERT_EQ(gpu_api_->user_queue_inactivate(queue), AMDF_STATUS_OK);
      EXPECT_EQ(gpu_api_->user_queue_inactivate(queue), AMDF_STATUS_OK);
      amdf_user_queue_mapping_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
          .structure_size = sizeof(info),
      };
      EXPECT_EQ(api_->user_queue_mapping_query_info(mapping, &info),
                AMDF_STATUS_OK);
    }
    if (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
      if (!(family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER)) {
        EXPECT_EQ(placement.placement_capabilities, 0u);
      }
      const amdf_gpu_kernel_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = ordinal,
      };
      amdf_kernel_queue_t* queue = nullptr;
      ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &create, &queue),
                AMDF_STATUS_OK);
      const auto release = [this](amdf_kernel_queue_t* value) {
        EXPECT_EQ(api_->kernel_queue_destroy(value), AMDF_STATUS_OK);
      };
      const std::unique_ptr<amdf_kernel_queue_t, decltype(release)> owner(
          queue, release);
      ASSERT_EQ(gpu_api_->kernel_queue_inactivate(queue), AMDF_STATUS_OK);
      EXPECT_EQ(gpu_api_->kernel_queue_inactivate(queue), AMDF_STATUS_OK);
    }
  }
}

class GpuMemoryObservationTest : public GpuDeviceFixture {};

class GpuQueuePriorityTest : public GpuDeviceFixture {};

TEST_F(GpuQueuePriorityTest, ConstructsEveryAdvertisedPriority) {
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  bool tested = false;
  for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
       ++ordinal) {
    SCOPED_TRACE(ordinal);
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    ASSERT_EQ(
        api_->endpoint_query_queue_family_info(endpoint_, ordinal, &family),
        AMDF_STATUS_OK);
    if (!(family.user_queue_capabilities &
          AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
      continue;
    }
    for (amdf_queue_priority_t priority :
         {AMDF_QUEUE_PRIORITY_LOW, AMDF_QUEUE_PRIORITY_NORMAL,
          AMDF_QUEUE_PRIORITY_HIGH}) {
      SCOPED_TRACE(priority);
      if (!(family.priority_capabilities & (1u << priority))) {
        continue;
      }
      const amdf_gpu_user_queue_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
          .structure_size = sizeof(create),
          .queue_family_ordinal = ordinal,
          .priority = priority,
          .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      };
      amdf_user_queue_t* queue = nullptr;
      ASSERT_EQ(gpu_api_->user_queue_create(device_, &create, &queue),
                AMDF_STATUS_OK);
      const auto release = [this](amdf_user_queue_t* value) {
        EXPECT_EQ(api_->user_queue_destroy(value), AMDF_STATUS_OK);
      };
      const std::unique_ptr<amdf_user_queue_t, decltype(release)> owner(
          queue, release);
      amdf_user_queue_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->user_queue_query_info(queue, &info), AMDF_STATUS_OK);
      EXPECT_EQ(info.priority, priority);
      tested = true;
    }
  }
  if (!tested) {
    GTEST_SKIP() << "endpoint has no host-published user queues";
  }
}

TEST_F(GpuMemoryObservationTest, PhysicalStorageReportsCapacityAndHeadroom) {
  amdf_memory_scope_info_t system_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(system_info),
  };
  ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &system_info),
            AMDF_STATUS_OK);
  for (amdf_memory_scope_t* scope : {system_scope_, local_scope_}) {
    if (!scope) {
      continue;
    }
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
    SCOPED_TRACE(info.kind);
    EXPECT_GT(info.capacity_byte_length, 0u);
    if (info.flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED) {
      EXPECT_EQ(info.capacity_byte_length, system_info.capacity_byte_length);
    }
    amdf_memory_scope_observation_t observation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_OBSERVATION,
        .structure_size = sizeof(observation),
    };
    if (info.flags & AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES) {
      ASSERT_EQ(api_->memory_scope_sample_observation(scope, &observation),
                AMDF_STATUS_OK);
      EXPECT_GT(observation.capacity_byte_length, 0u);
      if (!(info.flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED)) {
        EXPECT_EQ(observation.capacity_byte_length, info.capacity_byte_length);
      }
      EXPECT_LE(observation.available_byte_length,
                observation.capacity_byte_length);
    } else {
      EXPECT_EQ(api_->memory_scope_sample_observation(scope, &observation),
                amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    }
  }
}

TEST(GpuExtensionTest, ReportsCompiledAvailabilityBeforeCreatingInstance) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);

  const amdf_gpu_api_t* gpu_api = QueryGpuApi(api);

  ASSERT_NE(gpu_api, nullptr);
  EXPECT_EQ(gpu_api->structure_size, sizeof(amdf_gpu_api_t));
  EXPECT_EQ(gpu_api->extension_version, AMDF_GPU_EXTENSION_VERSION_LATEST);
  EXPECT_NE(gpu_api->endpoint_query_info, nullptr);
  EXPECT_NE(gpu_api->device_create, nullptr);
  EXPECT_NE(gpu_api->device_query_info, nullptr);
  EXPECT_NE(gpu_api->kernel_queue_create, nullptr);
  EXPECT_NE(gpu_api->kernel_queue_submit, nullptr);
  EXPECT_NE(gpu_api->user_queue_create, nullptr);
  EXPECT_NE(gpu_api->device_query_queue_clock_info, nullptr);
  EXPECT_NE(gpu_api->device_query_clock_sample, nullptr);
  EXPECT_NE(gpu_api->event_create, nullptr);
  EXPECT_NE(gpu_api->event_query_info, nullptr);
  EXPECT_NE(gpu_api->event_signal, nullptr);
  EXPECT_NE(gpu_api->event_wait, nullptr);
  EXPECT_NE(gpu_api->event_destroy, nullptr);
  EXPECT_NE(gpu_api->user_queue_query_scratch, nullptr);
  EXPECT_NE(gpu_api->user_queue_resume_scratch, nullptr);
  EXPECT_NE(gpu_api->user_queue_trim_scratch, nullptr);
  EXPECT_NE(gpu_api->user_queue_set_dispatch_timestamps, nullptr);
  EXPECT_NE(gpu_api->user_queue_inactivate, nullptr);
  EXPECT_NE(gpu_api->kernel_queue_inactivate, nullptr);
}

TEST(GpuExtensionTest, NegotiatesOriginalTableExtent) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* extension = nullptr;
  ASSERT_EQ(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                           AMDF_GPU_EXTENSION_VERSION_1, &extension),
      AMDF_STATUS_OK);
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  EXPECT_EQ(gpu->extension_version, AMDF_GPU_EXTENSION_VERSION_1);
  EXPECT_EQ(gpu->structure_size,
            offsetof(amdf_gpu_api_t, device_query_queue_clock_info));
  EXPECT_NE(gpu->user_queue_create, nullptr);
}

TEST(GpuExtensionTest, PreservesVersionTwoTableBoundary) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* extension = nullptr;
  ASSERT_EQ(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_2,
                           AMDF_GPU_EXTENSION_VERSION_2, &extension),
      AMDF_STATUS_OK);
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->extension_version, AMDF_GPU_EXTENSION_VERSION_2);
  EXPECT_EQ(gpu->structure_size, offsetof(amdf_gpu_api_t, event_create));
  EXPECT_NE(gpu->device_query_queue_clock_info, nullptr);
  EXPECT_NE(gpu->device_query_clock_sample, nullptr);
}

TEST(GpuExtensionTest, PreservesVersionThreeTableBoundary) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* extension = nullptr;
  ASSERT_EQ(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_3,
                           AMDF_GPU_EXTENSION_VERSION_3, &extension),
      AMDF_STATUS_OK);
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->extension_version, AMDF_GPU_EXTENSION_VERSION_3);
  EXPECT_EQ(gpu->structure_size,
            offsetof(amdf_gpu_api_t, user_queue_query_scratch));
  EXPECT_NE(gpu->event_destroy, nullptr);
}

TEST(GpuExtensionTest, PreservesVersionFiveTableBoundary) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* extension = nullptr;
  ASSERT_EQ(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_5,
                           AMDF_GPU_EXTENSION_VERSION_5, &extension),
      AMDF_STATUS_OK);
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->extension_version, AMDF_GPU_EXTENSION_VERSION_5);
  EXPECT_EQ(gpu->structure_size,
            offsetof(amdf_gpu_api_t, user_queue_inactivate));
  EXPECT_NE(gpu->user_queue_set_dispatch_timestamps, nullptr);
}

TEST(GpuExtensionTest, PreservesVersionFourTableBoundary) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* extension = nullptr;
  ASSERT_EQ(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_4,
                           AMDF_GPU_EXTENSION_VERSION_4, &extension),
      AMDF_STATUS_OK);
  const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->extension_version, AMDF_GPU_EXTENSION_VERSION_4);
  EXPECT_EQ(gpu->structure_size,
            offsetof(amdf_gpu_api_t, user_queue_set_dispatch_timestamps));
  EXPECT_NE(gpu->user_queue_trim_scratch, nullptr);
}

TEST(GpuExtensionTest, RejectsNullClockDeviceWithoutPublishingOutput) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const amdf_gpu_api_t* gpu = QueryGpuApi(api);
  ASSERT_NE(gpu, nullptr);
  const amdf_observation_clock_info_t sentinel = {UINT64_MAX, UINT32_MAX, 17};
  amdf_observation_clock_info_t info = sentinel;
  EXPECT_EQ(gpu->device_query_queue_clock_info(nullptr, 0, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&info, &sentinel, sizeof(info)), 0);
  amdf_observation_clock_sample_t sample_sentinel;
  std::memset(&sample_sentinel, 0x5a, sizeof(sample_sentinel));
  auto sample = sample_sentinel;
  EXPECT_EQ(gpu->device_query_clock_sample(nullptr, 0, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&sample, &sample_sentinel, sizeof(sample)), 0);
}

TEST(GpuExtensionTest, ReturnsStableImmutableTable) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);

  const amdf_gpu_api_t* first_api = QueryGpuApi(api);
  const amdf_gpu_api_t* second_api = QueryGpuApi(api);

  EXPECT_EQ(first_api, second_api);
}

TEST(GpuExtensionTest, RejectsUnsupportedVersionWithoutPublishingOutput) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);
  const void* const sentinel = reinterpret_cast<const void*>(uintptr_t{1});
  const void* extension_api = sentinel;

  const amdf_status_t status = api->query_extension(
      AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_LATEST + 1,
      AMDF_GPU_EXTENSION_VERSION_LATEST + 1, &extension_api);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_VERSION_MISMATCH);
  EXPECT_EQ(extension_api, sentinel);
}

class GpuEndpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    api_ = QueryApi();
    ASSERT_NE(api_, nullptr);
    gpu_api_ = QueryGpuApi(api_);
    ASSERT_NE(gpu_api_, nullptr);

    ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance_), AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (second_device_ != nullptr) {
      ASSERT_EQ(api_->device_destroy(second_device_), AMDF_STATUS_OK);
      second_device_ = nullptr;
    }
  }

  amdf_status_t OpenEngine(amdf_engine_kind_t engine_kind,
                           bool* out_engine_found) {
    *out_engine_found = false;
    uint32_t endpoint_count = 0;
    amdf_status_t status =
        api_->endpoint_enumerate(instance_, 0, nullptr, &endpoint_count);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    std::vector<amdf_endpoint_summary_t> summaries(endpoint_count);
    if (endpoint_count != 0) {
      status = api_->endpoint_enumerate(instance_, endpoint_count,
                                        summaries.data(), &endpoint_count);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
    }
    for (uint32_t endpoint_ordinal = 0; endpoint_ordinal < endpoint_count;
         ++endpoint_ordinal) {
      const amdf_endpoint_summary_t& summary = summaries[endpoint_ordinal];
      if (engine_kind == AMDF_ENGINE_KIND_GPU &&
          !GetCtsDeviceCache().IsGpuEndpointSelected(summary.id)) {
        continue;
      }
      if (summary.engine_kind == engine_kind) {
        status = GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint_);
        if (amdf_status_is_ok(status) && endpoint_ == nullptr) {
          return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
        }
        *out_engine_found = amdf_status_is_ok(status);
        return status;
      }
    }
    return AMDF_STATUS_OK;
  }

  amdf_gpu_device_create_info_t MakeDeviceCreateInfo() {
    amdf_gpu_device_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    return create_info;
  }

  void SetUpDevice() {
    const amdf_status_t status =
        GetCtsDeviceCache().GetGpuDevice(endpoint_, &device_);
    // The installed native ABI is qualified only at explicit activation.
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native GPU activation is unavailable for this lifetime";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK) << "domain=" << amdf_status_domain(status)
                                      << " code=" << amdf_status_code(status);
  }

  // Core table borrowed from the CTS provider.
  const amdf_api_t* api_ = nullptr;
  // GPU table borrowed from the CTS provider.
  const amdf_gpu_api_t* gpu_api_ = nullptr;
  // Shared instance used by all device tests.
  amdf_instance_t* instance_ = nullptr;
  // Shared endpoint selected without activating a device.
  amdf_endpoint_t* endpoint_ = nullptr;
  // Shared device materialized only by tests that require one.
  amdf_device_t* device_ = nullptr;
  // Case-owned sibling for the explicit native-device recreation test.
  amdf_device_t* second_device_ = nullptr;
};

TEST_F(GpuEndpointTest, ReturnsQualifiedCachedProfile) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status))
      << "domain=" << amdf_status_domain(open_status)
      << " code=" << amdf_status_code(open_status);
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }

  amdf_gpu_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  const amdf_status_t status = gpu_api_->endpoint_query_info(endpoint_, &info);
  if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
      amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    GTEST_SKIP() << "GPU endpoint is not qualified by this provider";
  }
  ASSERT_TRUE(amdf_status_is_ok(status))
      << "domain=" << amdf_status_domain(status)
      << " code=" << amdf_status_code(status);

  EXPECT_GT(info.gfx_ip.major, 0u);
  EXPECT_TRUE(info.compute.wavefront_size == 32u ||
              info.compute.wavefront_size == 64u);
  EXPECT_GT(info.compute.compute_unit_count, 0u);
  EXPECT_GT(info.compute.maximum_wave_count_per_compute_unit, 0u);
  EXPECT_GT(info.compute.maximum_scratch_wave_count_per_compute_unit, 0u);
  EXPECT_LE(info.compute.maximum_scratch_wave_count_per_compute_unit,
            info.compute.maximum_wave_count_per_compute_unit);
  EXPECT_GT(info.compute.local_data_share_byte_length, 0u);
  EXPECT_GT(info.topology.xcc_count, 0u);
  EXPECT_GT(info.topology.shader_engine_count_per_xcc, 0u);
  RecordProperty("native_unique_id", std::to_string(info.unique_id));

  amdf_gpu_endpoint_info_t second_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(second_info),
  };
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->endpoint_query_info(endpoint_, &second_info)));
  EXPECT_EQ(std::memcmp(&info, &second_info, sizeof(info)), 0);

  // Earlier consumers receive their complete prefix. An incomplete appended
  // field, and all bytes beyond the caller's extent, remain untouched.
  const uint32_t identity_offset =
      offsetof(amdf_gpu_endpoint_info_t, unique_id);
  const uint32_t resources_offset =
      offsetof(amdf_gpu_endpoint_info_t, execution_resources);
  for (uint32_t size : {identity_offset, resources_offset - 1, resources_offset,
                        uint32_t(sizeof(info) - 1)}) {
    amdf_gpu_endpoint_info_t prefix;
    std::memset(&prefix, 0xa5, sizeof(prefix));
    prefix.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
    prefix.structure_size = size;
    prefix.next = nullptr;
    ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &prefix),
              AMDF_STATUS_OK);
    EXPECT_EQ(prefix.structure_size, size);
    prefix.structure_size = info.structure_size;
    const uint32_t copied_size =
        size >= resources_offset ? resources_offset : identity_offset;
    EXPECT_EQ(std::memcmp(&prefix, &info, copied_size), 0);
    const auto* bytes = reinterpret_cast<const uint8_t*>(&prefix);
    for (size_t i = copied_size; i < sizeof(prefix); ++i) {
      EXPECT_EQ(bytes[i], 0xa5) << "size=" << size << " offset=" << i;
    }
  }
}

TEST_F(GpuEndpointTest, RejectsMalformedOutputWithoutMutation) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }

  EXPECT_EQ(amdf_status_code(gpu_api_->endpoint_query_info(endpoint_, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_gpu_endpoint_info_t info = {
      .structure_size = sizeof(info),
  };
  info.gfx_ip.major = UINT32_MAX;
  EXPECT_EQ(amdf_status_code(gpu_api_->endpoint_query_info(endpoint_, &info)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(info.gfx_ip.major, UINT32_MAX);

  info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  info.next = &info;
  EXPECT_EQ(amdf_status_code(gpu_api_->endpoint_query_info(endpoint_, &info)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(info.gfx_ip.major, UINT32_MAX);
}

TEST_F(GpuEndpointTest, RejectsXdnaEndpointWithoutMutation) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_XDNA, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no XDNA endpoint present";
  }

  amdf_gpu_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  info.gfx_ip.major = UINT32_MAX;
  const amdf_status_t status = gpu_api_->endpoint_query_info(endpoint_, &info);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(info.gfx_ip.major, UINT32_MAX);
}

TEST_F(GpuEndpointTest, ValidatesDeviceCreationArgumentsWithoutNativeWork) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }

  amdf_gpu_device_create_info_t create_info = MakeDeviceCreateInfo();
  amdf_device_t* output = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
  EXPECT_EQ(
      amdf_status_code(gpu_api_->device_create(nullptr, &create_info, &output)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeDeviceCreateInfo();
  create_info.reserved = 1;
  EXPECT_EQ(amdf_status_code(
                gpu_api_->device_create(endpoint_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  create_info = MakeDeviceCreateInfo();
  EXPECT_EQ(
      amdf_status_code(gpu_api_->device_create(endpoint_, nullptr, &output)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  EXPECT_EQ(amdf_status_code(
                gpu_api_->device_create(endpoint_, &create_info, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  create_info.type = AMDF_STRUCTURE_TYPE_NONE;
  EXPECT_EQ(amdf_status_code(
                gpu_api_->device_create(endpoint_, &create_info, &output)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});

  create_info = MakeDeviceCreateInfo();
  create_info.next = &create_info;
  EXPECT_EQ(amdf_status_code(
                gpu_api_->device_create(endpoint_, &create_info, &output)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(GpuEndpointTest, RejectsXdnaEndpointForDeviceCreation) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_XDNA, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no XDNA endpoint present";
  }

  const amdf_gpu_device_create_info_t create_info = MakeDeviceCreateInfo();
  amdf_device_t* output = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
  const amdf_status_t status =
      gpu_api_->device_create(endpoint_, &create_info, &output);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(GpuEndpointTest, MaterializesProgramIndependentDevice) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }

  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) {
    return;
  }
  ASSERT_NE(device_, nullptr);

  amdf_gpu_device_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_TRUE(amdf_status_is_ok(gpu_api_->device_query_info(device_, &info)));
  EXPECT_NE(info.id.words[0] | info.id.words[1], 0u);
  EXPECT_EQ(info.reset_epoch, 1u);
  amdf_gpu_device_info_t second_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(second_info),
  };
  ASSERT_TRUE(
      amdf_status_is_ok(gpu_api_->device_query_info(device_, &second_info)));
  EXPECT_EQ(std::memcmp(&info, &second_info, sizeof(info)), 0);

  EXPECT_EQ(amdf_status_code(api_->endpoint_close(endpoint_)),
            AMDF_STATUS_CODE_BUSY);
}

TEST_F(GpuEndpointTest, RejectsClockQueryArgumentsWithoutPublishingOutput) {
  bool engine_found = false;
  ASSERT_EQ(OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found), AMDF_STATUS_OK);
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }
  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) {
    return;
  }
  EXPECT_EQ(gpu_api_->device_query_queue_clock_info(device_, 0, nullptr),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  const amdf_observation_clock_info_t sentinel = {UINT64_MAX, UINT32_MAX, 17};
  amdf_observation_clock_info_t info = sentinel;
  EXPECT_EQ(gpu_api_->device_query_queue_clock_info(device_, UINT32_MAX, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(std::memcmp(&info, &sentinel, sizeof(info)), 0);
  EXPECT_EQ(gpu_api_->device_query_clock_sample(device_, 0, nullptr),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  amdf_observation_clock_sample_t sample_sentinel;
  std::memset(&sample_sentinel, 0x5a, sizeof(sample_sentinel));
  auto sample = sample_sentinel;
  EXPECT_EQ(gpu_api_->device_query_clock_sample(device_, UINT32_MAX, &sample),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(std::memcmp(&sample, &sample_sentinel, sizeof(sample)), 0);
}

TEST_F(GpuEndpointTest, RejectsMalformedDeviceInfoWithoutMutation) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }
  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) {
    return;
  }

  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(device_, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(nullptr, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_gpu_device_info_t info = {
      .structure_size = sizeof(info),
      .reset_epoch = UINT64_MAX,
  };
  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(device_, &info)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(info.reset_epoch, UINT64_MAX);

  info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  info.next = &info;
  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(device_, &info)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(info.reset_epoch, UINT64_MAX);
}

TEST_F(GpuEndpointTest, CreatesReclaimableDevicesFromOneEndpoint) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }
  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) {
    return;
  }
  amdf_gpu_device_info_t first_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(first_info),
  };
  ASSERT_TRUE(
      amdf_status_is_ok(gpu_api_->device_query_info(device_, &first_info)));
  if (!(first_info.features & AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION)) {
    GTEST_SKIP() << "device recreation is unavailable";
  }
  uint32_t scope_count = 0;
  const auto count_status =
      api_->device_enumerate_memory_scopes(device_, 0, nullptr, &scope_count);
  ASSERT_EQ(count_status,
            scope_count == 0
                ? AMDF_STATUS_OK
                : amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  std::vector<amdf_memory_scope_t*> first_scopes(scope_count);
  ASSERT_EQ(api_->device_enumerate_memory_scopes(
                device_, scope_count, first_scopes.data(), &scope_count),
            AMDF_STATUS_OK);
  std::vector<amdf_memory_scope_info_t> scope_infos(scope_count);
  for (uint32_t i = 0; i < scope_count; ++i) {
    scope_infos[i] = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope_infos[i]),
    };
    ASSERT_EQ(api_->memory_scope_query_info(first_scopes[i], &scope_infos[i]),
              AMDF_STATUS_OK);
    EXPECT_GT(scope_infos[i].capacity_byte_length, 0u);
  }
  const amdf_gpu_device_create_info_t create_info = MakeDeviceCreateInfo();
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->device_create(endpoint_, &create_info, &second_device_)));
  amdf_gpu_device_info_t second_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(second_info),
  };
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->device_query_info(second_device_, &second_info)));
  EXPECT_FALSE(amdf_device_id_is_equal(&first_info.id, &second_info.id));

  std::vector<amdf_memory_scope_t*> second_scopes(scope_count);
  uint32_t second_scope_count = 0;
  ASSERT_EQ(api_->device_enumerate_memory_scopes(second_device_, scope_count,
                                                 second_scopes.data(),
                                                 &second_scope_count),
            AMDF_STATUS_OK);
  EXPECT_EQ(second_scope_count, scope_count);
  EXPECT_EQ(first_scopes, second_scopes);

  ASSERT_TRUE(amdf_status_is_ok(api_->device_destroy(second_device_)));
  second_device_ = nullptr;
  for (uint32_t i = 0; i < scope_count; ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->memory_scope_query_info(second_scopes[i], &info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.kind, scope_infos[i].kind);
    EXPECT_EQ(info.flags, scope_infos[i].flags);
    EXPECT_EQ(info.capacity_byte_length, scope_infos[i].capacity_byte_length);
  }
}

}  // namespace
