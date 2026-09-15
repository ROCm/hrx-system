// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

const amdf_api_t* QueryApi() {
  const amdf_api_t* api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(amdf_cts_provider_query_api()(
      AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api)));
  return api;
}

const amdf_gpu_api_t* QueryGpuApi(const amdf_api_t* api) {
  const void* extension_api = nullptr;
  EXPECT_TRUE(amdf_status_is_ok(
      api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                           AMDF_GPU_EXTENSION_VERSION_LATEST, &extension_api)));
  return static_cast<const amdf_gpu_api_t*>(extension_api);
}

TEST(GpuExtensionTest, ReportsCompiledAvailabilityBeforeCreatingInstance) {
  const amdf_api_t* api = QueryApi();
  ASSERT_NE(api, nullptr);

  const amdf_gpu_api_t* gpu_api = QueryGpuApi(api);

  ASSERT_NE(gpu_api, nullptr);
  EXPECT_EQ(gpu_api->structure_size, sizeof(amdf_gpu_api_t));
  EXPECT_EQ(gpu_api->extension_version, AMDF_GPU_EXTENSION_VERSION_1);
  EXPECT_NE(gpu_api->endpoint_query_info, nullptr);
  EXPECT_NE(gpu_api->device_create, nullptr);
  EXPECT_NE(gpu_api->device_query_info, nullptr);
  EXPECT_NE(gpu_api->kernel_queue_create, nullptr);
  EXPECT_NE(gpu_api->kernel_queue_submit, nullptr);
  EXPECT_NE(gpu_api->endpoint_query_device_capabilities, nullptr);
  EXPECT_NE(gpu_api->user_queue_create, nullptr);
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

  amdf_status_t QueryDeviceCapabilities() {
    amdf_gpu_device_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    return gpu_api_->endpoint_query_device_capabilities(endpoint_,
                                                        &capabilities);
  }

  amdf_gpu_device_create_info_t MakeDeviceCreateInfo() {
    amdf_gpu_device_create_info_t create_info = {};
    create_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO;
    create_info.structure_size = sizeof(create_info);
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

  amdf_gpu_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
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

  amdf_gpu_endpoint_info_t second_info = {};
  second_info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  second_info.structure_size = sizeof(second_info);
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->endpoint_query_info(endpoint_, &second_info)));
  EXPECT_EQ(std::memcmp(&info, &second_info, sizeof(info)), 0);
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

  amdf_gpu_endpoint_info_t info = {};
  info.structure_size = sizeof(info);
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

  amdf_gpu_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
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

  const amdf_status_t lifetime_status = QueryDeviceCapabilities();
  if (lifetime_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "requested native lifetime is unavailable";
  }
  ASSERT_EQ(lifetime_status, AMDF_STATUS_OK);

  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) return;
  ASSERT_NE(device_, nullptr);

  amdf_gpu_device_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  info.structure_size = sizeof(info);
  ASSERT_TRUE(amdf_status_is_ok(gpu_api_->device_query_info(device_, &info)));
  EXPECT_NE(info.id.words[0] | info.id.words[1], 0u);
  EXPECT_EQ(info.reset_epoch, 1u);
  amdf_gpu_device_info_t second_info = {};
  second_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  second_info.structure_size = sizeof(second_info);
  ASSERT_TRUE(
      amdf_status_is_ok(gpu_api_->device_query_info(device_, &second_info)));
  EXPECT_EQ(std::memcmp(&info, &second_info, sizeof(info)), 0);

  EXPECT_EQ(amdf_status_code(api_->endpoint_close(endpoint_)),
            AMDF_STATUS_CODE_BUSY);
}

TEST_F(GpuEndpointTest, RejectsMalformedDeviceInfoWithoutMutation) {
  bool engine_found = false;
  const amdf_status_t open_status =
      OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found);
  ASSERT_TRUE(amdf_status_is_ok(open_status));
  if (!engine_found) {
    GTEST_SKIP() << "no GPU endpoint present";
  }
  const amdf_status_t lifetime_status = QueryDeviceCapabilities();
  if (lifetime_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "requested native lifetime is unavailable";
  }
  ASSERT_EQ(lifetime_status, AMDF_STATUS_OK);

  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) return;

  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(device_, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(gpu_api_->device_query_info(nullptr, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);

  amdf_gpu_device_info_t info = {};
  info.structure_size = sizeof(info);
  info.reset_epoch = UINT64_MAX;
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
  const amdf_status_t lifetime_status = QueryDeviceCapabilities();
  if (lifetime_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "requested native lifetime is unavailable";
  }
  ASSERT_EQ(lifetime_status, AMDF_STATUS_OK);

  ASSERT_NO_FATAL_FAILURE(SetUpDevice());
  if (IsSkipped()) return;
  amdf_gpu_device_info_t first_info = {};
  first_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  first_info.structure_size = sizeof(first_info);
  ASSERT_TRUE(
      amdf_status_is_ok(gpu_api_->device_query_info(device_, &first_info)));
  if (!(first_info.features & AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION)) {
    GTEST_SKIP() << "device recreation is unavailable";
  }
  const amdf_gpu_device_create_info_t create_info = MakeDeviceCreateInfo();
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->device_create(endpoint_, &create_info, &second_device_)));
  amdf_gpu_device_info_t second_info = {};
  second_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
  second_info.structure_size = sizeof(second_info);
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->device_query_info(second_device_, &second_info)));
  EXPECT_FALSE(amdf_device_id_is_equal(&first_info.id, &second_info.id));

  ASSERT_TRUE(amdf_status_is_ok(api_->device_destroy(second_device_)));
  second_device_ = nullptr;
}

TEST_F(GpuEndpointTest, QueriesInstanceCapabilitiesWithoutCreatingDevice) {
  bool engine_found = false;
  ASSERT_EQ(OpenEngine(AMDF_ENGINE_KIND_GPU, &engine_found), AMDF_STATUS_OK);
  if (!engine_found) GTEST_SKIP() << "No GPU endpoint";

  amdf_gpu_device_capabilities_t capabilities = {};
  capabilities.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CAPABILITIES;
  capabilities.structure_size = sizeof(capabilities);
  capabilities.features = UINT64_MAX;
  EXPECT_EQ(amdf_status_code(gpu_api_->endpoint_query_device_capabilities(
                nullptr, &capabilities)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(capabilities.features, UINT64_MAX);
  capabilities.next = &capabilities;
  EXPECT_EQ(amdf_status_code(gpu_api_->endpoint_query_device_capabilities(
                endpoint_, &capabilities)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(capabilities.features, UINT64_MAX);
  capabilities.next = nullptr;

  const amdf_status_t status =
      gpu_api_->endpoint_query_device_capabilities(endpoint_, &capabilities);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    EXPECT_EQ(capabilities.features, UINT64_MAX);
    amdf_gpu_device_create_info_t create_info = MakeDeviceCreateInfo();
    amdf_device_t* output = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
    EXPECT_EQ(gpu_api_->device_create(endpoint_, &create_info, &output),
              status);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
  } else {
    ASSERT_EQ(status, AMDF_STATUS_OK);
    amdf_gpu_device_capabilities_t second = capabilities;
    ASSERT_EQ(gpu_api_->endpoint_query_device_capabilities(endpoint_, &second),
              AMDF_STATUS_OK);
    EXPECT_EQ(second.features, capabilities.features);
  }
}

}  // namespace
