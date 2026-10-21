// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/gpu/cts/test_device.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libamdf/cts/util/native_identity.h"

namespace {

using iree::hal::amd::gpu::cts::DeviceAllocator;

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("rdna"), &driver_));
    ASSERT_NO_FATAL_FAILURE(ResolveSelectedGpu());
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, IREE_SV("rdna"), iree_allocator_system(), &count_, &infos_));
    bool selected_visible = false;
    for (iree_host_size_t i = 0; i < count_; ++i) {
      if (!selected_path_[0] ||
          iree_string_view_equal(infos_[i].path,
                                 iree_make_cstring_view(selected_path_))) {
        ordinal_ = i;
        selected_visible = true;
        break;
      }
    }
    if (!selected_visible) {
      if (selected_path_[0] &&
          selected_architecture_ == AMDF_GPU_ARCHITECTURE_RDNA) {
        FAIL() << "Runner-selected RDNA GPU is unavailable through rdna";
      }
      GTEST_SKIP() << "Selected GPU is unavailable through rdna";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &proactors_));
  }

  void TearDown() override {
    for (size_t i = 0; i < devices_.size(); ++i) {
      iree_hal_semaphore_release(semaphores_[i]);
      iree_hal_device_release(devices_[i]);
      EXPECT_EQ(allocators_[i].live_count.load(), 0u);
    }
    iree_async_proactor_pool_release(proactors_);
    iree_allocator_free(iree_allocator_system(), infos_);
    iree_hal_driver_release(driver_);
    iree_hal_driver_registry_free(registry_);
    iree_hal_amd_fabric_release(discovery_);
  }

  void ResolveSelectedGpu() {
    const char* selected_text = std::getenv("AMDF_CTS_GPU_NATIVE_IDENTITY");
    if (!selected_text) {
      return;
    }
    amdf_endpoint_native_identity_t selected = {};
    ASSERT_TRUE(amdf::cts::ParseNativeIdentity(selected_text, &selected))
        << "invalid AMDF_CTS_GPU_NATIVE_IDENTITY: " << selected_text;

    const amdf_api_t* api = nullptr;
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api),
        "amdf_query_api"));
    IREE_ASSERT_OK(
        iree_hal_amd_fabric_create(api, AMDF_NATIVE_LIFETIME_INSTANCE,
                                   iree_allocator_system(), &discovery_));
    const void* extension = nullptr;
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        api->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                             AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
        "query_extension(GPU)"));
    const auto* gpu = static_cast<const amdf_gpu_api_t*>(extension);
    for (uint32_t i = 0; i < discovery_->endpoint_count; ++i) {
      const amdf_endpoint_summary_t* summary = &discovery_->endpoints[i];
      if (summary->engine_kind != AMDF_ENGINE_KIND_GPU) {
        continue;
      }
      amdf_endpoint_t* endpoint = nullptr;
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          api->endpoint_open(discovery_->instance, &summary->id, &endpoint),
          "endpoint_open"));
      amdf_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
          .structure_size = sizeof(info),
      };
      iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          api->endpoint_query_info(endpoint, &info), "endpoint_query_info");
      const bool matches =
          iree_status_is_ok(status) &&
          amdf::cts::NativeIdentityIsEqual(info.native_identity, selected);
      amdf_gpu_endpoint_info_t gpu_info = {
          .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
          .structure_size = sizeof(gpu_info),
      };
      if (matches) {
        status = iree_status_join(
            status, IREE_HAL_AMD_STATUS_FROM_AMDF(
                        gpu->endpoint_query_info(endpoint, &gpu_info),
                        "gpu.endpoint_query_info"));
      }
      const amdf_status_t close_status = api->endpoint_close(endpoint);
      if (!amdf_status_is_ok(close_status)) {
        iree_hal_amd_fabric_retain(discovery_);
      }
      IREE_ASSERT_OK(iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(close_status, "endpoint_close")));
      if (!matches) {
        continue;
      }
      selected_architecture_ = amdf_gpu_endpoint_architecture(&gpu_info);
      std::snprintf(selected_path_, sizeof(selected_path_),
                    "id/%016" PRIx64 "%016" PRIx64, summary->id.words[0],
                    summary->id.words[1]);
      return;
    }
    FAIL() << "AMDF_CTS_GPU_NATIVE_IDENTITY does not identify an enumerated "
              "GPU: "
           << selected_text;
  }

  void Create(size_t index, iree_string_view_t path,
              iree_string_view_t scheme = IREE_SV("rdna")) {
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    const iree_string_pair_t options[] = {
        {.key = IREE_SV("queue_count"), .value = IREE_SV("2")}};
    iree::Status status = iree_hal_driver_create_device_by_path(
        driver_, scheme, path, IREE_ARRAYSIZE(options), options, &params,
        allocators_[index].allocator(), &devices_[index]);
    IREE_ASSERT_OK(status);
    EXPECT_TRUE(iree_string_view_equal(iree_hal_device_id(devices_[index]),
                                       IREE_SV("rdna")));
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 0), nullptr);
    ASSERT_NE(iree_hal_device_queue(devices_[index], 0, 1), nullptr);
  }

  void Run(size_t index) {
    iree_hal_device_t* device = devices_[index];
    iree_hal_semaphore_t** semaphore = &semaphores_[index];
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, semaphore));
    uint64_t first = 1, second = 2;
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 1), {1, semaphore, &first},
        {1, semaphore, &second}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        iree_hal_device_queue(device, 0, 0), iree_hal_semaphore_list_empty(),
        {1, semaphore, &first}, nullptr, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(*semaphore, second,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Public registry retaining the canonical AMD driver until explicitly freed.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Independent reference acquired through the RDNA scheme.
  iree_hal_driver_t* driver_ = nullptr;
  // Query-only owner used to resolve the runner's selected physical GPU.
  iree_hal_amd_fabric_t* discovery_ = nullptr;
  // Exact endpoint path selected by AMDF_CTS_GPU_NATIVE_IDENTITY, when set.
  char selected_path_[36] = {};
  // Architecture of the selected endpoint, established by passive discovery.
  amdf_gpu_architecture_t selected_architecture_ =
      AMDF_GPU_ARCHITECTURE_UNKNOWN;
  // Dense position of the selected endpoint in its scheme-local view.
  iree_host_size_t ordinal_ = 0;
  // Public discovery rows, independent of native execution resources.
  iree_hal_device_info_t* infos_ = nullptr;
  // Number of discovery rows in the selected scheme.
  iree_host_size_t count_ = 0;
  // Process progress service supplied to public device construction.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Device ownership includes asynchronous final worker cleanup.
  std::array<DeviceAllocator, 2> allocators_;
  // Independently provisioned devices sharing one native fabric.
  std::array<iree_hal_device_t*, 2> devices_ = {};
  // Cross-queue dependencies and final completions observed by this test.
  std::array<iree_hal_semaphore_t*, 2> semaphores_ = {};
};

TEST_F(DriverTest, DefaultAndNativeIdentityRetainOneOwner) {
  iree_hal_driver_t* xdna_driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry_, IREE_SV("xdna"), &xdna_driver));
  EXPECT_EQ(driver_, xdna_driver);
  iree_hal_driver_release(xdna_driver);
  iree_hal_driver_t* gpu_driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
      registry_, IREE_SV("amdgpu2"), &gpu_driver));
  EXPECT_EQ(driver_, gpu_driver);
  iree_hal_driver_release(gpu_driver);

  iree_host_size_t gpu_count = 0;
  iree_hal_device_info_t* gpu_infos = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, IREE_SV("amdgpu2"), iree_allocator_system(), &gpu_count,
      &gpu_infos));
  iree_string_view_t gpu_path = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < gpu_count; ++i) {
    if (gpu_infos[i].device_id == infos_[ordinal_].device_id) {
      gpu_path = gpu_infos[i].path;
      break;
    }
  }
  std::string gpu_path_storage;
  if (!iree_string_view_is_empty(gpu_path)) {
    gpu_path_storage.assign(gpu_path.data, gpu_path.size);
  }
  iree_allocator_free(iree_allocator_system(), gpu_infos);
  ASSERT_FALSE(gpu_path_storage.empty())
      << "Selected RDNA GPU is unavailable through amdgpu2";

  auto params = iree_hal_device_create_params_default();
  params.proactor_pool = proactors_;
  const iree_string_pair_t invalid[] = {
      {.key = IREE_SV("columns"), .value = IREE_SV("1")}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_driver_create_device_by_id(
          driver_, infos_[ordinal_].device_id, IREE_ARRAYSIZE(invalid), invalid,
          &params, allocators_[0].allocator(), &devices_[0]));
  ASSERT_EQ(devices_[0], nullptr);

  // The registry selects PROCESS native lifetime for host registration. Both
  // devices must share that owner: KFD cannot reacquire a destroyed primary VM.
  char ordinal[32];
  std::snprintf(ordinal, sizeof(ordinal), "%" PRIhsz, ordinal_);
  Create(0, ordinal_ ? iree_make_cstring_view(ordinal) : IREE_SV(""));
  if (HasFatalFailure()) {
    return;
  }
  Create(1, iree_make_cstring_view(gpu_path_storage.c_str()),
         IREE_SV("amdgpu2"));
  if (HasFatalFailure()) {
    return;
  }
  iree_hal_driver_release(driver_);
  driver_ = nullptr;
  iree_hal_driver_registry_free(registry_);
  registry_ = nullptr;
  Run(0);
  Run(1);
}

}  // namespace
