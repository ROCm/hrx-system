// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

TEST(AmdgpuDriverOptionsTest, DriverParamsAreRejectedUntilDefined) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  const iree_string_pair_t pair = iree_make_cstring_pair("unknown", "value");

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_options_parse(&options, (iree_string_pair_list_t){
                                                         /*.count=*/1,
                                                         /*.pairs=*/&pair,
                                                     }));
}

TEST(AmdgpuDriverOptionsTest, LogicalDeviceParamsAreRejectedUntilDefined) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  const iree_string_pair_t pair = iree_make_cstring_pair("unknown", "value");

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_logical_device_options_parse(
                            &options, (iree_string_pair_list_t){
                                          /*.count=*/1,
                                          /*.pairs=*/&pair,
                                      }));
}

TEST(AmdgpuDriverOptionsTest, LogicalDeviceDefaultsUseHostCopyPm4Publication) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  EXPECT_EQ(options.pm4_command_buffer_publication_mode,
            IREE_HAL_AMDGPU_PM4_COMMAND_BUFFER_PUBLICATION_MODE_HOST_COPY);
}

TEST(AmdgpuDriverOptionsTest, LogicalDeviceDefaultsDisableAsan) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  EXPECT_FALSE(options.asan.enabled);
  EXPECT_EQ(options.asan.report_policy,
            IREE_HAL_AMDGPU_ASAN_REPORT_POLICY_REPORT_ONLY);
  EXPECT_EQ(options.asan.shadow_mode, IREE_HAL_AMDGPU_ASAN_SHADOW_MODE_SPARSE);
  EXPECT_EQ(options.asan.shadow_backing,
            IREE_HAL_AMDGPU_ASAN_SHADOW_BACKING_DEVICE_LOCAL);
  EXPECT_EQ(options.asan.shadow_scale_shift,
            IREE_HAL_AMDGPU_SHADOW_MAP_DEFAULT_SCALE_SHIFT);
  EXPECT_EQ(options.asan.shadow_size, IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SIZE);
  EXPECT_EQ(options.asan.owned_application_size,
            IREE_HAL_AMDGPU_ASAN_DEFAULT_OWNED_APPLICATION_SIZE);
  EXPECT_EQ(options.asan.shadow_slab_size,
            IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SLAB_SIZE);
  EXPECT_EQ(options.asan.quarantine_size,
            IREE_HAL_AMDGPU_ASAN_DEFAULT_QUARANTINE_SIZE);
}

#if defined(IREE_SANITIZER_THREAD)
TEST(AmdgpuDriverOptionsTest, HostTsanRejectsAsanBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;

  const iree_hal_amdgpu_logical_device_host_compatibility_t compatibility =
      iree_hal_amdgpu_logical_device_options_query_host_compatibility(&options);
  EXPECT_EQ(
      compatibility,
      IREE_HAL_AMDGPU_LOGICAL_DEVICE_HOST_COMPATIBILITY_INCOMPATIBLE_HOST_TSAN_ASAN);
}
#endif  // IREE_SANITIZER_THREAD

TEST(AmdgpuDriverOptionsTest, RejectsMissingSearchPathStorageBeforeLoadingHsa) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  options.libhsa_search_paths = (iree_string_view_list_t){
      /*.count=*/1,
      /*.values=*/NULL,
  };

  iree_hal_driver_t* driver = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_create(IREE_SV("amdgpu"), &options,
                                    iree_allocator_system(), &driver));
  iree_hal_driver_release(driver);
}

TEST(AmdgpuDriverOptionsTest, RejectsMissingSearchPathDataBeforeLoadingHsa) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  const iree_string_view_t search_path = iree_make_string_view(NULL, 1);
  options.libhsa_search_paths = (iree_string_view_list_t){
      /*.count=*/1,
      /*.values=*/&search_path,
  };

  iree_hal_driver_t* driver = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_driver_create(IREE_SV("amdgpu"), &options,
                                    iree_allocator_system(), &driver));
  iree_hal_driver_release(driver);
}

static iree_status_t CreateDriverWithDefaultDeviceOptions(
    const iree_hal_amdgpu_logical_device_options_t* device_options) {
  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  options.default_device_options = *device_options;
  iree_hal_driver_t* driver = NULL;
  iree_status_t status = iree_hal_amdgpu_driver_create(
      IREE_SV("amdgpu"), &options, iree_allocator_system(), &driver);
  iree_hal_driver_release(driver);
  return status;
}

TEST(AmdgpuDriverOptionsTest, RejectsDeviceQueuePlacementBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.queue_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidQueuePlacementBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.queue_placement = (iree_hal_amdgpu_queue_placement_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, AcceptsCommandBufferModesBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;

  IREE_EXPECT_OK(
      iree_hal_amdgpu_logical_device_options_verify_supported_features(
          &options));

  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;

  IREE_EXPECT_OK(
      iree_hal_amdgpu_logical_device_options_verify_supported_features(
          &options));

  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;

  IREE_EXPECT_OK(
      iree_hal_amdgpu_logical_device_options_verify_supported_features(
          &options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidCommandBufferModeBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = (iree_hal_amdgpu_command_buffer_mode_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest,
     RejectsInvalidPm4CommandBufferPublicationBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.pm4_command_buffer_publication_mode =
      (iree_hal_amdgpu_pm4_command_buffer_publication_mode_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsExclusiveExecutionBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.exclusive_execution = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsNegativeActiveWaitBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.wait_active_for_ns = -1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsActiveWaitBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.wait_active_for_ns = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidAsanReportPolicyBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.report_policy = (iree_hal_amdgpu_asan_report_policy_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidAsanShadowModeBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_mode = (iree_hal_amdgpu_asan_shadow_mode_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidAsanShadowBackingBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_backing = (iree_hal_amdgpu_asan_shadow_backing_t)99;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

TEST(AmdgpuDriverOptionsTest, RejectsInvalidAsanGeometryBeforeLoadingHsa) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_size = 1000;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));

  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.owned_application_size = 1000;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));

  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_slab_size = 1000;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        CreateDriverWithDefaultDeviceOptions(&options));

  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.asan.enabled = 1;
  options.asan.shadow_scale_shift = 63;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        CreateDriverWithDefaultDeviceOptions(&options));
}

}  // namespace
}  // namespace iree::hal::amdgpu
