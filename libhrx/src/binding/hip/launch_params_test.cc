// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/launch_params.h"

#include <array>
#include <cstddef>

#include "iree/testing/gtest.h"

namespace {

void InitializeLaunchDevice(iree_hal_streaming_device_t* device) {
  device->max_grid_dim[0] = 1024;
  device->max_grid_dim[1] = 1024;
  device->max_grid_dim[2] = 1024;
  device->max_block_dim[0] = 1024;
  device->max_block_dim[1] = 1024;
  device->max_block_dim[2] = 64;
  device->max_threads_per_block = 1024;
  device->max_shared_memory_per_block = 64 * 1024;
}

void InitializeLaunchSymbol(iree_hal_streaming_symbol_t* symbol,
                            uint32_t maximum_threads_per_block,
                            uint32_t dynamic_shared_memory_size) {
  symbol->function_attributes.provided_flags =
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK |
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;
  symbol->function_attributes.maximum_threads_per_block =
      maximum_threads_per_block;
  symbol->function_attributes.maximum_configurable_dynamic_shared_memory_size =
      dynamic_shared_memory_size;
  iree_atomic_store(
      &symbol->function_attributes.configured_dynamic_shared_memory_size,
      dynamic_shared_memory_size, iree_memory_order_relaxed);
}

TEST(LaunchParamsTest, ParseLaunchExtraAcceptsBufferAndSize) {
  std::array<uint8_t, 16> buffer = {};
  size_t buffer_size = buffer.size();
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      buffer.data(),
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      HIP_LAUNCH_PARAM_END,
  };

  void* out_buffer = nullptr;
  size_t out_buffer_size = 0;
  EXPECT_EQ(hipSuccess,
            iree_hip_parse_launch_extra(extra, &out_buffer, &out_buffer_size));
  EXPECT_EQ(buffer.data(), out_buffer);
  EXPECT_EQ(buffer.size(), out_buffer_size);
}

TEST(LaunchParamsTest, ParseLaunchExtraPreservesZeroLengthSpan) {
  uint8_t buffer = 0;
  size_t buffer_size = 0;
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &buffer,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      HIP_LAUNCH_PARAM_END,
  };

  void* out_buffer = nullptr;
  size_t out_buffer_size = 1;
  EXPECT_EQ(hipSuccess,
            iree_hip_parse_launch_extra(extra, &out_buffer, &out_buffer_size));
  EXPECT_EQ(&buffer, out_buffer);
  EXPECT_EQ(0u, out_buffer_size);
}

TEST(LaunchParamsTest, ParseLaunchExtraRejectsMissingPair) {
  std::array<uint8_t, 16> buffer = {};
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      buffer.data(),
      HIP_LAUNCH_PARAM_END,
  };

  void* out_buffer = nullptr;
  size_t out_buffer_size = 0;
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_parse_launch_extra(extra, &out_buffer, &out_buffer_size));
}

TEST(LaunchParamsTest, ParseLaunchExtraRejectsDuplicateKeys) {
  std::array<uint8_t, 16> buffer = {};
  size_t buffer_size = buffer.size();
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      buffer.data(),
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      buffer.data(),
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      HIP_LAUNCH_PARAM_END,
  };

  void* out_buffer = nullptr;
  size_t out_buffer_size = 0;
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_parse_launch_extra(extra, &out_buffer, &out_buffer_size));
}

TEST(LaunchParamsTest, ParseLaunchExtraRejectsMarkerValues) {
  size_t buffer_size = 16;
  void* extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      HIP_LAUNCH_PARAM_END,
  };

  void* out_buffer = nullptr;
  size_t out_buffer_size = 0;
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_parse_launch_extra(extra, &out_buffer, &out_buffer_size));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationAcceptsDeviceLimits) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);
  iree_hal_streaming_symbol_t symbol = {};
  InitializeLaunchSymbol(&symbol, /*maximum_threads_per_block=*/512,
                         /*dynamic_shared_memory_size=*/32 * 1024);

  EXPECT_EQ(hipSuccess,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/32, /*grid_dim_y=*/8,
                /*grid_dim_z=*/1, /*block_dim_x=*/16, /*block_dim_y=*/16,
                /*block_dim_z=*/2, /*shared_memory_bytes=*/1024));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationRejectsInvalidDimensions) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);

  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, nullptr, /*grid_dim_x=*/0, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, nullptr, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1025, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationRejectsResourceExcess) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);
  iree_hal_streaming_symbol_t symbol = {};
  InitializeLaunchSymbol(&symbol, /*maximum_threads_per_block=*/256,
                         /*dynamic_shared_memory_size=*/4096);

  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/512, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/128, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/8192));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationHonorsKnownZeroLimit) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);
  iree_hal_streaming_symbol_t symbol = {};
  InitializeLaunchSymbol(&symbol, /*maximum_threads_per_block=*/256,
                         /*dynamic_shared_memory_size=*/0);

  EXPECT_EQ(hipSuccess,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/1));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationHonorsOptinFunctionLimit) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);
  iree_hal_streaming_symbol_t symbol = {};
  InitializeLaunchSymbol(&symbol, /*maximum_threads_per_block=*/256,
                         /*dynamic_shared_memory_size=*/96 * 1024);

  EXPECT_EQ(hipSuccess,
            iree_hip_validate_launch_configuration(
                &device, &symbol, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/80 * 1024));
}

TEST(LaunchParamsTest,
     ValidateLaunchConfigurationRejectsUnrepresentableSharedMemory) {
  iree_hal_streaming_device_t device = {};
  InitializeLaunchDevice(&device);
  device.max_shared_memory_per_block = UINT32_MAX;

  EXPECT_EQ(hipSuccess,
            iree_hip_validate_launch_configuration(
                &device, nullptr, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/UINT32_MAX));
  EXPECT_EQ(hipErrorInvalidConfiguration,
            iree_hip_validate_launch_configuration(
                &device, nullptr, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1,
                /*shared_memory_bytes=*/(size_t)UINT32_MAX + 1));
}

TEST(LaunchParamsTest, ValidateLaunchConfigurationRejectsMissingDevice) {
  EXPECT_EQ(hipErrorInvalidDevice,
            iree_hip_validate_launch_configuration(
                nullptr, nullptr, /*grid_dim_x=*/1, /*grid_dim_y=*/1,
                /*grid_dim_z=*/1, /*block_dim_x=*/1, /*block_dim_y=*/1,
                /*block_dim_z=*/1, /*shared_memory_bytes=*/0));
}

}  // namespace
