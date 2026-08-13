// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/function_attributes.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

iree_hal_device_spec_t* CreateDeviceSpec(
    uint32_t maximum_workgroup_invocations,
    uint64_t maximum_workgroup_local_memory_size,
    uint64_t maximum_workgroup_local_memory_size_optin) {
  iree_hal_device_dispatch_spec_t dispatch = {
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/
          maximum_workgroup_invocations,
      },
      /*.subgroup=*/{},
      /*.execution=*/
      {
          /*.unit_count=*/0,
          /*.group_count=*/0,
          /*.maximum_resident_workgroup_count=*/0,
          /*.maximum_resident_invocation_count=*/0,
          /*.maximum_resident_subgroup_count=*/0,
          /*.maximum_register_count=*/0,
          /*.maximum_workgroup_register_count=*/0,
          /*.maximum_local_memory_size=*/0,
          /*.maximum_workgroup_local_memory_size=*/
          maximum_workgroup_local_memory_size,
          /*.maximum_workgroup_local_memory_size_optin=*/
          maximum_workgroup_local_memory_size_optin,
      },
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/nullptr,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/nullptr,
      /*.dispatch=*/&dispatch,
  };
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_CHECK_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                            &device_spec));
  return device_spec;
}

iree_hal_executable_function_info_t MakeFunctionInfo() {
  return iree_hal_executable_function_info_t{
      /*.name=*/IREE_SV("test_kernel"),
      /*.flags=*/IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE,
      /*.constant_byte_length=*/0,
      /*.binding_count=*/0,
      /*.parameter_count=*/0,
      /*.maximum_workgroup_invocations=*/512,
      /*.workgroup_size=*/{},
      /*.resource_usage=*/
      {
          /*.provided_flags=*/
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL,
          /*.fixed_workgroup_local_memory_size=*/4096,
          /*.fixed_private_memory_size=*/64,
          /*.invocation_register_count=*/40,
      },
  };
}

TEST(FunctionAttributesTest, MapsGenericFactsAndDeviceLimits) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/64 * 1024,
                       /*maximum_workgroup_local_memory_size_optin=*/96 * 1024);
  const iree_hal_executable_function_info_t function_info = MakeFunctionInfo();

  iree_hal_streaming_function_attributes_t attributes;
  IREE_ASSERT_OK(iree_hal_streaming_function_attributes_initialize(
      device_spec, &function_info, &attributes));

  EXPECT_EQ(
      attributes.provided_flags,
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK |
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY |
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY |
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT |
          IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY);
  EXPECT_EQ(attributes.maximum_threads_per_block, 512u);
  EXPECT_EQ(attributes.fixed_shared_memory_size, 4096u);
  EXPECT_EQ(attributes.fixed_local_memory_size, 64u);
  EXPECT_EQ(attributes.register_count, 40u);
  EXPECT_EQ(iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
                &attributes),
            60u * 1024u);
  EXPECT_EQ(attributes.maximum_configurable_dynamic_shared_memory_size,
            92u * 1024u);

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, FallsBackToDeviceThreadLimit) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/64 * 1024,
                       /*maximum_workgroup_local_memory_size_optin=*/64 * 1024);
  iree_hal_executable_function_info_t function_info = MakeFunctionInfo();
  function_info.maximum_workgroup_invocations = 0;

  iree_hal_streaming_function_attributes_t attributes;
  IREE_ASSERT_OK(iree_hal_streaming_function_attributes_initialize(
      device_spec, &function_info, &attributes));
  EXPECT_EQ(attributes.maximum_threads_per_block, 1024u);

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, PreservesUnavailableResourceFacts) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/64 * 1024,
                       /*maximum_workgroup_local_memory_size_optin=*/64 * 1024);
  iree_hal_executable_function_info_t function_info = MakeFunctionInfo();
  function_info.resource_usage.provided_flags =
      IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_NONE;

  iree_hal_streaming_function_attributes_t attributes;
  IREE_ASSERT_OK(iree_hal_streaming_function_attributes_initialize(
      device_spec, &function_info, &attributes));
  EXPECT_EQ(
      attributes.provided_flags,
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK);
  EXPECT_EQ(attributes.fixed_shared_memory_size, 0u);
  EXPECT_EQ(attributes.fixed_local_memory_size, 0u);
  EXPECT_EQ(attributes.register_count, 0u);
  EXPECT_EQ(iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
                &attributes),
            0u);

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, DistinguishesKnownZeroDynamicLimit) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/4096,
                       /*maximum_workgroup_local_memory_size_optin=*/4096);
  iree_hal_executable_function_info_t function_info = MakeFunctionInfo();
  function_info.resource_usage.fixed_workgroup_local_memory_size = 4096;

  iree_hal_streaming_function_attributes_t attributes;
  IREE_ASSERT_OK(iree_hal_streaming_function_attributes_initialize(
      device_spec, &function_info, &attributes));
  EXPECT_TRUE(iree_all_bits_set(
      attributes.provided_flags,
      IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY));
  EXPECT_EQ(iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
                &attributes),
            0u);
  EXPECT_TRUE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &attributes, 0));
  EXPECT_FALSE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &attributes, 1));

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, SetterUsesOptinCeiling) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/64 * 1024,
                       /*maximum_workgroup_local_memory_size_optin=*/96 * 1024);
  const iree_hal_executable_function_info_t function_info = MakeFunctionInfo();

  iree_hal_streaming_function_attributes_t attributes;
  IREE_ASSERT_OK(iree_hal_streaming_function_attributes_initialize(
      device_spec, &function_info, &attributes));
  EXPECT_TRUE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &attributes, 92 * 1024));
  EXPECT_EQ(iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
                &attributes),
            92u * 1024u);
  EXPECT_FALSE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &attributes, 92 * 1024 + 1));
  EXPECT_EQ(iree_hal_streaming_function_attributes_dynamic_shared_memory_size(
                &attributes),
            92u * 1024u);

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, RejectsFunctionThreadLimitBeyondDevice) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/256,
                       /*maximum_workgroup_local_memory_size=*/64 * 1024,
                       /*maximum_workgroup_local_memory_size_optin=*/64 * 1024);
  const iree_hal_executable_function_info_t function_info = MakeFunctionInfo();

  iree_hal_streaming_function_attributes_t attributes;
  EXPECT_THAT(Status(iree_hal_streaming_function_attributes_initialize(
                  device_spec, &function_info, &attributes)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_device_spec_release(device_spec);
}

TEST(FunctionAttributesTest, RejectsFixedSharedMemoryBeyondDevice) {
  iree_hal_device_spec_t* device_spec =
      CreateDeviceSpec(/*maximum_workgroup_invocations=*/1024,
                       /*maximum_workgroup_local_memory_size=*/2048,
                       /*maximum_workgroup_local_memory_size_optin=*/2048);
  const iree_hal_executable_function_info_t function_info = MakeFunctionInfo();

  iree_hal_streaming_function_attributes_t attributes;
  EXPECT_THAT(Status(iree_hal_streaming_function_attributes_initialize(
                  device_spec, &function_info, &attributes)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_device_spec_release(device_spec);
}

}  // namespace
