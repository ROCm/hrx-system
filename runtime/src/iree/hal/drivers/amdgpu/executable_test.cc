// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using iree::testing::status::StatusIs;
using namespace test;

class ExecutableTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t ExecutableTest::host_allocator_;
iree_hal_amdgpu_libhsa_t ExecutableTest::libhsa_;
iree_hal_amdgpu_topology_t ExecutableTest::topology_;

TEST_F(ExecutableTest, PublishesAndEnforcesResourceLimits) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];

  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      IREE_SV("command_buffer_dispatch_multi_workgroup_test.bin"),
      executable.out()));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &function_info));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue(
          executable, function, IREE_HAL_QUEUE_AFFINITY_ANY, &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(function_info.maximum_workgroup_invocations,
            descriptor->limits.maximum_workgroup_invocations);
  EXPECT_EQ(function_info.resource_usage.provided_flags,
            IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL);
  EXPECT_EQ(function_info.resource_usage.fixed_workgroup_local_memory_size,
            descriptor->kernel_args.group_segment_size);
  EXPECT_EQ(function_info.resource_usage.fixed_private_memory_size,
            descriptor->kernel_args.private_segment_size);
  EXPECT_GT(function_info.resource_usage.invocation_register_count, 0u);

  const uint32_t maximum_dynamic_workgroup_local_memory_size =
      descriptor->limits.maximum_dynamic_workgroup_local_memory_size;
  ASSERT_GT(maximum_dynamic_workgroup_local_memory_size, 0u);
  ASSERT_LT(maximum_dynamic_workgroup_local_memory_size, UINT32_MAX);
  EXPECT_EQ(maximum_dynamic_workgroup_local_memory_size,
            physical_device->group_segment_max_size -
                descriptor->kernel_args.group_segment_size);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/1,
      /*.values=*/&binding,
  };
  iree_hal_dispatch_config_t config =
      iree_hal_make_static_dispatch_config(1, 1, 1);
  config.dynamic_workgroup_local_memory =
      maximum_dynamic_workgroup_local_memory_size;

  ASSERT_GT(physical_device->host_queue_count, 0u);
  iree_host_size_t operation_resource_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_validate_dispatch(
      &physical_device->host_queues[0], executable, function, config,
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE,
      &operation_resource_count));
  config.dynamic_workgroup_local_memory =
      maximum_dynamic_workgroup_local_memory_size + 1;
  EXPECT_THAT(Status(iree_hal_amdgpu_host_queue_validate_dispatch(
                  &physical_device->host_queues[0], executable, function,
                  config, iree_const_byte_span_empty(), bindings,
                  IREE_HAL_DISPATCH_FLAG_NONE, &operation_resource_count)),
              StatusIs(StatusCode::kOutOfRange));

  const auto record_aql_dispatch =
      [&](uint32_t dynamic_workgroup_local_memory_size) -> iree_status_t {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        test_device.base_device(), iree_hal_queue_family(test_device.queue()),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    iree_hal_dispatch_config_t dispatch_config =
        iree_hal_make_static_dispatch_config(1, 1, 1);
    dispatch_config.dynamic_workgroup_local_memory =
        dynamic_workgroup_local_memory_size;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        command_buffer, executable, function, dispatch_config,
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    return iree_hal_command_buffer_end(command_buffer);
  };
  IREE_EXPECT_OK(
      record_aql_dispatch(maximum_dynamic_workgroup_local_memory_size));
  EXPECT_THAT(Status(record_aql_dispatch(
                  maximum_dynamic_workgroup_local_memory_size + 1)),
              StatusIs(StatusCode::kOutOfRange));
}

}  // namespace
}  // namespace iree::hal::amdgpu
