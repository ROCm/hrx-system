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
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
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
      iree_hal_queue_family(test_device.queue()),
      IREE_SV("command_buffer_dispatch_multi_workgroup_test.bin"),
      executable.out()));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &function_info));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue_ordinal(
          executable, function, /*queue_ordinal=*/0, &descriptor));
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
        iree_hal_queue_family(test_device.queue()),
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

TEST_F(ExecutableTest, EnforcesUniformWorkgroupsAtDispatchBoundaries) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* host_queue = test_device.first_host_queue();
  ASSERT_NE(host_queue, nullptr);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  iree_hal_buffer_ref_t binding = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/1,
      /*.values=*/&binding,
  };

  struct Fixture {
    // Unique embedded code-object filename selected by the actual loader.
    const char* file_name;
    // Compiler-produced uniformity requirement expected from reflection.
    bool requires_uniform_workgroups;
  };
  const Fixture fixtures[] = {
      {"uniform_dispatch_metadata_fixture.bin", true},
      {"nonuniform_dispatch_metadata_fixture.bin", false},
  };
  const iree_hal_command_buffer_mode_t recording_modes[] = {
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
  };
  for (const Fixture& fixture : fixtures) {
    SCOPED_TRACE(fixture.file_name);
    Ref<iree_hal_executable_t> executable;
    IREE_ASSERT_OK(LoadCtsExecutable(iree_hal_queue_family(test_device.queue()),
                                     iree_make_cstring_view(fixture.file_name),
                                     executable.out()));
    const iree_hal_executable_function_t function =
        iree_hal_executable_function_from_index(0);
    iree_hal_executable_function_info_t function_info = {};
    IREE_ASSERT_OK(iree_hal_executable_function_info(executable, function,
                                                     &function_info));
    ASSERT_EQ(
        iree_any_bit_set(
            function_info.flags,
            IREE_HAL_EXECUTABLE_FUNCTION_FLAG_REQUIRES_UNIFORM_WORKGROUPS),
        fixture.requires_uniform_workgroups);
    ASSERT_TRUE(iree_any_bit_set(
        function_info.flags,
        IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC));
    ASSERT_EQ(function_info.binding_count, 1u);
    ASSERT_EQ(function_info.constant_byte_length, 0u);
    ASSERT_EQ(function_info.resource_usage.provided_flags,
              IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL);
    ASSERT_EQ(function_info.resource_usage.fixed_workgroup_local_memory_size,
              0u);
    ASSERT_EQ(function_info.resource_usage.fixed_private_memory_size, 0u);

    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
        nullptr;
    IREE_ASSERT_OK(
        iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue_ordinal(
            executable, function, /*queue_ordinal=*/0, &descriptor));
    ASSERT_NE(descriptor, nullptr);
    ASSERT_EQ(
        iree_any_bit_set(
            descriptor->export_flags,
            IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_UNIFORM_WORKGROUPS),
        fixture.requires_uniform_workgroups);
    iree_hal_dispatch_config_t ordinary_config =
        iree_hal_make_static_dispatch_config(2, 2, 2);
    for (iree_host_size_t i = 0; i < 3; ++i) {
      ordinary_config.workgroup_size[i] = 4;
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_executable_dispatch_limits_validate_workgroup_size(
            &descriptor->limits, ordinary_config.workgroup_size));
    iree_hal_dispatch_config_t exact_config = ordinary_config;
    for (iree_host_size_t i = 0; i < 3; ++i) {
      exact_config.workitem_count[i] = 8;
    }

    // This entry point only prepares and validates a plan. Even an unexpected
    // success for a rejected shape cannot publish a native dispatch packet.
    const auto validate_host_dispatch =
        [&](iree_hal_dispatch_config_t config,
            iree_hal_dispatch_flags_t flags) -> iree_status_t {
      iree_host_size_t operation_resource_count = 0;
      return iree_hal_amdgpu_host_queue_validate_dispatch(
          host_queue, executable, function, config,
          iree_const_byte_span_empty(), bindings, flags,
          &operation_resource_count);
    };
    // Recordings are destroyed locally and are never submitted. The executable
    // and output owners outlive every accepted or rejected recording.
    const auto record_aql_dispatch =
        [&](iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags,
            iree_hal_command_buffer_mode_t mode) -> iree_status_t {
      Ref<iree_hal_command_buffer_t> command_buffer;
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
          iree_hal_queue_family(test_device.queue()), mode,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH,
          /*binding_capacity=*/0, command_buffer.out()));
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
          command_buffer, executable, function, config,
          iree_const_byte_span_empty(), bindings, flags));
      return iree_hal_command_buffer_end(command_buffer);
    };

    IREE_EXPECT_OK(
        validate_host_dispatch(ordinary_config, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_EXPECT_OK(validate_host_dispatch(
        exact_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT));
    for (iree_hal_command_buffer_mode_t mode : recording_modes) {
      SCOPED_TRACE(mode);
      IREE_EXPECT_OK(record_aql_dispatch(ordinary_config,
                                         IREE_HAL_DISPATCH_FLAG_NONE, mode));
      IREE_EXPECT_OK(record_aql_dispatch(
          exact_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT, mode));
    }

    for (iree_host_size_t dimension = 0; dimension < 3; ++dimension) {
      SCOPED_TRACE(dimension);
      iree_hal_dispatch_config_t partial_config = exact_config;
      partial_config.workitem_count[dimension] = 7;
      if (fixture.requires_uniform_workgroups) {
        EXPECT_THAT(
            Status(validate_host_dispatch(
                partial_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT)),
            StatusIs(StatusCode::kInvalidArgument));
#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
        EXPECT_THAT(
            Status(record_aql_dispatch(
                partial_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT,
                IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT)),
            StatusIs(StatusCode::kInvalidArgument));
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
      } else {
        IREE_EXPECT_OK(validate_host_dispatch(
            partial_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT));
        for (iree_hal_command_buffer_mode_t mode : recording_modes) {
          SCOPED_TRACE(mode);
          IREE_EXPECT_OK(record_aql_dispatch(
              partial_config, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT,
              mode));
        }
      }
    }
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
