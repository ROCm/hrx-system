// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL executable creation, export metadata, and reflection.

#include <cstdint>
#include <string_view>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;
using ::testing::AnyOf;

class ExecutableTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    LoadExecutableOrSkipUnsupported("executable_test.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  // Probes whether export_info is available and reports parameter metadata.
  // Returns false without recording a test failure when the backend does not
  // implement reflection — callers use this to GTEST_SKIP().
  bool ExportHasParameterInfo(iree_hal_executable_function_t export_ordinal) {
    iree_hal_executable_function_info_t info;
    iree_status_t status =
        iree_hal_executable_function_info(executable_, export_ordinal, &info);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    return info.parameter_count != 0;
  }

  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(ExecutableTest, ExportCount) {
  ASSERT_EQ(iree_hal_executable_function_count(executable_), 1);
}

TEST_P(ExecutableTest, ExportInfoOutOfRange) {
  iree_hal_executable_function_info_t info;
  EXPECT_THAT(
      Status(iree_hal_executable_function_info(
          executable_, iree_hal_executable_function_from_index(100), &info)),
      StatusIs(StatusCode::kOutOfRange));
}

TEST_P(ExecutableTest, ExportInfo) {
  iree_hal_executable_function_info_t info;

  // export0: #hal.pipeline.layout<constants = 2, bindings = [
  //   #hal.pipeline.binding<storage_buffer>,
  //   #hal.pipeline.binding<storage_buffer>
  // ]>
  IREE_ASSERT_OK(iree_hal_executable_function_info(
      executable_, iree_hal_executable_function_from_index(0), &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size), "export0");
  EXPECT_EQ(
      info.flags & ~IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC,
      IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE);
  EXPECT_EQ(info.constant_byte_length, 2 * sizeof(uint32_t));
  EXPECT_EQ(info.binding_count, 2);
  EXPECT_EQ(info.resource_usage.provided_flags &
                ~IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL,
            IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_NONE);
  if (!iree_any_bit_set(
          info.resource_usage.provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY)) {
    EXPECT_EQ(info.resource_usage.fixed_workgroup_local_memory_size, 0u);
  }
  if (!iree_any_bit_set(
          info.resource_usage.provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_PRIVATE_MEMORY)) {
    EXPECT_EQ(info.resource_usage.fixed_private_memory_size, 0u);
  }
  if (!iree_any_bit_set(
          info.resource_usage.provided_flags,
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_INVOCATION_REGISTERS)) {
    EXPECT_EQ(info.resource_usage.invocation_register_count, 0u);
  }
  const uint64_t minimum_workgroup_invocations =
      static_cast<uint64_t>(info.workgroup_size[0]) * info.workgroup_size[1] *
      info.workgroup_size[2];
  if (info.maximum_workgroup_invocations != 0) {
    EXPECT_LE(minimum_workgroup_invocations,
              info.maximum_workgroup_invocations);
  }
}

TEST_P(ExecutableTest, ExportParametersOutOfRange) {
  if (!ExportHasParameterInfo(iree_hal_executable_function_from_index(0))) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_function_parameter_t parameters[64];
  EXPECT_THAT(Status(iree_hal_executable_function_parameters(
                  executable_, iree_hal_executable_function_from_index(100),
                  IREE_ARRAYSIZE(parameters), parameters)),
              StatusIs(StatusCode::kOutOfRange));
}

TEST_P(ExecutableTest, ExportParametersNoCapacity) {
  if (!ExportHasParameterInfo(iree_hal_executable_function_from_index(0))) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_function_parameter_t parameter;
  IREE_EXPECT_OK(iree_hal_executable_function_parameters(
      executable_, iree_hal_executable_function_from_index(0), /*capacity=*/0,
      &parameter));
}

TEST_P(ExecutableTest, ExportParameters) {
  if (!ExportHasParameterInfo(iree_hal_executable_function_from_index(0))) {
    GTEST_SKIP() << "parameter reflection not available";
  }

  iree_hal_executable_function_parameter_t parameters[64];
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable_, iree_hal_executable_function_from_index(0),
      IREE_ARRAYSIZE(parameters), parameters));
}

TEST_P(ExecutableTest, LookupExportByNameNotFound) {
  iree_hal_executable_function_t ordinal =
      iree_hal_executable_function_from_index(0);
  EXPECT_THAT(Status(iree_hal_executable_lookup_function_by_name(
                  executable_, IREE_SV("NOT_FOUND"), &ordinal)),
              StatusIs(StatusCode::kNotFound));
}

TEST_P(ExecutableTest, LookupExportByName) {
  iree_hal_executable_function_t ordinal =
      iree_hal_executable_function_from_index(0);

  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable_, IREE_SV("export0"), &ordinal));
  EXPECT_EQ(ordinal.value, 0);
}

TEST_P(ExecutableTest, TryLookupGlobalByNameNotFound) {
  bool found = true;
  iree_hal_executable_global_t global =
      iree_hal_executable_global_from_value(0);
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV("NOT_FOUND"), &found, &global));
  EXPECT_FALSE(found);
  EXPECT_FALSE(iree_hal_executable_global_is_valid(global));
}

TEST_P(ExecutableTest, LookupGlobalByName) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV("executable_test_global"), &found, &global));
  if (!found) GTEST_SKIP() << "executable testdata has no globals";
  ASSERT_TRUE(iree_hal_executable_global_is_valid(global));

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            "executable_test_global");
  ASSERT_EQ(info.byte_length, sizeof(uint64_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(global_buffer), sizeof(uint64_t));

  const uint64_t expected_value = 0xFEEDFACECAFEBEEFull;
  SemaphoreList empty_wait;
  SemaphoreList update_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
      &expected_value, /*source_offset=*/0, global_buffer, /*target_offset=*/0,
      sizeof(expected_value), IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      update_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint64_t> data = ReadBufferData<uint64_t>(global_buffer);
  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0], expected_value);
}

TEST_P(ExecutableTest, GlobalBufferVisibleToDispatch) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV("executable_test_global"), &found, &global));
  if (!found) GTEST_SKIP() << "executable testdata has no globals";

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  const uint64_t expected_value = 0xBADDEC0DEFEED123ull;
  SemaphoreList empty_wait;
  SemaphoreList update_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
      &expected_value, /*source_offset=*/0, global_buffer, /*target_offset=*/0,
      sizeof(expected_value), IREE_HAL_UPDATE_FLAG_NONE));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(sizeof(uint64_t), output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  const uint64_t fallback_value = 0xAAAAAAAA55555555ull;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(
      &fallback_value, sizeof(fallback_value), fallback_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[2];
  binding_refs[0] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  binding_refs[1] =
      iree_hal_make_buffer_ref(fallback_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(fallback_buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {1, 1};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, update_signal, dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint64_t> data = ReadBufferData<uint64_t>(output_buffer);
  ASSERT_EQ(data.size(), 1u);
  EXPECT_EQ(data[0], expected_value);
}

TEST_P(ExecutableTest, LookupGlobalByNameNotFound) {
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  EXPECT_THAT(Status(iree_hal_executable_lookup_global_by_name(
                  executable_, IREE_SV("NOT_FOUND"), &global)),
              StatusIs(StatusCode::kNotFound));
  EXPECT_FALSE(iree_hal_executable_global_is_valid(global));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(ExecutableTest);

}  // namespace iree::hal::cts
