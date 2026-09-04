// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN executable CTS coverage.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "iree/hal/cts/sanitizer/sanitizer_test_util.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/api.h"

namespace iree::hal::cts {

class AsanExecutableTest : public ::testing::TestWithParam<BackendInfo> {
 protected:
  void SetUp() override {
    std::string host_incompatibility_reason;
    if (!IsBackendHostCompatible(GetParam(), &host_incompatibility_reason)) {
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' is not compatible with this host: "
                   << host_incompatibility_reason;
    }

    iree_status_t status = asan_device_.Initialize(GetParam(), "asan");
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "Backend '" << GetParam().name
                   << "' unavailable on this system";
    }
    IREE_ASSERT_OK(status);

    iree_hal_executable_target_selection_result_t target_result;
    IREE_ASSERT_OK(
        SelectBackendExecutableTarget(device(), GetParam(), &target_result));
    if (target_result.outcome ==
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
      GTEST_SKIP() << "Executable target '"
                   << GetParam().executable_target_family << ":"
                   << GetParam().executable_target_key
                   << "' is unavailable on CTS backend/device '"
                   << GetParam().name << "'";
    }
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              target_result.outcome);

    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data = GetParam().executable_data(
        iree_make_cstring_view("asan_executable_test.bin"));
    IREE_ASSERT_OK(iree_hal_device_load_executable(
        device(), IREE_HAL_QUEUE_AFFINITY_ANY, target_result.target,
        &load_params, executable_.out()));
  }

  void TearDown() override {
    if (device()) {
      IREE_EXPECT_OK(
          iree_hal_device_queue_flush(device(), IREE_HAL_QUEUE_AFFINITY_ANY));
    }
  }

  iree_hal_device_t* device() const { return asan_device_.device(); }

  iree_hal_queue_t* queue() const { return asan_device_.queue(); }

  iree_hal_allocator_t* allocator() const { return asan_device_.allocator(); }

  SanitizerDeviceEventRecorder* recorder() const {
    return asan_device_.recorder();
  }

  SanitizerCachedBackendDevice asan_device_;
  Ref<iree_hal_executable_t> executable_;
};

TEST_P(AsanExecutableTest, PublishesConfigGlobal) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV(IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME), &found,
      &global));
  ASSERT_TRUE(found);

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            IREE_HAL_AMDGPU_ASAN_CONFIG_GLOBAL_NAME);
  ASSERT_EQ(info.byte_length, sizeof(iree_hal_amdgpu_asan_config_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  std::vector<iree_hal_amdgpu_asan_config_t> configs;
  IREE_ASSERT_OK(SanitizerReadBufferData(device(), queue(), allocator(),
                                         global_buffer, &configs));
  ASSERT_EQ(configs.size(), 1u);
  const iree_hal_amdgpu_asan_config_t& config = configs[0];
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_NE(config.shadow_base, 0u);
  EXPECT_EQ(config.shadow_size, IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SIZE);
  EXPECT_GE(config.shadow_slab_size,
            IREE_HAL_AMDGPU_ASAN_DEFAULT_SHADOW_SLAB_SIZE);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), 5 * sizeof(uint64_t),
                                             output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), sizeof(uint64_t),
                                             fallback_buffer.out()));

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

  const uint32_t constant_data[] = {0x4153414Eu, 0x43464721u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint64_t> output_data;
  IREE_ASSERT_OK(SanitizerReadBufferData(device(), queue(), allocator(),
                                         output_buffer, &output_data));
  ASSERT_EQ(output_data.size(), 5u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.shadow_base);
  EXPECT_EQ(output_data[3], config.shadow_size);
  EXPECT_EQ(output_data[4], config.shadow_slab_size);
}

TEST_P(AsanExecutableTest, PublishesFeedbackConfigGlobal) {
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV(IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME), &found,
      &global));
  ASSERT_TRUE(found);

  iree_hal_executable_global_info_t info;
  IREE_ASSERT_OK(iree_hal_executable_global_info(executable_, global, &info));
  EXPECT_EQ(std::string_view(info.name.data, info.name.size),
            IREE_HAL_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME);
  ASSERT_EQ(info.byte_length, sizeof(iree_hal_amdgpu_feedback_config_t));

  iree_hal_buffer_t* global_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, global, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));
  ASSERT_NE(global_buffer, nullptr);

  std::vector<iree_hal_amdgpu_feedback_config_t> configs;
  IREE_ASSERT_OK(SanitizerReadBufferData(device(), queue(), allocator(),
                                         global_buffer, &configs));
  ASSERT_EQ(configs.size(), 1u);
  const iree_hal_amdgpu_feedback_config_t& config = configs[0];
  EXPECT_EQ(config.record_length, sizeof(config));
  EXPECT_EQ(config.abi_version, IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0);
  EXPECT_NE(config.flags & IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED, 0u);
  EXPECT_NE(config.channel_base, 0u);
  EXPECT_NE(config.notify_signal.handle, 0u);
  EXPECT_NE(config.source_context, 0u);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), 5 * sizeof(uint64_t),
                                             output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), sizeof(uint64_t),
                                             fallback_buffer.out()));

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

  const uint32_t constant_data[] = {0x4644424Bu, 0x43464721u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint64_t> output_data;
  IREE_ASSERT_OK(SanitizerReadBufferData(device(), queue(), allocator(),
                                         output_buffer, &output_data));
  ASSERT_EQ(output_data.size(), 5u);
  EXPECT_EQ(output_data[0], config.record_length);
  EXPECT_EQ(output_data[1], config.flags);
  EXPECT_EQ(output_data[2], config.channel_base);
  EXPECT_EQ(output_data[3], config.notify_signal.handle);
  EXPECT_EQ(output_data[4], config.source_context);
}

TEST_P(AsanExecutableTest, ReportsAsanPacketThroughFeedback) {
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), sizeof(uint64_t),
                                             output_buffer.out()));
  Ref<iree_hal_buffer_t> fallback_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(allocator(), sizeof(uint64_t),
                                             fallback_buffer.out()));

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

  const uint32_t constant_data[] = {0x4153414Eu, 0x52505421u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  recorder()->Reset();
  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t report = recorder()->last_asan_report();
  EXPECT_EQ(report.record_length, sizeof(report));
  EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.fault_address, 0x123456789ABCDEFull);
  EXPECT_EQ(report.access_length, 16u);
  EXPECT_EQ(report.site_id, 0xC0DEFACEu);
  EXPECT_EQ(report.shadow_address, 0x56789ABCDEFull);
  EXPECT_EQ(report.shadow_value, 0xF0u);

  iree_hal_device_event_source_t source = recorder()->last_source();
  EXPECT_TRUE(iree_string_view_equal(source.driver_id, IREE_SV("amdgpu")));
  EXPECT_NE(source.executable_id, 0u);
  EXPECT_NE(source.physical_device_ordinal, UINT32_MAX);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AsanExecutableTest);

}  // namespace iree::hal::cts
