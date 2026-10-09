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
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"

namespace iree::hal::cts {

static iree_status_t QueryExecutableGlobalDeviceAddress(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_device_size_t expected_length, uint64_t* out_address) {
  *out_address = 0;
  bool found = false;
  iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
  IREE_RETURN_IF_ERROR(iree_hal_executable_try_lookup_global_by_name(
      executable, name, &found, &global));
  if (!found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "executable global `%.*s` not found",
                            (int)name.size, name.data);
  }

  iree_hal_executable_global_info_t info;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_global_info(executable, global, &info));
  if (info.byte_length != expected_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable global `%.*s` has length %" PRIu64 ", expected %" PRIu64,
        (int)name.size, name.data, (uint64_t)info.byte_length,
        (uint64_t)expected_length);
  }

  iree_hal_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_global_buffer(executable, global, &buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  const uint64_t allocation_address =
      (uint64_t)(uintptr_t)iree_hal_amdgpu_buffer_device_pointer(
          allocated_buffer);
  if (!allocation_address) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable global `%.*s` has no native AMDGPU allocation",
        (int)name.size, name.data);
  }
  if (!iree_checked_add_u64(allocation_address,
                            iree_hal_buffer_byte_offset(buffer), out_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable global `%.*s` address overflows",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t ReadAsanShadowBytes(
    iree_hal_amdgpu_logical_device_t* logical_device,
    uint64_t application_address, iree_device_size_t application_length,
    std::vector<uint8_t>* out_shadow_bytes) {
  out_shadow_bytes->clear();
  iree_hal_amdgpu_shadow_map_t* shadow_map =
      iree_hal_amdgpu_asan_state_shadow_map(&logical_device->asan);
  if (!shadow_map) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU ASAN shadow map is disabled");
  }

  iree_hal_amdgpu_shadow_map_range_t shadow_range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_shadow_map_calculate_range(
      shadow_map, application_address, application_length, &shadow_range));
  out_shadow_bytes->resize(shadow_range.shadow_length);
  return iree_hsa_memory_copy(
      IREE_LIBHSA(logical_device->asan.libhsa), out_shadow_bytes->data(),
      (void*)(uintptr_t)shadow_range.shadow_address, out_shadow_bytes->size());
}

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
      iree::Status unavailable_status(std::move(status));
      GTEST_SKIP() << unavailable_status.ToString();
    }
    IREE_ASSERT_OK(status);

    iree_hal_executable_target_selection_result_t target_result;
    IREE_ASSERT_OK(SelectBackendExecutableTarget(
        device(), iree_hal_queue_family(queue()), GetParam(), &target_result));
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

    executable_target_ = target_result.target;

    IREE_ASSERT_OK(
        LoadExecutable("asan_executable_test.bin", executable_.out()));
  }

  iree_status_t LoadExecutable(std::string_view executable_file,
                               iree_hal_executable_t** out_executable) {
    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data = GetParam().executable_data(
        iree_make_string_view(executable_file.data(), executable_file.size()));
    return iree_hal_executable_load(iree_hal_queue_family(queue()),
                                    executable_target_, &load_params,
                                    out_executable);
  }

  void TearDown() override {
    if (device()) {
      IREE_EXPECT_OK(iree_hal_queue_flush(queue()));
    }
  }

  iree_hal_device_t* device() const { return asan_device_.device(); }

  iree_hal_queue_t* queue() const { return asan_device_.queue(); }

  iree_hal_allocator_t* allocator() const { return asan_device_.allocator(); }

  SanitizerDeviceEventRecorder* recorder() const {
    return asan_device_.recorder();
  }

  SanitizerCachedBackendDevice asan_device_;
  const iree_hal_executable_target_t* executable_target_ = nullptr;
  Ref<iree_hal_executable_t> executable_;
};

TEST_P(AsanExecutableTest, RevokesAndRepublishesGlobalLayout) {
  constexpr iree_device_size_t kObjectLength = 16;
  constexpr iree_device_size_t kObjectAndRedzoneLength = 24;
  constexpr uint8_t kAddressableShadowValue = 0x00;
  constexpr uint8_t kHeapRedzoneShadowValue = 0xFA;
  const iree_string_view_t object_name =
      iree_make_cstring_view("iree_asan_global_layout_data");
  auto* logical_device =
      reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(device());

  Ref<iree_hal_executable_t> first_executable;
  IREE_ASSERT_OK(
      LoadExecutable("asan_global_layout_test.bin", first_executable.out()));
  uint64_t first_address = 0;
  IREE_ASSERT_OK(QueryExecutableGlobalDeviceAddress(
      first_executable, object_name, kObjectLength, &first_address));

  std::vector<uint8_t> shadow_bytes;
  IREE_ASSERT_OK(ReadAsanShadowBytes(logical_device, first_address,
                                     kObjectAndRedzoneLength, &shadow_bytes));
  ASSERT_EQ(shadow_bytes.size(), 3u);
  EXPECT_EQ(shadow_bytes[0], kAddressableShadowValue);
  EXPECT_EQ(shadow_bytes[1], kAddressableShadowValue);
  EXPECT_EQ(shadow_bytes[2], kHeapRedzoneShadowValue);

  first_executable.reset();
  IREE_ASSERT_OK(ReadAsanShadowBytes(logical_device, first_address,
                                     kObjectAndRedzoneLength, &shadow_bytes));
  ASSERT_EQ(shadow_bytes.size(), 3u);
  EXPECT_EQ(shadow_bytes[0], kHeapRedzoneShadowValue);
  EXPECT_EQ(shadow_bytes[1], kHeapRedzoneShadowValue);
  EXPECT_EQ(shadow_bytes[2], kHeapRedzoneShadowValue);

  Ref<iree_hal_executable_t> second_executable;
  IREE_ASSERT_OK(
      LoadExecutable("asan_global_layout_test.bin", second_executable.out()));
  uint64_t second_address = 0;
  IREE_ASSERT_OK(QueryExecutableGlobalDeviceAddress(
      second_executable, object_name, kObjectLength, &second_address));
  IREE_ASSERT_OK(ReadAsanShadowBytes(logical_device, second_address,
                                     kObjectAndRedzoneLength, &shadow_bytes));
  ASSERT_EQ(shadow_bytes.size(), 3u);
  EXPECT_EQ(shadow_bytes[0], kAddressableShadowValue);
  EXPECT_EQ(shadow_bytes[1], kAddressableShadowValue);
  EXPECT_EQ(shadow_bytes[2], kHeapRedzoneShadowValue);

  second_executable.reset();
  IREE_ASSERT_OK(ReadAsanShadowBytes(logical_device, second_address,
                                     kObjectAndRedzoneLength, &shadow_bytes));
  ASSERT_EQ(shadow_bytes.size(), 3u);
  EXPECT_EQ(shadow_bytes[0], kHeapRedzoneShadowValue);
  EXPECT_EQ(shadow_bytes[1], kHeapRedzoneShadowValue);
  EXPECT_EQ(shadow_bytes[2], kHeapRedzoneShadowValue);
}

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
  IREE_ASSERT_OK(
      iree_hal_executable_global_buffer(executable_, global, &global_buffer));
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
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue(), empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
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
  IREE_ASSERT_OK(
      iree_hal_executable_global_buffer(executable_, global, &global_buffer));
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
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue(), empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
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
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue(), empty_wait, dispatch_signal, executable_,
      iree_hal_executable_function_from_index(0),
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
