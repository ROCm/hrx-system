// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU ASAN manual hook CTS coverage.

#include <array>
#include <cstdint>

#include "iree/hal/cts/sanitizer/sanitizer_test_util.h"
#include "iree/hal/drivers/amdgpu/buffer.h"

namespace iree::hal::cts {

namespace {

enum ManualAsanHookSelector : uint32_t {
  kManualAsanHookLoad1 = 1,
  kManualAsanHookLoad2 = 2,
  kManualAsanHookLoad4 = 3,
  kManualAsanHookLoad8 = 4,
  kManualAsanHookLoad16 = 5,
  kManualAsanHookLoadN = 6,
  kManualAsanHookStore1 = 7,
  kManualAsanHookStore2 = 8,
  kManualAsanHookStore4 = 9,
  kManualAsanHookStore8 = 10,
  kManualAsanHookStore16 = 11,
  kManualAsanHookStoreN = 12,
  kManualAsanHookLoad1NoAbort = 13,
  kManualAsanHookLoad2NoAbort = 14,
  kManualAsanHookLoad4NoAbort = 15,
  kManualAsanHookLoad8NoAbort = 16,
  kManualAsanHookLoad16NoAbort = 17,
  kManualAsanHookLoadNNoAbort = 18,
  kManualAsanHookStore1NoAbort = 19,
  kManualAsanHookStore2NoAbort = 20,
  kManualAsanHookStore4NoAbort = 21,
  kManualAsanHookStore8NoAbort = 22,
  kManualAsanHookStore16NoAbort = 23,
  kManualAsanHookStoreNNoAbort = 24,
  kManualAsanHookReportLoad1 = 25,
  kManualAsanHookReportLoad2 = 26,
  kManualAsanHookReportLoad4 = 27,
  kManualAsanHookReportLoad8 = 28,
  kManualAsanHookReportLoad16 = 29,
  kManualAsanHookReportLoadN = 30,
  kManualAsanHookReportStore1 = 31,
  kManualAsanHookReportStore2 = 32,
  kManualAsanHookReportStore4 = 33,
  kManualAsanHookReportStore8 = 34,
  kManualAsanHookReportStore16 = 35,
  kManualAsanHookReportStoreN = 36,
  kManualAsanHookReportLoad1NoAbort = 37,
  kManualAsanHookReportLoad2NoAbort = 38,
  kManualAsanHookReportLoad4NoAbort = 39,
  kManualAsanHookReportLoad8NoAbort = 40,
  kManualAsanHookReportLoad16NoAbort = 41,
  kManualAsanHookReportLoadNNoAbort = 42,
  kManualAsanHookReportStore1NoAbort = 43,
  kManualAsanHookReportStore2NoAbort = 44,
  kManualAsanHookReportStore4NoAbort = 45,
  kManualAsanHookReportStore8NoAbort = 46,
  kManualAsanHookReportStore16NoAbort = 47,
  kManualAsanHookReportStoreNNoAbort = 48,
  kManualAsanHookPoisonRegion = 49,
  kManualAsanHookUnpoisonRegion = 50,
};

struct ManualAsanHookCase {
  // Kernel selector for the hook call.
  uint32_t selector;
  // Human-readable hook name used in assertion messages.
  const char* name;
  // Expected HAL ASAN access kind.
  iree_hal_device_asan_access_kind_t access_kind;
  // Expected access length in bytes.
  uint64_t access_length;
};

constexpr uint64_t kManualAsanBufferLength = 64;
constexpr uint64_t kManualAsanDynamicAccessLength = 23;
constexpr uint64_t kManualAsanSentinelAccessLength = 37;
constexpr uint32_t kManualAsanBindingEntrypoint = 0;
constexpr uint32_t kManualAsanRawAddressEntrypoint = 1;

constexpr std::array<ManualAsanHookCase, 48> kManualAsanHookCases = {{
    {kManualAsanHookLoad1, "__asan_load1",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 1},
    {kManualAsanHookLoad2, "__asan_load2",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 2},
    {kManualAsanHookLoad4, "__asan_load4",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 4},
    {kManualAsanHookLoad8, "__asan_load8",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 8},
    {kManualAsanHookLoad16, "__asan_load16",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 16},
    {kManualAsanHookLoadN, "__asan_loadN",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, kManualAsanDynamicAccessLength},
    {kManualAsanHookStore1, "__asan_store1",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 1},
    {kManualAsanHookStore2, "__asan_store2",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 2},
    {kManualAsanHookStore4, "__asan_store4",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 4},
    {kManualAsanHookStore8, "__asan_store8",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 8},
    {kManualAsanHookStore16, "__asan_store16",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 16},
    {kManualAsanHookStoreN, "__asan_storeN",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, kManualAsanDynamicAccessLength},
    {kManualAsanHookLoad1NoAbort, "__asan_load1_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 1},
    {kManualAsanHookLoad2NoAbort, "__asan_load2_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 2},
    {kManualAsanHookLoad4NoAbort, "__asan_load4_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 4},
    {kManualAsanHookLoad8NoAbort, "__asan_load8_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 8},
    {kManualAsanHookLoad16NoAbort, "__asan_load16_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 16},
    {kManualAsanHookLoadNNoAbort, "__asan_loadN_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, kManualAsanDynamicAccessLength},
    {kManualAsanHookStore1NoAbort, "__asan_store1_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 1},
    {kManualAsanHookStore2NoAbort, "__asan_store2_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 2},
    {kManualAsanHookStore4NoAbort, "__asan_store4_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 4},
    {kManualAsanHookStore8NoAbort, "__asan_store8_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 8},
    {kManualAsanHookStore16NoAbort, "__asan_store16_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 16},
    {kManualAsanHookStoreNNoAbort, "__asan_storeN_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, kManualAsanDynamicAccessLength},
    {kManualAsanHookReportLoad1, "__asan_report_load1",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 1},
    {kManualAsanHookReportLoad2, "__asan_report_load2",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 2},
    {kManualAsanHookReportLoad4, "__asan_report_load4",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 4},
    {kManualAsanHookReportLoad8, "__asan_report_load8",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 8},
    {kManualAsanHookReportLoad16, "__asan_report_load16",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 16},
    {kManualAsanHookReportLoadN, "__asan_report_load_n",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, kManualAsanDynamicAccessLength},
    {kManualAsanHookReportStore1, "__asan_report_store1",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 1},
    {kManualAsanHookReportStore2, "__asan_report_store2",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 2},
    {kManualAsanHookReportStore4, "__asan_report_store4",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 4},
    {kManualAsanHookReportStore8, "__asan_report_store8",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 8},
    {kManualAsanHookReportStore16, "__asan_report_store16",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 16},
    {kManualAsanHookReportStoreN, "__asan_report_store_n",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, kManualAsanDynamicAccessLength},
    {kManualAsanHookReportLoad1NoAbort, "__asan_report_load1_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 1},
    {kManualAsanHookReportLoad2NoAbort, "__asan_report_load2_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 2},
    {kManualAsanHookReportLoad4NoAbort, "__asan_report_load4_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 4},
    {kManualAsanHookReportLoad8NoAbort, "__asan_report_load8_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 8},
    {kManualAsanHookReportLoad16NoAbort, "__asan_report_load16_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, 16},
    {kManualAsanHookReportLoadNNoAbort, "__asan_report_load_n_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ, kManualAsanDynamicAccessLength},
    {kManualAsanHookReportStore1NoAbort, "__asan_report_store1_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 1},
    {kManualAsanHookReportStore2NoAbort, "__asan_report_store2_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 2},
    {kManualAsanHookReportStore4NoAbort, "__asan_report_store4_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 4},
    {kManualAsanHookReportStore8NoAbort, "__asan_report_store8_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 8},
    {kManualAsanHookReportStore16NoAbort, "__asan_report_store16_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, 16},
    {kManualAsanHookReportStoreNNoAbort, "__asan_report_store_n_noabort",
     IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE, kManualAsanDynamicAccessLength},
}};

static bool ManualAsanHookIsReportSelector(uint32_t selector) {
  return selector >= kManualAsanHookReportLoad1 &&
         selector <= kManualAsanHookReportStoreNNoAbort;
}

static iree_status_t DispatchManualAsanSelector(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_hal_buffer_ref_list_t bindings, uint32_t selector,
    uint32_t access_length, int32_t address_adjustment = 0) {
  const uint32_t constant_data[] = {
      selector,
      access_length,
      static_cast<uint32_t>(address_adjustment),
  };
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device, {0}, {1});
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_dispatch(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable,
      iree_hal_executable_function_from_index(kManualAsanBindingEntrypoint),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  return iree_hal_semaphore_list_wait(dispatch_signal, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t DispatchManualAsanRawAddress(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    uint64_t address, uint32_t selector, uint32_t access_length) {
  const uint32_t constant_data[] = {
      static_cast<uint32_t>(address),
      static_cast<uint32_t>(address >> 32),
      selector,
      access_length,
  };
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device, {0}, {1});
  iree_hal_buffer_ref_list_t empty_bindings = iree_hal_buffer_ref_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_dispatch(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable,
      iree_hal_executable_function_from_index(kManualAsanRawAddressEntrypoint),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, empty_bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  return iree_hal_semaphore_list_wait(dispatch_signal, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

}  // namespace

class ManualAsanExecutableTest : public ::testing::TestWithParam<BackendInfo> {
};

TEST_P(ManualAsanExecutableTest, ReportsCompatibleHooksThroughFeedback) {
  SanitizerCachedBackendDevice asan_device;
  iree_status_t status = asan_device.Initialize(GetParam(), "asan");
  if (iree_status_is_unavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Backend '" << GetParam().name
                 << "' unavailable on this system";
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_executable_cache_t> executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      asan_device.device(), iree_make_cstring_view("default"),
      executable_cache.out()));

  iree_const_byte_span_t executable_data = GetParam().executable_data(
      iree_make_cstring_view("manual_asan_test.bin"));
  ASSERT_FALSE(iree_const_byte_span_is_empty(executable_data));

  Ref<iree_hal_executable_t> executable;
  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format =
      iree_make_cstring_view(GetParam().executable_format);
  executable_params.executable_data = executable_data;
  status = iree_hal_executable_cache_prepare_executable(
      executable_cache, &executable_params, executable.out());
  if (iree_status_is_incompatible(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "Executable format '" << GetParam().executable_format
                 << "' is incompatible with CTS backend/device '"
                 << GetParam().name << "'";
  }
  IREE_ASSERT_OK(status);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(
      asan_device.allocator(), kManualAsanBufferLength, output_buffer.out()));

  iree_hal_buffer_ref_t binding_refs[1];
  binding_refs[0] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  asan_device.recorder()->Reset();
  IREE_ASSERT_OK(DispatchManualAsanSelector(asan_device.device(), executable,
                                            bindings, kManualAsanHookLoad2,
                                            /*access_length=*/2));
  IREE_ASSERT_OK(DispatchManualAsanSelector(
      asan_device.device(), executable, bindings, kManualAsanHookReportLoadN,
      static_cast<uint32_t>(kManualAsanSentinelAccessLength)));
  asan_device.recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t allocation_body_report =
      asan_device.recorder()->last_asan_report();
  EXPECT_EQ(allocation_body_report.access_kind,
            IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ);
  EXPECT_EQ(allocation_body_report.access_length,
            kManualAsanSentinelAccessLength);

  asan_device.recorder()->Reset();
  IREE_ASSERT_OK(DispatchManualAsanSelector(asan_device.device(), executable,
                                            bindings, kManualAsanHookLoad1,
                                            /*access_length=*/1,
                                            /*address_adjustment=*/-1));
  asan_device.recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t left_report =
      asan_device.recorder()->last_asan_report();
  EXPECT_EQ(left_report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ);
  EXPECT_EQ(left_report.access_length, 1u);
  EXPECT_NE(left_report.shadow_value, 0u);

  iree_hal_buffer_ref_t tail_binding_refs[1];
  tail_binding_refs[0] = iree_hal_make_buffer_ref(
      output_buffer, kManualAsanBufferLength - 1, /*length=*/1);
  iree_hal_buffer_ref_list_t tail_bindings = {
      /*.count=*/IREE_ARRAYSIZE(tail_binding_refs),
      /*.values=*/tail_binding_refs,
  };
  asan_device.recorder()->Reset();
  IREE_ASSERT_OK(DispatchManualAsanSelector(asan_device.device(), executable,
                                            tail_bindings, kManualAsanHookLoad2,
                                            /*access_length=*/2));
  asan_device.recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t tail_report =
      asan_device.recorder()->last_asan_report();
  EXPECT_EQ(tail_report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ);
  EXPECT_EQ(tail_report.access_length, 2u);
  EXPECT_NE(tail_report.shadow_value, 0u);

  Ref<iree_hal_buffer_t> stale_buffer;
  IREE_ASSERT_OK(SanitizerCreateDeviceBuffer(
      asan_device.allocator(), kManualAsanBufferLength, stale_buffer.out()));
  const uint64_t stale_address =
      (uint64_t)(uintptr_t)iree_hal_amdgpu_buffer_device_pointer(stale_buffer);
  ASSERT_NE(stale_address, 0u);
  stale_buffer.reset();
  asan_device.recorder()->Reset();
  IREE_ASSERT_OK(DispatchManualAsanRawAddress(
      asan_device.device(), executable, stale_address, kManualAsanHookLoad1,
      /*access_length=*/1));
  asan_device.recorder()->WaitForAsanReportCount(1);
  EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u);
  iree_hal_device_asan_report_t stale_report =
      asan_device.recorder()->last_asan_report();
  EXPECT_EQ(stale_report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ);
  EXPECT_EQ(stale_report.access_length, 1u);
  EXPECT_NE(stale_report.shadow_value, 0u);

  for (const ManualAsanHookCase& hook_case : kManualAsanHookCases) {
    const bool is_report_selector =
        ManualAsanHookIsReportSelector(hook_case.selector);

    IREE_ASSERT_OK(DispatchManualAsanSelector(
        asan_device.device(), executable, bindings,
        kManualAsanHookUnpoisonRegion,
        static_cast<uint32_t>(kManualAsanBufferLength)))
        << hook_case.name;
    asan_device.recorder()->Reset();

    if (!is_report_selector) {
      IREE_ASSERT_OK(DispatchManualAsanSelector(
          asan_device.device(), executable, bindings, hook_case.selector,
          static_cast<uint32_t>(hook_case.access_length)))
          << hook_case.name;
      IREE_ASSERT_OK(DispatchManualAsanSelector(
          asan_device.device(), executable, bindings,
          kManualAsanHookReportLoadN,
          static_cast<uint32_t>(kManualAsanSentinelAccessLength)))
          << hook_case.name;
      asan_device.recorder()->WaitForAsanReportCount(1);
      EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u)
          << hook_case.name;
      iree_hal_device_asan_report_t sentinel_report =
          asan_device.recorder()->last_asan_report();
      EXPECT_EQ(sentinel_report.access_kind,
                IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ)
          << hook_case.name;
      EXPECT_EQ(sentinel_report.access_length, kManualAsanSentinelAccessLength)
          << hook_case.name;
    }

    IREE_ASSERT_OK(DispatchManualAsanSelector(
        asan_device.device(), executable, bindings, kManualAsanHookPoisonRegion,
        static_cast<uint32_t>(kManualAsanBufferLength)))
        << hook_case.name;
    asan_device.recorder()->Reset();

    IREE_ASSERT_OK(DispatchManualAsanSelector(
        asan_device.device(), executable, bindings, hook_case.selector,
        static_cast<uint32_t>(hook_case.access_length)))
        << hook_case.name;

    asan_device.recorder()->WaitForAsanReportCount(1);
    EXPECT_EQ(asan_device.recorder()->asan_report_count(), 1u)
        << hook_case.name;
    iree_hal_device_asan_report_t report =
        asan_device.recorder()->last_asan_report();
    EXPECT_EQ(report.record_length, sizeof(report)) << hook_case.name;
    EXPECT_EQ(report.abi_version, IREE_HAL_DEVICE_ASAN_REPORT_ABI_VERSION_0)
        << hook_case.name;
    EXPECT_EQ(report.access_kind, hook_case.access_kind) << hook_case.name;
    EXPECT_NE(report.fault_address, 0u) << hook_case.name;
    EXPECT_EQ(report.access_length, hook_case.access_length) << hook_case.name;
    EXPECT_EQ(report.site_id, 0u) << hook_case.name;
    EXPECT_NE(report.shadow_address, 0u) << hook_case.name;
    EXPECT_NE(report.shadow_value, 0u) << hook_case.name;

    iree_hal_device_event_source_t source =
        asan_device.recorder()->last_source();
    EXPECT_TRUE(iree_string_view_equal(source.driver_id, IREE_SV("amdgpu")))
        << hook_case.name;
    EXPECT_NE(source.executable_id, 0u) << hook_case.name;
    EXPECT_NE(source.physical_device_ordinal, UINT32_MAX) << hook_case.name;

    IREE_ASSERT_OK(DispatchManualAsanSelector(
        asan_device.device(), executable, bindings,
        kManualAsanHookUnpoisonRegion,
        static_cast<uint32_t>(kManualAsanBufferLength)))
        << hook_case.name;
  }

  IREE_EXPECT_OK(iree_hal_device_queue_flush(asan_device.device(),
                                             IREE_HAL_QUEUE_AFFINITY_ANY));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(ManualAsanExecutableTest);

}  // namespace iree::hal::cts
