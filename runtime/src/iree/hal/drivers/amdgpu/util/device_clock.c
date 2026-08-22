// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/device_clock.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/target_platform.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>

static uint64_t iree_hal_amdgpu_device_clock_scale_to_nanoseconds(
    uint64_t timestamp, uint64_t frequency_hz) {
  if (frequency_hz == 1000000000ull) return timestamp;
  const uint64_t seconds = timestamp / frequency_hz;
  const uint64_t remainder = timestamp % frequency_hz;
  return seconds * 1000000000ull + (remainder * 1000000000ull) / frequency_hz;
}
#endif  // IREE_PLATFORM_WINDOWS

iree_status_t iree_hal_amdgpu_device_clock_counters_validate(
    uint32_t driver_uid,
    const iree_hal_amdgpu_device_clock_counters_t* counters) {
  IREE_ASSERT_ARGUMENT(counters);
  if (IREE_UNLIKELY(counters->device_clock_counter == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device clock source returned an invalid zero device_clock_counter for "
        "driver_uid=%" PRIu32,
        driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_cpu_timestamp_ns == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_cpu_timestamp_ns for driver_uid=%" PRIu32,
                            driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_system_timestamp == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_system_timestamp for driver_uid=%" PRIu32,
                            driver_uid);
  }
  if (IREE_UNLIKELY(counters->host_system_frequency_hz == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "device clock source returned an invalid zero "
                            "host_system_frequency_hz for driver_uid=%" PRIu32,
                            driver_uid);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_device_clock_source_initialize(
    iree_hal_amdgpu_device_clock_source_t* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  memset(out_source, 0, sizeof(*out_source));

#if defined(IREE_PLATFORM_LINUX)
  int kfd = -1;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kfd_open(&kfd));
  out_source->state.linux_kfd_file = (intptr_t)kfd;
  out_source->type = IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD;
#elif defined(IREE_PLATFORM_WINDOWS)
  LARGE_INTEGER frequency = {0};
  if (IREE_UNLIKELY(!QueryPerformanceFrequency(&frequency) ||
                    frequency.QuadPart <= 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "QueryPerformanceCounter frequency is unavailable");
  }
  out_source->state.windows_qpc_frequency_hz = (uint64_t)frequency.QuadPart;
  out_source->type =
      IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_WINDOWS_HSA_AGENT_INFO;
#endif  // IREE_PLATFORM_LINUX

  return iree_ok_status();
}

void iree_hal_amdgpu_device_clock_source_deinitialize(
    iree_hal_amdgpu_device_clock_source_t* source) {
  if (!source) return;
  if (source->type == IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD) {
    iree_hal_amdgpu_kfd_close((int)source->state.linux_kfd_file);
  }
  memset(source, 0, sizeof(*source));
}

iree_status_t iree_hal_amdgpu_device_clock_source_sample(
    const iree_hal_amdgpu_device_clock_source_t* source,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    uint32_t driver_uid,
    iree_hal_amdgpu_device_clock_counters_t* out_counters) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_counters);
  memset(out_counters, 0, sizeof(*out_counters));
  (void)libhsa;
  (void)device_agent;
  (void)driver_uid;

  switch (source->type) {
#if defined(IREE_PLATFORM_LINUX)
    case IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_LINUX_KFD: {
      iree_hal_amdgpu_kfd_clock_counters_t kfd_counters = {0};
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kfd_get_clock_counters(
          (int)source->state.linux_kfd_file, driver_uid, &kfd_counters));
      out_counters->device_clock_counter = kfd_counters.gpu_clock_counter;
      out_counters->host_cpu_timestamp_ns = kfd_counters.cpu_clock_counter;
      out_counters->host_system_timestamp = kfd_counters.system_clock_counter;
      out_counters->host_system_frequency_hz = kfd_counters.system_clock_freq;
      return iree_hal_amdgpu_device_clock_counters_validate(driver_uid,
                                                            out_counters);
    }
#elif defined(IREE_PLATFORM_WINDOWS)
    case IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_WINDOWS_HSA_AGENT_INFO: {
      IREE_ASSERT_ARGUMENT(libhsa);
      hsa_amd_clock_counters_t hsa_counters = {0};
      IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
          IREE_LIBHSA(libhsa), device_agent,
          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CLOCK_COUNTERS, &hsa_counters));
      out_counters->device_clock_counter = hsa_counters.gpu_clock_counter;
      out_counters->host_cpu_timestamp_ns =
          iree_hal_amdgpu_device_clock_scale_to_nanoseconds(
              hsa_counters.cpu_clock_counter,
              source->state.windows_qpc_frequency_hz);
      out_counters->host_system_timestamp = hsa_counters.system_clock_counter;
      out_counters->host_system_frequency_hz =
          hsa_counters.system_clock_frequency;
      return iree_hal_amdgpu_device_clock_counters_validate(driver_uid,
                                                            out_counters);
    }
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_WINDOWS
    case IREE_HAL_AMDGPU_DEVICE_CLOCK_SOURCE_TYPE_UNAVAILABLE:
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU device clock sampling is unavailable on this platform");
  }
}
