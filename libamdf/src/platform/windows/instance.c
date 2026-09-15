// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/instance.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/endpoint_snapshot.h"

amdf_status_t amdf_platform_instance_create(
    amdf_allocator_t host_allocator, amdf_platform_instance_t** out_instance) {
  amdf_platform_instance_t* instance = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*instance),
                  amdf_alignof(amdf_platform_instance_t), (void**)&instance);
  if (!amdf_status_is_ok(status)) return status;
  instance->host_allocator = host_allocator;
  InitializeSRWLock(&instance->native_lock);
  status = amdf_kmt_api_initialize(&instance->kmt);
  if (amdf_status_is_ok(status)) {
    *out_instance = instance;
  } else {
    amdf_free(host_allocator, instance);
  }
  return status;
}

void amdf_platform_instance_lock_native(amdf_platform_instance_t* instance) {
  AcquireSRWLockExclusive(&instance->native_lock);
}

uint64_t amdf_platform_instance_host_allocation_granularity(
    const amdf_platform_instance_t* instance) {
  (void)instance;
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwAllocationGranularity;
}

void amdf_platform_instance_unlock_native(amdf_platform_instance_t* instance) {
  ReleaseSRWLockExclusive(&instance->native_lock);
}

amdf_status_t amdf_platform_instance_destroy(
    amdf_platform_instance_t* instance) {
  const amdf_status_t status = amdf_kmt_api_deinitialize(&instance->kmt);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = instance->host_allocator;
    amdf_free(host_allocator, instance);
  }
  return status;
}

amdf_status_t amdf_platform_endpoint_enumerate(
    amdf_platform_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count) {
  if (!amdf_kmt_api_supports_endpoint_discovery(&instance->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return amdf_windows_endpoint_snapshot_enumerate(
      &instance->kmt, capacity, summaries, out_count, instance->host_allocator);
}
