// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_MONITORED_FENCE_H_
#define AMDF_SRC_PLATFORM_WINDOWS_MONITORED_FENCE_H_

#include "libamdf/src/kernel_queue_dependency.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// KMT-created monitored-fence provenance, openable by KMT and D3D12 consumers.
typedef struct amdf_windows_queue_progress_t {
  // Immutable native protocol and producer publication cost.
  amdf_kernel_queue_native_progress_t base;
  // Exact adapter LUID; cross-adapter synchronization is not implied.
  uint64_t adapter_luid;
  // Owning NT handle retained by the producer for every cold consumer import.
  HANDLE shared_handle;
} amdf_windows_queue_progress_t;

// Separately owned progress fence; its queue owns the borrowed KMT device/API.
typedef struct amdf_windows_monitored_fence_t {
  // Exportable identity borrowed by cold consumer bindings.
  amdf_windows_queue_progress_t progress;
  // Producer-domain native synchronization object.
  D3DKMT_HANDLE object;
  // CPU-visible progress. UINT64_MAX indicates removal, never retirement.
  const volatile uint64_t* pointer;
} amdf_windows_monitored_fence_t;

// Consumer-domain monitored-fence reference retained through accepted waits.
typedef struct amdf_windows_queue_dependency_t {
  // Native cleanup and per-operand publication cost.
  amdf_kernel_queue_native_dependency_t base;
  // Allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // KMT procedures borrowed from the consumer device.
  const amdf_kmt_api_t* api;
  // Native view imported into the consumer's exact KMT logical device.
  D3DKMT_HANDLE object;
} amdf_windows_queue_dependency_t;

// Cached API admission; no native work or allocation.
bool amdf_windows_monitored_fence_is_supported(const amdf_kmt_api_t* api);

// Cold transactional construction. No native command can reference the fence
// before success; failure leaves out_fence unchanged.
amdf_status_t amdf_windows_monitored_fence_initialize(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE device, uint64_t adapter_luid,
    amdf_windows_monitored_fence_t* out_fence);

// Consumes idle native state, including on cleanup error. The caller has proved
// retirement of every signal and released all dependent views.
amdf_status_t amdf_windows_monitored_fence_deinitialize(
    const amdf_kmt_api_t* api, amdf_windows_monitored_fence_t* fence);

// Cold transactional import of qualified KMT-created producer progress.
amdf_status_t amdf_windows_queue_dependency_create(
    amdf_allocator_t host_allocator, const amdf_kmt_api_t* api,
    D3DKMT_HANDLE device, uint64_t adapter_luid,
    const amdf_kernel_queue_native_progress_t* progress,
    amdf_kernel_queue_native_dependency_t** out_native);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_WINDOWS_MONITORED_FENCE_H_
