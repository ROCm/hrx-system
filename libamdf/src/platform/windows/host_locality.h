// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_HOST_LOCALITY_H_
#define AMDF_SRC_PLATFORM_WINDOWS_HOST_LOCALITY_H_

// Config Manager requires the Windows types established by the KMT header.
// clang-format off
#include "libamdf/src/platform/windows/kmt_api.h"
#include <cfgmgr32.h>
// clang-format on

// Explicit instance-owned native dependencies for CPU locality discovery.
typedef struct amdf_windows_host_locality_api_t {
  // Owned system Config Manager module, or NULL after release.
  HMODULE module;
  // Resolves the exact device instance returned by KMT.
  CONFIGRET(WINAPI* locate_device)(PDEVINST, DEVINSTID_W, ULONG);
  // Reads native device properties without enumerating devices.
  CONFIGRET(WINAPI* get_property)(DEVINST, const DEVPROPKEY*, DEVPROPTYPE*,
                                  PBYTE, PULONG, ULONG);
  // Converts Config Manager failures to documented Win32 status codes.
  DWORD(WINAPI* map_error)(CONFIGRET, DWORD);
  // Queries active CPU groups for an OS NUMA node, when the OS supports it.
  BOOL(WINAPI* get_node_masks)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
} amdf_windows_host_locality_api_t;

#ifdef __cplusplus
extern "C" {
#endif

// Loads system dependencies. Failure leaves the output unchanged.
amdf_status_t amdf_windows_host_locality_api_initialize(
    amdf_windows_host_locality_api_t* out_api);

// Releases the native module. Failure preserves ownership for retry.
amdf_status_t amdf_windows_host_locality_api_deinitialize(
    amdf_windows_host_locality_api_t* api);

// Queries one physical endpoint's host CPU node without activating execution.
// UINT32_MAX means unavailable or an attachment node without online CPUs.
// Native failures leave the output unchanged. Scratch uses host_allocator.
amdf_status_t amdf_windows_query_host_locality(
    const amdf_windows_host_locality_api_t* api, const amdf_kmt_api_t* kmt,
    D3DKMT_HANDLE adapter, uint32_t physical_adapter_index,
    uint32_t* out_numa_node, amdf_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_WINDOWS_HOST_LOCALITY_H_
