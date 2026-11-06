// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_METADATA_H_
#define AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_METADATA_H_

// Config Manager requires the Windows types established by the KMT header.
// clang-format off
#include "libamdf/src/platform/windows/kmt_api.h"
#include <cfgmgr32.h>
// clang-format on

// Explicit instance-owned native dependencies for passive endpoint metadata.
typedef struct amdf_windows_endpoint_metadata_api_t {
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
} amdf_windows_endpoint_metadata_api_t;

// Native attachment facts captured before activating an execution device.
typedef struct amdf_windows_endpoint_metadata_t {
  // OS attachment node with online CPUs, or UINT32_MAX when unavailable.
  uint32_t host_numa_node;
  // Physical PCI location, with availability independent of host CPU locality.
  amdf_endpoint_pci_address_t pci_address;
} amdf_windows_endpoint_metadata_t;

#ifdef __cplusplus
extern "C" {
#endif

// Loads system dependencies. Failure leaves the output unchanged.
amdf_status_t amdf_windows_endpoint_metadata_api_initialize(
    amdf_windows_endpoint_metadata_api_t* out_api);

// Releases the native module. Failure preserves ownership for retry.
amdf_status_t amdf_windows_endpoint_metadata_api_deinitialize(
    amdf_windows_endpoint_metadata_api_t* api);

// Queries one physical endpoint's attachment without activating execution.
// Native failures leave the output unchanged. Scratch uses host_allocator.
amdf_status_t amdf_windows_query_endpoint_metadata(
    const amdf_windows_endpoint_metadata_api_t* api, const amdf_kmt_api_t* kmt,
    D3DKMT_HANDLE adapter, uint32_t physical_adapter_index,
    amdf_allocator_t host_allocator,
    amdf_windows_endpoint_metadata_t* out_metadata);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_METADATA_H_
