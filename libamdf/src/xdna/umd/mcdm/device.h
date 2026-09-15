// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_DEVICE_H_
#define AMDF_SRC_XDNA_UMD_MCDM_DEVICE_H_

#include <stdint.h>

#include "libamdf/src/platform/windows/device_status.h"
#include "libamdf/src/platform/windows/kmt_api.h"
#include "libamdf/src/xdna/umd/device.h"

// Concrete Windows state backing one XDNA ordinary-address-domain device.
struct amdf_xdna_umd_device_t {
  // Host allocator copied for device and child metadata.
  amdf_allocator_t host_allocator;
  // Process-lifetime execution profile selected for this device.
  const amdf_xdna_endpoint_profile_t* profile;
  // KMT table borrowed from the endpoint's platform instance.
  const amdf_kmt_api_t* kmt;
  // Endpoint-owned adapter borrowed for context ABI qualification.
  D3DKMT_HANDLE adapter;
  // Logical KMT device owning paging and execution state.
  D3DKMT_HANDLE device;
  // Confirmed execution failure shared by this device's execution paths.
  amdf_kmt_device_status_t status;
  // Paging queue owned by this logical device.
  D3DKMT_HANDLE paging_queue;
  // Synchronization object owned by the paging queue.
  D3DKMT_HANDLE paging_sync_object;
  // CPU mapping of the paging queue's monitored fence.
  const volatile uint64_t* paging_fence;
};

#endif  // AMDF_SRC_XDNA_UMD_MCDM_DEVICE_H_
