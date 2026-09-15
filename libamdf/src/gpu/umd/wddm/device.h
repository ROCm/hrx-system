// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_DEVICE_H_
#define AMDF_SRC_GPU_UMD_WDDM_DEVICE_H_

#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/wddm/memory_profile.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/adapter.h"
#include "libamdf/src/platform/windows/device_status.h"
#include "libamdf/src/platform/windows/kmt_api.h"

// Concrete Windows state backing one program-independent GPU device.
struct amdf_gpu_umd_device_t {
  // Host allocator copied for this device and child metadata.
  amdf_allocator_t host_allocator;
  // KMT table borrowed from the endpoint's platform instance.
  const amdf_kmt_api_t* kmt;
  // Adapter handle borrowed from the endpoint owning this device.
  D3DKMT_HANDLE adapter;
  // Physical adapter represented by native private records.
  uint32_t physical_adapter_index;
  // Exact GPU MMU facts available to immutable memory-profile queries.
  amdf_windows_gpu_memory_capabilities_t memory_capabilities;
  // Loaded WKMI module outliving its borrowed API table and native adapter.
  amdf_gpu_wddm_wkmi_loader_t wkmi_loader;
  // Parsed private WKMI adapter state shared by allocations and queues.
  amdf_gpu_wddm_wkmi_adapter_t wkmi_adapter;
  // Logical KMT device owning paging and future execution state.
  D3DKMT_HANDLE device;
  // Confirmed execution failure shared by every queue on this device.
  amdf_kmt_device_status_t status;
  // Paging queue owned by this logical device.
  D3DKMT_HANDLE paging_queue;
  // Synchronization object owned by the paging queue.
  D3DKMT_HANDLE paging_sync_object;
  // CPU mapping of the paging queue's monitored fence.
  const volatile uint64_t* paging_fence;
};

#endif  // AMDF_SRC_GPU_UMD_WDDM_DEVICE_H_
