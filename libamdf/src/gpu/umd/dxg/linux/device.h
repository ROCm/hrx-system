// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_H_
#define AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_H_

#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/dxg/linux/logical_device.h"
#include "libamdf/src/gpu/umd/dxg/linux/memory_profile.h"
#include "libamdf/src/gpu/umd/dxg/linux/paging_queue.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/loader.h"

// Concrete direct Linux DXG state backing one GPU device.
struct amdf_gpu_umd_device_t {
  // Host allocator copied for this device and child metadata.
  amdf_allocator_t host_allocator;
  // Query adapter borrowed from the endpoint through device destruction.
  D3DKMT_HANDLE adapter;
  // DXGI adapter identity forwarded to private provider operations.
  uint64_t adapter_luid;
  // Physical adapter represented by the private provider state.
  uint32_t physical_adapter_index;
  // Exact GPU MMU facts available to immutable memory-profile queries.
  amdf_gpu_dxg_memory_capabilities_t memory_capabilities;
  // Immutable native target identity and queue-service facts.
  amdf_dxg_bridge_gpu_properties_t properties;
  // Loaded private bridge module outliving every call through its API table.
  amdf_gpu_dxg_rocdxg_loader_t bridge_loader;
  // Logical KMT device and private provider adapter owned together.
  amdf_gpu_linux_dxg_logical_device_t logical_device;
  // Paging queue owned by the logical device.
  amdf_gpu_linux_dxg_paging_queue_t paging_queue;
};

#endif  // AMDF_SRC_GPU_UMD_DXG_LINUX_DEVICE_H_
