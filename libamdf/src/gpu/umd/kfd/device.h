// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_DEVICE_H_
#define AMDF_SRC_GPU_UMD_KFD_DEVICE_H_

#include <stddef.h>

#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/kfd/reset_monitor.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"

typedef struct amdf_gpu_kfd_user_queue_native_api_t
    amdf_gpu_kfd_user_queue_native_api_t;

// Execution state borrowing its instance's KFD context and per-GPU VM binding.
struct amdf_gpu_umd_device_t {
  // Host allocator copied for device and child metadata.
  amdf_allocator_t host_allocator;
  // Borrowed instance KFD file selecting the primary or secondary context.
  int descriptor;
  // Borrowed instance render file whose VM is acquired by the KFD context.
  int render_descriptor;
  // Native lifetime inherited from the provider instance.
  amdf_native_lifetime_t native_lifetime;
  // Native identity, geometry and address limits established at creation.
  amdf_gpu_kfd_topology_t topology;
  // Native CPU page length in bytes.
  size_t page_size;
  // Qualified host cache-line length in bytes.
  uint32_t cache_line_size;
  // Native KFD queue operations borrowed through device destruction.
  const amdf_gpu_kfd_user_queue_native_api_t* user_queue_native_api;
  // Physical reset observer owned for every queue-qualified device.
  amdf_gpu_kfd_reset_monitor_t reset_monitor;
};

#endif  // AMDF_SRC_GPU_UMD_KFD_DEVICE_H_
