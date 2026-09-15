// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_DRM_DEVICE_H_
#define AMDF_SRC_XDNA_UMD_DRM_DEVICE_H_

#include "libamdf/src/xdna/umd/device.h"
#include "libamdf/src/xdna/umd/drm/buffer.h"

// One independent accel client and ordinary address domain, with native
// firmware-addressable storage when the profile supports execution.
struct amdf_xdna_umd_device_t {
  // Host allocator copied for device and child metadata.
  amdf_allocator_t host_allocator;
  // Process-lifetime execution profile selected for this device.
  const amdf_xdna_endpoint_profile_t* profile;
  // Fresh open file description owning all native handle namespaces.
  int descriptor;
  // Native host page size established during construction.
  size_t page_size;
  // Qualified CLFLUSH cache-line length in bytes.
  uint32_t cache_line_size;
  // Firmware heap and persistent host mapping backing private instruction
  // memory.
  amdf_linux_xdna_buffer_t heap;
};

#endif  // AMDF_SRC_XDNA_UMD_DRM_DEVICE_H_
