// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_DRM_CONTEXT_H_
#define AMDF_SRC_XDNA_UMD_DRM_CONTEXT_H_

#include <drm/amdxdna_accel.h>

#include "libamdf/src/atomics.h"
#include "libamdf/src/xdna/umd/context.h"
#include "libamdf/src/xdna/umd/drm/device.h"

// One schedulable hardware context and its private execution state.
struct amdf_xdna_umd_context_t {
  // Ordinary-address-domain device borrowed through context destruction.
  amdf_xdna_umd_device_t* device;
  // Hardware context, or AMDXDNA_INVALID_CTX_HANDLE after destruction.
  uint32_t handle;
  // Context's user-owned DRM timeline sync object, or zero after destruction.
  uint32_t completion_syncobj;
  // Exclusive lease for one public queue and its native timeline.
  amdf_atomic_uint32_t queue_leased;
  // First observed terminal firmware or bootstrap-admission failure.
  amdf_atomic_uint64_t terminal_status;
  // Last native sequence plus one, assigned under the exclusive queue lease.
  uint64_t last_native_sequence;
};

#endif  // AMDF_SRC_XDNA_UMD_DRM_CONTEXT_H_
