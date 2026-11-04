// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_ALLOCATION_FLAGS_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_ALLOCATION_FLAGS_H_

#include <cstdint>

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace amdf::rocdxg_bridge {

// Encodes public bridge allocation policies into the private ROCDXG payload.
constexpr uint32_t EncodeNativeAllocationFlags(
    amdf_dxg_bridge_gpu_allocation_flags_t flags) noexcept {
  uint32_t native_flags = 0;
  if ((flags & AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_FINE_GRAIN) != 0) {
    native_flags |= thunk_proxy::kFineGrain;
  }
  if ((flags & AMDF_DXG_BRIDGE_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED) != 0) {
    native_flags |= thunk_proxy::kKernarg;
  }
  // Queue storage remains in its ordinary physical domain. The proxy's
  // kUserQueue domain denotes a separate driver-owned SWS queue object; it is
  // not a property of caller-owned user-mode queue storage.
  return native_flags;
}

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_ALLOCATION_FLAGS_H_
