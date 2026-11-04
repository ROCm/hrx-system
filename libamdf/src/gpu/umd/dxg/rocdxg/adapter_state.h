// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

// Private ROCDXG state retained for one live bridge adapter.
struct amdf_dxg_bridge_gpu_adapter_t {
  // Parent-instance allocator copied for all bridge-controlled storage.
  amdf_allocator_t host_allocator = {};
  // Borrowed native transport installed only during synchronous bridge calls.
  amdf_dxg_bridge_native_transport_t native_transport = {};
  // Private topology context owned by this adapter.
  thunk_proxy::ChainContext* chain = nullptr;
  // Private physical-device context owned by this adapter.
  thunk_proxy::DeviceContext* device = nullptr;

  ~amdf_dxg_bridge_gpu_adapter_t();
};

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_ADAPTER_STATE_H_
