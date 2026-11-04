// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_TRANSPORT_H_
#define AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_TRANSPORT_H_

#include "libamdf/src/gpu/umd/dxg/bridge_api.h"

namespace amdf::rocdxg_bridge {

// Installs one borrowed transport for synchronous thunk-proxy calls on the
// current thread. Nested scopes restore their predecessor on destruction.
class NativeTransportScope {
 public:
  explicit NativeTransportScope(
      const amdf_dxg_bridge_native_transport_t& transport) noexcept;
  ~NativeTransportScope();

  NativeTransportScope(const NativeTransportScope&) = delete;
  NativeTransportScope& operator=(const NativeTransportScope&) = delete;

 private:
  // Previously active transport restored when this scope ends.
  const amdf_dxg_bridge_native_transport_t* previous_;
};

}  // namespace amdf::rocdxg_bridge

#endif  // AMDF_SRC_GPU_UMD_DXG_ROCDXG_NATIVE_TRANSPORT_H_
