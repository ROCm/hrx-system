// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU CTS backend factory helpers.

#ifndef IREE_HAL_DRIVERS_AMDGPU_CTS_BACKEND_FACTORIES_H_
#define IREE_HAL_DRIVERS_AMDGPU_CTS_BACKEND_FACTORIES_H_

#include "iree/hal/cts/util/registry.h"

namespace iree::hal::cts {

enum class AmdgpuCtsBackendMode {
  // Plain AMDGPU CTS backend using default logical-device options.
  kDefault,
  // AMDGPU CTS backend with production ASAN logical-device options enabled.
  kAsan,
  // AMDGPU CTS backend with production TSAN logical-device options enabled.
  kTsan,
};

BackendInfo MakeAmdgpuCtsBackendInfo(const char* name,
                                     AmdgpuCtsBackendMode mode);

}  // namespace iree::hal::cts

#endif  // IREE_HAL_DRIVERS_AMDGPU_CTS_BACKEND_FACTORIES_H_
