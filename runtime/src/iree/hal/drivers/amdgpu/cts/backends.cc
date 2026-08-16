// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the AMDGPU HAL driver.

#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amdgpu/cts/backend_factories.h"

namespace iree::hal::cts {

static bool amdgpu_registered_ =
    (CtsRegistry::RegisterBackend({
         "amdgpu",
         MakeAmdgpuCtsBackendInfo("amdgpu", AmdgpuCtsBackendMode::kDefault),
         {"async_queue", "file_io"},
     }),
     true);

}  // namespace iree::hal::cts
