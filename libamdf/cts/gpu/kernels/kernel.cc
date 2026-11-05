// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/kernel.h"

#include <cstdio>
#include <cstring>

namespace kernels {

const Kernel* KernelSet::Find(const amdf_gpu_endpoint_info_t& endpoint) const {
  const auto& gfx = endpoint.gfx_ip;
  const char* qualifier = "";
  // The physical target catalog distinguishes the strict gfx1250 A0
  // instruction set from the generic B0 instruction set.
  if (gfx.major == 12 && gfx.minor == 5 && gfx.stepping == 0) {
    switch (endpoint.asic_revision) {
      case 0:
        qualifier = "-strict";
        break;
      case 1:
        break;
      default:
        return nullptr;
    }
  }
  char selector[48];
  std::snprintf(selector, sizeof(selector), "gfx%u%x%x%s", gfx.major, gfx.minor,
                gfx.stepping, qualifier);
  for (const Kernel& kernel : variants) {
    if (std::strcmp(kernel.target, selector) == 0) {
      return &kernel;
    }
  }
  return nullptr;
}

}  // namespace kernels
