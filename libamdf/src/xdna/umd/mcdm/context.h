// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_CONTEXT_H_
#define AMDF_SRC_XDNA_UMD_MCDM_CONTEXT_H_

#include "libamdf/src/xdna/umd/context.h"
#include "libamdf/src/xdna/umd/mcdm/device.h"
#include "libamdf/src/xdna/umd/mcdm/native_abi.h"

typedef struct amdf_windows_xdna_kernel_execution_t
    amdf_windows_xdna_kernel_execution_t;

// Concrete Windows state backing one schedulable XDNA context.
struct amdf_xdna_umd_context_t {
  // Ordinary-address-domain device borrowed through context destruction.
  amdf_xdna_umd_device_t* device;
  // Program-independent KMT execution context.
  D3DKMT_HANDLE handle;
  // Driver-returned command aperture selector; zero is a valid value.
  uint32_t command_aperture_cookie;
  // Process-lifetime wire facts qualified against the installed driver.
  const amdf_windows_xdna_native_abi_t* native_abi;
  // Lazily prepared context-local instruction and kernel-queue state.
  amdf_windows_xdna_kernel_execution_t* kernel_execution;
};

#endif  // AMDF_SRC_XDNA_UMD_MCDM_CONTEXT_H_
