// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_NPU5_LEGACY_BOOTSTRAP_IMAGE_H_
#define AMDF_SRC_XDNA_UMD_MCDM_NPU5_LEGACY_BOOTSTRAP_IMAGE_H_

#include <stddef.h>
#include <stdint.h>

// Provider-owned compatibility image used only to admit a legacy NPU5 context.
extern const uint8_t amdf_windows_xdna_npu5_legacy_bootstrap_image[];

// Number of bytes in `amdf_windows_xdna_npu5_legacy_bootstrap_image`.
extern const size_t amdf_windows_xdna_npu5_legacy_bootstrap_image_size;

#endif  // AMDF_SRC_XDNA_UMD_MCDM_NPU5_LEGACY_BOOTSTRAP_IMAGE_H_
