// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_
#define LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses the numeric architecture from a gfx target identifier. Feature
// suffixes beginning with ':' are accepted. Returns false without modifying
// |out_architecture| when the identifier is malformed or out of range.
bool iree_hip_parse_gcn_arch_name(const char* name, int* out_architecture);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_DEVICE_PROPERTIES_H_
