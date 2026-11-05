// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_TOOL_INFO_PLATFORM_H_
#define AMDF_TOOL_INFO_PLATFORM_H_

#include <stdio.h>

#include "amdf/base.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host platform name for the report header, not a device capability.
const char* amdf_info_platform_name(void);

// Whether this output stream already supports terminal emphasis.
bool amdf_info_platform_has_color(FILE* file);

// Resolves a Linux character-device path without opening an execution file.
// Platforms without this namespace return UNSUPPORTED. Failure leaves the
// output unchanged; native status retains its original domain and code.
amdf_status_t amdf_info_platform_resolve_path(
    const char* path, amdf_endpoint_native_identity_t* out_identity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_TOOL_INFO_PLATFORM_H_
