// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_HOST_MEMORY_H_
#define AMDF_SRC_HOST_MEMORY_H_

#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares CPU-only system backing in the existing resource's native slot.
// Borrowed pages remain caller-owned; allocation owns only the host VM range.
amdf_status_t amdf_host_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_profile_t* profile,
    const amdf_memory_create_info_t* create_info, amdf_memory_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_HOST_MEMORY_H_
