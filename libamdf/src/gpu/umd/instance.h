// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_INSTANCE_H_
#define AMDF_SRC_GPU_UMD_INSTANCE_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_gpu_umd_instance_t amdf_gpu_umd_instance_t;

// Prepares instance-owned shared GPU connections during explicit device
// creation. The caller holds its native-state lock. The already-live instance
// owns `state`; any result may retain native progress there for retry or
// instance teardown. This operation does not create an execution device.
// A UMD whose shared connections already belong to the platform may leave
// state NULL and use that existing owner directly.
amdf_status_t amdf_gpu_umd_instance_prepare(
    amdf_gpu_umd_instance_t** state, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator);

// Releases shared native connections after all required devices are destroyed.
// Success frees the state; failure retains it for retry with no live devices.
amdf_status_t amdf_gpu_umd_instance_destroy(amdf_gpu_umd_instance_t* instance);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_INSTANCE_H_
