// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_
#define AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_

#include "libamdf/src/gpu/umd/instance.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the borrowed descriptor after successful instance preparation.
int amdf_gpu_kfd_instance_descriptor(const amdf_gpu_umd_instance_t* instance);

// Acquires or reuses this context's exact render-file VM for one GPU. The
// caller holds the owning platform instance's native-state lock. Any return
// may retain native progress on instance; only success publishes the borrowed
// descriptor and refines topology's memory facts from that same connection.
// Native memory qualification precedes VM bootstrap; failure leaves topology
// unchanged. KFD has no per-GPU inverse of ACQUIRE_VM, so the binding belongs
// to the instance even when no execution device currently borrows it.
amdf_status_t amdf_gpu_kfd_instance_prepare_vm(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology, size_t page_size,
    int* out_render_descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_
