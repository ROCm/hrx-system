// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_TARGET_MEMORY_H_
#define AMDF_SRC_GPU_UMD_KFD_TARGET_MEMORY_H_

#include "libamdf/src/gpu/umd/kfd/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sets complete expected memory facts from target/package identity and cached
// heap totals. The ordinary address envelope is architectural; device creation
// qualifies the installed aperture and placement on its native connection.
// An unrecognized identity returns false without changing topology.
bool amdf_gpu_kfd_target_memory_initialize(uint32_t pci_device_id,
                                           uint32_t page_size,
                                           amdf_gpu_kfd_topology_t* topology);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_KFD_TARGET_MEMORY_H_
