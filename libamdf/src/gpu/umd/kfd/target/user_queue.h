// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_TARGET_USER_QUEUE_H_
#define AMDF_SRC_GPU_UMD_KFD_TARGET_USER_QUEUE_H_

#include "libamdf/src/gpu/umd/kfd/target/queue_plan.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"

// Dense target-qualified KFD user-queue plans for one endpoint snapshot.
typedef struct amdf_gpu_kfd_user_queue_plans_t {
  // Plans in the same dense order used for public family ordinals.
  amdf_gpu_kfd_user_queue_plan_t values[AMDF_GPU_QUEUE_FAMILY_CAPACITY];
  // Number of initialized plans in `values`.
  uint32_t count;
} amdf_gpu_kfd_user_queue_plans_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects every exact target plan supported by the native endpoint facts.
void amdf_gpu_kfd_target_user_queue_plans_initialize(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size, amdf_gpu_kfd_user_queue_plans_t* out_plans);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_TARGET_USER_QUEUE_H_
