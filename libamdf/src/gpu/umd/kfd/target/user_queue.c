// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"

#include "libamdf/src/gpu/umd/kfd/target/gfx1151/pm4_queue.h"
#include "libamdf/src/gpu/umd/kfd/target/gfx1151/sdma_queue.h"

_Static_assert(AMDF_GPU_QUEUE_FAMILY_CAPACITY >= 2,
               "target queue plan capacity must fit PM4 and SDMA");

void amdf_gpu_kfd_target_user_queue_plans_initialize(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size, amdf_gpu_kfd_user_queue_plans_t* out_plans) {
  amdf_gpu_kfd_user_queue_plans_t plans = {0};
  amdf_gpu_kfd_user_queue_plan_t plan = {0};
  if (amdf_gpu_kfd_gfx1151_pm4_queue_plan(topology, page_size, cache_line_size,
                                          &plan)) {
    plans.values[plans.count++] = plan;
  }
  if (amdf_gpu_kfd_gfx1151_sdma_queue_plan(topology, page_size, cache_line_size,
                                           &plan)) {
    plans.values[plans.count++] = plan;
  }
  *out_plans = plans;
}
