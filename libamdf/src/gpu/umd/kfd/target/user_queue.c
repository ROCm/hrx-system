// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"

#include "libamdf/src/gpu/umd/kfd/hdp.h"
#include "libamdf/src/gpu/umd/kfd/target/aql_queue.h"
#include "libamdf/src/gpu/umd/kfd/target/pm4_queue.h"
#include "libamdf/src/gpu/umd/kfd/target/sdma_queue.h"

_Static_assert(AMDF_GPU_QUEUE_FAMILY_CAPACITY >= 3,
               "target queue plan capacity must fit AQL, PM4, and SDMA");

void amdf_gpu_kfd_target_user_queue_plans_initialize(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size, amdf_gpu_kfd_user_queue_plans_t* out_plans) {
  amdf_gpu_kfd_user_queue_plans_t plans = {0};
  amdf_gpu_kfd_user_queue_plan_t plan = {0};
  // CDNA consumes AQL; RDNA supports both compute packet languages. Native
  // storage and descriptor encodings are resolved before either family is
  // advertised, and the live constructor consumes the same plans.
  const bool cdna = topology->properties.gfx_ip.major == 9 &&
                    ((topology->properties.gfx_ip.minor == 0 &&
                      topology->properties.gfx_ip.stepping == 10) ||
                     (topology->properties.gfx_ip.minor == 4 &&
                      topology->properties.gfx_ip.stepping <= 2) ||
                     (topology->properties.gfx_ip.minor == 5 &&
                      topology->properties.gfx_ip.stepping == 0));
  const bool rdna = topology->properties.gfx_ip.major == 11 ||
                    (topology->properties.gfx_ip.major == 12 &&
                     (topology->properties.gfx_ip.minor == 0 ||
                      topology->properties.gfx_ip.minor == 5));
  // Preserve the PM4 family before AQL when both are present.
  if (rdna && amdf_gpu_kfd_pm4_queue_plan(topology, page_size, cache_line_size,
                                          &plan)) {
    plans.values[plans.count++] = plan;
  }
  if ((cdna || rdna) && amdf_gpu_kfd_aql_queue_plan(topology, page_size,
                                                    cache_line_size, &plan)) {
    plans.values[plans.count++] = plan;
  }
  // SDMA is a separate engine: its ring ABI does not depend on compute IP or
  // the number of compute XCCs. Its plan owns exact engine/format selection.
  if (amdf_gpu_kfd_sdma_queue_plan(topology, page_size, cache_line_size,
                                   &plan)) {
    plans.values[plans.count++] = plan;
  }
  const amdf_queue_host_publication_flags_t local_publication =
      topology->integrated ? AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN
                           : amdf_gpu_kfd_hdp_host_publication(topology);
  const bool host_visible_local =
      topology->integrated ||
      (topology->vram.total_byte_length != 0 &&
       topology->vram.visible_byte_length >= topology->vram.total_byte_length &&
       page_size <= 4096 && local_publication != 0);
  for (uint32_t i = 0; i < plans.count; ++i) {
    plans.values[i].family.placement_capabilities =
        AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
        (host_visible_local ? AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL : 0);
    plans.values[i].ring.placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL;
    plans.values[i].family.host_publication[AMDF_QUEUE_PLACEMENT_DEFAULT] =
        AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN;
    plans.values[i].family.host_publication[AMDF_QUEUE_PLACEMENT_HOST_LOCAL] =
        AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN;
    plans.values[i].family.host_publication[AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL] =
        host_visible_local ? local_publication
                           : AMDF_QUEUE_HOST_PUBLICATION_FLAG_UNKNOWN;
  }
  *out_plans = plans;
}
