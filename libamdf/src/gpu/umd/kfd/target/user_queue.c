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
  for (uint32_t i = 0; i < plans.count; ++i) {
    amdf_gpu_queue_family_properties_t* family = &plans.values[i].family;
    const amdf_queue_family_placement_info_t placement =
        amdf_gpu_kfd_target_user_queue_placement(topology, page_size,
                                                 family->command_type);
    // Passive discovery has physical heap totals but no activated memory
    // profile. Live queue construction qualifies the actual LOCAL route.
    if (family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
        topology->vram.total_byte_length != 0 &&
        (placement.placement_capabilities &
         AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL) != 0) {
      family->user_queue_capabilities |=
          AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION;
    }
    if (topology->properties.execution_resources.compute_units_per_resource !=
            0 &&
        (family->roles & AMDF_QUEUE_ROLE_COMPUTE) != 0) {
      family->user_queue_capabilities |=
          AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION;
    }
    // Cooperative support is qualified for modern CDNA and RDNA targets.
    // Older shader GWS protocols are not supported. KFD's GWS-named resource
    // still supplies native cooperative scheduling on these targets; its live
    // availability and exact GC identity qualify the queue writer independently
    // of the compiler ISA identity. Later MES writers are not qualified here.
    const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
    const bool cooperative_target =
        (properties->gfx_ip.major == 9 &&
         ((properties->gfx_ip.minor == 4 && properties->gfx_ip.stepping == 2) ||
          (properties->gfx_ip.minor == 5 &&
           properties->gfx_ip.stepping == 0))) ||
        properties->gfx_ip.major == 11;
    if (topology->global_wave_sync_count != 0 && topology->gc_ip.exact &&
        topology->gc_ip.major >= 9 && topology->gc_ip.major <= 11 &&
        cooperative_target && (family->roles & AMDF_QUEUE_ROLE_COMPUTE) != 0) {
      family->user_queue_capabilities |=
          AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH;
    }
    plans.values[i].ring.placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL;
  }
  *out_plans = plans;
}

amdf_queue_family_placement_info_t amdf_gpu_kfd_target_user_queue_placement(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    amdf_queue_command_type_t command_type) {
  const amdf_queue_host_publication_flags_t local_publication =
      topology->integrated ? AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN
                           : amdf_gpu_kfd_hdp_host_publication(topology);
  const bool host_visible_local =
      topology->integrated ||
      (topology->vram.total_byte_length != 0 &&
       topology->vram.visible_byte_length >= topology->vram.total_byte_length &&
       page_size <= 4096 && local_publication != 0);
  // AQL commits through a header in the same uncached ring as its body. Host
  // store ordering suffices for that relation even when the backing moves.
  // PM4 and SDMA commit through a separately mapped coherent write index.
  const amdf_queue_host_publication_flags_t ring_publication =
      command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
          ? AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN
          : local_publication;
  return (amdf_queue_family_placement_info_t){
      .placement_capabilities =
          AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
          (host_visible_local ? AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL
                              : 0),
      .host_publication =
          {
              [AMDF_QUEUE_PLACEMENT_DEFAULT] =
                  AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN,
              [AMDF_QUEUE_PLACEMENT_HOST_LOCAL] =
                  AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN,
              [AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL] =
                  host_visible_local ? ring_publication
                                     : AMDF_QUEUE_HOST_PUBLICATION_FLAG_UNKNOWN,
          },
  };
}
