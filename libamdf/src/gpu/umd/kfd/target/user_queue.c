// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"

#include <string.h>

#include "libamdf/src/gpu/umd/kfd/hdp.h"
#include "libamdf/src/gpu/umd/kfd/target/aql_queue.h"
#include "libamdf/src/gpu/umd/kfd/target/pm4_queue.h"
#include "libamdf/src/gpu/umd/kfd/target/sdma_queue.h"

_Static_assert(AMDF_GPU_QUEUE_FAMILY_CAPACITY >= 3,
               "target queue plan capacity must fit AQL, PM4, and SDMA");

// Restrictions remove individual services after native plans are qualified.
// Discovery and construction consume the same effective capabilities.
typedef struct amdf_gpu_kfd_queue_restriction_t {
  // Exact native GC identity, independent of compiler target aliases.
  amdf_gpu_kfd_ip_version_t gc_ip;
  // Exact installed module version, independent of the host kernel release.
  const char* driver_version;
  // Queue family whose otherwise supported capabilities are restricted.
  amdf_queue_command_type_t command_type;
  // Direct publication capabilities invalidated by this native defect.
  amdf_user_queue_capabilities_t denied_capabilities;
} amdf_gpu_kfd_queue_restriction_t;

static void amdf_gpu_kfd_restrict_user_queue_plans(
    const amdf_gpu_kfd_topology_t* topology,
    amdf_gpu_kfd_user_queue_plans_t* plans) {
  // The ROCm 7.2 driver retains the unaligned SDMA engine doorbell bases
  // repaired by Linux f145d5b3f740. Shader writes can reach the wrong engine;
  // host writes are unaffected. Custom backports retaining this version are
  // conservatively restricted too. Unknown versions do not match this table.
  static const amdf_gpu_kfd_queue_restriction_t restrictions[] = {
      {{9, 4, 3, true},
       "6.16.13",
       AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
       AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER},
      {{9, 4, 4, true},
       "6.16.13",
       AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
       AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER},
      {{9, 5, 0, true},
       "6.16.13",
       AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
       AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER},
  };
  for (size_t i = 0; i < sizeof(restrictions) / sizeof(restrictions[0]); ++i) {
    const amdf_gpu_kfd_queue_restriction_t* restriction = &restrictions[i];
    if (topology->gc_ip.exact != restriction->gc_ip.exact ||
        topology->gc_ip.major != restriction->gc_ip.major ||
        topology->gc_ip.minor != restriction->gc_ip.minor ||
        topology->gc_ip.revision != restriction->gc_ip.revision ||
        strcmp(topology->driver_version, restriction->driver_version) != 0) {
      continue;
    }
    for (uint32_t j = 0; j < plans->count; ++j) {
      amdf_gpu_queue_family_properties_t* family = &plans->values[j].family;
      if (family->command_type == restriction->command_type) {
        family->user_queue_capabilities &= ~restriction->denied_capabilities;
      }
    }
  }
}

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
    if (topology->properties.execution_resources.compute_units_per_resource !=
            0 &&
        (family->roles & AMDF_QUEUE_ROLE_COMPUTE) != 0) {
      family->user_queue_capabilities |=
          AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION;
    }
    // KFD exposes cooperative admission only when its scheduler/firmware
    // prerequisites are met. The qualified GC9-11 queue writers carry that
    // scheduling contract; later MES writers require independent qualification.
    if (topology->global_wave_sync_count != 0 && topology->gc_ip.exact &&
        topology->gc_ip.major >= 9 && topology->gc_ip.major <= 11 &&
        (family->roles & AMDF_QUEUE_ROLE_COMPUTE) != 0) {
      family->user_queue_capabilities |=
          AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH;
    }
    plans.values[i].ring.placement = AMDF_QUEUE_PLACEMENT_HOST_LOCAL;
  }
  amdf_gpu_kfd_restrict_user_queue_plans(topology, &plans);
  *out_plans = plans;
}

amdf_queue_family_placement_info_t amdf_gpu_kfd_target_user_queue_placement(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size) {
  const amdf_queue_host_publication_flags_t local_publication =
      topology->integrated ? AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN
                           : amdf_gpu_kfd_hdp_host_publication(topology);
  const bool host_visible_local =
      topology->integrated ||
      (topology->vram.total_byte_length != 0 &&
       topology->vram.visible_byte_length >= topology->vram.total_byte_length &&
       page_size <= 4096 && local_publication != 0);
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
                  host_visible_local ? local_publication
                                     : AMDF_QUEUE_HOST_PUBLICATION_FLAG_UNKNOWN,
          },
  };
}
