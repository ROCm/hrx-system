// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/endpoint_properties.h"

#include <stddef.h>

bool amdf_gpu_wddm_wkmi_endpoint_properties_translate(
    const amdf_wkmi_bridge_gpu_properties_t* provider_properties,
    amdf_gpu_endpoint_properties_t* out_properties) {
  if (provider_properties == NULL || out_properties == NULL) {
    return false;
  }
  const uint32_t xcc_count = provider_properties->xcc_count;
  const uint32_t shader_engine_count = provider_properties->shader_engine_count;
  // WKMI leaves this field zero when the private adapter record omits it. ROCr
  // uses 32 as the effective Windows limit for the same record shape.
  const uint32_t maximum_scratch_wave_count_per_compute_unit =
      provider_properties->maximum_scratch_wave_count_per_compute_unit
          ? provider_properties->maximum_scratch_wave_count_per_compute_unit
          : 32u;
  if (provider_properties->gfx_ip_major < 0 ||
      provider_properties->gfx_ip_minor < 0 ||
      provider_properties->gfx_ip_stepping < 0 || xcc_count == 0 ||
      shader_engine_count == 0 || shader_engine_count % xcc_count != 0) {
    return false;
  }

  amdf_gpu_endpoint_properties_t properties = {
      .gfx_ip =
          {
              .major = (uint32_t)provider_properties->gfx_ip_major,
              .minor = (uint32_t)provider_properties->gfx_ip_minor,
              .stepping = (uint32_t)provider_properties->gfx_ip_stepping,
          },
      // WKMI carries the complete HSAKMT capability word. The low four bits
      // are the ASIC revision used for physical compiler-target selection.
      .asic_revision = provider_properties->asic_revision & 0xFu,
      .compute =
          {
              .wavefront_size = provider_properties->wavefront_size,
              .compute_unit_count = provider_properties->compute_unit_count,
              .maximum_wave_count_per_compute_unit =
                  provider_properties->maximum_wave_count_per_compute_unit,
              .maximum_scratch_wave_count_per_compute_unit =
                  maximum_scratch_wave_count_per_compute_unit,
              .local_data_share_byte_length =
                  provider_properties->local_data_share_byte_length,
          },
      .topology =
          {
              .xcc_count = xcc_count,
              .shader_engine_count_per_xcc = shader_engine_count / xcc_count,
          },
  };
  if (provider_properties->supports_pm4_kernel_queue != 0) {
    properties.queue_families[properties.queue_family_count++] =
        (amdf_gpu_queue_family_properties_t){
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
            .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
            .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
                     AMDF_QUEUE_ROLE_CACHE_CONTROL,
            .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        };
  }
  if (provider_properties->supports_sdma_kernel_queue != 0) {
    // The native adapter record identifies the packet layout. Clients receive
    // encoding features and never need to reconstruct them from GFX identity.
    amdf_queue_format_features_t format_features = 0;
    if (provider_properties->gfx_ip_major >= 12) {
      format_features |= AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;
    }
    if (provider_properties->gfx_ip_major == 12 &&
        provider_properties->gfx_ip_minor >= 5) {
      format_features |= AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE;
    }
    properties.queue_families[properties.queue_family_count++] =
        (amdf_gpu_queue_family_properties_t){
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
            .format_features = format_features,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
        };
  }
  *out_properties = properties;
  return true;
}
