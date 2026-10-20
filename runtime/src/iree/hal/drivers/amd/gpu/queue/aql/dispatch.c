// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/dispatch.h"

iree_status_t iree_hal_amd_gpu_aql_dispatch_prepare(
    const iree_hal_amd_gpu_aql_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_dispatch_geometry_t* out_dispatch) {
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->base.metadata;
  // AMDHSA uses_dynamic_stack is distinct from fixed scratch supplied by CP.
  if (function->code_properties & (1u << 11)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AQL dynamic stacks require an allocation policy");
  }
  if (function->base.kernarg->requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel '%.*s' requires argument services 0x%x that this queue "
        "has not provisioned",
        (int)metadata->reflection_name.size, metadata->reflection_name.data,
        function->base.kernarg->requirements);
  }
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AQL workgroup clusters require native admission");
  }
  return iree_hal_amd_gpu_dispatch_prepare(
      &function->base, maximum_group_segment_byte_length, true, config,
      constants, binding_count, flags, out_dispatch);
}

iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_dispatch_packet(
    const iree_hal_amd_gpu_aql_function_t* function,
    const iree_hal_amd_gpu_dispatch_geometry_t* dispatch, uint32_t header,
    uint64_t kernarg_address, uint64_t completion_signal,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amd_gpu_aql_dispatch_t packet = {
      .private_segment_size =
          function->base.metadata->private_segment_fixed_size,
      .group_segment_size = dispatch->group_segment_byte_length,
      .kernel_object = function->kernel_object,
      .kernarg_address = kernarg_address,
      .completion_signal = completion_signal,
  };
  uint32_t dimensions = 1;
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = dispatch->workgroup_size[i];
    const uint32_t partial = dispatch->partial_size[i];
    packet.workgroup_size[i] = (uint16_t)size;
    packet.grid_size[i] =
        dispatch->workgroup_count[i] * size - (partial ? size - partial : 0);
    if (size != 1 || dispatch->workgroup_count[i] != 1) {
      dimensions = (uint32_t)(i + 1);
    }
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    dimensions = 3;
  }
  packet.header =
      header | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (dimensions << 16);
  return iree_hal_amd_gpu_aql_packet_dispatch(&packet);
}
