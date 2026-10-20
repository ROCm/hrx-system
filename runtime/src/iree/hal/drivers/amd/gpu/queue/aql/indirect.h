// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_INDIRECT_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_INDIRECT_H_

#include "iree/hal/drivers/amd/gpu/image/image.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/dispatch.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Device-time geometry publication. The target body is complete but its header
// remains INVALID until this invocation writes the grids and hidden counts.
// Every target access is a store; fixed launch facts never come from the ring.
typedef struct iree_hal_amd_gpu_aql_indirect_arguments_t {
  // GPU address of the exclusively reserved target packet.
  uint64_t packet;
  // GPU address of three execution-time uint32_t workgroup counts.
  uint64_t counts;
  // GPU address of the target's runtime-owned native arguments, or zero when
  // every count offset is absent.
  uint64_t arguments;
  // Complete admitted dispatch header/setup DWORD, published last.
  uint32_t header;
  // Full workgroup size in X/Y/Z, copied from the admitted launch.
  uint32_t workgroup_size[3];
  // Hidden block-count byte offsets in the target's argument segment.
  // UINT32_MAX means absent, including every caller-owned native argument.
  uint32_t count_offsets[3];
  // Initialized ABI padding, ignored by the shader.
  uint32_t reserved;
} iree_hal_amd_gpu_aql_indirect_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_indirect_arguments_t) == 56,
              "indirect kernargs match the compiled ABI");

// Returns relocation-free instructions and launch descriptor facts. This cold
// query never invokes a compiler or parses a code object.
iree_status_t iree_hal_amd_gpu_aql_indirect_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

// Captures fixed launch facts without inspecting any packet or user bytes.
// Prepared copies may coalesce adjacent hidden fields or describe disjoint
// ABI offsets. Caller-owned indirect arguments are never modified.
void iree_hal_amd_gpu_aql_indirect_prepare(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    const iree_hal_amd_gpu_dispatch_geometry_t* geometry,
    iree_hal_dispatch_flags_t flags, uint64_t packet, uint64_t counts,
    uint64_t arguments, uint32_t header,
    iree_hal_amd_gpu_aql_indirect_arguments_t* out_arguments);

// Ordered one-workitem producer. Both its argument bytes and the INVALID
// target's body must be complete before this packet's header is published.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_indirect_packet(
    uint64_t kernel_object, uint64_t arguments);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_INDIRECT_H_
