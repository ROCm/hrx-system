// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_COUNTS_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_COUNTS_H_

#include "iree/hal/driver/amd/gpu/queue/aql/indirect.h"
#include "iree/hal/driver/amd/gpu/queue/fixup.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable geometry for a launch whose device count buffer is ready when the
// invocation starts. Each descriptor owns disjoint grid/hidden-count words in
// the invocation's private instance. Ordinary fixups never write those words.
typedef struct iree_hal_amd_gpu_aql_count_t {
  // Byte offset of the native dispatch packet from the private instance base.
  uint32_t packet_offset;
  // Invocation value-table index holding the count buffer's base address.
  uint32_t value_index;
  // Full byte displacement from that count buffer base to its uint32_t XYZ.
  uint64_t count_offset;
  // Byte offset of runtime-owned kernargs from the private instance base.
  uint32_t argument_offset;
  // Admitted workgroup size for each native grid dimension.
  uint16_t workgroup_size[3];
  // Hidden-count byte offsets within the kernargs. UINT16_MAX means absent,
  // including every field of caller-owned indirect native arguments.
  uint16_t hidden_offsets[3];
} iree_hal_amd_gpu_aql_count_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_count_t) == 32,
              "two immutable count descriptors occupy one cache line");

// Sealed placement, shared by all instances without inspecting packet bytes.
typedef struct iree_hal_amd_gpu_aql_count_table_t {
  // Immutable descriptor-array byte offset from the resident program base.
  uint32_t offset;
  // Number of independent dispatch descriptors; zero selects ordinary fixups.
  uint32_t count;
} iree_hal_amd_gpu_aql_count_table_t;

// The ordinary fixup ABI is a prefix so the existing preparation launch can
// apply both kinds of disjoint writes. No dependent descriptor reads its
// target.
typedef struct iree_hal_amd_gpu_aql_count_arguments_t {
  // Ordinary replacements and this invocation's values/private target.
  iree_hal_amd_gpu_fixup_arguments_t fixup;
  // GPU address of the immutable descriptor array.
  uint64_t launches;
  // Number of descriptors processed after the ordinary fixup workgroups.
  uint32_t launch_count;
  // Initialized ABI padding, ignored by the shader.
  uint32_t reserved;
} iree_hal_amd_gpu_aql_count_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_count_arguments_t) == 48,
              "count preparation kernargs match the compiled ABI");

// Captures admitted launch/layout facts. Offsets are instance-relative after
// sealing; recording may initially supply data-relative argument_offset.
void iree_hal_amd_gpu_aql_count_prepare(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    const iree_hal_amd_gpu_dispatch_geometry_t* geometry,
    iree_hal_dispatch_flags_t flags, uint32_t packet_offset,
    uint32_t value_index, uint64_t count_offset, uint32_t argument_offset,
    iree_hal_amd_gpu_aql_count_t* out_count);

// Returns relocation-free instructions and native descriptor facts for the
// combined preparation kernel. This cold query never compiles or parses code.
iree_status_t iree_hal_amd_gpu_aql_count_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_COUNTS_H_
