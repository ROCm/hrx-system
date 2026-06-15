// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA kernel descriptor byte emission.
//
// The kernel descriptor is the loader-visible resource contract for one kernel.
// Loom keeps it as structured facts until final object layout gives us the
// descriptor-to-entry offset, then writes the fixed 64-byte AMDHSA descriptor
// directly into a native contribution.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_

#include "iree/base/api.h"
#include "loom/target/emit/native/amdgpu/metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Byte size of an AMDHSA code object v5 kernel descriptor.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH = 64,
  // Required descriptor symbol alignment in bytes.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ALIGNMENT = 64,
};

typedef enum loom_amdgpu_kernel_descriptor_bits_e {
  // Enable private-segment-buffer user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER = 1u << 0,
  // Enable dispatch-ptr user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR = 1u << 1,
  // Enable queue-ptr user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_QUEUE_PTR = 1u << 2,
  // Enable kernarg-segment-ptr user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR = 1u << 3,
  // Enable dispatch-id user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID = 1u << 4,
  // Enable flat-scratch-init user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_FLAT_SCRATCH_INIT = 1u << 5,
  // Enable private-segment-size user SGPRs.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = 1u << 6,
  // Enable architected private segment setup.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_PRIVATE_SEGMENT = 1u << 7,
  // Enable workgroup-id-x system SGPR setup.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X = 1u << 8,
  // Enable workgroup-id-y system SGPR setup.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y = 1u << 9,
  // Enable workgroup-id-z system SGPR setup.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z = 1u << 10,
  // Enable workgroup-info system SGPR setup.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_INFO = 1u << 11,
  // Request the architecturally initialized workitem-id-x VGPR.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X = 1u << 12,
  // Request the workitem-id-y VGPR after workitem-id-x.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y = 1u << 13,
  // Request the workitem-id-z VGPR after workitem-id-x/y.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z = 1u << 14,
  // Execute the kernel in wavefront-size-32 mode.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_WAVEFRONT_SIZE32 = 1u << 15,
  // Mark that the kernel may use a dynamically sized stack.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_USES_DYNAMIC_STACK = 1u << 16,
} loom_amdgpu_kernel_descriptor_bits_t;

// Bitset of loom_amdgpu_kernel_descriptor_bits_t values.
typedef uint32_t loom_amdgpu_kernel_descriptor_flags_t;

typedef struct loom_amdgpu_kernel_descriptor_t {
  // Processor name such as `gfx1100`.
  iree_string_view_t processor;
  // Fixed LDS/group segment size in bytes.
  uint32_t group_segment_fixed_size;
  // Fixed scratch/private segment size in bytes.
  uint32_t private_segment_fixed_size;
  // Kernel kernarg segment size in bytes.
  uint32_t kernarg_size;
  // Byte offset from the descriptor symbol to the kernel entry instruction.
  int64_t kernel_code_entry_byte_offset;
  // Physical SGPR high-water count for the kernel body.
  uint32_t next_free_sgpr;
  // Physical VGPR high-water count for the kernel body.
  uint32_t next_free_vgpr;
  // Total user SGPR count encoded in COMPUTE_PGM_RSRC2.
  uint32_t user_sgpr_count;
  // Descriptor flags controlling AMDHSA setup and code properties.
  loom_amdgpu_kernel_descriptor_flags_t flags;
} loom_amdgpu_kernel_descriptor_t;

// Initializes descriptor facts from the metadata kernel record used for the
// same kernel.
//
// Metadata-derived descriptor flags are initialized from |metadata_kernel|.
// Callers that require additional user SGPRs, system SGPRs, private-segment
// state, workitem IDs, or code properties must OR the corresponding flags
// before calling loom_amdgpu_kernel_descriptor_write.
iree_status_t loom_amdgpu_kernel_descriptor_initialize_from_metadata(
    iree_string_view_t processor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel,
    int64_t kernel_code_entry_byte_offset,
    loom_amdgpu_kernel_descriptor_t* out_descriptor);

// Verifies that descriptor facts still agree with the metadata kernel record
// for fields that both payloads expose.
iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel);

// Resolves descriptor workitem-id flags to the COMPUTE_PGM_RSRC2 field value.
iree_status_t loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
    loom_amdgpu_kernel_descriptor_flags_t flags, uint32_t* out_mode);

// Writes one AMDHSA kernel descriptor into |target_bytes|.
iree_status_t loom_amdgpu_kernel_descriptor_write(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    iree_byte_span_t target_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_DESCRIPTOR_H_
