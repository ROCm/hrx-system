// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_KERNARG_LAYOUT_H_
#define IREE_HAL_DRIVERS_AMDGPU_KERNARG_LAYOUT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_kernarg_layout_t
//===----------------------------------------------------------------------===//

// Runtime-supported maximum native kernarg byte length.
//
// Native kernarg offsets are represented as 16-bit byte offsets in the runtime
// dispatch layout. Code objects requiring larger kernarg packets need a future
// layout revision.
#define IREE_HAL_AMDGPU_KERNARG_LAYOUT_MAX_BYTE_LENGTH UINT16_MAX

// Sentinel value used when a kernel has no HIP/OpenCL implicit-argument suffix.
#define IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE UINT16_MAX

// Immutable native kernarg layout feature flags.
typedef enum iree_hal_amdgpu_kernarg_layout_flag_bits_e {
  // No layout flags are set.
  IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_NONE = 0u,
  // The native kernarg segment has a trailing HIP/OpenCL implicit-argument
  // suffix beginning at implicit_args_byte_offset.
  IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS = 1u << 0,
  // Kernarg storage must be zeroed before binding/constant emplacement.
  IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_REQUIRES_ZERO_FILL = 1u << 1,
  // Binding qword targets are a dense prefix matching binding ordinals.
  IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_PACKED_BINDING_PREFIX = 1u << 2,
  // Constant spans consume the constant stream as one contiguous source range.
  IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_CONTIGUOUS_CONSTANTS = 1u << 3,
} iree_hal_amdgpu_kernarg_layout_flag_bits_t;
typedef uint32_t iree_hal_amdgpu_kernarg_layout_flags_t;

// Mapping from one HAL binding ordinal to one native kernarg qword slot.
typedef struct iree_hal_amdgpu_kernarg_binding_slot_t {
  // Target qword index in the native kernarg segment.
  uint16_t target_qword_index;
} iree_hal_amdgpu_kernarg_binding_slot_t;

// Mapping from one HAL dispatch-constant byte slice to native kernarg bytes.
typedef struct iree_hal_amdgpu_kernarg_constant_span_t {
  // Target byte offset in the native kernarg segment.
  uint16_t target_byte_offset;
  // Source byte offset in the HAL dispatch constant stream.
  uint16_t source_byte_offset;
  // Byte length copied from the HAL dispatch constant stream.
  uint16_t byte_length;
} iree_hal_amdgpu_kernarg_constant_span_t;

// Native kernarg byte offsets of the HIP/OpenCL implicit ("hidden") argument
// fields the runtime synthesizes per dispatch.
//
// Each offset is the code object metadata's per-argument `.offset` for that
// hidden field, or IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE when the
// kernel does not declare the field (in which case it is left zero, matching
// the HIP default). This mirrors ROCr's per-field implicit-argument write
// (clr rocclr device/rocm/rocvirtual.cpp) rather than assuming a single fixed
// implicit-args struct at one base offset, which is invalid for hand-written
// assembly kernels (e.g. MIOpen Winograd) that declare only a partial hidden
// block. Undeclared fields (global offset, printf/hostcall buffers, remainder,
// and hidden_none padding) remain zero via the layout's zero-fill.
typedef struct iree_hal_amdgpu_kernarg_implicit_args_t {
  // Byte offset of hidden_block_count_x/y/z (uint32 each), or NONE.
  uint16_t block_count_offset[3];
  // Byte offset of hidden_group_size_x/y/z (uint16 each), or NONE.
  uint16_t group_size_offset[3];
  // Byte offset of hidden_grid_dims (uint16), or NONE.
  uint16_t grid_dims_offset;
  // Byte offset of hidden_dynamic_lds_size (uint32), or NONE.
  uint16_t dynamic_lds_size_offset;
} iree_hal_amdgpu_kernarg_implicit_args_t;

// Parameters used to build an immutable native kernarg layout.
typedef struct iree_hal_amdgpu_kernarg_layout_params_t {
  // Total native kernarg byte length reserved for dispatch.
  iree_host_size_t kernarg_byte_length;
  // Required native kernarg alignment in bytes.
  iree_host_size_t kernarg_alignment;
  // Number of bytes consumed from the HAL dispatch constant stream.
  iree_host_size_t constant_byte_length;
  // Native byte offset where the implicit-argument region begins (the lowest
  // hidden argument offset), or
  // IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE when the kernel has no
  // implicit arguments.
  iree_host_size_t implicit_args_byte_offset;
  // Number of entries in binding_slots.
  iree_host_size_t binding_count;
  // Dense binding ordinal to native kernarg qword target mappings.
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots;
  // Number of entries in constant_spans.
  iree_host_size_t constant_span_count;
  // Dense dispatch-constant stream to native kernarg byte mappings.
  const iree_hal_amdgpu_kernarg_constant_span_t* constant_spans;
  // Per-field native offsets of the synthesized implicit arguments. Only
  // meaningful when implicit_args_byte_offset != IMPLICIT_ARGS_NONE. The
  // producer must set every field the kernel does not declare to
  // IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE; a zero-initialized field
  // is interpreted as offset 0, not as absent.
  iree_hal_amdgpu_kernarg_implicit_args_t implicit_args;
} iree_hal_amdgpu_kernarg_layout_params_t;

// Immutable native kernarg layout for one executable export.
//
// The flexible binding_slots array is followed by constant_span_count
// iree_hal_amdgpu_kernarg_constant_span_t records. Use the accessors below
// instead of reconstructing table offsets at callsites.
typedef struct iree_hal_amdgpu_kernarg_layout_t {
  // Total native kernarg byte length reserved for dispatch.
  uint16_t kernarg_byte_length;
  // Required native kernarg alignment in bytes.
  uint16_t kernarg_alignment;
  // Number of HAL binding pointers consumed by the layout.
  uint16_t binding_count;
  // Number of constant span records following the binding table.
  uint16_t constant_span_count;
  // Number of bytes consumed from the HAL dispatch constant stream.
  uint16_t constant_byte_length;
  // Native byte offset where the implicit-argument region begins (the lowest
  // hidden argument offset), or
  // IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE when the kernel has no
  // implicit arguments. Used as the start of the region zeroed before per-field
  // implicit emplacement.
  uint16_t implicit_args_byte_offset;
  // Feature flags derived at layout construction time.
  iree_hal_amdgpu_kernarg_layout_flags_t flags;
  // Per-field native offsets of the synthesized implicit arguments. Only
  // meaningful when the IMPLICIT_ARGS flag is set.
  iree_hal_amdgpu_kernarg_implicit_args_t implicit_args;
  // Dense binding ordinal to native qword target mapping table.
  iree_hal_amdgpu_kernarg_binding_slot_t binding_slots[];
} iree_hal_amdgpu_kernarg_layout_t;

// Returns the binding slot table embedded in |layout|.
static inline const iree_hal_amdgpu_kernarg_binding_slot_t*
iree_hal_amdgpu_kernarg_layout_binding_slots(
    const iree_hal_amdgpu_kernarg_layout_t* layout) {
  return layout->binding_slots;
}

// Returns the constant span table embedded in |layout|.
static inline const iree_hal_amdgpu_kernarg_constant_span_t*
iree_hal_amdgpu_kernarg_layout_constant_spans(
    const iree_hal_amdgpu_kernarg_layout_t* layout) {
  return (
      const iree_hal_amdgpu_kernarg_constant_span_t*)(layout->binding_slots +
                                                      layout->binding_count);
}

// Calculates the byte length required to store a layout with the given counts.
iree_status_t iree_hal_amdgpu_kernarg_layout_storage_size(
    iree_host_size_t binding_count, iree_host_size_t constant_span_count,
    iree_host_size_t* out_storage_byte_length);

// Initializes |out_layout| in caller-provided storage.
//
// |storage_capacity| must be at least the value returned by
// iree_hal_amdgpu_kernarg_layout_storage_size for the parameter counts. The
// layout copies binding and constant tables from |params| and validates all
// source and target ranges before publishing derived flags.
iree_status_t iree_hal_amdgpu_kernarg_layout_initialize(
    const iree_hal_amdgpu_kernarg_layout_params_t* params,
    iree_host_size_t storage_capacity,
    iree_hal_amdgpu_kernarg_layout_t* out_layout);

// Populates explicit dispatch kernargs in already-reserved storage.
//
// |binding_ptrs| must provide |layout->binding_count| raw 64-bit device
// pointers. |constants| must provide |layout->constant_byte_length| bytes.
// |kernarg_ptr| must point to at least |layout->kernarg_byte_length| writable
// bytes. HIP/OpenCL implicit args are not populated by this function.
void iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
    const iree_hal_amdgpu_kernarg_layout_t* layout,
    const uint64_t* binding_ptrs, iree_const_byte_span_t constants,
    void* kernarg_ptr);

// Populates the HIP/OpenCL implicit ("hidden") kernargs in already-reserved
// storage, writing each declared implicit field at its own native offset.
//
// This zeroes the implicit region beginning at
// |layout|->implicit_args_byte_offset and then writes only the fields the code
// object metadata declared: hidden_block_count from |workgroup_count|,
// hidden_group_size from |workgroup_size|, hidden_grid_dims as 3, and
// hidden_dynamic_lds_size from |dynamic_workgroup_local_memory|. Undeclared
// fields (global offset, printf/hostcall buffers, remainder, padding) remain
// zero. Does nothing when |layout| is NULL or has no implicit arguments.
// |kernarg_ptr| must point to at least |layout|->kernarg_byte_length writable
// bytes.
void iree_hal_amdgpu_kernarg_layout_emplace_implicit_args(
    const iree_hal_amdgpu_kernarg_layout_t* layout,
    const uint32_t workgroup_count[3], const uint16_t workgroup_size[3],
    uint32_t dynamic_workgroup_local_memory, void* kernarg_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_KERNARG_LAYOUT_H_
