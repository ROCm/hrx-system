// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL-kernel ABI layout over target-low resources.
//
// This layer derives kernarg storage from function-local low.resource imports.
// It intentionally stays below LLVMIR/native artifact emission so the same
// resource ABI can feed the temporary assembly path, direct HSACO writing, and
// future backends.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernarg storage for one HAL binding pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE 8u

// Required kernarg alignment for HAL binding pointers.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT 8u

// Kernarg storage for one 32-bit direct dispatch constant word.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE 4u

// Required kernarg alignment for direct dispatch constant words.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT 4u

// Stable low.live_in source spelling for the AMDGPU kernarg segment pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE \
  "amdgpu.kernarg_segment_ptr"

// Stable low.live_in source ID for the AMDGPU kernarg segment pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE_ID \
  UINT64_C(0x7C8A03858206FDDC)

// Stable low.live_in source spelling for the AMDGPU dispatch packet pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_PTR_SOURCE "amdgpu.dispatch_ptr"

// Stable low.live_in source ID for the AMDGPU dispatch packet pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_PTR_SOURCE_ID \
  UINT64_C(0x36D73B4B3758D0B2)

// Stable low.live_in source spelling for workgroup_id.x in the first system
// SGPR after enabled user SGPRs.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE "amdgpu.workgroup_id.x"

// Stable low.live_in source ID for workgroup_id.x.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE_ID \
  UINT64_C(0x64E1C4EA699CDCC3)

// Stable low.live_in source spelling for workgroup_id.y after enabled lower
// workgroup-id dimensions.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE "amdgpu.workgroup_id.y"

// Stable low.live_in source ID for workgroup_id.y.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE_ID \
  UINT64_C(0x64E1C3EA699CDB10)

// Stable low.live_in source spelling for workgroup_id.z after enabled lower
// workgroup-id dimensions.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE "amdgpu.workgroup_id.z"

// Stable low.live_in source ID for workgroup_id.z.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE_ID \
  UINT64_C(0x64E1C6EA699CE029)

// Stable low.live_in source spelling for workitem_id.x in v0.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE "amdgpu.workitem_id.x"

// Stable low.live_in source ID for workitem_id.x.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE_ID \
  UINT64_C(0x599D0AE7D922CE17)

// Stable low.live_in source spelling for workitem_id.y in v1 on targets that
// expose unpacked workitem-id VGPRs.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE "amdgpu.workitem_id.y"

// Stable low.live_in source ID for workitem_id.y.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE_ID \
  UINT64_C(0x599D09E7D922CC64)

// Stable low.live_in source spelling for workitem_id.z in v2 on targets that
// expose unpacked workitem-id VGPRs.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE "amdgpu.workitem_id.z"

// Stable low.live_in source ID for workitem_id.z.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE_ID \
  UINT64_C(0x599D0CE7D922D17D)

// Stable low.live_in source spelling for targets that pack workitem_id.x/y into
// v0. Lowering must unpack logical dimensions before ordinary use.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE \
  "amdgpu.workitem_id.packed.xy"

// Stable low.live_in source ID for packed workitem_id.x/y.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XY_SOURCE_ID \
  UINT64_C(0x40BB6CD7335467E2)

// Stable low.live_in source spelling for targets that pack workitem_id.x/y/z
// into v0. Lowering must unpack logical dimensions before ordinary use.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE \
  "amdgpu.workitem_id.packed.xyz"

// Stable low.live_in source ID for packed workitem_id.x/y/z.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_PACKED_XYZ_SOURCE_ID \
  UINT64_C(0x52E189AC386C0748)

// Stable low.live_in source spelling for the M0 special register.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE "amdgpu.m0"

// Stable low.live_in source ID for the M0 special register.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE_ID UINT64_C(0x0667779E351A470C)

typedef struct loom_amdgpu_hal_kernarg_resource_t {
  // Defining low.resource op for diagnostics and cross-checks.
  const loom_op_t* resource_op;
  // Arena-owned resource name used in emitted AMDGPU metadata.
  iree_string_view_t name;
  // HAL binding ordinal used by the runtime dispatch path.
  uint32_t binding_index;
  // Byte offset of the pointer entry in the kernarg segment.
  uint32_t kernarg_offset;
  // Byte length of the pointer entry in the kernarg segment.
  uint32_t kernarg_size;
  // Byte alignment of the pointer entry in the kernarg segment.
  uint32_t kernarg_alignment;
  // Source/storage semantic type declared by the resource record.
  loom_type_t source_type;
  // Target-low value type produced by low.resource for this binding.
  loom_type_t abi_type;
} loom_amdgpu_hal_kernarg_resource_t;

typedef struct loom_amdgpu_hal_kernarg_direct_arg_t {
  // Entry block argument value loaded from the HAL constant segment.
  loom_value_id_t arg_id;
  // Metadata name copied from the entry block argument when present.
  iree_string_view_t name;
  // Entry block argument index before ABI materialization removes arguments.
  uint16_t argument_index;
  // Byte offset of the direct argument in the kernarg segment.
  uint32_t kernarg_offset;
  // Byte length of the direct argument in the kernarg segment.
  uint32_t kernarg_size;
  // Byte alignment of the direct argument in the kernarg segment.
  uint32_t kernarg_alignment;
  // Target-low value type produced by materializing this argument.
  loom_type_t abi_type;
} loom_amdgpu_hal_kernarg_direct_arg_t;

typedef struct loom_amdgpu_hal_kernel_abi_layout_t {
  // Target-low function operation whose resources are laid out.
  const loom_op_t* function_op;
  // Total kernarg segment size in bytes.
  uint32_t kernarg_segment_size;
  // Required kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // True when the kernel descriptor must request the kernarg segment pointer.
  bool uses_kernarg_segment_ptr;
  // HAL dispatch constant word count consumed by direct arguments.
  uint32_t constant_count;
  // Resource records in HAL binding/kernarg offset order.
  const loom_amdgpu_hal_kernarg_resource_t* resources;
  // Number of resource records in |resources|.
  iree_host_size_t resource_count;
  // Direct argument records in entry block argument order.
  const loom_amdgpu_hal_kernarg_direct_arg_t* direct_args;
  // Number of direct argument records in |direct_args|.
  iree_host_size_t direct_arg_count;
} loom_amdgpu_hal_kernel_abi_layout_t;

typedef struct loom_amdgpu_hal_kernel_abi_verify_result_t {
  // Number of AMDGPU HAL-kernel ABI errors emitted for the function.
  uint32_t error_count;
} loom_amdgpu_hal_kernel_abi_verify_result_t;

// Emits AMDGPU HAL-kernel ABI diagnostics for |function_op|.
//
// Status is reserved for diagnostic emission and table-allocation failures.
// User-visible ABI violations are emitted through |emitter| and counted in
// |out_result|.
iree_status_t loom_amdgpu_hal_kernel_abi_verify_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set, uint32_t max_errors,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_hal_kernel_abi_verify_result_t* out_result,
    iree_arena_allocator_t* arena);

// Derives the AMDGPU HAL-kernel ABI layout for |function_op|.
//
// Supports low.resource imports with kind hal_binding, dense unique binding
// indexes starting at zero, and result type reg<amdgpu.sgpr x2>. The
// source_type attribute records the high-level resource handle type, but the
// AMDGPU ABI layout is determined by the import kind and target-low result
// type. The kernarg segment stores one 64-bit global pointer per binding in
// binding-index order, then tightly packed 32-bit direct constant words for
// remaining entry block arguments. Later lowering materializes the target
// buffer descriptor value consumed by packets that need one.
//
// The function must already have passed
// loom_amdgpu_hal_kernel_abi_verify_low. Status is reserved for allocation and
// contract misuse that would otherwise make layout construction unsafe.
iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena);

// Returns true when |function_op| carries a prepared AMDGPU HAL ABI layout.
bool loom_amdgpu_hal_kernel_abi_has_layout_attr(const loom_op_t* function_op);

// Builds a structured ABI layout snapshot attribute for a prepared
// low.kernel.def.
//
// AMDGPU HAL binding materialization consumes low.resource ops and may remove
// entry block arguments, so later emission cannot recover binding metadata by
// walking the rewritten body. The snapshot is owned by the kernel op rather
// than a companion metadata op to preserve function-local pass semantics.
iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr);

// Loads a prepared AMDGPU HAL ABI layout snapshot from low.kernel.def.
iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_attr(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout,
    iree_arena_allocator_t* arena);

// Returns true if |value_id| is defined by the kernarg segment pointer live-in.
bool loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the dispatch packet pointer live-in.
bool loom_amdgpu_hal_kernel_abi_is_dispatch_ptr_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workgroup_id.x live-in.
bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_x_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workgroup_id.y live-in.
bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_y_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workgroup_id.z live-in.
bool loom_amdgpu_hal_kernel_abi_is_workgroup_id_z_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workitem_id.x live-in.
bool loom_amdgpu_hal_kernel_abi_is_workitem_id_x_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workitem_id.y live-in.
bool loom_amdgpu_hal_kernel_abi_is_workitem_id_y_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the workitem_id.z live-in.
bool loom_amdgpu_hal_kernel_abi_is_workitem_id_z_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the packed workitem_id.x/y live-in.
bool loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xy_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the packed workitem_id.x/y/z
// live-in.
bool loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xyz_live_in(
    const loom_module_t* module, loom_value_id_t value_id);

// Returns true if |value_id| is defined by the M0 special-register live-in.
bool loom_amdgpu_hal_kernel_abi_is_m0_live_in(const loom_module_t* module,
                                              loom_value_id_t value_id);

// Finds AMDGPU ABI live-ins that require fixed physical locations during
// allocation.
//
// The returned array is arena-owned. The current ABI fixes hidden user SGPRs in
// AMDHSA order: dispatch pointer, then kernarg segment pointer. Each pointer
// consumes two SGPRs starting at the next available user-SGPR location.
// workgroup_id.x/y/z live-ins use the SGPRs immediately following enabled user
// SGPRs. Unpacked workitem_id.x/y/z live-ins use v0/v1/v2, and packed
// workitem-id live-ins use v0 when present.
//
// The function must already have passed
// loom_amdgpu_hal_kernel_abi_verify_low. Status is reserved for allocation and
// contract misuse that would otherwise make fixed-value construction unsafe.
iree_status_t loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_fixed_value_t** out_fixed_values,
    iree_host_size_t* out_fixed_value_count, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_
