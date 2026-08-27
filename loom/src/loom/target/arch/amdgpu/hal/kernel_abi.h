// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL-kernel ABI layout over authored kernel signatures.
//
// Source lowering records the declared kernel.def parameter sequence in a
// low.kernel.def ABI snapshot. Function-local low.resource imports and low
// entry-block arguments are use sites attached to that declaration; they do not
// define the exported ABI shape after cleanup removes unused values. This layer
// stays below LLVMIR/native artifact emission so the same ABI can feed the
// temporary assembly path, direct HSACO writing, and future backends.

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

// Stable low.live_in source spelling for the AMDGPU dispatch packet ID.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_ID_SOURCE "amdgpu.dispatch_id"

// Stable low.live_in source ID for the AMDGPU dispatch packet ID.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DISPATCH_ID_SOURCE_ID \
  UINT64_C(0x186DBCDA81EDC8D5)

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

// Stable low.live_in source spelling for the packed cluster-local workgroup
// coordinates and cluster extent in gfx1250 TTMP6.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_SOURCE \
  "amdgpu.cluster_workgroup_info"

// Stable low.live_in source ID for the packed cluster workgroup information.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_SOURCE_ID \
  UINT64_C(0x7A5A999A4B91578F)

// Stable low.live_in source spelling for gfx1250 TTMP6 when only the x
// cluster-local workgroup coordinate is enabled.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_X_SOURCE \
  "amdgpu.cluster_workgroup_info.x"

// Stable low.live_in source ID for x-only packed cluster workgroup state.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_X_SOURCE_ID \
  UINT64_C(0x11B87FDAAA3D6151)

// Stable low.live_in source spelling for gfx1250 TTMP6 when the x and y
// cluster-local workgroup coordinates are enabled.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_XY_SOURCE \
  "amdgpu.cluster_workgroup_info.xy"

// Stable low.live_in source ID for x/y packed cluster workgroup state.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_XY_SOURCE_ID \
  UINT64_C(0x59E2688F464C16F8)

// Stable low.live_in source spelling for gfx1250 TTMP6 when the x and z
// cluster-local workgroup coordinates are enabled.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_XZ_SOURCE \
  "amdgpu.cluster_workgroup_info.xz"

// Stable low.live_in source ID for x/z packed cluster workgroup state.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_WORKGROUP_INFO_XZ_SOURCE_ID \
  UINT64_C(0x59E26B8F464C1C11)

// Stable low.live_in source spelling for the packed architected y/z
// workgroup or cluster coordinates in TTMP7.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_YZ_SOURCE "amdgpu.cluster_id_yz"

// Stable low.live_in source ID for packed cluster_id.y/z.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_YZ_SOURCE_ID \
  UINT64_C(0x5CC9CA25D8E49D3B)

// Stable low.live_in source spelling for the architected y workgroup or
// cluster coordinate in TTMP7.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_Y_SOURCE "amdgpu.cluster_id_y"

// Stable low.live_in source ID for cluster_id.y.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_Y_SOURCE_ID \
  UINT64_C(0x68A4B9A3B10CE223)

// Stable low.live_in source spelling for the architected z workgroup or
// cluster coordinate in TTMP7.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_Z_SOURCE "amdgpu.cluster_id_z"

// Stable low.live_in source ID for cluster_id.z.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_Z_SOURCE_ID \
  UINT64_C(0x68A4BAA3B10CE3D6)

// Stable low.live_in source spelling for the architected x workgroup or
// cluster coordinate in TTMP9.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_X_SOURCE "amdgpu.cluster_id_x"

// Stable low.live_in source ID for cluster_id.x.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_CLUSTER_ID_X_SOURCE_ID \
  UINT64_C(0x68A4B8A3B10CE070)

typedef enum loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flag_bits_e {
  // Dispatch must initialize the x workgroup-coordinate launch state.
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X = 1u << 0,
  // Dispatch must initialize the y workgroup-coordinate launch state.
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y = 1u << 1,
  // Dispatch must initialize the z workgroup-coordinate launch state.
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z = 1u << 2,
  // Workgroup-coordinate launch-state flags known by the AMDGPU HAL ABI.
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_KNOWN_FLAGS =
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_X |
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Y |
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAUNCH_WORKGROUP_ID_Z,
} loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flag_bits_t;

// Bitset of loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flag_bits_t values.
typedef uint32_t loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t;

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

// Stable source kinds carried by AMDGPU ABI low.live_in ops.
typedef enum loom_amdgpu_hal_kernel_abi_source_kind_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_UNKNOWN = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR = 2,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID = 3,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X = 4,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Y = 5,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z = 6,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X = 7,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Y = 8,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z = 9,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY = 10,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ = 11,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_M0 = 12,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO = 13,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_YZ = 14,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_X = 15,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_Y = 16,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_ID_Z = 17,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_X = 18,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_XY = 19,
  LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_CLUSTER_WORKGROUP_INFO_XZ = 20,
} loom_amdgpu_hal_kernel_abi_source_kind_t;

// Maximum number of fixed physical values required by one AMDGPU HAL kernel
// ABI. This covers hidden user SGPRs, three coordinate dimensions, packed
// workitem state, M0, and the fixed architected launch-state sources.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_FIXED_VALUE_COUNT (8u + 2u * 3u)

// Returns the stable low.live_in source spelling for |source_kind|, or an
// empty string for unknown/invalid kinds.
iree_string_view_t loom_amdgpu_hal_kernel_abi_source_name(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind);

typedef struct loom_amdgpu_hal_kernarg_resource_t {
  // Defining low.resource op for diagnostics and cross-checks.
  const loom_op_t* resource_op;
  // Arena-owned resource name used in emitted AMDGPU metadata.
  iree_string_view_t name;
  // HAL binding ordinal used by the runtime dispatch path.
  uint32_t binding_index;
  // Authored source kernel parameter index for native metadata ordering.
  uint32_t parameter_index;
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
  // Authored source kernel parameter index for native metadata ordering.
  uint32_t parameter_index;
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
  // Authored source kernel parameter count represented by this ABI layout.
  uint32_t parameter_count;
  // Total kernarg segment size in bytes.
  uint32_t kernarg_segment_size;
  // Required kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // True when surviving parameter uses require the kernarg segment pointer.
  // Source lowering may initialize this conservatively; HAL ABI materialization
  // refines it after low-level dead-code elimination.
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
  // Bitset of verified ABI source kinds present in the function.
  uint64_t live_in_source_bits;
  // Launch workgroup-coordinate state required by the verified low function.
  loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t
      launch_workgroup_id_flags;
  // Number of hidden user SGPRs consumed by verified ABI live-ins.
  uint32_t user_sgpr_count;
  // Fixed physical values retained from verified ABI live-ins.
  loom_low_allocation_fixed_value_t
      fixed_values[LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_FIXED_VALUE_COUNT];
  // Number of populated entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
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
// Source-lowered low.kernel.def ops carry a prepared ABI snapshot preserving
// the source parameter count and native kernarg offsets. Direct low IR without
// a snapshot falls back to dense low.resource imports with kind hal_binding,
// unique binding indexes starting at zero, and result type reg<amdgpu.sgpr x2>.
// Later lowering materializes the target buffer descriptor value consumed by
// packets that need one.
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

// Returns the ABI source kind for |value_id| when it is defined by an AMDGPU
// low.live_in op, or UNKNOWN otherwise.
loom_amdgpu_hal_kernel_abi_source_kind_t
loom_amdgpu_hal_kernel_abi_live_in_source_kind(const loom_module_t* module,
                                               loom_value_id_t value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_H_
