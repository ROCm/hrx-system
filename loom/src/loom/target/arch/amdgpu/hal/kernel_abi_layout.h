// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL kernel parameter layout and its authored ABI dictionary.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// ABI resource arrays use the attribute representation's 16-bit count.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT UINT16_MAX

// Kernarg storage for one HAL binding pointer.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE 8u

// Required kernarg alignment for HAL binding pointers.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT 8u

// Kernarg storage for one 32-bit direct dispatch constant word. Logical
// payloads narrower than a word occupy its low-address bytes. Padding is
// unspecified; source lowering establishes each consumer's carrier contract.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE 4u

// Required kernarg alignment for direct dispatch constant words.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT 4u

// Maximum physical word count carried by one direct dispatch argument. This
// covers the widest source vector representation admitted by AMDGPU lowering.
#define LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT 32u

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
  // Entry block argument loaded from the HAL constant segment, or INVALID
  // when only its ABI metadata remains after argument materialization.
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
  // Byte length consumed from the dense HAL dispatch constant stream.
  uint32_t payload_size;
  // Byte alignment of the direct argument in the kernarg segment.
  uint32_t kernarg_alignment;
  // Target-low value type produced by materializing this argument.
  loom_type_t abi_type;
} loom_amdgpu_hal_kernarg_direct_arg_t;

// HSA V5/V6 implicit launch arguments follow the explicitly authored arguments.
// The low three bits select complete group counts, the next three partial
// sizes.
enum {
  LOOM_AMDGPU_HAL_BLOCK_COUNT_X = 0,
  LOOM_AMDGPU_HAL_REMAINDER_X = 3,
  LOOM_AMDGPU_HAL_GEOMETRY_ARG_COUNT = 6,
};

static inline uint32_t loom_amdgpu_hal_geometry_arg_offset(uint32_t field) {
  return field < LOOM_AMDGPU_HAL_REMAINDER_X
             ? field * 4
             : 18 + (field - LOOM_AMDGPU_HAL_REMAINDER_X) * 2;
}

static inline uint32_t loom_amdgpu_hal_geometry_arg_size(uint32_t field) {
  return field < LOOM_AMDGPU_HAL_REMAINDER_X ? 4 : 2;
}

// Byte extent up to the last requested implicit field, aligned like kernarg.
static inline uint32_t loom_amdgpu_hal_geometry_byte_length(uint32_t fields) {
  if (!fields) {
    return 0;
  }
  const uint32_t last = 31u - iree_math_count_leading_zeros_u32(fields);
  return (uint32_t)iree_align_uint64(
      loom_amdgpu_hal_geometry_arg_offset(last) +
          loom_amdgpu_hal_geometry_arg_size(last),
      8);
}

typedef struct loom_amdgpu_hal_kernel_abi_layout_t {
  // Target-low function operation whose resources are laid out.
  const loom_op_t* function_op;
  // Authored source kernel parameter count represented by this ABI layout.
  uint32_t parameter_count;
  // Aligned byte extent of the explicit parameters and base of HSA implicit
  // arguments.
  uint32_t explicit_kernarg_size;
  // Required kernarg segment alignment in bytes.
  uint32_t kernarg_segment_alignment;
  // True when the entry reads explicit or implicit kernel arguments.
  // Source lowering may initialize this conservatively; HAL ABI materialization
  // refines it after low-level dead-code elimination.
  bool uses_kernarg_segment_ptr;
  // HSA implicit launch fields consumed by the lowered entry point.
  uint8_t geometry_args;
  // Physical kernarg word count occupied by direct arguments.
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

// Returns whether the kernel declares an ABI parameter layout.
bool loom_amdgpu_hal_kernel_abi_has_layout_attr(const loom_op_t* function_op);

// Encodes an admitted layout as the kernel's stable parameter contract.
iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr);

// Admits an authored ABI dictionary and retains its decoded layout. Rejection
// emits a source diagnostic and leaves |out_accepted| false, including without
// a callback. Status represents allocation or diagnostic sink failure.
iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_attr(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_accepted, loom_amdgpu_hal_kernel_abi_layout_t* out_layout);

// Derives the dense layout for a function with no ABI dictionary. The caller
// has verified live argument types and the resource index domain. Authored
// dictionary layouts are retained directly by verification. Status represents
// arena allocation failure.
iree_status_t loom_amdgpu_hal_kernel_abi_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_KERNEL_ABI_LAYOUT_H_
