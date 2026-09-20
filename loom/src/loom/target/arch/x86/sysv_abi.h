// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 platform ABI classification retained in target-low function layouts.

#ifndef LOOM_TARGET_ARCH_X86_SYSV_ABI_H_
#define LOOM_TARGET_ARCH_X86_SYSV_ABI_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// SysV x86-64 architectural GPR ordinals shared by the target descriptor
// views. These are boundary locations, not allocation preferences.
enum {
  LOOM_X86_SYSV_GPR_RAX = 0,
  LOOM_X86_SYSV_GPR_RCX = 1,
  LOOM_X86_SYSV_GPR_RDX = 2,
  LOOM_X86_SYSV_GPR_RBX = 3,
  LOOM_X86_SYSV_GPR_RSP = 4,
  LOOM_X86_SYSV_GPR_RBP = 5,
  LOOM_X86_SYSV_GPR_RSI = 6,
  LOOM_X86_SYSV_GPR_RDI = 7,
  LOOM_X86_SYSV_GPR_R8 = 8,
  LOOM_X86_SYSV_GPR_R9 = 9,
  LOOM_X86_SYSV_GPR_R10 = 10,
  LOOM_X86_SYSV_GPR_R11 = 11,
  LOOM_X86_SYSV_GPR_R12 = 12,
  LOOM_X86_SYSV_GPR_R13 = 13,
  LOOM_X86_SYSV_GPR_R14 = 14,
  LOOM_X86_SYSV_GPR_R15 = 15,
};

// Decoded scalar SysV ABI layout for one target-low function signature.
//
// Locations are serialized as signed integers. Nonnegative values are
// physical-register ordinals in the corresponding Low value's register class.
// Negative values encode outgoing byte offsets from the caller's pre-call
// stack pointer as -(offset + 1); the callee observes the same arguments eight
// bytes higher after the return address is pushed.
typedef struct loom_x86_sysv_abi_layout_t {
  // Argument locations in signature order.
  const int64_t* argument_locations;
  // Number of entries in |argument_locations|.
  iree_host_size_t argument_count;
  // Result locations in signature order.
  const int64_t* result_locations;
  // Number of entries in |result_locations|.
  iree_host_size_t result_count;
  // Bytes occupied by the caller's outgoing stack-argument area.
  uint32_t stack_argument_bytes;
} loom_x86_sysv_abi_layout_t;

// Returns the serialized location for a stack byte offset.
static inline int64_t loom_x86_sysv_abi_stack_location(uint32_t byte_offset) {
  return -(int64_t)byte_offset - 1;
}

// Returns true when |location| names a physical register.
static inline bool loom_x86_sysv_abi_location_is_register(int64_t location) {
  return location >= 0;
}

// Decodes the stack byte offset in a negative serialized location.
static inline uint32_t loom_x86_sysv_abi_stack_location_offset(
    int64_t location) {
  IREE_ASSERT(location < 0);
  return (uint32_t)-(location + 1);
}

// Classifies a one-unit GPR signature according to the x86-64 SysV ABI.
// Unsupported signatures return OK with |out_supported| false and an empty
// layout. Arrays in a supported layout are owned by |arena|.
iree_status_t loom_x86_sysv_abi_layout_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    iree_arena_allocator_t* arena, loom_x86_sysv_abi_layout_t* out_layout,
    bool* out_supported);

// Materializes the canonical target-low ABI dictionary for |layout|.
iree_status_t loom_x86_sysv_abi_layout_make_attr(
    loom_module_t* module, const loom_x86_sysv_abi_layout_t* layout,
    loom_attribute_t* out_attr);

// Decodes and validates an authored ABI dictionary against its Low signature.
// The returned arrays borrow the canonical module attribute storage.
iree_status_t loom_x86_sysv_abi_layout_parse(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_slice_t attrs, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, iree_arena_allocator_t* scratch_arena,
    loom_x86_sysv_abi_layout_t* out_layout);

// Decodes and validates the retained ABI layout on |function_op| against its
// Low signature. |function_op| must be low.func.def or low.func.decl. The
// returned arrays borrow the canonical module attribute storage.
iree_status_t loom_x86_sysv_abi_function_layout_parse(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* function_op, iree_arena_allocator_t* scratch_arena,
    loom_x86_sysv_abi_layout_t* out_layout);

// Returns true when |physical_register| is clobbered by a SysV call.
bool loom_x86_sysv_gpr_is_caller_saved(uint32_t physical_register);

// Returns true when |physical_register| must be preserved by a SysV callee.
bool loom_x86_sysv_gpr_is_callee_saved(uint32_t physical_register);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_SYSV_ABI_H_
