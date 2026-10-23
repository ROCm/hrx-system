// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent structural layout of verified C source signatures.

#ifndef LOOM_TARGET_ABI_C_SIGNATURE_LAYOUT_H_
#define LOOM_TARGET_ABI_C_SIGNATURE_LAYOUT_H_

#include "iree/base/internal/arena.h"
#include "loom/ops/func/source_signature.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_c_abi_leaf_flags_t;
enum loom_c_abi_leaf_flag_bits_e {
  // Every enclosing aggregate places this leaf at its required source
  // alignment. Platform classifiers can reject packed objects without
  // reconstructing the source type graph.
  LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED = 1u << 0,
  // Leaf is one source pointer projecting to a buffer base and byte offset.
  LOOM_C_ABI_LEAF_FLAG_POINTER = 1u << 1,
  // Leaf is one native vector value.
  LOOM_C_ABI_LEAF_FLAG_VECTOR = 1u << 2,
  // Leaf has C Boolean semantics and must normalize loaded bits to zero or one.
  LOOM_C_ABI_LEAF_FLAG_BOOLEAN = 1u << 3,
};

// One terminal scalar, vector, or pointer occurrence inside a root C object.
// Pointers remain one source leaf while projecting to two logical Loom values:
// a buffer base and byte offset.
typedef struct loom_c_abi_leaf_t {
  // Byte offset from the beginning of the root source object.
  uint64_t byte_offset;
  // Exact source object footprint of this leaf in bytes.
  uint64_t byte_length;
  // Source type table index identifying the terminal type and semantics.
  uint32_t type_index;
  // First flattened logical Loom value carrying this leaf.
  uint16_t logical_offset;
  // Number of logical Loom values carrying this leaf.
  uint8_t logical_count;
  // loom_c_abi_leaf_flag_bits_e bitset.
  loom_c_abi_leaf_flags_t flags;
} loom_c_abi_leaf_t;
static_assert(sizeof(loom_c_abi_leaf_t) == 24,
              "C ABI source leaves must be exactly 24 bytes");

// Structural layout of one root C argument or result object.
typedef struct loom_c_abi_object_t {
  // Exact source object footprint in bytes.
  uint64_t byte_length;
  // Required source object alignment in bytes.
  uint64_t byte_alignment;
  // Root source type table index.
  uint32_t type_index;
  // First flattened logical Loom value carrying this object.
  uint16_t logical_offset;
  // Number of flattened logical Loom values carrying this object.
  uint16_t logical_count;
  // First leaf in the corresponding argument or result leaf array.
  uint16_t leaf_offset;
  // Number of terminal leaves contained by this object.
  uint16_t leaf_count;
} loom_c_abi_object_t;
static_assert(sizeof(loom_c_abi_object_t) == 32,
              "C ABI source objects must be exactly 32 bytes");

// Normalized structural facts for one verified C source signature. Source
// attribute arrays borrow immutable module storage; all object and leaf arrays
// belong to the caller-provided result arena.
typedef struct loom_c_abi_signature_layout_t {
  // Retained source signature attribute slices.
  loom_func_c_abi_signature_t signature;
  // Argument objects in source signature order.
  const loom_c_abi_object_t* arguments;
  // Result objects in source signature order.
  const loom_c_abi_object_t* results;
  // Argument leaves in flattened logical order.
  const loom_c_abi_leaf_t* argument_leaves;
  // Result leaves in flattened logical order.
  const loom_c_abi_leaf_t* result_leaves;
  // Total number of argument leaves.
  uint16_t argument_leaf_count;
  // Total number of result leaves.
  uint16_t result_leaf_count;
  // Total number of flattened logical argument values.
  uint16_t argument_logical_count;
  // Total number of flattened logical result values.
  uint16_t result_logical_count;
} loom_c_abi_signature_layout_t;

// Normalizes one verified C source signature into root objects and terminal
// leaves. The scratch arena holds only type-table leaf counts; the result arena
// owns every retained array. Verification has already established source type
// indices, object bounds, nesting depth, and 16-bit logical component totals,
// so allocation is the only fallible operation.
iree_status_t loom_c_abi_signature_layout_build(
    const loom_func_c_abi_signature_t* signature,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    loom_c_abi_signature_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ABI_C_SIGNATURE_LAYOUT_H_
