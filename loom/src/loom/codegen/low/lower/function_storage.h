// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded source allocations backed by structural Low function storage.

#ifndef LOOM_CODEGEN_LOW_LOWER_FUNCTION_STORAGE_H_
#define LOOM_CODEGEN_LOW_LOWER_FUNCTION_STORAGE_H_

#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;

typedef enum loom_low_lower_function_storage_source_e {
  // Byte-addressable buffer allocation.
  LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_BUFFER = 0,
  // Typed allocation with scalar elements.
  LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_SCALAR_VIEW,
  // Typed allocation with opaque reference elements.
  LOOM_LOW_LOWER_FUNCTION_STORAGE_SOURCE_REFERENCE_VIEW,
} loom_low_lower_function_storage_source_t;

// Target capability for a source allocation space. This is independent of the
// final physical placement, which belongs to the target's emission frame.
typedef struct loom_low_lower_function_storage_mapping_t {
  // Source scratch-root space accepted by this mapping.
  loom_value_fact_memory_space_t memory_space;
  // Structural Low reservation space preserving the source ownership scope.
  loom_storage_space_t storage_space;
  // One-unit descriptor-set register class holding the reservation's address.
  uint16_t address_register_class;
  // Source representation admitted by this mapping. Typed reference cells
  // can occupy a different physical bank from byte-addressable storage.
  loom_low_lower_function_storage_source_t source;
} loom_low_lower_function_storage_mapping_t;

typedef struct loom_low_lower_function_storage_config_t {
  // Borrowed immutable mappings, or NULL when allocations use another path.
  const loom_low_lower_function_storage_mapping_t* mappings;
  // Number of mappings, with at most one per source kind and memory space.
  iree_host_size_t count;
} loom_low_lower_function_storage_config_t;

// Function-arena-owned decision consumed directly by Low construction.
typedef struct loom_low_lower_function_storage_plan_t {
  // Proven finite maximum byte length of the source allocation.
  int64_t byte_length;
  // Source-required power-of-two base alignment.
  int64_t byte_alignment;
  // Structural Low space selected by the target capability.
  loom_storage_space_t storage_space;
} loom_low_lower_function_storage_plan_t;

// Proves that a fixed slot preserves the allocation's per-execution identity.
// Roots outside repeated control require only indexed execution facts. Repeated
// roots share the function's retained storage analysis with physical packing.
// An unproven lifetime emits a target diagnostic and sets |out_supported|
// false; status reports construction or diagnostic-sink failure.
iree_status_t loom_low_lower_function_storage_check_lifetime(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_supported);

// Selects one buffer.alloca or view.alloca using the active policy and retained
// facts. An unsupported space remains unselected for other lowering mechanisms.
// An invalid extent or lifetime is selected but diagnosed, with a NULL plan;
// the shared diagnostic boundary prevents emission of that function.
iree_status_t loom_low_lower_function_storage_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected,
    const loom_low_lower_function_storage_plan_t** out_plan);

// Emits the retained reservation/address pair using the source buffer's
// selected binding type. No extent operand is queried during emission.
iree_status_t loom_low_lower_function_storage_emit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_function_storage_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_FUNCTION_STORAGE_H_
