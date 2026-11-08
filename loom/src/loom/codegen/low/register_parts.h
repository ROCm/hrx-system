// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_REGISTER_PARTS_H_
#define LOOM_CODEGEN_LOW_REGISTER_PARTS_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_register_part_continuation_t
    loom_low_register_part_continuation_t;

// Returns the declared parts for a verified register-class alternative.
static inline uint32_t loom_low_register_parts_operand_mask(
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_operand_t* operand, uint16_t register_class_id) {
  const loom_low_reg_class_alt_t* alternative =
      loom_low_operand_reg_class_alt(descriptors, operand, register_class_id);
  return alternative->register_part_id == LOOM_LOW_REGISTER_PART_NONE
             ? descriptors->reg_classes[register_class_id]
                   .full_register_part_mask
             : descriptors->register_parts[alternative->register_part_id].mask;
}

// A use whose required parts were not yet available during the defining walk.
// All diagnostic metadata is borrowed from the verified module/descriptor set.
typedef struct loom_low_register_part_requirement_t {
  // Operation containing the use.
  const loom_op_t* op;
  // Operation or descriptor name for diagnostics.
  iree_string_view_t op_name;
  // Operand field name for diagnostics.
  iree_string_view_t field_name;
  // Operand field location for diagnostics.
  loom_diagnostic_field_ref_t field_ref;
  // Value supplying the required register parts.
  loom_value_id_t value;
  // Parts that must be present in the value's final defined mask.
  uint32_t mask;
} loom_low_register_part_requirement_t;

// Function-local register-part facts used by verification and allocation.
// The caller publishes each value's written mask once and records its single
// tied source when present. Missing use facts are deferred, not diagnosed from
// traversal order. The caller owns zeroed mask storage and the arena; masks
// use module scratch for verification or a local domain for allocation.
typedef struct loom_low_register_parts_t {
  // Value-indexed masks, borrowed for the complete verification invocation.
  loom_value_u32_scratch_t* masks;
  // Acquired local domain when compiler analysis owns the module ordinal map.
  // NULL selects the verifier's module scratch above.
  const loom_local_value_domain_t* value_domain;
  // Local-ordinal masks when |value_domain| is non-NULL.
  uint32_t* local_masks;
  // Function-lifetime arena for sparse continuations and requirements.
  iree_arena_allocator_t* arena;
  // Incomplete tied-result dependencies retained during the defining walk.
  struct {
    // Sparse result-to-source continuation records.
    loom_low_register_part_continuation_t* values;
    // Number of retained dependencies.
    iree_host_size_t count;
    // Allocated continuation entries.
    iree_host_size_t capacity;
  } continuations;
  // Uses requiring a check after all defining facts are available.
  struct {
    // Original use facts and their borrowed diagnostic locations.
    loom_low_register_part_requirement_t* values;
    // Number of deferred use checks.
    iree_host_size_t count;
    // Allocated requirement entries.
    iree_host_size_t capacity;
  } requirements;
} loom_low_register_parts_t;

// Returns the currently known mask for a structurally verified value ID.
static inline uint32_t loom_low_register_parts_mask(
    const loom_low_register_parts_t* parts, loom_value_id_t value) {
  if (parts->value_domain != NULL) {
    return parts->local_masks[loom_local_value_domain_ordinal(
        parts->value_domain, value)];
  }
  return loom_value_u32_scratch_load(parts->masks, value);
}

// Publishes a definition with no inherited register parts.
static inline void loom_low_register_parts_define(
    loom_low_register_parts_t* parts, loom_value_id_t value, uint32_t mask) {
  if (parts->value_domain != NULL) {
    parts->local_masks[loom_local_value_domain_ordinal(parts->value_domain,
                                                       value)] = mask;
    return;
  }
  loom_value_u32_scratch_store(parts->masks, value, mask);
}

// Publishes written parts plus parts preserved from |source|. A full mask needs
// no retained dependency. Each result has at most one continuation source.
iree_status_t loom_low_register_parts_continue(loom_low_register_parts_t* parts,
                                               loom_value_id_t result,
                                               loom_value_id_t source,
                                               uint32_t written_mask,
                                               uint32_t full_mask);

// Retains only requirements not satisfied by already known parts. Requirements
// never alter a definition's mask; the caller diagnoses remaining failures
// after resolve, using the retained metadata and final value masks.
iree_status_t loom_low_register_parts_require(
    loom_low_register_parts_t* parts,
    const loom_low_register_part_requirement_t* requirement);

// Resolves all retained continuations after the defining walk. Each source
// chain is memoized once; cycles receive the union of their own written parts.
// Uses O(K log K) time and O(K) arena storage for K incomplete continuations,
// with no IR traversal, per-use recovery, recursion or fixed-point rescanning.
iree_status_t loom_low_register_parts_resolve(loom_low_register_parts_t* parts);

// Computes defined register parts for one verified, immutable function domain.
// Invoked only when allocation selects spills in part-bearing classes. Masks
// are dense by local ordinal; UINT32_MAX denotes a complete value. Returns NULL
// without allocating when every definition writes a complete register. The
// caller retains selected masks before releasing |arena|.
iree_status_t loom_low_register_parts_analyze(
    const loom_local_value_domain_t* value_domain,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, uint32_t** out_masks);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_REGISTER_PARTS_H_
