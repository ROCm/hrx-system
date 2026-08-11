// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low function representation and target binding.
//
// This layer resolves a Low function's intrinsic representation contract to
// dense Low descriptor tables, then verifies that the representation can
// encode the target selected by authored or invocation-refined target facts.
// The descriptor table ABI itself remains IR-agnostic.

#ifndef LOOM_CODEGEN_LOW_TARGET_BINDING_H_
#define LOOM_CODEGEN_LOW_TARGET_BINDING_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolved low target context for one low function.
typedef struct loom_low_resolved_target_t {
  // Immutable function target facts selected for this Low function, or NULL
  // for a portable representation that does not require a hardware target.
  const loom_target_facts_t* target_facts;
  // Borrowed effective target name without the leading '@', or empty when
  // |target_facts| is NULL.
  iree_string_view_t target_name;
  // Borrowed descriptor-set key selected by the Low function representation
  // contract.
  iree_string_view_t descriptor_set_key;
  // Feature bitset projected from the function target facts.
  uint64_t feature_bits;
  // Descriptor set found in the caller-provided registry.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_low_resolved_target_t;

// Returns the common target bundle projected into |target->target_facts|.
static inline const loom_target_bundle_t* loom_low_resolved_target_bundle(
    const loom_low_resolved_target_t* target) {
  return target ? loom_target_facts_bundle(target->target_facts) : NULL;
}

typedef struct loom_low_register_type_resolver_t {
  // Descriptor set defining the resolved descriptor register-class IDs.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_low_register_type_resolver_t;

// Returns a resolver that borrows |descriptor_set|.
static inline loom_low_register_type_resolver_t
loom_low_register_type_resolver_for_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set) {
  return (loom_low_register_type_resolver_t){
      /*.descriptor_set=*/descriptor_set,
  };
}

// Resolves a Loom register type to a descriptor-set-local register class.
// |out_descriptor_register_class| may be NULL when only the dense descriptor ID
// is needed. Returns false when |type| is not a register type or its class is
// not defined by the descriptor set.
bool loom_low_register_type_resolver_try_resolve(
    const loom_low_register_type_resolver_t* resolver, loom_type_t type,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class);

// Returns true when |type| resolves and its register class contains every
// requested flag bit.
bool loom_low_register_type_resolver_has_class_flags(
    const loom_low_register_type_resolver_t* resolver, loom_type_t type,
    loom_low_reg_class_flags_t flags);

typedef enum loom_low_descriptor_packet_kind_e {
  // Not a descriptor-backed low packet.
  LOOM_LOW_DESCRIPTOR_PACKET_NONE = 0,
  // low.op descriptor packet.
  LOOM_LOW_DESCRIPTOR_PACKET_OP = 1,
  // low.const descriptor packet.
  LOOM_LOW_DESCRIPTOR_PACKET_CONST = 2,
} loom_low_descriptor_packet_kind_t;

// Target-bound descriptor row for one descriptor-backed low packet.
//
// Canonical Low IR stores a required dense descriptor ordinal in the enclosing
// function's selected representation contract. Compiler consumers project the
// corresponding descriptor pointer directly; stable descriptor spellings are
// recovered only at text, bytecode, and diagnostic boundaries.
typedef struct loom_low_descriptor_packet_t {
  // Operation represented by this packet record.
  const loom_op_t* op;
  // Descriptor packet kind, or NONE for non-packet ops.
  loom_low_descriptor_packet_kind_t kind;
  // Dense descriptor ordinal in the function's representation contract.
  uint32_t descriptor_ordinal;
  // Borrowed descriptor row in the function's representation contract.
  const loom_low_descriptor_t* descriptor;
} loom_low_descriptor_packet_t;

// Returns the descriptor packet kind for |op|.
static inline loom_low_descriptor_packet_kind_t loom_low_descriptor_packet_kind(
    const loom_op_t* op) {
  if (loom_low_op_isa(op)) return LOOM_LOW_DESCRIPTOR_PACKET_OP;
  if (loom_low_const_isa(op)) return LOOM_LOW_DESCRIPTOR_PACKET_CONST;
  return LOOM_LOW_DESCRIPTOR_PACKET_NONE;
}

// Returns the required descriptor ordinal for a descriptor-backed packet.
static inline uint32_t loom_low_descriptor_packet_ordinal(
    const loom_op_t* op, loom_low_descriptor_packet_kind_t kind) {
  return kind == LOOM_LOW_DESCRIPTOR_PACKET_OP ? loom_low_op_descriptor(op)
                                               : loom_low_const_descriptor(op);
}

// Projects one verified Low packet into its descriptor row.
//
// The function-scoped Low verifier proves that packet ordinals are in range.
// Subsequent compiler passes use that invariant directly: this helper performs
// no lookup, fallback, or repeated validation.
static inline void loom_low_descriptor_packet_initialize(
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    loom_low_descriptor_packet_t* out_packet) {
  const loom_low_descriptor_packet_kind_t kind =
      loom_low_descriptor_packet_kind(op);
  *out_packet = (loom_low_descriptor_packet_t){
      /*.op=*/op,
      /*.kind=*/kind,
  };
  if (kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) return;
  out_packet->descriptor_ordinal = loom_low_descriptor_packet_ordinal(op, kind);
  out_packet->descriptor =
      &descriptor_set->descriptors[out_packet->descriptor_ordinal];
}

// Returns the packet field index used to attach descriptor diagnostics.
static inline uint16_t loom_low_descriptor_packet_attribute_index(
    const loom_low_descriptor_packet_t* packet) {
  if (packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_OP) {
    return loom_low_op_descriptor_ATTR_INDEX;
  }
  return loom_low_const_descriptor_ATTR_INDEX;
}

// Returns the stable descriptor spelling for diagnostics and presentation.
// Compiler matching and dispatch must use |descriptor_ordinal| or
// |descriptor| instead.
static inline iree_string_view_t loom_low_descriptor_packet_diagnostic_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_packet_t* packet) {
  return loom_low_descriptor_set_string(descriptor_set,
                                        packet->descriptor->key_string_offset);
}

// Resolves the function target facts and descriptor set for |low_func_op|
// using caller-owned symbol facts.
//
// User IR failures are emitted through |emitter| and leave
// out_target->descriptor_set NULL. Infrastructure failures are returned as
// status. |low_func_op| must be a target-low function definition or
// declaration. |function_target_facts| supplies invocation-refined facts that
// already include the function contract when non-NULL; otherwise facts are
// resolved from the authored target witness. A targetless function resolves
// only its explicit representation descriptor set and leaves the target facts
// unset, independent of its ABI. The arena backing |symbol_facts| and
// |function_target_facts| must outlive |out_target|.
iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_op_t* low_func_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TARGET_BINDING_H_
