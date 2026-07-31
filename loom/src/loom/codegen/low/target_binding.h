// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Low function representation and target binding.
//
// This layer connects a Low function's intrinsic representation contract and
// effective target facts to dense Low descriptor tables. The descriptor table
// ABI itself remains IR-agnostic.

#ifndef LOOM_CODEGEN_LOW_TARGET_BINDING_H_
#define LOOM_CODEGEN_LOW_TARGET_BINDING_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolved low target context for one low function.
typedef struct loom_low_resolved_target_t {
  // Immutable effective facts selected for this Low function.
  const loom_target_facts_t* target_facts;
  // Borrowed effective target name without the leading '@'.
  iree_string_view_t target_name;
  // Borrowed descriptor-set key selected by the Low function representation
  // contract.
  iree_string_view_t descriptor_set_key;
  // Feature bitset projected from the effective target facts.
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

typedef enum loom_low_descriptor_packet_kind_e {
  // Not a descriptor-backed low packet.
  LOOM_LOW_DESCRIPTOR_PACKET_NONE = 0,
  // low.op descriptor packet.
  LOOM_LOW_DESCRIPTOR_PACKET_OP = 1,
  // low.const descriptor packet.
  LOOM_LOW_DESCRIPTOR_PACKET_CONST = 2,
} loom_low_descriptor_packet_kind_t;

typedef enum loom_low_descriptor_packet_resolution_e {
  // Not a descriptor-backed low packet.
  LOOM_LOW_DESCRIPTOR_PACKET_RESOLUTION_NONE = 0,
  // Packet key resolved to a descriptor row in the selected target contract.
  LOOM_LOW_DESCRIPTOR_PACKET_RESOLUTION_RESOLVED = 1,
  // Packet key was not present in the selected target contract.
  LOOM_LOW_DESCRIPTOR_PACKET_RESOLUTION_MISSING = 2,
  // Packet carried an explicit descriptor ordinal whose row key differs from
  // the packet key in the selected target contract.
  LOOM_LOW_DESCRIPTOR_PACKET_RESOLUTION_ORDINAL_KEY_MISMATCH = 3,
} loom_low_descriptor_packet_resolution_t;

// Target-bound descriptor row for one descriptor-backed low packet.
//
// Text IR names descriptor packets with stable spellings and stores an
// unresolved ordinal sentinel. This boundary resolves the spelling through the
// selected descriptor set; compiled in-memory consumers carry the descriptor
// pointer directly.
typedef struct loom_low_resolved_descriptor_packet_t {
  // Operation represented by this packet record.
  const loom_op_t* op;
  // Descriptor packet kind, or NONE for non-packet ops.
  loom_low_descriptor_packet_kind_t kind;
  // Borrowed textual descriptor key used for diagnostics.
  iree_string_view_t key;
  // Attribute index containing |key| in text-form IR.
  uint16_t key_attr_index;
  // Resolution state for descriptor-backed packets.
  loom_low_descriptor_packet_resolution_t resolution;
  // Dense descriptor ordinal used for explicit packet rows or lookup results.
  uint32_t descriptor_ordinal;
  // Borrowed descriptor row key, or empty when no row was resolved.
  iree_string_view_t descriptor_key;
  // Descriptor row in |target->descriptor_set|, or NULL when unresolved.
  const loom_low_descriptor_t* descriptor;
} loom_low_resolved_descriptor_packet_t;

// Resolves the effective target facts and descriptor set for |low_func_op|
// using caller-owned symbol facts.
//
// User IR failures are emitted through |emitter| and leave
// out_target->descriptor_set NULL. Infrastructure failures are returned as
// status. |low_func_op| must be a target-low function definition or
// declaration. The arena backing |symbol_facts| must outlive |out_target|.
iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target);

// Resolves |op| as a descriptor-backed low packet in |target|.
//
// Non-packet ops return OK with kind NONE. Missing user descriptors return OK
// with a non-NONE kind and NULL descriptor so callers can emit diagnostics
// using their own error domains and continue when appropriate.
iree_status_t loom_low_resolve_descriptor_packet(
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_op_t* op, loom_low_resolved_descriptor_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TARGET_BINDING_H_
