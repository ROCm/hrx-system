// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_REFERENCE_PLAN_H_
#define LOOM_TARGET_ARCH_VM_REFERENCE_PLAN_H_

#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/intern_table.h"
#include "loom/target/arch/vm/types.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// A canonical external identity and its provisional-to-final ordinal mapping.
// The key and provisional ordinal move together during canonical ordering;
// final ordinals are then stored at their original provisional row indexes.
typedef struct loom_vm_reference_entry_t {
  // Exact identity whose strings borrow immutable source or type metadata.
  loom_type_reference_key_t key;
  // Insertion-order ordinal carried with the key during sorting.
  uint16_t provisional_ordinal;
  // Final ordinal for the key originally stored at this physical row index.
  uint16_t ordinal;
} loom_vm_reference_entry_t;

// Metadata-scoped reference identities. Zero initialization creates an empty
// plan without allocating storage. Binding interns only participating source
// identities and canonical keys; repeated occurrences and unrelated fields
// require no additional storage. Image ordinals are independent of provider
// ordinals.
//
// Finalization ends binding and orders keys for serialization. The index is no
// longer queried after keys move. Plan storage may be reclaimed after ordinals
// have been remapped and metadata serialized; function emission needs neither
// this plan nor its borrowed strings. Discard the plan after any binding
// failure.
typedef struct loom_vm_reference_plan_t {
  // Source names, managed declaration IDs, and keys in disjoint index domains.
  loom_intern_table_t index;
  // Stable chunks of 32 key rows beyond the inline prefix.
  loom_segmented_storage_t overflow;
  // Inline prefix for common modules with few external reference identities.
  loom_vm_reference_entry_t inline_entries[8];
  // Number of canonical keys, bounded by the image's u16 ordinal space.
  uint32_t count;
  // Number of namespaces partitioning the finalized key order.
  uint32_t group_count;
} loom_vm_reference_plan_t;

// Resolves a source type binding once and returns its provisional key ordinal.
// |type| is a supported reference signature type. An undeclared external
// identity or image ordinal exhaustion emits a diagnostic at |op| and leaves
// |out_bound| false, including with no diagnostic callback. Status is reserved
// for allocation and diagnostic sink failures.
iree_status_t loom_vm_reference_plan_bind(loom_vm_reference_plan_t* plan,
                                          const loom_module_t* module,
                                          loom_type_t type,
                                          iree_arena_allocator_t* arena,
                                          iree_diagnostic_emitter_t emitter,
                                          const loom_op_t* op, bool* out_bound,
                                          uint16_t* out_ordinal);

// Sorts canonical keys and establishes namespace counts and the ordinal map.
// Returns true when any provisional ordinal differs from its final ordinal.
// No allocation or source/registry lookup occurs during finalization.
bool loom_vm_reference_plan_finalize(loom_vm_reference_plan_t* plan);

// Returns one canonical key after finalization, in serialized image order.
const loom_type_reference_key_t* loom_vm_reference_plan_key(
    const loom_vm_reference_plan_t* plan, uint16_t ordinal);

// Maps an owned signature field's provisional ordinal to its final ordinal.
uint16_t loom_vm_reference_plan_ordinal(const loom_vm_reference_plan_t* plan,
                                        uint16_t provisional_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_REFERENCE_PLAN_H_
