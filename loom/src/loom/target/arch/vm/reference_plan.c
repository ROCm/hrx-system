// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/reference_plan.h"

#include "loom/ir/structural_hash.h"
#include "loom/util/adaptive_sort.h"

// Bucket values store a u16 canonical ordinal and an index-domain tag. Static
// type names retain their full string ID; inline handles use the validated
// canonical declaration binding. Both source domains resolve only once.
#define LOOM_VM_REFERENCE_KEY_TAG (1u << 16)
#define LOOM_VM_REFERENCE_DECLARATION_TAG (1u << 17)
#define LOOM_VM_REFERENCE_SEGMENT_CAPACITY 32u

static const loom_vm_reference_entry_t* loom_vm_reference_entry(
    const loom_vm_reference_plan_t* plan, iree_host_size_t ordinal) {
  if (ordinal < IREE_ARRAYSIZE(plan->inline_entries)) {
    return &plan->inline_entries[ordinal];
  }
  ordinal -= IREE_ARRAYSIZE(plan->inline_entries);
  const loom_vm_reference_entry_t* entries =
      loom_segmented_storage_const_segment(
          &plan->overflow,
          (uint32_t)(ordinal / LOOM_VM_REFERENCE_SEGMENT_CAPACITY));
  return &entries[ordinal % LOOM_VM_REFERENCE_SEGMENT_CAPACITY];
}

static loom_vm_reference_entry_t* loom_vm_reference_entry_mutable(
    loom_vm_reference_plan_t* plan, iree_host_size_t ordinal) {
  return (loom_vm_reference_entry_t*)loom_vm_reference_entry(plan, ordinal);
}

static int loom_vm_reference_key_compare(const loom_type_reference_key_t* lhs,
                                         const loom_type_reference_key_t* rhs) {
  const int comparison =
      iree_string_view_compare(lhs->namespace_name, rhs->namespace_name);
  return comparison ? comparison
                    : iree_string_view_compare(lhs->type_name, rhs->type_name);
}

static uint32_t loom_vm_reference_key_hash(
    const loom_type_reference_key_t* key) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_bytes(hash, key->namespace_name.data,
                                        key->namespace_name.size);
  hash = loom_structural_hash_mix_bytes(hash, key->type_name.data,
                                        key->type_name.size);
  return loom_structural_hash_finalize(hash);
}

typedef struct loom_vm_reference_query_t {
  // Binding plan owning the canonical key rows.
  const loom_vm_reference_plan_t* plan;
  // Candidate key, or NULL when probing the source-name domain.
  const loom_type_reference_key_t* key;
  // Source identity domain when |key| is NULL.
  uint32_t source_tag;
} loom_vm_reference_query_t;

static bool loom_vm_reference_equal(const void* context, uint32_t index) {
  const loom_vm_reference_query_t* query = context;
  if (!query->key) {
    // Source hashes are a bijection of the complete u32 ID within each domain.
    return (index & ~UINT32_C(0xFFFF)) == query->source_tag;
  }
  return (index & LOOM_VM_REFERENCE_KEY_TAG) != 0 &&
         loom_vm_reference_key_compare(
             query->key,
             &loom_vm_reference_entry(query->plan, (uint16_t)index)->key) == 0;
}

iree_status_t loom_vm_reference_plan_bind(loom_vm_reference_plan_t* plan,
                                          const loom_module_t* module,
                                          loom_type_t type,
                                          iree_arena_allocator_t* arena,
                                          uint16_t* out_ordinal) {
  if (!plan->count) {
    IREE_RETURN_IF_ERROR(loom_intern_table_initialize(arena, 32, &plan->index));
    loom_segmented_storage_initialize(
        LOOM_VM_REFERENCE_SEGMENT_CAPACITY * sizeof(loom_vm_reference_entry_t),
        iree_alignof(loom_vm_reference_entry_t), &plan->overflow);
  }
  const uint32_t source_tag =
      loom_type_is_reference(type) ? LOOM_VM_REFERENCE_DECLARATION_TAG : 0;
  const uint32_t source_id =
      loom_type_is_reference(type) ? loom_type_reference_declaration_id(type)
      : loom_type_is_buffer(type)  ? LOOM_STRING_ID_INVALID
                                   : loom_type_dialect_name_id(type);
  // Odd multiplication modulo 2^32 preserves every source ID bit, including
  // the built-in buffer sentinel. It is an identity encoding, not a lossy hash.
  const uint32_t source_hash = source_id * 2654435769u;
  const loom_vm_reference_query_t source_query = {.plan = plan,
                                                  .source_tag = source_tag};
  loom_intern_probe_t source = loom_intern_table_probe(
      &plan->index, source_hash, loom_vm_reference_equal, &source_query);
  if (source.index != UINT32_MAX) {
    *out_ordinal = (uint16_t)source.index;
    return iree_ok_status();
  }
  loom_type_reference_key_t key;
  if (!loom_vm_type_reference_key(module, type, &key)) {
    const iree_string_view_t spelling =
        loom_string_table_get(&module->strings, source_id);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM signature type '%.*s' has no managed reference identity",
        (int)spelling.size, spelling.data);
  }
  const uint32_t key_hash = loom_vm_reference_key_hash(&key);
  const loom_vm_reference_query_t key_query = {.plan = plan, .key = &key};
  loom_intern_probe_t canonical = loom_intern_table_probe(
      &plan->index, key_hash, loom_vm_reference_equal, &key_query);
  uint16_t ordinal = (uint16_t)canonical.index;
  if (canonical.index == UINT32_MAX) {
    if (plan->count == (uint32_t)UINT16_MAX + 1) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM reference type count exceeds u16 ordinals");
    }
    if (plan->count >= IREE_ARRAYSIZE(plan->inline_entries) &&
        (plan->count - IREE_ARRAYSIZE(plan->inline_entries)) %
                LOOM_VM_REFERENCE_SEGMENT_CAPACITY ==
            0) {
      void* entries = NULL;
      IREE_RETURN_IF_ERROR(
          loom_segmented_storage_append(&plan->overflow, arena, &entries));
    }
    ordinal = (uint16_t)plan->count++;
    *loom_vm_reference_entry_mutable(plan, ordinal) =
        (loom_vm_reference_entry_t){.key = key, .provisional_ordinal = ordinal};
    IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
        arena, &plan->index, key_hash, /*insertion_count=*/1, &canonical.slot));
    loom_intern_table_insert(&plan->index, canonical.slot, key_hash,
                             ordinal | LOOM_VM_REFERENCE_KEY_TAG);
    // Canonical insertion can occupy the source slot or grow the table.
    source = loom_intern_table_probe(&plan->index, source_hash,
                                     loom_vm_reference_equal, &source_query);
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      arena, &plan->index, source_hash, /*insertion_count=*/1, &source.slot));
  loom_intern_table_insert(&plan->index, source.slot, source_hash,
                           ordinal | source_tag);
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static bool loom_vm_reference_less(void* context,
                                   const loom_vm_reference_entry_t* lhs,
                                   const loom_vm_reference_entry_t* rhs) {
  (void)context;
  return loom_vm_reference_key_compare(&lhs->key, &rhs->key) < 0;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_ACCESSOR(loom_vm_reference_sort,
                                        loom_vm_reference_entry_t,
                                        loom_vm_reference_plan_t*,
                                        loom_vm_reference_entry_mutable, void*,
                                        loom_vm_reference_less)

bool loom_vm_reference_plan_finalize(loom_vm_reference_plan_t* plan) {
  loom_vm_reference_sort(NULL, plan, plan->count);
  plan->group_count = 0;
  bool changed = false;
  const loom_type_reference_key_t* previous = NULL;
  for (uint32_t i = 0; i < plan->count; ++i) {
    const loom_vm_reference_entry_t* entry = loom_vm_reference_entry(plan, i);
    if (!previous || !iree_string_view_equal(previous->namespace_name,
                                             entry->key.namespace_name)) {
      ++plan->group_count;
    }
    previous = &entry->key;
    changed |= entry->provisional_ordinal != i;
    loom_vm_reference_entry_mutable(plan, entry->provisional_ordinal)->ordinal =
        (uint16_t)i;
  }
  return changed;
}

const loom_type_reference_key_t* loom_vm_reference_plan_key(
    const loom_vm_reference_plan_t* plan, uint16_t ordinal) {
  return &loom_vm_reference_entry(plan, ordinal)->key;
}

uint16_t loom_vm_reference_plan_ordinal(const loom_vm_reference_plan_t* plan,
                                        uint16_t provisional_ordinal) {
  return loom_vm_reference_entry(plan, provisional_ordinal)->ordinal;
}
