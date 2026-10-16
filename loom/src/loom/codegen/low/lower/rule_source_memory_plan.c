// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_source_memory_plan.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_plan.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/ir/structural_hash.h"

typedef struct loom_low_lower_rule_byte_offset_binding_t {
  // Resolved emit owning the arithmetic definition at its original position.
  const loom_low_lower_resolved_emit_t* definition;
  // Low SSA published by the definition and read directly by later uses.
  loom_value_id_t value;
  // First operand using the byte offset within the defining emit.
  uint16_t operand_ordinal;
} loom_low_lower_rule_byte_offset_binding_t;

struct loom_low_lower_rule_byte_offset_entry_t {
  // Actual source emission block; definitions never cross block boundaries.
  const loom_block_t* block;
  // Descriptor namespace for the selected arithmetic contract.
  const loom_low_lower_rule_set_t* rule_set;
  // Interned arithmetic carrier, conversions, and descriptor choices.
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer;
  // Source observation's canonical dynamic term-set identity.
  iree_host_size_t identity;
  // Exact static byte contribution included in the returned physical value.
  int64_t static_bias;
  // Function-owned binding shared by all uses of this physical recipe.
  loom_low_lower_rule_byte_offset_binding_t* binding;
  // Next colliding physical recipe in the bucket.
  loom_low_lower_rule_byte_offset_entry_t* next;
  // Actual source position of the current defining emit.
  uint64_t block_ordinal;
  // Row ordinal within the defining selected rule.
  uint16_t emit_ordinal;
  // Retained hash used when growing the construction table.
  uint32_t hash;
};

static iree_status_t loom_low_lower_rule_byte_offsets_grow(
    loom_low_lower_rule_source_memory_plan_builder_t* builder) {
  const iree_host_size_t bucket_count =
      builder->bucket_count ? builder->bucket_count * 2 : 32;
  loom_low_lower_rule_byte_offset_entry_t** buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < builder->bucket_count; ++i) {
    loom_low_lower_rule_byte_offset_entry_t* entry = builder->buckets[i];
    while (entry != NULL) {
      loom_low_lower_rule_byte_offset_entry_t* next = entry->next;
      const iree_host_size_t bucket = entry->hash & (bucket_count - 1);
      entry->next = buckets[bucket];
      buckets[bucket] = entry;
      entry = next;
    }
  }
  builder->buckets = buckets;
  builder->bucket_count = bucket_count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_source_memory_plan_byte_offset(
    loom_low_lower_context_t* context,
    loom_low_lower_rule_source_memory_plan_builder_t* builder,
    const loom_low_lower_selected_plan_t* selected_plan,
    const loom_low_lower_resolved_emit_t* resolved_emit, uint16_t emit_ordinal,
    uint16_t operand_ordinal, const void** out_binding) {
  const loom_low_lower_rule_set_t* rule_set = selected_plan->rule_set;
  const loom_low_lower_source_memory_t* source_memory =
      &rule_set
           ->source_memories[resolved_emit->emit->source_memory_ordinal - 1];
  const loom_low_source_memory_access_plan_t* access =
      &selected_plan->source_memory->access;
  const loom_low_lower_source_memory_byte_offset_materializer_t* materializer =
      loom_low_lower_rule_set_source_memory_byte_offset_materializer(
          rule_set, source_memory);
  const loom_low_lower_value_ref_t* ref =
      &rule_set->value_refs[resolved_emit->emit->operand_ref_start +
                            operand_ordinal];
  // A reused view base may already include a constant. Complete offsets always
  // include the canonical bias; dynamic offsets include only that embedded
  // contribution. Equal physical values can share across both reference kinds.
  const int64_t static_bias =
      ref->kind == LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET
          ? access->static_byte_offset
          : (loom_low_lower_rule_source_memory_reuses_view_base(source_memory,
                                                                access)
                 ? access->dynamic_view_base_value_static_byte_offset
                 : 0);
  const iree_host_size_t identity =
      selected_plan->source_memory->dynamic_byte_offset_identity;
  const loom_op_t* emission_op =
      selected_plan->emission_source_node_index
          ? selected_plan->data
                .source_nodes[selected_plan->emission_source_node_index]
          : selected_plan->source_op;
  const loom_block_t* block = emission_op->parent_block;
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u64(hash, identity);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)block);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)rule_set);
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)materializer);
  hash = loom_structural_hash_mix_u64(hash, (uint64_t)static_bias);
  hash = loom_structural_hash_finalize(hash);
  loom_low_lower_rule_byte_offset_entry_t* entry =
      builder->bucket_count
          ? builder->buckets[hash & (builder->bucket_count - 1)]
          : NULL;
  for (; entry != NULL; entry = entry->next) {
    if (entry->hash == hash && entry->identity == identity &&
        entry->block == block && entry->rule_set == rule_set &&
        entry->materializer == materializer &&
        entry->static_bias == static_bias) {
      break;
    }
  }
  if (entry == NULL) {
    if (builder->entry_count >= builder->bucket_count * 3 / 4) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_byte_offsets_grow(builder));
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(builder->arena, sizeof(*entry), (void**)&entry));
    loom_low_lower_rule_byte_offset_binding_t* binding = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*binding), (void**)&binding));
    *binding = (loom_low_lower_rule_byte_offset_binding_t){
        .definition = resolved_emit,
        .value = LOOM_VALUE_ID_INVALID,
        .operand_ordinal = operand_ordinal,
    };
    const iree_host_size_t bucket = hash & (builder->bucket_count - 1);
    *entry = (loom_low_lower_rule_byte_offset_entry_t){
        .block = block,
        .rule_set = rule_set,
        .materializer = materializer,
        .identity = identity,
        .static_bias = static_bias,
        .binding = binding,
        .next = builder->buckets[bucket],
        .block_ordinal = emission_op->block_ordinal,
        .emit_ordinal = emit_ordinal,
        .hash = hash,
    };
    builder->buckets[bucket] = entry;
    ++builder->entry_count;
  } else if (emission_op->block_ordinal < entry->block_ordinal ||
             (emission_op->block_ordinal == entry->block_ordinal &&
              (emit_ordinal < entry->emit_ordinal ||
               (emit_ordinal == entry->emit_ordinal &&
                operand_ordinal < entry->binding->operand_ordinal)))) {
    entry->block_ordinal = emission_op->block_ordinal;
    entry->emit_ordinal = emit_ordinal;
    entry->binding->definition = resolved_emit;
    entry->binding->operand_ordinal = operand_ordinal;
  }
  *out_binding = entry->binding;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_source_memory_emit_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access,
    const loom_low_lower_resolved_emit_t* resolved_emit,
    uint16_t operand_ordinal, loom_value_id_t* out_value) {
  loom_low_lower_rule_byte_offset_binding_t* binding =
      (loom_low_lower_rule_byte_offset_binding_t*)
          loom_low_lower_resolved_emit_materializer(resolved_emit,
                                                    operand_ordinal);
  if (binding->definition == resolved_emit &&
      binding->operand_ordinal == operand_ordinal) {
    const loom_low_lower_value_ref_t* ref =
        &rule_set->value_refs[resolved_emit->emit->operand_ref_start +
                              operand_ordinal];
    if (ref->kind == LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_materialize_source_memory_complete_byte_offset(
              context, rule_set, source_op, source_memory, access,
              &binding->value));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_materialize_source_memory_byte_offset(
              context, rule_set, source_op, source_memory, access,
              &binding->value));
    }
  }
  *out_value = binding->value;
  return iree_ok_status();
}
