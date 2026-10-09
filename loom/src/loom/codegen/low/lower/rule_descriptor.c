// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_descriptor.h"

#include <string.h>

#include "loom/target/registers.h"

static uint32_t** loom_low_lower_rule_descriptor_cache_maps(
    loom_low_lower_rule_descriptor_cache_t* cache) {
  return (uint32_t**)(cache + 1);
}

iree_status_t loom_low_lower_rule_descriptor_cache_select(
    loom_low_lower_rule_set_list_t rule_sets,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_lower_rule_descriptor_cache_t** inout_cache) {
  for (loom_low_lower_rule_descriptor_cache_t** link = inout_cache;
       *link != NULL; link = &(*link)->next) {
    loom_low_lower_rule_descriptor_cache_t* cache = *link;
    if (cache->descriptor_set != descriptor_set ||
        cache->rule_sets.values != rule_sets.values ||
        cache->rule_sets.count != rule_sets.count) {
      continue;
    }
    if (link != inout_cache) {
      *link = cache->next;
      cache->next = *inout_cache;
      *inout_cache = cache;
    }
    return iree_ok_status();
  }

  const iree_host_size_t maps_byte_length = rule_sets.count * sizeof(uint32_t*);
  loom_low_lower_rule_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*cache) + maps_byte_length, (void**)&cache));
  *cache = (loom_low_lower_rule_descriptor_cache_t){
      .arena = arena,
      .next = *inout_cache,
      .rule_sets = rule_sets,
      .descriptor_set = descriptor_set,
  };
  memset(loom_low_lower_rule_descriptor_cache_maps(cache), 0, maps_byte_length);
  *inout_cache = cache;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_descriptor_cache_resolve(
    loom_low_lower_rule_descriptor_cache_t* cache, uint16_t rule_set_index,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  const loom_low_lower_rule_set_t* rule_set =
      cache->rule_sets.values[rule_set_index];
  uint32_t** maps = loom_low_lower_rule_descriptor_cache_maps(cache);
  uint32_t* ordinals = maps[rule_set_index];
  if (ordinals == NULL) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(cache->arena, rule_set->descriptor_ref_count,
                                  sizeof(*ordinals), (void**)&ordinals));
    memset(ordinals, 0, rule_set->descriptor_ref_count * sizeof(*ordinals));
    maps[rule_set_index] = ordinals;
  }

  // Zero is unresolved, UINT32_MAX is missing, and all other entries are
  // descriptor ordinals plus one.
  uint32_t ordinal = ordinals[descriptor_ref];
  if (ordinal == 0) {
    const iree_string_view_t key = loom_low_lower_rule_set_string(
        rule_set, rule_set->descriptor_refs[descriptor_ref].key_string_ref);
    const uint32_t resolved =
        loom_low_descriptor_set_lookup_descriptor(cache->descriptor_set, key);
    ordinal = resolved == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ? UINT32_MAX
                                                           : resolved + 1;
    ordinals[descriptor_ref] = ordinal;
  }
  *out_descriptor = ordinal == UINT32_MAX
                        ? NULL
                        : loom_low_descriptor_set_descriptor_at(
                              cache->descriptor_set, ordinal - 1);
  return iree_ok_status();
}

const loom_low_operand_t* loom_low_lower_rule_descriptor_result_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  IREE_ASSERT_LT(result_index, descriptor->result_count);
  return loom_low_descriptor_operand(descriptor_set, descriptor, result_index);
}

iree_status_t loom_low_lower_rule_descriptor_result_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint16_t result_index, loom_type_t* out_type) {
  *out_type = loom_type_none();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operand =
      loom_low_lower_rule_descriptor_result_operand(descriptor_set, descriptor,
                                                    result_index);
  IREE_ASSERT_EQ(operand->reg_class_alt_count, 1);
  const uint32_t alt_index = operand->reg_class_alt_start;
  IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  IREE_ASSERT_NE(alt->reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  IREE_ASSERT_FALSE(
      iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE));
  IREE_ASSERT_GT(operand->unit_count, 0);
  return loom_low_lower_make_register_type(context, alt->reg_class_id,
                                           operand->unit_count, out_type);
}

const loom_low_operand_t* loom_low_lower_rule_descriptor_packet_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t operand_index) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (loom_low_operand_role_is_packet_operand(operand->role) &&
        operand->source_value_index == operand_index) {
      return operand;
    }
  }
  IREE_ASSERT_UNREACHABLE("trusted descriptor packet operand must exist");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_low_lower_rule_descriptor_copy_operand_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint16_t operand_index, loom_type_t source_type, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT(loom_low_type_is_register(source_type));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operand =
      loom_low_lower_rule_descriptor_packet_operand(descriptor_set, descriptor,
                                                    operand_index);
  const uint32_t source_unit_count =
      loom_low_register_type_unit_count(source_type);
  // Per-lane emissions copy an aggregate before slicing it into descriptor
  // packet operands. The descriptor selects the register class for each lane;
  // the copy preserves the complete aggregate carrier width.
  IREE_ASSERT_EQ(source_unit_count % operand->unit_count, 0);

  const uint16_t source_reg_class_id =
      loom_low_register_type_class_id(source_type);
  uint16_t target_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  uint16_t register_alternative_count = 0;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    ++register_alternative_count;
    target_reg_class_id = alt->reg_class_id;
    if (alt->reg_class_id == source_reg_class_id) {
      *out_type = source_type;
      return iree_ok_status();
    }
  }
  IREE_ASSERT_EQ(register_alternative_count, 1);
  IREE_ASSERT_NE(target_reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  if (loom_type_register_has_value_type(source_type)) {
    return loom_low_lower_make_typed_register_type(
        context, target_reg_class_id, source_unit_count,
        *loom_type_register_value_type(source_type), out_type);
  }
  return loom_low_lower_make_register_type(context, target_reg_class_id,
                                           source_unit_count, out_type);
}
