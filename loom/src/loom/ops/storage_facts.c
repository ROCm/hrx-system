// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/storage_facts.h"

#include <string.h>

#include "loom/util/math.h"

// Payload tag namespace owned by the storage fact domain.
#define LOOM_STORAGE_FACT_PAYLOAD_REFERENCE 1

static bool loom_storage_facts_query_reference_from_table(
    const loom_value_fact_table_t* table, loom_value_facts_t facts,
    loom_storage_reference_facts_t* out_reference) {
  if (!table) return false;
  const void* payload = NULL;
  iree_host_size_t payload_length = 0;
  if (!loom_value_facts_query_extension_payload(
          &table->context, facts, LOOM_STORAGE_FACT_PAYLOAD_REFERENCE, &payload,
          &payload_length) ||
      payload_length != sizeof(*out_reference)) {
    return false;
  }
  if (out_reference) memcpy(out_reference, payload, sizeof(*out_reference));
  return true;
}

bool loom_storage_facts_query_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_storage_reference_facts_t* out_reference) {
  return context && loom_storage_facts_query_reference_from_table(
                        context->table, facts, out_reference);
}

iree_status_t loom_storage_facts_make_reference(
    loom_fact_context_t* context,
    const loom_storage_reference_facts_t* reference,
    loom_value_facts_t* out_facts) {
  if (!context || !reference ||
      reference->backing_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_storage_space_is_valid(reference->storage_space) ||
      reference->minimum_alignment == 0) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_value_facts_make_extension_payload(
      context, LOOM_STORAGE_FACT_PAYLOAD_REFERENCE, reference,
      sizeof(*reference), out_facts);
}

iree_status_t loom_storage_facts_make_reserve(
    loom_fact_context_t* context, loom_value_id_t storage_value_id,
    loom_storage_space_t storage_space, int64_t byte_length,
    int64_t byte_alignment, loom_value_facts_t* out_facts) {
  if (storage_value_id == LOOM_VALUE_ID_INVALID ||
      !loom_storage_space_is_valid(storage_space) || byte_length <= 0 ||
      !loom_is_power_of_two_i64(byte_alignment)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_storage_reference_facts_t reference = {
      .byte_offset = loom_value_facts_exact_i64(0),
      .target_byte_offset = loom_value_facts_unknown(),
      .valid_byte_length = loom_value_facts_exact_i64(byte_length),
      .minimum_alignment = (uint64_t)byte_alignment,
      .backing_value_id = storage_value_id,
      .storage_space = storage_space,
  };
  return loom_storage_facts_make_reference(context, &reference, out_facts);
}

static uint64_t loom_storage_gcd_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0) return rhs;
  if (rhs == 0) return lhs;
  while (rhs != 0) {
    uint64_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

static uint64_t loom_storage_offset_alignment(uint64_t base_alignment,
                                              int64_t byte_offset) {
  if (byte_offset == 0) return base_alignment;
  uint64_t offset_magnitude =
      byte_offset > 0 ? (uint64_t)byte_offset : (uint64_t)(-byte_offset);
  uint64_t alignment = loom_storage_gcd_u64(base_alignment, offset_magnitude);
  return alignment == 0 ? 1 : alignment;
}

iree_status_t loom_storage_facts_make_static_view(
    loom_fact_context_t* context, loom_value_facts_t source_facts,
    int64_t byte_offset, int64_t byte_length, loom_value_facts_t* out_facts) {
  loom_storage_reference_facts_t source = {0};
  if (byte_offset < 0 || byte_length <= 0 ||
      !loom_storage_facts_query_reference(context, source_facts, &source)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t static_offset = loom_value_facts_exact_i64(byte_offset);
  loom_storage_reference_facts_t reference = source;
  loom_value_facts_addi(&source.byte_offset, &static_offset,
                        &reference.byte_offset);
  loom_value_facts_addi(&source.target_byte_offset, &static_offset,
                        &reference.target_byte_offset);
  reference.valid_byte_length = loom_value_facts_exact_i64(byte_length);
  reference.minimum_alignment =
      loom_storage_offset_alignment(source.minimum_alignment, byte_offset);
  return loom_storage_facts_make_reference(context, &reference, out_facts);
}

static bool loom_storage_reference_equal(
    const loom_value_fact_table_t* lhs_table,
    const loom_storage_reference_facts_t* lhs,
    const loom_value_fact_table_t* rhs_table,
    const loom_storage_reference_facts_t* rhs) {
  return lhs->minimum_alignment == rhs->minimum_alignment &&
         lhs->backing_value_id == rhs->backing_value_id &&
         lhs->storage_space == rhs->storage_space &&
         loom_value_fact_table_facts_equal(lhs_table, lhs->byte_offset,
                                           rhs_table, rhs->byte_offset) &&
         loom_value_fact_table_facts_equal(lhs_table, lhs->target_byte_offset,
                                           rhs_table,
                                           rhs->target_byte_offset) &&
         loom_value_fact_table_facts_equal(lhs_table, lhs->valid_byte_length,
                                           rhs_table, rhs->valid_byte_length);
}

static bool loom_storage_extensions_equal(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs) {
  (void)domain;
  (void)module;
  (void)type;
  loom_storage_reference_facts_t lhs_reference = {0};
  loom_storage_reference_facts_t rhs_reference = {0};
  if (!loom_storage_facts_query_reference_from_table(lhs_table, lhs,
                                                     &lhs_reference) ||
      !loom_storage_facts_query_reference_from_table(rhs_table, rhs,
                                                     &rhs_reference)) {
    return lhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE &&
           rhs.extension_id == LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  }
  return loom_storage_reference_equal(lhs_table, &lhs_reference, rhs_table,
                                      &rhs_reference);
}

static iree_status_t loom_storage_clone_reference(
    loom_value_fact_table_t* target,
    const loom_value_fact_table_t* source_table,
    const loom_storage_reference_facts_t* source_reference,
    loom_storage_reference_facts_t* out_reference) {
  *out_reference = *source_reference;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
      target, source_table, source_reference->byte_offset,
      &out_reference->byte_offset));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
      target, source_table, source_reference->target_byte_offset,
      &out_reference->target_byte_offset));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
      target, source_table, source_reference->valid_byte_length,
      &out_reference->valid_byte_length));
  return iree_ok_status();
}

static iree_status_t loom_storage_clone_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* source, loom_value_facts_t facts,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  loom_storage_reference_facts_t source_reference = {0};
  if (!loom_storage_facts_query_reference_from_table(source, facts,
                                                     &source_reference)) {
    inout_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return iree_ok_status();
  }
  loom_storage_reference_facts_t reference = {0};
  IREE_RETURN_IF_ERROR(loom_storage_clone_reference(
      target, source, &source_reference, &reference));
  return loom_storage_facts_make_reference(&target->context, &reference,
                                           inout_facts);
}

static void loom_storage_meet_scalar_facts(loom_value_facts_t lhs,
                                           loom_value_facts_t rhs,
                                           loom_value_facts_t* out) {
  lhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  rhs.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
  if (loom_value_facts_is_float(lhs) || loom_value_facts_is_float(rhs)) {
    *out = loom_value_facts_unknown();
    return;
  }
  loom_value_facts_meet(&lhs, &rhs, out);
}

static bool loom_storage_references_share_backing(
    const loom_storage_reference_facts_t* lhs,
    const loom_storage_reference_facts_t* rhs) {
  return lhs->backing_value_id == rhs->backing_value_id &&
         lhs->storage_space == rhs->storage_space;
}

static iree_status_t loom_storage_meet_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  loom_storage_reference_facts_t lhs_reference = {0};
  loom_storage_reference_facts_t rhs_reference = {0};
  if (!loom_storage_facts_query_reference_from_table(lhs_table, lhs,
                                                     &lhs_reference) ||
      !loom_storage_facts_query_reference_from_table(rhs_table, rhs,
                                                     &rhs_reference) ||
      !loom_storage_references_share_backing(&lhs_reference, &rhs_reference)) {
    inout_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return iree_ok_status();
  }

  loom_storage_reference_facts_t reference = {
      .minimum_alignment = loom_storage_gcd_u64(
          lhs_reference.minimum_alignment, rhs_reference.minimum_alignment),
      .backing_value_id = lhs_reference.backing_value_id,
      .storage_space = lhs_reference.storage_space,
  };
  if (reference.minimum_alignment == 0) reference.minimum_alignment = 1;
  loom_storage_meet_scalar_facts(lhs_reference.byte_offset,
                                 rhs_reference.byte_offset,
                                 &reference.byte_offset);
  loom_storage_meet_scalar_facts(lhs_reference.target_byte_offset,
                                 rhs_reference.target_byte_offset,
                                 &reference.target_byte_offset);
  loom_storage_meet_scalar_facts(lhs_reference.valid_byte_length,
                                 rhs_reference.valid_byte_length,
                                 &reference.valid_byte_length);
  return loom_storage_facts_make_reference(&target->context, &reference,
                                           inout_facts);
}

static iree_status_t loom_storage_widen_fact_field(
    loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    loom_value_facts_t* out_facts) {
  if (loom_value_fact_table_facts_equal(previous_table, previous, next_table,
                                        next)) {
    return loom_value_fact_table_clone_fact(target, next_table, next,
                                            out_facts);
  }
  *out_facts = loom_value_facts_unknown();
  return iree_ok_status();
}

static iree_status_t loom_storage_widen_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    uint32_t iteration, loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  (void)iteration;
  loom_storage_reference_facts_t previous_reference = {0};
  loom_storage_reference_facts_t next_reference = {0};
  if (!loom_storage_facts_query_reference_from_table(previous_table, previous,
                                                     &previous_reference) ||
      !loom_storage_facts_query_reference_from_table(next_table, next,
                                                     &next_reference) ||
      !loom_storage_references_share_backing(&previous_reference,
                                             &next_reference)) {
    inout_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return iree_ok_status();
  }

  loom_storage_reference_facts_t reference = {
      .minimum_alignment = previous_reference.minimum_alignment ==
                                   next_reference.minimum_alignment
                               ? next_reference.minimum_alignment
                               : 1,
      .backing_value_id = next_reference.backing_value_id,
      .storage_space = next_reference.storage_space,
  };
  IREE_RETURN_IF_ERROR(loom_storage_widen_fact_field(
      target, previous_table, previous_reference.byte_offset, next_table,
      next_reference.byte_offset, &reference.byte_offset));
  IREE_RETURN_IF_ERROR(loom_storage_widen_fact_field(
      target, previous_table, previous_reference.target_byte_offset, next_table,
      next_reference.target_byte_offset, &reference.target_byte_offset));
  IREE_RETURN_IF_ERROR(loom_storage_widen_fact_field(
      target, previous_table, previous_reference.valid_byte_length, next_table,
      next_reference.valid_byte_length, &reference.valid_byte_length));
  return loom_storage_facts_make_reference(&target->context, &reference,
                                           inout_facts);
}

const loom_value_fact_domain_t loom_storage_fact_domain = {
    .extensions_equal = loom_storage_extensions_equal,
    .clone_extension = loom_storage_clone_extension,
    .meet_extension = loom_storage_meet_extension,
    .widen_extension = loom_storage_widen_extension,
};
