// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/builder.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder) {
  out_builder->module = module;
  out_builder->arena = arena;
  out_builder->ip.block = block;
  out_builder->ip.parent_op = NULL;
  out_builder->ip.before_op = NULL;
  out_builder->on_op_finalized.fn = NULL;
  out_builder->on_op_finalized.user_data = NULL;
  out_builder->on_argument_defined = (loom_builder_argument_callback_t){0};
  out_builder->reserved_value_ids = NULL;
  out_builder->reserved_value_count = 0;
  out_builder->reserved_value_next = 0;
}

void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block) {
  builder->ip.block = block;
  builder->ip.before_op = NULL;
}

loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region) {
  loom_builder_ip_t saved = builder->ip;
  builder->ip.block = loom_region_entry_block(region);
  builder->ip.parent_op = parent_op;
  builder->ip.before_op = NULL;
  return saved;
}

void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  builder->ip.parent_op = op->parent_op;
  builder->ip.before_op = (loom_op_t*)op;
}

void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  builder->ip.parent_op = op->parent_op;
  builder->ip.before_op = op->next_op;
}

loom_builder_ip_t loom_builder_save(const loom_builder_t* builder) {
  return builder->ip;
}

void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip) {
  builder->ip = ip;
}

iree_status_t loom_builder_reserve_values(loom_builder_t* builder,
                                          iree_host_size_t count,
                                          loom_value_id_t* out_value_ids) {
  if (builder->reserved_value_count > 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot reserve values: %" PRIhsz
                            " values already reserved",
                            builder->reserved_value_count);
  }
  loom_type_t none_type = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_define_value(builder->module, none_type,
                                                  &out_value_ids[i]));
  }
  builder->reserved_value_ids = out_value_ids;
  builder->reserved_value_count = count;
  builder->reserved_value_next = 0;
  return iree_ok_status();
}

iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id) {
  if (builder->reserved_value_next < builder->reserved_value_count) {
    loom_value_id_t id =
        builder->reserved_value_ids[builder->reserved_value_next++];
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(builder->module, id, type));
    *out_value_id = id;
    return iree_ok_status();
  }
  return loom_module_define_value(builder->module, type, out_value_id);
}

iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(
      loom_module_define_value(builder->module, type, out_value_id));
  IREE_RETURN_IF_ERROR(
      loom_block_add_arg(builder->module, block, *out_value_id));
  return builder->on_argument_defined.fn
             ? builder->on_argument_defined.fn(
                   builder->on_argument_defined.user_data, *out_value_id)
             : iree_ok_status();
}

iree_status_t loom_builder_create_region(loom_builder_t* builder, loom_op_t* op,
                                         uint8_t region_index,
                                         loom_block_t** out_entry_block) {
  if (!builder || !builder->module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no module");
  }
  if (!op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "region owner op is NULL");
  }
  if (region_index >= op->region_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "region index %u is out of range for op with %u region(s)",
        (unsigned)region_index, (unsigned)op->region_count);
  }
  loom_region_t* region = NULL;
  IREE_RETURN_IF_ERROR(
      loom_module_allocate_region(builder->module, 1, &region));
  loom_op_regions(op)[region_index] = region;
  if (out_entry_block) {
    *out_entry_block = loom_region_entry_block(region);
  }
  return iree_ok_status();
}

iree_status_t loom_builder_define_result(loom_builder_t* builder,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_result) {
  return loom_builder_define_value(builder, result_type, out_result);
}

iree_status_t loom_builder_define_results(loom_builder_t* builder,
                                          const loom_type_t* result_types,
                                          iree_host_size_t result_count,
                                          loom_value_id_t* result_storage) {
  if (result_count == 0) {
    return iree_ok_status();
  }
  if (!result_types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "result type storage is NULL for non-zero result count");
  }
  if (!result_storage) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "result id storage is NULL for non-zero result count");
  }
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_result(builder, result_types[i],
                                                    &result_storage[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_builder_copy_tied_results(
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_op_t* op) {
  if (tied_result_count == 0) {
    return iree_ok_status();
  }
  if (!op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tied result owner op is NULL");
  }
  if (!tied_results) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "tied result storage is NULL for non-zero tied result count");
  }
  if (tied_result_count > op->tied_result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tied result count %" PRIhsz
                            " exceeds op tied result storage count %u",
                            tied_result_count, (unsigned)op->tied_result_count);
  }
  memcpy(loom_op_tied_results(op), tied_results,
         tied_result_count * sizeof(loom_tied_result_t));
  return iree_ok_status();
}

iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id) {
  return loom_module_intern_string(builder->module, string, out_string_id);
}

iree_status_t loom_builder_check_count_range(iree_host_size_t count,
                                             iree_host_size_t max_count,
                                             iree_string_view_t label) {
  if (count <= max_count) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "%.*s count %" PRIhsz " exceeds max %" PRIhsz,
                          (int)label.size, label.data, count, max_count);
}

iree_status_t loom_builder_copy_i64_array_attr_storage(loom_builder_t* builder,
                                                       const int64_t* values,
                                                       iree_host_size_t count,
                                                       iree_string_view_t label,
                                                       int64_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "i64-array storage output is NULL");
  }
  *out_storage = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, count, sizeof(**out_storage), (void**)out_storage));
  memcpy(*out_storage, values, count * sizeof(**out_storage));
  return iree_ok_status();
}

iree_status_t loom_builder_copy_string_array_attr_storage(
    loom_builder_t* builder, loom_string_id_array_t values,
    iree_string_view_t label, const loom_string_id_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string-array storage output is NULL");
  }
  *out_storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_builder_check_count_range(values.count, UINT16_MAX, label));
  if (values.count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  loom_string_id_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, values.count, sizeof(*storage), (void**)&storage));
  memcpy(storage, values.values, values.count * sizeof(*storage));
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_enum_array_attr_storage(
    loom_builder_t* builder, loom_enum_array_t values, iree_string_view_t label,
    const uint8_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "enum-array storage output is NULL");
  }
  *out_storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_builder_check_count_range(values.count, UINT16_MAX, label));
  if (values.count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, values.count, sizeof(*storage), (void**)&storage));
  memcpy(storage, values.values, values.count * sizeof(*storage));
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_signed_enum_set_attr_storage(
    loom_builder_t* builder, loom_signed_enum_set_t set,
    iree_string_view_t label, const uint64_t** out_storage,
    uint16_t* out_word_count) {
  if (!out_storage || !out_word_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signed enum-set storage outputs must both be non-NULL");
  }
  *out_storage = NULL;
  *out_word_count = 0;

  iree_host_size_t canonical_word_count = 0;
  iree_status_t status =
      loom_signed_enum_set_canonical_word_count(set, &canonical_word_count);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_copy(status, label);
  }
  if (canonical_word_count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  uint64_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, canonical_word_count * 2,
                                sizeof(*storage), (void**)&storage));
  memcpy(storage, set.words, canonical_word_count * sizeof(*storage));
  memcpy(storage + canonical_word_count, set.words + set.word_count,
         canonical_word_count * sizeof(*storage));
  *out_storage = storage;
  *out_word_count = (uint16_t)canonical_word_count;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_symbol_array_attr_storage(
    loom_builder_t* builder, loom_symbol_ref_array_t values,
    iree_string_view_t label, const loom_symbol_ref_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol-array storage output is NULL");
  }
  *out_storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_builder_check_count_range(values.count, UINT16_MAX, label));
  if (values.count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!values.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  loom_symbol_ref_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, values.count, sizeof(*storage), (void**)&storage));
  memcpy(storage, values.values, values.count * sizeof(*storage));
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_copy_predicate_list_attr_storage(
    loom_builder_t* builder, const loom_predicate_t* predicates,
    iree_host_size_t count, iree_string_view_t label,
    loom_predicate_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate-list storage output is NULL");
  }
  *out_storage = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!predicates) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero count",
                            (int)label.size, label.data);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, count, sizeof(**out_storage), (void**)out_storage));
  memcpy(*out_storage, predicates, count * sizeof(**out_storage));
  return iree_ok_status();
}

iree_status_t loom_builder_copy_bytes_attr_storage(
    loom_builder_t* builder, iree_const_byte_span_t bytes,
    iree_string_view_t label, const uint8_t** out_storage) {
  if (!out_storage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte storage output is NULL");
  }
  *out_storage = NULL;
  if (bytes.data_length == 0) {
    return iree_ok_status();
  }
  if (bytes.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "%.*s byte length %" PRIhsz " exceeds max %u",
                            (int)label.size, label.data, bytes.data_length,
                            UINT32_MAX);
  }
  if (!builder || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no arena");
  }
  if (!bytes.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s storage is NULL for non-zero byte length",
                            (int)label.size, label.data);
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bytes.data_length, sizeof(*storage), (void**)&storage));
  memcpy(storage, bytes.data, bytes.data_length);
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t loom_builder_set_operand_dict(
    loom_builder_t* builder, loom_named_value_slice_t named_values,
    loom_value_id_t* operand_storage, loom_attribute_t* out_names_attr) {
  if (!out_names_attr) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand dictionary names attribute output is NULL");
  }
  *out_names_attr = loom_attr_absent();
  if (!builder || !builder->module || !builder->arena) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no module or arena");
  }
  if (named_values.count == 0) {
    return iree_ok_status();
  }
  if (!named_values.entries) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty operand dictionary has a NULL entry pointer");
  }
  if (!operand_storage) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty operand dictionary has a NULL operand storage pointer");
  }
  if (named_values.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "operand dictionary has %" PRIhsz
                            " entries, max %u",
                            named_values.count, (unsigned)UINT16_MAX);
  }

  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    const loom_named_value_t entry = named_values.entries[i];
    if (entry.reserved != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary entry reserved bits must be zero");
    }
    if (entry.name_id == LOOM_STRING_ID_INVALID ||
        entry.name_id >= builder->module->strings.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary key string id %u is out of range (module has "
          "%" PRIhsz " strings)",
          entry.name_id, builder->module->strings.count);
    }
    if (entry.value_id == LOOM_VALUE_ID_INVALID ||
        entry.value_id >= builder->module->values.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand dictionary value id %u is out of range (module has %" PRIhsz
          " values)",
          entry.value_id, builder->module->values.count);
    }
  }

  loom_named_attr_t* name_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->module->arena, named_values.count,
                                sizeof(*name_entries), (void**)&name_entries));
  // Until canonical positions are assigned, each entry carries its paired SSA
  // value. Sorting the owned entries keeps names and values together without
  // retaining a separate permutation or caller-owned payload.
  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    name_entries[i] = (loom_named_attr_t){
        .name_id = named_values.entries[i].name_id,
        .reserved = 0,
        .value = loom_attr_i64(named_values.entries[i].value_id),
    };
  }
  loom_module_sort_attr_dict_entries(builder->module, name_entries,
                                     named_values.count);
  for (iree_host_size_t i = 1; i < named_values.count; ++i) {
    if (name_entries[i - 1].name_id == name_entries[i].name_id) {
      iree_string_view_t name = loom_string_table_get(&builder->module->strings,
                                                      name_entries[i].name_id);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate operand dictionary key '%.*s'",
                              (int)name.size, name.data);
    }
  }
  for (iree_host_size_t i = 0; i < named_values.count; ++i) {
    operand_storage[i] = (loom_value_id_t)name_entries[i].value.i64;
    name_entries[i].value = loom_attr_i64((int64_t)i);
  }
  *out_names_attr =
      loom_make_canonical_attr_dict(name_entries, named_values.count);
  return iree_ok_status();
}

static iree_status_t loom_builder_validate_operand_segments(
    const loom_op_vtable_t* vtable, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count) {
  uint8_t expected_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (expected_segment_count == 0) {
    if (operand_segment_count != 0 || operand_segment_counts) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "operand segment counts provided for non-segmented op");
    }
    return iree_ok_status();
  }
  if (!operand_segment_counts ||
      operand_segment_count != expected_segment_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "segmented op requires %u operand segment count(s), got %u",
        expected_segment_count, operand_segment_count);
  }
  uint32_t total_count = 0;
  for (uint8_t i = 0; i < operand_segment_count; ++i) {
    const loom_operand_descriptor_t* descriptor =
        &vtable->operand_descriptors[i];
    const uint16_t segment_count = operand_segment_counts[i];
    if (!iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC)) {
      const bool optional =
          iree_any_bit_set(descriptor->flags, LOOM_OPERAND_OPTIONAL);
      if ((!optional && segment_count != 1) ||
          (optional && segment_count > 1)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "operand segment %u has %u value(s), expected %s", i,
            (unsigned)segment_count, optional ? "0 or 1" : "1");
      }
    }
    total_count += segment_count;
  }
  if (total_count != operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operand segment counts sum to %u but op has %u operand(s)",
        (unsigned)total_count, (unsigned)operand_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_builder_allocate_op_storage(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  if (!builder->ip.block || !builder->module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no insertion block or module");
  }
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(builder->module->context, kind);
  IREE_RETURN_IF_ERROR(loom_builder_validate_operand_segments(
      vtable, operand_count, operand_segment_counts, operand_segment_count));

  iree_host_size_t successors_size =
      (iree_host_size_t)successor_count * sizeof(loom_block_t*);
  iree_host_size_t regions_size =
      (iree_host_size_t)region_count * sizeof(loom_region_t*);
  iree_host_size_t operands_size =
      (iree_host_size_t)operand_count * sizeof(loom_value_id_t);
  iree_host_size_t results_size =
      (iree_host_size_t)result_count * sizeof(loom_value_id_t);
  iree_host_size_t operand_use_indices_size =
      (iree_host_size_t)operand_count * sizeof(loom_use_index_t);
  iree_host_size_t tied_size =
      (iree_host_size_t)tied_result_count * sizeof(loom_tied_result_t);
  iree_host_size_t operand_segment_counts_size =
      (iree_host_size_t)operand_segment_count * sizeof(uint16_t);

  iree_host_size_t before_attrs = sizeof(loom_op_t) + successors_size +
                                  regions_size + operands_size + results_size +
                                  operand_use_indices_size + tied_size;
  iree_host_size_t aligned_before_attrs =
      (attribute_count > 0 || operand_segment_count > 0)
          ? iree_host_align(before_attrs, iree_alignof(loom_attribute_t))
          : before_attrs;
  iree_host_size_t attrs_size =
      (iree_host_size_t)attribute_count * sizeof(loom_attribute_t);
  iree_host_size_t total_size =
      aligned_before_attrs + attrs_size +
      (iree_host_size_t)attribute_count * sizeof(loom_attribute_owner_id_t) +
      operand_segment_counts_size;

  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(builder->arena, total_size, &allocation));
  memset(allocation, 0, total_size);

  loom_op_t* op = (loom_op_t*)allocation;
  op->kind = kind;
  op->operand_count = operand_count;
  op->result_count = result_count;
  op->successor_count = successor_count;
  op->region_count = region_count;
  op->tied_result_count = tied_result_count;
  op->attribute_count = attribute_count;
  op->traits = vtable ? vtable->traits : LOOM_TRAIT_UNKNOWN_EFFECTS;
  op->location = location;
  op->parent_op = builder->ip.parent_op;
  loom_use_index_t* operand_use_indices = loom_op_operand_use_indices(op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    operand_use_indices[i] = LOOM_USE_INDEX_INVALID;
  }
  if (operand_segment_count > 0) {
    memcpy(loom_op_operand_segment_counts(op), operand_segment_counts,
           operand_segment_counts_size);
  }

  if (!builder->ip.before_op) {
    IREE_RETURN_IF_ERROR(
        loom_block_append_op(builder->module, builder->ip.block, op));
  } else {
    IREE_RETURN_IF_ERROR(loom_block_insert_before_op(
        builder->module, builder->ip.block, builder->ip.before_op, op));
  }

  *out_op = op;
  return iree_ok_status();
}

iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, /*operand_segment_counts=*/NULL,
      /*operand_segment_count=*/0, result_count, /*successor_count=*/0,
      region_count, tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, /*operand_segment_counts=*/NULL,
      /*operand_segment_count=*/0, result_count, successor_count, region_count,
      tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_segmented_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, operand_segment_counts,
      operand_segment_count, result_count, /*successor_count=*/0, region_count,
      tied_result_count, attribute_count, location, out_op);
}

iree_status_t loom_builder_allocate_segmented_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op) {
  return loom_builder_allocate_op_storage(
      builder, kind, operand_count, operand_segment_counts,
      operand_segment_count, result_count, successor_count, region_count,
      tied_result_count, attribute_count, location, out_op);
}

//===----------------------------------------------------------------------===//
// Use-def list maintenance
//===----------------------------------------------------------------------===//

void loom_module_link_symbol_defining_op(loom_module_t* module, loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  if (!vtable || !vtable->symbol_def || !vtable->attr_descriptors) {
    return;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return;
  }
  loom_attribute_t* attrs = loom_op_attrs(op);
  loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[symbol_attr_index]);
  if (loom_symbol_ref_is_valid(ref) && ref.module_id == 0 &&
      ref.symbol_id < module->symbols.count) {
    loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
    if (symbol->defining_op && symbol->defining_op != op) {
      return;
    }
    symbol->defining_op = op;
    symbol->definition = vtable->symbol_def;
    symbol->kind = vtable->symbol_def->bytecode_kind;
    const uint8_t visibility_attr_index_plus_one =
        vtable->symbol_def->visibility_attr_index_plus_one;
    if (visibility_attr_index_plus_one) {
      const uint8_t visibility_attr_index = visibility_attr_index_plus_one - 1;
      symbol->flags &= (loom_symbol_flags_t)~LOOM_SYMBOL_FLAG_PUBLIC;
      if (visibility_attr_index < op->attribute_count &&
          loom_attr_as_enum(attrs[visibility_attr_index]) != 0) {
        symbol->flags |= LOOM_SYMBOL_FLAG_PUBLIC;
      }
    }
    symbol->flags &= (loom_symbol_flags_t)~LOOM_SYMBOL_FLAG_LINK_ONCE;
    const uint8_t linkage_attr_index_plus_one =
        vtable->symbol_def->linkage_attr_index_plus_one;
    if (linkage_attr_index_plus_one) {
      const uint8_t linkage_attr_index = linkage_attr_index_plus_one - 1;
      if (linkage_attr_index < op->attribute_count &&
          loom_attr_as_enum(attrs[linkage_attr_index]) ==
              LOOM_SYMBOL_LINKAGE_LINK_ONCE) {
        symbol->flags |= LOOM_SYMBOL_FLAG_LINK_ONCE;
      }
    }
    symbol->flags &= (loom_symbol_flags_t)~LOOM_SYMBOL_FLAG_RETAIN;
    const uint8_t retain_attr_index_plus_one =
        vtable->symbol_def->retain_attr_index_plus_one;
    if (retain_attr_index_plus_one) {
      const uint8_t retain_attr_index = retain_attr_index_plus_one - 1;
      if (retain_attr_index < op->attribute_count &&
          loom_attr_as_enum(attrs[retain_attr_index]) != 0) {
        symbol->flags |= LOOM_SYMBOL_FLAG_RETAIN;
      }
    }
  }
}

iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op) {
  // Verify reserved values were fully consumed.
  if (builder->reserved_value_count > 0) {
    if (builder->reserved_value_next != builder->reserved_value_count) {
      iree_host_size_t consumed = builder->reserved_value_next;
      iree_host_size_t reserved = builder->reserved_value_count;
      builder->reserved_value_ids = NULL;
      builder->reserved_value_count = 0;
      builder->reserved_value_next = 0;
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "reserved %" PRIhsz
                              " value(s) but op consumed %" PRIhsz,
                              reserved, consumed);
    }
    builder->reserved_value_ids = NULL;
    builder->reserved_value_count = 0;
    builder->reserved_value_next = 0;
  }

  IREE_RETURN_IF_ERROR(loom_op_record_operand_uses(builder->module, op));
  // Set the def pointer on each result value.
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_module_value(builder->module, results[i])->def =
          loom_value_def_make_op(op, i);
    }
  }
  // Result type uses are installed when values are defined or their types
  // change. Finalization only assigns the operation definition site.
  IREE_RETURN_IF_ERROR(
      loom_module_refresh_op_attribute_uses(builder->module, op));
  // Wire the symbol table entry for symbol-defining ops so that
  // loom_func_like_cast can find the defining op without a scan.
  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
  loom_op_refresh_effective_traits(builder->module, op);
  if (vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    loom_module_link_symbol_defining_op(builder->module, op, vtable);
  }
  loom_module_record_op_summaries(builder->module, op);
  // Notify the rewriter (or other listener) that the op's direct fields are
  // fully wired. Callers may populate nested regions after their parent
  // builder returns.
  if (builder->on_op_finalized.fn) {
    IREE_RETURN_IF_ERROR(
        builder->on_op_finalized.fn(builder->on_op_finalized.user_data, op));
  }
  return iree_ok_status();
}
