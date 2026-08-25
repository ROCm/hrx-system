// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/bytecode_template_contract.h"

#include <string.h>

#include "loom/format/bytecode/varint.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/condition.h"

static loom_template_provider_kind_t loom_link_bytecode_template_provider_kind(
    loom_bytecode_symbol_kind_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF:
      return LOOM_TEMPLATE_PROVIDER_KIND_DEF;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL:
      return LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL;
    default:
      IREE_ASSERT_UNREACHABLE("bytecode symbol is not a template provider");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static void loom_link_bytecode_template_attribute_cursor(
    const loom_link_bytecode_template_contract_reader_t* reader,
    const loom_bytecode_symbol_attribute_metadata_t* attribute,
    loom_bytecode_cursor_t* out_cursor) {
  IREE_ASSERT(attribute->value_offset <= reader->bytecode.data_length);
  IREE_ASSERT(attribute->value_length <=
              reader->bytecode.data_length - attribute->value_offset);
  loom_bytecode_cursor_initialize(
      reader->bytecode.data + (iree_host_size_t)attribute->value_offset,
      (iree_host_size_t)attribute->value_length, out_cursor);
}

static const loom_bytecode_symbol_attribute_metadata_t*
loom_link_bytecode_template_retained_attribute(
    const loom_bytecode_symbol_metadata_t* symbol, uint8_t ordinal_plus_one) {
  if (ordinal_plus_one == 0) return NULL;
  IREE_ASSERT(ordinal_plus_one <= symbol->attribute_count);
  return &symbol->attributes[ordinal_plus_one - 1];
}

static iree_status_t loom_link_bytecode_template_read_svarint(
    loom_bytecode_cursor_t* cursor, int64_t* out_value) {
  uint64_t zigzag = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &zigzag));
  *out_value = (int64_t)((zigzag >> 1) ^ -(zigzag & 1));
  return iree_ok_status();
}

// Decodes the closed immediate domain used by target facts and conditions.
// Structural bytecode validation has already established kind, bounds, and
// descriptor compatibility; this projection owns only persistent storage.
static iree_status_t loom_link_bytecode_template_decode_immediate(
    loom_link_bytecode_template_contract_reader_t* reader,
    const loom_attr_descriptor_t* descriptor,
    loom_bytecode_attr_kind_t wire_kind, bool allow_source_string,
    loom_bytecode_cursor_t* cursor, loom_attribute_t* out_attribute) {
  switch (wire_kind) {
    case LOOM_BYTECODE_ATTR_I64: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_I64);
      int64_t value = 0;
      IREE_RETURN_IF_ERROR(
          loom_link_bytecode_template_read_svarint(cursor, &value));
      *out_attribute = loom_attr_i64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_F64: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_F64);
      uint64_t bits = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_cursor_read_u64_le(cursor, &bits));
      double value = 0.0;
      memcpy(&value, &bits, sizeof(value));
      *out_attribute = loom_attr_f64(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_BOOL: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_BOOL);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_cursor_read_u8(cursor, &value));
      IREE_ASSERT(value <= 1);
      *out_attribute = loom_attr_bool(value != 0);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_ENUM: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_ENUM);
      uint8_t value = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_cursor_read_u8(cursor, &value));
      *out_attribute = loom_attr_enum(value);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_STRING: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_STRING);
      if (!allow_source_string) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "target condition parameters must not carry module-local strings");
      }
      uint64_t source_string_ordinal = 0;
      IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &source_string_ordinal));
      IREE_ASSERT(source_string_ordinal < reader->metadata->strings.count);
      *out_attribute =
          loom_attr_string((loom_string_id_t)source_string_ordinal);
      return iree_ok_status();
    }
    case LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET: {
      IREE_ASSERT(descriptor->attr_kind == LOOM_ATTR_SIGNED_ENUM_SET);
      uint8_t word_count = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_cursor_read_u8(cursor, &word_count));
      IREE_ASSERT(word_count <= LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT);
      uint64_t* words = NULL;
      const iree_host_size_t total_word_count =
          (iree_host_size_t)word_count * 2;
      if (total_word_count != 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            reader->arena, total_word_count, sizeof(*words), (void**)&words));
      }
      for (iree_host_size_t i = 0; i < total_word_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_cursor_read_u64_le(cursor, &words[i]));
      }
      *out_attribute = loom_attr_signed_enum_set(words, word_count);
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target applicability attribute kind %u is not self-contained",
          (unsigned)wire_kind);
  }
}

static iree_status_t loom_link_bytecode_template_decode_condition(
    loom_link_bytecode_template_contract_reader_t* reader,
    loom_parameterized_attr_kind_t expected_family_kind,
    loom_bytecode_cursor_t* cursor, loom_attribute_t* out_condition) {
  uint64_t family_name_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &family_name_ordinal));
  IREE_ASSERT(family_name_ordinal < reader->metadata->strings.count);
  const loom_parameterized_attr_descriptor_t* family =
      loom_context_lookup_parameterized_attr_by_name(
          reader->context,
          reader->metadata->strings.values[family_name_ordinal]);
  IREE_ASSERT(family != NULL);
  IREE_ASSERT(expected_family_kind == LOOM_PARAMETERIZED_ATTR_KIND_ANY ||
              expected_family_kind == family->kind);

  uint64_t present_count = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &present_count));
  IREE_ASSERT(present_count <= family->parameter_count);
  loom_attribute_t* slots = NULL;
  if (family->parameter_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(reader->arena, family->parameter_count,
                                  sizeof(*slots), (void**)&slots));
    memset(slots, 0, family->parameter_count * sizeof(*slots));
  }
  for (uint64_t i = 0; i < present_count; ++i) {
    uint64_t parameter_name_ordinal = 0;
    IREE_RETURN_IF_ERROR(loom_uvarint_decode(cursor, &parameter_name_ordinal));
    IREE_ASSERT(parameter_name_ordinal < reader->metadata->strings.count);
    uint8_t parameter_index = LOOM_ATTR_INDEX_NONE;
    const loom_attr_descriptor_t* parameter = loom_attr_descriptor_find_by_name(
        family->parameter_descriptors, family->parameter_count,
        reader->metadata->strings.values[parameter_name_ordinal],
        &parameter_index);
    IREE_ASSERT(parameter != NULL);
    uint8_t raw_kind = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_cursor_read_u8(cursor, &raw_kind));
    IREE_ASSERT(raw_kind < LOOM_BYTECODE_ATTR_COUNT);
    IREE_RETURN_IF_ERROR(loom_link_bytecode_template_decode_immediate(
        reader, parameter, (loom_bytecode_attr_kind_t)raw_kind,
        /*allow_source_string=*/false, cursor, &slots[parameter_index]));
  }
  *out_condition = loom_make_parameterized_attr(family->kind, slots,
                                                family->parameter_count);
  return iree_ok_status();
}

static iree_status_t loom_link_bytecode_template_decode_conditions(
    loom_link_bytecode_template_contract_reader_t* reader,
    const loom_bytecode_symbol_metadata_t* symbol,
    const loom_func_like_vtable_t* function,
    const loom_target_condition_t** out_conditions,
    uint16_t* out_condition_count) {
  *out_conditions = NULL;
  *out_condition_count = 0;
  const loom_bytecode_symbol_attribute_metadata_t* attribute =
      loom_link_bytecode_template_retained_attribute(
          symbol, symbol->template_requires_attribute_ordinal_plus_one);
  if (attribute == NULL) return iree_ok_status();
  IREE_ASSERT(attribute->attribute_index == function->requires_attr_index);
  IREE_ASSERT(attribute->kind == LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY);

  loom_bytecode_cursor_t cursor;
  loom_link_bytecode_template_attribute_cursor(reader, attribute, &cursor);
  uint64_t condition_count = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(&cursor, &condition_count));
  IREE_ASSERT(condition_count <= UINT16_MAX);
  loom_target_condition_t* conditions = NULL;
  if (condition_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, (iree_host_size_t)condition_count, sizeof(*conditions),
        (void**)&conditions));
  }
  const loom_parameterized_attr_kind_t expected_family_kind =
      reader->metadata->ops.entries[symbol->defining_op_ordinal]
          .vtable->attr_descriptors[function->requires_attr_index]
          .reference.parameterized_attr_kind;
  for (uint64_t i = 0; i < condition_count; ++i) {
    loom_attribute_t value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_link_bytecode_template_decode_condition(
        reader, expected_family_kind, &cursor, &value));
    const loom_target_condition_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(
        loom_target_condition_resolve(reader->context, value, &descriptor));
    conditions[i] = (loom_target_condition_t){
        .descriptor = descriptor,
        .value = value,
    };
  }
  IREE_ASSERT(loom_bytecode_cursor_is_empty(&cursor));
  *out_conditions = conditions;
  *out_condition_count = (uint16_t)condition_count;
  return iree_ok_status();
}

static iree_status_t loom_link_bytecode_template_target_symbol_ordinal(
    loom_link_bytecode_template_contract_reader_t* reader,
    const loom_bytecode_symbol_metadata_t* symbol,
    const loom_func_like_vtable_t* function, uint32_t* out_symbol_ordinal) {
  *out_symbol_ordinal = UINT32_MAX;
  const loom_bytecode_symbol_attribute_metadata_t* attribute =
      loom_link_bytecode_template_retained_attribute(
          symbol, symbol->template_target_attribute_ordinal_plus_one);
  if (attribute == NULL) return iree_ok_status();
  IREE_ASSERT(attribute->attribute_index == function->target_attr_index);
  IREE_ASSERT(attribute->kind == LOOM_BYTECODE_ATTR_SYMBOL);
  loom_bytecode_cursor_t cursor;
  loom_link_bytecode_template_attribute_cursor(reader, attribute, &cursor);
  uint64_t source_name_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_uvarint_decode(&cursor, &source_name_ordinal));
  IREE_ASSERT(loom_bytecode_cursor_is_empty(&cursor));
  if (source_name_ordinal == 0) return iree_ok_status();
  const bool found = loom_bytecode_module_metadata_lookup_symbol_ordinal(
      reader->metadata, (uint32_t)source_name_ordinal, out_symbol_ordinal);
  IREE_ASSERT(found);
  return iree_ok_status();
}

static bool loom_link_bytecode_template_target_is_projected(
    const loom_link_bytecode_template_contract_reader_t* reader,
    uint32_t source_symbol_ordinal) {
  return reader->targets.projected_words != NULL &&
         (reader->targets.projected_words[source_symbol_ordinal >> 6] &
          (UINT64_C(1) << (source_symbol_ordinal & 63u))) != 0;
}

static iree_status_t loom_link_bytecode_template_prepare_target_cache(
    loom_link_bytecode_template_contract_reader_t* reader) {
  if (reader->targets.projected_words != NULL) return iree_ok_status();
  const iree_host_size_t symbol_count = reader->metadata->symbol_count;
  const iree_host_size_t word_count = (symbol_count + 63) / 64;
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, symbol_count, sizeof(*reader->targets.values),
        (void**)&reader->targets.values));
    memset(reader->targets.values, 0,
           symbol_count * sizeof(*reader->targets.values));
  }
  if (word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        reader->arena, word_count, sizeof(*reader->targets.projected_words),
        (void**)&reader->targets.projected_words));
    memset(reader->targets.projected_words, 0,
           word_count * sizeof(*reader->targets.projected_words));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_bytecode_template_project_target_facts(
    loom_link_bytecode_template_contract_reader_t* reader,
    uint32_t source_symbol_ordinal,
    const loom_target_facts_t** out_target_facts) {
  IREE_ASSERT(source_symbol_ordinal < reader->metadata->symbol_count);
  IREE_RETURN_IF_ERROR(
      loom_link_bytecode_template_prepare_target_cache(reader));
  if (loom_link_bytecode_template_target_is_projected(reader,
                                                      source_symbol_ordinal)) {
    *out_target_facts = reader->targets.values[source_symbol_ordinal];
    return iree_ok_status();
  }

  const loom_bytecode_symbol_metadata_t* symbol =
      &reader->metadata->symbols[source_symbol_ordinal];
  const loom_op_vtable_t* vtable = NULL;
  if (symbol->defining_op_ordinal != UINT32_MAX) {
    IREE_ASSERT(symbol->defining_op_ordinal < reader->metadata->ops.count);
    vtable = reader->metadata->ops.entries[symbol->defining_op_ordinal].vtable;
  }
  const loom_target_facts_t* facts = NULL;
  if (vtable != NULL && vtable->target_like != NULL) {
    loom_attribute_t* attributes = NULL;
    if (vtable->attribute_count != 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(reader->arena, vtable->attribute_count,
                                    sizeof(*attributes), (void**)&attributes));
      memset(attributes, 0, vtable->attribute_count * sizeof(*attributes));
    }
    for (uint8_t i = 0; i < symbol->attribute_count; ++i) {
      const loom_bytecode_symbol_attribute_metadata_t* attribute =
          &symbol->attributes[i];
      if (attribute->attribute_index ==
          vtable->target_like->symbol_attr_index) {
        continue;
      }
      loom_bytecode_cursor_t cursor;
      loom_link_bytecode_template_attribute_cursor(reader, attribute, &cursor);
      IREE_RETURN_IF_ERROR(loom_link_bytecode_template_decode_immediate(
          reader, &vtable->attr_descriptors[attribute->attribute_index],
          attribute->kind, /*allow_source_string=*/true, &cursor,
          &attributes[attribute->attribute_index]));
      IREE_ASSERT(loom_bytecode_cursor_is_empty(&cursor));
    }

    const loom_target_like_descriptor_t* descriptor =
        vtable->target_like->descriptor;
    const uint8_t selector =
        loom_attr_as_enum(attributes[vtable->target_like->selector_attr_index]);
    const loom_target_bundle_t* row_bundle =
        loom_target_bundle_table_lookup(descriptor->bundle_table, selector);
    IREE_ASSERT(row_bundle != NULL);
    loom_target_facts_t* mutable_facts = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(reader->arena, descriptor->fact_type->storage_size,
                            (void**)&mutable_facts));
    const loom_target_record_view_t record = {
        .descriptor = descriptor,
        .name = symbol->name,
        .attributes = attributes,
        .attribute_count = vtable->attribute_count,
        .selector = selector,
        .strings =
            {
                .values = reader->metadata->strings.values,
                .count = reader->metadata->strings.count,
            },
    };
    loom_target_facts_project_record(&record, row_bundle, mutable_facts);
    facts = mutable_facts;
  }

  reader->targets.values[source_symbol_ordinal] = facts;
  reader->targets.projected_words[source_symbol_ordinal >> 6] |=
      UINT64_C(1) << (source_symbol_ordinal & 63u);
  *out_target_facts = facts;
  return iree_ok_status();
}

iree_status_t loom_link_bytecode_template_contract_reader_initialize(
    iree_const_byte_span_t bytecode, loom_context_t* context,
    const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* arena,
    loom_link_bytecode_template_contract_reader_t* out_reader) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_reader);
  *out_reader = (loom_link_bytecode_template_contract_reader_t){
      .bytecode = bytecode,
      .context = context,
      .metadata = metadata,
      .arena = arena,
      .contracts = {.count = metadata->symbol_count},
  };
  if (metadata->symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, metadata->symbol_count, sizeof(*out_reader->contracts.values),
        (void**)&out_reader->contracts.values));
    memset(out_reader->contracts.values, 0,
           metadata->symbol_count * sizeof(*out_reader->contracts.values));
  }
  return iree_ok_status();
}

iree_status_t loom_link_bytecode_template_contract_reader_load(
    loom_link_bytecode_template_contract_reader_t* reader,
    uint32_t source_symbol_ordinal,
    const loom_link_bytecode_template_contract_t** out_contract) {
  IREE_ASSERT_ARGUMENT(reader);
  IREE_ASSERT_ARGUMENT(out_contract);
  IREE_ASSERT(source_symbol_ordinal < reader->contracts.count);
  if (reader->contracts.values[source_symbol_ordinal] != NULL) {
    *out_contract = reader->contracts.values[source_symbol_ordinal];
    return iree_ok_status();
  }

  const loom_bytecode_symbol_metadata_t* symbol =
      &reader->metadata->symbols[source_symbol_ordinal];
  IREE_ASSERT(symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF ||
              symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL);
  IREE_ASSERT(symbol->kernel_workload_argument_count == 0);
  IREE_ASSERT(symbol->template_family_symbol_ordinal <
              reader->metadata->symbol_count);
  IREE_ASSERT(symbol->defining_op_ordinal < reader->metadata->ops.count);
  const loom_op_vtable_t* vtable =
      reader->metadata->ops.entries[symbol->defining_op_ordinal].vtable;
  IREE_ASSERT(vtable->func_like != NULL);

  loom_link_bytecode_template_contract_t* contract = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(reader->arena, sizeof(*contract), (void**)&contract));
  *contract = (loom_link_bytecode_template_contract_t){
      .provider =
          {
              .kind = loom_link_bytecode_template_provider_kind(symbol->kind),
              .has_body = symbol->body_region_payload_ordinal_plus_one != 0,
              .argument_count = symbol->argument_count,
              .result_count = symbol->result_count,
              .predicate_count = symbol->predicate_count,
              .name = symbol->name,
              .priority = (int64_t)symbol->priority,
              .predicates = symbol->predicates,
          },
      .target_symbol_ordinal = UINT32_MAX,
  };
  IREE_RETURN_IF_ERROR(loom_link_bytecode_template_target_symbol_ordinal(
      reader, symbol, vtable->func_like, &contract->target_symbol_ordinal));
  IREE_RETURN_IF_ERROR(loom_link_bytecode_template_decode_conditions(
      reader, symbol, vtable->func_like, &contract->provider.target_conditions,
      &contract->provider.target_condition_count));
  if (contract->target_symbol_ordinal != UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_link_bytecode_template_project_target_facts(
        reader, contract->target_symbol_ordinal,
        &contract->provider.target_facts));
  }
  reader->contracts.values[source_symbol_ordinal] = contract;
  *out_contract = contract;
  return iree_ok_status();
}
