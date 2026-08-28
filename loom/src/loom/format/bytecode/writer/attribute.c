// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/attribute.h"

#include <string.h>

static iree_status_t loom_bytecode_write_attr_value_at_depth(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t aggregate_depth);

static iree_status_t loom_bytecode_write_parameterized_attr_payload(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind, uint8_t aggregate_depth) {
  if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aggregate attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  const loom_parameterized_attr_descriptor_t* family_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_get_parameterized_attr(
      numbering, attr, descriptor, expected_descriptor_kind,
      &family_descriptor));
  uint8_t present_count = 0;
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_parameter_is_present(
        family_descriptor, attr.parameterized_slots[i], i, &present));
    if (present) ++present_count;
  }
  uint32_t family_name_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, loom_bstring_view(family_descriptor->name), &family_name_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, family_name_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_count));
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    const loom_attribute_t value = attr.parameterized_slots[i];
    if (loom_attr_is_absent(value)) continue;
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    uint32_t parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(parameter_descriptor),
        &parameter_name_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, parameter_name_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value_at_depth(
        writer, numbering, value_numbering, value, parameter_descriptor,
        aggregate_depth + 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_attr_value_at_depth(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t aggregate_depth) {
  switch (attr.kind) {
    case LOOM_ATTR_I64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_I64));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_svarint(writer, attr.i64));
      break;
    }
    case LOOM_ATTR_U64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_U64));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.u64));
      break;
    }
    case LOOM_ATTR_F64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_F64));
      uint8_t bytes[8];
      memcpy(bytes, &attr.f64, 8);
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, bytes, 8));
      break;
    }
    case LOOM_ATTR_STRING: {
      uint32_t string_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &string_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_STRING));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_BOOL));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, attr.raw ? 1 : 0));
      break;
    }
    case LOOM_ATTR_ENUM: {
      uint8_t ordinal = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_ordinal(attr, descriptor, &ordinal));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_ENUM));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, ordinal));
      break;
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      loom_enum_array_t array = loom_enum_array_empty();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_array(attr, descriptor, &array));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_ENUM_ARRAY));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, array.count));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write(writer, array.values, array.count));
      break;
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      loom_signed_enum_set_t set = loom_signed_enum_set_empty();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_signed_enum_set(attr, descriptor, &set));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, (uint8_t)set.word_count));
      for (iree_host_size_t i = 0; i < set.word_count * 2; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u64_le(writer, set.words[i]));
      }
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_I64_ARRAY));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_svarint(writer, attr.i64_array[i]));
      }
      break;
    }
    case LOOM_ATTR_BYTES: {
      iree_const_byte_span_t bytes = loom_attr_as_bytes(attr);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_BYTES));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, bytes.data_length));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(writer, bytes.data,
                                                           bytes.data_length));
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      uint32_t string_writer_id = 0;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_SYMBOL));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      break;
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_symbol_ref_array_t array = loom_symbol_ref_array_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_get_symbol_collection(
          numbering, attr, descriptor, &array));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, attr.kind == LOOM_ATTR_SYMBOL_SET
                      ? LOOM_BYTECODE_ATTR_SYMBOL_SET
                      : LOOM_BYTECODE_ATTR_SYMBOL_ARRAY));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, array.count));
      for (iree_host_size_t i = 0; i < array.count; ++i) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[array.values[i].symbol_id];
        uint32_t string_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(writer, string_writer_id));
      }
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      uint32_t type_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, type, &type_writer_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_TYPE));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_PREDICATE_LIST));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->kind));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(writer, predicate->arg_count));
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          uint8_t tag = predicate->arg_tags[arg_index];
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, tag));
          switch (tag) {
            case LOOM_PRED_ARG_VALUE: {
              if (value_numbering) {
                uint32_t value_number = 0;
                IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
                    value_numbering,
                    (loom_value_id_t)predicate->args[arg_index],
                    &value_number));
                IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                    writer, value_number));
              } else {
                loom_string_id_t name_id =
                    (loom_string_id_t)predicate->args[arg_index];
                uint32_t string_writer_id = 0;
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_numbering_intern_module_string(
                        numbering, name_id, &string_writer_id));
                IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                    writer, string_writer_id));
              }
              break;
            }
            case LOOM_PRED_ARG_CONST: {
              IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_svarint(
                  writer, predicate->args[arg_index]));
              break;
            }
            default:
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "unknown predicate arg tag %d", (int)tag);
          }
        }
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_u8(writer, LOOM_BYTECODE_ATTR_DICT));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      // Dict attrs are stored canonically at IR construction time; emit that
      // order directly so logically equivalent dicts produce stable bytes.
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_named_attr_t* entry = &attr.dict_entries[i];
        uint32_t key_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, entry->name_id, &key_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value_at_depth(
            writer, numbering, value_numbering, entry->value, NULL,
            aggregate_depth + 1));
      }
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_ENCODING));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.encoding_id));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_PARAMETERIZED));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_parameterized_attr_payload(
          writer, numbering, value_numbering, attr, descriptor,
          LOOM_ATTR_PARAMETERIZED, aggregate_depth));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (!descriptor ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY ||
          aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attr.count > 0 && !attr.parameterized_array)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array has no descriptor or malformed "
            "storage");
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
          writer, LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_write_parameterized_attr_payload(
            writer, numbering, value_numbering, attr.parameterized_array[i],
            descriptor, LOOM_ATTR_PARAMETERIZED_ARRAY, aggregate_depth + 1));
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported attribute kind %d", (int)attr.kind);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_attr_value(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  return loom_bytecode_write_attr_value_at_depth(
      writer, numbering, value_numbering, attr, descriptor,
      /*aggregate_depth=*/0);
}

iree_status_t loom_bytecode_write_scoped_enum(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    loom_attribute_t attr) {
  if (attr.kind != LOOM_ATTR_SCOPED_ENUM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scoped enum field has attribute kind %u",
                            (unsigned)attr.kind);
  }
  if (!numbering->low_repr.active_descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum attribute is outside a representation contract");
  }
  const iree_string_view_t key =
      loom_low_repr_descriptor_key(&numbering->low_repr.environment,
                                   numbering->low_repr.active_descriptor_set,
                                   loom_attr_as_scoped_enum(attr));
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scoped enum ordinal is outside the active representation contract");
  }
  uint32_t string_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, key, &string_writer_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
      writer, LOOM_BYTECODE_ATTR_SCOPED_ENUM));
  return loom_bytecode_page_writer_write_uvarint(writer, string_writer_id);
}

static iree_status_t loom_bytecode_emit_attr_value_at_depth(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t aggregate_depth);

static iree_status_t loom_bytecode_emit_parameterized_attr_payload(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    loom_attr_kind_t expected_descriptor_kind, uint8_t aggregate_depth) {
  if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aggregate attribute nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
  }
  const loom_parameterized_attr_descriptor_t* family_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_bytecode_get_parameterized_attr(
      numbering, attr, descriptor, expected_descriptor_kind,
      &family_descriptor));
  uint8_t present_count = 0;
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_parameter_is_present(
        family_descriptor, attr.parameterized_slots[i], i, &present));
    if (present) ++present_count;
  }
  uint32_t family_name_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
      numbering, loom_bstring_view(family_descriptor->name), &family_name_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, family_name_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, present_count));
  for (uint8_t i = 0; i < family_descriptor->parameter_count; ++i) {
    const loom_attribute_t value = attr.parameterized_slots[i];
    if (loom_attr_is_absent(value)) continue;
    const loom_attr_descriptor_t* parameter_descriptor =
        &family_descriptor->parameter_descriptors[i];
    uint32_t parameter_name_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(parameter_descriptor),
        &parameter_name_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_uvarint(builder, parameter_name_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value_at_depth(
        builder, numbering, value_numbering, value, parameter_descriptor,
        aggregate_depth + 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_emit_attr_value_at_depth(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor,
    uint8_t aggregate_depth) {
  switch (attr.kind) {
    case LOOM_ATTR_I64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_I64));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_svarint(builder, attr.i64));
      break;
    }
    case LOOM_ATTR_U64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_U64));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.u64));
      break;
    }
    case LOOM_ATTR_F64: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_F64));
      uint64_t bits = 0;
      memcpy(&bits, &attr.f64, sizeof(bits));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, bits));
      break;
    }
    case LOOM_ATTR_STRING: {
      uint32_t string_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr.string_id, &string_writer_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_STRING));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, string_writer_id));
      break;
    }
    case LOOM_ATTR_BOOL: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_BOOL));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, attr.raw ? 1 : 0));
      break;
    }
    case LOOM_ATTR_ENUM: {
      uint8_t ordinal = 0;
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_ordinal(attr, descriptor, &ordinal));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_ENUM));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, ordinal));
      break;
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      loom_enum_array_t array = loom_enum_array_empty();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_enum_array(attr, descriptor, &array));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_ENUM_ARRAY));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, array.count));
      if (array.count > 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            builder,
            iree_make_string_view((const char*)array.values, array.count)));
      }
      break;
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      loom_signed_enum_set_t set = loom_signed_enum_set_empty();
      IREE_RETURN_IF_ERROR(
          loom_bytecode_get_signed_enum_set(attr, descriptor, &set));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_SIGNED_ENUM_SET));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, (uint8_t)set.word_count));
      for (iree_host_size_t i = 0; i < set.word_count * 2; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_u64_le(builder, set.words[i]));
      }
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_I64_ARRAY));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_svarint(builder, attr.i64_array[i]));
      }
      break;
    }
    case LOOM_ATTR_BYTES: {
      iree_const_byte_span_t bytes = loom_attr_as_bytes(attr);
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_BYTES));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, bytes.data_length));
      if (bytes.data_length > 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            builder,
            iree_make_string_view((const char*)bytes.data, bytes.data_length)));
      }
      break;
    }
    case LOOM_ATTR_SYMBOL: {
      loom_symbol_ref_t ref = attr.symbol;
      uint32_t string_writer_id = 0;
      if (loom_symbol_ref_is_valid(ref) &&
          ref.symbol_id < numbering->module->symbols.count) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[ref.symbol_id];
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_SYMBOL));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, string_writer_id));
      break;
    }
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      loom_symbol_ref_array_t array = loom_symbol_ref_array_empty();
      IREE_RETURN_IF_ERROR(loom_bytecode_get_symbol_collection(
          numbering, attr, descriptor, &array));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(
          builder, attr.kind == LOOM_ATTR_SYMBOL_SET
                       ? LOOM_BYTECODE_ATTR_SYMBOL_SET
                       : LOOM_BYTECODE_ATTR_SYMBOL_ARRAY));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, array.count));
      for (iree_host_size_t i = 0; i < array.count; ++i) {
        const loom_symbol_t* target =
            &numbering->module->symbols.entries[array.values[i].symbol_id];
        uint32_t string_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, target->name_id, &string_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_uvarint(builder, string_writer_id));
      }
      break;
    }
    case LOOM_ATTR_TYPE: {
      if (attr.type_id >= numbering->module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u out of range (module has %" PRIhsz " types)",
            (unsigned)attr.type_id, numbering->module->types.count);
      }
      loom_type_t type = numbering->module->types.entries[attr.type_id];
      uint32_t type_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
          numbering, type, &type_writer_id));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_TYPE));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, type_writer_id));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_PREDICATE_LIST));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, predicate->kind));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_u8(builder, predicate->arg_count));
        for (uint8_t arg_index = 0; arg_index < predicate->arg_count;
             ++arg_index) {
          uint8_t tag = predicate->arg_tags[arg_index];
          IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(builder, tag));
          switch (tag) {
            case LOOM_PRED_ARG_VALUE: {
              if (value_numbering) {
                uint32_t value_number = 0;
                IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
                    value_numbering,
                    (loom_value_id_t)predicate->args[arg_index],
                    &value_number));
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_emit_uvarint(builder, value_number));
              } else {
                loom_string_id_t name_id =
                    (loom_string_id_t)predicate->args[arg_index];
                uint32_t string_writer_id = 0;
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_numbering_intern_module_string(
                        numbering, name_id, &string_writer_id));
                IREE_RETURN_IF_ERROR(
                    loom_bytecode_emit_uvarint(builder, string_writer_id));
              }
              break;
            }
            case LOOM_PRED_ARG_CONST: {
              IREE_RETURN_IF_ERROR(loom_bytecode_emit_svarint(
                  builder, predicate->args[arg_index]));
              break;
            }
            default:
              return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "unknown predicate arg tag %d", (int)tag);
          }
        }
      }
      break;
    }
    case LOOM_ATTR_DICT: {
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_DICT));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_named_attr_t* entry = &attr.dict_entries[i];
        uint32_t key_writer_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
            numbering, entry->name_id, &key_writer_id));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_emit_uvarint(builder, key_writer_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_attr_value_at_depth(
            builder, numbering, value_numbering, entry->value, NULL,
            aggregate_depth + 1));
      }
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_ENCODING));
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_uvarint(builder, attr.encoding_id));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_emit_u8(builder, LOOM_BYTECODE_ATTR_PARAMETERIZED));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_parameterized_attr_payload(
          builder, numbering, value_numbering, attr, descriptor,
          LOOM_ATTR_PARAMETERIZED, aggregate_depth));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (!descriptor ||
          descriptor->attr_kind != LOOM_ATTR_PARAMETERIZED_ARRAY ||
          aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attr.count > 0 && !attr.parameterized_array)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parameterized attribute array has no descriptor or malformed "
            "storage");
      }
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_u8(
          builder, LOOM_BYTECODE_ATTR_PARAMETERIZED_ARRAY));
      IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, attr.count));
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_bytecode_emit_parameterized_attr_payload(
            builder, numbering, value_numbering, attr.parameterized_array[i],
            descriptor, LOOM_ATTR_PARAMETERIZED_ARRAY, aggregate_depth + 1));
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported attribute kind %d", (int)attr.kind);
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_attr_value(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    loom_attribute_t attr, const loom_attr_descriptor_t* descriptor) {
  return loom_bytecode_emit_attr_value_at_depth(
      builder, numbering, value_numbering, attr, descriptor,
      /*aggregate_depth=*/0);
}
