// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/descriptors_verify.h"

#include <inttypes.h>

static iree_status_t loom_low_verify_pointer_for_count(const void* pointer,
                                                       uint32_t count,
                                                       const char* table_name) {
  if (count != 0 && pointer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set table '%s' is required when "
                            "its count is non-zero",
                            table_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_span(uint32_t start, uint32_t count,
                                          uint32_t total_count,
                                          const char* table_name) {
  if (start > total_count || count > total_count - start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor set span [%" PRIu32 ", %" PRIu32
                            ") exceeds table '%s' count %" PRIu32,
                            start, start + count, table_name, total_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_known_flags(uint16_t flags,
                                                 uint16_t known_mask,
                                                 const char* table_name,
                                                 uint32_t table_index) {
  const uint16_t unknown_flags = flags & (uint16_t)~known_mask;
  if (unknown_flags == 0) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "low %s %" PRIu32
                          " has unknown generic flag bits 0x%04x",
                          table_name, table_index, (unsigned)unknown_flags);
}

static uint64_t loom_low_descriptor_bit_mask(uint16_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  if (bit_count >= 64) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << bit_count) - 1;
}

static iree_status_t loom_low_descriptor_set_string_impl(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, bool allow_none,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    if (allow_none) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string offset is NONE");
  }
  if (descriptor_set->string_table.data == NULL ||
      descriptor_set->string_table.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string table is empty");
  }
  if (string_offset >= descriptor_set->string_table.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " exceeds string table length %" PRIu32,
                            string_offset,
                            descriptor_set->string_table.data_length);
  }
  loom_bstring_t bstring = NULL;
  if (!loom_bstring_table_try_get(&descriptor_set->string_table, string_offset,
                                  &bstring)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " does not name a complete B-string",
                            string_offset);
  }
  *out_string = loom_bstring_view(bstring);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_required_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/false, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid required string field '%s'",
                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_optional_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/true, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid optional string field '%s'",
                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_non_empty_required_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name,
    iree_string_view_t* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/false, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid required string field '%s'",
                                  field_name);
  }
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string field '%s' is "
                            "empty",
                            field_name);
  }
  if (out_value != NULL) {
    *out_value = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_stable_id_field(uint64_t actual_id,
                                                     iree_string_view_t key,
                                                     const char* field_name) {
  if (iree_string_view_is_empty(key)) {
    if (actual_id != LOOM_LOW_STABLE_ID_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor set %s must be NONE when its key is absent",
          field_name);
    }
    return iree_ok_status();
  }
  const uint64_t expected_id = loom_low_descriptor_stable_id_from_key(key);
  if (actual_id != expected_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor set %s 0x%016" PRIx64
        " does not match key-derived stable id 0x%016" PRIx64,
        field_name, actual_id, expected_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_tables_present(
    const loom_low_descriptor_set_t* descriptor_set) {
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptors, descriptor_set->descriptor_count,
      "descriptors"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptor_refs, descriptor_set->descriptor_ref_count,
      "descriptor_refs"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_forms, descriptor_set->asm_form_count, "asm_forms"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_operand_indices,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_immediates, descriptor_set->asm_immediate_count,
      "asm_immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->native_asm_values, descriptor_set->native_asm_value_count,
      "native_asm_values"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operands, descriptor_set->operand_count, "operands"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->immediates, descriptor_set->immediate_count,
      "immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->immediate_encoding_slices,
      descriptor_set->immediate_encoding_slice_count,
      "immediate_encoding_slices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->enum_domains, descriptor_set->enum_domain_count,
      "enum_domains"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->enum_values, descriptor_set->enum_value_count,
      "enum_values"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->effects, descriptor_set->effect_count, "effects"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->constraints, descriptor_set->constraint_count,
      "constraints"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->storage_leases, descriptor_set->storage_lease_count,
      "storage_leases"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->reg_classes, descriptor_set->reg_class_count,
      "reg_classes"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->register_parts, descriptor_set->register_part_count,
      "register_parts"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->reg_class_alts, descriptor_set->reg_class_alt_count,
      "reg_class_alts"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->schedule_classes, descriptor_set->schedule_class_count,
      "schedule_classes"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->issue_uses, descriptor_set->issue_use_count,
      "issue_uses"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->resources, descriptor_set->resource_count, "resources"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->hazards, descriptor_set->hazard_count, "hazards"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->pressure_deltas, descriptor_set->pressure_delta_count,
      "pressure_deltas"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->feature_mask_words,
      descriptor_set->feature_mask_word_count, "feature_mask_words"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->encoding_field_values,
      descriptor_set->encoding_field_value_count, "encoding_field_values"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operand_forms, descriptor_set->operand_form_count,
      "operand_forms"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operand_form_matches,
      descriptor_set->operand_form_match_count, "operand_form_matches"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operand_form_operand_indices,
      descriptor_set->operand_form_operand_index_count,
      "operand_form_operand_indices"));
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_refs(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->descriptor_ref_count !=
      descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map count %" PRIu32
                            " does not match descriptor count %" PRIu32,
                            descriptor_set->descriptor_ref_count,
                            descriptor_set->descriptor_count);
  }
  iree_string_view_t previous_key = iree_string_view_empty();
  for (uint32_t i = 0; i < descriptor_set->descriptor_ref_count; ++i) {
    const loom_low_descriptor_ref_t* descriptor_ref =
        &descriptor_set->descriptor_refs[i];
    iree_string_view_t ref_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_ref->key_string_offset, /*allow_none=*/false,
        &ref_key));
    if (i > 0 && iree_string_view_compare(previous_key, ref_key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor reference map is not strictly sorted near '%.*s'",
          (int)ref_key.size, ref_key.data);
    }
    if (descriptor_ref->descriptor_ordinal >=
        descriptor_set->descriptor_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low descriptor reference '%.*s' points at descriptor ordinal "
          "%" PRIu32 " but only %" PRIu32 " descriptors exist",
          (int)ref_key.size, ref_key.data, descriptor_ref->descriptor_ordinal,
          descriptor_set->descriptor_count);
    }
    iree_string_view_t descriptor_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set,
        descriptor_set->descriptors[descriptor_ref->descriptor_ordinal]
            .key_string_offset,
        /*allow_none=*/false, &descriptor_key));
    if (!iree_string_view_equal(ref_key, descriptor_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor reference key '%.*s' does not "
                              "match descriptor %" PRIu32 " key '%.*s'",
                              (int)ref_key.size, ref_key.data,
                              descriptor_ref->descriptor_ordinal,
                              (int)descriptor_key.size, descriptor_key.data);
    }
    previous_key = ref_key;
  }
  return iree_ok_status();
}

static bool loom_low_operand_role_is_valid(loom_low_operand_role_t role) {
  switch (role) {
    case LOOM_LOW_OPERAND_ROLE_RESULT:
    case LOOM_LOW_OPERAND_ROLE_OPERAND:
    case LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT:
    case LOOM_LOW_OPERAND_ROLE_PREDICATE:
    case LOOM_LOW_OPERAND_ROLE_RESOURCE:
    case LOOM_LOW_OPERAND_ROLE_IMPLICIT:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_asm_operand_indices(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index,
    uint32_t start, uint16_t count, bool expect_result) {
  for (uint16_t i = 0; i < count; ++i) {
    const uint16_t operand_index =
        descriptor_set->asm_operand_indices[start + i];
    if (operand_index >= descriptor->operand_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low asm form for descriptor %" PRIu32 " references operand %" PRIu16
          " but descriptor has only %" PRIu16 " operands",
          descriptor_index, operand_index, descriptor->operand_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (descriptor_set->asm_operand_indices[start + j] == operand_index) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm form for descriptor %" PRIu32
                                " references operand %" PRIu16
                                " more than once",
                                descriptor_index, operand_index);
      }
    }
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + operand_index];
    if (expect_result) {
      if (operand_index >= descriptor->result_count ||
          operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm form for descriptor %" PRIu32
                                " result operand %" PRIu16
                                " does not name a descriptor result",
                                descriptor_index, operand_index);
      }
    } else if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " operand %" PRIu16
                              " does not name an explicit packet operand",
                              descriptor_index, operand_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_immediates(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < asm_form->immediate_count; ++i) {
    const uint32_t row_index = asm_form->immediate_start + i;
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[row_index];
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm form for descriptor %" PRIu32
                              " references immediate %" PRIu16
                              " but descriptor has only %" PRIu16 " immediates",
                              descriptor_index, asm_immediate->immediate_index,
                              descriptor->immediate_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      const loom_low_asm_immediate_t* previous =
          &descriptor_set->asm_immediates[asm_form->immediate_start + j];
      if (previous->immediate_index == asm_immediate->immediate_index) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low asm form for descriptor %" PRIu32
            " references immediate %" PRIu16 " more than once",
            descriptor_index, asm_immediate->immediate_index);
      }
    }

    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, asm_immediate->name_string_offset, /*allow_none=*/true,
        &name));
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE &&
        iree_string_view_is_empty(name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " has an empty immediate name",
                              descriptor_index);
    }
    if (!iree_string_view_is_empty(name)) {
      for (uint16_t j = 0; j < i; ++j) {
        const loom_low_asm_immediate_t* previous =
            &descriptor_set->asm_immediates[asm_form->immediate_start + j];
        iree_string_view_t previous_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
            descriptor_set, previous->name_string_offset, /*allow_none=*/true,
            &previous_name));
        if (iree_string_view_equal(previous_name, name)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " uses immediate name '%.*s' more than once",
                                  descriptor_index, (int)name.size, name.data);
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_low_native_asm_value_kind_is_valid(
    loom_low_native_asm_value_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL:
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT:
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_OPERAND:
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64:
    case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_native_asm_values(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < asm_form->native_assembly_value_count; ++i) {
    const uint32_t row_index = asm_form->native_assembly_value_start + i;
    const loom_low_native_asm_value_t* value =
        &descriptor_set->native_asm_values[row_index];
    if (!loom_low_native_asm_value_kind_is_valid(value->kind)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " has invalid native value kind %u",
                              descriptor_index, (unsigned)value->kind);
    }
    if (value->kind == LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL) {
      iree_string_view_t literal = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
          descriptor_set, value->literal_string_offset,
          "native_asm_value.literal", &literal));
      (void)literal;
      if (value->index != 0 || value->bit_width != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low asm form for descriptor %" PRIu32
            " literal native value must not set index or bit width",
            descriptor_index);
      }
      continue;
    }
    if (value->literal_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " non-literal native value has a literal string",
                              descriptor_index);
    }
    switch (value->kind) {
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT: {
        if (value->index >= descriptor->operand_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low asm form for descriptor %" PRIu32
              " native result references operand %" PRIu16
              " but descriptor has only %" PRIu16 " operands",
              descriptor_index, value->index, descriptor->operand_count);
        }
        const loom_low_operand_t* operand =
            &descriptor_set->operands[descriptor->operand_start + value->index];
        if (value->index >= descriptor->result_count ||
            operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " native result operand %" PRIu16
                                  " does not name a descriptor result",
                                  descriptor_index, value->index);
        }
        if (value->bit_width != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " native result must not set bit width",
                                  descriptor_index);
        }
        break;
      }
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_OPERAND: {
        if (value->index >= descriptor->operand_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low asm form for descriptor %" PRIu32
              " native operand references operand %" PRIu16
              " but descriptor has only %" PRIu16 " operands",
              descriptor_index, value->index, descriptor->operand_count);
        }
        const loom_low_operand_t* operand =
            &descriptor_set->operands[descriptor->operand_start + value->index];
        if (!loom_low_operand_role_is_packet_operand(operand->role)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " native operand %" PRIu16
                                  " does not name an explicit packet operand",
                                  descriptor_index, value->index);
        }
        if (value->bit_width != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " native operand must not set bit width",
                                  descriptor_index);
        }
        break;
      }
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64:
        if (value->index >= descriptor->immediate_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low asm form for descriptor %" PRIu32
              " native immediate references immediate %" PRIu16
              " but descriptor has only %" PRIu16 " immediates",
              descriptor_index, value->index, descriptor->immediate_count);
        }
        if (value->bit_width != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low asm form for descriptor %" PRIu32
              " native i64 immediate must not set bit width",
              descriptor_index);
        }
        break;
      case LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX:
        if (value->index >= descriptor->immediate_count) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low asm form for descriptor %" PRIu32
              " native immediate references immediate %" PRIu16
              " but descriptor has only %" PRIu16 " immediates",
              descriptor_index, value->index, descriptor->immediate_count);
        }
        if (value->bit_width == 0 || value->bit_width > 32) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " native unsigned-hex bit width is invalid",
                                  descriptor_index);
        }
        break;
      default:
        IREE_ASSERT_UNREACHABLE("validated above");
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_form(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t asm_form_index,
    iree_string_view_t* out_mnemonic) {
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[asm_form_index];
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, asm_form->mnemonic_string_offset, "asm_form.mnemonic",
      &mnemonic));
  if (asm_form->native_assembly_mnemonic_string_offset !=
      LOOM_LOW_STRING_OFFSET_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
        descriptor_set, asm_form->native_assembly_mnemonic_string_offset,
        "asm_form.native_assembly_mnemonic", NULL));
  }
  if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm form %" PRIu32
                            " references descriptor ordinal %" PRIu32
                            " but only %" PRIu32 " descriptors exist",
                            asm_form_index, asm_form->descriptor_ordinal,
                            descriptor_set->descriptor_count);
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[asm_form->descriptor_ordinal];
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->result_operand_index_start,
      asm_form->result_operand_index_count,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->operand_index_start, asm_form->operand_index_count,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->immediate_start, asm_form->immediate_count,
      descriptor_set->asm_immediate_count, "asm_immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->native_assembly_value_start,
      asm_form->native_assembly_value_count,
      descriptor_set->native_asm_value_count, "native_asm_values"));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_operand_indices(
      descriptor_set, descriptor, asm_form->descriptor_ordinal,
      asm_form->result_operand_index_start,
      asm_form->result_operand_index_count, /*expect_result=*/true));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_operand_indices(
      descriptor_set, descriptor, asm_form->descriptor_ordinal,
      asm_form->operand_index_start, asm_form->operand_index_count,
      /*expect_result=*/false));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_immediates(
      descriptor_set, asm_form, descriptor, asm_form->descriptor_ordinal));
  IREE_RETURN_IF_ERROR(loom_low_verify_native_asm_values(
      descriptor_set, asm_form, descriptor, asm_form->descriptor_ordinal));
  if (out_mnemonic != NULL) {
    *out_mnemonic = mnemonic;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_forms(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->asm_form_count == 0) {
    if (descriptor_set->asm_operand_index_count != 0 ||
        descriptor_set->asm_immediate_count != 0 ||
        descriptor_set->native_asm_value_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor set has asm operand, immediate, or native value rows "
          "but no asm forms");
    }
    return iree_ok_status();
  }
  iree_string_view_t previous_mnemonic = iree_string_view_empty();
  for (uint32_t i = 0; i < descriptor_set->asm_form_count; ++i) {
    const loom_low_asm_form_t* asm_form = &descriptor_set->asm_forms[i];
    iree_string_view_t mnemonic = iree_string_view_empty();
    if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
      // Shared backing storage may carry extension rows for larger descriptor
      // set views. Smaller views keep those rows sorted for lookup, but the
      // extension view owns full payload validation.
      IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
          descriptor_set, asm_form->mnemonic_string_offset, "asm_form.mnemonic",
          &mnemonic));
      if (asm_form->native_assembly_mnemonic_string_offset !=
          LOOM_LOW_STRING_OFFSET_NONE) {
        IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
            descriptor_set, asm_form->native_assembly_mnemonic_string_offset,
            "asm_form.native_assembly_mnemonic", NULL));
      }
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_verify_asm_form(descriptor_set, i, &mnemonic));
    }
    if (i > 0 && iree_string_view_compare(previous_mnemonic, mnemonic) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm forms are not strictly sorted near "
                              "mnemonic '%.*s'",
                              (int)mnemonic.size, mnemonic.data);
    }
    previous_mnemonic = mnemonic;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_binary_constraint(
    const loom_low_constraint_t* constraint, const char* constraint_name) {
  if (constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low %s constraint requires an rhs operand",
                            constraint_name);
  }
  if (constraint->lhs_operand_index == constraint->rhs_operand_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low %s constraint cannot reference the same operand twice",
        constraint_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_constraints(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->lhs_operand_index >= descriptor->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor constraint lhs operand %" PRIu16
                              " exceeds descriptor operand count %" PRIu16,
                              constraint->lhs_operand_index,
                              descriptor->operand_count);
    }
    if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE &&
        constraint->rhs_operand_index >= descriptor->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor constraint rhs operand %" PRIu16
                              " exceeds descriptor operand count %" PRIu16,
                              constraint->rhs_operand_index,
                              descriptor->operand_count);
    }
    const loom_low_operand_t* lhs =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->lhs_operand_index];
    const loom_low_operand_t* rhs = NULL;
    if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE) {
      rhs = &descriptor_set->operands[descriptor->operand_start +
                                      constraint->rhs_operand_index];
    }
    switch (constraint->kind) {
      case LOOM_LOW_CONSTRAINT_KIND_TIED: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "tied"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT ||
            !loom_low_operand_role_is_packet_operand(rhs->role)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low tied constraint requires a result lhs and packet operand "
              "rhs");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "commutable"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_OPERAND ||
            rhs->role != LOOM_LOW_OPERAND_ROLE_OPERAND) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low commutable constraint requires two packet operand rows");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "destructive"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT ||
            !loom_low_operand_role_is_packet_operand(rhs->role)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low destructive constraint requires a result lhs and packet "
              "operand rhs");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
        if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE ||
            lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low early-clobber constraint requires one result operand");
        }
        break;
      case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
      case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
        if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE ||
            lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low unary descriptor constraint requires one result operand");
        }
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

static bool loom_low_operand_form_match_kind_is_valid(
    loom_low_operand_form_match_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64:
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_EXACT_I64:
      return true;
    default:
      return false;
  }
}

static bool loom_low_operand_form_immediate_action_is_valid(
    loom_low_operand_form_immediate_action_t action) {
  switch (action) {
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE:
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_SET_MATCHED_I64:
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_ADD_MATCHED_I64:
      return true;
    default:
      return false;
  }
}

static uint16_t loom_low_descriptor_packet_operand_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  uint16_t count = 0;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    if (loom_low_operand_role_is_packet_operand(operands[i].role) &&
        !iree_any_bit_set(operands[i].flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_low_verify_descriptor_operand_forms(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const uint16_t source_packet_operand_count =
      loom_low_descriptor_packet_operand_count(descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor->operand_form_count; ++i) {
    const uint32_t form_index = descriptor->operand_form_start + i;
    const loom_low_operand_form_t* form =
        &descriptor_set->operand_forms[form_index];
    if (form->replacement_descriptor_ordinal >=
        descriptor_set->descriptor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor %" PRIu32 " operand form %" PRIu32
                              " references replacement descriptor %" PRIu32
                              " but only %" PRIu32 " descriptors exist",
                              descriptor_index, form_index,
                              form->replacement_descriptor_ordinal,
                              descriptor_set->descriptor_count);
    }
    if (form->match_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " operand form %" PRIu32
                              " has no match predicates",
                              descriptor_index, form_index);
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_span(
        form->match_start, form->match_count,
        descriptor_set->operand_form_match_count, "operand_form_matches"));
    for (uint16_t match_index = 0; match_index < form->match_count;
         ++match_index) {
      const loom_low_operand_form_match_t* match =
          &descriptor_set
               ->operand_form_matches[form->match_start + match_index];
      if (match->source_operand_index >= descriptor->operand_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low descriptor %" PRIu32 " operand form %" PRIu32 " match %" PRIu16
            " references source operand %" PRIu16
            " but descriptor has only %" PRIu16 " operands",
            descriptor_index, form_index, match_index,
            match->source_operand_index, descriptor->operand_count);
      }
      const loom_low_operand_t* source_operand =
          &descriptor_set->operands[descriptor->operand_start +
                                    match->source_operand_index];
      if (!loom_low_operand_role_is_packet_operand(source_operand->role) ||
          iree_any_bit_set(source_operand->flags,
                           LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand form %" PRIu32 " match %" PRIu16
            " source operand %" PRIu16 " is not an explicit packet operand",
            descriptor_index, form_index, match_index,
            match->source_operand_index);
      }
      if (match->source_packet_operand_index >= source_packet_operand_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low descriptor %" PRIu32 " operand form %" PRIu32 " match %" PRIu16
            " references source packet operand %" PRIu16
            " but descriptor has only %" PRIu16 " packet operands",
            descriptor_index, form_index, match_index,
            match->source_packet_operand_index, source_packet_operand_count);
      }
      if (!loom_low_operand_form_match_kind_is_valid(match->match_kind)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand form %" PRIu32 " match %" PRIu16
            " has unknown match kind %u",
            descriptor_index, form_index, match_index, match->match_kind);
      }
    }
    if (!loom_low_operand_form_immediate_action_is_valid(
            form->immediate_action)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " operand form %" PRIu32
                              " has unknown immediate action %u",
                              descriptor_index, form_index,
                              form->immediate_action);
    }
    switch (form->immediate_action) {
      case LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE:
        if (form->source_immediate_index !=
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->replacement_immediate_index !=
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->immediate_match_index !=
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low descriptor %" PRIu32 " operand form %" PRIu32
              " has immediate indexes without an immediate action",
              descriptor_index, form_index);
        }
        break;
      case LOOM_LOW_OPERAND_FORM_IMMEDIATE_SET_MATCHED_I64:
        if (form->source_immediate_index !=
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->replacement_immediate_index ==
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->immediate_match_index >= form->match_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low descriptor %" PRIu32 " operand form %" PRIu32
              " has malformed SET_MATCHED_I64 immediate indexes",
              descriptor_index, form_index);
        }
        break;
      case LOOM_LOW_OPERAND_FORM_IMMEDIATE_ADD_MATCHED_I64:
        if (form->source_immediate_index ==
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->replacement_immediate_index ==
                LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE ||
            form->immediate_match_index >= form->match_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low descriptor %" PRIu32 " operand form %" PRIu32
              " has malformed ADD_MATCHED_I64 immediate indexes",
              descriptor_index, form_index);
        }
        break;
      default:
        break;
    }

    const loom_low_descriptor_t* replacement =
        &descriptor_set->descriptors[form->replacement_descriptor_ordinal];
    if (form->source_immediate_index != LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE &&
        form->source_immediate_index >= descriptor->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor %" PRIu32 " operand form %" PRIu32
                              " references source immediate %" PRIu16
                              " but descriptor has only %" PRIu16 " immediates",
                              descriptor_index, form_index,
                              form->source_immediate_index,
                              descriptor->immediate_count);
    }
    if (form->replacement_immediate_index !=
            LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE &&
        form->replacement_immediate_index >= replacement->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low descriptor %" PRIu32 " operand form %" PRIu32
          " references replacement immediate %" PRIu16
          " but replacement descriptor %" PRIu32 " has only %" PRIu16
          " immediates",
          descriptor_index, form_index, form->replacement_immediate_index,
          form->replacement_descriptor_ordinal, replacement->immediate_count);
    }
    if (replacement->result_count != descriptor->result_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " operand form %" PRIu32
          " replacement descriptor %" PRIu32 " has result count %" PRIu16
          " instead of %" PRIu16,
          descriptor_index, form_index, form->replacement_descriptor_ordinal,
          replacement->result_count, descriptor->result_count);
    }
    const uint16_t replacement_packet_operand_count =
        loom_low_descriptor_packet_operand_count(descriptor_set, replacement);
    if (form->operand_map_count != replacement_packet_operand_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " operand form %" PRIu32 " maps %" PRIu16
          " operands but replacement descriptor %" PRIu32 " has %" PRIu16
          " packet operands",
          descriptor_index, form_index, form->operand_map_count,
          form->replacement_descriptor_ordinal,
          replacement_packet_operand_count);
    }
    IREE_RETURN_IF_ERROR(
        loom_low_verify_span(form->operand_map_start, form->operand_map_count,
                             descriptor_set->operand_form_operand_index_count,
                             "operand_form_operand_indices"));
    for (uint16_t map_index = 0; map_index < form->operand_map_count;
         ++map_index) {
      const uint16_t source_packet_operand_index =
          descriptor_set->operand_form_operand_indices[form->operand_map_start +
                                                       map_index];
      if (source_packet_operand_index >= source_packet_operand_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low descriptor %" PRIu32 " operand form %" PRIu32
            " operand map row %" PRIu16
            " references source packet operand "
            "%" PRIu16 " but descriptor has only %" PRIu16 " packet operands",
            descriptor_index, form_index, map_index,
            source_packet_operand_index, source_packet_operand_count);
      }
    }
  }
  return iree_ok_status();
}

static bool loom_low_immediate_kind_is_valid(loom_low_immediate_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      return true;
    default:
      return false;
  }
}

static bool loom_low_effect_kind_is_valid(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return true;
    default:
      return false;
  }
}

static bool loom_low_memory_space_is_valid(loom_low_memory_space_t space) {
  switch (space) {
    case LOOM_LOW_MEMORY_SPACE_NONE:
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_VM_REF:
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return true;
    default:
      return false;
  }
}

static bool loom_low_storage_lease_kind_is_valid(
    loom_low_storage_lease_kind_t kind) {
  return kind == LOOM_LOW_STORAGE_LEASE_SOURCE_READ ||
         kind == LOOM_LOW_STORAGE_LEASE_RESULT_WRITE;
}

static bool loom_low_storage_lease_attachment_is_valid(
    loom_low_storage_lease_attachment_t attachment) {
  return attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND ||
         attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT;
}

static bool loom_low_storage_lease_release_scope_is_valid(
    loom_low_storage_lease_release_scope_t scope) {
  return scope == LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS;
}

static bool loom_low_constraint_kind_is_valid(loom_low_constraint_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_CONSTRAINT_KIND_TIED:
    case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE:
    case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE:
    case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
    case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
    case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_latency_kind_is_valid(loom_low_latency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LATENCY_KIND_EXACT:
    case LOOM_LOW_LATENCY_KIND_ESTIMATE:
    case LOOM_LOW_LATENCY_KIND_VARIABLE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_model_quality_is_valid(loom_low_model_quality_t quality) {
  switch (quality) {
    case LOOM_LOW_MODEL_QUALITY_EXACT:
    case LOOM_LOW_MODEL_QUALITY_CALIBRATED:
    case LOOM_LOW_MODEL_QUALITY_ESTIMATED:
    case LOOM_LOW_MODEL_QUALITY_FALLBACK:
      return true;
    default:
      return false;
  }
}

static bool loom_low_resource_kind_is_valid(loom_low_resource_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_KIND_SCALAR_ALU:
    case LOOM_LOW_RESOURCE_KIND_VECTOR_ALU:
    case LOOM_LOW_RESOURCE_KIND_MATRIX:
    case LOOM_LOW_RESOURCE_KIND_LOAD:
    case LOOM_LOW_RESOURCE_KIND_STORE:
    case LOOM_LOW_RESOURCE_KIND_CONTROL:
    case LOOM_LOW_RESOURCE_KIND_ADDRESS:
      return true;
    default:
      return false;
  }
}

static bool loom_low_hazard_kind_is_valid(loom_low_hazard_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_KIND_MIN_DISTANCE:
    case LOOM_LOW_HAZARD_KIND_WAIT_COUNTER:
    case LOOM_LOW_HAZARD_KIND_BYPASS:
    case LOOM_LOW_HAZARD_KIND_FUSION:
      return true;
    default:
      return false;
  }
}

static bool loom_low_hazard_reference_kind_is_valid(
    loom_low_hazard_reference_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE:
    case LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER:
    case LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET:
      return true;
    default:
      return false;
  }
}

static loom_low_schedule_class_flags_t
loom_low_schedule_flags_required_for_effect(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE;
    case LOOM_LOW_EFFECT_KIND_CALL:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL;
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL;
    default:
      return 0;
  }
}

static bool loom_low_effect_kind_requires_side_effecting_flag(
    loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_WRITE:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_descriptor_encoding_contract(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const bool is_pseudo =
      iree_all_bits_set(descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO);
  if (is_pseudo && descriptor->encoding_id != LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is pseudo but has target encoding id %" PRIu16,
                            descriptor_index, descriptor->encoding_id);
  }
  if (is_pseudo && descriptor->encoding_format_id != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32
        " is pseudo but has target encoding format id %" PRIu16,
        descriptor_index, descriptor->encoding_format_id);
  }
  if (!is_pseudo && descriptor->encoding_id == LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no target encoding id but is not pseudo",
                            descriptor_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->encoding_field_value_start,
      descriptor->encoding_field_value_count,
      descriptor_set->encoding_field_value_count, "encoding_field_values"));
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const uint32_t field_value_index =
        descriptor->encoding_field_value_start + i;
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set->encoding_field_values[field_value_index];
    if (field_value->encoding_field_id == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " fixed encoding field value %" PRIu32
                              " has field id zero",
                              descriptor_index, field_value_index);
    }
    for (uint16_t j = 0; j < i; ++j) {
      const loom_low_encoding_field_value_t* previous =
          &descriptor_set
               ->encoding_field_values[descriptor->encoding_field_value_start +
                                       j];
      if (previous->encoding_field_id == field_value->encoding_field_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32
                                " repeats fixed encoding field id %" PRIu16,
                                descriptor_index,
                                field_value->encoding_field_id);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_operand_encoding_fields(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t operand_index = 0; operand_index < descriptor->operand_count;
       ++operand_index) {
    const loom_low_operand_t* operand = &operands[operand_index];
    if (operand->encoding_field_id == 0) {
      continue;
    }
    for (uint16_t field_index = 0;
         field_index < descriptor->encoding_field_value_count; ++field_index) {
      const loom_low_encoding_field_value_t* field_value =
          &descriptor_set
               ->encoding_field_values[descriptor->encoding_field_value_start +
                                       field_index];
      if (field_value->encoding_field_id == operand->encoding_field_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand row %" PRIu16
            " shares fixed encoding field id %" PRIu16,
            descriptor_index, operand_index, operand->encoding_field_id);
      }
    }
    for (uint16_t previous_index = 0; previous_index < operand_index;
         ++previous_index) {
      const loom_low_operand_t* previous = &operands[previous_index];
      if (previous->encoding_field_id != operand->encoding_field_id) {
        continue;
      }
      if (loom_low_descriptor_operands_are_tied(
              descriptor_set, descriptor, previous_index, operand_index)) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " operand rows %" PRIu16 " and %" PRIu16
          " share encoding field id %" PRIu16 " without a tied constraint",
          descriptor_index, previous_index, operand_index,
          operand->encoding_field_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_effect_contract(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const bool is_side_effecting = iree_all_bits_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING);
  const bool is_terminator =
      iree_all_bits_set(descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  const bool is_dead_removable = iree_all_bits_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE);
  if (is_side_effecting && is_dead_removable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is both side-effecting and dead-removable",
                            descriptor_index);
  }
  if (is_terminator && !is_side_effecting) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is a terminator but is not side-effecting",
                            descriptor_index);
  }
  if (descriptor->effect_count == 0 && is_side_effecting) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is side-effecting but has no effect rows",
                            descriptor_index);
  }
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }

  bool has_control_effect = false;
  bool has_convergent_effect = false;
  loom_low_schedule_class_flags_t required_schedule_flags = 0;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const uint32_t effect_index = descriptor->effect_start + i;
    const loom_low_effect_t* effect = &descriptor_set->effects[effect_index];
    if (effect->kind == LOOM_LOW_EFFECT_KIND_READ ||
        effect->kind == LOOM_LOW_EFFECT_KIND_WRITE) {
      if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_NONE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32 " effect %" PRIu32
                                " reads or writes no memory space",
                                descriptor_index, effect_index);
      }
    }
    if (effect->kind == LOOM_LOW_EFFECT_KIND_CONTROL) {
      has_control_effect = true;
    }
    if (effect->kind == LOOM_LOW_EFFECT_KIND_CONVERGENT) {
      has_convergent_effect = true;
    }
    if (!is_side_effecting &&
        loom_low_effect_kind_requires_side_effecting_flag(effect->kind)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " effect %" PRIu32
                              " requires the side-effecting descriptor flag",
                              descriptor_index, effect_index);
    }
    required_schedule_flags |=
        loom_low_schedule_flags_required_for_effect(effect->kind);
  }

  if (is_terminator && !has_control_effect) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is a terminator but has no control effect",
                            descriptor_index);
  }
  if (is_dead_removable && has_convergent_effect) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is convergent but dead-removable",
                            descriptor_index);
  }
  if (has_control_effect && !is_terminator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has a control effect but is not a terminator",
                            descriptor_index);
  }
  if (descriptor->schedule_class_id != LOOM_LOW_SCHEDULE_CLASS_NONE &&
      required_schedule_flags != 0) {
    const loom_low_schedule_class_t* schedule_class =
        &descriptor_set->schedule_classes[descriptor->schedule_class_id];
    const loom_low_schedule_class_flags_t missing_flags =
        required_schedule_flags & ~schedule_class->flags;
    if (missing_flags != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " effect rows require schedule flags 0x%x but "
                              "schedule class %" PRIu16 " is missing 0x%x",
                              descriptor_index, required_schedule_flags,
                              descriptor->schedule_class_id, missing_flags);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor %" PRIu32 " canonical asm form ordinal %" PRIu32
        " exceeds asm form count %" PRIu32,
        descriptor_index, descriptor->canonical_asm_form_ordinal,
        descriptor_set->asm_form_count);
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_form->descriptor_ordinal != descriptor_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32 " canonical asm form ordinal %" PRIu32
        " belongs to descriptor %" PRIu32,
        descriptor_index, descriptor->canonical_asm_form_ordinal,
        asm_form->descriptor_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_operand_roles(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_index = descriptor->operand_start + i;
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    if (operand->role == LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " operand row %" PRIu16
          " uses OPERAND_RESULT, but SSA low packets require separate result "
          "and operand rows plus an explicit constraint",
          descriptor_index, i);
    }
    if (i < descriptor->result_count) {
      if (operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32 " result row %" PRIu16
                                " has non-result role %u",
                                descriptor_index, i, (unsigned)operand->role);
      }
      continue;
    }
    if (operand->role == LOOM_LOW_OPERAND_ROLE_RESULT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " operand row %" PRIu16
                              " appears after the result prefix but has "
                              "result role",
                              descriptor_index, i);
    }
    if (operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT &&
        !iree_all_bits_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " implicit operand row %" PRIu16
                              " must set LOOM_LOW_OPERAND_FLAG_IMPLICIT",
                              descriptor_index, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_state_operands(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_index = descriptor->operand_start + i;
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    const uint16_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (iree_any_bit_set(operand->flags,
                         LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE)) {
      if (!iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand row %" PRIu16
            " has schedule-only state without the implicit flag",
            descriptor_index, i);
      }
      if (!iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand row %" PRIu16
            " has schedule-only state without a state read flag",
            descriptor_index, i);
      }
      if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low descriptor %" PRIu32 " operand row %" PRIu16
            " has schedule-only state with a state write flag",
            descriptor_index, i);
      }
    }
    if (state_flags == 0) {
      continue;
    }
    if (operand->reg_class_alt_count != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " state operand row %" PRIu16
          " must name exactly one register-class alternative",
          descriptor_index, i);
    }
    const uint32_t alt_index = operand->reg_class_alt_start;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor %" PRIu32
                              " state operand row %" PRIu16
                              " has an out-of-range register-class "
                              "alternative",
                              descriptor_index, i);
    }
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) ||
        alt->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        alt->reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " state operand row %" PRIu16
                              " must name one concrete register class",
                              descriptor_index, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_storage_lease_attachment_unit_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index,
    const loom_low_descriptor_storage_lease_t* lease,
    uint16_t* out_unit_count) {
  *out_unit_count = 0;
  if (lease->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT) {
    if (lease->attachment_index >= descriptor->result_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low descriptor %" PRIu32 " storage lease references result %" PRIu16
          " but descriptor has only %" PRIu16 " results",
          descriptor_index, lease->attachment_index, descriptor->result_count);
    }
    const loom_low_operand_t* result =
        &descriptor_set
             ->operands[descriptor->operand_start + lease->attachment_index];
    *out_unit_count = result->unit_count;
    return iree_ok_status();
  }

  uint16_t packet_operand_index = 0;
  for (uint16_t operand_index = descriptor->result_count;
       operand_index < descriptor->operand_count; ++operand_index) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + operand_index];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    if (packet_operand_index == lease->attachment_index) {
      *out_unit_count = operand->unit_count;
      return iree_ok_status();
    }
    ++packet_operand_index;
  }
  return iree_make_status(
      IREE_STATUS_OUT_OF_RANGE,
      "low descriptor %" PRIu32
      " storage lease references packet operand %" PRIu16
      " but descriptor has only %" PRIu16 " packet operands",
      descriptor_index, lease->attachment_index, packet_operand_index);
}

static iree_status_t loom_low_verify_storage_lease(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index,
    uint16_t descriptor_lease_index) {
  const uint32_t lease_index =
      descriptor->storage_lease_start + descriptor_lease_index;
  const loom_low_descriptor_storage_lease_t* lease =
      &descriptor_set->storage_leases[lease_index];
  if (lease->kind == LOOM_LOW_STORAGE_LEASE_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage lease %" PRIu32 " has unknown kind",
                            lease_index);
  }
  if (!loom_low_storage_lease_kind_is_valid(lease->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage lease %" PRIu32 " has invalid kind %u",
                            lease_index, (unsigned)lease->kind);
  }
  if (lease->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32 " has unknown attachment", lease_index);
  }
  if (!loom_low_storage_lease_attachment_is_valid(lease->attachment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage lease %" PRIu32
                            " has invalid attachment %u",
                            lease_index, (unsigned)lease->attachment);
  }
  if (lease->release_scope == LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_UNKNOWN) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32 " has unknown release scope", lease_index);
  }
  if (!loom_low_storage_lease_release_scope_is_valid(lease->release_scope)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage lease %" PRIu32
                            " has invalid release scope %u",
                            lease_index, (unsigned)lease->release_scope);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      lease->flags,
      LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE |
          LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY |
          LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY,
      "storage lease", lease_index));
  if (iree_all_bits_set(
          lease->flags,
          LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY |
              LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32
        " both releases before and carries across a block boundary",
        lease_index);
  }
  if (lease->unit_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low storage lease %" PRIu32 " has zero unit count",
                            lease_index);
  }
  if (lease->release_class_id == UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32 " has no release class", lease_index);
  }
  if (lease->release_action_id == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32 " has no release action", lease_index);
  }
  if (lease->release_reason_id == UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low storage lease %" PRIu32 " has no release reason", lease_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, lease->release_class_name_string_offset,
      "storage_lease.release_class", NULL));
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, lease->release_action_name_string_offset,
      "storage_lease.release_action", NULL));
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, lease->release_reason_name_string_offset,
      "storage_lease.release_reason", NULL));

  uint16_t attachment_unit_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_verify_storage_lease_attachment_unit_count(
      descriptor_set, descriptor, descriptor_index, lease,
      &attachment_unit_count));
  if (lease->unit_offset > attachment_unit_count ||
      lease->unit_count > attachment_unit_count - lease->unit_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor %" PRIu32 " storage lease %" PRIu16
        " unit range [%" PRIu32 ", %" PRIu32
        ") exceeds attached value unit count %" PRIu16,
        descriptor_index, descriptor_lease_index, lease->unit_offset,
        lease->unit_offset + lease->unit_count, attachment_unit_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_index) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(descriptor->flags,
                                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR |
                                      LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE |
                                      LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO,
                                  "descriptor", descriptor_index));
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, descriptor->key_string_offset, "descriptor.key",
      &descriptor_key));
  if (descriptor->stable_id == LOOM_LOW_STABLE_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no stable descriptor ID",
                            descriptor_index);
  }
  const uint64_t expected_stable_id =
      loom_low_descriptor_stable_id_from_key(descriptor_key);
  if (descriptor->stable_id != expected_stable_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " key '%.*s' stable ID "
                            "0x%016" PRIx64 " should be 0x%016" PRIx64,
                            descriptor_index, (int)descriptor_key.size,
                            descriptor_key.data, descriptor->stable_id,
                            expected_stable_id);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor->mnemonic_string_offset,
      "descriptor.mnemonic"));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor->semantic_tag_string_offset,
      "descriptor.semantic_tag"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->feature_mask_word_start, descriptor->feature_mask_word_count,
      descriptor_set->feature_mask_word_count, "feature_mask_words"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(descriptor->operand_start, descriptor->operand_count,
                           descriptor_set->operand_count, "operands"));
  if (descriptor->result_count > descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32 " has result count %" PRIu16
        " greater than operand count %" PRIu16,
        descriptor_index, descriptor->result_count, descriptor->operand_count);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->immediate_start, descriptor->immediate_count,
      descriptor_set->immediate_count, "immediates"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(descriptor->effect_start, descriptor->effect_count,
                           descriptor_set->effect_count, "effects"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->constraint_start, descriptor->constraint_count,
      descriptor_set->constraint_count, "constraints"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->storage_lease_start, descriptor->storage_lease_count,
      descriptor_set->storage_lease_count, "storage_leases"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->operand_form_start, descriptor->operand_form_count,
      descriptor_set->operand_form_count, "operand_forms"));
  if (descriptor->schedule_class_id != LOOM_LOW_SCHEDULE_CLASS_NONE &&
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor %" PRIu32
                            " references schedule class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            descriptor_index, descriptor->schedule_class_id,
                            descriptor_set->schedule_class_count);
  }
  if (descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no schedule class; use an explicit fallback "
                            "schedule class for unknown scheduling",
                            descriptor_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_operand_roles(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_state_operands(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_encoding_contract(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_effect_contract(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_descriptor_constraints(descriptor_set, descriptor));
  for (uint16_t i = 0; i < descriptor->storage_lease_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_storage_lease(
        descriptor_set, descriptor, descriptor_index, i));
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_operand_forms(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_operand_encoding_fields(
      descriptor_set, descriptor, descriptor_index));
  return loom_low_verify_descriptor_canonical_asm_form(
      descriptor_set, descriptor, descriptor_index);
}

static iree_status_t loom_low_verify_operand(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t operand_index) {
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      operand->flags,
      LOOM_LOW_OPERAND_FLAG_IMPLICIT | LOOM_LOW_OPERAND_FLAG_TIED |
          LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER | LOOM_LOW_OPERAND_FLAG_OPTIONAL |
          LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE |
          LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE,
      "operand", operand_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, operand->field_name_string_offset, "operand.field_name"));
  if (operand->role == LOOM_LOW_OPERAND_ROLE_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32 " has unknown role",
                            operand_index);
  }
  if (!loom_low_operand_role_is_valid(operand->role)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32 " has invalid role %u",
                            operand_index, (unsigned)operand->role);
  }
  if (!loom_low_operand_address_map_kind_is_valid(operand->address_map_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32
                            " has invalid address map %u",
                            operand_index, (unsigned)operand->address_map_kind);
  }
  if (operand->address_map_kind == LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT &&
      operand->addressable_unit_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low operand %" PRIu32
        " has a direct address map with bounded addressable units",
        operand_index);
  }
  const bool has_addressable_assignment =
      operand->role == LOOM_LOW_OPERAND_ROLE_RESULT ||
      (loom_low_operand_role_is_packet_operand(operand->role) &&
       !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT));
  if (loom_low_operand_address_map_kind_has_low_window(
          operand->address_map_kind) &&
      !has_addressable_assignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low operand %" PRIu32
        " has a bounded address map without an explicit value operand",
        operand_index);
  }
  if (loom_low_operand_address_map_kind_has_low_window(
          operand->address_map_kind) &&
      operand->addressable_unit_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32
                            " has a bounded address map with no addressable "
                            "units",
                            operand_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      operand->reg_class_alt_start, operand->reg_class_alt_count,
      descriptor_set->reg_class_alt_count, "reg_class_alts"));
  if (loom_low_operand_address_map_kind_has_low_window(
          operand->address_map_kind)) {
    if (operand->addressable_unit_count < operand->unit_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low operand %" PRIu32
          " has a bounded address map that covers fewer units than the "
          "operand consumes",
          operand_index);
    }
    bool has_concrete_alt = false;
    for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
      const uint16_t alt_index = operand->reg_class_alt_start + i;
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (!iree_any_bit_set(alt->flags,
                            LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) &&
          alt->reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
        has_concrete_alt = true;
        break;
      }
    }
    if (!has_concrete_alt) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low operand %" PRIu32
          " has a bounded address map without a concrete register-class "
          "alternative",
          operand_index);
    }
  }
  if (operand->register_part_id != LOOM_LOW_REGISTER_PART_NONE) {
    if (operand->register_part_id >= descriptor_set->register_part_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low operand %" PRIu32
                              " references register part %" PRIu16
                              " but only %" PRIu32 " register parts exist",
                              operand_index, operand->register_part_id,
                              descriptor_set->register_part_count);
    }
    uint16_t concrete_alt_count = 0;
    uint16_t concrete_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
      const uint16_t alt_index = operand->reg_class_alt_start + i;
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (iree_all_bits_set(alt->flags,
                            LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        continue;
      }
      concrete_reg_class_id = alt->reg_class_id;
      ++concrete_alt_count;
    }
    const loom_low_register_part_t* register_part =
        &descriptor_set->register_parts[operand->register_part_id];
    if (concrete_alt_count != 1 ||
        concrete_reg_class_id != register_part->reg_class_id) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low operand %" PRIu32
                              " uses register part %" PRIu16
                              " but does not name exactly that register class",
                              operand_index, operand->register_part_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_immediate(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t immediate_index) {
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      immediate->flags,
      LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC | LOOM_LOW_IMMEDIATE_FLAG_RELATIVE |
          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE,
      "immediate", immediate_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, immediate->field_name_string_offset,
      "immediate.field_name"));
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32 " has unknown kind",
                            immediate_index);
  }
  if (!loom_low_immediate_kind_is_valid(immediate->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32 " has invalid kind %u",
                            immediate_index, (unsigned)immediate->kind);
  }
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ENUM) {
    if (immediate->enum_domain_id == LOOM_LOW_ENUM_DOMAIN_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low enum immediate %" PRIu32 " has no enum domain", immediate_index);
    }
    if (immediate->enum_domain_id >= descriptor_set->enum_domain_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low enum immediate %" PRIu32
                              " references enum domain %" PRIu16
                              " but only %" PRIu32 " domains exist",
                              immediate_index, immediate->enum_domain_id,
                              descriptor_set->enum_domain_count);
    }
  } else if (immediate->enum_domain_id != LOOM_LOW_ENUM_DOMAIN_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low non-enum immediate %" PRIu32
                            " references enum domain %" PRIu16,
                            immediate_index, immediate->enum_domain_id);
  }
  if (immediate->bit_width > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32
                            " has unsupported bit width %" PRIu16,
                            immediate_index, immediate->bit_width);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      immediate->encoding_slice_start, immediate->encoding_slice_count,
      descriptor_set->immediate_encoding_slice_count,
      "immediate_encoding_slices"));
  if (immediate->encoding_field_id != 0 &&
      immediate->encoding_slice_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32
                            " uses both direct and sliced encoding fields",
                            immediate_index);
  }
  if (immediate->encoding_slice_count != 0) {
    uint64_t covered_bits = 0;
    for (uint16_t i = 0; i < immediate->encoding_slice_count; ++i) {
      const uint32_t slice_index = immediate->encoding_slice_start + i;
      const loom_low_immediate_encoding_slice_t* slice =
          &descriptor_set->immediate_encoding_slices[slice_index];
      if (slice->encoding_field_id == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low immediate encoding slice %" PRIu32
                                " has encoding field id zero",
                                slice_index);
      }
      if (slice->bit_count == 0 || slice->bit_count > 64 ||
          slice->source_bit_offset > immediate->bit_width ||
          slice->bit_count > immediate->bit_width - slice->source_bit_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low immediate encoding slice %" PRIu32
            " source range [%u, %u) exceeds immediate %" PRIu32
            " bit width %" PRIu16,
            slice_index, slice->source_bit_offset,
            slice->source_bit_offset + slice->bit_count, immediate_index,
            immediate->bit_width);
      }
      const uint64_t slice_bits = loom_low_descriptor_bit_mask(slice->bit_count)
                                  << slice->source_bit_offset;
      if (iree_any_bit_set(covered_bits, slice_bits)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low immediate encoding slice %" PRIu32
            " overlaps another slice for immediate %" PRIu32,
            slice_index, immediate_index);
      }
      covered_bits |= slice_bits;
    }
    const uint64_t expected_bits =
        loom_low_descriptor_bit_mask(immediate->bit_width);
    if (covered_bits != expected_bits) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low immediate %" PRIu32
                              " encoding slices cover 0x%016" PRIx64
                              " instead of 0x%016" PRIx64,
                              immediate_index, covered_bits, expected_bits);
    }
  }
  if (!iree_any_bit_set(immediate->flags,
                        LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    if (immediate->default_value != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low immediate %" PRIu32
                              " has a default value without the "
                              "default-value flag",
                              immediate_index);
    }
    return iree_ok_status();
  }
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                  ? INT64_MAX
                                  : (int64_t)immediate->unsigned_max;
      if (immediate->default_value < immediate->signed_min ||
          immediate->default_value > maximum) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low signed immediate %" PRIu32
                                " default value %" PRId64 " is out of range",
                                immediate_index, immediate->default_value);
      }
      break;
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      if (immediate->default_value < 0 ||
          (uint64_t)immediate->default_value > immediate->unsigned_max) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low unsigned immediate %" PRIu32
                                " default value %" PRId64 " is out of range",
                                immediate_index, immediate->default_value);
      }
      break;
    case LOOM_LOW_IMMEDIATE_KIND_ENUM: {
      const loom_low_enum_domain_t* domain =
          &descriptor_set->enum_domains[immediate->enum_domain_id];
      IREE_RETURN_IF_ERROR(loom_low_verify_span(
          domain->value_start, domain->value_count,
          descriptor_set->enum_value_count, "enum_values"));
      bool found = false;
      for (uint16_t i = 0; i < domain->value_count; ++i) {
        const loom_low_enum_value_t* value =
            &descriptor_set->enum_values[domain->value_start + i];
        if (value->value == immediate->default_value) {
          found = true;
          break;
        }
      }
      if (!found) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low enum immediate %" PRIu32
                                " default value %" PRId64
                                " is not in its enum domain",
                                immediate_index, immediate->default_value);
      }
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_enum_domain(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t enum_domain_index) {
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[enum_domain_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, domain->name_string_offset, "enum_domain.name"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(domain->value_start, domain->value_count,
                           descriptor_set->enum_value_count, "enum_values"));
  if (domain->value_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low enum domain %" PRIu32 " has no values",
                            enum_domain_index);
  }
  iree_string_view_t previous_token = iree_string_view_empty();
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const uint32_t value_index = domain->value_start + i;
    const loom_low_enum_value_t* value =
        &descriptor_set->enum_values[value_index];
    iree_string_view_t token = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, value->token_string_offset, /*allow_none=*/false,
        &token));
    if (i != 0 && iree_string_view_compare(previous_token, token) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low enum domain %" PRIu32
                              " value tokens are not strictly sorted",
                              enum_domain_index);
    }
    previous_token = token;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_enum_value(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t enum_value_index) {
  const loom_low_enum_value_t* value =
      &descriptor_set->enum_values[enum_value_index];
  return loom_low_verify_required_string(
      descriptor_set, value->token_string_offset, "enum_value.token");
}

static iree_status_t loom_low_verify_reg_class(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t reg_class_index) {
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      reg_class->flags,
      LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY | LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE |
          LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE,
      "register class", reg_class_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, reg_class->name_string_offset, "reg_class.name"));
  if (reg_class->alloc_unit_bits == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low register class %" PRIu32
                            " has zero allocation-unit width",
                            reg_class_index);
  }
  if (reg_class->full_register_part_mask == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low register class %" PRIu32
                            " has an empty full register-part mask",
                            reg_class_index);
  }
  if (!loom_low_spill_slot_space_is_valid(
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register class %" PRIu32 " has unknown spill slot space %u",
        reg_class_index, (unsigned)reg_class->spill_slot_space);
  }
  const bool is_virtual_only =
      iree_all_bits_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY);
  const bool is_physical =
      iree_all_bits_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  if (is_virtual_only == is_physical) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register class %" PRIu32
        " must name exactly one virtual or physical storage kind",
        reg_class_index);
  }
  if (is_virtual_only && reg_class->allocatable_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low virtual register class %" PRIu32
                            " has non-zero allocatable count %" PRIu16,
                            reg_class_index, reg_class->allocatable_count);
  }
  if (is_physical && reg_class->allocatable_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low physical register class %" PRIu32
                            " has zero allocatable count",
                            reg_class_index);
  }
  if (reg_class->spill_class_id != LOOM_LOW_REG_CLASS_NONE &&
      reg_class->spill_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low register class %" PRIu32
                            " references spill class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            reg_class_index, reg_class->spill_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_register_part(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t register_part_index) {
  const loom_low_register_part_t* register_part =
      &descriptor_set->register_parts[register_part_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, register_part->name_string_offset, "register_part.name"));
  if (register_part->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low register part %" PRIu32
                            " references register class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            register_part_index, register_part->reg_class_id,
                            descriptor_set->reg_class_count);
  }
  if (register_part->mask == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low register part %" PRIu32 " has an empty mask",
                            register_part_index);
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[register_part->reg_class_id];
  if ((register_part->mask & ~reg_class->full_register_part_mask) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register part %" PRIu32 " mask 0x%08" PRIx32
        " exceeds register class %" PRIu16 " full mask 0x%08" PRIx32,
        register_part_index, register_part->mask, register_part->reg_class_id,
        reg_class->full_register_part_mask);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_reg_class_alt(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t alt_index) {
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(alt->flags,
                                  LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED |
                                      LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE |
                                      LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY,
                                  "register-class alternative", alt_index));
  const bool is_immediate =
      (alt->flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) != 0;
  if (is_immediate && alt->reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register-class alternative %" PRIu32
        " marks an immediate but also references register class %" PRIu16,
        alt_index, alt->reg_class_id);
  }
  if (!is_immediate && alt->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low register-class alternative %" PRIu32
                            " references register class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            alt_index, alt->reg_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t schedule_class_index) {
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[schedule_class_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(schedule_class->flags,
                                  LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL,
                                  "schedule class", schedule_class_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, schedule_class->name_string_offset,
      "schedule_class.name"));
  if (schedule_class->latency_kind == LOOM_LOW_LATENCY_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown latency kind",
                            schedule_class_index);
  }
  if (!loom_low_latency_kind_is_valid(schedule_class->latency_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low schedule class %" PRIu32 " has invalid latency kind %u",
        schedule_class_index, (unsigned)schedule_class->latency_kind);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown model quality",
                            schedule_class_index);
  }
  if (!loom_low_model_quality_is_valid(schedule_class->model_quality)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low schedule class %" PRIu32 " has invalid model quality %u",
        schedule_class_index, (unsigned)schedule_class->model_quality);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT &&
      schedule_class->latency_kind != LOOM_LOW_LATENCY_KIND_EXACT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has exact model quality without exact latency",
                            schedule_class_index);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_FALLBACK &&
      schedule_class->latency_kind != LOOM_LOW_LATENCY_KIND_VARIABLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " is a fallback model without variable latency",
                            schedule_class_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->issue_use_start, schedule_class->issue_use_count,
      descriptor_set->issue_use_count, "issue_uses"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->hazard_start, schedule_class->hazard_count,
      descriptor_set->hazard_count, "hazards"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->pressure_delta_start,
      schedule_class->pressure_delta_count,
      descriptor_set->pressure_delta_count, "pressure_deltas"));
  return iree_ok_status();
}

static iree_status_t loom_low_verify_issue_use(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t issue_use_index) {
  const loom_low_issue_use_t* issue_use =
      &descriptor_set->issue_uses[issue_use_index];
  if (issue_use->resource_id >= descriptor_set->resource_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low issue-use %" PRIu32
                            " references resource %" PRIu16 " but only %" PRIu32
                            " resources exist",
                            issue_use_index, issue_use->resource_id,
                            descriptor_set->resource_count);
  }
  if (issue_use->cycles == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low issue-use %" PRIu32 " has zero occupied cycles", issue_use_index);
  }
  if (issue_use->units == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low issue-use %" PRIu32
                            " consumes zero resource units",
                            issue_use_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_resource(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t resource_index) {
  const loom_low_resource_t* resource =
      &descriptor_set->resources[resource_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, resource->name_string_offset, "resource.name"));
  if (resource->kind == LOOM_LOW_RESOURCE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low resource %" PRIu32 " has unknown kind",
                            resource_index);
  }
  if (!loom_low_resource_kind_is_valid(resource->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low resource %" PRIu32 " has invalid kind %u",
                            resource_index, (unsigned)resource->kind);
  }
  if (resource->capacity_per_cycle == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low resource %" PRIu32 " has zero capacity per cycle", resource_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_effect(uint32_t effect_index,
                                            const loom_low_effect_t* effect) {
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      effect->flags,
      LOOM_LOW_EFFECT_FLAG_ORDERED | LOOM_LOW_EFFECT_FLAG_DEPENDENCY, "effect",
      effect_index));
  if (effect->kind == LOOM_LOW_EFFECT_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32 " has unknown kind",
                            effect_index);
  }
  if (!loom_low_effect_kind_is_valid(effect->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32 " has invalid kind %u",
                            effect_index, (unsigned)effect->kind);
  }
  if (!loom_low_memory_space_is_valid(effect->memory_space)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32
                            " has invalid memory space %u",
                            effect_index, (unsigned)effect->memory_space);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_constraint(
    uint32_t constraint_index, const loom_low_constraint_t* constraint) {
  if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low constraint %" PRIu32 " has unknown kind",
                            constraint_index);
  }
  if (!loom_low_constraint_kind_is_valid(constraint->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low constraint %" PRIu32 " has invalid kind %u",
                            constraint_index, (unsigned)constraint->kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_hazard(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t hazard_index) {
  const loom_low_hazard_t* hazard = &descriptor_set->hazards[hazard_index];
  if (hazard->kind == LOOM_LOW_HAZARD_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has unknown kind",
                            hazard_index);
  }
  if (!loom_low_hazard_kind_is_valid(hazard->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has invalid kind %u",
                            hazard_index, (unsigned)hazard->kind);
  }
  if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has unknown reference kind",
                            hazard_index);
  }
  if (!loom_low_hazard_reference_kind_is_valid(hazard->reference_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32
                            " has invalid reference kind %u",
                            hazard_index, (unsigned)hazard->reference_kind);
  }
  if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE &&
      hazard->reference_id >= descriptor_set->resource_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low hazard %" PRIu32 " references resource %" PRIu16
        " but only %" PRIu32 " resources exist",
        hazard_index, hazard->reference_id, descriptor_set->resource_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_pressure_delta(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t pressure_delta_index) {
  const loom_low_pressure_delta_t* pressure_delta =
      &descriptor_set->pressure_deltas[pressure_delta_index];
  if (pressure_delta->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low pressure delta %" PRIu32
                            " references register class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            pressure_delta_index, pressure_delta->reg_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_verify(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->abi_version != LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set ABI version %" PRIu32
                            " does not match supported version %u",
                            descriptor_set->abi_version,
                            LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_tables_present(descriptor_set));
  iree_string_view_t set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, descriptor_set->key_string_offset, "set.key", &set_key));
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
      descriptor_set, descriptor_set->target_key_string_offset,
      /*allow_none=*/true, &target_key));
  IREE_RETURN_IF_ERROR(loom_low_verify_stable_id_field(
      descriptor_set->stable_id, set_key, "stable_id"));
  IREE_RETURN_IF_ERROR(loom_low_verify_stable_id_field(
      descriptor_set->target_stable_id, target_key, "target_stable_id"));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor_set->feature_key_string_offset,
      "set.feature"));

  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_operand(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_immediate(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->enum_domain_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_enum_domain(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->enum_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_enum_value(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->effect_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_effect(i, &descriptor_set->effects[i]));
  }
  for (uint32_t i = 0; i < descriptor_set->constraint_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_constraint(i, &descriptor_set->constraints[i]));
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_reg_class(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->register_part_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_register_part(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_alt_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_reg_class_alt(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->schedule_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_schedule_class(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->issue_use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_issue_use(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->resource_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_resource(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->hazard_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_hazard(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->pressure_delta_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_pressure_delta(descriptor_set, i));
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_forms(descriptor_set));
  return loom_low_verify_descriptor_refs(descriptor_set);
}

static iree_status_t loom_low_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
      descriptor_set, descriptor_set->key_string_offset, /*allow_none=*/false,
      out_key));
  if (iree_string_view_is_empty(*out_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set key is empty");
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_registry_verify(
    const loom_low_descriptor_registry_t* registry) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (registry->descriptor_set_count != 0 &&
      registry->descriptor_sets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry entries are required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry providers are required");
  }
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(registry, i);
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify(descriptor_set));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_key(descriptor_set, &key));
    for (iree_host_size_t j = i + 1; j < descriptor_set_count; ++j) {
      const loom_low_descriptor_set_t* other_descriptor_set =
          loom_low_descriptor_registry_descriptor_set_at(registry, j);
      iree_string_view_t other_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_low_descriptor_set_key(other_descriptor_set, &other_key));
      if (iree_string_view_equal(key, other_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "low descriptor registry has duplicate descriptor set key '%.*s'",
            (int)key.size, key.data);
      }
      if (descriptor_set->stable_id == other_descriptor_set->stable_id) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "low descriptor registry has duplicate descriptor set stable ID "
            "0x%016" PRIx64,
            descriptor_set->stable_id);
      }
    }
  }
  return iree_ok_status();
}
