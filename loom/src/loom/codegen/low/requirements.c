// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/requirements.h"

#include <inttypes.h>

static iree_status_t loom_low_requirements_get_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name,
    iree_string_view_t* out_string) {
  *out_string = loom_low_descriptor_set_string(descriptor_set, string_offset);
  if (out_string->size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor %s is required by the selected "
                            "descriptor requirements",
                            field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_requirements_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t* out_key) {
  return loom_low_requirements_get_string(
      descriptor_set, descriptor->key_string_offset, "key", out_key);
}

static iree_status_t loom_low_requirements_verify_core_tables(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t set_key) {
  if (descriptor_set->descriptor_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor set '%.*s' has no descriptors",
                            (int)set_key.size, set_key.data);
  }
  if (descriptor_set->descriptor_ref_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low descriptor set '%.*s' has no descriptor key map",
        (int)set_key.size, set_key.data);
  }
  if (descriptor_set->reg_class_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor set '%.*s' has no register classes",
                            (int)set_key.size, set_key.data);
  }
  if (descriptor_set->reg_class_alt_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low descriptor set '%.*s' has no register-class alternatives",
        (int)set_key.size, set_key.data);
  }
  if (descriptor_set->schedule_class_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor set '%.*s' has no schedule classes",
                            (int)set_key.size, set_key.data);
  }
  if (descriptor_set->resource_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor set '%.*s' has no resources",
                            (int)set_key.size, set_key.data);
  }
  return iree_ok_status();
}

static bool loom_low_requirements_schedule_is_zero_cost(
    const loom_low_schedule_class_t* schedule_class) {
  return schedule_class->latency_cycles == 0 &&
         schedule_class->latency_kind == LOOM_LOW_LATENCY_KIND_EXACT &&
         schedule_class->flags == 0 &&
         schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT;
}

static iree_status_t loom_low_requirements_verify_operand_reg_classes(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t set_key,
    const loom_low_descriptor_t* descriptor,
    iree_string_view_t descriptor_key) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_index = descriptor->operand_start + i;
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    if (operand->reg_class_alt_count == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low descriptor set '%.*s' descriptor '%.*s' operand %" PRIu16
          " has no register-class alternatives",
          (int)set_key.size, set_key.data, (int)descriptor_key.size,
          descriptor_key.data, i);
    }
    bool has_register_or_immediate_alternative = false;
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[operand->reg_class_alt_start + j];
      if (iree_all_bits_set(alt->flags,
                            LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        has_register_or_immediate_alternative = true;
        continue;
      }
      if (alt->reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
        has_register_or_immediate_alternative = true;
      }
    }
    if (!has_register_or_immediate_alternative) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low descriptor set '%.*s' descriptor '%.*s' operand %" PRIu16
          " has only absent register-class alternatives",
          (int)set_key.size, set_key.data, (int)descriptor_key.size,
          descriptor_key.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_requirements_verify_descriptor_schedule(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t set_key,
    const loom_low_descriptor_t* descriptor,
    iree_string_view_t descriptor_key) {
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  const bool has_effect_or_control_payload =
      descriptor->effect_count != 0 ||
      iree_any_bit_set(descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                           LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  if (schedule_class->issue_use_count != 0) return iree_ok_status();
  if (!has_effect_or_control_payload &&
      loom_low_requirements_schedule_is_zero_cost(schedule_class)) {
    return iree_ok_status();
  }

  iree_string_view_t schedule_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_requirements_get_string(
      descriptor_set, schedule_class->name_string_offset, "schedule class name",
      &schedule_name));
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low descriptor set '%.*s' descriptor '%.*s' uses schedule class '%.*s' "
      "without issue-resource rows",
      (int)set_key.size, set_key.data, (int)descriptor_key.size,
      descriptor_key.data, (int)schedule_name.size, schedule_name.data);
}

static iree_status_t loom_low_requirements_verify_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t set_key,
    const loom_low_descriptor_t* descriptor,
    loom_low_descriptor_requirement_flags_t requirements) {
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_requirements_descriptor_key(
      descriptor_set, descriptor, &descriptor_key));

  if (iree_any_bit_set(requirements,
                       LOOM_LOW_DESCRIPTOR_REQUIREMENT_MNEMONICS)) {
    iree_string_view_t mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_requirements_get_string(
                             descriptor_set, descriptor->mnemonic_string_offset,
                             "mnemonic", &mnemonic),
                         "low descriptor set '%.*s' descriptor '%.*s'",
                         (int)set_key.size, set_key.data,
                         (int)descriptor_key.size, descriptor_key.data);
  }
  if (iree_any_bit_set(requirements,
                       LOOM_LOW_DESCRIPTOR_REQUIREMENT_SEMANTIC_TAGS)) {
    iree_string_view_t semantic_tag = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_requirements_get_string(descriptor_set,
                                         descriptor->semantic_tag_string_offset,
                                         "semantic tag", &semantic_tag),
        "low descriptor set '%.*s' descriptor '%.*s'", (int)set_key.size,
        set_key.data, (int)descriptor_key.size, descriptor_key.data);
  }
  if (iree_any_bit_set(requirements,
                       LOOM_LOW_DESCRIPTOR_REQUIREMENT_OPERAND_REG_CLASSES)) {
    IREE_RETURN_IF_ERROR(loom_low_requirements_verify_operand_reg_classes(
        descriptor_set, set_key, descriptor, descriptor_key));
  }
  if (iree_any_bit_set(requirements,
                       LOOM_LOW_DESCRIPTOR_REQUIREMENT_ISSUE_RESOURCES)) {
    IREE_RETURN_IF_ERROR(loom_low_requirements_verify_descriptor_schedule(
        descriptor_set, set_key, descriptor, descriptor_key));
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_verify_requirements(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_descriptor_requirement_flags_t requirements) {
  iree_string_view_t set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_requirements_get_string(
      descriptor_set, descriptor_set->key_string_offset, "set key", &set_key));
  if (iree_any_bit_set(requirements,
                       LOOM_LOW_DESCRIPTOR_REQUIREMENT_CORE_TABLES)) {
    IREE_RETURN_IF_ERROR(
        loom_low_requirements_verify_core_tables(descriptor_set, set_key));
  }
  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_requirements_verify_descriptor(
        descriptor_set, set_key, &descriptor_set->descriptors[i],
        requirements));
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_registry_verify_requirements(
    const loom_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements) {
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify_requirements(
        loom_low_descriptor_registry_descriptor_set_at(registry, i),
        requirements));
  }
  return iree_ok_status();
}
