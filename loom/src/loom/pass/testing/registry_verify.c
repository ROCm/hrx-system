// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/testing/registry_verify.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "loom/pass/environment.h"

typedef struct loom_pass_statistic_int64_alignment_t {
  // Leading byte used to compute int64_t aggregate alignment.
  char prefix;
  // Aligned int64_t member.
  int64_t value;
} loom_pass_statistic_int64_alignment_t;

static bool loom_pass_descriptor_key_matches_info(
    const loom_pass_descriptor_t* descriptor) {
  if (!descriptor->info) return false;
  const loom_pass_info_t* info = descriptor->info();
  return info && iree_string_view_equal(descriptor->key, info->name);
}

static bool loom_pass_info_has_option_def(const loom_pass_info_t* info,
                                          iree_string_view_t option_name) {
  for (uint16_t i = 0; i < info->option_count; ++i) {
    if (iree_string_view_equal(info->option_defs[i].name, option_name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_registry_verify_option_schema(
    const loom_pass_descriptor_t* descriptor, const loom_pass_info_t* info) {
  if (descriptor->option_schema_count != info->option_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' has %u option schemas but pass info declares "
        "%u options",
        (int)descriptor->key.size, descriptor->key.data,
        (unsigned)descriptor->option_schema_count,
        (unsigned)info->option_count);
  }
  if (info->option_count > 0 && !info->option_defs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' pass info declares options without option defs",
        (int)descriptor->key.size, descriptor->key.data);
  }
  if (descriptor->option_schema_count > 0 && !descriptor->option_schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' has no option schema",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (descriptor->option_schema_count > 0 && !descriptor->create) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' declares options but has no create callback",
        (int)descriptor->key.size, descriptor->key.data);
  }
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_string_view_is_empty(schema->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema %u has no name",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (!loom_pass_info_has_option_def(info, schema->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema '%.*s' has no matching pass "
          "info option",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data);
    }
    if (iree_any_bit_set(schema->flags, ~(loom_pass_option_schema_flags_t)
                                            LOOM_PASS_OPTION_SCHEMA_REQUIRED)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' option schema '%.*s' has unsupported flags "
          "0x%" PRIx32,
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, schema->flags);
    }
    if (i > 0) {
      const loom_pass_option_schema_t* previous =
          &descriptor->option_schema[i - 1];
      int comparison = iree_string_view_compare(previous->name, schema->name);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' option schemas must be strictly sorted; "
            "'%.*s' precedes '%.*s'",
            (int)descriptor->key.size, descriptor->key.data,
            (int)previous->name.size, previous->name.data,
            (int)schema->name.size, schema->name.data);
      }
    }
    switch (schema->kind) {
      case LOOM_PASS_OPTION_SCHEMA_STRING:
        if (schema->enum_values || schema->enum_value_count != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' string option schema '%.*s' has enum "
              "values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        break;
      case LOOM_PASS_OPTION_SCHEMA_UINT32:
        if (schema->minimum_uint32 > schema->maximum_uint32) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' uint32 option schema '%.*s' has invalid "
              "range",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        if (schema->enum_values || schema->enum_value_count != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' uint32 option schema '%.*s' has enum "
              "values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        break;
      case LOOM_PASS_OPTION_SCHEMA_ENUM:
        if (schema->enum_value_count == 0 || !schema->enum_values) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "pass descriptor '%.*s' enum option schema '%.*s' has no values",
              (int)descriptor->key.size, descriptor->key.data,
              (int)schema->name.size, schema->name.data);
        }
        for (uint16_t j = 0; j < schema->enum_value_count; ++j) {
          const loom_pass_option_enum_value_t* value = &schema->enum_values[j];
          if (iree_string_view_is_empty(value->value)) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "pass descriptor '%.*s' enum option schema '%.*s' value "
                "%u is empty",
                (int)descriptor->key.size, descriptor->key.data,
                (int)schema->name.size, schema->name.data, (unsigned)j);
          }
          if (j > 0) {
            const loom_pass_option_enum_value_t* previous =
                &schema->enum_values[j - 1];
            int comparison =
                iree_string_view_compare(previous->value, value->value);
            if (comparison >= 0) {
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "pass descriptor '%.*s' enum option schema '%.*s' values "
                  "must be strictly sorted; '%.*s' precedes '%.*s'",
                  (int)descriptor->key.size, descriptor->key.data,
                  (int)schema->name.size, schema->name.data,
                  (int)previous->value.size, previous->value.data,
                  (int)value->value.size, value->value.data);
            }
          }
        }
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' option schema '%.*s' has unknown kind %d",
            (int)descriptor->key.size, descriptor->key.data,
            (int)schema->name.size, schema->name.data, (int)schema->kind);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_registry_verify_requirements(
    const loom_pass_descriptor_t* descriptor) {
  if (descriptor->requirement_count > 0 && !descriptor->requirement_defs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' has no requirement defs",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  for (uint16_t i = 0; i < descriptor->requirement_count; ++i) {
    const loom_pass_requirement_def_t* requirement =
        &descriptor->requirement_defs[i];
    if (!requirement->capability_type ||
        iree_string_view_is_empty(requirement->capability_type->name) ||
        !requirement->capability_type->satisfies_requirement) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' requirement %u has no satisfying "
          "capability type",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (iree_string_view_is_empty(requirement->key)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' requirement %u has no key",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (iree_string_view_is_empty(requirement->description)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' requirement '%.*s' has no description",
          (int)descriptor->key.size, descriptor->key.data,
          (int)requirement->key.size, requirement->key.data);
    }
    if (i > 0) {
      const loom_pass_requirement_def_t* previous =
          &descriptor->requirement_defs[i - 1];
      int comparison =
          iree_string_view_compare(previous->key, requirement->key);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' requirements must be strictly sorted; "
            "'%.*s' precedes '%.*s'",
            (int)descriptor->key.size, descriptor->key.data,
            (int)previous->key.size, previous->key.data,
            (int)requirement->key.size, requirement->key.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_registry_verify_statistics(
    const loom_pass_descriptor_t* descriptor, const loom_pass_info_t* info) {
  const loom_pass_statistic_layout_t* layout = info->statistic_layout;
  if (!layout) {
    return iree_ok_status();
  }
  if (layout->field_count > LOOM_PASS_STATISTIC_FIELD_COUNT_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' statistic layout has %u fields but the maximum "
        "is %u",
        (int)descriptor->key.size, descriptor->key.data,
        (unsigned)layout->field_count,
        (unsigned)LOOM_PASS_STATISTIC_FIELD_COUNT_MAX);
  }
  if (layout->field_count == 0) {
    if (layout->fields) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' statistic layout has fields without a count",
          (int)descriptor->key.size, descriptor->key.data);
    }
    return iree_ok_status();
  }
  if (!layout->fields) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' declares statistic fields without metadata",
        (int)descriptor->key.size, descriptor->key.data);
  }
  if (layout->storage_size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor '%.*s' declares statistic fields without storage",
        (int)descriptor->key.size, descriptor->key.data);
  }
  for (uint16_t i = 0; i < layout->field_count; ++i) {
    const loom_pass_statistic_field_t* field = &layout->fields[i];
    if (iree_string_view_is_empty(field->name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' statistic field %u has no name",
          (int)descriptor->key.size, descriptor->key.data, (unsigned)i);
    }
    if (iree_string_view_is_empty(field->description)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' statistic field '%.*s' has no description",
          (int)descriptor->key.size, descriptor->key.data,
          (int)field->name.size, field->name.data);
    }
    if (field->offset > layout->storage_size ||
        layout->storage_size - field->offset < sizeof(int64_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' statistic field '%.*s' offset %" PRIhsz
          " is outside storage size %" PRIhsz,
          (int)descriptor->key.size, descriptor->key.data,
          (int)field->name.size, field->name.data, field->offset,
          layout->storage_size);
    }
    const iree_host_size_t int64_alignment =
        offsetof(loom_pass_statistic_int64_alignment_t, value);
    if (field->offset % int64_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' statistic field '%.*s' offset %" PRIhsz
          " is not int64 aligned",
          (int)descriptor->key.size, descriptor->key.data,
          (int)field->name.size, field->name.data, field->offset);
    }
    for (uint16_t j = 0; j < i; ++j) {
      const loom_pass_statistic_field_t* previous = &layout->fields[j];
      if (iree_string_view_equal(previous->name, field->name)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' statistic field '%.*s' is duplicated",
            (int)descriptor->key.size, descriptor->key.data,
            (int)field->name.size, field->name.data);
      }
      if (previous->offset == field->offset) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass descriptor '%.*s' statistic field '%.*s' reuses offset "
            "%" PRIhsz,
            (int)descriptor->key.size, descriptor->key.data,
            (int)field->name.size, field->name.data, field->offset);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pass_registry_verify(const loom_pass_registry_t* registry) {
  if (!registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass registry is required");
  }
  if (registry->descriptor_count > 0 && !registry->descriptors) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass registry descriptors are required");
  }
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[i];
    if (iree_string_view_is_empty(descriptor->key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor %zu has no key", i);
    }
    if (!descriptor->info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' has no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    const loom_pass_info_t* info = descriptor->info();
    if (!info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' returned no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    if (!loom_pass_descriptor_key_matches_info(descriptor)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor key '%.*s' does not match pass info name '%.*s'",
          (int)descriptor->key.size, descriptor->key.data, (int)info->name.size,
          info->name.data);
    }
    IREE_RETURN_IF_ERROR(
        loom_pass_registry_verify_option_schema(descriptor, info));
    IREE_RETURN_IF_ERROR(
        loom_pass_registry_verify_statistics(descriptor, info));
    IREE_RETURN_IF_ERROR(loom_pass_registry_verify_requirements(descriptor));
    if (iree_any_bit_set(
            descriptor->flags,
            ~(loom_pass_descriptor_flags_t)LOOM_PASS_DESCRIPTOR_UNAVAILABLE)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' has unsupported flags 0x%" PRIx32,
          (int)descriptor->key.size, descriptor->key.data, descriptor->flags);
    }
    if (iree_any_bit_set(descriptor->flags, LOOM_PASS_DESCRIPTOR_UNAVAILABLE) &&
        iree_string_view_is_empty(descriptor->unavailable_reason)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass descriptor '%.*s' is unavailable without a reason",
          (int)descriptor->key.size, descriptor->key.data);
    }
    switch (info->kind) {
      case LOOM_PASS_MODULE:
        if (!descriptor->module_run) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module pass descriptor '%.*s' has no run callback",
              (int)descriptor->key.size, descriptor->key.data);
        }
        break;
      case LOOM_PASS_FUNCTION:
        if (!descriptor->function_run) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "function pass descriptor '%.*s' has no run callback",
              (int)descriptor->key.size, descriptor->key.data);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pass descriptor '%.*s' has unknown kind %d",
                                (int)descriptor->key.size, descriptor->key.data,
                                (int)info->kind);
    }
    if (i > 0) {
      const loom_pass_descriptor_t* previous = &registry->descriptors[i - 1];
      int comparison = iree_string_view_compare(previous->key, descriptor->key);
      if (comparison >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pass registry keys must be strictly sorted; '%.*s' precedes "
            "'%.*s'",
            (int)previous->key.size, previous->key.data,
            (int)descriptor->key.size, descriptor->key.data);
      }
    }
  }
  return iree_ok_status();
}
