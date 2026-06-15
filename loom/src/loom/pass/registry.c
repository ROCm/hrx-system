// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/registry.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/pass/pipeline.h"

iree_status_t loom_pass_registry_lookup(
    const loom_pass_registry_t* registry, iree_string_view_t key,
    const loom_pass_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  iree_host_size_t low = 0;
  iree_host_size_t high = registry->descriptor_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[mid];
    int comparison = iree_string_view_compare(key, descriptor->key);
    if (comparison == 0) {
      *out_descriptor = descriptor;
      return iree_ok_status();
    } else if (comparison < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_registry_storage_insert(
    loom_pass_registry_storage_t* storage,
    const loom_pass_descriptor_t* descriptor) {
  if (storage->registry.descriptor_count >=
      IREE_ARRAYSIZE(storage->descriptors)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pass registry storage capacity exceeded");
  }

  iree_host_size_t insert_index = 0;
  while (insert_index < storage->registry.descriptor_count) {
    const loom_pass_descriptor_t* existing =
        &storage->descriptors[insert_index];
    int comparison = iree_string_view_compare(descriptor->key, existing->key);
    if (comparison == 0) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "duplicate pass descriptor key '%.*s'",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    if (comparison < 0) {
      break;
    }
    ++insert_index;
  }

  const iree_host_size_t move_count =
      storage->registry.descriptor_count - insert_index;
  if (move_count != 0) {
    memmove(&storage->descriptors[insert_index + 1],
            &storage->descriptors[insert_index],
            move_count * sizeof(storage->descriptors[0]));
  }
  storage->descriptors[insert_index] = *descriptor;
  ++storage->registry.descriptor_count;
  return iree_ok_status();
}

iree_status_t loom_pass_registry_storage_initialize_from_registries(
    const loom_pass_registry_t* const* registries,
    iree_host_size_t registry_count,
    loom_pass_registry_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_pass_registry_storage_t){
      .registry =
          {
              .descriptors = out_storage->descriptors,
          },
  };

  for (iree_host_size_t registry_index = 0; registry_index < registry_count;
       ++registry_index) {
    const loom_pass_registry_t* registry = registries[registry_index];
    if (registry == NULL) {
      continue;
    }
    for (iree_host_size_t descriptor_index = 0;
         descriptor_index < registry->descriptor_count; ++descriptor_index) {
      IREE_RETURN_IF_ERROR(loom_pass_registry_storage_insert(
          out_storage, &registry->descriptors[descriptor_index]));
    }
  }
  return iree_ok_status();
}

const loom_pass_registry_t* loom_pass_registry_storage_registry(
    const loom_pass_registry_storage_t* storage) {
  return &storage->registry;
}

bool loom_pass_descriptor_is_available(
    const loom_pass_descriptor_t* descriptor) {
  return descriptor &&
         !iree_any_bit_set(descriptor->flags, LOOM_PASS_DESCRIPTOR_UNAVAILABLE);
}

typedef struct loom_pass_option_validation_context_t {
  // Descriptor whose option schema is being applied.
  const loom_pass_descriptor_t* descriptor;
  // Seen bit per descriptor option schema entry.
  bool* seen_options;
} loom_pass_option_validation_context_t;

static const loom_pass_option_schema_t* loom_pass_descriptor_find_option_schema(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t option_name,
    uint16_t* out_schema_index) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_string_view_equal(schema->name, option_name)) {
      *out_schema_index = i;
      return schema;
    }
  }
  return NULL;
}

static bool loom_pass_option_schema_find_enum_value(
    const loom_pass_option_schema_t* schema, iree_string_view_t option_value,
    uint16_t* out_enum_value_index) {
  for (uint16_t i = 0; i < schema->enum_value_count; ++i) {
    if (iree_string_view_equal(schema->enum_values[i].value, option_value)) {
      *out_enum_value_index = i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pass_option_schema_decode_uint32(
    const loom_pass_descriptor_t* descriptor,
    const loom_pass_option_schema_t* schema, iree_string_view_t option_value,
    uint32_t* out_value) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
      descriptor->key, schema->name, option_value, &value));
  if (value < schema->minimum_uint32 || value > schema->maximum_uint32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' must be in range %" PRIu32 "..%" PRIu32,
        (int)descriptor->key.size, descriptor->key.data, (int)schema->name.size,
        schema->name.data, schema->minimum_uint32, schema->maximum_uint32);
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_pass_option_schema_validate_assignment(
    void* user_data, iree_string_view_t option_name,
    iree_string_view_t option_value) {
  loom_pass_option_validation_context_t* context =
      (loom_pass_option_validation_context_t*)user_data;
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  uint16_t schema_index = 0;
  const loom_pass_option_schema_t* schema =
      loom_pass_descriptor_find_option_schema(descriptor, option_name,
                                              &schema_index);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (context->seen_options[schema_index]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  context->seen_options[schema_index] = true;

  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      return iree_ok_status();
    case LOOM_PASS_OPTION_SCHEMA_UINT32: {
      uint32_t uint32_value = 0;
      return loom_pass_option_schema_decode_uint32(descriptor, schema,
                                                   option_value, &uint32_value);
    }
    case LOOM_PASS_OPTION_SCHEMA_ENUM: {
      uint16_t enum_value_index = 0;
      if (loom_pass_option_schema_find_enum_value(schema, option_value,
                                                  &enum_value_index)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has invalid enum value '%.*s'",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)option_value.size,
          option_value.data);
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }
}

static iree_status_t loom_pass_option_schema_validate_required(
    const loom_pass_descriptor_t* descriptor, const bool* seen_options) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED) &&
        !seen_options[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing required option '%.*s' for pass '%.*s'",
                              (int)schema->name.size, schema->name.data,
                              (int)descriptor->key.size, descriptor->key.data);
    }
  }
  return iree_ok_status();
}

static bool loom_pass_option_schema_has_required(
    const loom_pass_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_pass_descriptor_validate_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options) {
  if (!descriptor || !descriptor->info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass descriptor with pass info is required for option validation");
  }
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  iree_string_view_t trimmed_options = iree_string_view_trim(options);
  bool has_options = !iree_string_view_is_empty(trimmed_options);
  if (has_options && !descriptor->create) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }
  if (descriptor->option_schema_count == 0) {
    if (!has_options) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }
  if (!descriptor->option_schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' has no option schema",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (!has_options && !loom_pass_option_schema_has_required(descriptor)) {
    return iree_ok_status();
  }

  bool* seen_options = NULL;
  iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_allocator_malloc(
      allocator, descriptor->option_schema_count * sizeof(*seen_options),
      (void**)&seen_options);
  if (iree_status_is_ok(status)) {
    memset(seen_options, 0,
           descriptor->option_schema_count * sizeof(*seen_options));
  }

  loom_pass_option_validation_context_t context = {
      .descriptor = descriptor,
      .seen_options = seen_options,
  };
  if (iree_status_is_ok(status) && has_options) {
    status = loom_pass_options_parse(
        info->name, trimmed_options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_pass_option_schema_validate_assignment,
            .user_data = &context,
        });
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_pass_option_schema_validate_required(descriptor, seen_options);
  }

  if (seen_options) {
    iree_allocator_free(allocator, seen_options);
  }
  return status;
}

typedef struct loom_pass_option_decode_context_t {
  // Descriptor whose option schema is being applied.
  const loom_pass_descriptor_t* descriptor;
  // Mutable decoded options indexed by descriptor option schema order.
  loom_pass_decoded_option_t* decoded_options;
} loom_pass_option_decode_context_t;

static iree_status_t loom_pass_option_schema_decode_assignment(
    void* user_data, iree_string_view_t option_name,
    iree_string_view_t option_value) {
  loom_pass_option_decode_context_t* context =
      (loom_pass_option_decode_context_t*)user_data;
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  uint16_t schema_index = 0;
  const loom_pass_option_schema_t* schema =
      loom_pass_descriptor_find_option_schema(descriptor, option_name,
                                              &schema_index);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }

  loom_pass_decoded_option_t* decoded = &context->decoded_options[schema_index];
  if (decoded->present) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  decoded->present = true;

  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      decoded->string_value = option_value;
      return iree_ok_status();
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      return loom_pass_option_schema_decode_uint32(
          descriptor, schema, option_value, &decoded->uint32_value);
    case LOOM_PASS_OPTION_SCHEMA_ENUM:
      if (loom_pass_option_schema_find_enum_value(schema, option_value,
                                                  &decoded->enum_value_index)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has invalid enum value '%.*s'",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)option_value.size,
          option_value.data);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }
}

static iree_status_t loom_pass_option_schema_validate_decoded_required(
    const loom_pass_descriptor_t* descriptor,
    const loom_pass_decoded_option_t* decoded_options) {
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    const loom_pass_option_schema_t* schema = &descriptor->option_schema[i];
    if (iree_all_bits_set(schema->flags,
                          (loom_pass_option_schema_flags_t)
                              LOOM_PASS_OPTION_SCHEMA_REQUIRED) &&
        !decoded_options[i].present) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing required option '%.*s' for pass '%.*s'",
                              (int)schema->name.size, schema->name.data,
                              (int)descriptor->key.size, descriptor->key.data);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pass_descriptor_decode_options(
    const loom_pass_descriptor_t* descriptor, iree_string_view_t options,
    iree_arena_allocator_t* arena, loom_pass_decoded_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  iree_string_view_t trimmed_options = iree_string_view_trim(options);
  bool has_options = !iree_string_view_is_empty(trimmed_options);
  if (has_options && descriptor->option_schema_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options, got '{%.*s}'",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)options.size, options.data);
  }

  loom_pass_decoded_option_t* decoded_options = NULL;
  iree_status_t status = iree_ok_status();
  if (descriptor->option_schema_count > 0) {
    status = iree_arena_allocate_array(arena, descriptor->option_schema_count,
                                       sizeof(*decoded_options),
                                       (void**)&decoded_options);
    if (iree_status_is_ok(status)) {
      memset(decoded_options, 0,
             descriptor->option_schema_count * sizeof(*decoded_options));
      for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
        decoded_options[i].schema = &descriptor->option_schema[i];
      }
    }
  }

  loom_pass_option_decode_context_t context = {
      .descriptor = descriptor,
      .decoded_options = decoded_options,
  };
  if (iree_status_is_ok(status) && has_options) {
    status = loom_pass_options_parse(
        info->name, trimmed_options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_pass_option_schema_decode_assignment,
            .user_data = &context,
        });
  }
  if (iree_status_is_ok(status) && decoded_options) {
    status = loom_pass_option_schema_validate_decoded_required(descriptor,
                                                               decoded_options);
  }
  if (iree_status_is_ok(status)) {
    *out_options = (loom_pass_decoded_options_t){
        .descriptor = descriptor,
        .options = decoded_options,
        .option_count = descriptor->option_schema_count,
    };
  }
  return status;
}

typedef struct loom_pass_attr_option_decode_context_t {
  // Module that owns option key and string value IDs.
  const loom_module_t* module;
  // Arena that owns decoded option storage.
  iree_arena_allocator_t* arena;
  // Descriptor whose option schema is being applied.
  const loom_pass_descriptor_t* descriptor;
  // Mutable decoded options indexed by descriptor option schema order.
  loom_pass_decoded_option_t* decoded_options;
} loom_pass_attr_option_decode_context_t;

static iree_status_t loom_pass_attr_option_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id, const char* label,
    iree_string_view_t* out_string) {
  if (!module || !out_string || string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %s string id", label);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_pass_attr_option_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_string) {
  if (source.size == 0) {
    *out_string = source;
    return iree_ok_status();
  }
  char* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target_data));
  memcpy(target_data, source.data, source.size);
  *out_string = iree_make_string_view(target_data, source.size);
  return iree_ok_status();
}

static iree_status_t loom_pass_attr_option_decode_uint32(
    const loom_pass_descriptor_t* descriptor,
    const loom_pass_option_schema_t* schema, const loom_attribute_t* value_attr,
    uint32_t* out_value) {
  if (value_attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' must be an integer attribute",
        (int)descriptor->key.size, descriptor->key.data, (int)schema->name.size,
        schema->name.data);
  }
  if (value_attr->i64 < 0 || value_attr->i64 > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' option '%.*s' value %" PRId64
                            " cannot be represented as uint32",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)schema->name.size, schema->name.data,
                            value_attr->i64);
  }
  uint32_t value = (uint32_t)value_attr->i64;
  if (value < schema->minimum_uint32 || value > schema->maximum_uint32) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' must be in range %" PRIu32 "..%" PRIu32,
        (int)descriptor->key.size, descriptor->key.data, (int)schema->name.size,
        schema->name.data, schema->minimum_uint32, schema->maximum_uint32);
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_pass_attr_option_decode_string(
    const loom_pass_attr_option_decode_context_t* context,
    const loom_pass_option_schema_t* schema, const loom_attribute_t* value_attr,
    iree_string_view_t* out_value) {
  if (value_attr->kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' option '%.*s' must be a string attribute",
        (int)context->descriptor->key.size, context->descriptor->key.data,
        (int)schema->name.size, schema->name.data);
  }
  iree_string_view_t source = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_attr_option_string_from_id(
      context->module, value_attr->string_id, "pass option value", &source));
  return loom_pass_attr_option_copy_string(context->arena, source, out_value);
}

static iree_status_t loom_pass_attr_option_decode_assignment(
    loom_pass_attr_option_decode_context_t* context,
    const loom_named_attr_t* option_attr) {
  const loom_pass_descriptor_t* descriptor = context->descriptor;
  iree_string_view_t option_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_attr_option_string_from_id(
      context->module, option_attr->name_id, "pass option name", &option_name));

  uint16_t schema_index = 0;
  const loom_pass_option_schema_t* schema =
      loom_pass_descriptor_find_option_schema(descriptor, option_name,
                                              &schema_index);
  if (!schema) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }

  loom_pass_decoded_option_t* decoded = &context->decoded_options[schema_index];
  if (decoded->present) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate option '%.*s' for pass '%.*s'",
                            (int)option_name.size, option_name.data,
                            (int)descriptor->key.size, descriptor->key.data);
  }
  decoded->present = true;

  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      return loom_pass_attr_option_decode_string(
          context, schema, &option_attr->value, &decoded->string_value);
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      return loom_pass_attr_option_decode_uint32(
          descriptor, schema, &option_attr->value, &decoded->uint32_value);
    case LOOM_PASS_OPTION_SCHEMA_ENUM: {
      iree_string_view_t enum_value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_pass_attr_option_decode_string(
          context, schema, &option_attr->value, &enum_value));
      if (loom_pass_option_schema_find_enum_value(schema, enum_value,
                                                  &decoded->enum_value_index)) {
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has invalid enum value '%.*s'",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)enum_value.size,
          enum_value.data);
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass '%.*s' option '%.*s' has unknown schema kind %d",
          (int)descriptor->key.size, descriptor->key.data,
          (int)schema->name.size, schema->name.data, (int)schema->kind);
  }
}

iree_status_t loom_pass_descriptor_decode_attr_options(
    const loom_pass_descriptor_t* descriptor, const loom_module_t* module,
    loom_named_attr_slice_t options, iree_arena_allocator_t* arena,
    loom_pass_decoded_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (options.count > 0 && descriptor->option_schema_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass '%.*s' does not accept options",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  loom_pass_decoded_option_t* decoded_options = NULL;
  iree_status_t status = iree_ok_status();
  if (descriptor->option_schema_count > 0) {
    status = iree_arena_allocate_array(arena, descriptor->option_schema_count,
                                       sizeof(*decoded_options),
                                       (void**)&decoded_options);
    if (iree_status_is_ok(status)) {
      memset(decoded_options, 0,
             descriptor->option_schema_count * sizeof(*decoded_options));
      for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
        decoded_options[i].schema = &descriptor->option_schema[i];
      }
    }
  }

  loom_pass_attr_option_decode_context_t context = {
      .module = module,
      .arena = arena,
      .descriptor = descriptor,
      .decoded_options = decoded_options,
  };
  for (iree_host_size_t i = 0; i < options.count && iree_status_is_ok(status);
       ++i) {
    status =
        loom_pass_attr_option_decode_assignment(&context, &options.entries[i]);
  }
  if (iree_status_is_ok(status) && decoded_options) {
    status = loom_pass_option_schema_validate_decoded_required(descriptor,
                                                               decoded_options);
  }
  if (iree_status_is_ok(status)) {
    *out_options = (loom_pass_decoded_options_t){
        .descriptor = descriptor,
        .options = decoded_options,
        .option_count = descriptor->option_schema_count,
    };
  }
  return status;
}
