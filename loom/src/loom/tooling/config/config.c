// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/config/config.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/attribute.h"
#include "loom/ir/encoding.h"
#include "loom/ops/config/contract.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/encoding/roles.h"
#include "loom/tooling/io/file.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

void loom_tooling_config_materialize_options_initialize(
    loom_tooling_config_materialize_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
}

void loom_tooling_config_set_initialize(
    iree_allocator_t host_allocator,
    loom_tooling_config_set_t* out_config_set) {
  IREE_ASSERT_ARGUMENT(out_config_set);
  memset(out_config_set, 0, sizeof(*out_config_set));
  out_config_set->host_allocator = host_allocator;
}

void loom_tooling_config_set_deinitialize(
    loom_tooling_config_set_t* config_set) {
  if (!config_set) {
    return;
  }
  for (iree_host_size_t i = 0; i < config_set->binding_count; ++i) {
    iree_allocator_free(config_set->host_allocator,
                        (void*)config_set->bindings[i].key.data);
    iree_allocator_free(config_set->host_allocator,
                        (void*)config_set->bindings[i].value.data);
  }
  iree_allocator_free(config_set->host_allocator, config_set->bindings);
  iree_allocator_t host_allocator = config_set->host_allocator;
  memset(config_set, 0, sizeof(*config_set));
  config_set->host_allocator = host_allocator;
}

static iree_string_view_t loom_tooling_config_normalize_key(
    iree_string_view_t key) {
  key = iree_string_view_trim(key);
  (void)iree_string_view_consume_prefix_char(&key, '@');
  return iree_string_view_trim(key);
}

iree_status_t loom_tooling_config_parse_assignment(
    iree_string_view_t assignment, loom_tooling_config_binding_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  iree_string_view_t key = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  if (iree_string_view_split(assignment, '=', &key, &value) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config assignment must use key=value syntax, got '%.*s'",
        (int)assignment.size, assignment.data);
  }
  key = loom_tooling_config_normalize_key(key);
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config assignment key must not be empty");
  }
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config assignment for '%.*s' must provide a non-empty value",
        (int)key.size, key.data);
  }
  *out_binding = (loom_tooling_config_binding_t){
      .key = key,
      .value = value,
  };
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_clone_string_view(
    iree_allocator_t allocator, iree_string_view_t value,
    iree_string_view_t* out_value) {
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_add(value.size, 1, &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "config string length overflow");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, byte_length, (void**)&data));
  memcpy(data, value.data, value.size);
  data[value.size] = '\0';
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_set_reserve(
    loom_tooling_config_set_t* config_set, iree_host_size_t minimum_capacity) {
  if (config_set->binding_capacity >= minimum_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      config_set->binding_capacity ? config_set->binding_capacity : 4;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "config binding capacity overflow");
    }
    new_capacity *= 2;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      config_set->host_allocator, new_capacity, sizeof(*config_set->bindings),
      (void**)&config_set->bindings));
  config_set->binding_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_set_check_duplicate(
    const loom_tooling_config_set_t* config_set, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < config_set->binding_count; ++i) {
    if (!iree_string_view_equal(config_set->bindings[i].key, key)) {
      continue;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate config binding for '%.*s'",
                            (int)key.size, key.data);
  }
  return iree_ok_status();
}

iree_status_t loom_tooling_config_set_append(
    loom_tooling_config_set_t* config_set, iree_string_view_t key,
    iree_string_view_t value) {
  IREE_ASSERT_ARGUMENT(config_set);
  key = loom_tooling_config_normalize_key(key);
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config binding key must not be empty");
  }
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config binding for '%.*s' must provide a non-empty value",
        (int)key.size, key.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_tooling_config_set_check_duplicate(config_set, key));
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_add(config_set->binding_count, 1,
                                  &minimum_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "config binding count overflow");
  }
  IREE_RETURN_IF_ERROR(
      loom_tooling_config_set_reserve(config_set, minimum_capacity));

  iree_string_view_t copied_key = iree_string_view_empty();
  iree_status_t status = loom_tooling_config_clone_string_view(
      config_set->host_allocator, key, &copied_key);
  iree_string_view_t copied_value = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_config_clone_string_view(config_set->host_allocator,
                                                   value, &copied_value);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(config_set->host_allocator, (void*)copied_key.data);
    return status;
  }

  config_set->bindings[config_set->binding_count++] =
      (loom_tooling_config_binding_t){
          .key = copied_key,
          .value = copied_value,
      };
  return iree_ok_status();
}

iree_status_t loom_tooling_config_set_append_assignment(
    loom_tooling_config_set_t* config_set, iree_string_view_t assignment) {
  IREE_ASSERT_ARGUMENT(config_set);
  loom_tooling_config_binding_t binding = {0};
  IREE_RETURN_IF_ERROR(
      loom_tooling_config_parse_assignment(assignment, &binding));
  return loom_tooling_config_set_append(config_set, binding.key, binding.value);
}

static iree_status_t loom_tooling_config_append_unescaped_json_string(
    iree_string_view_t escaped_string, iree_string_builder_t* builder,
    iree_host_size_t* out_length) {
  iree_host_size_t unescaped_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(escaped_string, 0, NULL, &unescaped_length));
  char* target = NULL;
  if (unescaped_length > 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_inline(builder, unescaped_length, &target));
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        escaped_string, unescaped_length, target, &unescaped_length));
  }
  *out_length = unescaped_length;
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_append_json_key(
    iree_string_view_t prefix, iree_string_view_t escaped_key,
    iree_string_builder_t* builder) {
  if (!iree_string_view_is_empty(prefix)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, prefix));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "."));
  }
  iree_host_size_t key_length = 0;
  IREE_RETURN_IF_ERROR(loom_tooling_config_append_unescaped_json_string(
      escaped_key, builder, &key_length));
  if (key_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config JSON object keys must not be empty");
  }
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_set_append_json_value(
    loom_tooling_config_set_t* config_set, iree_string_view_t key,
    iree_json_value_type_t value_type, iree_string_view_t value) {
  switch (value_type) {
    case IREE_JSON_VALUE_TYPE_STRING: {
      iree_string_builder_t value_builder;
      iree_string_builder_initialize(config_set->host_allocator,
                                     &value_builder);
      iree_host_size_t value_length = 0;
      iree_status_t status = loom_tooling_config_append_unescaped_json_string(
          value, &value_builder, &value_length);
      if (iree_status_is_ok(status)) {
        status = loom_tooling_config_set_append(
            config_set, key, iree_string_builder_view(&value_builder));
      }
      iree_string_builder_deinitialize(&value_builder);
      return status;
    }
    case IREE_JSON_VALUE_TYPE_NUMBER:
    case IREE_JSON_VALUE_TYPE_TRUE:
    case IREE_JSON_VALUE_TYPE_FALSE:
      return loom_tooling_config_set_append(config_set, key, value);
    case IREE_JSON_VALUE_TYPE_ARRAY:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config JSON binding '%.*s' uses an unsupported array value",
          (int)key.size, key.data);
    case IREE_JSON_VALUE_TYPE_NULL:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config JSON binding '%.*s' must not be null",
                              (int)key.size, key.data);
    case IREE_JSON_VALUE_TYPE_OBJECT:
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config JSON binding '%.*s' has unsupported value type",
          (int)key.size, key.data);
  }
}

typedef struct loom_tooling_config_json_object_state_t {
  loom_tooling_config_set_t* config_set;
  iree_string_view_t prefix;
  uint8_t depth;
} loom_tooling_config_json_object_state_t;

static iree_status_t loom_tooling_config_set_append_json_object_impl(
    loom_tooling_config_set_t* config_set, iree_string_view_t object,
    iree_string_view_t prefix, uint8_t depth);

static iree_status_t loom_tooling_config_set_append_json_member(
    void* user_data, iree_string_view_t escaped_key,
    iree_json_value_type_t value_type, iree_string_view_t value) {
  loom_tooling_config_json_object_state_t* state =
      (loom_tooling_config_json_object_state_t*)user_data;
  iree_string_builder_t key_builder;
  iree_string_builder_initialize(state->config_set->host_allocator,
                                 &key_builder);
  iree_status_t status = loom_tooling_config_append_json_key(
      state->prefix, escaped_key, &key_builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t key = iree_string_builder_view(&key_builder);
    if (value_type == IREE_JSON_VALUE_TYPE_OBJECT) {
      status = loom_tooling_config_set_append_json_object_impl(
          state->config_set, value, key, (uint8_t)(state->depth + 1));
    } else {
      status = loom_tooling_config_set_append_json_value(state->config_set, key,
                                                         value_type, value);
    }
  }
  iree_string_builder_deinitialize(&key_builder);
  return status;
}

static iree_status_t loom_tooling_config_set_append_json_object_impl(
    loom_tooling_config_set_t* config_set, iree_string_view_t object,
    iree_string_view_t prefix, uint8_t depth) {
  if (depth >= 128) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config JSON object nesting is too deep");
  }
  loom_tooling_config_json_object_state_t state = {
      .config_set = config_set,
      .prefix = prefix,
      .depth = depth,
  };
  return iree_json_enumerate_object_typed(
      object, loom_tooling_config_set_append_json_member, &state);
}

iree_status_t loom_tooling_config_set_append_json_object(
    loom_tooling_config_set_t* config_set, iree_string_view_t json_object) {
  IREE_ASSERT_ARGUMENT(config_set);
  iree_string_view_t cursor = json_object;
  iree_string_view_t object = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (!iree_string_view_is_empty(cursor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected trailing content after config JSON");
  }
  return loom_tooling_config_set_append_json_object_impl(
      config_set, object, iree_string_view_empty(), /*depth=*/0);
}

iree_status_t loom_tooling_config_set_append_json_file(
    loom_tooling_config_set_t* config_set, iree_string_view_t path,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(config_set);
  path = iree_string_view_trim(path);
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config JSON file requires a filesystem path, not stdin");
  }

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status =
      loom_tooling_read_input_file(path, host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_config_set_append_json_object(
        config_set, loom_tooling_file_contents_string_view(contents));
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(status, "config JSON file '%.*s'",
                                    (int)path.size, path.data);
  }
  iree_io_file_contents_free(contents);
  return status;
}

static iree_string_view_t loom_tooling_config_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!module || !symbol || symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return IREE_SV("<invalid>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_tooling_config_lookup_symbol(
    loom_module_t* module, iree_string_view_t key, uint16_t* out_symbol_id) {
  loom_string_id_t name_id = loom_module_lookup_string(module, key);
  if (name_id == LOOM_STRING_ID_INVALID) {
    *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
    return iree_ok_status();
  }
  *out_symbol_id = loom_module_find_symbol(module, name_id);
  return iree_ok_status();
}

static bool loom_tooling_config_symbol_is_config(const loom_symbol_t* symbol) {
  return symbol && symbol->defining_op &&
         loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG);
}

static iree_status_t loom_tooling_config_copy_attr(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_attribute_t source_attr, uint8_t depth, loom_attribute_t* out_attr);

static iree_status_t loom_tooling_config_copy_string_id(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_string_id_t source_string_id, loom_string_id_t* out_target_string_id) {
  if (source_string_id == LOOM_STRING_ID_INVALID) {
    *out_target_string_id = LOOM_STRING_ID_INVALID;
    return iree_ok_status();
  }
  if (source_string_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config value references invalid string id %u",
                            (unsigned)source_string_id);
  }
  return loom_module_intern_string(
      target_module, source_module->strings.entries[source_string_id],
      out_target_string_id);
}

static iree_status_t loom_tooling_config_copy_named_attrs(
    const loom_module_t* source_module, loom_module_t* target_module,
    const loom_named_attr_t* source_entries, iree_host_size_t count,
    uint8_t depth, loom_named_attr_t** out_target_entries) {
  if (count == 0) {
    *out_target_entries = NULL;
    return iree_ok_status();
  }
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config value attribute nesting exceeds max %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  loom_named_attr_t* target_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&target_module->arena, count,
                                                 sizeof(*target_entries),
                                                 (void**)&target_entries));
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_copy_string_id(
        source_module, target_module, source_entries[i].name_id,
        &target_entries[i].name_id));
    IREE_RETURN_IF_ERROR(loom_tooling_config_copy_attr(
        source_module, target_module, source_entries[i].value,
        (uint8_t)(depth + 1), &target_entries[i].value));
    target_entries[i].reserved = 0;
  }
  *out_target_entries = target_entries;
  return iree_ok_status();
}

static iree_status_t loom_tooling_config_copy_encoding(
    const loom_module_t* source_module, loom_module_t* target_module,
    uint16_t source_encoding_id, uint8_t depth,
    uint16_t* out_target_encoding_id) {
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config encoding nesting exceeds max %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  const loom_encoding_t* source_encoding =
      loom_module_encoding(source_module, source_encoding_id);
  if (!source_encoding) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config value references invalid encoding id %u",
                            (unsigned)source_encoding_id);
  }

  loom_encoding_t target_encoding = {0};
  IREE_RETURN_IF_ERROR(loom_tooling_config_copy_string_id(
      source_module, target_module, source_encoding->name_id,
      &target_encoding.name_id));
  IREE_RETURN_IF_ERROR(loom_tooling_config_copy_string_id(
      source_module, target_module, source_encoding->alias_id,
      &target_encoding.alias_id));
  target_encoding.attribute_count = source_encoding->attribute_count;
  loom_named_attr_t* target_entries = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_config_copy_named_attrs(
      source_module, target_module, source_encoding->attributes,
      source_encoding->attribute_count, (uint8_t)(depth + 1), &target_entries));
  target_encoding.attributes = target_entries;
  return loom_module_add_encoding(target_module, &target_encoding,
                                  out_target_encoding_id);
}

static iree_status_t loom_tooling_config_copy_attr(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_attribute_t source_attr, uint8_t depth, loom_attribute_t* out_attr) {
  *out_attr = source_attr;
  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
      return iree_ok_status();
    case LOOM_ATTR_STRING: {
      loom_string_id_t target_string_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_tooling_config_copy_string_id(
          source_module, target_module, source_attr.string_id,
          &target_string_id));
      *out_attr = loom_attr_string(target_string_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_I64_ARRAY: {
      if (source_attr.count == 0) {
        *out_attr = loom_attr_i64_array(NULL, 0);
        return iree_ok_status();
      }
      int64_t* target_values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &target_module->arena, source_attr.count, sizeof(*target_values),
          (void**)&target_values));
      memcpy(target_values, source_attr.i64_array,
             (iree_host_size_t)source_attr.count * sizeof(*target_values));
      *out_attr = loom_attr_i64_array(target_values, source_attr.count);
      return iree_ok_status();
    }
    case LOOM_ATTR_DICT: {
      if (source_attr.count == 0) {
        *out_attr = loom_make_canonical_attr_dict(NULL, 0);
        return iree_ok_status();
      }
      loom_named_attr_t* target_entries = NULL;
      IREE_RETURN_IF_ERROR(loom_tooling_config_copy_named_attrs(
          source_module, target_module, source_attr.dict_entries,
          source_attr.count, depth, &target_entries));
      return loom_module_make_canonical_attr_dict(
          target_module,
          loom_make_named_attr_slice(target_entries, source_attr.count),
          out_attr);
    }
    case LOOM_ATTR_ENCODING: {
      uint16_t target_encoding_id = 0;
      IREE_RETURN_IF_ERROR(loom_tooling_config_copy_encoding(
          source_module, target_module,
          (uint16_t)loom_attr_as_encoding_id(source_attr), (uint8_t)(depth + 1),
          &target_encoding_id));
      *out_attr = loom_attr_encoding(target_encoding_id);
      return iree_ok_status();
    }
    case LOOM_ATTR_SYMBOL:
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_PREDICATE_LIST:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config encoding value contains unsupported attribute kind %u",
          (unsigned)source_attr.kind);
    case LOOM_ATTR_ANY:
    case LOOM_ATTR_COUNT_:
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown config attribute kind %u",
                              (unsigned)source_attr.kind);
  }
}

static iree_status_t loom_tooling_config_parse_encoding_value(
    loom_module_t* module, loom_type_t type, iree_string_view_t value_text,
    iree_arena_block_pool_t* block_pool, loom_attribute_t* out_attr,
    iree_allocator_t host_allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  iree_status_t status = iree_string_builder_append_cstring(
      &builder, "config.def @__config_value = ");
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, value_text);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, " : ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_text_print_type(type, module, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }

  loom_module_t* parsed_module = NULL;
  if (iree_status_is_ok(status)) {
    loom_text_parse_options_t parse_options = {
        .max_errors = 20,
    };
    status = loom_text_parse(iree_string_builder_view(&builder),
                             IREE_SV("<config>"), module->context, block_pool,
                             &parse_options, &parsed_module);
  }
  if (iree_status_is_ok(status) && !parsed_module) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to parse config encoding value '%.*s'",
                              (int)value_text.size, value_text.data);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* op = loom_module_block(parsed_module)->first_op;
    if (!op || !loom_config_def_isa(op)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "failed to parse config encoding value");
    } else {
      status = loom_tooling_config_copy_attr(parsed_module, module,
                                             loom_config_def_value(op),
                                             /*depth=*/0, out_attr);
    }
  }

  if (parsed_module) {
    loom_module_free(parsed_module);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_tooling_config_check_encoding_role(
    const loom_module_t* module, iree_string_view_t key, loom_type_t type,
    loom_attribute_t value) {
  loom_encoding_role_t expected_role = loom_type_encoding_role(type);
  if (expected_role == LOOM_ENCODING_ROLE_UNKNOWN) {
    return iree_ok_status();
  }
  const loom_encoding_t* encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(value));
  loom_encoding_role_t actual_role =
      loom_encoding_static_role(module, encoding);
  if (actual_role == expected_role) {
    return iree_ok_status();
  }
  iree_string_view_t expected = loom_encoding_role_description(expected_role);
  iree_string_view_t actual = loom_encoding_role_description(actual_role);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' expects %.*s, got %.*s", (int)key.size,
                          key.data, (int)expected.size, expected.data,
                          (int)actual.size, actual.data);
}

static iree_status_t loom_tooling_config_parse_scalar_value(
    iree_string_view_t key, loom_type_t type, iree_string_view_t value_text,
    loom_attribute_t* out_attr) {
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    if (iree_string_view_equal_case(value_text, IREE_SV("true"))) {
      *out_attr = loom_attr_bool(true);
      return iree_ok_status();
    }
    if (iree_string_view_equal_case(value_text, IREE_SV("false"))) {
      *out_attr = loom_attr_bool(false);
      return iree_ok_status();
    }
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(value_text, &value) ||
        (value != 0 && value != 1)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config '%.*s' expects i1 value true/false/0/1, got '%.*s'",
          (int)key.size, key.data, (int)value_text.size, value_text.data);
    }
    *out_attr = loom_attr_bool(value != 0);
    return iree_ok_status();
  }

  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
      loom_scalar_type_is_integer(scalar_type)) {
    int64_t value = 0;
    if (!iree_string_view_atoi_int64(value_text, &value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config '%.*s' expects integer value, got '%.*s'",
                              (int)key.size, key.data, (int)value_text.size,
                              value_text.data);
    }
    int64_t domain_lo = INT64_MIN;
    int64_t domain_hi = INT64_MAX;
    if (loom_scalar_type_integer_domain(scalar_type, &domain_lo, &domain_hi) &&
        (value < domain_lo || value > domain_hi)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "config '%.*s' value %" PRId64 " is outside the %s domain [%" PRId64
          ", %" PRId64 "]",
          (int)key.size, key.data, value, loom_scalar_type_name(scalar_type),
          domain_lo, domain_hi);
    }
    *out_attr = loom_attr_i64(value);
    return iree_ok_status();
  }

  if (loom_scalar_type_is_float(scalar_type)) {
    double value = 0.0;
    if (!iree_string_view_atod(value_text, &value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config '%.*s' expects floating-point value, got '%.*s'",
          (int)key.size, key.data, (int)value_text.size, value_text.data);
    }
    *out_attr = loom_attr_f64(value);
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' has unsupported scalar type %s",
                          (int)key.size, key.data,
                          loom_scalar_type_name(scalar_type));
}

static iree_status_t loom_tooling_config_parse_value(
    loom_module_t* module, iree_string_view_t key, loom_type_t type,
    iree_string_view_t value_text, iree_arena_block_pool_t* block_pool,
    loom_attribute_t* out_attr) {
  if (loom_type_is_scalar(type)) {
    return loom_tooling_config_parse_scalar_value(key, type, value_text,
                                                  out_attr);
  }
  if (loom_type_is_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_parse_encoding_value(
        module, type, value_text, block_pool, out_attr,
        iree_allocator_system()));
    return loom_tooling_config_check_encoding_role(module, key, type,
                                                   *out_attr);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "config '%.*s' has unsupported type", (int)key.size,
                          key.data);
}

static iree_status_t loom_tooling_config_replace_with_def(
    loom_module_t* module, loom_op_t* old_op, loom_symbol_ref_t symbol,
    loom_attribute_t value, loom_type_t type) {
  loom_block_t* block = old_op->parent_block;
  loom_op_t* before_op = old_op->next_op;
  loom_op_t* parent_op = old_op->parent_op;
  loom_location_id_t location = old_op->location;
  loom_value_id_t old_result = loom_config_symbol_result_value(old_op);

  IREE_RETURN_IF_ERROR(loom_op_erase(module, old_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, block, &builder);
  builder.ip.parent_op = parent_op;
  builder.ip.before_op = before_op;
  loom_op_t* new_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_config_def_build(&builder, symbol, value, type, location, &new_op));
  if (old_result != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, old_result, loom_config_def_type(new_op)));
  }
  return iree_ok_status();
}

iree_status_t loom_tooling_config_materialize_module(
    loom_module_t* module,
    const loom_tooling_config_materialize_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(block_pool);
  const loom_tooling_config_set_t* config_set = options->config_set;
  const iree_host_size_t binding_count =
      config_set ? config_set->binding_count : 0;
  if (config_set && !config_set->bindings && binding_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config bindings pointer is NULL");
  }

  loom_tooling_config_materialize_result_t result = {0};
  if (binding_count == 0) {
    if (out_result) *out_result = result;
    return iree_ok_status();
  }

  const bool require_matches = iree_any_bit_set(
      options->flags, LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES);
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const loom_tooling_config_binding_t* binding = &config_set->bindings[i];
    iree_string_view_t key = binding->key;
    iree_string_view_t value_text = binding->value;
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_tooling_config_lookup_symbol(module, key, &symbol_id));
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      if (require_matches) {
        return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown config '%.*s'",
                                (int)key.size, key.data);
      }
      ++result.ignored_count;
      continue;
    }

    loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
    if (!loom_tooling_config_symbol_is_config(symbol)) {
      if (!require_matches && symbol->definition == NULL &&
          symbol->defining_op == NULL) {
        ++result.ignored_count;
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "symbol '%.*s' is not a config symbol",
                              (int)key.size, key.data);
    }
    loom_op_t* old_op = symbol->defining_op;
    loom_value_id_t config_value = loom_config_symbol_result_value(old_op);
    if (config_value == LOOM_VALUE_ID_INVALID ||
        config_value >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config '%.*s' has no result value",
                              (int)key.size, key.data);
    }
    loom_type_t config_type = loom_module_value_type(module, config_value);

    loom_attribute_t value = {0};
    IREE_RETURN_IF_ERROR(loom_tooling_config_parse_value(
        module, key, config_type, value_text, block_pool, &value));
    if (loom_config_decl_isa(old_op)) {
      IREE_RETURN_IF_ERROR(loom_config_check_value_constraints(
          loom_tooling_config_symbol_name(module, symbol), config_type,
          loom_config_decl_type(old_op), value,
          loom_config_decl_predicates(old_op)));
    }
    IREE_RETURN_IF_ERROR(loom_tooling_config_replace_with_def(
        module, old_op, (loom_symbol_ref_t){0, symbol_id}, value, config_type));
    ++result.materialized_count;
  }

  if (out_result) *out_result = result;
  return iree_ok_status();
}

iree_status_t loom_tooling_config_require_resolved_module(
    const loom_module_t* module,
    loom_tooling_config_resolution_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  loom_tooling_config_resolution_result_t result = {0};
  iree_string_view_t first_unresolved_name = iree_string_view_empty();

  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!symbol->defining_op || !loom_config_decl_isa(symbol->defining_op)) {
      continue;
    }
    if (result.unresolved_count == 0) {
      first_unresolved_name = loom_tooling_config_symbol_name(module, symbol);
    }
    ++result.unresolved_count;
  }

  if (out_result) *out_result = result;
  if (result.unresolved_count == 0) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "unresolved config '@%.*s' remains for final compilation (%" PRIhsz
      " unresolved config%s total)",
      (int)first_unresolved_name.size, first_unresolved_name.data,
      result.unresolved_count, result.unresolved_count == 1 ? "" : "s");
}

static iree_status_t loom_tooling_config_write_json_string_field(
    loom_output_stream_t* stream, const char* name, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"%s\":", name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_tooling_config_write_printed_type_string(
    const loom_module_t* module, loom_type_t type,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_text_print_type(type, module, &escape_stream));
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_tooling_config_write_printed_attr_string(
    const loom_module_t* module, loom_attribute_t attr,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(
      loom_text_print_attribute(&attr, module, &escape_stream));
  return loom_output_stream_write_char(stream, '"');
}

static iree_string_view_t loom_tooling_config_predicate_arg_kind_name(
    uint8_t tag) {
  switch ((loom_predicate_arg_tag_t)tag) {
    case LOOM_PRED_ARG_NONE:
      return IREE_SV("none");
    case LOOM_PRED_ARG_VALUE:
      return IREE_SV("value");
    case LOOM_PRED_ARG_CONST:
      return IREE_SV("const");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_tooling_config_format_predicate_arg_json(
    const loom_predicate_t* predicate, uint8_t arg_index,
    loom_output_stream_t* stream) {
  uint8_t tag = LOOM_PRED_ARG_NONE;
  int64_t value = 0;
  if (arg_index < predicate->arg_count) {
    tag = predicate->arg_tags[arg_index];
    value = predicate->args[arg_index];
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_tooling_config_write_json_string_field(
      stream, "kind", loom_tooling_config_predicate_arg_kind_name(tag)));
  switch ((loom_predicate_arg_tag_t)tag) {
    case LOOM_PRED_ARG_VALUE: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"value_id\":%" PRId64, value));
      break;
    }
    case LOOM_PRED_ARG_CONST: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"value\":%" PRId64, value));
      break;
    }
    default:
      break;
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_tooling_config_format_predicate_json(
    const loom_predicate_t* predicate, loom_output_stream_t* stream) {
  const char* kind_name = loom_predicate_kind_name(predicate->kind);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_tooling_config_write_json_string_field(
      stream, "kind",
      kind_name ? iree_make_cstring_view(kind_name) : IREE_SV("unknown")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"args\":["));
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
    IREE_RETURN_IF_ERROR(
        loom_tooling_config_format_predicate_arg_json(predicate, i, stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_tooling_config_format_predicates_json(
    loom_attribute_t predicates, loom_output_stream_t* stream) {
  if (predicates.kind != LOOM_ATTR_ABSENT &&
      predicates.kind != LOOM_ATTR_PREDICATE_LIST) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config constraints must be a predicate list");
  }
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST && predicates.count > 0 &&
      !predicates.predicate_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config constraints predicate list is missing");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  if (predicates.kind == LOOM_ATTR_PREDICATE_LIST) {
    for (uint16_t i = 0; i < predicates.count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
      IREE_RETURN_IF_ERROR(loom_tooling_config_format_predicate_json(
          &predicates.predicate_list[i], stream));
    }
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_tooling_config_format_schema_entry_json(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_output_stream_t* stream) {
  loom_op_t* op = symbol->defining_op;
  const bool is_decl = loom_config_decl_isa(op);
  const bool is_def = loom_config_def_isa(op);
  if (!is_decl && !is_def) {
    iree_string_view_t symbol_name =
        loom_tooling_config_symbol_name(module, symbol);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config symbol '@%.*s' is not a config.decl/def",
                            (int)symbol_name.size, symbol_name.data);
  }

  loom_value_id_t config_value = loom_config_symbol_result_value(op);
  if (config_value == LOOM_VALUE_ID_INVALID ||
      config_value >= module->values.count) {
    iree_string_view_t symbol_name =
        loom_tooling_config_symbol_name(module, symbol);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config symbol '@%.*s' has no result value",
                            (int)symbol_name.size, symbol_name.data);
  }
  loom_type_t config_type = loom_module_value_type(module, config_value);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_tooling_config_write_json_string_field(
      stream, "name", loom_tooling_config_symbol_name(module, symbol)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_tooling_config_write_json_string_field(
      stream, "state", is_decl ? IREE_SV("decl") : IREE_SV("def")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"required\":%s,\"type\":", is_decl ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_tooling_config_write_printed_type_string(
      module, config_type, stream));
  if (is_def) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"default\":"));
    IREE_RETURN_IF_ERROR(loom_tooling_config_write_printed_attr_string(
        module, loom_config_def_value(op), stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"constraints\":"));
  IREE_RETURN_IF_ERROR(loom_tooling_config_format_predicates_json(
      loom_config_decl_predicates(op), stream));
  return loom_output_stream_write_cstring(stream, "}");
}

iree_status_t loom_tooling_config_format_schema_json(
    const loom_module_t* module, loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\"configs\":["));
  iree_host_size_t config_count = 0;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!loom_tooling_config_symbol_is_config(symbol)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, config_count == 0 ? "" : ","));
    IREE_RETURN_IF_ERROR(
        loom_tooling_config_format_schema_entry_json(module, symbol, stream));
    ++config_count;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "],\"count\":%" PRIhsz "}", config_count));
  return iree_ok_status();
}
