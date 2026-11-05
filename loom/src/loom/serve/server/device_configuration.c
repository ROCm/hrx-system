// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/device_configuration.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

static loom_serve_pool_options_t loom_serve_device_configuration_defaults(
    void) {
  return (loom_serve_pool_options_t){
      .elastic = true,
      .weight_slab_size = UINT64_C(1) << 30,
      .state_slab_size = UINT64_C(64) << 20,
  };
}

typedef struct loom_serve_device_configuration_parser_t {
  // Output under construction; deinitialization owns partial string storage.
  loom_serve_device_configuration_t* configuration;
  // Next unused byte of input-sized decoded string storage.
  char* next_string;
} loom_serve_device_configuration_parser_t;

static iree_status_t loom_serve_device_configuration_string(
    loom_serve_device_configuration_parser_t* parser,
    iree_string_view_t escaped, iree_string_view_t* out_value) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(escaped, escaped.size,
                                                 parser->next_string, &length));
  const iree_string_view_t value =
      iree_make_string_view(parser->next_string, length);
  if (!length || memchr(value.data, 0, length) ||
      !iree_unicode_utf8_validate(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "configuration strings must be nonempty UTF-8 without NUL");
  }
  parser->next_string += length;
  *out_value = value;
  return iree_ok_status();
}

typedef struct loom_serve_device_configuration_fields_t {
  // Decoder owning storage for string values in one device or pool entry.
  loom_serve_device_configuration_parser_t* parser;
  // Decoded strings or raw JSON values; null data marks omission.
  iree_string_view_t values[8];
} loom_serve_device_configuration_fields_t;

static const char* const loom_serve_device_configuration_field_names[] = {
    "devices",           "pool_backing", "weight_slab_bytes",
    "state_slab_bytes",  "memory_bytes", "workspace_idle_ms",
    "parameter_idle_ms", "placement"};

static iree_status_t loom_serve_device_configuration_field(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_fields_t* fields = self;
  iree_host_size_t i = 0;
  while (i < IREE_ARRAYSIZE(loom_serve_device_configuration_field_names) &&
         !iree_string_view_equal(
             key, iree_make_cstring_view(
                      loom_serve_device_configuration_field_names[i]))) {
    ++i;
  }
  if (i == IREE_ARRAYSIZE(loom_serve_device_configuration_field_names)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown pool field '%.*s'", (int)key.size,
                            key.data);
  }
  const bool type_allowed =
      (type == IREE_JSON_VALUE_TYPE_ARRAY && i == 0) ||
      (type == IREE_JSON_VALUE_TYPE_STRING && i > 0 && i < 5) ||
      (type == IREE_JSON_VALUE_TYPE_NUMBER && i >= 2 && i < 7) ||
      (type == IREE_JSON_VALUE_TYPE_OBJECT && i == 7);
  if (fields->values[i].data || !type_allowed) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate or wrongly typed pool field '%s'",
                            loom_serve_device_configuration_field_names[i]);
  }
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_string(
                             fields->parser, value, &fields->values[i]),
                         "pool field '%s'",
                         loom_serve_device_configuration_field_names[i]);
  } else {
    fields->values[i] = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_uri(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_fields_t* fields = self;
  if (!iree_string_view_equal(key, IREE_SV("uri")) ||
      type != IREE_JSON_VALUE_TYPE_STRING || fields->values[0].data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device entries require exactly one 'uri' string");
  }
  return loom_serve_device_configuration_string(fields->parser, value,
                                                &fields->values[0]);
}

static iree_status_t loom_serve_device_configuration_name(
    loom_serve_device_configuration_parser_t* parser, iree_string_view_t key,
    iree_string_view_t* out_name) {
  IREE_RETURN_IF_ERROR(
      loom_serve_device_configuration_string(parser, key, out_name));
  for (iree_host_size_t c = 0; c < out_name->size; ++c) {
    if ((uint8_t)out_name->data[c] < 0x21 ||
        (uint8_t)out_name->data[c] > 0x7e) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device and pool keys must be visible ASCII");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_entry(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_parser_t* parser = self;
  loom_serve_device_configuration_t* configuration = parser->configuration;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT ||
      configuration->count == IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "devices requires at most %d named objects",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  const iree_host_size_t index = configuration->count;
  loom_serve_device_options_t* options = &configuration->options[index];
  IREE_RETURN_IF_ERROR(
      loom_serve_device_configuration_name(parser, key, &options->name));
  for (iree_host_size_t i = 0; i < index; ++i) {
    if (iree_string_view_equal(configuration->options[i].name, options->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate device key '%.*s'",
                              (int)options->name.size, options->name.data);
    }
  }
  loom_serve_device_configuration_fields_t fields = {.parser = parser};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
                           value, loom_serve_device_configuration_uri, &fields),
                       "device '%.*s'", (int)options->name.size,
                       options->name.data);
  if (!fields.values[0].data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device '%.*s' requires 'uri'",
                            (int)options->name.size, options->name.data);
  }
  options->uri = fields.values[0];
  ++configuration->count;
  return iree_ok_status();
}

typedef struct loom_serve_device_configuration_members_t {
  // Decoded configuration and string storage retained through construction.
  loom_serve_device_configuration_parser_t* parser;
  // Resolved consumer set; each member must occur exactly once.
  uint64_t mask;
} loom_serve_device_configuration_members_t;

static iree_status_t loom_serve_device_configuration_find_device(
    const loom_serve_device_configuration_t* configuration,
    iree_string_view_t name, uint32_t* out_ordinal) {
  for (iree_host_size_t i = 0; i < configuration->count; ++i) {
    if (iree_string_view_equal(configuration->options[i].name, name)) {
      *out_ordinal = (uint32_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown pool device '%.*s'", (int)name.size,
                          name.data);
}

static iree_status_t loom_serve_device_configuration_member(
    void* self, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_members_t* members = self;
  if (type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool devices must be an array of device keys");
  }
  iree_string_view_t name;
  IREE_RETURN_IF_ERROR(
      loom_serve_device_configuration_string(members->parser, value, &name));
  uint32_t ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_serve_device_configuration_find_device(
      members->parser->configuration, name, &ordinal));
  const uint64_t bit = UINT64_C(1) << ordinal;
  if (members->mask & bit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate pool device '%.*s'", (int)name.size,
                            name.data);
  }
  members->mask |= bit;
  return iree_ok_status();
}

typedef struct loom_serve_device_configuration_placement_t {
  // Decoded device inventory and string storage owned by configuration parsing.
  loom_serve_device_configuration_parser_t* parser;
  // Single explicit selector; AUTOMATIC marks an empty object during parsing.
  loom_serve_pool_placement_t value;
} loom_serve_device_configuration_placement_t;

static iree_status_t loom_serve_device_configuration_placement(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_placement_t* placement = self;
  if (placement->value.kind != LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "placement requires exactly one selector");
  }
  if (iree_string_view_equal(key, IREE_SV("device")) &&
      type == IREE_JSON_VALUE_TYPE_STRING) {
    iree_string_view_t name;
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_string(
        placement->parser, value, &name));
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_find_device(
        placement->parser->configuration, name, &placement->value.ordinal));
    placement->value.kind = LOOM_SERVE_POOL_PLACEMENT_DEVICE;
  } else if (iree_string_view_equal(key, IREE_SV("host_numa")) &&
             type == IREE_JSON_VALUE_TYPE_NUMBER) {
    uint64_t ordinal = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(value, &ordinal));
    if (ordinal > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "host NUMA node exceeds the OS node namespace");
    }
    placement->value.kind = LOOM_SERVE_POOL_PLACEMENT_HOST_NUMA;
    placement->value.ordinal = (uint32_t)ordinal;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "placement requires a 'device' string or 'host_numa' integer");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_pool(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_device_configuration_parser_t* parser = self;
  loom_serve_device_configuration_t* configuration = parser->configuration;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pools must contain named objects");
  }
  const iree_host_size_t index = configuration->pool_count;
  loom_serve_pool_configuration_t pool = {
      .options = loom_serve_device_configuration_defaults()};
  IREE_RETURN_IF_ERROR(
      loom_serve_device_configuration_name(parser, key, &pool.binding.name));
  for (iree_host_size_t i = 0; i < index; ++i) {
    if (iree_string_view_equal(configuration->pools[i].binding.name,
                               pool.binding.name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "duplicate pool key '%.*s'",
          (int)pool.binding.name.size, pool.binding.name.data);
    }
  }
  loom_serve_device_configuration_fields_t fields = {.parser = parser};
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(
          value, loom_serve_device_configuration_field, &fields),
      "pool '%.*s'", (int)pool.binding.name.size, pool.binding.name.data);
  loom_serve_device_configuration_members_t members = {.parser = parser};
  if (fields.values[0].data) {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields.values[0], loom_serve_device_configuration_member, &members));
  }
  if (!members.mask) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "pool '%.*s' requires nonempty 'devices'",
        (int)pool.binding.name.size, pool.binding.name.data);
  }
  pool.binding.device_mask = members.mask;
  loom_serve_pool_options_t* options = &pool.options;
  if (fields.values[7].data) {
    loom_serve_device_configuration_placement_t placement = {.parser = parser};
    IREE_RETURN_IF_ERROR(
        iree_json_enumerate_object_typed(
            fields.values[7], loom_serve_device_configuration_placement,
            &placement),
        "pool '%.*s' placement", (int)pool.binding.name.size,
        pool.binding.name.data);
    if (placement.value.kind == LOOM_SERVE_POOL_PLACEMENT_AUTOMATIC) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pool placement requires one selector");
    }
    options->placement = placement.value;
  }
  if (fields.values[1].data) {
    if (iree_string_view_equal(fields.values[1], IREE_SV("fixed"))) {
      options->elastic = false;
    } else if (!iree_string_view_equal(fields.values[1], IREE_SV("elastic"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pool_backing must be elastic or fixed");
    }
  }
  uint64_t numbers[] = {options->weight_slab_size, options->state_slab_size, 0,
                        0, 0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(numbers) && iree_status_is_ok(status); ++i) {
    if (fields.values[i + 2].data) {
      if (i < 3) {
        iree_device_size_t size = 0;
        status =
            iree_string_view_parse_device_size(fields.values[i + 2], &size);
        numbers[i] = size;
      } else {
        status = iree_json_parse_uint64(fields.values[i + 2], &numbers[i]);
      }
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status, "pool '%.*s' field '%s'", (int)pool.binding.name.size,
            pool.binding.name.data,
            loom_serve_device_configuration_field_names[i + 2]);
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (numbers[3] > INT64_MAX / 1000000 || numbers[4] > INT64_MAX / 1000000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pool idle interval is out of range");
  }
  if (!options->elastic && numbers[2]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "a physical budget requires elastic backing");
  }
  options->weight_slab_size = (iree_device_size_t)numbers[0];
  options->state_slab_size = (iree_device_size_t)numbers[1];
  options->memory_limit = numbers[2];
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      configuration->host_allocator, index + 1, sizeof(*configuration->pools),
      (void**)&configuration->pools));
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      configuration->host_allocator, index + 1, sizeof(*configuration->idle),
      (void**)&configuration->idle));
  configuration->pools[index] = pool;
  configuration->idle[index] = (loom_serve_pool_idle_policy_t){
      .workspace_timeout = (iree_duration_t)numbers[3] * 1000000,
      .parameter_timeout = (iree_duration_t)numbers[4] * 1000000};
  ++configuration->pool_count;
  return iree_ok_status();
}

static iree_status_t loom_serve_device_configuration_root(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  iree_string_view_t* fields = self;
  iree_host_size_t index;
  if (iree_string_view_equal(key, IREE_SV("devices"))) {
    index = 0;
  } else if (iree_string_view_equal(key, IREE_SV("pools"))) {
    index = 1;
  } else if (iree_string_view_equal(key, IREE_SV("wait_mode"))) {
    index = 2;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown machine configuration field '%.*s'",
                            (int)key.size, key.data);
  }
  const iree_json_value_type_t expected_type =
      index == 2 ? IREE_JSON_VALUE_TYPE_STRING : IREE_JSON_VALUE_TYPE_OBJECT;
  if (fields[index].data || type != expected_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate or wrongly typed machine configuration field '%.*s'",
        (int)key.size, key.data);
  }
  fields[index] = value;
  return iree_ok_status();
}

iree_status_t loom_serve_device_configuration_initialize(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  *out_configuration = (loom_serve_device_configuration_t){0};
  const iree_host_size_t storage_size = json.size;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing device configuration JSON");
  }
  iree_string_view_t fields[3] = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_device_configuration_root, fields));
  if (!fields[0].data || !fields[1].data) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "machine configuration requires 'devices' and 'pools'");
  }
  char* strings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, (void**)&strings));
  out_configuration->host_allocator = host_allocator;
  out_configuration->strings = strings;
  loom_serve_device_configuration_parser_t parser = {
      .configuration = out_configuration,
      .next_string = out_configuration->strings};
  iree_status_t status = iree_json_enumerate_object_typed(
      fields[0], loom_serve_device_configuration_entry, &parser);
  if (iree_status_is_ok(status) && !out_configuration->count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "devices must contain at least one named device");
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_object_typed(
        fields[1], loom_serve_device_configuration_pool, &parser);
  }
  if (iree_status_is_ok(status) && !out_configuration->pool_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pools must contain at least one named pool");
  }
  iree_string_view_t wait_mode = IREE_SV("active");
  if (iree_status_is_ok(status) && fields[2].data) {
    status =
        loom_serve_device_configuration_string(&parser, fields[2], &wait_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_configuration_set_wait_mode(out_configuration,
                                                           wait_mode);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_device_configuration_deinitialize(out_configuration);
  }
  return status;
}

iree_status_t loom_serve_device_configuration_initialize_uri(
    iree_string_view_t uri, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  *out_configuration = (loom_serve_device_configuration_t){0};
  if (!uri.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device is required; specify a HAL device URI");
  }
  if (memchr(uri.data, 0, uri.size) || !iree_unicode_utf8_validate(uri)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device URI must be UTF-8 without NUL");
  }
  char* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      host_allocator, iree_make_const_byte_span(uri.data, uri.size),
      (void**)&strings));
  out_configuration->host_allocator = host_allocator;
  out_configuration->strings = strings;
  out_configuration->count = 1;
  out_configuration->options[0].name = IREE_SV("device");
  out_configuration->options[0].uri =
      iree_make_string_view(out_configuration->strings, uri.size);
  out_configuration->options[0].completion_wait_flags =
      IREE_ASYNC_WAIT_FLAG_ACTIVE;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*out_configuration->pools),
                            (void**)&out_configuration->pools);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*out_configuration->idle),
                              (void**)&out_configuration->idle);
  }
  if (iree_status_is_ok(status)) {
    out_configuration->pool_count = 1;
    out_configuration->pools[0] = (loom_serve_pool_configuration_t){
        .binding = {.name = IREE_SV("pool"), .device_mask = 1},
        .options = loom_serve_device_configuration_defaults(),
    };
  } else {
    loom_serve_device_configuration_deinitialize(out_configuration);
  }
  return status;
}

iree_status_t loom_serve_device_configuration_set_wait_mode(
    loom_serve_device_configuration_t* configuration,
    iree_string_view_t wait_mode) {
  iree_async_wait_flags_t flags;
  if (iree_string_view_equal(wait_mode, IREE_SV("active"))) {
    flags = IREE_ASYNC_WAIT_FLAG_ACTIVE;
  } else if (iree_string_view_equal(wait_mode, IREE_SV("blocking"))) {
    flags = IREE_ASYNC_WAIT_FLAG_NONE;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wait_mode must be active or blocking");
  }
  for (iree_host_size_t i = 0; i < configuration->count; ++i) {
    configuration->options[i].completion_wait_flags = flags;
  }
  return iree_ok_status();
}

void loom_serve_device_configuration_deinitialize(
    loom_serve_device_configuration_t* configuration) {
  iree_allocator_free(configuration->host_allocator, configuration->strings);
  iree_allocator_free(configuration->host_allocator, configuration->pools);
  iree_allocator_free(configuration->host_allocator, configuration->idle);
  memset(configuration, 0, sizeof(*configuration));
}
