// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/configuration.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

typedef struct configuration_field_t {
  // Literal catalog key; escaped keys are rejected.
  const char* name;
  // Required JSON type for this field.
  iree_json_value_type_t type;
  // Raw borrowed JSON value, with null data marking an absent field.
  iree_string_view_t value;
} configuration_field_t;

typedef struct configuration_fields_t {
  // Number of mutable declared fields.
  iree_host_size_t count;
  // Borrowed field destinations for this object.
  configuration_field_t* values;
} configuration_fields_t;

static iree_status_t configuration_field_visit(void* self,
                                               iree_string_view_t key,
                                               iree_json_value_type_t type,
                                               iree_string_view_t value) {
  configuration_fields_t* fields = self;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    configuration_field_t* field = &fields->values[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || field->type != type) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed catalog field '%s'",
                              field->name);
    }
    field->value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown catalog field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t configuration_string(const configuration_field_t* field,
                                          iree_string_view_t* out_value,
                                          iree_allocator_t allocator) {
  if (!field->value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "catalog requires nonempty '%s'", field->name);
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, field->value.size + 1, (void**)&data));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(field->value, field->value.size, data, &length);
  const iree_string_view_t value = iree_make_string_view(data, length);
  if (iree_status_is_ok(status) &&
      (!length || !iree_unicode_utf8_validate(value) ||
       memchr(data, 0, length))) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "catalog strings must be nonempty UTF-8 without NUL");
  }
  if (iree_status_is_ok(status)) {
    data[length] = 0;
    *out_value = value;
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

static iree_status_t configuration_size(const configuration_field_t* field,
                                        uint64_t minimum, uint64_t maximum,
                                        iree_host_size_t* value) {
  if (!field->value.data) {
    return iree_ok_status();
  }
  uint64_t number = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(field->value, &number));
  if (number < minimum || number > maximum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "catalog field '%s' must be in [%" PRIu64
                            ", %" PRIu64 "]",
                            field->name, minimum, maximum);
  }
  *value = (iree_host_size_t)number;
  return iree_ok_status();
}

static iree_status_t configuration_count(void* self, iree_host_size_t index,
                                         iree_json_value_type_t type,
                                         iree_string_view_t value) {
  (void)value;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "models must contain objects");
  }
  *(iree_host_size_t*)self = index + 1;
  return iree_ok_status();
}

static iree_status_t configuration_model(void* self, iree_host_size_t index,
                                         iree_json_value_type_t type,
                                         iree_string_view_t value) {
  (void)type;
  loom_serve_configuration_t* configuration = self;
  loom_serve_model_configuration_t* model = &configuration->models[index];
  *model = (loom_serve_model_configuration_t){.rows = 4,
                                              .context_capacity = 16384,
                                              .pool_capacity = 65536,
                                              .prefill_capacity = 512,
                                              .pending_requests = 32,
                                              .max_tokens = 512,
                                              .continuation_epochs = 1};
  configuration->model_count = index + 1;
  configuration_field_t fields[] = {
      {"kind", IREE_JSON_VALUE_TYPE_STRING},
      {"name", IREE_JSON_VALUE_TYPE_STRING},
      {"source", IREE_JSON_VALUE_TYPE_STRING},
      {"weights", IREE_JSON_VALUE_TYPE_STRING},
      {"tokenizer", IREE_JSON_VALUE_TYPE_STRING},
      {"rows", IREE_JSON_VALUE_TYPE_NUMBER},
      {"context_capacity", IREE_JSON_VALUE_TYPE_NUMBER},
      {"pool_capacity", IREE_JSON_VALUE_TYPE_NUMBER},
      {"prefill_capacity", IREE_JSON_VALUE_TYPE_NUMBER},
      {"checkpoint_capacity", IREE_JSON_VALUE_TYPE_NUMBER},
      {"pending_requests", IREE_JSON_VALUE_TYPE_NUMBER},
      {"max_tokens", IREE_JSON_VALUE_TYPE_NUMBER},
      {"mtp_depth", IREE_JSON_VALUE_TYPE_NUMBER},
      {"continuation_epochs", IREE_JSON_VALUE_TYPE_NUMBER}};
  configuration_fields_t context = {IREE_ARRAYSIZE(fields), fields};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, configuration_field_visit, &context));
  if (!iree_string_view_equal(fields[0].value, IREE_SV("text"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "catalog model kind must be 'text'");
  }
  iree_string_view_t* strings[] = {&model->name, &model->source,
                                   &model->weights, &model->tokenizer};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(strings) && iree_status_is_ok(status); ++i) {
    status = configuration_string(&fields[i + 1], strings[i],
                                  configuration->allocator);
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  for (iree_host_size_t i = 0; i < model->name.size; ++i) {
    const uint8_t c = (uint8_t)model->name.data[i];
    if (c < 0x21 || c > 0x7e) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "model names must fit a visible ASCII HTTP header selector");
    }
  }
  IREE_RETURN_IF_ERROR(configuration_size(&fields[5], 1, 16, &model->rows));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[6], 1, 262144, &model->context_capacity));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[7], 1, 4194304, &model->pool_capacity));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[8], 1, 512, &model->prefill_capacity));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[9], 0, IREE_HOST_SIZE_MAX,
                                          &model->checkpoint_capacity));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[10], 1, IREE_HOST_SIZE_MAX,
                                          &model->pending_requests));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[11], 1, 16384, &model->max_tokens));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[12], 0, 3, &model->mtp_depth));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[13], 1, 2, &model->continuation_epochs));
  if ((model->mtp_depth != 0 && model->mtp_depth != 3) ||
      (model->mtp_depth && model->prefill_capacity < 4) ||
      (model->continuation_epochs == 2 && model->mtp_depth != 3)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MTP requires depth 3 and at least four tokens; "
                            "two-epoch continuation requires MTP");
  }
  for (iree_host_size_t i = 0; i < index; ++i) {
    if (iree_string_view_equal(configuration->models[i].name, model->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate model name");
    }
  }
  return iree_ok_status();
}

void loom_serve_configuration_deinitialize(
    loom_serve_configuration_t* configuration) {
  for (iree_host_size_t i = 0; i < configuration->model_count; ++i) {
    loom_serve_model_configuration_t* model = &configuration->models[i];
    iree_allocator_free(configuration->allocator, (void*)model->name.data);
    iree_allocator_free(configuration->allocator, (void*)model->source.data);
    iree_allocator_free(configuration->allocator, (void*)model->weights.data);
    iree_allocator_free(configuration->allocator, (void*)model->tokenizer.data);
  }
  iree_allocator_free(configuration->allocator, configuration->models);
  memset(configuration, 0, sizeof(*configuration));
}

iree_status_t loom_serve_configuration_initialize(
    iree_string_view_t json, loom_serve_configuration_t* out_configuration,
    iree_allocator_t host_allocator) {
  *out_configuration =
      (loom_serve_configuration_t){.allocator = host_allocator};
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing catalog JSON");
  }
  configuration_field_t models = {"models", IREE_JSON_VALUE_TYPE_ARRAY};
  configuration_fields_t context = {1, &models};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, configuration_field_visit, &context));
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      models.value, configuration_count, &count));
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "catalog requires at least one model");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, count, sizeof(*out_configuration->models),
      (void**)&out_configuration->models));
  iree_status_t status = iree_json_enumerate_array_typed(
      models.value, configuration_model, out_configuration);
  if (!iree_status_is_ok(status)) {
    loom_serve_configuration_deinitialize(out_configuration);
  }
  return status;
}
