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
  // Allowed model kinds as bit positions, or zero for common fields.
  uint32_t kinds;
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

static iree_status_t configuration_text(const configuration_field_t* fields,
                                        loom_serve_text_configuration_t* text,
                                        iree_allocator_t allocator) {
  *text = (loom_serve_text_configuration_t){.rows = 4,
                                            .context_capacity = 16384,
                                            .pool_capacity = 65536,
                                            .prefill_capacity = 512,
                                            .max_tokens = 512,
                                            .continuation_epochs = 1};
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[3], &text->weights, allocator));
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[4], &text->tokenizer, allocator));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[5], 1, 16, &text->rows));
  text->admission_overtakes = text->rows;
  IREE_RETURN_IF_ERROR(configuration_size(&fields[23], 0, INT32_MAX,
                                          &text->admission_overtakes));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[6], 1, 262144, &text->context_capacity));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[7], 1, 4194304, &text->pool_capacity));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[8], 1, 262144, &text->prefill_capacity));
  if (text->prefill_capacity > text->context_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prefill capacity exceeds context capacity");
  }
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[9], 0, INT32_MAX, &text->checkpoint_capacity));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[22], 0, INT32_MAX - 1,
                                          &text->session_cache_capacity));
  if (text->session_cache_capacity &&
      text->checkpoint_capacity >
          INT32_MAX - text->session_cache_capacity - 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "combined endpoint capacity exceeds INT32_MAX");
  }
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[11], 1, 16384, &text->max_tokens));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[12], 0, 3, &text->mtp_depth));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[13], 1, 2, &text->continuation_epochs));
  if ((text->mtp_depth != 0 && text->mtp_depth != 3) ||
      (text->mtp_depth && text->prefill_capacity < 4) ||
      (text->continuation_epochs == 2 && text->mtp_depth != 3)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MTP requires depth 3 and at least four tokens; "
                            "two-epoch continuation requires MTP");
  }
  return iree_ok_status();
}

static iree_status_t configuration_image(
    const configuration_field_t* fields,
    loom_serve_image_configuration_t* image, iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[14], &image->checkpoint, allocator));
  if (fields[15].value.data) {
    IREE_RETURN_IF_ERROR(
        configuration_string(&fields[15], &image->adapter, allocator));
  }
  iree_host_size_t width = 384, height = 384, text_tokens = 512;
  IREE_RETURN_IF_ERROR(configuration_size(&fields[16], 1, UINT32_MAX, &width));
  IREE_RETURN_IF_ERROR(configuration_size(&fields[17], 1, UINT32_MAX, &height));
  IREE_RETURN_IF_ERROR(
      configuration_size(&fields[18], 1, UINT32_MAX, &text_tokens));
  image->width = (uint32_t)width;
  image->height = (uint32_t)height;
  image->text_tokens = (uint32_t)text_tokens;
  return iree_ok_status();
}

static iree_status_t configuration_model_parse(
    iree_string_view_t value, loom_serve_model_configuration_t* model,
    iree_allocator_t allocator) {
  *model = (loom_serve_model_configuration_t){.allocator = allocator,
                                              .pending_requests = 32};
  configuration_field_t fields[] = {
      {"kind", IREE_JSON_VALUE_TYPE_STRING},
      {"name", IREE_JSON_VALUE_TYPE_STRING},
      {"source", IREE_JSON_VALUE_TYPE_STRING},
      {"weights", IREE_JSON_VALUE_TYPE_STRING,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"tokenizer", IREE_JSON_VALUE_TYPE_STRING,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"rows", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"context_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"pool_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"prefill_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"checkpoint_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"pending_requests", IREE_JSON_VALUE_TYPE_NUMBER},
      {"max_tokens", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"mtp_depth", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"continuation_epochs", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"checkpoint", IREE_JSON_VALUE_TYPE_STRING,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      {"adapter", IREE_JSON_VALUE_TYPE_STRING,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      {"width", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      {"height", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      {"text_tokens", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      {"assets", IREE_JSON_VALUE_TYPE_STRING,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
                (1u << LOOM_SERVE_MODEL_KIND_DECISION)},
      {"options", IREE_JSON_VALUE_TYPE_OBJECT,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
                (1u << LOOM_SERVE_MODEL_KIND_DECISION)},
      {"admission", IREE_JSON_VALUE_TYPE_STRING},
      {"session_cache_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      {"admission_overtakes", IREE_JSON_VALUE_TYPE_NUMBER,
       .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)}};
  configuration_fields_t context = {IREE_ARRAYSIZE(fields), fields};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, configuration_field_visit, &context));
  if (iree_string_view_equal(fields[0].value, IREE_SV("text"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_TEXT;
  } else if (iree_string_view_equal(fields[0].value, IREE_SV("image"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_IMAGE;
  } else if (iree_string_view_equal(fields[0].value, IREE_SV("embedding"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_EMBEDDING;
  } else if (iree_string_view_equal(fields[0].value, IREE_SV("decision"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_DECISION;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "catalog model kind must be text, image, embedding or decision");
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    if (fields[i].value.data && fields[i].kinds &&
        !(fields[i].kinds & (1u << model->kind))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "catalog field '%s' does not apply to kind '%.*s'", fields[i].name,
          (int)fields[0].value.size, fields[0].value.data);
    }
  }
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[1], &model->name, allocator));
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[2], &model->source, allocator));
  for (iree_host_size_t i = 0; i < model->name.size; ++i) {
    const uint8_t c = (uint8_t)model->name.data[i];
    if (c < 0x21 || c > 0x7e) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "model names must fit a visible ASCII HTTP header selector");
    }
  }
  IREE_RETURN_IF_ERROR(configuration_size(&fields[10], 1, IREE_HOST_SIZE_MAX,
                                          &model->pending_requests));
  if (fields[21].value.data) {
    if (iree_string_view_equal(fields[21].value, IREE_SV("drained"))) {
      model->start_drained = true;
    } else if (!iree_string_view_equal(fields[21].value, IREE_SV("open"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "admission must be open or drained");
    }
  }
  if (model->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    return configuration_text(fields, &model->text, allocator);
  } else if (model->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    return configuration_image(fields, &model->image, allocator);
  }
  IREE_RETURN_IF_ERROR(
      configuration_string(&fields[19], &model->finite.assets, allocator));
  const iree_string_view_t options =
      fields[20].value.data ? fields[20].value : IREE_SV("{}");
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      allocator, iree_make_const_byte_span(options.data, options.size),
      (void**)&data));
  model->finite.options = iree_make_string_view(data, options.size);
  return iree_ok_status();
}

void loom_serve_model_configuration_deinitialize(
    loom_serve_model_configuration_t* model) {
  const iree_allocator_t allocator = model->allocator;
  iree_allocator_free(allocator, (void*)model->name.data);
  iree_allocator_free(allocator, (void*)model->source.data);
  if (model->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    iree_allocator_free(allocator, (void*)model->text.weights.data);
    iree_allocator_free(allocator, (void*)model->text.tokenizer.data);
  } else if (model->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    iree_allocator_free(allocator, (void*)model->image.checkpoint.data);
    iree_allocator_free(allocator, (void*)model->image.adapter.data);
  } else if (model->kind == LOOM_SERVE_MODEL_KIND_EMBEDDING ||
             model->kind == LOOM_SERVE_MODEL_KIND_DECISION) {
    iree_allocator_free(allocator, (void*)model->finite.assets.data);
    iree_allocator_free(allocator, (void*)model->finite.options.data);
  }
  memset(model, 0, sizeof(*model));
}

iree_status_t loom_serve_model_configuration_initialize(
    iree_string_view_t json, loom_serve_model_configuration_t* out_model,
    iree_allocator_t host_allocator) {
  *out_model = (loom_serve_model_configuration_t){0};
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing model JSON");
  }
  iree_status_t status =
      configuration_model_parse(object, out_model, host_allocator);
  if (!iree_status_is_ok(status)) {
    loom_serve_model_configuration_deinitialize(out_model);
  }
  return status;
}

static iree_status_t configuration_model(void* self, iree_host_size_t index,
                                         iree_json_value_type_t type,
                                         iree_string_view_t value) {
  loom_serve_configuration_t* configuration = self;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model must be an object");
  }
  loom_serve_model_configuration_t* model = &configuration->models[index];
  configuration->model_count = index + 1;
  IREE_RETURN_IF_ERROR(
      configuration_model_parse(value, model, configuration->allocator));
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
    loom_serve_model_configuration_deinitialize(&configuration->models[i]);
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
    return iree_ok_status();
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
