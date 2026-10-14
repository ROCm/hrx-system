// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/configuration.h"

#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/internal/unicode.h"
#include "iree/io/file_contents.h"

typedef struct configuration_field_t {
  // Literal model key; escaped keys are rejected.
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

static iree_status_t loom_serve_model_configuration_field_visit(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  configuration_fields_t* fields = self;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    configuration_field_t* field = &fields->values[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || field->type != type) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed model field '%s'",
                              field->name);
    }
    field->value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown model field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_serve_model_configuration_string(
    const configuration_field_t* field, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  if (!field->value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model requires nonempty '%s'", field->name);
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, field->value.size + 1, (void**)&data));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(field->value, field->value.size, data, &length);
  const iree_string_view_t value = iree_make_string_view(data, length);
  if (iree_status_is_ok(status) &&
      (!length || !iree_unicode_utf8_validate(value) ||
       memchr(data, 0, length))) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "model strings must be nonempty UTF-8 without NUL");
  }
  if (iree_status_is_ok(status)) {
    data[length] = 0;
    *out_value = value;
  } else {
    iree_allocator_free(host_allocator, data);
  }
  return status;
}

static iree_status_t loom_serve_model_configuration_size(
    const configuration_field_t* field, uint64_t minimum, uint64_t maximum,
    iree_host_size_t* value) {
  if (!field->value.data) {
    return iree_ok_status();
  }
  uint64_t number = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(field->value, &number));
  if (number < minimum || number > maximum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model field '%s' must be in [%" PRIu64 ", %" PRIu64
                            "]",
                            field->name, minimum, maximum);
  }
  *value = (iree_host_size_t)number;
  return iree_ok_status();
}

static iree_status_t loom_serve_model_configuration_options(
    const configuration_field_t* field, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  const iree_string_view_t value =
      field->value.data ? field->value : IREE_SV("{}");
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      host_allocator, iree_make_const_byte_span(value.data, value.size),
      (void**)&data));
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

static iree_status_t loom_serve_model_configuration_component(
    iree_string_view_t value,
    loom_serve_finite_component_configuration_t* component,
    iree_allocator_t host_allocator) {
  configuration_field_t fields[] = {{"prepare", IREE_JSON_VALUE_TYPE_STRING},
                                    {"control", IREE_JSON_VALUE_TYPE_STRING},
                                    {"assets", IREE_JSON_VALUE_TYPE_STRING},
                                    {"options", IREE_JSON_VALUE_TYPE_OBJECT}};
  configuration_fields_t context = {IREE_ARRAYSIZE(fields), fields};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, loom_serve_model_configuration_field_visit, &context));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[0], host_allocator, &component->prepare));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[1], host_allocator, &component->control));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[2], host_allocator, &component->assets));
  return loom_serve_model_configuration_options(&fields[3], host_allocator,
                                                &component->options);
}

static iree_status_t loom_serve_model_configuration_source_count(
    void* self, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_STRING || !value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sources must contain nonempty path strings");
  }
  *(iree_host_size_t*)self = index + 1;
  return iree_ok_status();
}

static iree_status_t loom_serve_model_configuration_source(
    void* self, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_model_configuration_t* model = self;
  configuration_field_t field = {"sources", type, value};
  return loom_serve_model_configuration_string(
      &field, model->host_allocator,
      &((iree_string_view_t*)model->sources.values)[index]);
}

typedef enum definition_field_e {
  DEFINITION_KIND,
  DEFINITION_PREPARE,
  DEFINITION_CONTROL,
  DEFINITION_WEIGHTS,
  DEFINITION_TOKENIZER,
  DEFINITION_ROWS,
  DEFINITION_CONTEXT_CAPACITY,
  DEFINITION_POOL_CAPACITY,
  DEFINITION_PREFILL_CAPACITY,
  DEFINITION_CHECKPOINT_CAPACITY,
  DEFINITION_PENDING_REQUESTS,
  DEFINITION_MAX_TOKENS,
  DEFINITION_DRAFT_DEPTH,
  DEFINITION_CONTINUATION_EPOCHS,
  DEFINITION_CHECKPOINT,
  DEFINITION_ADAPTER,
  DEFINITION_WIDTH,
  DEFINITION_HEIGHT,
  DEFINITION_TEXT_TOKENS,
  DEFINITION_ASSETS,
  DEFINITION_OPTIONS,
  DEFINITION_SOURCES,
  DEFINITION_SESSION_CACHE_CAPACITY,
  DEFINITION_ADMISSION_OVERTAKES,
  DEFINITION_PREFIX_CACHE_CAPACITY,
  DEFINITION_COMMAND_QUANTUM,
  DEFINITION_VARIANT_CAPACITY,
  DEFINITION_IMAGE,
} definition_field_t;

static iree_status_t loom_serve_model_configuration_text(
    const configuration_field_t* fields, loom_serve_text_configuration_t* text,
    iree_allocator_t host_allocator) {
  *text = (loom_serve_text_configuration_t){.rows = 4,
                                            .context_capacity = 16384,
                                            .pool_capacity = 65536,
                                            .prefill_capacity = 512,
                                            .max_tokens = 512,
                                            .continuation_epochs = 1};
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[DEFINITION_WEIGHTS], host_allocator, &text->weights));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[DEFINITION_TOKENIZER], host_allocator, &text->tokenizer));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_ROWS], 1, 16, &text->rows));
  text->admission_overtakes = text->rows;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_ADMISSION_OVERTAKES], 0, INT32_MAX,
      &text->admission_overtakes));
  IREE_RETURN_IF_ERROR(
      loom_serve_model_configuration_size(&fields[DEFINITION_CONTEXT_CAPACITY],
                                          1, 262144, &text->context_capacity));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_POOL_CAPACITY], 1, 4194304, &text->pool_capacity));
  IREE_RETURN_IF_ERROR(
      loom_serve_model_configuration_size(&fields[DEFINITION_PREFILL_CAPACITY],
                                          1, 262144, &text->prefill_capacity));
  if (text->prefill_capacity > text->context_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prefill capacity exceeds context capacity");
  }
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_CHECKPOINT_CAPACITY], 0, INT32_MAX,
      &text->checkpoint_capacity));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_SESSION_CACHE_CAPACITY], 0, INT32_MAX - 1,
      &text->session_cache_capacity));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_PREFIX_CACHE_CAPACITY], 0, INT32_MAX,
      &text->prefix_cache_capacity));
  const uint64_t endpoint_capacity =
      (uint64_t)text->checkpoint_capacity + text->prefix_cache_capacity +
      (text->session_cache_capacity ? text->session_cache_capacity + 1 : 0);
  if (endpoint_capacity > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "combined endpoint capacity exceeds INT32_MAX");
  }
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_MAX_TOKENS], 1, 16384, &text->max_tokens));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_DRAFT_DEPTH], 0, 7, &text->draft_depth));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_CONTINUATION_EPOCHS], 1, 2,
      &text->continuation_epochs));
  if (text->prefill_capacity < text->draft_depth + 1 ||
      (text->continuation_epochs == 2 && !text->draft_depth)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "drafting requires capacity for an anchor and "
                            "every proposal; two-epoch continuation requires "
                            "nonzero draft depth");
  }
  if (fields[DEFINITION_IMAGE].value.data) {
    IREE_RETURN_IF_ERROR(loom_serve_model_configuration_component(
        fields[DEFINITION_IMAGE].value, &text->image, host_allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_model_configuration_image(
    const configuration_field_t* fields,
    loom_serve_image_configuration_t* image, iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[DEFINITION_CHECKPOINT], host_allocator, &image->checkpoint));
  if (fields[DEFINITION_ADAPTER].value.data) {
    IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
        &fields[DEFINITION_ADAPTER], host_allocator, &image->adapter));
  }
  iree_host_size_t width = 384, height = 384, text_tokens = 512;
  iree_host_size_t command_quantum = 0;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_WIDTH], 1, UINT32_MAX, &width));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_HEIGHT], 1, UINT32_MAX, &height));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_TEXT_TOKENS], 1, UINT32_MAX, &text_tokens));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_COMMAND_QUANTUM], 0, UINT32_MAX, &command_quantum));
  image->width = (uint32_t)width;
  image->height = (uint32_t)height;
  image->text_tokens = (uint32_t)text_tokens;
  image->command_quantum = (uint32_t)command_quantum;
  return iree_ok_status();
}

static iree_status_t loom_serve_model_configuration_definition_parse(
    iree_string_view_t value, loom_serve_model_configuration_t* model,
    iree_allocator_t host_allocator) {
  configuration_field_t fields[] = {
      [DEFINITION_KIND] = {"kind", IREE_JSON_VALUE_TYPE_STRING},
      [DEFINITION_PREPARE] = {"prepare", IREE_JSON_VALUE_TYPE_STRING},
      [DEFINITION_CONTROL] = {"control", IREE_JSON_VALUE_TYPE_STRING},
      [DEFINITION_WEIGHTS] = {"weights", IREE_JSON_VALUE_TYPE_STRING,
                              .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_TOKENIZER] = {"tokenizer", IREE_JSON_VALUE_TYPE_STRING,
                                .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_ROWS] = {"rows", IREE_JSON_VALUE_TYPE_NUMBER,
                           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_CONTEXT_CAPACITY] = {"context_capacity",
                                       IREE_JSON_VALUE_TYPE_NUMBER,
                                       .kinds =
                                           (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_POOL_CAPACITY] = {"pool_capacity",
                                    IREE_JSON_VALUE_TYPE_NUMBER,
                                    .kinds =
                                        (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_PREFILL_CAPACITY] = {"prefill_capacity",
                                       IREE_JSON_VALUE_TYPE_NUMBER,
                                       .kinds =
                                           (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_CHECKPOINT_CAPACITY] =
          {"checkpoint_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_PENDING_REQUESTS] = {"pending_requests",
                                       IREE_JSON_VALUE_TYPE_NUMBER},
      [DEFINITION_MAX_TOKENS] = {"max_tokens", IREE_JSON_VALUE_TYPE_NUMBER,
                                 .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_DRAFT_DEPTH] = {"draft_depth", IREE_JSON_VALUE_TYPE_NUMBER,
                                  .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_CONTINUATION_EPOCHS] =
          {"continuation_epochs", IREE_JSON_VALUE_TYPE_NUMBER,
           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_CHECKPOINT] = {"checkpoint", IREE_JSON_VALUE_TYPE_STRING,
                                 .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_ADAPTER] = {"adapter", IREE_JSON_VALUE_TYPE_STRING,
                              .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_WIDTH] = {"width", IREE_JSON_VALUE_TYPE_NUMBER,
                            .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_HEIGHT] = {"height", IREE_JSON_VALUE_TYPE_NUMBER,
                             .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_TEXT_TOKENS] = {"text_tokens", IREE_JSON_VALUE_TYPE_NUMBER,
                                  .kinds = (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_ASSETS] = {"assets", IREE_JSON_VALUE_TYPE_STRING,
                             .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
                                      (1u << LOOM_SERVE_MODEL_KIND_DECISION)},
      [DEFINITION_OPTIONS] = {"options", IREE_JSON_VALUE_TYPE_OBJECT},
      [DEFINITION_SOURCES] = {"sources", IREE_JSON_VALUE_TYPE_ARRAY},
      [DEFINITION_SESSION_CACHE_CAPACITY] =
          {"session_cache_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_ADMISSION_OVERTAKES] =
          {"admission_overtakes", IREE_JSON_VALUE_TYPE_NUMBER,
           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_PREFIX_CACHE_CAPACITY] =
          {"prefix_cache_capacity", IREE_JSON_VALUE_TYPE_NUMBER,
           .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)},
      [DEFINITION_COMMAND_QUANTUM] = {"command_quantum",
                                      IREE_JSON_VALUE_TYPE_NUMBER,
                                      .kinds =
                                          (1u << LOOM_SERVE_MODEL_KIND_IMAGE)},
      [DEFINITION_VARIANT_CAPACITY] = {"variant_capacity",
                                       IREE_JSON_VALUE_TYPE_NUMBER},
      [DEFINITION_IMAGE] = {"image", IREE_JSON_VALUE_TYPE_OBJECT,
                            .kinds = (1u << LOOM_SERVE_MODEL_KIND_TEXT)}};
  configuration_fields_t context = {IREE_ARRAYSIZE(fields), fields};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, loom_serve_model_configuration_field_visit, &context));
  if (iree_string_view_equal(fields[DEFINITION_KIND].value, IREE_SV("text"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_TEXT;
  } else if (iree_string_view_equal(fields[DEFINITION_KIND].value,
                                    IREE_SV("image"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_IMAGE;
  } else if (iree_string_view_equal(fields[DEFINITION_KIND].value,
                                    IREE_SV("embedding"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_EMBEDDING;
  } else if (iree_string_view_equal(fields[DEFINITION_KIND].value,
                                    IREE_SV("decision"))) {
    model->kind = LOOM_SERVE_MODEL_KIND_DECISION;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "model kind must be text, image, embedding or decision");
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    if (fields[i].value.data && fields[i].kinds &&
        !(fields[i].kinds & (1u << model->kind))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model field '%s' does not apply to kind '%.*s'",
                              fields[i].name,
                              (int)fields[DEFINITION_KIND].value.size,
                              fields[DEFINITION_KIND].value.data);
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[DEFINITION_PREPARE], host_allocator, &model->sources.prepare));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[DEFINITION_CONTROL], host_allocator, &model->sources.control));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      fields[DEFINITION_SOURCES].value,
      loom_serve_model_configuration_source_count, &model->sources.count));
  if (!model->sources.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model requires at least one source provider");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, model->sources.count, sizeof(*model->sources.values),
      (void**)&model->sources.values));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      fields[DEFINITION_SOURCES].value, loom_serve_model_configuration_source,
      model));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_PENDING_REQUESTS], 1, IREE_HOST_SIZE_MAX,
      &model->pending_requests));
  model->variant_capacity = 16;
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[DEFINITION_VARIANT_CAPACITY], 0, INT32_MAX,
      &model->variant_capacity));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_options(
      &fields[DEFINITION_OPTIONS], host_allocator, &model->options));
  if (model->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    return loom_serve_model_configuration_text(fields, &model->text,
                                               host_allocator);
  }
  if (model->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    return loom_serve_model_configuration_image(fields, &model->image,
                                                host_allocator);
  }
  return loom_serve_model_configuration_string(
      &fields[DEFINITION_ASSETS], host_allocator, &model->finite.assets);
}

void loom_serve_model_configuration_deinitialize(
    loom_serve_model_configuration_t* model) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_allocator_free(host_allocator, (void*)model->name.data);
  iree_allocator_free(host_allocator,
                      (void*)model->sources.base_directory.data);
  iree_allocator_free(host_allocator, (void*)model->sources.prepare.data);
  iree_allocator_free(host_allocator, (void*)model->sources.control.data);
  loom_serve_package_release(model->sources.package);
  for (iree_host_size_t i = 0;
       model->sources.values && i < model->sources.count; ++i) {
    iree_allocator_free(host_allocator, (void*)model->sources.values[i].data);
  }
  iree_allocator_free(host_allocator, (void*)model->sources.values);
  iree_allocator_free(host_allocator, (void*)model->options.data);
  if (model->kind == LOOM_SERVE_MODEL_KIND_TEXT) {
    iree_allocator_free(host_allocator, (void*)model->text.weights.data);
    iree_allocator_free(host_allocator, (void*)model->text.tokenizer.data);
    iree_allocator_free(host_allocator, (void*)model->text.image.prepare.data);
    iree_allocator_free(host_allocator, (void*)model->text.image.control.data);
    iree_allocator_free(host_allocator, (void*)model->text.image.assets.data);
    iree_allocator_free(host_allocator, (void*)model->text.image.options.data);
  } else if (model->kind == LOOM_SERVE_MODEL_KIND_IMAGE) {
    iree_allocator_free(host_allocator, (void*)model->image.checkpoint.data);
    iree_allocator_free(host_allocator, (void*)model->image.adapter.data);
  } else if (model->kind == LOOM_SERVE_MODEL_KIND_EMBEDDING ||
             model->kind == LOOM_SERVE_MODEL_KIND_DECISION) {
    iree_allocator_free(host_allocator, (void*)model->finite.assets.data);
  }
  memset(model, 0, sizeof(*model));
}

static iree_status_t loom_serve_model_configuration_model_parse(
    iree_string_view_t json, iree_string_view_t document_directory,
    loom_serve_package_t* package, iree_allocator_t host_allocator,
    loom_serve_model_configuration_t* out_model) {
  *out_model =
      (loom_serve_model_configuration_t){.host_allocator = host_allocator,
                                         .device_index = IREE_HOST_SIZE_MAX,
                                         .pending_requests = 32};
  out_model->sources.package = package;
  loom_serve_package_retain(package);
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing model JSON");
  }
  configuration_field_t fields[] = {
      {"name", IREE_JSON_VALUE_TYPE_STRING},
      {"definition", IREE_JSON_VALUE_TYPE_OBJECT},
      {"admission", IREE_JSON_VALUE_TYPE_STRING},
      {"base_directory", IREE_JSON_VALUE_TYPE_STRING},
      {"device", IREE_JSON_VALUE_TYPE_NUMBER}};
  configuration_fields_t context = {IREE_ARRAYSIZE(fields), fields};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_model_configuration_field_visit, &context));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
      &fields[0], host_allocator, &out_model->name));
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_size(
      &fields[4], 0, IREE_HOST_SIZE_MAX - 1, &out_model->device_index));
  for (iree_host_size_t i = 0; i < out_model->name.size; ++i) {
    const uint8_t c = (uint8_t)out_model->name.data[i];
    if (c < 0x21 || c > 0x7e) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "model names must fit a visible ASCII HTTP header selector");
    }
  }
  if (fields[2].value.data) {
    if (iree_string_view_equal(fields[2].value, IREE_SV("drained"))) {
      out_model->start_drained = true;
    } else if (!iree_string_view_equal(fields[2].value, IREE_SV("open"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "admission must be open or drained");
    }
  }
  if (fields[3].value.data) {
    IREE_RETURN_IF_ERROR(loom_serve_model_configuration_string(
        &fields[3], host_allocator, &out_model->sources.base_directory));
    if (!loom_serve_path_is_absolute(out_model->sources.base_directory)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "explicit base_directory must be absolute");
    }
  } else if (document_directory.size) {
    char* base = NULL;
    IREE_RETURN_IF_ERROR(iree_file_path_join(
        document_directory, iree_string_view_empty(), host_allocator, &base));
    out_model->sources.base_directory = iree_make_cstring_view(base);
  }
  IREE_RETURN_IF_ERROR(loom_serve_model_configuration_definition_parse(
      fields[1].value, out_model, host_allocator));
  if (out_model->sources.base_directory.size) {
    return iree_ok_status();
  }
  if (package) {
    // The package resolves program and metadata locators. Only explicit
    // checkpoint files need a local base here; source-declared parameter paths
    // are checked when their preparation declarations are consumed.
    const iree_string_view_t checkpoint =
        out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT ? out_model->text.weights
        : out_model->kind == LOOM_SERVE_MODEL_KIND_IMAGE
            ? out_model->image.adapter
            : iree_string_view_empty();
    if (checkpoint.size && !loom_serve_path_is_absolute(checkpoint)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "relative checkpoint paths require base_directory");
    }
    return iree_ok_status();
  }
  // Without a document base, every native-owned locator must stand alone.
  // Source-declared checkpoint/policy paths are checked when they are bound.
  const iree_string_view_t paths[] = {
      out_model->sources.prepare,
      out_model->sources.control,
      out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT ? out_model->text.weights
      : out_model->kind == LOOM_SERVE_MODEL_KIND_IMAGE
          ? out_model->image.checkpoint
          : out_model->finite.assets,
      out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT ? out_model->text.tokenizer
      : out_model->kind == LOOM_SERVE_MODEL_KIND_IMAGE
          ? out_model->image.adapter
          : iree_string_view_empty(),
      out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT
          ? out_model->text.image.prepare
          : iree_string_view_empty(),
      out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT
          ? out_model->text.image.control
          : iree_string_view_empty(),
      out_model->kind == LOOM_SERVE_MODEL_KIND_TEXT
          ? out_model->text.image.assets
          : iree_string_view_empty()};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
    if (paths[i].size && !loom_serve_path_is_absolute(paths[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "relative model paths require base_directory");
    }
  }
  for (iree_host_size_t i = 0; i < out_model->sources.count; ++i) {
    if (!loom_serve_path_is_absolute(out_model->sources.values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "relative source paths require base_directory");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_model_configuration_initialize(
    iree_string_view_t json, iree_string_view_t document_directory,
    loom_serve_package_t* package, iree_allocator_t host_allocator,
    loom_serve_model_configuration_t* out_model) {
  iree_status_t status = loom_serve_model_configuration_model_parse(
      json, document_directory, package, host_allocator, out_model);
  if (!iree_status_is_ok(status)) {
    loom_serve_model_configuration_deinitialize(out_model);
  }
  return status;
}

iree_status_t loom_serve_model_configuration_load(
    iree_string_view_t path, iree_allocator_t host_allocator,
    loom_serve_model_configuration_t* out_model) {
  *out_model = (loom_serve_model_configuration_t){0};
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(path, host_allocator, &contents));
  iree_string_view_t directory = iree_file_path_dirname(path);
  if (!directory.size) {
    directory = IREE_SV(".");
  }
  iree_status_t status = loom_serve_model_configuration_initialize(
      iree_make_string_view((const char*)contents->const_buffer.data,
                            contents->const_buffer.data_length),
      directory, NULL, host_allocator, out_model);
  iree_io_file_contents_free(contents);
  return status;
}

iree_status_t loom_serve_model_configuration_resolve_device(
    const loom_serve_model_configuration_t* model,
    iree_host_size_t device_count, iree_host_size_t* out_device_index) {
  iree_host_size_t index = model->device_index;
  if (index == IREE_HOST_SIZE_MAX) {
    if (device_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model '%.*s' requires an explicit device "
                              "ordinal for a %zu-device group",
                              (int)model->name.size, model->name.data,
                              device_count);
    }
    index = 0;
  }
  if (index >= device_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "model '%.*s' selects device %zu, but the group has %zu devices",
        (int)model->name.size, model->name.data, index, device_count);
  }
  *out_device_index = index;
  return iree_ok_status();
}
