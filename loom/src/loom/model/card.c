// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/model/card.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

struct loom_model_card_t {
  // Shared immutable ownership across preparation and runtime consumers.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the card and its arena blocks.
  iree_allocator_t host_allocator;
  // Private cold-path block pool; no global cache or synchronization owner.
  iree_arena_block_pool_t block_pool;
  // Owns selected descriptors, decoded strings, and retained JSON.
  iree_arena_allocator_t arena;
  // Published only after complete selection validation.
  loom_model_selection_t selection;
};

typedef struct loom_model_json_field_t {
  // Decoded key used for semantic identity and duplicate detection.
  iree_string_view_t key;
  // Original escaped key, retained for lossless JSON composition.
  iree_string_view_t raw_key;
  // Type established by the JSON parser.
  iree_json_value_type_t type;
  // Original value; strings exclude their surrounding quotes.
  iree_string_view_t value;
} loom_model_json_field_t;

typedef struct loom_model_json_object_t {
  // Arena owner used while constructing fields.
  loom_model_card_t* card;
  // Number of initialized fields.
  iree_host_size_t count;
  // Arena-owned fields in document order.
  loom_model_json_field_t* fields;
} loom_model_json_object_t;

static iree_status_t loom_model_copy(loom_model_card_t* card,
                                     iree_string_view_t value,
                                     iree_string_view_t* out_value) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&card->arena, value.size, (void**)&data));
  if (value.size) {
    memcpy(data, value.data, value.size);
  }
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

static iree_status_t loom_model_string(loom_model_card_t* card,
                                       iree_string_view_t raw,
                                       iree_string_view_t* out_value) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&card->arena, raw.size, (void**)&data));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(raw, raw.size, data, &length));
  *out_value = iree_make_string_view(data, length);
  if (!length || memchr(data, 0, length) ||
      !iree_unicode_utf8_validate(*out_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model identities and paths require nonempty "
                            "UTF-8 strings without NUL");
  }
  return iree_ok_status();
}

static iree_status_t loom_model_count_field(void* self, iree_string_view_t key,
                                            iree_json_value_type_t type,
                                            iree_string_view_t value) {
  ++*(iree_host_size_t*)self;
  return iree_ok_status();
}

static iree_status_t loom_model_parse_field(void* self, iree_string_view_t key,
                                            iree_json_value_type_t type,
                                            iree_string_view_t value) {
  loom_model_json_object_t* object = self;
  loom_model_json_field_t* field = &object->fields[object->count];
  IREE_RETURN_IF_ERROR(loom_model_string(object->card, key, &field->key));
  for (iree_host_size_t i = 0; i < object->count; ++i) {
    if (iree_string_view_equal(field->key, object->fields[i].key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate model field '%.*s'",
                              (int)field->key.size, field->key.data);
    }
  }
  field->raw_key = key;
  field->type = type;
  field->value = value;
  ++object->count;
  return iree_ok_status();
}

static iree_status_t loom_model_check_fields(
    const loom_model_json_object_t* object, const char* allowed) {
  if (!allowed) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < object->count; ++i) {
    iree_string_view_t remaining = iree_make_cstring_view(allowed);
    bool found = false;
    while (remaining.size) {
      iree_string_view_t name;
      iree_string_view_split(remaining, ' ', &name, &remaining);
      if (iree_string_view_equal(name, object->fields[i].key)) {
        found = true;
        break;
      }
    }
    if (!found) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "unknown model field '%.*s'",
          (int)object->fields[i].key.size, object->fields[i].key.data);
    }
  }
  return iree_ok_status();
}

// Each object is decoded once at the external boundary. Downstream selection
// consumes these fields instead of rescanning JSON for each property.
static iree_status_t loom_model_object(loom_model_card_t* card,
                                       iree_string_view_t json,
                                       const char* allowed,
                                       loom_model_json_object_t* out_object) {
  *out_object = (loom_model_json_object_t){.card = card};
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(json, loom_model_count_field, &count));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&card->arena, count,
                                                 sizeof(*out_object->fields),
                                                 (void**)&out_object->fields));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      json, loom_model_parse_field, out_object));
  return loom_model_check_fields(out_object, allowed);
}

static const loom_model_json_field_t* loom_model_field(
    const loom_model_json_object_t* object, iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < object->count; ++i) {
    if (iree_string_view_equal(object->fields[i].key, key)) {
      return &object->fields[i];
    }
  }
  return NULL;
}

static iree_status_t loom_model_value(const loom_model_json_object_t* object,
                                      const char* key,
                                      iree_json_value_type_t type,
                                      bool required,
                                      iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  const loom_model_json_field_t* field =
      loom_model_field(object, iree_make_cstring_view(key));
  if (!field) {
    if (!required) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "model requires '%s'",
                            key);
  }
  if (field->type != type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wrong type for model field '%s'", key);
  }
  *out_value = field->value;
  return iree_ok_status();
}

static iree_status_t loom_model_string_field(
    const loom_model_json_object_t* object, const char* key, bool required,
    iree_string_view_t* out_value) {
  iree_string_view_t raw;
  IREE_RETURN_IF_ERROR(loom_model_value(
      object, key, IREE_JSON_VALUE_TYPE_STRING, required, &raw));
  *out_value = iree_string_view_empty();
  if (!raw.data) {
    return iree_ok_status();
  }
  return loom_model_string(object->card, raw, out_value);
}

static iree_status_t loom_model_object_field(
    const loom_model_json_object_t* object, const char* key, bool required,
    const char* allowed, loom_model_json_object_t* out_object) {
  iree_string_view_t json;
  IREE_RETURN_IF_ERROR(loom_model_value(
      object, key, IREE_JSON_VALUE_TYPE_OBJECT, required, &json));
  return loom_model_object(object->card, json.data ? json : IREE_SV("{}"),
                           allowed, out_object);
}

static iree_status_t loom_model_choose(loom_model_json_object_t* choices,
                                       iree_string_view_t requested,
                                       const char* label,
                                       iree_string_view_t* out_name,
                                       iree_string_view_t* out_json) {
  const loom_model_json_field_t* field = NULL;
  if (requested.size) {
    field = loom_model_field(choices, requested);
    if (!field) {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown %s '%.*s'", label,
                              (int)requested.size, requested.data);
    }
  } else if (choices->count == 1) {
    field = &choices->fields[0];
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select a %s explicitly (%zu choices)", label,
                            choices->count);
  }
  if (field->type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s '%.*s' must be an object", label,
                            (int)field->key.size, field->key.data);
  }
  *out_name = field->key;
  *out_json = field->value;
  return iree_ok_status();
}

static iree_status_t loom_model_path(iree_string_view_t path) {
  if (!path.size || memchr(path.data, '\\', path.size) ||
      memchr(path.data, ':', path.size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model file paths must be normalized and relative");
  }
  iree_string_view_t remaining = path;
  do {
    iree_string_view_t component;
    int split = iree_string_view_split(remaining, '/', &component, &remaining);
    if (!component.size || iree_string_view_equal(component, IREE_SV(".")) ||
        iree_string_view_equal(component, IREE_SV("..")) ||
        (split >= 0 && !remaining.size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "model path '%.*s' must be normalized and relative", (int)path.size,
          path.data);
    }
  } while (remaining.size);
  return iree_ok_status();
}

static iree_status_t loom_model_file(loom_model_card_t* card,
                                     iree_json_value_type_t type,
                                     iree_string_view_t value, bool provider,
                                     bool parameter,
                                     loom_model_json_object_t* out_object,
                                     loom_model_file_t* out_file) {
  *out_file = (loom_model_file_t){0};
  *out_object = (loom_model_json_object_t){.card = card};
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    IREE_RETURN_IF_ERROR(loom_model_string(card, value, &out_file->path));
  } else if (type == IREE_JSON_VALUE_TYPE_OBJECT) {
    IREE_RETURN_IF_ERROR(
        loom_model_object(card, value,
                          provider    ? "path origin format options data"
                          : parameter ? "path origin format transform"
                                      : "path origin format",
                          out_object));
    IREE_RETURN_IF_ERROR(
        loom_model_string_field(out_object, "path", true, &out_file->path));
    IREE_RETURN_IF_ERROR(loom_model_string_field(out_object, "format", false,
                                                 &out_file->format));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model files require a path string or file object");
  }
  return loom_model_path(out_file->path);
}

typedef struct loom_model_files_parse_t {
  // Destination arena owner.
  loom_model_card_t* card;
  // Preallocated ordered output file array.
  loom_model_file_t* files;
  // Parameter files may retain acquisition operations ignored by runtime IO.
  bool parameter;
} loom_model_files_parse_t;

static iree_status_t loom_model_file_visit(void* self, iree_host_size_t index,
                                           iree_json_value_type_t type,
                                           iree_string_view_t value) {
  loom_model_files_parse_t* parse = self;
  loom_model_json_object_t object;
  return loom_model_file(parse->card, type, value, false, parse->parameter,
                         &object, &parse->files[index]);
}

static iree_status_t loom_model_files(const loom_model_json_object_t* object,
                                      const char* key, bool required,
                                      bool parameter,
                                      loom_model_file_list_t* out_files) {
  iree_string_view_t json;
  IREE_RETURN_IF_ERROR(loom_model_value(object, key, IREE_JSON_VALUE_TYPE_ARRAY,
                                        required, &json));
  *out_files = (loom_model_file_list_t){0};
  if (!json.data) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_json_array_length(json, &out_files->count));
  if (required && !out_files->count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model '%s' requires at least one file", key);
  }
  loom_model_files_parse_t parse = {
      .card = object->card,
      .parameter = parameter,
  };
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&object->card->arena, out_files->count,
                                sizeof(*parse.files), (void**)&parse.files));
  out_files->values = parse.files;
  return iree_json_enumerate_array_typed(json, loom_model_file_visit, &parse);
}

typedef struct loom_model_strings_parse_t {
  // Destination arena owner.
  loom_model_card_t* card;
  // Preallocated ordered output string array.
  iree_string_view_t* values;
} loom_model_strings_parse_t;

static iree_status_t loom_model_string_visit(void* self, iree_host_size_t index,
                                             iree_json_value_type_t type,
                                             iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model name lists require strings");
  }
  loom_model_strings_parse_t* parse = self;
  return loom_model_string(parse->card, value, &parse->values[index]);
}

static iree_status_t loom_model_strings(const loom_model_json_object_t* object,
                                        const char* key, bool required,
                                        iree_string_view_list_t* out_strings) {
  iree_string_view_t json;
  IREE_RETURN_IF_ERROR(loom_model_value(object, key, IREE_JSON_VALUE_TYPE_ARRAY,
                                        required, &json));
  *out_strings = (iree_string_view_list_t){0};
  if (!json.data) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_json_array_length(json, &out_strings->count));
  loom_model_strings_parse_t parse = {.card = object->card};
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&object->card->arena, out_strings->count,
                                sizeof(*parse.values), (void**)&parse.values));
  out_strings->values = parse.values;
  return iree_json_enumerate_array_typed(json, loom_model_string_visit, &parse);
}

typedef struct loom_model_providers_parse_t {
  // Destination arena owner.
  loom_model_card_t* card;
  // Preallocated ordered output providers.
  loom_model_provider_t* values;
} loom_model_providers_parse_t;

static iree_status_t loom_model_provider_visit(void* self,
                                               iree_host_size_t index,
                                               iree_json_value_type_t type,
                                               iree_string_view_t value) {
  loom_model_providers_parse_t* parse = self;
  loom_model_provider_t* provider = &parse->values[index];
  loom_model_json_object_t object;
  IREE_RETURN_IF_ERROR(loom_model_file(parse->card, type, value, true, false,
                                       &object, &provider->file));
  IREE_RETURN_IF_ERROR(
      loom_model_strings(&object, "options", false, &provider->options));
  return loom_model_files(&object, "data", false, false, &provider->data);
}

static iree_status_t loom_model_providers(
    const loom_model_json_object_t* object, const char* key,
    loom_model_provider_list_t* out_providers) {
  iree_string_view_t json;
  IREE_RETURN_IF_ERROR(
      loom_model_value(object, key, IREE_JSON_VALUE_TYPE_ARRAY, true, &json));
  IREE_RETURN_IF_ERROR(iree_json_array_length(json, &out_providers->count));
  loom_model_providers_parse_t parse = {.card = object->card};
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&object->card->arena, out_providers->count,
                                sizeof(*parse.values), (void**)&parse.values));
  out_providers->values = parse.values;
  return iree_json_enumerate_array_typed(json, loom_model_provider_visit,
                                         &parse);
}

static iree_status_t loom_model_parameters(
    const loom_model_json_object_t* variant,
    loom_model_selection_t* selection) {
  loom_model_json_object_t parameters;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(variant, "parameters", false, NULL, &parameters));
  loom_model_parameter_t* values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&variant->card->arena, parameters.count,
                                sizeof(*values), (void**)&values));
  selection->parameters = values;
  selection->parameter_count = parameters.count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < parameters.count && iree_status_is_ok(status); ++i) {
    const loom_model_json_field_t* field = &parameters.fields[i];
    values[i].scope = field->key;
    if (field->type != IREE_JSON_VALUE_TYPE_OBJECT) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameter scopes require index objects");
    } else {
      loom_model_json_object_t index;
      status = loom_model_object(variant->card, field->value, "origin files",
                                 &index);
      if (iree_status_is_ok(status)) {
        status =
            loom_model_files(&index, "files", true, true, &values[i].files);
      }
    }
  }
  return status;
}

static iree_status_t loom_model_tokenizers(
    const loom_model_json_object_t* variant,
    loom_model_selection_t* selection) {
  loom_model_json_object_t tokenizers;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(variant, "tokenizers", false, NULL, &tokenizers));
  loom_model_tokenizer_t* values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&variant->card->arena, tokenizers.count,
                                sizeof(*values), (void**)&values));
  selection->tokenizers = values;
  selection->tokenizer_count = tokenizers.count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < tokenizers.count && iree_status_is_ok(status); ++i) {
    const loom_model_json_field_t* field = &tokenizers.fields[i];
    values[i].name = field->key;
    loom_model_json_object_t file;
    status = loom_model_file(variant->card, field->type, field->value, false,
                             false, &file, &values[i].file);
  }
  return status;
}

static iree_status_t loom_model_voices(const loom_model_json_object_t* variant,
                                       loom_model_selection_t* selection) {
  loom_model_json_object_t voices;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(variant, "voices", false, NULL, &voices));
  loom_model_voice_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &variant->card->arena, voices.count, sizeof(*values), (void**)&values));
  selection->voices = values;
  selection->voice_count = voices.count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < voices.count && iree_status_is_ok(status);
       ++i) {
    const loom_model_json_field_t* field = &voices.fields[i];
    values[i].name = field->key;
    if (field->type != IREE_JSON_VALUE_TYPE_OBJECT) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "voices require recording objects");
      continue;
    }
    loom_model_json_object_t voice;
    status = loom_model_object(variant->card, field->value, "file content_type",
                               &voice);
    const loom_model_json_field_t* file = NULL;
    if (iree_status_is_ok(status)) {
      file = loom_model_field(&voice, IREE_SV("file"));
      if (!file) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "voice recording requires a file");
      }
    }
    if (iree_status_is_ok(status)) {
      loom_model_json_object_t file_object;
      status = loom_model_file(variant->card, file->type, file->value, false,
                               false, &file_object, &values[i].file);
      if (iree_status_is_ok(status) && values[i].file.format.size) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "voice recordings use content_type instead of file.format");
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_model_string_field(&voice, "content_type", false,
                                       &values[i].content_type);
    }
  }
  return status;
}

static iree_status_t loom_model_config_append(
    const loom_model_json_field_t* field, bool* comma,
    iree_string_builder_t* builder) {
  if (field->type == IREE_JSON_VALUE_TYPE_OBJECT ||
      field->type == IREE_JSON_VALUE_TYPE_ARRAY ||
      field->type == IREE_JSON_VALUE_TYPE_NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "model config values require scalar compiler values");
  }
  if (field->type == IREE_JSON_VALUE_TYPE_NUMBER) {
    double number = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_double(field->value, &number));
    if (!isfinite(number)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model config numbers must be finite");
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "%s\"%.*s\":", *comma ? "," : "", (int)field->raw_key.size,
      field->raw_key.data));
  const char* quote = field->type == IREE_JSON_VALUE_TYPE_STRING ? "\"" : "";
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "%s%.*s%s", quote, (int)field->value.size, field->value.data,
      quote));
  *comma = true;
  return iree_ok_status();
}

static iree_status_t loom_model_config(const loom_model_json_object_t* programs,
                                       const loom_model_json_object_t* variant,
                                       iree_string_view_t* out_config) {
  loom_model_json_object_t defaults, overrides;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(programs, "config", false, NULL, &defaults));
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(variant, "config", false, NULL, &overrides));
  iree_string_builder_t builder;
  iree_string_builder_initialize(programs->card->host_allocator, &builder);
  iree_status_t status = iree_string_builder_append_cstring(&builder, "{");
  bool comma = false;
  for (iree_host_size_t i = 0; i < defaults.count && iree_status_is_ok(status);
       ++i) {
    if (!loom_model_field(&overrides, defaults.fields[i].key)) {
      status = loom_model_config_append(&defaults.fields[i], &comma, &builder);
    }
  }
  for (iree_host_size_t i = 0; i < overrides.count && iree_status_is_ok(status);
       ++i) {
    status = loom_model_config_append(&overrides.fields[i], &comma, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "}");
  }
  if (iree_status_is_ok(status)) {
    status = loom_model_copy(programs->card, iree_string_builder_view(&builder),
                             out_config);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

// The shared JSON reader also accepts comments and trailing commas. Cards use
// plain JSON so jq, schema tooling, and every host consume the same language.
static iree_status_t loom_model_json_syntax(iree_string_view_t json) {
  bool quoted = false;
  bool escaped = false;
  char previous = 0;
  for (iree_host_size_t i = 0; i < json.size; ++i) {
    const char c = json.data[i];
    if (quoted) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        quoted = false;
      }
    } else if (c == '"') {
      quoted = true;
    } else if (c == '/' || ((c == '}' || c == ']') && previous == ',')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model cards require JSON without comments or "
                              "trailing commas");
    }
    if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
      previous = c;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_model_identifier(iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const char c = name.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
          (i && c >= '0' && c <= '9'))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "model service kind and component role must be a symbol identifier");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_model_version(
    const loom_model_json_object_t* definition, uint32_t* out_version) {
  iree_string_view_t version;
  IREE_RETURN_IF_ERROR(loom_model_value(
      definition, "version", IREE_JSON_VALUE_TYPE_NUMBER, false, &version));
  *out_version = 0;
  if (version.data) {
    uint64_t number = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(version, &number));
    if (!number || number > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "model service version must be in [1, 2^32-1]");
    }
    *out_version = (uint32_t)number;
  }
  return iree_ok_status();
}

static iree_status_t loom_model_settings(
    const loom_model_json_object_t* definition,
    const loom_model_json_object_t* preset, iree_string_view_t* out_options,
    iree_string_view_t* out_execution) {
  iree_string_view_list_t accepted_options;
  IREE_RETURN_IF_ERROR(
      loom_model_strings(definition, "options", false, &accepted_options));
  loom_model_json_object_t options;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(preset, "options", false, NULL, &options));
  for (iree_host_size_t i = 0; i < options.count; ++i) {
    bool accepted = false;
    for (iree_host_size_t j = 0; j < accepted_options.count; ++j) {
      accepted |= iree_string_view_equal(options.fields[i].key,
                                         accepted_options.values[j]);
    }
    if (!accepted) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected service does not accept option '%.*s'",
                              (int)options.fields[i].key.size,
                              options.fields[i].key.data);
    }
  }
  IREE_RETURN_IF_ERROR(loom_model_value(
      preset, "options", IREE_JSON_VALUE_TYPE_OBJECT, false, out_options));
  IREE_RETURN_IF_ERROR(loom_model_value(
      preset, "execution", IREE_JSON_VALUE_TYPE_OBJECT, false, out_execution));
  if (!out_options->data) {
    *out_options = IREE_SV("{}");
  }
  if (!out_execution->data) {
    *out_execution = IREE_SV("{}");
  }
  return iree_ok_status();
}

static iree_status_t loom_model_components(
    const loom_model_json_object_t* service,
    const loom_model_json_object_t* preset, loom_model_selection_t* selection) {
  loom_model_json_object_t definitions, settings;
  IREE_RETURN_IF_ERROR(loom_model_object_field(service, "components", false,
                                               NULL, &definitions));
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(preset, "components", false, NULL, &settings));
  for (iree_host_size_t i = 0; i < settings.count; ++i) {
    if (!loom_model_field(&definitions, settings.fields[i].key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "preset names undeclared component '%.*s'",
                              (int)settings.fields[i].key.size,
                              settings.fields[i].key.data);
    }
  }
  loom_model_card_t* card = service->card;
  loom_model_component_t* components = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&card->arena, definitions.count,
                                sizeof(*components), (void**)&components));
  for (iree_host_size_t i = 0; i < definitions.count; ++i) {
    loom_model_component_t* component = &components[i];
    component->name = definitions.fields[i].key;
    IREE_RETURN_IF_ERROR(loom_model_identifier(component->name));
    loom_model_json_object_t definition, values;
    char* prefix = NULL;
    iree_host_size_t prefix_length = 0;
    if (!iree_host_size_checked_add(selection->kind.size, component->name.size,
                                    &prefix_length) ||
        !iree_host_size_checked_add(prefix_length, 1, &prefix_length)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "component symbol namespace exceeds host storage");
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&card->arena, prefix_length, (void**)&prefix));
    memcpy(prefix, selection->kind.data, selection->kind.size);
    prefix[selection->kind.size] = '_';
    memcpy(prefix + selection->kind.size + 1, component->name.data,
           component->name.size);
    component->export_prefix = iree_make_string_view(prefix, prefix_length);
    if (definitions.fields[i].type != IREE_JSON_VALUE_TYPE_OBJECT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "component definition must be an object");
    }
    IREE_RETURN_IF_ERROR(loom_model_object(
        card, definitions.fields[i].value,
        "version options interface constraints", &definition));
    const loom_model_json_field_t* setting =
        loom_model_field(&settings, component->name);
    if (setting && setting->type != IREE_JSON_VALUE_TYPE_OBJECT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "component preset must be an object");
    }
    IREE_RETURN_IF_ERROR(
        loom_model_object(card, setting ? setting->value : IREE_SV("{}"),
                          "options execution", &values));
    IREE_RETURN_IF_ERROR(loom_model_version(&definition, &component->version));
    IREE_RETURN_IF_ERROR(loom_model_settings(
        &definition, &values, &component->options, &component->execution));
  }
  selection->component_count = definitions.count;
  selection->components = components;
  return iree_ok_status();
}

static iree_status_t loom_model_select(loom_model_card_t* card,
                                       iree_string_view_t json,
                                       const loom_model_selector_t* selector) {
  IREE_RETURN_IF_ERROR(loom_model_json_syntax(json));
  IREE_RETURN_IF_ERROR(loom_model_copy(card, json, &json));
  iree_string_view_t document;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&json, &document));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&json));
  if (json.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing model card JSON");
  }
  loom_model_json_object_t root;
  IREE_RETURN_IF_ERROR(loom_model_object(card, document, NULL, &root));
  iree_string_view_t version;
  IREE_RETURN_IF_ERROR(loom_model_value(
      &root, "schema_version", IREE_JSON_VALUE_TYPE_NUMBER, true, &version));
  uint64_t schema_version = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(version, &schema_version));
  if (schema_version != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "model schema version %" PRIu64
                            " is unsupported; this host supports version 1",
                            schema_version);
  }
  IREE_RETURN_IF_ERROR(loom_model_check_fields(
      &root,
      "$schema schema_version model origins programs services variants "
      "presets"));
  loom_model_selection_t* selection = &card->selection;
  loom_model_json_object_t metadata;
  IREE_RETURN_IF_ERROR(loom_model_object_field(
      &root, "model", true, "id name description license origin implementation",
      &metadata));
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&metadata, "id", true, &selection->model_id));
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&metadata, "name", true, &selection->model_name));
  loom_model_json_object_t presets;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(&root, "presets", false, NULL, &presets));
  iree_string_view_t preset_json = IREE_SV("{}");
  if (selector->preset.size ||
      (!selector->service.size && !selector->variant.size &&
       !selector->programs.size && !selector->partition.size &&
       presets.count)) {
    IREE_RETURN_IF_ERROR(loom_model_choose(&presets, selector->preset, "preset",
                                           &selection->selector.preset,
                                           &preset_json));
  }
  loom_model_json_object_t preset;
  IREE_RETURN_IF_ERROR(loom_model_object(
      card, preset_json,
      "service variant programs partition options execution request workload "
      "memory components",
      &preset));
  loom_model_selector_t defaults = {0};
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&preset, "service", false, &defaults.service));
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&preset, "variant", false, &defaults.variant));
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&preset, "programs", false, &defaults.programs));
  IREE_RETURN_IF_ERROR(loom_model_string_field(&preset, "partition", false,
                                               &defaults.partition));
  loom_model_json_object_t variants;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(&root, "variants", true, NULL, &variants));
  iree_string_view_t variant_json;
  IREE_RETURN_IF_ERROR(loom_model_choose(
      &variants, selector->variant.size ? selector->variant : defaults.variant,
      "variant", &selection->selector.variant, &variant_json));
  loom_model_json_object_t variant, service_variants;
  IREE_RETURN_IF_ERROR(loom_model_object(
      card, variant_json, "description services partitions", &variant));
  iree_string_view_t partition_json, services_json;
  IREE_RETURN_IF_ERROR(loom_model_value(&variant, "partitions",
                                        IREE_JSON_VALUE_TYPE_OBJECT, false,
                                        &partition_json));
  IREE_RETURN_IF_ERROR(loom_model_value(&variant, "services",
                                        IREE_JSON_VALUE_TYPE_OBJECT, false,
                                        &services_json));
  if (!!partition_json.data == !!services_json.data) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "model variant requires exactly one of services or partitions");
  }
  iree_string_view_t partition_name =
      selector->partition.size ? selector->partition : defaults.partition;
  if (partition_json.data) {
    loom_model_json_object_t partitions;
    IREE_RETURN_IF_ERROR(
        loom_model_object(card, partition_json, NULL, &partitions));
    IREE_RETURN_IF_ERROR(
        loom_model_choose(&partitions, partition_name, "partition",
                          &selection->selector.partition, &partition_json));
    IREE_RETURN_IF_ERROR(loom_model_object(card, partition_json,
                                           "description services", &variant));
  } else if (partition_name.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected model variant is unpartitioned");
  }
  IREE_RETURN_IF_ERROR(loom_model_object_field(&variant, "services", true, NULL,
                                               &service_variants));
  iree_string_view_t service_variant_json;
  IREE_RETURN_IF_ERROR(loom_model_choose(
      &service_variants,
      selector->service.size ? selector->service : defaults.service, "service",
      &selection->selector.service, &service_variant_json));
  loom_model_json_object_t service_variant;
  IREE_RETURN_IF_ERROR(loom_model_object(
      card, service_variant_json,
      "programs config parameters tokenizers voices", &service_variant));
  loom_model_json_object_t services;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(&root, "services", true, NULL, &services));
  iree_string_view_t service_json;
  IREE_RETURN_IF_ERROR(
      loom_model_choose(&services, selection->selector.service, "service",
                        &selection->selector.service, &service_json));
  loom_model_json_object_t service;
  IREE_RETURN_IF_ERROR(loom_model_object(
      card, service_json,
      "kind version options interface constraints components", &service));
  IREE_RETURN_IF_ERROR(
      loom_model_string_field(&service, "kind", true, &selection->kind));
  IREE_RETURN_IF_ERROR(loom_model_identifier(selection->kind));
  IREE_RETURN_IF_ERROR(loom_model_version(&service, &selection->version));
  IREE_RETURN_IF_ERROR(loom_model_components(&service, &preset, selection));
  iree_string_view_list_t compatible;
  IREE_RETURN_IF_ERROR(
      loom_model_strings(&service_variant, "programs", true, &compatible));
  iree_string_view_t programs_name =
      selector->programs.size ? selector->programs : defaults.programs;
  if (!programs_name.size && compatible.count == 1) {
    programs_name = compatible.values[0];
  }
  bool found = false;
  for (iree_host_size_t i = 0; i < compatible.count; ++i) {
    found |= iree_string_view_equal(programs_name, compatible.values[i]);
  }
  if (!found) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "select a compatible program set for service '%.*s'",
        (int)selection->selector.service.size,
        selection->selector.service.data);
  }
  loom_model_json_object_t program_sets;
  IREE_RETURN_IF_ERROR(
      loom_model_object_field(&root, "programs", true, NULL, &program_sets));
  iree_string_view_t programs_json;
  IREE_RETURN_IF_ERROR(
      loom_model_choose(&program_sets, programs_name, "program set",
                        &selection->selector.programs, &programs_json));
  loom_model_json_object_t programs;
  IREE_RETURN_IF_ERROR(
      loom_model_object(card, programs_json,
                        "origin directory inputs libraries config", &programs));
  IREE_RETURN_IF_ERROR(loom_model_string_field(&programs, "directory", false,
                                               &selection->directory));
  if (!selection->directory.size) {
    selection->directory = IREE_SV(".");
  }
  if (!iree_string_view_equal(selection->directory, IREE_SV("."))) {
    IREE_RETURN_IF_ERROR(loom_model_path(selection->directory));
  }
  IREE_RETURN_IF_ERROR(
      loom_model_providers(&programs, "inputs", &selection->inputs));
  IREE_RETURN_IF_ERROR(
      loom_model_providers(&programs, "libraries", &selection->libraries));
  if (!selection->inputs.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model program sets require at least one input");
  }
  IREE_RETURN_IF_ERROR(
      loom_model_config(&programs, &service_variant, &selection->config));
  IREE_RETURN_IF_ERROR(loom_model_parameters(&service_variant, selection));
  IREE_RETURN_IF_ERROR(loom_model_tokenizers(&service_variant, selection));
  IREE_RETURN_IF_ERROR(loom_model_voices(&service_variant, selection));
  IREE_RETURN_IF_ERROR(loom_model_settings(
      &service, &preset, &selection->options, &selection->execution));
  IREE_RETURN_IF_ERROR(loom_model_value(&preset, "request",
                                        IREE_JSON_VALUE_TYPE_OBJECT, false,
                                        &selection->request));
  if (!selection->request.data) {
    selection->request = IREE_SV("{}");
  }
  return iree_ok_status();
}

iree_status_t loom_model_card_parse(iree_string_view_t json,
                                    const loom_model_selector_t* selector,
                                    iree_allocator_t host_allocator,
                                    loom_model_card_t** out_card) {
  *out_card = NULL;
  loom_model_card_t* card = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*card), (void**)&card));
  iree_atomic_ref_count_init(&card->ref_count);
  card->host_allocator = host_allocator;
  iree_arena_block_pool_initialize(4096, host_allocator, &card->block_pool);
  iree_arena_initialize(&card->block_pool, &card->arena);
  const loom_model_selector_t empty_selector = {0};
  iree_status_t status =
      loom_model_select(card, json, selector ? selector : &empty_selector);
  if (iree_status_is_ok(status)) {
    *out_card = card;
  } else {
    loom_model_card_release(card);
  }
  return status;
}

void loom_model_card_retain(loom_model_card_t* card) {
  if (card) {
    iree_atomic_ref_count_inc(&card->ref_count);
  }
}

void loom_model_card_release(loom_model_card_t* card) {
  if (card && iree_atomic_ref_count_dec(&card->ref_count) == 1) {
    iree_arena_deinitialize(&card->arena);
    iree_arena_block_pool_deinitialize(&card->block_pool);
    iree_allocator_free(card->host_allocator, card);
  }
}

const loom_model_selection_t* loom_model_card_selection(
    const loom_model_card_t* card) {
  return &card->selection;
}

const loom_model_parameter_t* loom_model_selection_parameter(
    const loom_model_selection_t* selection, iree_string_view_t scope) {
  for (iree_host_size_t i = 0; i < selection->parameter_count; ++i) {
    if (iree_string_view_equal(selection->parameters[i].scope, scope)) {
      return &selection->parameters[i];
    }
  }
  return NULL;
}

const loom_model_tokenizer_t* loom_model_selection_tokenizer(
    const loom_model_selection_t* selection, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < selection->tokenizer_count; ++i) {
    if (iree_string_view_equal(selection->tokenizers[i].name, name)) {
      return &selection->tokenizers[i];
    }
  }
  return NULL;
}

const loom_model_component_t* loom_model_selection_component(
    const loom_model_selection_t* selection, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < selection->component_count; ++i) {
    if (iree_string_view_equal(selection->components[i].name, name)) {
      return &selection->components[i];
    }
  }
  return NULL;
}
