// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/decision/protocol.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

enum { DECISION_MAX_QUESTIONS = 1024, DECISION_MAX_OPTIONS = 255 };

typedef enum decision_kind_e {
  DECISION_CHOICE = 0,
  DECISION_SCORE = 1,
  DECISION_NOUL = 2,
} decision_kind_t;

typedef struct decision_question_t {
  // Owned decoded question ID; source sees its original escaped range.
  iree_string_view_t id;
  // Protocol answer family, independent of the model architecture.
  decision_kind_t kind;
  // First option in the request's contiguous option table.
  iree_host_size_t first_option;
  // Number of options and expected probabilities for this question.
  iree_host_size_t option_count;
} decision_question_t;

typedef struct decision_option_t {
  // Owned decoded choice ID; empty for implicit score/noul ordinals.
  iree_string_view_t id;
  // Criterion type, preserving structured score legends.
  iree_json_value_type_t type;
  // Criterion range in original JSON, before the final packet data offset.
  iree_host_size_t offset;
  // Criterion bytes; strings exclude quotes and retain escapes.
  iree_host_size_t length;
} decision_option_t;

typedef struct decision_request_t {
  // Initial shared request member, including the owned source packet.
  loom_serve_finite_request_t base;
  // Allocator owning decoded IDs and native association tables.
  iree_allocator_t allocator;
  // Owned initialized question records in original request order.
  decision_question_t* questions;
  // Owned option records in each question's original public order.
  decision_option_t* options;
  // Number of initialized option records.
  iree_host_size_t option_count;
  // Borrowed original JSON inside base.input, stable through encoding.
  iree_string_view_t json;
} decision_request_t;

typedef struct decision_field_t {
  // Literal schema key; escaped schema keys are not part of this profile.
  const char* name;
  // Permitted JSON types as a bitmask.
  uint32_t types;
  // Observed type after public validation.
  iree_json_value_type_t type;
  // Borrowed original value; null data means absent.
  iree_string_view_t value;
} decision_field_t;

typedef struct decision_fields_t {
  // Number of declared fields.
  iree_host_size_t count;
  // Borrowed mutable destinations.
  decision_field_t* fields;
} decision_fields_t;

static iree_status_t decision_field(void* self, iree_string_view_t key,
                                    iree_json_value_type_t type,
                                    iree_string_view_t value) {
  decision_fields_t* fields = self;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    decision_field_t* field = &fields->fields[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || !(field->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed field '%s'",
                              field->name);
    }
    field->type = type;
    field->value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported decision field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t decision_fields(iree_string_view_t object,
                                     iree_host_size_t count,
                                     decision_field_t* fields) {
  decision_fields_t state = {count, fields};
  return iree_json_enumerate_object_typed(object, decision_field, &state);
}

static iree_status_t decision_id(iree_string_view_t raw,
                                 iree_string_view_t* output,
                                 iree_allocator_t allocator) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, raw.size + 1, (void**)&data));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(raw, raw.size, data, &length);
  const iree_string_view_t id = iree_make_string_view(data, length);
  if (iree_status_is_ok(status) &&
      (!length || !iree_unicode_utf8_validate(id))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "IDs must be nonempty UTF-8");
  }
  if (iree_status_is_ok(status)) {
    *output = id;
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

// Duplicate detection sorts borrowed views, never the client association table.
static int decision_compare_ids(const void* lhs, const void* rhs) {
  return iree_string_view_compare(*(const iree_string_view_t*)lhs,
                                  *(const iree_string_view_t*)rhs);
}

static iree_status_t decision_unique(iree_host_size_t count,
                                     iree_string_view_t* ids) {
  qsort(ids, count, sizeof(*ids), decision_compare_ids);
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (iree_string_view_equal(ids[i - 1], ids[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate decoded ID");
    }
  }
  return iree_ok_status();
}

static iree_status_t decision_record(iree_string_builder_t* builder,
                                     iree_host_size_t count,
                                     const uint64_t* values) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_inline(builder, count * 8, &data));
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u64(data + i * 8, values[i]);
  }
  return iree_ok_status();
}

typedef struct decision_parser_t {
  // Borrowed complete request JSON during synchronous initialization.
  iree_string_view_t body;
  // Owned request under construction.
  decision_request_t* request;
  // Native option records under construction; ownership transfers even on
  // error.
  iree_string_builder_t options;
  // Indexed source question records with JSON-relative offsets.
  iree_string_builder_t question_records;
  // Indexed source option records with JSON-relative offsets.
  iree_string_builder_t option_records;
} decision_parser_t;

static iree_status_t decision_option(decision_parser_t* parser,
                                     iree_string_view_t key,
                                     iree_json_value_type_t type,
                                     iree_string_view_t value) {
  decision_request_t* request = parser->request;
  decision_option_t* option = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(
      &parser->options, sizeof(*option), (char**)&option));
  *option = (decision_option_t){
      .type = type,
      .offset =
          value.data ? (iree_host_size_t)(value.data - parser->body.data) : 0,
      .length = value.size};
  ++request->option_count;
  if (key.data) {
    IREE_RETURN_IF_ERROR(decision_id(key, &option->id, request->allocator));
  }
  const uint64_t record[5] = {
      key.data ? (uint64_t)(key.data - parser->body.data) : 0, key.size,
      option->offset, option->length, type};
  return decision_record(&parser->option_records, IREE_ARRAYSIZE(record),
                         record);
}

// Choice permits null rubrics; all other textual schema values preserve a
// string, object or array, never an implicit numeric/bool-to-string conversion.
static bool decision_is_content(iree_json_value_type_t type) {
  return type == IREE_JSON_VALUE_TYPE_STRING ||
         type == IREE_JSON_VALUE_TYPE_OBJECT ||
         type == IREE_JSON_VALUE_TYPE_ARRAY;
}

static iree_status_t decision_choice_option(void* self, iree_string_view_t key,
                                            iree_json_value_type_t type,
                                            iree_string_view_t value) {
  decision_parser_t* parser = self;
  const decision_question_t* question =
      &parser->request->questions[parser->request->base.count - 1];
  if ((!decision_is_content(type) && type != IREE_JSON_VALUE_TYPE_NULL) ||
      parser->request->option_count - question->first_option ==
          DECISION_MAX_OPTIONS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "choice requires at most 255 string/object/array/null options");
  }
  return decision_option(parser, key, type, value);
}

static iree_status_t decision_score_option(void* self, iree_host_size_t index,
                                           iree_json_value_type_t type,
                                           iree_string_view_t value) {
  if (!decision_is_content(type) || index == 10) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "score requires 2-10 string/object/array levels");
  }
  return decision_option(self, iree_string_view_empty(), type, value);
}

static iree_status_t decision_question(void* self, iree_string_view_t key,
                                       iree_json_value_type_t type,
                                       iree_string_view_t value) {
  (void)type;
  decision_parser_t* parser = self;
  decision_request_t* request = parser->request;
  decision_question_t* question = &request->questions[request->base.count++];
  IREE_RETURN_IF_ERROR(decision_id(key, &question->id, request->allocator));
  const uint32_t content = (1u << IREE_JSON_VALUE_TYPE_STRING) |
                           (1u << IREE_JSON_VALUE_TYPE_OBJECT) |
                           (1u << IREE_JSON_VALUE_TYPE_ARRAY);
  decision_field_t fields[] = {
      {.name = "type", .types = 1u << IREE_JSON_VALUE_TYPE_STRING},
      {.name = "instructions", .types = content},
      {.name = "criteria",
       .types = (1u << IREE_JSON_VALUE_TYPE_OBJECT) |
                (1u << IREE_JSON_VALUE_TYPE_ARRAY)}};
  IREE_RETURN_IF_ERROR(decision_fields(value, IREE_ARRAYSIZE(fields), fields));
  if (!fields[0].value.data || !fields[1].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "questions require type and instructions");
  }
  char name[6];
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(fields[0].value, 0, NULL, &length));
  if (length > sizeof(name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported question type");
  }
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(fields[0].value, sizeof(name), name, &length));
  const iree_string_view_t kind = iree_make_string_view(name, length);
  question->first_option = request->option_count;
  if (iree_string_view_equal(kind, IREE_SV("choice"))) {
    question->kind = DECISION_CHOICE;
    if (!fields[2].value.data ||
        fields[2].type != IREE_JSON_VALUE_TYPE_OBJECT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "choice requires criteria map");
    }
    IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
        fields[2].value, decision_choice_option, parser));
    const iree_host_size_t count =
        request->option_count - question->first_option;
    if (!count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "choice requires at least one option");
    }
    const decision_option_t* options =
        (const decision_option_t*)iree_string_builder_buffer(&parser->options);
    iree_string_view_t ids[DECISION_MAX_OPTIONS];
    for (iree_host_size_t i = 0; i < count; ++i) {
      ids[i] = options[question->first_option + i].id;
    }
    IREE_RETURN_IF_ERROR(decision_unique(count, ids));
  } else if (iree_string_view_equal(kind, IREE_SV("score"))) {
    question->kind = DECISION_SCORE;
    if (!fields[2].value.data || fields[2].type != IREE_JSON_VALUE_TYPE_ARRAY) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "score requires criteria array");
    }
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[2].value, decision_score_option, parser));
    if (request->option_count - question->first_option < 2) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "score requires at least two levels");
    }
  } else if (iree_string_view_equal(kind, IREE_SV("noul"))) {
    question->kind = DECISION_NOUL;
    decision_field_t criteria[] = {{.name = "false", .types = content},
                                   {.name = "true", .types = content}};
    if (fields[2].value.data) {
      if (fields[2].type != IREE_JSON_VALUE_TYPE_OBJECT) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "noul criteria must be an object");
      }
      IREE_RETURN_IF_ERROR(
          decision_fields(fields[2].value, IREE_ARRAYSIZE(criteria), criteria));
    }
    IREE_RETURN_IF_ERROR(decision_option(
        parser, iree_string_view_empty(),
        criteria[0].value.data ? criteria[0].type : IREE_JSON_VALUE_TYPE_NULL,
        criteria[0].value));
    IREE_RETURN_IF_ERROR(decision_option(
        parser, iree_string_view_empty(),
        criteria[1].value.data ? criteria[1].type : IREE_JSON_VALUE_TYPE_NULL,
        criteria[1].value));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported question type");
  }
  question->option_count = request->option_count - question->first_option;
  const uint64_t record[8] = {
      (uint64_t)(key.data - parser->body.data),
      key.size,
      question->kind,
      (uint64_t)(fields[1].value.data - parser->body.data),
      fields[1].value.size,
      fields[1].type,
      question->first_option,
      question->option_count};
  return decision_record(&parser->question_records, IREE_ARRAYSIZE(record),
                         record);
}

static iree_status_t decision_count(void* self, iree_string_view_t key,
                                    iree_json_value_type_t type,
                                    iree_string_view_t value) {
  (void)key;
  (void)value;
  iree_host_size_t* count = self;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT || *count == DECISION_MAX_QUESTIONS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "questions requires at most 1024 objects");
  }
  ++*count;
  return iree_ok_status();
}

static iree_status_t decision_packet(decision_parser_t* parser,
                                     const decision_field_t* state) {
  decision_request_t* request = parser->request;
  const iree_string_view_t questions =
      iree_string_builder_view(&parser->question_records);
  const iree_string_view_t options =
      iree_string_builder_view(&parser->option_records);
  const iree_host_size_t offset = 32 + questions.size + options.size;
  const iree_host_size_t length = offset + parser->body.size;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(length, 8, request->allocator, &buffer));
  iree_byte_span_t bytes;
  iree_status_t status = iree_vm_buffer_map_write(buffer, 0, length, &bytes);
  if (iree_status_is_ok(status)) {
    iree_unaligned_store_le_u64(
        bytes.data, offset + (state->value.data - parser->body.data));
    iree_unaligned_store_le_u64(bytes.data + 8, state->value.size);
    iree_unaligned_store_le_u64(bytes.data + 16, state->type);
    iree_unaligned_store_le_u64(bytes.data + 24, request->base.count);
    memcpy(bytes.data + 32, questions.data, questions.size);
    memcpy(bytes.data + 32 + questions.size, options.data, options.size);
    memcpy(bytes.data + offset, parser->body.data, parser->body.size);
    for (iree_host_size_t i = 0; i < request->base.count; ++i) {
      uint8_t* record = bytes.data + 32 + i * 64;
      iree_unaligned_store_le_u64(record,
                                  offset + iree_unaligned_load_le_u64(record));
      iree_unaligned_store_le_u64(
          record + 24, offset + iree_unaligned_load_le_u64(record + 24));
    }
    for (iree_host_size_t i = 0; i < request->option_count; ++i) {
      uint8_t* record = bytes.data + 32 + questions.size + i * 40;
      iree_unaligned_store_le_u64(record,
                                  offset + iree_unaligned_load_le_u64(record));
      iree_unaligned_store_le_u64(
          record + 16, offset + iree_unaligned_load_le_u64(record + 16));
    }
    status = iree_vm_buffer_subspan(buffer, 0, length,
                                    IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                    request->allocator, &request->base.input);
    if (iree_status_is_ok(status)) {
      request->json = iree_make_string_view((const char*)bytes.data + offset,
                                            parser->body.size);
    }
  }
  iree_vm_buffer_release(buffer);
  return status;
}

// IREE's parser also accepts JSONC. This wire profile preserves structured
// rubric values in response legends, so their original bytes must be strict
// JSON: no comments/trailing commas or invalid decoded Unicode scalars.
static iree_status_t decision_require_json(iree_string_view_t body) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "JSON must be UTF-8");
  }
  char previous = 0;
  while (body.size) {
    const char c = body.data[0];
    if (c == '"') {
      iree_string_view_t value;
      IREE_RETURN_IF_ERROR(iree_json_consume_string(&body, &value));
      iree_host_size_t length = 0;
      IREE_RETURN_IF_ERROR(iree_json_unescape_string(value, 0, NULL, &length));
      previous = '"';
      continue;
    }
    if (c == '/' || ((c == '}' || c == ']') && previous == ',')) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "SystemOne requires JSON without comments or trailing commas");
    }
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
      previous = c;
    }
    body = iree_string_view_substr(body, 1, IREE_HOST_SIZE_MAX);
  }
  return iree_ok_status();
}

static iree_status_t decision_parse(decision_parser_t* parser) {
  decision_request_t* request = parser->request;
  IREE_RETURN_IF_ERROR(decision_require_json(parser->body));
  iree_string_view_t cursor = parser->body, object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing request JSON");
  }
  decision_field_t fields[] = {
      {.name = "model", .types = 1u << IREE_JSON_VALUE_TYPE_STRING},
      {.name = "state",
       .types = (1u << IREE_JSON_VALUE_TYPE_STRING) |
                (1u << IREE_JSON_VALUE_TYPE_OBJECT) |
                (1u << IREE_JSON_VALUE_TYPE_ARRAY)},
      {.name = "questions", .types = 1u << IREE_JSON_VALUE_TYPE_OBJECT}};
  IREE_RETURN_IF_ERROR(decision_fields(object, IREE_ARRAYSIZE(fields), fields));
  if (!fields[0].value.data || !fields[1].value.data || !fields[2].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model, state and questions are required");
  }
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      fields[2].value, decision_count, &count));
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "questions must not be empty");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      request->allocator, count, sizeof(*request->questions),
      (void**)&request->questions));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      fields[2].value, decision_question, parser));
  iree_string_view_t* ids = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(request->allocator, count,
                                                   sizeof(*ids), (void**)&ids));
  for (iree_host_size_t i = 0; i < count; ++i) {
    ids[i] = request->questions[i].id;
  }
  iree_status_t status = decision_unique(count, ids);
  iree_allocator_free(request->allocator, ids);
  if (iree_status_is_ok(status)) {
    status = decision_packet(parser, &fields[1]);
  }
  return status;
}

static void decision_deinitialize(loom_serve_finite_request_t* base) {
  decision_request_t* request = (decision_request_t*)base;
  for (iree_host_size_t i = 0; i < base->count; ++i) {
    iree_allocator_free(request->allocator,
                        (void*)request->questions[i].id.data);
  }
  for (iree_host_size_t i = 0; i < request->option_count; ++i) {
    iree_allocator_free(request->allocator, (void*)request->options[i].id.data);
  }
  iree_allocator_free(request->allocator, request->questions);
  iree_allocator_free(request->allocator, request->options);
  iree_vm_buffer_release(base->input);
  memset(request, 0, sizeof(*request));
}

static iree_status_t decision_initialize(iree_string_view_t body,
                                         loom_serve_finite_request_t* base,
                                         iree_allocator_t allocator) {
  decision_request_t* request = (decision_request_t*)base;
  request->allocator = allocator;
  decision_parser_t parser = {.body = body, .request = request};
  iree_string_builder_initialize(allocator, &parser.options);
  iree_string_builder_initialize(allocator, &parser.question_records);
  iree_string_builder_initialize(allocator, &parser.option_records);
  iree_status_t status = decision_parse(&parser);
  request->options =
      (decision_option_t*)iree_string_builder_take_storage(&parser.options);
  iree_string_builder_deinitialize(&parser.options);
  iree_string_builder_deinitialize(&parser.question_records);
  iree_string_builder_deinitialize(&parser.option_records);
  return status;
}

static double decision_probability(iree_const_byte_span_t row,
                                   iree_host_size_t index) {
  return (double)iree_unaligned_load_le_f32(row.data + index * 4);
}

static iree_status_t decision_answer(const decision_request_t* request,
                                     const decision_question_t* question,
                                     iree_const_byte_span_t row,
                                     iree_string_builder_t* output) {
  const iree_host_size_t count = question->option_count;
  if (row.data_length != count * 4) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "source probability count does not match question");
  }
  double sum = 0, score = 0;
  iree_host_size_t mode = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const double p = decision_probability(row, i);
    if (!isfinite(p) || p < 0 || p > 1) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "source produced invalid probability");
    }
    sum += p;
    score += p * i;
    if (p > decision_probability(row, mode)) {
      mode = i;
    }
  }
  // The source softmax is F32. Permit accumulated rounding, not a host repair
  // of logits or an unnormalized head; all values are encoded unchanged.
  if (fabs(sum - 1.0) > 4.0 * FLT_EPSILON * count) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "source probabilities are not normalized");
  }
  if (question->kind == DECISION_NOUL) {
    return iree_string_builder_append_format(
        output, "{\"type\":\"noul\",\"noul\":%.9g}",
        decision_probability(row, 1));
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  double confidence = 1;
  if (question->kind == DECISION_CHOICE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output, "{\"type\":\"choice\",\"choice\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, request->options[question->first_option + mode].id));
    if (count > 1) {
      confidence =
          fmax(0, (decision_probability(row, mode) * count - 1) / (count - 1));
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "{\"type\":\"score\",\"score\":%.9g,\"legend\":{", score));
    double spread = 0, uniform_spread = 0;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      const decision_option_t* option =
          &request->options[question->first_option + i];
      spread += decision_probability(row, i) * fabs((double)i - (double)mode);
      uniform_spread += fabs((double)i - (double)(count - 1) / 2.0) / count;
      const char* quote =
          option->type == IREE_JSON_VALUE_TYPE_STRING ? "\"" : "";
      status = iree_string_builder_append_format(
          output, "%s\"%zu\":%s%.*s%s", i ? "," : "", i, quote,
          (int)option->length, request->json.data + option->offset, quote);
    }
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
    confidence = fmax(0, 1 - spread / uniform_spread);
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"confidence\":%.9g,\"probabilities\":{", confidence));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    if (i) {
      status = iree_string_builder_append_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      if (question->kind == DECISION_CHOICE) {
        status = loom_json_write_escaped_string(
            &stream, request->options[question->first_option + i].id);
      } else {
        status = iree_string_builder_append_format(output, "\"%zu\"", i);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(output, ":%.9g",
                                                 decision_probability(row, i));
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "}}");
  }
  return status;
}

static iree_status_t decision_response(const loom_serve_finite_request_t* base,
                                       iree_string_view_t model,
                                       const iree_const_byte_span_t* rows,
                                       uint64_t input_tokens,
                                       iree_string_builder_t* output) {
  const decision_request_t* request = (const decision_request_t*)base;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"model\":"));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, model));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"answers\":{"));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < base->count && iree_status_is_ok(status);
       ++i) {
    if (i) {
      status = iree_string_builder_append_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_json_write_escaped_string(&stream, request->questions[i].id);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, ":");
    }
    if (iree_status_is_ok(status)) {
      status =
          decision_answer(request, &request->questions[i], rows[i], output);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        output,
        "},\"usage\":{\"input_tokens\":%" PRIu64 ",\"output_tokens\":0}}",
        input_tokens);
  }
  return status;
}

const loom_serve_finite_protocol_t* loom_serve_decision_protocol(void) {
  static const loom_serve_finite_protocol_t protocol = {
      .name = "decision",
      .endpoint = "/v1/systemone",
      .invalid_status = "422 Unprocessable Entity",
      .request_size = sizeof(decision_request_t),
      .initialize = decision_initialize,
      .deinitialize = decision_deinitialize,
      .response = decision_response};
  return &protocol;
}
