// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/configuration.h"

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/io/file_contents.h"

iree_status_t loom_serve_realtime_configuration_initialize(
    iree_string_view_t document, iree_allocator_t host_allocator,
    loom_serve_realtime_configuration_t* configuration) {
  *configuration = (loom_serve_realtime_configuration_t){0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(configuration->strings);
       ++i) {
    iree_string_builder_initialize(host_allocator, &configuration->strings[i]);
  }
  if (!iree_unicode_utf8_validate(document)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "assistant document requires UTF-8");
  }
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&document));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&document, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&document));
  if (document.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing assistant document content");
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"name", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"text", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"recognition", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"speech", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"voice", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"instructions", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"max_output_tokens", 1u << IREE_JSON_VALUE_TYPE_NUMBER},
      {"seed", 1u << IREE_JSON_VALUE_TYPE_NUMBER}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  if ((present & 31) != 31) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "assistant requires name, text, recognition, speech, and voice");
  }
  iree_string_view_t* fields[] = {&configuration->profile.name,
                                  &configuration->profile.text,
                                  &configuration->profile.recognition,
                                  &configuration->profile.speech,
                                  &configuration->profile.voice,
                                  &configuration->profile.instructions};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(fields) && iree_status_is_ok(status); ++i) {
    if (!(present & (1u << i))) {
      continue;
    }
    status = loom_serve_realtime_string_decode(
        values[i].text, i == 5 ? 65536 : 128, &configuration->strings[i]);
    *fields[i] = iree_string_builder_view(&configuration->strings[i]);
    if (iree_status_is_ok(status) && i < 5 && !fields[i]->size) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "assistant selector '%s' is empty", rules[i].name);
    }
  }
  uint64_t maximum = 256;
  if (iree_status_is_ok(status) && (present & 64)) {
    status = iree_json_parse_uint64(values[6].text, &maximum);
    if (iree_status_is_ok(status) && (!maximum || maximum > UINT32_MAX)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "max_output_tokens requires a positive uint32");
    }
  }
  configuration->profile.max_output_tokens = (uint32_t)maximum;
  if (iree_status_is_ok(status) && (present & 128)) {
    status =
        iree_json_parse_uint64(values[7].text, &configuration->profile.seed);
  }
  return status;
}

iree_status_t loom_serve_realtime_configuration_load(
    iree_string_view_t path, iree_allocator_t host_allocator,
    loom_serve_realtime_configuration_t* configuration) {
  *configuration = (loom_serve_realtime_configuration_t){0};
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_read(path, host_allocator, &contents));
  const iree_status_t status = loom_serve_realtime_configuration_initialize(
      iree_make_string_view((const char*)contents->const_buffer.data,
                            contents->const_buffer.data_length),
      host_allocator, configuration);
  iree_io_file_contents_free(contents);
  return status;
}

void loom_serve_realtime_configuration_deinitialize(
    loom_serve_realtime_configuration_t* configuration) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(configuration->strings);
       ++i) {
    iree_string_builder_deinitialize(&configuration->strings[i]);
  }
}
