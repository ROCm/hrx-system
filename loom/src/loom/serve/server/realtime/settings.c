// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/settings.h"

#include <inttypes.h>
#include <math.h>

#include "iree/base/internal/unicode.h"

#define REALTIME_TYPE(type) (1u << IREE_JSON_VALUE_TYPE_##type)

bool loom_serve_realtime_string_is(iree_string_view_t encoded,
                                   iree_string_view_t expected) {
  char decoded[128];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(encoded, sizeof(decoded), decoded, &length);
  const bool equal =
      iree_status_is_ok(status) &&
      iree_string_view_equal(iree_make_string_view(decoded, length), expected);
  iree_status_ignore(status);
  return equal;
}

iree_status_t loom_serve_realtime_settings_initialize(
    const loom_serve_realtime_profile_t* profile,
    iree_allocator_t host_allocator, loom_serve_realtime_settings_t* settings) {
  *settings = (loom_serve_realtime_settings_t){
      .host_allocator = host_allocator,
      .output = {.audio = true,
                 .max_output_tokens = profile->max_output_tokens
                                          ? profile->max_output_tokens
                                          : 256},
      .turn = {.threshold = 0.5f,
               .prefix_padding_ms = 300,
               .silence_duration_ms = 500},
      .create_response = true,
      .interrupt_response = true};
  iree_string_builder_initialize(host_allocator,
                                 &settings->output.instructions);
  iree_string_builder_initialize(host_allocator, &settings->output.voice);
  iree_string_builder_initialize(host_allocator, &settings->language);
  iree_string_builder_initialize(host_allocator, &settings->prompt);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      &settings->output.instructions, profile->instructions));
  return iree_string_builder_append_string(&settings->output.voice,
                                           profile->voice);
}

static iree_status_t loom_serve_realtime_output_settings_copy(
    const loom_serve_realtime_output_settings_t* source,
    iree_allocator_t host_allocator,
    loom_serve_realtime_output_settings_t* target) {
  *target = *source;
  loom_serve_realtime_tools_retain(target->tools);
  iree_string_builder_initialize(host_allocator, &target->instructions);
  iree_string_builder_initialize(host_allocator, &target->voice);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      &target->instructions, iree_string_builder_view(&source->instructions)));
  return iree_string_builder_append_string(
      &target->voice, iree_string_builder_view(&source->voice));
}

static void loom_serve_realtime_output_settings_deinitialize(
    loom_serve_realtime_output_settings_t* settings) {
  loom_serve_realtime_tools_release(settings->tools);
  iree_string_builder_deinitialize(&settings->instructions);
  iree_string_builder_deinitialize(&settings->voice);
}

void loom_serve_realtime_settings_deinitialize(
    loom_serve_realtime_settings_t* settings) {
  loom_serve_realtime_output_settings_deinitialize(&settings->output);
  iree_string_builder_deinitialize(&settings->language);
  iree_string_builder_deinitialize(&settings->prompt);
}

static iree_status_t loom_serve_realtime_format_parse(
    iree_string_view_t object) {
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", REALTIME_TYPE(STRING)}, {"rate", REALTIME_TYPE(NUMBER)}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  uint64_t rate = 0;
  if (present & 2) {
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(values[1].text, &rate));
  }
  if (present != 3 || rate != 24000 ||
      !loom_serve_realtime_string_is(values[0].text, IREE_SV("audio/pcm"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "realtime audio requires mono PCM16 at 24000 Hz");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_transcription_parse(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object) {
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"model", REALTIME_TYPE(STRING)},
      {"language", REALTIME_TYPE(STRING)},
      {"prompt", REALTIME_TYPE(STRING)}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  if ((present & 1) &&
      !loom_serve_realtime_string_is(values[0].text, profile->recognition)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transcription model must match the assistant profile");
  }
  if (present & 2) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[1].text, 128, &settings->language));
  }
  if (present & 4) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[2].text, 16384, &settings->prompt));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_turn_parse(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile,
    const loom_serve_realtime_event_value_t* value) {
  if (value->type == IREE_JSON_VALUE_TYPE_NULL) {
    settings->detect_turns = false;
    return iree_ok_status();
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", REALTIME_TYPE(STRING)},
      {"threshold", REALTIME_TYPE(NUMBER)},
      {"prefix_padding_ms", REALTIME_TYPE(NUMBER)},
      {"silence_duration_ms", REALTIME_TYPE(NUMBER)},
      {"create_response", REALTIME_TYPE(TRUE) | REALTIME_TYPE(FALSE)},
      {"interrupt_response", REALTIME_TYPE(TRUE) | REALTIME_TYPE(FALSE)}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      value->text, IREE_ARRAYSIZE(rules), rules, values, &present));
  if (!(present & 1) ||
      !loom_serve_realtime_string_is(values[0].text, IREE_SV("server_vad"))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "turn_detection requires server_vad or null");
  }
  if (!profile->activity.size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "assistant has no activity deployment");
  }
  if (present & 2) {
    double threshold = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_double(values[1].text, &threshold));
    if (!isfinite(threshold) || threshold < 0 || threshold > 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "activity threshold requires a value in [0, 1]");
    }
    settings->turn.threshold = (float)threshold;
  }
  uint32_t* durations[] = {&settings->turn.prefix_padding_ms,
                           &settings->turn.silence_duration_ms};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(durations) && iree_status_is_ok(status); ++i) {
    if (present & (1u << (i + 2))) {
      uint64_t duration = 0;
      status = iree_json_parse_uint64(values[i + 2].text, &duration);
      if (iree_status_is_ok(status) && duration > UINT32_MAX) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "turn duration requires uint32 milliseconds");
      }
      *durations[i] = (uint32_t)duration;
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_host_size_t capacity =
      profile->input_capacity ? profile->input_capacity
                              : LOOM_SERVE_REALTIME_INPUT_CAPACITY_DEFAULT;
  if ((uint64_t)settings->turn.prefix_padding_ms * 24 >= capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pre-roll must fit assistant input_capacity");
  }
  if (present & 16) {
    settings->create_response = values[4].type == IREE_JSON_VALUE_TYPE_TRUE;
  }
  if (present & 32) {
    settings->interrupt_response = values[5].type == IREE_JSON_VALUE_TYPE_TRUE;
  }
  settings->detect_turns = true;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_output_audio_parse(
    loom_serve_realtime_output_settings_t* settings,
    iree_string_view_t object) {
  static const loom_serve_realtime_field_rule_t output_rules[] = {
      {"format", REALTIME_TYPE(OBJECT)},
      {"voice", REALTIME_TYPE(STRING)},
      {"speed", REALTIME_TYPE(NUMBER)}};
  loom_serve_realtime_event_value_t output_values[IREE_ARRAYSIZE(output_rules)];
  uint32_t output_present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(output_rules), output_rules, output_values,
      &output_present));
  if (output_present & 1) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_format_parse(output_values[0].text));
  }
  if (output_present & 2) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        output_values[1].text, 1024, &settings->voice));
    if (!settings->voice.size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "voice is empty");
    }
  }
  if (output_present & 4) {
    double speed = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_double(output_values[2].text, &speed));
    if (speed != 1.0) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "realtime speech speed must be 1");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_audio_parse(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object) {
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"input", REALTIME_TYPE(OBJECT)}, {"output", REALTIME_TYPE(OBJECT)}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  if (present & 1) {
    static const loom_serve_realtime_field_rule_t input_rules[] = {
        {"format", REALTIME_TYPE(OBJECT)},
        {"transcription", REALTIME_TYPE(OBJECT)},
        {"turn_detection", REALTIME_TYPE(NULL) | REALTIME_TYPE(OBJECT)},
        {"noise_reduction", REALTIME_TYPE(NULL)}};
    loom_serve_realtime_event_value_t input_values[IREE_ARRAYSIZE(input_rules)];
    uint32_t input_present = 0;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
        values[0].text, IREE_ARRAYSIZE(input_rules), input_rules, input_values,
        &input_present));
    if (input_present & 1) {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_format_parse(input_values[0].text));
    }
    if (input_present & 2) {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_transcription_parse(
          settings, profile, input_values[1].text));
    }
    if (input_present & 4) {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_turn_parse(settings, profile, &input_values[2]));
    }
  }
  if (present & 2) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_output_audio_parse(
        &settings->output, values[1].text));
  }
  return iree_ok_status();
}

enum {
  LOOM_SERVE_REALTIME_SETTING_TYPE,
  LOOM_SERVE_REALTIME_SETTING_INSTRUCTIONS,
  LOOM_SERVE_REALTIME_SETTING_AUDIO,
  LOOM_SERVE_REALTIME_SETTING_MODALITIES,
  LOOM_SERVE_REALTIME_SETTING_MAX_OUTPUT_TOKENS,
  LOOM_SERVE_REALTIME_SETTING_TOOLS,
  LOOM_SERVE_REALTIME_SETTING_TOOL_CHOICE,
  LOOM_SERVE_REALTIME_SETTING_METADATA,
  LOOM_SERVE_REALTIME_SETTING_COUNT,
};

typedef struct loom_serve_realtime_settings_fields_t {
  // Indexed values borrowing the command through transactional application.
  loom_serve_realtime_event_value_t values[LOOM_SERVE_REALTIME_SETTING_COUNT];
  // Present fields, including explicit empty strings and nulls.
  uint32_t present;
} loom_serve_realtime_settings_fields_t;

static iree_status_t loom_serve_realtime_settings_parse(
    iree_string_view_t object, bool response,
    loom_serve_realtime_settings_fields_t* fields) {
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", REALTIME_TYPE(STRING)},
      {"instructions", REALTIME_TYPE(STRING)},
      {"audio", REALTIME_TYPE(OBJECT)},
      {"output_modalities", REALTIME_TYPE(ARRAY)},
      {"max_output_tokens", REALTIME_TYPE(NUMBER)},
      {"tools", REALTIME_TYPE(ARRAY)},
      {"tool_choice", REALTIME_TYPE(STRING)},
      {"metadata", REALTIME_TYPE(OBJECT) | REALTIME_TYPE(NULL)}};
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, fields->values, &fields->present));
  const uint32_t inapplicable =
      response ? (1u << LOOM_SERVE_REALTIME_SETTING_TYPE)
               : (1u << LOOM_SERVE_REALTIME_SETTING_METADATA);
  if (fields->present & inapplicable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inapplicable session or response settings");
  }
  if ((fields->present & (1u << LOOM_SERVE_REALTIME_SETTING_TYPE)) &&
      !loom_serve_realtime_string_is(
          fields->values[LOOM_SERVE_REALTIME_SETTING_TYPE].text,
          IREE_SV("realtime"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "assistant sessions require type realtime");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_output_settings_apply(
    loom_serve_realtime_output_settings_t* settings,
    const loom_serve_realtime_settings_fields_t* fields) {
  const uint32_t present = fields->present;
  const loom_serve_realtime_event_value_t* values = fields->values;
  if (present & (1u << LOOM_SERVE_REALTIME_SETTING_INSTRUCTIONS)) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[LOOM_SERVE_REALTIME_SETTING_INSTRUCTIONS].text, 65536,
        &settings->instructions));
  }
  if (present & (1u << LOOM_SERVE_REALTIME_SETTING_MODALITIES)) {
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_json_array_length(
        values[LOOM_SERVE_REALTIME_SETTING_MODALITIES].text, &count));
    iree_string_view_t modality = iree_string_view_empty();
    if (count == 1) {
      IREE_RETURN_IF_ERROR(iree_json_array_get(
          values[LOOM_SERVE_REALTIME_SETTING_MODALITIES].text, 0, &modality));
    }
    if (count != 1 ||
        (!loom_serve_realtime_string_is(modality, IREE_SV("audio")) &&
         !loom_serve_realtime_string_is(modality, IREE_SV("text")))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "output_modalities requires one of audio or text");
    }
    settings->audio = loom_serve_realtime_string_is(modality, IREE_SV("audio"));
  }
  if (present & (1u << LOOM_SERVE_REALTIME_SETTING_MAX_OUTPUT_TOKENS)) {
    uint64_t limit = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(
        values[LOOM_SERVE_REALTIME_SETTING_MAX_OUTPUT_TOKENS].text, &limit));
    if (!limit || limit > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "max_output_tokens requires a positive uint32");
    }
    settings->max_output_tokens = (uint32_t)limit;
  }
  if (present & (1u << LOOM_SERVE_REALTIME_SETTING_TOOLS)) {
    loom_serve_realtime_tools_t* tools = NULL;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_tools_parse(
        values[LOOM_SERVE_REALTIME_SETTING_TOOLS].text,
        settings->instructions.allocator, &tools));
    loom_serve_realtime_tools_release(settings->tools);
    settings->tools = tools;
  }
  if (present & (1u << LOOM_SERVE_REALTIME_SETTING_TOOL_CHOICE)) {
    const iree_string_view_t choice =
        values[LOOM_SERVE_REALTIME_SETTING_TOOL_CHOICE].text;
    if (!loom_serve_realtime_string_is(choice, IREE_SV("none")) &&
        !loom_serve_realtime_string_is(choice, IREE_SV("auto"))) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "tool_choice supports auto or none");
    }
    settings->tool_choice_none =
        loom_serve_realtime_string_is(choice, IREE_SV("none"));
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_settings_prepare(
    const loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object,
    bool input_open, loom_serve_realtime_settings_t* candidate) {
  *candidate = *settings;
  iree_string_builder_initialize(settings->host_allocator,
                                 &candidate->language);
  iree_string_builder_initialize(settings->host_allocator, &candidate->prompt);
  iree_status_t status = loom_serve_realtime_output_settings_copy(
      &settings->output, settings->host_allocator, &candidate->output);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &candidate->language, iree_string_builder_view(&settings->language));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &candidate->prompt, iree_string_builder_view(&settings->prompt));
  }
  loom_serve_realtime_settings_fields_t fields;
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_settings_parse(object, false, &fields);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_realtime_output_settings_apply(&candidate->output, &fields);
  }
  if (iree_status_is_ok(status) &&
      (fields.present & (1u << LOOM_SERVE_REALTIME_SETTING_AUDIO))) {
    status = loom_serve_realtime_audio_parse(
        candidate, profile,
        fields.values[LOOM_SERVE_REALTIME_SETTING_AUDIO].text);
  }
  if (iree_status_is_ok(status) && input_open &&
      (candidate->detect_turns != settings->detect_turns ||
       candidate->turn.threshold != settings->turn.threshold ||
       candidate->turn.prefix_padding_ms != settings->turn.prefix_padding_ms ||
       candidate->turn.silence_duration_ms !=
           settings->turn.silence_duration_ms ||
       candidate->create_response != settings->create_response ||
       candidate->interrupt_response != settings->interrupt_response ||
       !iree_string_view_equal(iree_string_builder_view(&candidate->language),
                               iree_string_builder_view(&settings->language)) ||
       !iree_string_view_equal(iree_string_builder_view(&candidate->prompt),
                               iree_string_builder_view(&settings->prompt)))) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "commit or clear microphone input before changing its policy");
  }
  return status;
}

iree_string_view_t loom_serve_realtime_output_settings_tools(
    const loom_serve_realtime_output_settings_t* settings) {
  return settings->tool_choice_none
             ? iree_string_view_empty()
             : loom_serve_realtime_tools_native(settings->tools);
}

bool loom_serve_realtime_output_settings_text_equal(
    const loom_serve_realtime_output_settings_t* left,
    const loom_serve_realtime_output_settings_t* right) {
  return iree_string_view_equal(
             iree_string_builder_view(&left->instructions),
             iree_string_builder_view(&right->instructions)) &&
         iree_string_view_equal(
             loom_serve_realtime_output_settings_tools(left),
             loom_serve_realtime_output_settings_tools(right));
}

enum {
  LOOM_SERVE_REALTIME_METADATA_ENTRIES = 16,
  LOOM_SERVE_REALTIME_METADATA_KEY_CODEPOINTS = 64,
  LOOM_SERVE_REALTIME_METADATA_VALUE_CODEPOINTS = 512,
};

typedef struct loom_serve_realtime_metadata_parser_t {
  // Canonical response-owned JSON, independent of the incoming receipt.
  iree_string_builder_t* output;
  // Decoded keys retained only while checking this bounded object's uniqueness.
  char keys[LOOM_SERVE_REALTIME_METADATA_ENTRIES]
           [4 * LOOM_SERVE_REALTIME_METADATA_KEY_CODEPOINTS];
  // Byte lengths preserve embedded NULs in decoded keys.
  iree_host_size_t lengths[LOOM_SERVE_REALTIME_METADATA_ENTRIES];
  // Number of fully validated and serialized entries.
  iree_host_size_t count;
} loom_serve_realtime_metadata_parser_t;

static iree_status_t loom_serve_realtime_metadata_string_decode(
    iree_string_view_t encoded, iree_host_size_t limit, char* storage,
    iree_string_view_t* out_decoded) {
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(encoded, 4 * limit, storage, &length);
  if (iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata string exceeds %" PRIhsz " codepoints",
                            limit);
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_string_view_t decoded = iree_make_string_view(storage, length);
  if (!iree_unicode_utf8_validate(decoded) ||
      iree_unicode_utf8_codepoint_count(decoded) > limit) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "metadata string requires valid UTF-8 within %" PRIhsz " codepoints",
        limit);
  }
  *out_decoded = decoded;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_metadata_visit(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_realtime_metadata_parser_t* parser = user_data;
  if (parser->count == LOOM_SERVE_REALTIME_METADATA_ENTRIES ||
      type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata requires at most 16 string-valued keys");
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_metadata_string_decode(
      key, LOOM_SERVE_REALTIME_METADATA_KEY_CODEPOINTS,
      parser->keys[parser->count], &key));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < parser->count && iree_status_is_ok(status);
       ++i) {
    if (iree_string_view_equal(
            key, iree_make_string_view(parser->keys[i], parser->lengths[i]))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate metadata key");
    }
  }
  IREE_RETURN_IF_ERROR(status);
  char decoded_value[4 * LOOM_SERVE_REALTIME_METADATA_VALUE_CODEPOINTS];
  IREE_RETURN_IF_ERROR(loom_serve_realtime_metadata_string_decode(
      value, LOOM_SERVE_REALTIME_METADATA_VALUE_CODEPOINTS, decoded_value,
      &value));
  if (parser->count) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(parser->output, ","));
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(parser->output, key));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(parser->output, ":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(parser->output, value));
  parser->lengths[parser->count++] = key.size;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_response_settings_initialize(
    const loom_serve_realtime_settings_t* defaults,
    iree_string_view_t overrides,
    loom_serve_realtime_response_settings_t* settings) {
  iree_string_builder_initialize(defaults->host_allocator, &settings->metadata);
  IREE_RETURN_IF_ERROR(loom_serve_realtime_output_settings_copy(
      &defaults->output, defaults->host_allocator, &settings->output));
  if (!overrides.size) {
    return iree_ok_status();
  }
  loom_serve_realtime_settings_fields_t fields;
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_settings_parse(overrides, true, &fields));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_output_settings_apply(&settings->output, &fields));
  if (fields.present & (1u << LOOM_SERVE_REALTIME_SETTING_AUDIO)) {
    static const loom_serve_realtime_field_rule_t rules[] = {
        {"output", REALTIME_TYPE(OBJECT)}};
    loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
    uint32_t present = 0;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
        fields.values[LOOM_SERVE_REALTIME_SETTING_AUDIO].text,
        IREE_ARRAYSIZE(rules), rules, values, &present));
    if (present) {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_output_audio_parse(
          &settings->output, values[0].text));
    }
  }
  if ((fields.present & (1u << LOOM_SERVE_REALTIME_SETTING_METADATA)) &&
      fields.values[LOOM_SERVE_REALTIME_SETTING_METADATA].type ==
          IREE_JSON_VALUE_TYPE_OBJECT) {
    loom_serve_realtime_metadata_parser_t parser = {.output =
                                                        &settings->metadata};
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&settings->metadata, "{"));
    IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
        fields.values[LOOM_SERVE_REALTIME_SETTING_METADATA].text,
        loom_serve_realtime_metadata_visit, &parser));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&settings->metadata, "}"));
  }
  return iree_ok_status();
}

void loom_serve_realtime_response_settings_deinitialize(
    loom_serve_realtime_response_settings_t* settings) {
  loom_serve_realtime_output_settings_deinitialize(&settings->output);
  iree_string_builder_deinitialize(&settings->metadata);
}

iree_status_t loom_serve_realtime_settings_write(
    const loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, uint64_t session_id,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(output,
                                        "{\"id\":\"sess_%" PRIu64
                                        "\",\"object\":\"realtime.session\","
                                        "\"type\":\"realtime\",\"model\":",
                                        session_id));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(output, profile->name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"instructions\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&settings->output.instructions)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"output_modalities\":[\"%s\"],\"max_output_tokens\":%" PRIu32
      ",\"tools\":",
      settings->output.audio ? "audio" : "text",
      settings->output.max_output_tokens));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_tools_write(settings->output.tools, output));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"tool_choice\":\"%s\",\"audio\":{\"input\":{"
      "\"format\":{\"type\":\"audio/pcm\",\"rate\":24000},"
      "\"turn_detection\":",
      settings->output.tool_choice_none ? "none" : "auto"));
  if (settings->detect_turns) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "{\"type\":\"server_vad\",\"threshold\":%.9g,"
        "\"prefix_padding_ms\":%" PRIu32 ",\"silence_duration_ms\":%" PRIu32
        ",\"create_response\":%s,\"interrupt_response\":%s}",
        settings->turn.threshold, settings->turn.prefix_padding_ms,
        settings->turn.silence_duration_ms,
        settings->create_response ? "true" : "false",
        settings->interrupt_response ? "true" : "false"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, ",\"noise_reduction\":null,\"transcription\":{\"model\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_string_write(output, profile->recognition));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"language\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&settings->language)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"prompt\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&settings->prompt)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output,
      "}},\"output\":{\"format\":{\"type\":\"audio/pcm\",\"rate\":24000},"
      "\"voice\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&settings->output.voice)));
  return iree_string_builder_append_cstring(output, ",\"speed\":1}}}");
}

#undef REALTIME_TYPE
