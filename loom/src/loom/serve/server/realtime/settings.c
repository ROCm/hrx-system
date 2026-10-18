// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/settings.h"

#include <inttypes.h>
#include <math.h>

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
      .audio = true,
      .turn = {.threshold = 0.5f,
               .prefix_padding_ms = 300,
               .silence_duration_ms = 500},
      .create_response = true,
      .interrupt_response = true,
      .max_output_tokens =
          profile->max_output_tokens ? profile->max_output_tokens : 256};
  iree_string_builder_initialize(host_allocator, &settings->instructions);
  iree_string_builder_initialize(host_allocator, &settings->voice);
  iree_string_builder_initialize(host_allocator, &settings->language);
  iree_string_builder_initialize(host_allocator, &settings->prompt);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      &settings->instructions, profile->instructions));
  return iree_string_builder_append_string(&settings->voice, profile->voice);
}

void loom_serve_realtime_settings_deinitialize(
    loom_serve_realtime_settings_t* settings) {
  iree_string_builder_deinitialize(&settings->instructions);
  iree_string_builder_deinitialize(&settings->voice);
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
    static const loom_serve_realtime_field_rule_t output_rules[] = {
        {"format", REALTIME_TYPE(OBJECT)},
        {"voice", REALTIME_TYPE(STRING)},
        {"speed", REALTIME_TYPE(NUMBER)}};
    loom_serve_realtime_event_value_t
        output_values[IREE_ARRAYSIZE(output_rules)];
    uint32_t output_present = 0;
    IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
        values[1].text, IREE_ARRAYSIZE(output_rules), output_rules,
        output_values, &output_present));
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
      IREE_RETURN_IF_ERROR(
          iree_json_parse_double(output_values[2].text, &speed));
      if (speed != 1.0) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "realtime speech speed must be 1");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_settings_parse(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object,
    bool response) {
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", REALTIME_TYPE(STRING)},
      {"instructions", REALTIME_TYPE(STRING)},
      {"audio", REALTIME_TYPE(OBJECT)},
      {"output_modalities", REALTIME_TYPE(ARRAY)},
      {"max_output_tokens", REALTIME_TYPE(NUMBER)},
      {"tools", REALTIME_TYPE(ARRAY)},
      {"tool_choice", REALTIME_TYPE(STRING)}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(rules), rules, values, &present));
  if (response && (present & (1 | 4))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "response overrides cannot change session audio or type");
  }
  if ((present & 1) &&
      !loom_serve_realtime_string_is(values[0].text, IREE_SV("realtime"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "assistant sessions require type realtime");
  }
  if (present & 2) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        values[1].text, 65536, &settings->instructions));
  }
  if (present & 4) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_audio_parse(settings, profile, values[2].text));
  }
  if (present & 8) {
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_json_array_length(values[3].text, &count));
    iree_string_view_t modality = iree_string_view_empty();
    if (count == 1) {
      IREE_RETURN_IF_ERROR(iree_json_array_get(values[3].text, 0, &modality));
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
  if (present & 16) {
    uint64_t limit = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(values[4].text, &limit));
    if (!limit || limit > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "max_output_tokens requires a positive uint32");
    }
    settings->max_output_tokens = (uint32_t)limit;
  }
  if (present & 32) {
    iree_host_size_t count = 0;
    IREE_RETURN_IF_ERROR(iree_json_array_length(values[5].text, &count));
    if (count) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "function tools are not enabled for this session");
    }
  }
  if ((present & 64) &&
      !loom_serve_realtime_string_is(values[6].text, IREE_SV("none")) &&
      !loom_serve_realtime_string_is(values[6].text, IREE_SV("auto"))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "tool_choice requires none or auto with empty tools");
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_settings_update(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object,
    bool response, bool input_open, bool* out_instructions_changed) {
  if (out_instructions_changed) {
    *out_instructions_changed = false;
  }
  const loom_serve_realtime_profile_t current = {
      .voice = iree_string_builder_view(&settings->voice),
      .instructions = iree_string_builder_view(&settings->instructions),
      .max_output_tokens = settings->max_output_tokens};
  loom_serve_realtime_settings_t candidate;
  iree_status_t status = loom_serve_realtime_settings_initialize(
      &current, settings->host_allocator, &candidate);
  candidate.audio = settings->audio;
  candidate.detect_turns = settings->detect_turns;
  candidate.turn = settings->turn;
  candidate.create_response = settings->create_response;
  candidate.interrupt_response = settings->interrupt_response;
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &candidate.language, iree_string_builder_view(&settings->language));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &candidate.prompt, iree_string_builder_view(&settings->prompt));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_settings_parse(&candidate, profile, object,
                                                response);
  }
  if (iree_status_is_ok(status) && input_open &&
      (candidate.detect_turns != settings->detect_turns ||
       candidate.turn.threshold != settings->turn.threshold ||
       candidate.turn.prefix_padding_ms != settings->turn.prefix_padding_ms ||
       candidate.turn.silence_duration_ms !=
           settings->turn.silence_duration_ms ||
       candidate.create_response != settings->create_response ||
       candidate.interrupt_response != settings->interrupt_response ||
       !iree_string_view_equal(iree_string_builder_view(&candidate.language),
                               iree_string_builder_view(&settings->language)) ||
       !iree_string_view_equal(iree_string_builder_view(&candidate.prompt),
                               iree_string_builder_view(&settings->prompt)))) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "commit or clear microphone input before changing its policy");
  }
  if (iree_status_is_ok(status)) {
    if (out_instructions_changed) {
      *out_instructions_changed = !iree_string_view_equal(
          iree_string_builder_view(&settings->instructions),
          iree_string_builder_view(&candidate.instructions));
    }
    loom_serve_realtime_settings_t previous = *settings;
    *settings = candidate;
    candidate = previous;
  }
  loom_serve_realtime_settings_deinitialize(&candidate);
  return status;
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
      output, iree_string_builder_view(&settings->instructions)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"output_modalities\":[\"%s\"],\"max_output_tokens\":%" PRIu32
      ",\"tools\":[],\"tool_choice\":\"none\",\"audio\":{\"input\":{"
      "\"format\":{\"type\":\"audio/pcm\",\"rate\":24000},"
      "\"turn_detection\":",
      settings->audio ? "audio" : "text", settings->max_output_tokens));
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
      output, iree_string_builder_view(&settings->voice)));
  return iree_string_builder_append_cstring(output, ",\"speed\":1}}}");
}

#undef REALTIME_TYPE
