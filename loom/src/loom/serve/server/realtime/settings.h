// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_SETTINGS_H_
#define LOOM_SERVE_SERVER_REALTIME_SETTINGS_H_

#include "loom/serve/server/realtime/event.h"
#include "loom/serve/server/realtime/turn.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { LOOM_SERVE_REALTIME_INPUT_CAPACITY_DEFAULT = 48000 };

// Explicit composition of independently registered and scheduled deployments.
// All strings are borrowed from application configuration through router exit.
typedef struct loom_serve_realtime_profile_t {
  // Public assistant name selected by /v1/realtime?model=NAME.
  iree_string_view_t name;
  // Text deployment, with retained-session storage and two checkpoint slots.
  iree_string_view_t text;
  // Recognition deployment supplying source-defined input geometry.
  iree_string_view_t recognition;
  // Optional independent activity deployment for automatic microphone turns.
  iree_string_view_t activity;
  // Speech deployment supplying continuing text input and waveform geometry.
  iree_string_view_t speech;
  // Default source-owned voice ID; contains no model selector.
  iree_string_view_t voice;
  // Default instructions passed to the text model as a system message.
  iree_string_view_t instructions;
  // Default output-token ceiling; zero uses 256.
  uint32_t max_output_tokens;
  // Explicit reproducibility key for independent native requests.
  uint64_t seed;
  // Immutable original 24 kHz microphone ring capacity; zero uses the default.
  iree_host_size_t input_capacity;
} loom_serve_realtime_profile_t;

// Generation policy shared by session defaults and independent response copies.
typedef struct loom_serve_realtime_output_settings_t {
  // Text instructions, copied before response composition.
  iree_string_builder_t instructions;
  // Source voice selected independently of speech deployment identity.
  iree_string_builder_t voice;
  // Text-only versus audio output; generated audio also carries its transcript.
  bool audio;
  // Explicit maximum native output token count.
  uint32_t max_output_tokens;
} loom_serve_realtime_output_settings_t;

typedef struct loom_serve_realtime_settings_t {
  // Allocator owning independently mutable session settings.
  iree_allocator_t host_allocator;
  // Defaults copied before each response applies its own overrides.
  loom_serve_realtime_output_settings_t output;
  // Optional recognition language selector.
  iree_string_builder_t language;
  // Optional recognition prompt, copied into each independent audio item.
  iree_string_builder_t prompt;
  // Server activity detection is enabled; false selects explicit manual turns.
  bool detect_turns;
  // Continuous detector threshold, pre-roll, and silence policy.
  loom_serve_realtime_turn_options_t turn;
  // A detected stop requests a response once committed transcripts are ready.
  bool create_response;
  // A detected onset cancels the currently active response.
  bool interrupt_response;
} loom_serve_realtime_settings_t;

typedef struct loom_serve_realtime_response_settings_t {
  // Owned immutable policy, independent of later session defaults.
  loom_serve_realtime_output_settings_t output;
  // Owned canonical JSON object; empty storage denotes null metadata.
  iree_string_builder_t metadata;
} loom_serve_realtime_response_settings_t;

iree_status_t loom_serve_realtime_settings_initialize(
    const loom_serve_realtime_profile_t* profile,
    iree_allocator_t host_allocator, loom_serve_realtime_settings_t* settings);
void loom_serve_realtime_settings_deinitialize(
    loom_serve_realtime_settings_t* settings);

// Transactionally applies the supported GA session settings.
// Unknown fields and unsupported formats/turn policies reject explicitly. A
// requested transcription model must match the selected assistant composition.
// Optional out_instructions_changed reports whether committed text
// policy changed. Audio/voice/output-budget changes preserve compatible text
// prefixes. input_open prevents changes to input policy until explicit
// commit/clear; such changes cannot reinterpret already classified samples.
iree_status_t loom_serve_realtime_settings_update(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object,
    bool input_open, bool* out_instructions_changed);

// Copies defaults, then applies one response's output policy and metadata.
// overrides is empty or a complete JSON object. Input policy and session type
// are not response settings. Metadata has at most 16 distinct keys, 64 Unicode
// codepoints per key, and string values of at most 512 codepoints. The result
// owns every selected byte before the command receipt is released, and remains
// deinitializable after failure. No model work or session mutation occurs.
iree_status_t loom_serve_realtime_response_settings_initialize(
    const loom_serve_realtime_settings_t* defaults,
    iree_string_view_t overrides,
    loom_serve_realtime_response_settings_t* settings);
void loom_serve_realtime_response_settings_deinitialize(
    loom_serve_realtime_response_settings_t* settings);

// Appends the actual effective session settings, including turn policy.
iree_status_t loom_serve_realtime_settings_write(
    const loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, uint64_t session_id,
    iree_string_builder_t* output);

// Exact decoded comparison for bounded protocol vocabulary. Invalid escapes or
// a value longer than 128 decoded bytes cannot equal these protocol literals.
bool loom_serve_realtime_string_is(iree_string_view_t encoded,
                                   iree_string_view_t expected);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_SETTINGS_H_
