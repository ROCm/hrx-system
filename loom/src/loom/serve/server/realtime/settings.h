// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_SETTINGS_H_
#define LOOM_SERVE_SERVER_REALTIME_SETTINGS_H_

#include "loom/serve/server/realtime/event.h"

#ifdef __cplusplus
extern "C" {
#endif

// Explicit composition of independently registered and scheduled deployments.
// All strings are borrowed from application configuration through router exit.
typedef struct loom_serve_realtime_profile_t {
  // Public assistant name selected by /v1/realtime?model=NAME.
  iree_string_view_t name;
  // Text deployment, with retained-session storage and two checkpoint slots.
  iree_string_view_t text;
  // Recognition deployment supplying source-defined input geometry.
  iree_string_view_t recognition;
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
} loom_serve_realtime_profile_t;

typedef struct loom_serve_realtime_settings_t {
  // Allocator owning independently mutable session settings.
  iree_allocator_t host_allocator;
  // Text instructions, copied before response composition.
  iree_string_builder_t instructions;
  // Source voice selected independently of speech deployment identity.
  iree_string_builder_t voice;
  // Optional recognition language selector.
  iree_string_builder_t language;
  // Optional recognition prompt, copied into each independent audio item.
  iree_string_builder_t prompt;
  // Text-only versus audio output; generated audio also carries its transcript.
  bool audio;
  // Explicit maximum native output token count.
  uint32_t max_output_tokens;
} loom_serve_realtime_settings_t;

iree_status_t loom_serve_realtime_settings_initialize(
    const loom_serve_realtime_profile_t* profile,
    iree_allocator_t host_allocator, loom_serve_realtime_settings_t* settings);
void loom_serve_realtime_settings_deinitialize(
    loom_serve_realtime_settings_t* settings);

// Transactionally applies the supported manual-turn GA session settings.
// Unknown fields and unsupported formats/turn policies reject explicitly. A
// requested transcription model must match the selected assistant composition.
// response restricts the object to per-response overrides of text policy and
// output modality; session type and input/output audio settings are not
// overrides. Optional out_instructions_changed reports whether committed text
// policy changed. Audio/voice/output-budget changes preserve compatible text
// prefixes.
iree_status_t loom_serve_realtime_settings_update(
    loom_serve_realtime_settings_t* settings,
    const loom_serve_realtime_profile_t* profile, iree_string_view_t object,
    bool response, bool* out_instructions_changed);

// Appends the actual effective session settings, including manual turn policy.
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
