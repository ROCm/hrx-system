// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_CONFIGURATION_H_
#define LOOM_SERVE_SERVER_REALTIME_CONFIGURATION_H_

#include "loom/serve/server/realtime/settings.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_configuration_t {
  // Profile view borrowing the decoded configuration strings below.
  loom_serve_realtime_profile_t profile;
  // Owned name, text, recognition, speech, voice, instructions, and activity.
  iree_string_builder_t strings[7];
} loom_serve_realtime_configuration_t;

// Parses one complete assistant document with required name/text/recognition/
// speech/voice and optional activity/instructions/max_output_tokens/seed/
// input_capacity/execution_weight. Every selector names an independent
// registration; this document owns no model weights.
// Failure leaves a deinitializable value and performs no model or file I/O.
iree_status_t loom_serve_realtime_configuration_initialize(
    iree_string_view_t document, iree_allocator_t host_allocator,
    loom_serve_realtime_configuration_t* configuration);
// Cold startup file I/O, outside the serialized serving owner.
iree_status_t loom_serve_realtime_configuration_load(
    iree_string_view_t path, iree_allocator_t host_allocator,
    loom_serve_realtime_configuration_t* configuration);
void loom_serve_realtime_configuration_deinitialize(
    loom_serve_realtime_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_CONFIGURATION_H_
