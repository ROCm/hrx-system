// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/telemetry.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t weight_activity_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_weight_activity_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  return iree_string_builder_append_format(
      output,
      ",\"activations\":%" PRIu64 ",\"deactivations\":%" PRIu64
      ",\"read_bytes\":%" PRIu64 ",\"prepared_bytes\":%" PRIu64
      ",\"groups\":%" PRIu64,
      value.activations, value.deactivations, value.read_bytes,
      value.prepared_bytes, value.groups);
}

static const loom_serve_telemetry_event_t weight_activity_event = {
    "weight_activity", weight_activity_format};

void loom_serve_telemetry_record_weights(
    loom_serve_telemetry_stream_t* stream, iree_time_t time_ns,
    const loom_serve_weight_activity_t* activity) {
  loom_serve_telemetry_record_t record = {.time_ns = time_ns,
                                          .event = &weight_activity_event};
  memcpy(record.data.words, activity, sizeof(*activity));
  loom_serve_telemetry_stream_append(stream, &record);
}
