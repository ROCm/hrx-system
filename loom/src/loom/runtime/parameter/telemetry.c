// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/telemetry.h"

#include <string.h>

static iree_status_t loom_runtime_parameter_activity_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_runtime_parameter_activity_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("activations", value.activations),
      loom_runtime_telemetry_field_uint64("deactivations", value.deactivations),
      loom_runtime_telemetry_field_uint64("read_bytes", value.read_bytes),
      loom_runtime_telemetry_field_uint64("prepared_bytes",
                                          value.prepared_bytes),
      loom_runtime_telemetry_field_uint64("groups", value.groups),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static const loom_runtime_telemetry_event_t parameter_activity_event = {
    "parameter_activity", loom_runtime_parameter_activity_fields};

void loom_runtime_parameter_record_activity(
    loom_runtime_telemetry_stream_t* stream, iree_time_t time_ns,
    const loom_runtime_parameter_activity_t* activity) {
  loom_runtime_telemetry_record_t record = {.time_ns = time_ns,
                                            .event = &parameter_activity_event};
  memcpy(record.data.words, activity, sizeof(*activity));
  loom_runtime_telemetry_stream_append(stream, &record);
}
