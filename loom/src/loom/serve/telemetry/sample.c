// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/sample.h"

#include <inttypes.h>

iree_status_t loom_serve_telemetry_sample_write_json(
    const loom_runtime_telemetry_topic_t* topic, const uint64_t* values,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"topic\":\"%s\",\"version\":%u,\"fields\":[", topic->name,
      topic->version));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < topic->count && iree_status_is_ok(status);
       ++i) {
    const loom_runtime_telemetry_metric_t* field = &topic->fields[i];
    status = iree_string_builder_append_format(
        output,
        "%s{\"name\":\"%s\",\"unit\":\"%s\",\"kind\":\"%s\",\"value\":"
        "\"%" PRIu64 "\"}",
        i ? "," : "", field->name, field->unit,
        field->kind == LOOM_RUNTIME_TELEMETRY_COUNTER ? "counter" : "gauge",
        values[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]}");
  }
  return status;
}
