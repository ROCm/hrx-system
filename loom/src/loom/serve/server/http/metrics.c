// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/metrics.h"

#include <inttypes.h>

#include "loom/serve/telemetry/latency.h"
#include "loom/serve/telemetry/record.h"

// Prometheus labels preserve UTF-8 and escape only backslash, quote and LF.
// JSON escaping is different for control characters and Unicode separators.
static iree_status_t loom_serve_http_metrics_label(
    iree_string_builder_t* output, iree_string_view_t value) {
  iree_status_t status = iree_string_builder_append_cstring(output, "\"");
  iree_host_size_t start = 0;
  for (iree_host_size_t i = 0; i < value.size && iree_status_is_ok(status);
       ++i) {
    const char* escape = NULL;
    switch (value.data[i]) {
      case '\\':
        escape = "\\\\";
        break;
      case '"':
        escape = "\\\"";
        break;
      case '\n':
        escape = "\\n";
        break;
    }
    if (escape) {
      status = iree_string_builder_append_string(
          output, iree_make_string_view(value.data + start, i - start));
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(output, escape);
      }
      start = i + 1;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output, iree_string_view_substr(value, start, IREE_STRING_VIEW_NPOS));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "\"");
  }
  return status;
}

static iree_status_t loom_serve_http_metrics_identity(
    iree_string_builder_t* output, const loom_serve_registered_model_t* model) {
  const loom_serve_service_t* service = &model->service;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "model="));
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_label(output, service->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ",kind="));
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_label(
      output, iree_make_cstring_view(service->kind)));
  if (model->incarnation) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",incarnation=\"%" PRIu64 "\"", model->incarnation));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_metrics_sample_begin(
    iree_string_builder_t* output, const char* metric,
    const loom_serve_registered_model_t* model, const char* label,
    const char* value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "loom_serve_model_%s{", metric));
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_identity(output, model));
  if (label) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(output, ",%s=", label));
    IREE_RETURN_IF_ERROR(
        loom_serve_http_metrics_label(output, iree_make_cstring_view(value)));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_metrics_sample(
    iree_string_builder_t* output, const char* metric,
    const loom_serve_registered_model_t* model, const char* label,
    const char* label_value, double value) {
  if (!model->service.name.size) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_sample_begin(
      output, metric, model, label, label_value));
  return iree_string_builder_append_format(output, "} %.9g\n", value);
}

static iree_status_t loom_serve_http_metrics_integer(
    iree_string_builder_t* output, const char* metric,
    const loom_serve_registered_model_t* model, const char* component,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_sample_begin(
      output, metric, model, "component", component));
  return iree_string_builder_append_format(output, "} %" PRIu64 "\n", value);
}

static iree_status_t loom_serve_http_metrics_parameters(
    loom_serve_registry_t* registry, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output,
      "# HELP loom_serve_model_parameter_pins Accepted-consumer and explicit-"
      "retention pins, excluding persistent policy.\n"
      "# TYPE loom_serve_model_parameter_pins gauge\n"
      "# HELP loom_serve_model_parameter_admission_waiting Whether new work "
      "is waiting for parameter admission.\n"
      "# TYPE loom_serve_model_parameter_admission_waiting gauge\n"
      "# HELP loom_serve_model_parameter_admission_waits_total Distinct "
      "parameter admission waits, not repeated polls or request count.\n"
      "# TYPE loom_serve_model_parameter_admission_waits_total counter\n"
      "# HELP loom_serve_model_parameter_admission_wait_seconds_total Time "
      "waiting for parameter admission, including cancelled and open waits.\n"
      "# TYPE loom_serve_model_parameter_admission_wait_seconds_total counter\n"
      "# HELP loom_serve_model_parameter_admission_wait_age_seconds Current "
      "parameter admission wait age, or zero when not waiting.\n"
      "# TYPE loom_serve_model_parameter_admission_wait_age_seconds gauge\n"));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    if (model->phase != LOOM_SERVE_REGISTRATION_READY ||
        !model->registration.parameters) {
      continue;
    }
    for (loom_serve_model_component_t component = 0;
         component < LOOM_SERVE_MODEL_COMPONENT_COUNT &&
         iree_status_is_ok(status);
         ++component) {
      if (!loom_serve_registry_has_component(model, component)) {
        continue;
      }
      const loom_serve_registration_parameters_t parameters =
          model->registration.parameters(model->registration.self, component);
      if (!parameters.prepared) {
        continue;
      }
      const loom_serve_residency_snapshot_t state = parameters.parameters;
      const char* name = loom_serve_model_component_name(component);
      status = loom_serve_http_metrics_integer(output, "parameter_pins", model,
                                               name, state.pin_count);
      if (iree_status_is_ok(status)) {
        status = loom_serve_http_metrics_integer(
            output, "parameter_admission_waiting", model, name,
            state.admission.waiting);
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_http_metrics_integer(
            output, "parameter_admission_waits_total", model, name,
            state.admission.count);
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_http_metrics_sample(
            output, "parameter_admission_wait_seconds_total", model,
            "component", name, state.admission.duration_ns / 1e9);
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_http_metrics_sample(
            output, "parameter_admission_wait_age_seconds", model, "component",
            name, state.admission.age_ns / 1e9);
      }
    }
  }
  return status;
}

typedef struct loom_serve_metrics_counter_field_t {
  // Suffix shared by model and server Prometheus metric names.
  const char* name;
  // Counter units and ownership semantics.
  const char* description;
  // Byte offset of a uint64_t in loom_serve_telemetry_counters_t.
  iree_host_size_t offset;
} loom_serve_metrics_counter_field_t;

static const loom_serve_metrics_counter_field_t metrics_counter_fields[] = {
    {"requests", "Requests reaching a model protocol, including rejections.",
     offsetof(loom_serve_telemetry_counters_t, requests)},
    {"completed", "Successful responses copied into the HTTP carrier.",
     offsetof(loom_serve_telemetry_counters_t, completed)},
    {"rejected", "Protocol, source validation or capacity rejections.",
     offsetof(loom_serve_telemetry_counters_t, rejected)},
    {"cancelled", "Requests abandoned by peers or orderly shutdown.",
     offsetof(loom_serve_telemetry_counters_t, cancelled)},
    {"failed", "Requests terminated by transport or execution failures.",
     offsetof(loom_serve_telemetry_counters_t, failed)},
    {"input_tokens",
     "Tokens in completed model work, including abandoned work.",
     offsetof(loom_serve_telemetry_counters_t, input_tokens)},
    {"output_tokens", "Selected model output tokens, including EOS.",
     offsetof(loom_serve_telemetry_counters_t, output_tokens)},
    {"batches", "Completed model invocations or cohorts, not HTTP requests.",
     offsetof(loom_serve_telemetry_counters_t, batches)},
};

static uint64_t loom_serve_http_metrics_counter_value(
    const loom_serve_telemetry_counters_t* counters,
    const loom_serve_metrics_counter_field_t* field) {
  return *(const uint64_t*)((const uint8_t*)counters + field->offset);
}

static iree_status_t loom_serve_http_metrics_counters(
    loom_serve_registry_t* registry, iree_string_builder_t* output) {
  loom_serve_telemetry_counters_t total;
  const bool available = loom_serve_registry_counters(registry, &total);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "# HELP loom_serve_server_counters_available Every model publishes work "
      "counters.\n"
      "# TYPE loom_serve_server_counters_available gauge\n"
      "loom_serve_server_counters_available %d\n",
      available));
  for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(metrics_counter_fields);
       ++j) {
    const loom_serve_metrics_counter_field_t* field =
        &metrics_counter_fields[j];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "# HELP loom_serve_model_%s_total %s\n"
        "# TYPE loom_serve_model_%s_total counter\n",
        field->name, field->description, field->name));
    for (iree_host_size_t i = 0; i < loom_serve_registry_capacity(registry);
         ++i) {
      const loom_serve_registered_model_t* model =
          loom_serve_registry_at(registry, i);
      const loom_serve_service_t* service = &model->service;
      if (!service->telemetry.counters) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "loom_serve_model_%s_total{", field->name));
      IREE_RETURN_IF_ERROR(loom_serve_http_metrics_identity(output, model));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "} %" PRIu64 "\n",
          loom_serve_http_metrics_counter_value(service->telemetry.counters,
                                                field)));
    }
    if (available) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          "# HELP loom_serve_server_%s_total %s\n"
          "# TYPE loom_serve_server_%s_total counter\n"
          "loom_serve_server_%s_total %" PRIu64 "\n",
          field->name, field->description, field->name, field->name,
          loom_serve_http_metrics_counter_value(&total, field)));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_metrics_latency_prefix(
    iree_string_builder_t* output, const char* metric, const char* suffix,
    const loom_serve_registered_model_t* model, const char* outcome) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "loom_serve_%s_%s_seconds_%s{", model ? "model" : "server",
      metric, suffix));
  if (model) {
    IREE_RETURN_IF_ERROR(loom_serve_http_metrics_identity(output, model));
  }
  if (outcome) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "%soutcome=\"%s\"", model ? "," : "", outcome));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_http_metrics_histogram(
    iree_string_builder_t* output, const char* metric,
    const loom_serve_registered_model_t* model, const char* outcome,
    const loom_serve_telemetry_latency_histogram_t* histogram) {
  const char* separator = model || outcome ? "," : "";
  uint64_t count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TELEMETRY_LATENCY_BUCKET_COUNT;
       ++i) {
    count += histogram->buckets[i];
    IREE_RETURN_IF_ERROR(loom_serve_http_metrics_latency_prefix(
        output, metric, "bucket", model, outcome));
    if (i + 1 == LOOM_SERVE_TELEMETRY_LATENCY_BUCKET_COUNT) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "%sle=\"+Inf\"} %" PRIu64 "\n", separator, count));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "%sle=\"%.9g\"} %" PRIu64 "\n", separator,
          loom_serve_telemetry_latency_bound_ns(i) / 1e9, count));
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_latency_prefix(
      output, metric, "count", model, outcome));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(output, "} %" PRIu64 "\n", count));
  IREE_RETURN_IF_ERROR(loom_serve_http_metrics_latency_prefix(
      output, metric, "sum", model, outcome));
  return iree_string_builder_append_format(output, "} %.9g\n",
                                           histogram->sum_ns / 1e9);
}

static iree_status_t loom_serve_http_metrics_latency_values(
    iree_string_builder_t* output, const loom_serve_registered_model_t* model,
    const loom_serve_telemetry_latency_t* latency) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(latency->outcomes); ++i) {
    IREE_RETURN_IF_ERROR(loom_serve_http_metrics_histogram(
        output, "request_duration", model,
        loom_serve_telemetry_outcome_name((loom_serve_telemetry_outcome_t)i),
        &latency->outcomes[i]));
  }
  return loom_serve_http_metrics_histogram(output, "first_output", model, NULL,
                                           &latency->first_output);
}

static iree_status_t loom_serve_http_metrics_latency(
    loom_serve_registry_t* registry, iree_string_builder_t* output) {
  loom_serve_telemetry_latency_t total;
  const bool available = loom_serve_registry_latency(registry, &total);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "# HELP loom_serve_server_latency_available Every model publishes "
      "request latency distributions.\n"
      "# TYPE loom_serve_server_latency_available gauge\n"
      "loom_serve_server_latency_available %d\n"
      "# HELP loom_serve_model_request_duration_seconds Model protocol receipt "
      "through terminal owner retirement, not peer receipt.\n"
      "# TYPE loom_serve_model_request_duration_seconds histogram\n"
      "# HELP loom_serve_model_first_output_seconds Model protocol receipt "
      "through first meaningful output accepted by the carrier, not peer "
      "receipt.\n"
      "# TYPE loom_serve_model_first_output_seconds histogram\n",
      available));
  for (iree_host_size_t i = 0; i < loom_serve_registry_capacity(registry);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    if (model->service.telemetry.latency) {
      IREE_RETURN_IF_ERROR(loom_serve_http_metrics_latency_values(
          output, model, model->service.telemetry.latency));
    }
  }
  if (available) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_server_request_duration_seconds Model protocol "
        "receipt through terminal owner retirement, including removed "
        "registrations.\n"
        "# TYPE loom_serve_server_request_duration_seconds histogram\n"
        "# HELP loom_serve_server_first_output_seconds Model protocol receipt "
        "through first meaningful output accepted by the carrier, including "
        "removed registrations.\n"
        "# TYPE loom_serve_server_first_output_seconds histogram\n"));
    IREE_RETURN_IF_ERROR(
        loom_serve_http_metrics_latency_values(output, NULL, &total));
  }
  return iree_ok_status();
}

iree_status_t loom_serve_http_metrics_format(loom_serve_registry_t* registry,
                                             iree_string_builder_t* output) {
  for (iree_host_size_t i = 0; i < loom_serve_registry_capacity(registry);
       ++i) {
    loom_serve_registered_model_t* model = loom_serve_registry_at(registry, i);
    model->snapshot = loom_serve_registry_snapshot(model);
  }
  iree_string_builder_reset(output);
  iree_status_t status = iree_string_builder_append_cstring(
      output,
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
      "Connection: close\r\n\r\n"
      "# HELP loom_serve_model_queued_requests Requests outside the active "
      "invocation or cohort.\n"
      "# TYPE loom_serve_model_queued_requests gauge\n");
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status =
        loom_serve_http_metrics_sample(output, "queued_requests", model, NULL,
                                       NULL, model->snapshot.queued_requests);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_model_active_requests Requests owned by the active "
        "invocation or cohort, not hardware utilization.\n"
        "# TYPE loom_serve_model_active_requests gauge\n");
  }
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status =
        loom_serve_http_metrics_sample(output, "active_requests", model, NULL,
                                       NULL, model->snapshot.active_requests);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_model_phase Current scheduler ownership phase.\n"
        "# TYPE loom_serve_model_phase gauge\n");
  }
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status = loom_serve_http_metrics_sample(output, "phase", model, "phase",
                                            model->snapshot.phase, 1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_model_oldest_queued_age_seconds Time since the "
        "oldest "
        "queued request was accepted, or zero if empty.\n"
        "# TYPE loom_serve_model_oldest_queued_age_seconds gauge\n");
  }
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status = loom_serve_http_metrics_sample(
        output, "oldest_queued_age_seconds", model, NULL, NULL,
        model->snapshot.oldest_queued_age / 1e9);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_model_oldest_active_age_seconds Time since the "
        "oldest "
        "active request was accepted, including prior queue residence, or zero "
        "if empty.\n"
        "# TYPE loom_serve_model_oldest_active_age_seconds gauge\n");
  }
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status = loom_serve_http_metrics_sample(
        output, "oldest_active_age_seconds", model, NULL, NULL,
        model->snapshot.oldest_active_age / 1e9);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        output,
        "# HELP loom_serve_model_held_requests Requests held outside model "
        "admission.\n"
        "# TYPE loom_serve_model_held_requests gauge\n"
        "# HELP loom_serve_model_admission_open Whether new requests may enter "
        "the model.\n"
        "# TYPE loom_serve_model_admission_open gauge\n");
  }
  for (iree_host_size_t i = 0;
       i < loom_serve_registry_capacity(registry) && iree_status_is_ok(status);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    status = loom_serve_http_metrics_sample(output, "held_requests", model,
                                            NULL, NULL, model->held_count);
    if (iree_status_is_ok(status)) {
      status = loom_serve_http_metrics_sample(output, "admission_open", model,
                                              NULL, NULL, !model->closed);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_metrics_parameters(registry, output);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_metrics_counters(registry, output);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_metrics_latency(registry, output);
  }
  return status;
}
