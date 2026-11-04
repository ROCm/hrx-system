// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/observation.h"

#include <inttypes.h>

#include "loom/serve/service/kind.h"
#include "loom/serve/telemetry/sample.h"
#include "loom/serve/telemetry/stream.h"
#include "loom/util/json.h"

typedef struct loom_serve_observation_source_t {
  // Owned serialized source facts, independent of the service lifetime.
  iree_string_builder_t content;
  // Content identity and its sampled unchanged boundary.
  loom_serve_observation_version_t version;
} loom_serve_observation_source_t;

struct loom_serve_observation_t {
  // Allocator for metadata and lazily grown, reusable source storage.
  iree_allocator_t host_allocator;
  // Borrowed registration truth, accessed only by the serialized owner.
  loom_serve_registry_t* registry;
  // Borrowed application resources, live until every observer has retired.
  const loom_serve_resources_t* resources;
  // Immutable resource sources preceding model slots.
  iree_host_size_t resource_count;
  // Number of registry slots, excluding server, device, and pool sources.
  iree_host_size_t capacity;
  // Monotonic revision advanced by exact source-content changes.
  uint64_t revision;
  // Server, resources, then registry-slot sources; empty content is absence.
  loom_serve_observation_source_t sources[];
};

iree_status_t loom_serve_observation_allocate(
    loom_serve_registry_t* registry, const loom_serve_resources_t* resources,
    iree_allocator_t host_allocator,
    loom_serve_observation_t** out_observation) {
  *out_observation = NULL;
  const iree_host_size_t capacity = loom_serve_registry_capacity(registry);
  const iree_host_size_t resource_count =
      loom_serve_resources_observation_count(resources);
  loom_serve_observation_t* observation = NULL;
  iree_host_size_t source_count = 0;
  if (!iree_host_size_checked_add(capacity, resource_count, &source_count) ||
      !iree_host_size_checked_add(source_count, 1, &source_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "observation source inventory exceeds addressable storage");
  }
  iree_host_size_t total_size = 0, sources_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*observation), &total_size,
      IREE_STRUCT_FIELD(source_count, loom_serve_observation_source_t,
                        &sources_offset)));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&observation));
  observation->host_allocator = host_allocator;
  observation->registry = registry;
  observation->resources = resources;
  observation->resource_count = resource_count;
  observation->capacity = capacity;
  for (iree_host_size_t i = 0; i <= capacity + resource_count; ++i) {
    iree_string_builder_initialize(host_allocator,
                                   &observation->sources[i].content);
  }
  *out_observation = observation;
  return iree_ok_status();
}

void loom_serve_observation_free(loom_serve_observation_t* observation) {
  if (!observation) {
    return;
  }
  for (iree_host_size_t i = 0;
       i <= observation->capacity + observation->resource_count; ++i) {
    iree_string_builder_deinitialize(&observation->sources[i].content);
  }
  iree_allocator_free(observation->host_allocator, observation);
}

uint64_t loom_serve_observation_revision(
    const loom_serve_observation_t* observation) {
  return observation->revision;
}

static iree_status_t loom_serve_observation_update(
    loom_serve_observation_t* observation, iree_host_size_t index,
    iree_time_t now, iree_string_view_t content) {
  loom_serve_observation_source_t* source = &observation->sources[index];
  if (!iree_string_view_equal(iree_string_builder_view(&source->content),
                              content)) {
    // Reserve before changing either the copied facts or their revision.
    IREE_RETURN_IF_ERROR(
        iree_string_builder_reserve(&source->content, content.size + 1));
    iree_string_builder_reset(&source->content);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(&source->content, content));
    source->version.previous_revision = source->version.revision;
    source->version.previous_time_ns = source->version.time_ns;
    source->version.revision = ++observation->revision;
  }
  source->version.time_ns = now;
  return iree_ok_status();
}

static iree_status_t loom_serve_observation_counters(
    const loom_serve_telemetry_counters_t* counters,
    iree_string_builder_t* output) {
  if (!counters) {
    return iree_string_builder_append_cstring(output, "null");
  }
  return iree_string_builder_append_format(
      output,
      "{\"requests\":\"%" PRIu64 "\",\"completed\":\"%" PRIu64
      "\",\"rejected\":\"%" PRIu64 "\",\"cancelled\":\"%" PRIu64
      "\",\"failed\":\"%" PRIu64 "\",\"input_tokens\":\"%" PRIu64
      "\",\"output_tokens\":\"%" PRIu64 "\",\"batches\":\"%" PRIu64 "\"}",
      counters->requests, counters->completed, counters->rejected,
      counters->cancelled, counters->failed, counters->input_tokens,
      counters->output_tokens, counters->batches);
}

static iree_status_t loom_serve_observation_topic(
    void* self, const loom_runtime_telemetry_topic_t* topic,
    const uint64_t* values) {
  iree_string_builder_t* output = self;
  // Each extension follows the common topic's array entry.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ","));
  return loom_serve_telemetry_sample_write_json(topic, values, output);
}

static iree_status_t loom_serve_observation_epoch(
    iree_time_t since_ns, iree_string_builder_t* output) {
  return since_ns
             ? iree_string_builder_append_format(
                   output, ",\"since_ns\":\"%" PRId64 "\"", since_ns)
             : iree_string_builder_append_cstring(output, ",\"since_ns\":null");
}

static iree_status_t loom_serve_observation_model(
    const loom_serve_registered_model_t* model, iree_host_size_t index,
    iree_string_builder_t* output) {
  const loom_serve_service_t* service = &model->service;
  const loom_serve_service_snapshot_t snapshot =
      loom_serve_registry_snapshot(model);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"slot\":%zu,\"incarnation\":\"%" PRIu64 "\",\"name\":", index,
      model->incarnation));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_string_view_substr(service->name, 0, 256)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"name_truncated\":%s,\"kind\":\"%s\",\"phase\":",
      service->name.size > 256 ? "true" : "false",
      loom_serve_model_kind_name(service->kind)));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_make_cstring_view(snapshot.phase)));
  if (model->configuration.device.size) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"device\":%zu", model->configuration.device_index));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"device\":null"));
  }
  if (model->configuration.pool.size) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"pool\":%zu", model->configuration.pool_index));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"pool\":null"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"queued\":%zu,\"active\":%zu,\"held\":%zu,\"admission_open\":%s,"
      "\"counters\":",
      snapshot.queued_requests, snapshot.active_requests, model->held_count,
      model->closed ? "false" : "true"));
  IREE_RETURN_IF_ERROR(
      loom_serve_observation_counters(service->telemetry.counters, output));
  IREE_RETURN_IF_ERROR(loom_serve_observation_epoch(
      service->telemetry.stream
          ? loom_serve_telemetry_stream_identity(service->telemetry.stream)
                .created_ns
          : 0,
      output));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"topics\":["));
  static const loom_runtime_telemetry_metric_t fields[] = {
      {"oldest_queued_age", "nanoseconds", LOOM_RUNTIME_TELEMETRY_GAUGE},
      {"oldest_active_age", "nanoseconds", LOOM_RUNTIME_TELEMETRY_GAUGE},
  };
  const loom_runtime_telemetry_topic_t topic = {"activity", 1,
                                                IREE_ARRAYSIZE(fields), fields};
  const uint64_t values[] = {(uint64_t)snapshot.oldest_queued_age,
                             (uint64_t)snapshot.oldest_active_age};
  IREE_RETURN_IF_ERROR(
      loom_serve_telemetry_sample_write_json(&topic, values, output));
  if (model->phase == LOOM_SERVE_REGISTRATION_READY &&
      service->telemetry.sample) {
    IREE_RETURN_IF_ERROR(service->telemetry.sample(
        service->self, (loom_runtime_telemetry_sample_sink_t){
                           output, loom_serve_observation_topic}));
  }
  return iree_string_builder_append_cstring(output, "]}");
}

iree_status_t loom_serve_observation_sample(
    loom_serve_observation_t* observation, iree_time_t now,
    iree_host_size_t* cursor, iree_string_builder_t* output) {
  loom_serve_registry_t* registry = observation->registry;
  iree_string_builder_reset(output);
  if (!*cursor) {
    loom_serve_telemetry_counters_t counters;
    const bool available = loom_serve_registry_counters(registry, &counters);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "{\"type\":\"server\",\"model_capacity\":%zu,\"device_count\":%zu,"
        "\"pool_count\":%zu,\"counters\":",
        loom_serve_registry_capacity(registry),
        observation->resources->device_group
            ? loom_serve_device_group_count(
                  observation->resources->device_group)
            : 0,
        observation->resources->device_group
            ? loom_serve_device_group_pool_count(
                  observation->resources->device_group)
            : 0));
    IREE_RETURN_IF_ERROR(
        loom_serve_observation_counters(available ? &counters : NULL, output));
    IREE_RETURN_IF_ERROR(loom_serve_observation_epoch(
        loom_serve_registry_counters_since_ns(registry), output));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
    IREE_RETURN_IF_ERROR(loom_serve_observation_update(
        observation, 0, now, iree_string_builder_view(output)));
    *cursor = 1;
    return iree_ok_status();
  }
  if (*cursor <= observation->resource_count) {
    IREE_RETURN_IF_ERROR(loom_serve_resources_write_observation(
        observation->resources, *cursor - 1, output));
    IREE_RETURN_IF_ERROR(loom_serve_observation_update(
        observation, *cursor, now, iree_string_builder_view(output)));
    ++*cursor;
    return iree_ok_status();
  }
  const iree_host_size_t first = *cursor - observation->resource_count - 1;
  const iree_host_size_t capacity = loom_serve_registry_capacity(registry);
  const iree_host_size_t end = first + iree_min(capacity - first, 64);
  iree_host_size_t next = first;
  iree_status_t status = iree_ok_status();
  while (next < end && iree_status_is_ok(status)) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, next++);
    if (model->service.name.size) {
      status = loom_serve_observation_model(model, next - 1, output);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_observation_update(
          observation, next + observation->resource_count, now,
          iree_string_builder_view(output));
    }
    if (model->service.name.size) {
      break;
    }
  }
  if (iree_status_is_ok(status)) {
    *cursor = next == capacity ? 0 : next + observation->resource_count + 1;
  }
  return status;
}

iree_status_t loom_serve_observation_write_page(
    const loom_serve_observation_t* observation, iree_host_size_t* cursor,
    iree_string_builder_t* output,
    loom_serve_observation_version_t* out_version) {
  *out_version = (loom_serve_observation_version_t){0};
  if (*cursor <= observation->resource_count) {
    const loom_serve_observation_source_t* source =
        &observation->sources[*cursor];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, iree_string_builder_view(&source->content)));
    *out_version = source->version;
    ++*cursor;
    return iree_ok_status();
  }
  const iree_host_size_t first = *cursor - observation->resource_count - 1;
  const iree_host_size_t capacity = observation->capacity;
  const iree_host_size_t end = first + iree_min(capacity - first, 64);
  iree_host_size_t next = first;
  const loom_serve_observation_source_t* source = NULL;
  while (next < end && !source) {
    const loom_serve_observation_source_t* candidate =
        &observation->sources[++next + observation->resource_count];
    if (iree_string_builder_size(&candidate->content)) {
      source = candidate;
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "{\"type\":\"models\",\"first\":%zu,\"next\":%zu,\"done\":%s,\"model\":",
      first, next, next == capacity ? "true" : "false"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      output,
      source ? iree_string_builder_view(&source->content) : IREE_SV("null")));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  if (source) {
    *out_version = source->version;
  } else {
    // Empty ranges were sampled too; sending later does not refresh absence.
    out_version->time_ns =
        observation->sources[next + observation->resource_count]
            .version.time_ns;
  }
  *cursor = next == capacity ? 0 : next + observation->resource_count + 1;
  return iree_ok_status();
}
