// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/resources.h"

#include <inttypes.h>

#include "iree/base/internal/json.h"
#include "loom/runtime/storage/telemetry.h"
#include "loom/serve/runtime/execution/workspace.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/telemetry/sample.h"
#include "loom/util/json.h"

iree_host_size_t loom_serve_resources_observation_count(
    const loom_serve_resources_t* resources) {
  return resources->device_group
             ? loom_serve_device_group_count(resources->device_group) +
                   loom_serve_device_group_pool_count(resources->device_group)
             : 0;
}

iree_status_t loom_serve_resources_sample(
    const loom_serve_resources_t* resources, iree_host_size_t pool_index,
    loom_runtime_telemetry_sample_sink_t sink) {
  loom_serve_pool_t* pool =
      loom_serve_device_group_pool_at(resources->device_group, pool_index);
  const loom_serve_memory_pool_t* retained = loom_serve_pool_memory(pool);
  const loom_serve_pool_statistics_t memory = loom_serve_pool_statistics(pool);
  IREE_RETURN_IF_ERROR(
      loom_runtime_storage_sample_pool("storage", &memory.storage, sink));
  if (retained) {
    static const loom_runtime_telemetry_metric_t fields[] = {
        {"reserved_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"committed_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"cached_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"pending_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"releasing_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"abandoned_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"peak_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"budget_limit", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE},
        {"released_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_COUNTER},
    };
    const loom_runtime_telemetry_topic_t topic = {
        "elastic", 1, IREE_ARRAYSIZE(fields), fields};
    const uint64_t limit = loom_serve_memory_pool_limit(retained);
    const uint64_t values[] = {
        memory.retained.reserved_bytes,  memory.retained.committed_bytes,
        memory.retained.cached_bytes,    memory.retained.pending_bytes,
        memory.retained.releasing_bytes, memory.retained.abandoned_bytes,
        memory.retained.peak_bytes,      limit,
        memory.retained.released_bytes,
    };
    IREE_RETURN_IF_ERROR(sink.emit(sink.self, &topic, values));
  }
  return loom_runtime_storage_sample_pool("workspace", &memory.workspace, sink);
}

typedef struct loom_serve_resources_topic_writer_t {
  // Borrowed formatter bounded by the current-state page extent.
  iree_string_builder_t* output;
  // Separates topics without assuming any particular provider's first topic.
  bool has_topic;
} loom_serve_resources_topic_writer_t;

static iree_status_t loom_serve_resources_write_topic(
    void* self, const loom_runtime_telemetry_topic_t* topic,
    const uint64_t* values) {
  loom_serve_resources_topic_writer_t* writer = self;
  if (writer->has_topic) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(writer->output, ","));
  }
  writer->has_topic = true;
  return loom_serve_telemetry_sample_write_json(topic, values, writer->output);
}

static iree_status_t loom_serve_resources_write_device_observation(
    const loom_serve_resources_t* resources, iree_host_size_t device_index,
    iree_string_builder_t* output) {
  loom_serve_device_t* device =
      loom_serve_device_group_at(resources->device_group, device_index);
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(
          iree_hal_device_spec(loom_serve_device_handle(device)));
  const iree_string_view_t key = loom_serve_device_name(device);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"type\":\"device\",\"index\":%zu,\"key\":", device_index));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_string_view_substr(key, 0, 256)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"key_truncated\":%s,\"name\":", key.size > 256 ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, identity->display_name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"driver\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, identity->driver_id));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"backend\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, identity->backend_id));
  return iree_string_builder_append_cstring(output, ",\"topics\":[]}");
}

static iree_status_t loom_serve_resources_write_members(
    const loom_serve_resources_t* resources, uint64_t device_mask,
    loom_output_stream_t* output) {
  iree_status_t status = loom_output_stream_write_cstring(output, "[");
  bool first = true;
  for (iree_host_size_t i = 0;
       i < loom_serve_device_group_count(resources->device_group) &&
       iree_status_is_ok(status);
       ++i) {
    if (!(device_mask & (UINT64_C(1) << i))) {
      continue;
    }
    status =
        loom_output_stream_write_format(output, "%s%zu", first ? "" : ",", i);
    first = false;
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "]");
  }
  return status;
}

iree_status_t loom_serve_resources_write_observation(
    const loom_serve_resources_t* resources, iree_host_size_t resource_index,
    iree_string_builder_t* output) {
  const iree_host_size_t device_count =
      loom_serve_device_group_count(resources->device_group);
  if (resource_index < device_count) {
    return loom_serve_resources_write_device_observation(
        resources, resource_index, output);
  }
  const iree_host_size_t pool_index = resource_index - device_count;
  const loom_serve_pool_binding_t* binding =
      &loom_serve_device_group_pool_bindings(
          resources->device_group)[pool_index];
  loom_serve_pool_t* pool =
      loom_serve_device_group_pool_at(resources->device_group, pool_index);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"type\":\"pool\",\"index\":%zu,\"key\":", pool_index));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_string_view_substr(binding->name, 0, 256)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"key_truncated\":%s,\"backing\":\"%s\",\"devices\":",
      binding->name.size > 256 ? "true" : "false",
      loom_serve_pool_memory(pool) ? "elastic" : "fixed"));
  IREE_RETURN_IF_ERROR(loom_serve_resources_write_members(
      resources, binding->device_mask, &stream));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"topics\":["));
  loom_serve_resources_topic_writer_t writer = {.output = output};
  IREE_RETURN_IF_ERROR(loom_serve_resources_sample(
      resources, pool_index,
      (loom_runtime_telemetry_sample_sink_t){
          &writer, loom_serve_resources_write_topic}));
  return iree_string_builder_append_cstring(output, "]}");
}

static iree_status_t loom_serve_resources_write_physical_devices(
    const iree_hal_device_identity_spec_t* identity,
    loom_output_stream_t* output) {
  iree_status_t status = loom_output_stream_write_cstring(output, "[");
  for (iree_host_size_t i = 0;
       i < identity->physical_device_count && iree_status_is_ok(status); ++i) {
    const iree_hal_physical_device_spec_t* physical =
        &identity->physical_devices[i];
    status = loom_output_stream_write_format(
        output, "%s{\"ordinal\":%" PRIu32 ",\"name\":", i ? "," : "",
        physical->physical_ordinal);
    if (iree_status_is_ok(status)) {
      status = loom_json_write_escaped_string(output,
                                              physical->identity.display_name);
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_cstring(output, ",\"path\":");
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_write_escaped_string(output,
                                              physical->identity.backend_path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_cstring(output, "}");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "]");
  }
  return status;
}

static iree_status_t loom_serve_resources_write_memory(
    const loom_serve_pool_t* pool, loom_output_stream_t* output) {
  const loom_serve_memory_pool_t* retained = loom_serve_pool_memory(pool);
  const loom_serve_pool_statistics_t memory = loom_serve_pool_statistics(pool);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      "{\"backing\":\"%s\",\"retained\":", retained ? "elastic" : "fixed"));
  if (retained) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(output, "{\"limit_bytes\":"));
    const uint64_t limit = loom_serve_memory_pool_limit(retained);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(output, "%" PRIu64, limit));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output,
        ",\"reserved_bytes\":%" PRIu64 ",\"committed_bytes\":%" PRIu64
        ",\"abandoned_bytes\":%" PRIu64 ",\"cached_bytes\":%" PRIu64
        ",\"pending_bytes\":%" PRIu64 ",\"releasing_bytes\":%" PRIu64
        ",\"peak_bytes\":%" PRIu64 ",\"released_bytes\":%" PRIu64 "}",
        memory.retained.reserved_bytes, memory.retained.committed_bytes,
        memory.retained.abandoned_bytes, memory.retained.cached_bytes,
        memory.retained.pending_bytes, memory.retained.releasing_bytes,
        memory.retained.peak_bytes, memory.retained.released_bytes));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  return loom_output_stream_write_format(
      output,
      ",\"storage\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"abandoned_bytes\":%" PRIu64
      "}"
      ",\"workspace\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"free_bytes\":%" PRIu64
      ",\"quarantined_bytes\":%" PRIu64 ",\"budget_limit\":%" PRIu64
      ",\"reservation_count\":%" PRIu32 ",\"slab_count\":%" PRIu32
      ",\"reserve_count\":%" PRIu64 ",\"release_count\":%" PRIu64
      ",\"reuse_count\":%" PRIu64 ",\"reuse_miss_count\":%" PRIu64
      ",\"fresh_count\":%" PRIu64 ",\"exhausted_count\":%" PRIu64
      ",\"over_budget_count\":%" PRIu64 ",\"wait_count\":%" PRIu64
      ",\"quarantine_eviction_count\":%" PRIu64 "}}",
      memory.storage.bytes_reserved, memory.storage.bytes_committed,
      memory.storage.bytes_abandoned, memory.workspace.bytes_reserved,
      memory.workspace.bytes_committed, memory.workspace.bytes_free,
      memory.workspace.bytes_quarantined, memory.workspace.budget_limit,
      memory.workspace.reservation_count, memory.workspace.slab_count,
      memory.workspace.reserve_count, memory.workspace.release_count,
      memory.workspace.reuse_count, memory.workspace.reuse_miss_count,
      memory.workspace.fresh_count, memory.workspace.exhausted_count,
      memory.workspace.over_budget_count, memory.workspace.wait_count,
      memory.workspace.quarantine_eviction_count);
}

static iree_status_t loom_serve_resources_write_device(
    const loom_serve_resources_t* resources, iree_host_size_t device_index,
    loom_output_stream_t* output) {
  loom_serve_device_t* device =
      loom_serve_device_group_at(resources->device_group, device_index);
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(
          iree_hal_device_spec(loom_serve_device_handle(device)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "{\"key\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(output, loom_serve_device_name(device)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, ",\"id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(output, identity->logical_device_id));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, ",\"name\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(output, identity->display_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"driver\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(output, identity->driver_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"backend\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(output, identity->backend_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"physical_devices\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_resources_write_physical_devices(identity, output));
  return loom_output_stream_write_cstring(output, "}");
}

static iree_status_t loom_serve_resources_write_pool(
    const loom_serve_resources_t* resources, iree_host_size_t pool_index,
    loom_output_stream_t* output) {
  const loom_serve_pool_binding_t* binding =
      &loom_serve_device_group_pool_bindings(
          resources->device_group)[pool_index];
  const loom_serve_pool_idle_policy_t idle =
      resources->idle_policy_count ? resources->idle_policies[pool_index]
                                   : (loom_serve_pool_idle_policy_t){0};
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "{\"key\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(output, binding->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"devices\":"));
  IREE_RETURN_IF_ERROR(loom_serve_resources_write_members(
      resources, binding->device_mask, output));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      ",\"idle\":{\"workspace_ms\":%" PRId64 ",\"parameter_ms\":%" PRId64
      "},\"memory\":",
      idle.workspace_timeout / 1000000, idle.parameter_timeout / 1000000));
  IREE_RETURN_IF_ERROR(loom_serve_resources_write_memory(
      loom_serve_device_group_pool_at(resources->device_group, pool_index),
      output));
  return loom_output_stream_write_cstring(output, "}");
}

iree_status_t loom_serve_resources_write_json(
    const loom_serve_resources_t* resources, loom_output_stream_t* output) {
  iree_status_t status =
      loom_output_stream_write_cstring(output, "{\"listeners\":[");
  for (iree_host_size_t i = 0;
       i < resources->listener_count && iree_status_is_ok(status); ++i) {
    char storage[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t address;
    status = iree_async_address_format(
        loom_serve_http_listener_address(resources->listeners[i]),
        sizeof(storage), storage, &address);
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_cstring(
          output, i ? ",{\"address\":" : "{\"address\":");
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_write_escaped_string(output, address);
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write_format(
          output, ",\"waiting_for_capacity\":%s}",
          loom_serve_http_listener_is_waiting_for_capacity(
              resources->listeners[i])
              ? "true"
              : "false");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "],\"devices\":[");
  }
  const iree_host_size_t device_count =
      resources->device_group
          ? loom_serve_device_group_count(resources->device_group)
          : 0;
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    if (i) {
      status = loom_output_stream_write_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_resources_write_device(resources, i, output);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "],\"pools\":[");
  }
  const iree_host_size_t pool_count =
      resources->device_group
          ? loom_serve_device_group_pool_count(resources->device_group)
          : 0;
  for (iree_host_size_t i = 0; i < pool_count && iree_status_is_ok(status);
       ++i) {
    if (i) {
      status = loom_output_stream_write_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_resources_write_pool(resources, i, output);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "]}");
  }
  return status;
}

typedef struct loom_serve_resources_trim_fields_t {
  // Borrowed escaped key; null data marks an absent pool field.
  iree_string_view_t pool;
  // Borrowed JSON integer spelling; null data marks an absent target field.
  iree_string_view_t target_bytes;
} loom_serve_resources_trim_fields_t;

static iree_status_t loom_serve_resources_trim_field(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_resources_trim_fields_t* fields = self;
  iree_string_view_t* field = NULL;
  iree_json_value_type_t expected = IREE_JSON_VALUE_TYPE_NUMBER;
  if (iree_string_view_equal(key, IREE_SV("pool"))) {
    field = &fields->pool;
    expected = IREE_JSON_VALUE_TYPE_STRING;
  } else if (iree_string_view_equal(key, IREE_SV("target_bytes"))) {
    field = &fields->target_bytes;
  }
  if (!field || field->data || type != expected) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown, duplicate or wrongly typed trim field "
                            "'%.*s'",
                            (int)key.size, key.data);
  }
  *field = value;
  return iree_ok_status();
}

iree_status_t loom_serve_resources_parse_trim(
    const loom_serve_resources_t* resources, iree_string_view_t value,
    iree_allocator_t host_allocator, iree_host_size_t* out_pool_index,
    uint64_t* out_target_bytes) {
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&value, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&value));
  if (value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing trim request JSON");
  }
  loom_serve_resources_trim_fields_t fields = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_resources_trim_field, &fields));
  if (!fields.pool.data || !fields.target_bytes.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trim requires pool and target_bytes");
  }
  uint64_t target_bytes = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_parse_uint64(fields.target_bytes, &target_bytes));
  if (!fields.pool.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trim requires a nonempty pool key");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, fields.pool.size,
                                             (void**)&storage));
  iree_host_size_t length = 0;
  iree_status_t status = iree_json_unescape_string(
      fields.pool, fields.pool.size, storage, &length);
  const iree_host_size_t pool_count =
      resources->device_group
          ? loom_serve_device_group_pool_count(resources->device_group)
          : 0;
  const loom_serve_pool_binding_t* pools =
      resources->device_group
          ? loom_serve_device_group_pool_bindings(resources->device_group)
          : NULL;
  iree_host_size_t index = 0;
  if (iree_status_is_ok(status)) {
    const iree_string_view_t name = iree_make_string_view(storage, length);
    while (index < pool_count &&
           !iree_string_view_equal(name, pools[index].name)) {
      ++index;
    }
    if (index == pool_count) {
      status =
          iree_make_status(IREE_STATUS_NOT_FOUND, "unknown trim pool '%.*s'",
                           (int)name.size, name.data);
    }
  }
  iree_allocator_free(host_allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_pool_index = index;
    *out_target_bytes = target_bytes;
  }
  return status;
}

iree_status_t loom_serve_resources_trim(const loom_serve_resources_t* resources,
                                        iree_host_size_t pool_index,
                                        uint64_t target_bytes,
                                        loom_output_stream_t* output) {
  IREE_RETURN_IF_ERROR(loom_serve_pool_trim(
      loom_serve_device_group_pool_at(resources->device_group, pool_index),
      target_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output, "{\"target_bytes\":%" PRIu64 ",\"pool\":", target_bytes));
  IREE_RETURN_IF_ERROR(
      loom_serve_resources_write_pool(resources, pool_index, output));
  return loom_output_stream_write_cstring(output, "}");
}

static iree_status_t loom_serve_resources_program_trim_field(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  iree_string_view_t* name = self;
  if (!iree_string_view_equal(key, IREE_SV("device")) || name->data ||
      type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program trim requires only a device key");
  }
  *name = value;
  return iree_ok_status();
}

iree_status_t loom_serve_resources_parse_trim_programs(
    const loom_serve_resources_t* resources, iree_string_view_t value,
    iree_allocator_t host_allocator, iree_host_size_t* out_device_index) {
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&value, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&value));
  if (value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing program trim request JSON");
  }
  iree_string_view_t escaped = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_resources_program_trim_field, &escaped));
  if (!escaped.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "program trim requires a nonempty device key");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, escaped.size, (void**)&storage));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(escaped, escaped.size, storage, &length);
  const iree_string_view_list_t names =
      resources->device_group
          ? loom_serve_device_group_names(resources->device_group)
          : (iree_string_view_list_t){0};
  iree_host_size_t index = 0;
  if (iree_status_is_ok(status)) {
    const iree_string_view_t name = iree_make_string_view(storage, length);
    while (index < names.count &&
           !iree_string_view_equal(name, names.values[index])) {
      ++index;
    }
    if (index == names.count) {
      status =
          iree_make_status(IREE_STATUS_NOT_FOUND, "unknown trim device '%.*s'",
                           (int)name.size, name.data);
    }
  }
  iree_allocator_free(host_allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_device_index = index;
  }
  return status;
}

iree_status_t loom_serve_resources_trim_programs(
    const loom_serve_resources_t* resources, iree_host_size_t device_index,
    loom_output_stream_t* output) {
  loom_serve_device_trim_programs(
      loom_serve_device_group_at(resources->device_group, device_index));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, "{\"device\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_resources_write_device(resources, device_index, output));
  return loom_output_stream_write_cstring(output, "}");
}
