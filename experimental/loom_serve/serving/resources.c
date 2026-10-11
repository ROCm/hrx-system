// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/resources.h"

#include <inttypes.h>

#include "iree/base/internal/json.h"
#include "loom/util/json.h"

static iree_status_t resources_write_physical_devices(
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

static iree_status_t resources_write_memory(const loom_serve_device_t* device,
                                            loom_output_stream_t* output) {
  const loom_serve_memory_pool_t* retained =
      loom_serve_device_memory_pool(device);
  const loom_serve_device_memory_statistics_t memory =
      loom_serve_device_memory_statistics(device);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      "{\"backing\":\"%s\",\"retained\":", retained ? "elastic" : "fixed"));
  if (retained) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(output, "{\"limit_bytes\":"));
    const uint64_t limit = loom_serve_memory_pool_limit(retained);
    if (limit == UINT64_MAX) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_format(output, "%" PRIu64, limit));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output,
        ",\"reserved_bytes\":%" PRIu64 ",\"committed_bytes\":%" PRIu64
        ",\"peak_bytes\":%" PRIu64 ",\"released_bytes\":%" PRIu64 "}",
        memory.retained.reserved_bytes, memory.retained.committed_bytes,
        memory.retained.peak_bytes, memory.retained.released_bytes));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  return loom_output_stream_write_format(
      output,
      ",\"workspace\":{\"reserved_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 ",\"free_bytes\":%" PRIu64
      ",\"quarantined_bytes\":%" PRIu64 ",\"budget_limit\":%" PRIu64
      ",\"reservation_count\":%" PRIu32 "}}",
      memory.workspace.bytes_reserved, memory.workspace.bytes_committed,
      memory.workspace.bytes_free, memory.workspace.bytes_quarantined,
      memory.workspace.budget_limit, memory.workspace.reservation_count);
}

static iree_status_t resources_write_device(loom_serve_device_t* device,
                                            iree_host_size_t index,
                                            loom_output_stream_t* output) {
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(
          iree_hal_device_spec(loom_serve_device_handle(device)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(output, "{\"index\":%zu,\"id\":", index));
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
  IREE_RETURN_IF_ERROR(resources_write_physical_devices(identity, output));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"memory\":"));
  IREE_RETURN_IF_ERROR(resources_write_memory(device, output));
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
      status = loom_output_stream_write_cstring(output, "}");
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
      status = resources_write_device(
          loom_serve_device_group_at(resources->device_group, i), i, output);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(output, "]}");
  }
  return status;
}

typedef struct resources_trim_fields_t {
  // Borrowed JSON integer spelling; null data marks an absent device field.
  iree_string_view_t device;
  // Borrowed JSON integer spelling; null data marks an absent target field.
  iree_string_view_t target_bytes;
} resources_trim_fields_t;

static iree_status_t resources_trim_field(void* self, iree_string_view_t key,
                                          iree_json_value_type_t type,
                                          iree_string_view_t value) {
  resources_trim_fields_t* fields = self;
  iree_string_view_t* field = NULL;
  if (iree_string_view_equal(key, IREE_SV("device"))) {
    field = &fields->device;
  } else if (iree_string_view_equal(key, IREE_SV("target_bytes"))) {
    field = &fields->target_bytes;
  }
  if (!field || field->data || type != IREE_JSON_VALUE_TYPE_NUMBER) {
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
    iree_host_size_t* out_device_index, uint64_t* out_target_bytes) {
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&value, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&value));
  if (value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing trim request JSON");
  }
  resources_trim_fields_t fields = {0};
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(object, resources_trim_field, &fields));
  if (!fields.device.data || !fields.target_bytes.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trim requires device and target_bytes");
  }
  uint64_t index = 0;
  uint64_t target_bytes = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(fields.device, &index));
  IREE_RETURN_IF_ERROR(
      iree_json_parse_uint64(fields.target_bytes, &target_bytes));
  const iree_host_size_t count =
      resources->device_group
          ? loom_serve_device_group_count(resources->device_group)
          : 0;
  if (index >= count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trim device index %" PRIu64
                            " is outside the published device group",
                            index);
  }
  *out_device_index = (iree_host_size_t)index;
  *out_target_bytes = target_bytes;
  return iree_ok_status();
}

iree_status_t loom_serve_resources_trim(const loom_serve_resources_t* resources,
                                        iree_host_size_t device_index,
                                        uint64_t target_bytes,
                                        loom_output_stream_t* output) {
  loom_serve_device_t* device =
      loom_serve_device_group_at(resources->device_group, device_index);
  IREE_RETURN_IF_ERROR(loom_serve_device_trim(device, target_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output, "{\"target_bytes\":%" PRIu64 ",\"device\":", target_bytes));
  IREE_RETURN_IF_ERROR(resources_write_device(device, device_index, output));
  return loom_output_stream_write_cstring(output, "}");
}
