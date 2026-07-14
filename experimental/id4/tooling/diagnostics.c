// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/diagnostics.h"

#include <errno.h>
#include <string.h>

#include "experimental/id4/pipeline/json.h"
#include "experimental/id4/tooling/filesystem.h"

static iree_string_view_t id4_tooling_diagnostics_event_kind_name(
    id4_pipeline_diagnostic_event_kind_t kind) {
  switch (kind) {
    case ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE:
      return IREE_SV("lifecycle");
    case ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PLAN:
      return IREE_SV("plan");
    case ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB:
      return IREE_SV("parameter_slab");
    case ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL:
      return IREE_SV("kernel");
    case ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_TIMING:
      return IREE_SV("timing");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t id4_tooling_diagnostics_append_json_field_string(
    iree_string_builder_t* builder, const char* key, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", key));
  return id4_pipeline_json_append_string(builder, value);
}

static iree_status_t id4_tooling_diagnostics_append_host_size_or_null(
    iree_string_builder_t* builder, iree_host_size_t value) {
  if (value == IREE_HOST_SIZE_MAX) {
    return iree_string_builder_append_cstring(builder, "null");
  }
  return iree_string_builder_append_format(builder, "%" PRIu64,
                                           (uint64_t)value);
}

static iree_status_t id4_tooling_diagnostics_append_parameter_slab_json(
    iree_string_builder_t* builder,
    const id4_pipeline_parameter_slab_diagnostic_t* parameter_slab) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"parameter_slab\":{"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "\"slab_index\":%" PRIu64 ",\"request_index\":%" PRIu64
      ",\"placement_id\":%u,\"device_index\":%" PRIu64
      ",\"queue_affinity\":%" PRIu64
      ",\"memory_type\":%u"
      ",\"memory_access\":%u,\"buffer_usage\":%u"
      ",\"parameter_offset\":%" PRIu64 ",\"buffer_offset\":%" PRIu64
      ",\"length\":%" PRIu64 ",\"slab_byte_length\":%" PRIu64
      ",\"slab_alignment\":%" PRIu64 ",\"request_count\":%" PRIu64 ",",
      (uint64_t)parameter_slab->slab_index,
      (uint64_t)parameter_slab->request_index, parameter_slab->placement_id,
      (uint64_t)parameter_slab->device_index,
      (uint64_t)parameter_slab->queue_affinity, parameter_slab->memory_type,
      parameter_slab->memory_access, parameter_slab->buffer_usage,
      parameter_slab->parameter_offset, parameter_slab->buffer_offset,
      parameter_slab->length, (uint64_t)parameter_slab->slab_byte_length,
      (uint64_t)parameter_slab->slab_alignment,
      (uint64_t)parameter_slab->request_count));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "scope", parameter_slab->scope));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "parameter_key", parameter_slab->parameter_key));
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t id4_tooling_diagnostics_append_parameter_load_json(
    iree_string_builder_t* builder,
    const id4_pipeline_parameter_load_diagnostic_t* parameter_load) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"parameter_load\":{\"slab_index\":%" PRIu64 ",\"load_group_index\":",
      (uint64_t)parameter_load->slab_index));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_host_size_or_null(
      builder, parameter_load->load_group_index));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "load_group_kind", parameter_load->load_group_kind));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, ",\"first_consumer_execution_ordinal\":"));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_host_size_or_null(
      builder, parameter_load->first_consumer_execution_ordinal));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, ",\"submit_execution_ordinal\":"));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_host_size_or_null(
      builder, parameter_load->submit_execution_ordinal));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "first_load_step_name", parameter_load->first_load_step_name));
  return iree_string_builder_append_format(
      builder,
      ",\"load_step_offset\":%" PRIu64 ",\"load_step_count\":%" PRIu64
      ",\"staging_slot_count\":%" PRIu64
      ",\"staging_slot_byte_length\":%" PRIu64
      ",\"staging_total_byte_length\":%" PRIu64
      ",\"staging_chunk_count\":%" PRIu64 ",\"logical_source_count\":%" PRIu64
      ",\"source_gather_batch_count\":%" PRIu64
      ",\"source_byte_length\":%" PRIu64 ",\"target_byte_length\":%" PRIu64
      ",\"encoder_dispatch_count\":%" PRIu64 "}",
      (uint64_t)parameter_load->load_step_offset,
      (uint64_t)parameter_load->load_step_count,
      (uint64_t)parameter_load->staging_slot_count,
      (uint64_t)parameter_load->staging_slot_byte_length,
      (uint64_t)parameter_load->staging_total_byte_length,
      (uint64_t)parameter_load->staging_chunk_count,
      (uint64_t)parameter_load->logical_source_count,
      (uint64_t)parameter_load->source_gather_batch_count,
      (uint64_t)parameter_load->source_byte_length,
      (uint64_t)parameter_load->target_byte_length,
      (uint64_t)parameter_load->encoder_dispatch_count);
}

static iree_status_t id4_tooling_diagnostics_append_kernel_json(
    iree_string_builder_t* builder,
    const id4_pipeline_kernel_diagnostic_t* kernel) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"kernel\":{"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "\"config_binding_count\":%" PRIu64 ",\"artifact_byte_length\":%" PRIu64
      ",\"inferred_executable_byte_length\":%" PRIu64
      ",\"diagnostic_index\":%" PRIu64
      ",\"diagnostic_severity\":%d"
      ",\"queue_affinity\":%" PRIu64 ",\"caching_mode\":%u,",
      (uint64_t)kernel->config_binding_count,
      (uint64_t)kernel->artifact_byte_length,
      (uint64_t)kernel->inferred_executable_byte_length,
      (uint64_t)kernel->diagnostic_index, kernel->diagnostic_severity,
      (uint64_t)kernel->queue_affinity, (uint32_t)kernel->caching_mode));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "phase", kernel->phase));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "source_identifier", kernel->source_identifier));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "module_path", kernel->module_path));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "target_processor", kernel->target_processor));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "loom_artifact_format", kernel->loom_artifact_format));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "hal_executable_format", kernel->hal_executable_format));
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t id4_tooling_diagnostics_append_timing_json(
    iree_string_builder_t* builder,
    const id4_pipeline_timing_diagnostic_t* timing) {
  return iree_string_builder_append_format(
      builder,
      ",\"timing\":{\"start_time_ns\":%" PRIi64 ",\"end_time_ns\":%" PRIi64
      ",\"duration_ns\":%" PRIi64 "}",
      timing->start_time_ns, timing->end_time_ns, timing->duration_ns);
}

static iree_status_t id4_tooling_diagnostics_format_event(
    const id4_pipeline_diagnostic_event_t* event,
    iree_allocator_t host_allocator, iree_string_builder_t* builder) {
  iree_string_builder_initialize(host_allocator, builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "kind", id4_tooling_diagnostics_event_kind_name(event->kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "stage", event->stage_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "key", event->key));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_json_field_string(
      builder, "message", event->message));
  if (event->parameter_slab) {
    IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_parameter_slab_json(
        builder, event->parameter_slab));
  }
  if (event->parameter_load) {
    IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_append_parameter_load_json(
        builder, event->parameter_load));
  }
  if (event->kernel) {
    IREE_RETURN_IF_ERROR(
        id4_tooling_diagnostics_append_kernel_json(builder, event->kernel));
  }
  if (event->timing) {
    IREE_RETURN_IF_ERROR(
        id4_tooling_diagnostics_append_timing_json(builder, event->timing));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t id4_tooling_diagnostics_file_sink_emit(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  if (!user_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostics file sink is required");
  }
  if (!event) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic event is required");
  }
  id4_tooling_diagnostics_file_sink_t* file_sink =
      (id4_tooling_diagnostics_file_sink_t*)user_data;
  if (!file_sink->event_log_file) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "diagnostics file sink is not initialized");
  }

  iree_string_builder_t builder;
  iree_status_t status = id4_tooling_diagnostics_format_event(
      event, file_sink->host_allocator, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_builder_view(&builder);
    if (fwrite(line.data, 1, line.size, file_sink->event_log_file) !=
            line.size ||
        fputc('\n', file_sink->event_log_file) == EOF ||
        fflush(file_sink->event_log_file) != 0) {
      const int write_errno = errno;
      status = iree_make_status(write_errno
                                    ? iree_status_code_from_errno(write_errno)
                                    : IREE_STATUS_DATA_LOSS,
                                "failed to write diagnostics event");
    }
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t id4_tooling_diagnostics_file_sink_initialize(
    iree_string_view_t directory, iree_allocator_t host_allocator,
    id4_tooling_diagnostics_file_sink_t* out_file_sink,
    id4_pipeline_diagnostics_sink_t* out_sink) {
  IREE_ASSERT_ARGUMENT(out_file_sink);
  IREE_ASSERT_ARGUMENT(out_sink);
  memset(out_file_sink, 0, sizeof(*out_file_sink));
  memset(out_sink, 0, sizeof(*out_sink));
  IREE_RETURN_IF_ERROR(id4_tooling_ensure_directory(directory, host_allocator));

  iree_string_view_t event_log_path = iree_string_view_empty();
  iree_status_t status = id4_tooling_format_child_path(
      directory, IREE_SV("events.jsonl"), host_allocator, &event_log_path);
  FILE* file = NULL;
  if (iree_status_is_ok(status)) {
    file = fopen(event_log_path.data, "wb");
    if (!file) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "failed to open diagnostics event log");
    }
  }
  if (iree_status_is_ok(status)) {
    out_file_sink->host_allocator = host_allocator;
    out_file_sink->event_log_path = event_log_path;
    out_file_sink->event_log_file = file;
    out_sink->emit = id4_tooling_diagnostics_file_sink_emit;
    out_sink->user_data = out_file_sink;
  } else {
    if (file) fclose(file);
    id4_tooling_free_path(&event_log_path, host_allocator);
  }
  return status;
}

iree_status_t id4_tooling_diagnostics_file_sink_deinitialize(
    id4_tooling_diagnostics_file_sink_t* file_sink) {
  if (!file_sink) return iree_ok_status();
  iree_status_t status = iree_ok_status();
  if (file_sink->event_log_file && fclose(file_sink->event_log_file) != 0) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "failed to close diagnostics event log");
  }
  id4_tooling_free_path(&file_sink->event_log_path, file_sink->host_allocator);
  memset(file_sink, 0, sizeof(*file_sink));
  return status;
}
