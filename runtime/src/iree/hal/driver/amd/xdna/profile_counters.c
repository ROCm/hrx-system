// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/profile_counters.h"

#include <stdlib.h>

#include "iree/hal/driver/amd/status.h"
#include "iree/hal/driver/amd/xdna/executable_measurement.h"

// A session is deliberately finite. Native geometry may reduce this bound.
#define IREE_HAL_AMD_XDNA_COUNTER_CAPACITY 4096u
#define IREE_HAL_AMD_XDNA_COUNTER_BYTES (2 * sizeof(amdf_xdna_timer_record_t))

// Native acceptance order is also firmware result order. No marker identity
// changes between invocations, and no result lookup occurs during execution.
typedef struct iree_hal_amd_xdna_counter_entry_t {
  // Retained executable and borrowed immutable function attribution.
  iree_hal_amd_xdna_invocation_info_t info;
  // Public queue submission identity.
  uint64_t submission_id;
  // Stable recording identity, or zero for direct dispatch.
  uint64_t command_buffer_id;
  // Recording ordinal, or UINT32_MAX for direct dispatch.
  uint32_t command_index;
} iree_hal_amd_xdna_counter_entry_t;

struct iree_hal_amd_xdna_profile_counters_t {
  // Allocator owning this object and placed ledger/selection storage.
  iree_allocator_t host_allocator;
  // Borrowed native context and result-control owner.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed arena owning range-mode backing.
  iree_hal_amd_xdna_command_arena_t* command_arena;
  // Borrowed sink retained by the enclosing profile session.
  iree_hal_profile_sink_t* sink;
  // Session identity copied from the enclosing recorder.
  iree_hal_profile_chunk_metadata_t metadata;
  // Copied filter; the optional pattern points into placed storage.
  iree_hal_profile_capture_filter_t filter;
  // Copied counter set name in placed storage.
  iree_string_view_t name;
  // True for native counters, false for executable metadata alone.
  bool measuring;
  // First capture preparation failure, reported and consumed at end.
  iree_status_t failure;
  // Accepted matching invocations omitted from the finite attribution ledger.
  uint64_t dropped_count;
  // Available reservations; rejected commands conservatively consume a slot.
  uint32_t remaining;
  // Number of accepted rows in the ledger.
  uint32_t count;
  // Placed ledger in native acceptance order.
  iree_hal_amd_xdna_counter_entry_t* entries;
  // Placed cold sort view used only at end to deduplicate executable metadata.
  iree_hal_amd_xdna_counter_entry_t** sorted;
  // Reusable cold serialization storage; never touched during submission.
  iree_byte_span_t scratch;
  // Exact association supplied unchanged to every native control.
  amdf_xdna_result_buffer_binding_t binding;
  // Independent whole-memory result backing; empty in range mode.
  iree_hal_amd_fabric_memory_t memory;
  // Arena-owned result range; empty in whole-memory mode.
  iree_hal_amd_xdna_command_allocation_t range;
  // Borrowed host mapping for cache maintenance.
  amdf_host_mapping_t* mapping;
  // Borrowed exact host result span.
  iree_byte_span_t contents;
};

static iree_status_t iree_hal_amd_xdna_profile_counters_write(
    iree_hal_amd_xdna_profile_counters_t* counters,
    iree_string_view_t content_type, iree_host_size_t span_count,
    const iree_const_byte_span_t* spans) {
  iree_hal_profile_chunk_metadata_t metadata = counters->metadata;
  metadata.content_type = content_type;
  if (span_count == 1) {
    return iree_hal_profile_sink_write(counters->sink, &metadata, 1, spans);
  }
  // Typed sinks consume complete records per iovec. Flatten variable-size
  // metadata here, after retirement, reusing the largest cold record buffer.
  iree_host_size_t length = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    length += spans[i].data_length;
  }
  if (length > counters->scratch.data_length) {
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        counters->host_allocator, length, (void**)&counters->scratch.data));
    counters->scratch.data_length = length;
  }
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    if (spans[i].data_length) {
      memcpy(counters->scratch.data + offset, spans[i].data,
             spans[i].data_length);
    }
    offset += spans[i].data_length;
  }
  const iree_const_byte_span_t record =
      iree_make_const_byte_span(counters->scratch.data, length);
  return iree_hal_profile_sink_write(counters->sink, &metadata, 1, &record);
}

static void iree_hal_amd_xdna_profile_counters_free(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  for (uint32_t i = 0; i < counters->count; ++i) {
    iree_hal_executable_release(counters->entries[i].info.executable);
  }
  if (counters->context->result_buffer_retained) {
    // Keep the allocation capsule as well as its borrowed native parents.
    return;
  }
  iree_hal_amd_xdna_command_arena_release(counters->command_arena,
                                          &counters->range);
  iree_hal_amd_fabric_memory_deinitialize(
      counters->context->api, counters->context->event_sink, &counters->memory);
  iree_allocator_free(counters->host_allocator, counters->scratch.data);
  iree_allocator_free(counters->host_allocator, counters);
}

static iree_status_t iree_hal_amd_xdna_profile_counters_control_status(
    iree_hal_amd_xdna_profile_counters_t* counters, amdf_status_t status,
    const char* operation) {
  if (!amdf_status_is_ok(status)) {
    counters->context->result_buffer_retained = true;
    iree_hal_amd_xdna_context_report(counters->context, status, operation);
  }
  return IREE_HAL_AMD_STATUS_FROM_AMDF(status, operation);
}

static iree_status_t iree_hal_amd_xdna_profile_counters_allocate_result(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  iree_hal_amd_xdna_context_t* context = counters->context;
  amdf_xdna_result_buffer_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_RESULT_BUFFER_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->context_query_result_buffer_info(context->handle, &info),
      "xdna.context_query_result_buffer_info"));
  if (info.timer_clock.counter_bit_count != 64) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "XDNA timer requires a qualified 64-bit counter");
  }
  // Both native modes use the context's prepared instruction profile. Range
  // mode must share its indivisible aperture with live executable commands.
  if (info.memory_scope != context->command_source.scope ||
      info.memory_profile_ordinal != context->command_source.profile.ordinal) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "XDNA results require a separate memory profile");
  }
  uint64_t granularity = info.byte_length_granularity;
  uint64_t minimum = info.minimum_byte_length;
  uint64_t maximum = info.maximum_byte_length;
  if (info.binding_mode == AMDF_XDNA_RESULT_BUFFER_BINDING_WHOLE_MEMORY) {
    const amdf_memory_profile_t* profile = &context->command_source.profile;
    uint64_t a = granularity;
    uint64_t b = profile->allocation.byte_length_granularity;
    while (b) {
      const uint64_t remainder = a % b;
      a = b;
      b = remainder;
    }
    if (!iree_checked_mul_u64(granularity / a,
                              profile->allocation.byte_length_granularity,
                              &granularity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA result allocation granularity overflows");
    }
    maximum = iree_min(maximum, profile->allocation.maximum_byte_length);
  }
  uint64_t length = iree_max(minimum, IREE_HAL_AMD_XDNA_COUNTER_CAPACITY *
                                          IREE_HAL_AMD_XDNA_COUNTER_BYTES);
  if (!iree_checked_add_u64(length, granularity - 1, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA result allocation size overflows");
  }
  length = iree_min(length / granularity, maximum / granularity) * granularity;
  if (length < minimum || length < IREE_HAL_AMD_XDNA_COUNTER_BYTES) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "XDNA result geometry cannot hold a timer pair");
  }
  counters->remaining =
      (uint32_t)iree_min(IREE_HAL_AMD_XDNA_COUNTER_CAPACITY,
                         length / IREE_HAL_AMD_XDNA_COUNTER_BYTES);
  if (info.binding_mode == AMDF_XDNA_RESULT_BUFFER_BINDING_WHOLE_MEMORY) {
    const iree_hal_amd_fabric_memory_options_t options = {
        .scope = info.memory_scope,
        .profile = context->command_source.profile,
        .access_count = 1,
        .accesses = &context->command_source.access,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
        context->api, &options, length, info.byte_offset_alignment,
        context->event_sink, &counters->memory, NULL));
    counters->binding.memory = counters->memory.handle;
    counters->mapping = counters->memory.mapping;
    counters->contents = counters->memory.contents;
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_command_arena_allocate_range(
        counters->command_arena, length,
        iree_max(info.byte_offset_alignment,
                 context->target.instruction_alignment),
        &counters->range));
    counters->binding.memory = counters->range.memory;
    counters->binding.byte_offset = counters->range.memory_byte_offset;
    counters->mapping = counters->range.mapping;
    counters->contents = counters->range.contents;
  }
  counters->binding.byte_length = length;
  memset(counters->contents.data, 0, length);
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(
          counters->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
          counters->binding.byte_offset, length),
      "host_mapping_cache_control(result clear)"));
  const amdf_status_t status = context->xdna->context_attach_result_buffer(
      context->handle, &counters->binding);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    // Initial attachment was rejected before native ownership changed. The
    // ordinary failure cleanup can release this previously unbound destination.
    return IREE_HAL_AMD_STATUS_FROM_AMDF(status,
                                         "xdna.context_attach_result_buffer");
  }
  return iree_hal_amd_xdna_profile_counters_control_status(
      counters, status, "xdna.context_attach_result_buffer");
}

iree_status_t iree_hal_amd_xdna_profile_counters_begin(
    const iree_hal_device_profiling_options_t* options,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_profile_counters_t** out_counters) {
  *out_counters = NULL;
  const bool measuring = iree_any_bit_set(
      options->data_families, IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES);
  if (!measuring &&
      !iree_any_bit_set(options->data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }
  if (context->result_buffer_retained) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "XDNA result association has not retired");
  }
  if (options->counter_set_count > 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA exposes one invocation cycle counter set");
  }
  iree_string_view_t name = IREE_SV("xdna");
  if (options->counter_set_count) {
    const iree_hal_profile_counter_set_selection_t* selection =
        &options->counter_sets[0];
    if (selection->flags || !measuring ||
        (selection->counter_name_count && !selection->counter_names)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid XDNA counter selection");
    }
    for (iree_host_size_t i = 0; i < selection->counter_name_count; ++i) {
      if (!iree_string_view_equal(selection->counter_names[i],
                                  IREE_SV("invocation_cycles"))) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "XDNA counter '%.*s' is unavailable",
                                (int)selection->counter_names[i].size,
                                selection->counter_names[i].data);
      }
    }
    if (selection->name.size) {
      name = selection->name;
    }
  }
  if (name.size > UINT32_MAX - sizeof(iree_hal_profile_counter_set_record_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA counter set name is too long");
  }
  const iree_string_view_t pattern =
      options->capture_filter.executable_function_pattern;
  iree_host_size_t size = 0, entries_offset = 0, sorted_offset = 0;
  iree_host_size_t name_offset = 0, pattern_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_profile_counters_t), &size,
      IREE_STRUCT_FIELD(IREE_HAL_AMD_XDNA_COUNTER_CAPACITY,
                        iree_hal_amd_xdna_counter_entry_t, &entries_offset),
      IREE_STRUCT_FIELD(IREE_HAL_AMD_XDNA_COUNTER_CAPACITY,
                        iree_hal_amd_xdna_counter_entry_t*, &sorted_offset),
      IREE_STRUCT_FIELD(name.size, char, &name_offset),
      IREE_STRUCT_FIELD(pattern.size, char, &pattern_offset)));
  iree_hal_amd_xdna_profile_counters_t* counters = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&counters));
  counters->host_allocator = host_allocator;
  counters->context = context;
  counters->command_arena = command_arena;
  counters->sink = options->sink;
  counters->metadata = *metadata;
  counters->measuring = measuring;
  counters->remaining = IREE_HAL_AMD_XDNA_COUNTER_CAPACITY;
  counters->entries = (void*)((uint8_t*)counters + entries_offset);
  counters->sorted = (void*)((uint8_t*)counters + sorted_offset);
  counters->name =
      iree_make_string_view((char*)counters + name_offset, name.size);
  memcpy((void*)counters->name.data, name.data, name.size);
  counters->filter = options->capture_filter;
  counters->filter.executable_function_pattern =
      iree_make_string_view((char*)counters + pattern_offset, pattern.size);
  if (pattern.size) {
    memcpy((void*)counters->filter.executable_function_pattern.data,
           pattern.data, pattern.size);
  }
  iree_status_t status =
      measuring ? iree_hal_amd_xdna_profile_counters_allocate_result(counters)
                : iree_ok_status();
  if (iree_status_is_ok(status)) {
    *out_counters = counters;
  } else {
    iree_hal_amd_xdna_profile_counters_free(counters);
  }
  return status;
}

static bool iree_hal_amd_xdna_profile_counters_matches(
    const iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_profile_queue_event_t* event, uint32_t command_index,
    iree_string_view_t name) {
  return iree_hal_profile_capture_filter_matches_location(
             &counters->filter, event->command_buffer_id, command_index,
             event->physical_device_ordinal, event->queue_ordinal) &&
         (!iree_any_bit_set(
              counters->filter.flags,
              IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) ||
          iree_string_view_match_pattern(
              name, counters->filter.executable_function_pattern));
}

static bool iree_hal_amd_xdna_profile_counters_reserve(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  if (!counters->remaining || !iree_status_is_ok(counters->failure)) {
    return false;
  }
  --counters->remaining;
  return true;
}

void iree_hal_amd_xdna_profile_counters_prepare(
    iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_profile_queue_event_t* event, uint32_t command_index,
    iree_hal_amd_xdna_invocation_t* invocation,
    amdf_xdna_kernel_command_t* command) {
  if (!counters || !counters->measuring) {
    return;
  }
  const iree_hal_amd_xdna_invocation_info_t info =
      iree_hal_amd_xdna_invocation_info(invocation);
  if (iree_hal_amd_xdna_profile_counters_matches(counters, event, command_index,
                                                 info.name) &&
      iree_hal_amd_xdna_profile_counters_reserve(counters)) {
    counters->failure =
        iree_hal_amd_xdna_invocation_prepare_measurement(invocation, command);
  }
}

void iree_hal_amd_xdna_profile_counters_accept(
    iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_profile_queue_event_t* event, uint32_t command_index,
    iree_hal_amd_xdna_invocation_t* invocation,
    const amdf_xdna_kernel_command_t* command) {
  if (!counters) {
    return;
  }
  const iree_hal_amd_xdna_invocation_info_t info =
      iree_hal_amd_xdna_invocation_info(invocation);
  if (counters->measuring) {
    if (!iree_hal_amd_xdna_invocation_is_measured(invocation, command)) {
      if (iree_hal_amd_xdna_profile_counters_matches(
              counters, event, command_index, info.name)) {
        ++counters->dropped_count;
      }
      return;
    }
  } else if (!iree_hal_amd_xdna_profile_counters_matches(
                 counters, event, command_index, info.name)) {
    return;
  } else if (!iree_hal_amd_xdna_profile_counters_reserve(counters)) {
    ++counters->dropped_count;
    return;
  }
  iree_hal_amd_xdna_counter_entry_t* entry =
      &counters->entries[counters->count++];
  *entry = (iree_hal_amd_xdna_counter_entry_t){
      .info = info,
      .submission_id = event->submission_id,
      .command_buffer_id = event->command_buffer_id,
      .command_index = command_index,
  };
  iree_hal_executable_retain(info.executable);
}

static int iree_hal_amd_xdna_profile_counters_compare(const void* lhs,
                                                      const void* rhs) {
  const iree_hal_amd_xdna_counter_entry_t* a =
      *(const iree_hal_amd_xdna_counter_entry_t* const*)lhs;
  const iree_hal_amd_xdna_counter_entry_t* b =
      *(const iree_hal_amd_xdna_counter_entry_t* const*)rhs;
  return (a->info.executable_id > b->info.executable_id) -
         (a->info.executable_id < b->info.executable_id);
}

static iree_status_t iree_hal_amd_xdna_profile_counters_emit_executable(
    iree_hal_amd_xdna_profile_counters_t* counters,
    const iree_hal_amd_xdna_invocation_info_t* info) {
  iree_hal_profile_executable_record_t executable =
      iree_hal_profile_executable_record_default();
  executable.executable_id = info->executable_id;
  executable.function_count =
      (uint32_t)iree_hal_executable_function_count(info->executable);
  iree_const_byte_span_t span =
      iree_make_const_byte_span(&executable, sizeof(executable));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_profile_counters_write(
      counters, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES, 1, &span));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < executable.function_count && iree_status_is_ok(status); ++i) {
    iree_hal_executable_function_info_t function;
    status = iree_hal_executable_function_info(
        info->executable, iree_hal_executable_function_from_index(i),
        &function);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (function.name.size >
        UINT32_MAX - sizeof(iree_hal_profile_executable_function_record_t)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "XDNA function profile record is too long");
      break;
    }
    iree_hal_profile_executable_function_record_t record =
        iree_hal_profile_executable_function_record_default();
    record.record_length += (uint32_t)function.name.size;
    record.executable_id = info->executable_id;
    record.function_ordinal = i;
    record.binding_count = function.binding_count;
    record.parameter_count = function.parameter_count;
    memcpy(record.workgroup_size, function.workgroup_size,
           sizeof(record.workgroup_size));
    record.name_length = (uint32_t)function.name.size;
    const iree_const_byte_span_t spans[] = {
        iree_make_const_byte_span(&record, sizeof(record)),
        iree_make_const_byte_span(function.name.data, function.name.size),
    };
    status = iree_hal_amd_xdna_profile_counters_write(
        counters, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS, 2, spans);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_profile_counters_emit_metadata(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  for (uint32_t i = 0; i < counters->count; ++i) {
    counters->sorted[i] = &counters->entries[i];
  }
  qsort(counters->sorted, counters->count, sizeof(*counters->sorted),
        iree_hal_amd_xdna_profile_counters_compare);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < counters->count && iree_status_is_ok(status); ++i) {
    if (!i || counters->sorted[i]->info.executable_id !=
                  counters->sorted[i - 1]->info.executable_id) {
      status = iree_hal_amd_xdna_profile_counters_emit_executable(
          counters, &counters->sorted[i]->info);
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_profile_counters_validate(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  const amdf_xdna_timer_record_t* records =
      (const void*)counters->contents.data;
  for (uint32_t i = 0; i < counters->count; ++i) {
    if (records[2 * i].id != IREE_HAL_AMD_XDNA_MEASUREMENT_BEGIN_ID ||
        records[2 * i + 1].id != IREE_HAL_AMD_XDNA_MEASUREMENT_END_ID) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "XDNA timer pair %u is missing or malformed", i);
    }
  }
  for (iree_host_size_t i = counters->count * IREE_HAL_AMD_XDNA_COUNTER_BYTES;
       i < counters->contents.data_length; ++i) {
    if (counters->contents.data[i]) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "XDNA timer results exceed accepted commands");
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_profile_counters_emit_samples(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  iree_hal_profile_counter_set_record_t set =
      iree_hal_profile_counter_set_record_default();
  set.record_length += (uint32_t)counters->name.size;
  set.counter_set_id = 1;
  set.physical_device_ordinal = counters->metadata.physical_device_ordinal;
  set.counter_count = set.sample_value_count = 1;
  set.name_length = (uint32_t)counters->name.size;
  const iree_const_byte_span_t set_spans[] = {
      iree_make_const_byte_span(&set, sizeof(set)),
      iree_make_const_byte_span(counters->name.data, counters->name.size),
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_profile_counters_write(
      counters, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS, 2, set_spans));
  const iree_string_view_t block = IREE_SV("partition_shim");
  const iree_string_view_t name = IREE_SV("invocation_cycles");
  const iree_string_view_t description = IREE_SV(
      "Compiler-declared same-epoch invocation interval in partition-shim "
      "timer cycles");
  iree_hal_profile_counter_record_t counter =
      iree_hal_profile_counter_record_default();
  counter.record_length +=
      (uint32_t)(block.size + name.size + description.size);
  counter.unit = IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES;
  counter.counter_set_id = 1;
  counter.physical_device_ordinal = counters->metadata.physical_device_ordinal;
  counter.sample_value_count = 1;
  counter.block_name_length = (uint32_t)block.size;
  counter.name_length = (uint32_t)name.size;
  counter.description_length = (uint32_t)description.size;
  const iree_const_byte_span_t counter_spans[] = {
      iree_make_const_byte_span(&counter, sizeof(counter)),
      iree_make_const_byte_span(block.data, block.size),
      iree_make_const_byte_span(name.data, name.size),
      iree_make_const_byte_span(description.data, description.size),
  };
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_profile_counters_write(
      counters, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS, 4, counter_spans));
  const amdf_xdna_timer_record_t* records =
      (const void*)counters->contents.data;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < counters->count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_xdna_counter_entry_t* entry = &counters->entries[i];
    struct {
      // Portable sample header, with no claimed cross-command device epoch.
      iree_hal_profile_counter_sample_record_t record;
      // Same-epoch partition-shim cycle delta, including natural 64-bit wrap.
      uint64_t value;
    } sample = {.record = iree_hal_profile_counter_sample_record_default()};
    sample.record.record_length = sizeof(sample);
    sample.record.scope = IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH;
    sample.record.sample_id = i + 1;
    sample.record.counter_set_id = 1;
    sample.record.submission_id = entry->submission_id;
    sample.record.command_buffer_id = entry->command_buffer_id;
    sample.record.command_index = entry->command_index;
    sample.record.executable_id = entry->info.executable_id;
    sample.record.function_ordinal = entry->info.function_ordinal;
    sample.record.stream_id = counters->metadata.stream_id;
    sample.record.physical_device_ordinal =
        counters->metadata.physical_device_ordinal;
    sample.record.queue_ordinal = counters->metadata.queue_ordinal;
    sample.record.sample_value_count = 1;
    sample.value = amdf_xdna_timer_record_timestamp(&records[2 * i + 1]) -
                   amdf_xdna_timer_record_timestamp(&records[2 * i]);
    const iree_const_byte_span_t span =
        iree_make_const_byte_span(&sample, sizeof(sample));
    status = iree_hal_amd_xdna_profile_counters_write(
        counters, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES, 1, &span);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_profile_counters_emit_loss(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  if (!counters->dropped_count) {
    return iree_ok_status();
  }
  iree_hal_profile_chunk_metadata_t metadata = counters->metadata;
  metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  if (counters->measuring) {
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
    metadata.dropped_record_count = counters->dropped_count;
  } else {
    // Omitted invocations may reuse already-emitted executables. Without
    // retaining their identities, the number of missing metadata rows is
    // unknown.
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
  }
  return iree_hal_profile_sink_write(counters->sink, &metadata, 0, NULL);
}

iree_status_t iree_hal_amd_xdna_profile_counters_end(
    iree_hal_amd_xdna_profile_counters_t* counters) {
  if (!counters) {
    return iree_ok_status();
  }
  iree_status_t status = counters->failure;
  counters->failure = iree_ok_status();
  if (counters->measuring) {
    iree_hal_amd_xdna_context_t* context = counters->context;
    iree_status_t collect_status =
        iree_hal_amd_xdna_profile_counters_control_status(
            counters,
            context->xdna->context_collect_result_buffer(context->handle,
                                                         &counters->binding),
            "xdna.context_collect_result_buffer");
    if (iree_status_is_ok(collect_status)) {
      // Detach even when cache acquisition fails: native writers are already
      // idle, and healthy removal still discharges the backing obligation.
      collect_status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          context->api->host_mapping_cache_control(
              counters->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
              counters->binding.byte_offset, counters->binding.byte_length),
          "host_mapping_cache_control(result acquire)");
      collect_status = iree_status_join(
          collect_status, iree_hal_amd_xdna_profile_counters_control_status(
                              counters,
                              context->xdna->context_detach_result_buffer(
                                  context->handle, &counters->binding),
                              "xdna.context_detach_result_buffer"));
    }
    status = iree_status_join(status, collect_status);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_xdna_profile_counters_validate(counters);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_profile_counters_emit_metadata(counters);
  }
  if (iree_status_is_ok(status) && counters->measuring) {
    status = iree_hal_amd_xdna_profile_counters_emit_samples(counters);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_profile_counters_emit_loss(counters);
  }
  iree_hal_amd_xdna_profile_counters_free(counters);
  return status;
}
