// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/profile.h"

#include <inttypes.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/utils/profile_event_ring.h"

// Host metadata has no native resource or executable ownership. Its lifetime
// can extend to an explicit profile flush after the original code is unloaded.
struct iree_hal_amd_gpu_profile_executable_t {
  // Independent refcounted host metadata resource.
  iree_hal_resource_t resource;
  // Allocator for this object and its trailing load and function records.
  iree_allocator_t host_allocator;
  // Stable identity and complete executable summary.
  iree_hal_profile_executable_record_t record;
  // Packed function records and names, owned by this allocation.
  iree_const_byte_span_t functions;
  // Dense function-name views into functions, indexed by function ordinal.
  iree_string_view_t* function_names;
  // Retained original host artifact, without executable or native ownership.
  iree_byte_sequence_t* source;
  // Shared image identity followed by the original artifact bytes at emission.
  iree_hal_profile_executable_code_object_record_t code_object;
  // Contiguous original bytes, kept alive by source without a payload copy.
  iree_const_byte_span_t image;
  // Packed load records, owned by this allocation, including queue variants.
  iree_const_byte_span_t loads;
  // Last emitted device-local session, accessed only by serialized flush/end.
  uint64_t emitted_session;
};

struct iree_hal_amd_gpu_profile_commands_t {
  // Independent host metadata resource, without resident IB ownership.
  iree_hal_resource_t resource;
  // Allocator for the object and recording-time dispatch array.
  iree_allocator_t host_allocator;
  // Public recording identity, independent of native program placements.
  iree_hal_profile_command_buffer_record_t record;
  // Dense dispatch templates; each owns its executable metadata reference.
  iree_hal_amd_gpu_profile_dispatch_t* dispatches;
  // Initialized dispatch templates.
  uint32_t count;
  // Allocated recording-time array capacity.
  uint32_t capacity;
  // Last emitted device-local session, accessed only by serialized flush/end.
  uint64_t emitted_session;
};

static iree_atomic_int64_t iree_hal_amd_gpu_profile_next_executable_id =
    IREE_ATOMIC_VAR_INIT(1);

static void iree_hal_amd_gpu_profile_executable_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amd_gpu_profile_executable_t* metadata = (void*)resource;
  iree_byte_sequence_release(metadata->source);
  iree_allocator_free(metadata->host_allocator, metadata);
}

iree_status_t iree_hal_amd_gpu_profile_executable_create(
    iree_hal_executable_t* executable, const iree_hal_amd_gpu_image_t* image,
    uint32_t load_count, const uint64_t* load_addresses,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_executable_t** out_metadata) {
  *out_metadata = NULL;
  iree_const_byte_span_t image_bytes;
  const bool contiguous =
      iree_byte_sequence_try_get_contiguous_span(image->source, &image_bytes);
  IREE_ASSERT(contiguous, "native executable storage must be contiguous");
  if (image_bytes.data_length >
      UINT32_MAX - sizeof(iree_hal_profile_executable_code_object_record_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU profile code-object record exceeds 4 GiB");
  }
  const iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  iree_host_size_t function_bytes = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < function_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_executable_function_info_t info;
    status = iree_hal_executable_function_info(
        executable, iree_hal_executable_function_from_index((uint32_t)i),
        &info);
    if (iree_status_is_ok(status) &&
        (!iree_host_size_checked_add(
             function_bytes,
             sizeof(iree_hal_profile_executable_function_record_t),
             &function_bytes) ||
         !iree_host_size_checked_add(function_bytes, info.name.size,
                                     &function_bytes))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "GPU profile metadata size overflows");
    }
  }
  iree_host_size_t size = 0, loads_offset = 0, names_offset = 0;
  iree_host_size_t functions_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_amd_gpu_profile_executable_t), &size,
        IREE_STRUCT_FIELD(load_count,
                          iree_hal_profile_executable_code_object_load_record_t,
                          &loads_offset),
        IREE_STRUCT_FIELD(function_count, iree_string_view_t, &names_offset),
        IREE_STRUCT_FIELD(function_bytes, uint8_t, &functions_offset));
  }
  iree_hal_amd_gpu_profile_executable_t* metadata = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, size, (void**)&metadata);
  }
  if (iree_status_is_ok(status)) {
    static const iree_hal_resource_vtable_t vtable = {
        .destroy = iree_hal_amd_gpu_profile_executable_destroy,
    };
    iree_hal_resource_initialize(&vtable, &metadata->resource);
    metadata->host_allocator = host_allocator;
    metadata->record = iree_hal_profile_executable_record_default();
    metadata->record.executable_id = (uint64_t)iree_atomic_fetch_add(
        &iree_hal_amd_gpu_profile_next_executable_id, 1,
        iree_memory_order_relaxed);
    metadata->record.function_count = (uint32_t)function_count;
    metadata->source = image->source;
    iree_byte_sequence_retain(metadata->source);
    metadata->image = image_bytes;
    metadata->code_object =
        iree_hal_profile_executable_code_object_record_default();
    metadata->code_object.record_length += (uint32_t)image_bytes.data_length;
    metadata->code_object.executable_id = metadata->record.executable_id;
    metadata->code_object.code_object_id = metadata->record.executable_id;
    metadata->code_object.data_length = image_bytes.data_length;
    metadata->loads = iree_make_const_byte_span(
        (uint8_t*)metadata + loads_offset,
        load_count *
            sizeof(iree_hal_profile_executable_code_object_load_record_t));
    uint8_t* cursor = (uint8_t*)metadata + loads_offset;
    for (uint32_t i = 0; i < load_count; ++i) {
      iree_hal_profile_executable_code_object_load_record_t load =
          iree_hal_profile_executable_code_object_load_record_default();
      load.physical_device_ordinal = 0;
      load.executable_id = metadata->record.executable_id;
      load.code_object_id = metadata->code_object.code_object_id;
      load.load_delta = (int64_t)(load_addresses[i] - image->virtual_base);
      load.load_size = image->byte_length;
      memcpy(cursor, &load, sizeof(load));
      cursor += sizeof(load);
    }
    cursor = (uint8_t*)metadata + functions_offset;
    metadata->functions = iree_make_const_byte_span(cursor, function_bytes);
    metadata->function_names = (void*)((uint8_t*)metadata + names_offset);
    for (iree_host_size_t i = 0;
         i < function_count && iree_status_is_ok(status); ++i) {
      iree_hal_executable_function_info_t info;
      status = iree_hal_executable_function_info(
          executable, iree_hal_executable_function_from_index((uint32_t)i),
          &info);
      if (!iree_status_is_ok(status)) {
        continue;
      }
      iree_hal_profile_executable_function_record_t record =
          iree_hal_profile_executable_function_record_default();
      record.record_length = (uint32_t)(sizeof(record) + info.name.size);
      record.executable_id = metadata->record.executable_id;
      record.function_ordinal = (uint32_t)i;
      record.constant_byte_length = info.constant_byte_length;
      record.binding_count = info.binding_count;
      record.parameter_count = info.parameter_count;
      memcpy(record.workgroup_size, info.workgroup_size,
             sizeof(record.workgroup_size));
      record.name_length = (uint32_t)info.name.size;
      memcpy(cursor, &record, sizeof(record));
      if (info.name.size) {
        memcpy(cursor + sizeof(record), info.name.data, info.name.size);
      }
      metadata->function_names[i] = iree_make_string_view(
          (const char*)cursor + sizeof(record), info.name.size);
      cursor += record.record_length;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_metadata = metadata;
  } else {
    iree_hal_amd_gpu_profile_executable_release(metadata);
  }
  return status;
}

void iree_hal_amd_gpu_profile_executable_retain(
    iree_hal_amd_gpu_profile_executable_t* metadata) {
  iree_hal_resource_retain(metadata);
}
void iree_hal_amd_gpu_profile_executable_release(
    iree_hal_amd_gpu_profile_executable_t* metadata) {
  iree_hal_resource_release(metadata);
}
uint64_t iree_hal_amd_gpu_profile_executable_id(
    const iree_hal_amd_gpu_profile_executable_t* metadata) {
  return metadata->record.executable_id;
}

static void iree_hal_amd_gpu_profile_commands_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amd_gpu_profile_commands_t* commands = (void*)resource;
  for (uint32_t i = 0; i < commands->count; ++i) {
    iree_hal_amd_gpu_profile_executable_release(
        commands->dispatches[i].executable);
  }
  iree_allocator_free(commands->host_allocator, commands->dispatches);
  iree_allocator_free(commands->host_allocator, commands);
}

iree_status_t iree_hal_amd_gpu_profile_commands_create(
    const iree_hal_profile_command_buffer_record_t* record,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_commands_t** out_commands) {
  *out_commands = NULL;
  iree_hal_amd_gpu_profile_commands_t* commands = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*commands),
                                             (void**)&commands));
  static const iree_hal_resource_vtable_t vtable = {
      .destroy = iree_hal_amd_gpu_profile_commands_destroy,
  };
  iree_hal_resource_initialize(&vtable, &commands->resource);
  commands->host_allocator = host_allocator;
  commands->record = *record;
  *out_commands = commands;
  return iree_ok_status();
}
void iree_hal_amd_gpu_profile_commands_retain(
    iree_hal_amd_gpu_profile_commands_t* commands) {
  iree_hal_resource_retain(commands);
}
void iree_hal_amd_gpu_profile_commands_release(
    iree_hal_amd_gpu_profile_commands_t* commands) {
  iree_hal_resource_release(commands);
}
iree_status_t iree_hal_amd_gpu_profile_commands_append(
    iree_hal_amd_gpu_profile_commands_t* commands,
    iree_hal_amd_gpu_profile_dispatch_t dispatch) {
  if (commands->count == commands->capacity) {
    const uint64_t capacity =
        commands->capacity ? (uint64_t)commands->capacity * 2 : 16;
    if (capacity > UINT32_MAX / sizeof(dispatch)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "GPU profile command metadata exceeds 4 GiB");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        commands->host_allocator, (iree_host_size_t)capacity * sizeof(dispatch),
        (void**)&commands->dispatches));
    commands->capacity = (uint32_t)capacity;
  }
  iree_hal_amd_gpu_profile_executable_retain(dispatch.executable);
  commands->dispatches[commands->count++] = dispatch;
  return iree_ok_status();
}
uint32_t iree_hal_amd_gpu_profile_commands_count(
    const iree_hal_amd_gpu_profile_commands_t* commands) {
  return commands ? commands->count : 0;
}

// Each queue retains this many public operations between explicit flushes.
#define IREE_HAL_AMD_GPU_PROFILE_CAPACITY 4096

typedef struct iree_hal_amd_gpu_profile_metadata_t {
  // Borrowed from commands, or directly retained for a direct dispatch.
  iree_hal_amd_gpu_profile_executable_t* executable;
  // Retained host recording metadata, absent for direct dispatches.
  iree_hal_amd_gpu_profile_commands_t* commands;
} iree_hal_amd_gpu_profile_metadata_t;

typedef struct iree_hal_amd_gpu_profile_stream_t {
  // Protects ring publication and flush commits, never a sink callback.
  iree_slim_mutex_t mutex;
  // Queue-local acceptance identity, independent of native chunk epochs.
  iree_atomic_int64_t next_submission_id;
  // Bounded records stored in the session's trailing allocation.
  iree_hal_profile_event_ring_t events;
  // Completed device events, never pointers to still-live native samples.
  iree_hal_profile_event_ring_t dispatches;
  // Host metadata ownership for each physical dispatch-ring slot.
  iree_hal_amd_gpu_profile_metadata_t* metadata;
} iree_hal_amd_gpu_profile_stream_t;

struct iree_hal_amd_gpu_profile_t {
  // Allocator owning this session and all inline stream storage.
  iree_allocator_t host_allocator;
  // Retained output sink; only explicit lifecycle calls may invoke it.
  iree_hal_profile_sink_t* sink;
  // Immutable session metadata shared by every chunk.
  iree_hal_profile_chunk_metadata_t metadata;
  // Number of provisioned queues and initialized stream mutexes.
  uint32_t queue_count;
  // Admitted immutable families; controls capture and stream emission.
  iree_hal_device_profiling_data_families_t families;
  // Immutable predicates; the optional pattern lives in trailing storage.
  iree_hal_profile_capture_filter_t filter;
  // Inline queue streams, indexed directly by physical queue ordinal.
  iree_hal_amd_gpu_profile_stream_t* streams;
};

static void iree_hal_amd_gpu_profile_metadata_release(
    iree_hal_amd_gpu_profile_metadata_t metadata) {
  if (metadata.commands) {
    iree_hal_amd_gpu_profile_commands_release(metadata.commands);
  } else {
    iree_hal_amd_gpu_profile_executable_release(metadata.executable);
  }
}

static void iree_hal_amd_gpu_profile_free(iree_hal_amd_gpu_profile_t* profile) {
  for (uint32_t i = 0; i < profile->queue_count; ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[i];
    for (uint64_t j = stream->dispatches.read_position;
         j != stream->dispatches.write_position; ++j) {
      iree_hal_amd_gpu_profile_metadata_release(
          stream->metadata[j & stream->dispatches.mask]);
    }
    iree_slim_mutex_deinitialize(&stream->mutex);
  }
  iree_hal_profile_sink_release(profile->sink);
  iree_allocator_free(profile->host_allocator, profile);
}

static iree_status_t iree_hal_amd_gpu_profile_write(
    iree_hal_amd_gpu_profile_t* profile, iree_string_view_t content_type,
    uint32_t queue_ordinal, iree_const_byte_span_t bytes) {
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = content_type;
  metadata.queue_ordinal = queue_ordinal;
  metadata.stream_id = queue_ordinal == UINT32_MAX ? 0 : queue_ordinal + 1ull;
  return iree_hal_profile_sink_write(profile->sink, &metadata, 1, &bytes);
}

static iree_status_t iree_hal_amd_gpu_profile_write_executable(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_executable_t* executable) {
  if (executable->emitted_session == profile->metadata.session_id) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES, UINT32_MAX,
      iree_make_const_byte_span(&executable->record,
                                sizeof(executable->record))));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS, UINT32_MAX,
      executable->functions));
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS;
  const iree_const_byte_span_t image_spans[] = {
      iree_make_const_byte_span(&executable->code_object,
                                sizeof(executable->code_object)),
      executable->image,
  };
  IREE_RETURN_IF_ERROR(iree_hal_profile_sink_write(
      profile->sink, &metadata, IREE_ARRAYSIZE(image_spans), image_spans));
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS,
      UINT32_MAX, executable->loads));
  executable->emitted_session = profile->metadata.session_id;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_profile_flush_dispatches(
    iree_hal_amd_gpu_profile_t* profile, uint32_t ordinal) {
  iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[ordinal];
  iree_hal_profile_event_ring_snapshot_t snapshot;
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status =
      iree_hal_profile_event_ring_snapshot(&stream->dispatches, &snapshot);
  iree_slim_mutex_unlock(&stream->mutex);
  if (!iree_status_is_ok(status) ||
      (!snapshot.record_count && !snapshot.dropped_record_count)) {
    return status;
  }
  for (uint64_t i = 0; i < snapshot.record_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_gpu_profile_metadata_t* metadata =
        &stream
             ->metadata[(snapshot.read_position + i) & stream->dispatches.mask];
    status = iree_hal_amd_gpu_profile_write_executable(profile,
                                                       metadata->executable);
    if (iree_status_is_ok(status) && metadata->commands &&
        metadata->commands->emitted_session != profile->metadata.session_id) {
      status = iree_hal_amd_gpu_profile_write(
          profile, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS, UINT32_MAX,
          iree_make_const_byte_span(&metadata->commands->record,
                                    sizeof(metadata->commands->record)));
      if (iree_status_is_ok(status)) {
        metadata->commands->emitted_session = profile->metadata.session_id;
      }
    }
  }
  if (iree_status_is_ok(status) &&
      (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
    iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
    metadata.queue_ordinal = ordinal;
    metadata.stream_id = ordinal + 1ull;
    metadata.dropped_record_count = snapshot.dropped_record_count;
    if (snapshot.dropped_record_count) {
      metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    }
    status = iree_hal_profile_sink_write(
        profile->sink, &metadata, snapshot.record_span_count,
        snapshot.record_span_count ? snapshot.record_spans : NULL);
  } else if (iree_status_is_ok(status) && snapshot.dropped_record_count) {
    // A lost dispatch may reference an executable already emitted elsewhere.
    // Metadata is incomplete, but the number of missing unique identities is
    // unknown; it is not the number of dropped dispatches.
    iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
    metadata.queue_ordinal = ordinal;
    metadata.stream_id = ordinal + 1ull;
    metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    status = iree_hal_profile_sink_write(profile->sink, &metadata, 0, NULL);
  }
  if (iree_status_is_ok(status)) {
    for (uint64_t i = 0; i < snapshot.record_count; ++i) {
      iree_hal_amd_gpu_profile_metadata_release(
          stream->metadata[(snapshot.read_position + i) &
                           stream->dispatches.mask]);
    }
    iree_slim_mutex_lock(&stream->mutex);
    iree_hal_profile_event_ring_commit_snapshot(&stream->dispatches, &snapshot);
    iree_slim_mutex_unlock(&stream->mutex);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_begin(
    const iree_hal_device_profiling_options_t* options, uint64_t session_id,
    uint32_t queue_count, uint64_t timestamp_frequency_hz,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_profile_t** out_profile) {
  *out_profile = NULL;
  iree_hal_device_profiling_data_families_t families = options->data_families;
  if (!families &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options)) {
    families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  }
  if (!families) {
    return iree_ok_status();
  }
  const uint64_t supported = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                             IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                             IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  if (families & ~supported) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported GPU profiling data families 0x%" PRIx64,
        families & ~supported);
  }
  if (options->counter_set_count) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GPU queue profiling does not support hardware counters");
  }
  if ((families & IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS) &&
      !timestamp_frequency_hz) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GPU dispatch profiling requires a native queue clock");
  }
  iree_host_size_t record_count = 0;
  if (!iree_host_size_checked_mul(
          queue_count, IREE_HAL_AMD_GPU_PROFILE_CAPACITY, &record_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU profiling stream capacity overflows");
  }
  const iree_host_size_t dispatch_count =
      (families & (IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                   IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA))
          ? record_count
          : 0;
  iree_host_size_t size = 0, streams_offset = 0, records_offset = 0;
  iree_host_size_t dispatches_offset = 0, metadata_offset = 0,
                   pattern_offset = 0;
  const iree_string_view_t pattern =
      iree_any_bit_set(
          options->capture_filter.flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN)
          ? options->capture_filter.executable_function_pattern
          : iree_string_view_empty();
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_profile_t), &size,
      IREE_STRUCT_FIELD(queue_count, iree_hal_amd_gpu_profile_stream_t,
                        &streams_offset),
      IREE_STRUCT_FIELD(record_count, iree_hal_profile_queue_event_t,
                        &records_offset),
      IREE_STRUCT_FIELD(dispatch_count, iree_hal_profile_dispatch_event_t,
                        &dispatches_offset),
      IREE_STRUCT_FIELD(dispatch_count, iree_hal_amd_gpu_profile_metadata_t,
                        &metadata_offset),
      IREE_STRUCT_FIELD(pattern.size, char, &pattern_offset)));
  iree_hal_amd_gpu_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&profile));
  profile->host_allocator = host_allocator;
  profile->sink = options->sink;
  iree_hal_profile_sink_retain(profile->sink);
  profile->queue_count = queue_count;
  profile->families = families;
  profile->filter = options->capture_filter;
  profile->filter.executable_function_pattern = iree_string_view_empty();
  if (pattern.size) {
    char* target = (char*)profile + pattern_offset;
    memcpy(target, pattern.data, pattern.size);
    profile->filter.executable_function_pattern =
        iree_make_string_view(target, pattern.size);
  }
  profile->streams = (void*)((uint8_t*)profile + streams_offset);
  profile->metadata = iree_hal_profile_chunk_metadata_default();
  profile->metadata.name = IREE_SV("amd-gpu");
  profile->metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  profile->metadata.session_id = session_id;
  profile->metadata.physical_device_ordinal = 0;
  iree_hal_profile_queue_event_t* records =
      (void*)((uint8_t*)profile + records_offset);
  for (uint32_t i = 0; i < queue_count; ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[i];
    iree_slim_mutex_initialize(&stream->mutex);
    iree_atomic_store(&stream->next_submission_id, 1,
                      iree_memory_order_relaxed);
    iree_hal_profile_event_ring_initialize(
        records + (iree_host_size_t)i * IREE_HAL_AMD_GPU_PROFILE_CAPACITY,
        sizeof(*records), IREE_HAL_AMD_GPU_PROFILE_CAPACITY, &stream->events);
    iree_hal_profile_dispatch_event_t* dispatch_records =
        (void*)((uint8_t*)profile + dispatches_offset);
    iree_hal_profile_event_ring_initialize(
        dispatch_count
            ? dispatch_records +
                  (iree_host_size_t)i * IREE_HAL_AMD_GPU_PROFILE_CAPACITY
            : NULL,
        sizeof(*dispatch_records),
        dispatch_count ? IREE_HAL_AMD_GPU_PROFILE_CAPACITY : 0,
        &stream->dispatches);
    iree_hal_amd_gpu_profile_metadata_t* metadata =
        (void*)((uint8_t*)profile + metadata_offset);
    stream->metadata =
        dispatch_count
            ? metadata + (iree_host_size_t)i * IREE_HAL_AMD_GPU_PROFILE_CAPACITY
            : NULL;
  }
  iree_status_t status =
      iree_hal_profile_sink_begin_session(profile->sink, &profile->metadata);
  const bool begun = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    iree_hal_profile_device_record_t record =
        iree_hal_profile_device_record_default();
    record.physical_device_ordinal = 0;
    record.queue_count = queue_count;
    if (timestamp_frequency_hz) {
      record.flags |= IREE_HAL_PROFILE_DEVICE_FLAG_TIMESTAMP_FREQUENCY;
      record.timestamp_frequency_hz = timestamp_frequency_hz;
    }
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, UINT32_MAX,
        iree_make_const_byte_span(&record, sizeof(record)));
  }
  for (uint32_t i = 0; i < queue_count && iree_status_is_ok(status); ++i) {
    iree_hal_profile_queue_record_t record =
        iree_hal_profile_queue_record_default();
    record.physical_device_ordinal = 0;
    record.queue_ordinal = i;
    record.stream_id = i + 1ull;
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, i,
        iree_make_const_byte_span(&record, sizeof(record)));
  }
  if (iree_status_is_ok(status)) {
    *out_profile = profile;
  } else {
    if (begun) {
      const iree_status_code_t code = iree_status_code(status);
      status = iree_status_join(
          status, iree_hal_profile_sink_end_session(profile->sink,
                                                    &profile->metadata, code));
    }
    iree_hal_amd_gpu_profile_free(profile);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_flush(
    iree_hal_amd_gpu_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < profile->queue_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amd_gpu_profile_flush_dispatches(profile, i);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    iree_hal_amd_gpu_profile_stream_t* stream = &profile->streams[i];
    iree_hal_profile_event_ring_snapshot_t snapshot;
    iree_slim_mutex_lock(&stream->mutex);
    status = iree_hal_profile_event_ring_snapshot(&stream->events, &snapshot);
    iree_slim_mutex_unlock(&stream->mutex);
    if (!iree_status_is_ok(status) ||
        (!snapshot.record_count && !snapshot.dropped_record_count)) {
      continue;
    }
    iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
    metadata.queue_ordinal = i;
    metadata.stream_id = i + 1ull;
    metadata.dropped_record_count = snapshot.dropped_record_count;
    if (snapshot.dropped_record_count) {
      metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    }
    status = iree_hal_profile_sink_write(
        profile->sink, &metadata, snapshot.record_span_count,
        snapshot.record_span_count ? snapshot.record_spans : NULL);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&stream->mutex);
      iree_hal_profile_event_ring_commit_snapshot(&stream->events, &snapshot);
      iree_slim_mutex_unlock(&stream->mutex);
    }
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_end(
    iree_hal_amd_gpu_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  iree_status_t status = iree_hal_amd_gpu_profile_flush(profile);
  const iree_status_code_t code = iree_status_code(status);
  status =
      iree_status_join(status, iree_hal_profile_sink_end_session(
                                   profile->sink, &profile->metadata, code));
  iree_hal_amd_gpu_profile_free(profile);
  return status;
}

void iree_hal_amd_gpu_profile_operation_initialize(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    iree_hal_profile_queue_event_type_t type, iree_host_size_t wait_count,
    iree_host_size_t signal_count, iree_host_size_t operation_count,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  *operation = (iree_hal_amd_gpu_profile_operation_t){
      .event = iree_hal_profile_queue_event_default()};
  iree_hal_profile_queue_event_t* event = &operation->event;
  event->type = type;
  event->host_time_ns = iree_time_now();
  event->submission_id =
      iree_atomic_fetch_add(&profile->streams[queue_ordinal].next_submission_id,
                            1, iree_memory_order_relaxed);
  event->physical_device_ordinal = 0;
  event->queue_ordinal = queue_ordinal;
  event->stream_id = queue_ordinal + 1ull;
  event->wait_count = (uint32_t)iree_min(wait_count, UINT32_MAX);
  event->signal_count = (uint32_t)iree_min(signal_count, UINT32_MAX);
  event->operation_count = (uint32_t)iree_min(operation_count, UINT32_MAX);
  event->dependency_strategy =
      wait_count ? IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE
                 : IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE;
}

void iree_hal_amd_gpu_profile_operation_record(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation, bool ready,
    uint32_t native_wait_count) {
  if (!operation || operation->recorded) {
    return;
  }
  operation->recorded = true;
  iree_hal_profile_queue_event_t* event = &operation->event;
  event->ready_host_time_ns = ready ? iree_time_now() : 0;
  event->barrier_count = native_wait_count;
  if (native_wait_count &&
      !iree_any_bit_set(event->flags,
                        IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED)) {
    event->dependency_strategy =
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER;
  }
  if (!(profile->families & IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return;
  }
  iree_hal_amd_gpu_profile_stream_t* stream =
      &profile->streams[event->queue_ordinal];
  uint64_t position = 0, event_id = 0;
  iree_slim_mutex_lock(&stream->mutex);
  if (iree_hal_profile_event_ring_try_append(&stream->events, &position,
                                             &event_id)) {
    event->event_id = event_id;
    *(iree_hal_profile_queue_event_t*)iree_hal_profile_event_ring_record_at(
        &stream->events, position) = *event;
  }
  iree_slim_mutex_unlock(&stream->mutex);
}

bool iree_hal_amd_gpu_profile_captures_dispatches(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile && (profile->families &
                     (IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA));
}

bool iree_hal_amd_gpu_profile_timestamps_dispatches(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile &&
         (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS);
}

bool iree_hal_amd_gpu_profile_captures_commands(
    const iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    uint64_t command_buffer_id) {
  if (!iree_hal_amd_gpu_profile_captures_dispatches(profile)) {
    return false;
  }
  const iree_hal_profile_capture_filter_t* filter = &profile->filter;
  return (!(filter->flags &
            IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID) ||
          filter->command_buffer_id == command_buffer_id) &&
         (!(filter->flags &
            IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL) ||
          filter->physical_device_ordinal == 0) &&
         (!(filter->flags &
            IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL) ||
          filter->queue_ordinal == queue_ordinal);
}

bool iree_hal_amd_gpu_profile_captures_dispatch(
    const iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal,
    uint64_t command_buffer_id, uint32_t command_index,
    iree_string_view_t function_name) {
  if (!iree_hal_amd_gpu_profile_captures_dispatches(profile)) {
    return false;
  }
  const iree_hal_profile_capture_filter_t* filter = &profile->filter;
  return iree_hal_profile_capture_filter_matches_location(
             filter, command_buffer_id, command_index, 0, queue_ordinal) &&
         (!(filter->flags &
            IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) ||
          iree_string_view_match_pattern(function_name,
                                         filter->executable_function_pattern));
}

void iree_hal_amd_gpu_profile_operation_complete(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_operation_t* operation,
    const iree_hal_amd_gpu_profile_completion_t* completion) {
  if (!completion->count) {
    return;
  }
  iree_hal_amd_gpu_profile_stream_t* stream =
      &profile->streams[operation->event.queue_ordinal];
  // Only completed native owners reach this boundary. Each retained record
  // copies its sample before invocation storage can return to the producer.
  iree_slim_mutex_lock(&stream->mutex);
  for (uint32_t i = 0; i < completion->count; ++i) {
    const iree_hal_amd_gpu_profile_dispatch_t* dispatch =
        operation->commands
            ? &operation->commands->dispatches[completion->first + i]
            : &operation->dispatch;
    // Direct dispatch selection precedes sample reservation. Recorded commands
    // retain one native sample layout and select during this existing walk.
    if (operation->commands && profile->filter.flags) {
      const iree_string_view_t name =
          (profile->filter.flags &
           IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN)
              ? dispatch->executable
                    ->function_names[dispatch->event.function_ordinal]
              : iree_string_view_empty();
      if (!iree_hal_amd_gpu_profile_captures_dispatch(
              profile, operation->event.queue_ordinal,
              dispatch->event.command_buffer_id, dispatch->event.command_index,
              name)) {
        continue;
      }
    }
    uint64_t position = 0, event_id = 0;
    if (!iree_hal_profile_event_ring_try_append(&stream->dispatches, &position,
                                                &event_id)) {
      continue;
    }
    iree_hal_profile_dispatch_event_t* event =
        iree_hal_profile_event_ring_record_at(&stream->dispatches, position);
    *event = dispatch->event;
    event->event_id = event_id;
    event->submission_id = operation->event.submission_id;
    if (completion->samples) {
      const iree_hal_amd_gpu_profile_sample_t* sample =
          (const void*)((const uint8_t*)completion->samples +
                        (iree_host_size_t)i * completion->sample_stride);
      event->start_tick = sample->begin;
      event->end_tick = sample->end;
    }
    stream->metadata[position & stream->dispatches.mask] =
        (iree_hal_amd_gpu_profile_metadata_t){
            .executable = dispatch->executable,
            .commands = operation->commands,
        };
    if (operation->commands) {
      iree_hal_amd_gpu_profile_commands_retain(operation->commands);
    } else {
      iree_hal_amd_gpu_profile_executable_retain(dispatch->executable);
    }
  }
  iree_slim_mutex_unlock(&stream->mutex);
}
