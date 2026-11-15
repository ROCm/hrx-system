// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/profile.h"

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
  // Protects placement publication and bounded flush copies, never callouts.
  iree_slim_mutex_t loads_mutex;
  // Packed retained placement history, extended only on queue acquisition.
  iree_byte_span_t loads;
  // Initial inline records; a grown loads array owns its separate allocation.
  uint8_t* initial_loads;
  // Capacity of loads in records, including unused trailing storage.
  iree_host_size_t load_capacity;
  // Records already emitted in emitted_session; reset when a new session emits.
  iree_host_size_t emitted_load_count;
  // Last emitted device-local session, accessed only by serialized flush/end.
  iree_hal_profile_session_id_t emitted_session;
};

iree_status_t iree_hal_amd_gpu_profile_executable_add_load(
    iree_hal_amd_gpu_profile_executable_t* metadata, uint64_t virtual_base,
    uint64_t byte_length, uint64_t address) {
  if (!metadata) {
    return iree_ok_status();
  }
  const iree_host_size_t record_size =
      sizeof(iree_hal_profile_executable_code_object_load_record_t);
  // The executable serializes placement additions. Published records are
  // immutable, so allocation and copying need no exclusion from flush readers.
  const iree_host_size_t count = metadata->loads.data_length / record_size;
  iree_status_t status = iree_ok_status();
  uint8_t* records = NULL;
  iree_host_size_t capacity = metadata->load_capacity;
  uint8_t* retired_records = NULL;
  if (count == metadata->load_capacity) {
    if (!iree_host_size_checked_mul_add(4, count, 2, &capacity)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "GPU profile placement history overflows");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(metadata->host_allocator, capacity,
                                           record_size, (void**)&records);
    }
    if (iree_status_is_ok(status)) {
      memcpy(records, metadata->loads.data, metadata->loads.data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_executable_code_object_load_record_t load =
        iree_hal_profile_executable_code_object_load_record_default();
    load.physical_device_ordinal = 0;
    load.executable_id = metadata->record.executable_id;
    load.code_object_id = metadata->code_object.code_object_id;
    load.load_delta = (int64_t)(address - virtual_base);
    load.load_size = byte_length;
    iree_slim_mutex_lock(&metadata->loads_mutex);
    if (records) {
      retired_records = metadata->loads.data;
      metadata->loads.data = records;
      metadata->load_capacity = capacity;
    }
    memcpy(metadata->loads.data + metadata->loads.data_length, &load,
           sizeof(load));
    metadata->loads.data_length += record_size;
    iree_slim_mutex_unlock(&metadata->loads_mutex);
  }
  if (retired_records && retired_records != metadata->initial_loads) {
    iree_allocator_free(metadata->host_allocator, retired_records);
  }
  return status;
}

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
  iree_hal_profile_session_id_t emitted_session;
};

static iree_atomic_int64_t iree_hal_amd_gpu_profile_next_executable_id =
    IREE_ATOMIC_VAR_INIT(1);

static void iree_hal_amd_gpu_profile_executable_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amd_gpu_profile_executable_t* metadata = (void*)resource;
  if (metadata->loads.data != metadata->initial_loads) {
    iree_allocator_free(metadata->host_allocator, metadata->loads.data);
  }
  iree_slim_mutex_deinitialize(&metadata->loads_mutex);
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
    iree_slim_mutex_initialize(&metadata->loads_mutex);
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
    metadata->loads = iree_make_byte_span(
        (uint8_t*)metadata + loads_offset,
        load_count *
            sizeof(iree_hal_profile_executable_code_object_load_record_t));
    metadata->initial_loads = metadata->loads.data;
    metadata->load_capacity = load_count;
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
  // Queue metadata accepted by the sink; accessed only by explicit flush/end.
  bool emitted;
  // Queue-local acceptance identity, independent of native chunk epochs.
  iree_atomic_int64_t next_submission_id;
  // Bounded records stored beside the owning stream allocation.
  iree_hal_profile_event_ring_t events;
  // Completed device events, never pointers to still-live native samples.
  iree_hal_profile_event_ring_t dispatches;
  // Host metadata ownership for each physical dispatch-ring slot.
  iree_hal_amd_gpu_profile_metadata_t* metadata;
  // Completed counter samples: a sample record followed by its values.
  // Disabled unless the session samples counters.
  iree_hal_profile_event_ring_t counter_samples;
  // Separately owned storage of counter_samples, or NULL.
  void* counter_records;
} iree_hal_amd_gpu_profile_stream_t;

struct iree_hal_amd_gpu_profile_t {
  // Allocator owning this session, initial streams, and later stream additions.
  iree_allocator_t host_allocator;
  // Retained output sink; only explicit lifecycle calls may invoke it.
  iree_hal_profile_sink_t* sink;
  // Immutable session metadata shared by every chunk.
  iree_hal_profile_chunk_metadata_t metadata;
  // Highest installed queue coordinate plus one; protected by streams_mutex.
  uint32_t queue_count;
  // Initial streams share the session allocation; later streams own theirs.
  uint32_t initial_queue_count;
  // Protects directory publication and snapshots, never allocation or output.
  iree_slim_mutex_t streams_mutex;
  // Last emitted device metadata; accessed only by explicit flush/end.
  iree_hal_profile_device_record_t device;
  // Admitted immutable families; controls capture and stream emission.
  iree_hal_device_profiling_data_families_t families;
  // Immutable predicates; the optional pattern lives in trailing storage.
  iree_hal_profile_capture_filter_t filter;
  // Stable directory for the device's bounded native coordinates. Only actual
  // participants allocate rings; ordinary devices allocate no profiling state.
  iree_hal_amd_gpu_profile_stream_t* streams[256];
  // The device's single thread-trace bracket. A publisher claims it, host
  // completion fills |pending|, and the next flush emits and frees it. Each
  // owner touches |pending| only in its own state, so it needs no lock.
  struct {
    // Borrowed device resources, or NULL when the session cannot trace.
    const iree_hal_amd_gpu_profile_trace_storage_t* storage;
    // One of iree_hal_amd_gpu_profile_trace_state_e.
    iree_atomic_int32_t state;
    // Granted claims not returned by cancel; bounded by storage->trace_limit.
    iree_atomic_int32_t started;
    // Matching operations refused while an earlier trace held the bracket.
    iree_atomic_int32_t skipped;
    // Identifier of the last completed trace; completion-owned.
    uint64_t last_trace_id;
    // Completed trace awaiting flush.
    struct {
      // Trace identity; shader engine and data length are filled per engine.
      iree_hal_profile_executable_trace_record_t record;
      // Retained code-object owner of every dispatch inside the bracket.
      iree_hal_amd_gpu_profile_metadata_t metadata;
      // Dispatches executed inside the bracket.
      uint32_t dispatch_count;
      // Matching operations refused while this trace held the bracket.
      uint32_t skipped_count;
    } pending;
    // Engines whose capture record and trace chunk a failed flush already
    // wrote; flush-owned.
    uint32_t emitted_captures;
    uint32_t emitted_traces;
  } trace;
  // Hardware counter sampling. Publishers claim the sampled range, host
  // completion turns raw samples into records and flush emits them.
  struct {
    // Borrowed device resources, or NULL when the session cannot sample.
    const iree_hal_amd_gpu_profile_counter_storage_t* storage;
    // Byte size of one sample record with its values.
    iree_host_size_t record_size;
    // Owner of the sampled range: (queue ordinal + 1) << 32 | its operations
    // in flight, or zero when no queue holds it.
    iree_atomic_int64_t window;
    // Dispatches the session's bound still admits.
    iree_atomic_int64_t remaining;
    // A claim found the bound exhausted.
    iree_atomic_int32_t limit_reached;
    // Identifier of the next sample record.
    iree_atomic_int64_t next_sample_id;
    // Cumulative facts reported by the capture record.
    iree_atomic_int64_t sampled_operations;
    iree_atomic_int64_t sampled_dispatches;
    iree_atomic_int64_t skipped_operations;
    iree_atomic_int64_t dropped_samples;
    iree_atomic_int64_t invalid_samples;
    iree_atomic_int64_t foreign_states;
    // Set and counter metadata accepted by the sink; flush-owned.
    bool metadata_emitted;
    // Last capture record accepted by the sink; flush-owned.
    bool capture_emitted;
    iree_hal_profile_counter_capture_record_t capture;
  } counters;
};

// Identifier of the session's only counter set.
#define IREE_HAL_AMD_GPU_PROFILE_COUNTER_SET_ID 1

enum iree_hal_amd_gpu_profile_trace_state_e {
  // No trace in flight; a publisher may claim the bracket.
  IREE_HAL_AMD_GPU_PROFILE_TRACE_FREE = 0,
  // Claimed by a publication that has not completed on the host.
  IREE_HAL_AMD_GPU_PROFILE_TRACE_CLAIMED = 1,
  // Stopped and collected; buffers stay untouched until flush emits them.
  IREE_HAL_AMD_GPU_PROFILE_TRACE_COMPLETE = 2,
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
  if (iree_atomic_load(&profile->trace.state, iree_memory_order_acquire) ==
      IREE_HAL_AMD_GPU_PROFILE_TRACE_COMPLETE) {
    iree_hal_amd_gpu_profile_metadata_release(profile->trace.pending.metadata);
  }
  for (uint32_t i = 0; i < profile->queue_count; ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = profile->streams[i];
    if (!stream) {
      continue;
    }
    for (uint64_t j = stream->dispatches.read_position;
         j != stream->dispatches.write_position; ++j) {
      iree_hal_amd_gpu_profile_metadata_release(
          stream->metadata[j & stream->dispatches.mask]);
    }
    iree_slim_mutex_deinitialize(&stream->mutex);
    iree_allocator_free(profile->host_allocator, stream->counter_records);
    if (i >= profile->initial_queue_count) {
      iree_allocator_free(profile->host_allocator, stream);
    }
  }
  iree_slim_mutex_deinitialize(&profile->streams_mutex);
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

static iree_status_t iree_hal_amd_gpu_profile_write_executable_metadata(
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
  executable->emitted_session = profile->metadata.session_id;
  executable->emitted_load_count = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_profile_write_executable(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_executable_t* executable) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_profile_write_executable_metadata(profile, executable));
  iree_slim_mutex_lock(&executable->loads_mutex);
  const iree_host_size_t record_size =
      sizeof(iree_hal_profile_executable_code_object_load_record_t);
  // A finite prefix prevents concurrent placement from extending this flush.
  const iree_host_size_t count = executable->loads.data_length / record_size;
  iree_slim_mutex_unlock(&executable->loads_mutex);
  // Sink input must survive placement-history growth without owning another
  // allocation. Bound shared exclusion to a small copy, independent of history.
  iree_hal_profile_executable_code_object_load_record_t records[32];
  iree_status_t status = iree_ok_status();
  while (executable->emitted_load_count < count && iree_status_is_ok(status)) {
    const iree_host_size_t batch_count = iree_min(
        count - executable->emitted_load_count, IREE_ARRAYSIZE(records));
    iree_slim_mutex_lock(&executable->loads_mutex);
    memcpy(
        records,
        executable->loads.data + executable->emitted_load_count * record_size,
        batch_count * record_size);
    iree_slim_mutex_unlock(&executable->loads_mutex);
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS,
        UINT32_MAX,
        iree_make_const_byte_span(records, batch_count * record_size));
    if (iree_status_is_ok(status)) {
      executable->emitted_load_count += batch_count;
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_gpu_profile_flush_dispatches(
    iree_hal_amd_gpu_profile_t* profile, uint32_t ordinal,
    iree_hal_amd_gpu_profile_stream_t* stream) {
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

static void iree_hal_amd_gpu_profile_stream_initialize(
    iree_hal_amd_gpu_profile_stream_t* stream,
    iree_hal_profile_queue_event_t* records,
    iree_hal_profile_dispatch_event_t* dispatches,
    iree_hal_amd_gpu_profile_metadata_t* metadata) {
  iree_slim_mutex_initialize(&stream->mutex);
  stream->emitted = false;
  iree_atomic_store(&stream->next_submission_id, 1, iree_memory_order_relaxed);
  iree_hal_profile_event_ring_initialize(records, sizeof(*records),
                                         IREE_HAL_AMD_GPU_PROFILE_CAPACITY,
                                         &stream->events);
  iree_hal_profile_event_ring_initialize(
      dispatches, sizeof(*dispatches),
      dispatches ? IREE_HAL_AMD_GPU_PROFILE_CAPACITY : 0, &stream->dispatches);
  stream->metadata = metadata;
  iree_hal_profile_event_ring_initialize(NULL, 0, 0, &stream->counter_samples);
  stream->counter_records = NULL;
}

// Gives a stream its counter-sample records. The stream is not yet visible to
// completion: the session has no queue, or the queue is still registering.
static iree_status_t iree_hal_amd_gpu_profile_stream_allocate_counters(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_stream_t* stream) {
  if (!profile->counters.storage || stream->counter_records) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      profile->host_allocator, IREE_HAL_AMD_GPU_PROFILE_CAPACITY,
      profile->counters.record_size, &stream->counter_records));
  iree_hal_profile_event_ring_initialize(
      stream->counter_records, profile->counters.record_size,
      IREE_HAL_AMD_GPU_PROFILE_CAPACITY, &stream->counter_samples);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_profile_register_queue(
    iree_hal_amd_gpu_profile_t* profile, uint32_t queue_ordinal) {
  if (!profile) {
    return iree_ok_status();
  }
  // Membership serializes registration of each coordinate. Flush only reads
  // the directory, and installed streams live until the session ends.
  if (profile->streams[queue_ordinal]) {
    return iree_ok_status();
  }
  const uint32_t dispatch_count =
      iree_hal_amd_gpu_profile_captures_dispatches(profile)
          ? IREE_HAL_AMD_GPU_PROFILE_CAPACITY
          : 0;
  iree_host_size_t size = 0, records_offset = 0, dispatches_offset = 0;
  iree_host_size_t metadata_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_gpu_profile_stream_t), &size,
      IREE_STRUCT_FIELD(IREE_HAL_AMD_GPU_PROFILE_CAPACITY,
                        iree_hal_profile_queue_event_t, &records_offset),
      IREE_STRUCT_FIELD(dispatch_count, iree_hal_profile_dispatch_event_t,
                        &dispatches_offset),
      IREE_STRUCT_FIELD(dispatch_count, iree_hal_amd_gpu_profile_metadata_t,
                        &metadata_offset));
  iree_hal_amd_gpu_profile_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(profile->host_allocator, size, (void**)&stream);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_profile_stream_initialize(
        stream, (void*)((uint8_t*)stream + records_offset),
        dispatch_count ? (void*)((uint8_t*)stream + dispatches_offset) : NULL,
        dispatch_count ? (void*)((uint8_t*)stream + metadata_offset) : NULL);
    status = iree_hal_amd_gpu_profile_stream_allocate_counters(profile, stream);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_deinitialize(&stream->mutex);
      iree_allocator_free(profile->host_allocator, stream);
      return status;
    }
    iree_slim_mutex_lock(&profile->streams_mutex);
    profile->streams[queue_ordinal] = stream;
    profile->queue_count = iree_max(profile->queue_count, queue_ordinal + 1);
    iree_slim_mutex_unlock(&profile->streams_mutex);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_profile_begin(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_profile_session_id_t session_id, uint32_t queue_count,
    uint64_t timestamp_frequency_hz, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_profile_t** out_profile) {
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
  const uint64_t supported =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES |
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  if (families & ~supported) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported GPU profiling data families 0x%" PRIx64,
        families & ~supported);
  }
  if (options->counter_set_count &&
      !(families & IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "GPU counter selections require the counter-sample family");
  }
  // A counter sample carries its dispatch's tick range.
  if ((families & (IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                   IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES)) &&
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
  // Traces are attributed through the captured dispatches' code objects, and
  // counter samples through the dispatches themselves.
  const iree_host_size_t dispatch_count =
      (families & (IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                   IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                   IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES |
                   IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES))
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
  profile->initial_queue_count = queue_count;
  memset(&profile->trace, 0, sizeof(profile->trace));
  memset(&profile->counters, 0, sizeof(profile->counters));
  iree_slim_mutex_initialize(&profile->streams_mutex);
  profile->families = families;
  profile->filter = options->capture_filter;
  profile->filter.executable_function_pattern = iree_string_view_empty();
  if (pattern.size) {
    char* target = (char*)profile + pattern_offset;
    memcpy(target, pattern.data, pattern.size);
    profile->filter.executable_function_pattern =
        iree_make_string_view(target, pattern.size);
  }
  iree_hal_amd_gpu_profile_stream_t* streams =
      (void*)((uint8_t*)profile + streams_offset);
  profile->metadata = iree_hal_profile_chunk_metadata_default();
  profile->metadata.name = IREE_SV("amd-gpu");
  profile->metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_SESSION;
  profile->metadata.session_id = session_id;
  profile->metadata.physical_device_ordinal = 0;
  iree_hal_profile_queue_event_t* records =
      (void*)((uint8_t*)profile + records_offset);
  for (uint32_t i = 0; i < queue_count; ++i) {
    profile->streams[i] = &streams[i];
    iree_hal_profile_dispatch_event_t* dispatch_records =
        (void*)((uint8_t*)profile + dispatches_offset);
    iree_hal_amd_gpu_profile_metadata_t* metadata =
        (void*)((uint8_t*)profile + metadata_offset);
    const iree_host_size_t offset =
        (iree_host_size_t)i * IREE_HAL_AMD_GPU_PROFILE_CAPACITY;
    iree_hal_amd_gpu_profile_stream_initialize(
        &streams[i], records + offset,
        dispatch_count ? dispatch_records + offset : NULL,
        dispatch_count ? metadata + offset : NULL);
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
    profile->device = record;
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
    profile->streams[i]->emitted = iree_status_is_ok(status);
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

// Emits the completed trace and frees the bracket. Engines a failed attempt
// already wrote are not repeated; the bracket stays held until all are out.
static iree_status_t iree_hal_amd_gpu_profile_flush_trace(
    iree_hal_amd_gpu_profile_t* profile) {
  const iree_hal_amd_gpu_profile_trace_storage_t* storage =
      profile->trace.storage;
  const iree_hal_amd_gpu_pm4_thread_trace_config_t* config = &storage->config;
  const iree_hal_amd_gpu_profile_metadata_t* owner =
      &profile->trace.pending.metadata;
  // Every code object that ran inside the bracket precedes the trace, whether
  // or not its dispatch event passed the capture filter.
  iree_status_t status = iree_ok_status();
  if (owner->commands) {
    for (uint32_t i = 0;
         i < owner->commands->count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amd_gpu_profile_write_executable(
          profile, owner->commands->dispatches[i].executable);
    }
    if (iree_status_is_ok(status) &&
        owner->commands->emitted_session != profile->metadata.session_id) {
      status = iree_hal_amd_gpu_profile_write(
          profile, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS, UINT32_MAX,
          iree_make_const_byte_span(&owner->commands->record,
                                    sizeof(owner->commands->record)));
      if (iree_status_is_ok(status)) {
        owner->commands->emitted_session = profile->metadata.session_id;
      }
    }
  } else {
    status =
        iree_hal_amd_gpu_profile_write_executable(profile, owner->executable);
  }
  const uint32_t engines =
      config->engine_mask & ((1u << config->engine_count) - 1);
  // Sampled once per trace, when its first record goes out: every refusal
  // between this trace's claim and now waited on this trace.
  if (!profile->trace.emitted_captures) {
    profile->trace.pending.skipped_count = (uint32_t)iree_atomic_exchange(
        &profile->trace.skipped, 0, iree_memory_order_relaxed);
  }
  for (uint32_t i = 0; i < config->engine_count && iree_status_is_ok(status);
       ++i) {
    if (!(engines & (1u << i))) {
      continue;
    }
    iree_hal_amd_gpu_pm4_thread_trace_result_t result;
    iree_hal_amd_gpu_pm4_thread_trace_read_info(
        config, i,
        storage->info_words +
            (iree_host_size_t)i *
                IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_WORD_COUNT,
        &result);
    const bool incomplete = !result.collected || !result.finish_done ||
                            result.write_error || result.buffer_full ||
                            result.dropped_count;
    iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
    metadata.queue_ordinal = profile->trace.pending.record.queue_ordinal;
    metadata.stream_id = profile->trace.pending.record.stream_id;
    if (!(profile->trace.emitted_captures & (1u << i))) {
      iree_hal_profile_executable_trace_capture_record_t capture =
          iree_hal_profile_executable_trace_capture_record_default();
      capture.trace_id = profile->trace.pending.record.trace_id;
      capture.shader_engine = i;
      capture.physical_device_ordinal = 0;
      capture.shader_engine_mask = engines;
      capture.target_compute_unit = config->workgroup_processor;
      capture.simd_select = config->simd;
      capture.dispatch_count = profile->trace.pending.dispatch_count;
      capture.buffer_byte_length = config->buffer_length;
      capture.written_byte_length = result.written_length;
      capture.dropped_count = result.collected ? result.dropped_count : 0;
      capture.hardware_status = result.collected ? result.status : 0;
      capture.token_mask = config->profile->token_mask_value;
      capture.trace_limit = storage->trace_limit;
      capture.skipped_count = profile->trace.pending.skipped_count;
      if (storage->stable_power_state) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_STABLE_POWER_STATE;
      }
      if (result.finish_done) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_FINISHED;
      }
      if (result.buffer_full) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_BUFFER_FULL;
      }
      if (result.write_error) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_WRITE_ERROR;
      }
      if (owner->commands) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_COMMAND_BUFFER_SCOPE;
      }
      if (!result.collected) {
        capture.flags |=
            IREE_HAL_PROFILE_EXECUTABLE_TRACE_CAPTURE_FLAG_UNCOLLECTED;
      }
      metadata.content_type =
          IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACE_CAPTURES;
      const iree_const_byte_span_t span =
          iree_make_const_byte_span(&capture, sizeof(capture));
      status = iree_hal_profile_sink_write(profile->sink, &metadata, 1, &span);
      if (iree_status_is_ok(status)) {
        profile->trace.emitted_captures |= 1u << i;
      }
    }
    // No bytes means no trace chunk: an empty artifact would claim a partition
    // the capture record already reports as uncollected or silent.
    if (iree_status_is_ok(status) && result.written_length &&
        !(profile->trace.emitted_traces & (1u << i))) {
      iree_hal_profile_executable_trace_record_t record =
          profile->trace.pending.record;
      record.shader_engine = i;
      record.data_length = result.written_length;
      metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES;
      metadata.flags = profile->metadata.flags;
      metadata.dropped_record_count = result.dropped_count;
      if (incomplete) {
        metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
      }
      const iree_const_byte_span_t spans[] = {
          iree_make_const_byte_span(&record, sizeof(record)),
          iree_make_const_byte_span(storage->buffers[i],
                                    (iree_host_size_t)result.written_length),
      };
      status = iree_hal_profile_sink_write(profile->sink, &metadata,
                                           IREE_ARRAYSIZE(spans), spans);
      if (iree_status_is_ok(status)) {
        profile->trace.emitted_traces |= 1u << i;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_profile_metadata_release(*owner);
    profile->trace.pending.metadata = (iree_hal_amd_gpu_profile_metadata_t){0};
    profile->trace.emitted_captures = 0;
    profile->trace.emitted_traces = 0;
    iree_atomic_store(&profile->trace.state,
                      IREE_HAL_AMD_GPU_PROFILE_TRACE_FREE,
                      iree_memory_order_release);
  }
  return status;
}

// Emits the counter set and one record per selected counter, once.
static iree_status_t iree_hal_amd_gpu_profile_flush_counter_metadata(
    iree_hal_amd_gpu_profile_t* profile) {
  const iree_hal_amd_gpu_pm4_perf_counter_config_t* config =
      &profile->counters.storage->config;
  const iree_string_view_t set_name = IREE_SV("amdgpu.perfmon");
  struct {
    iree_hal_profile_counter_set_record_t record;
    char name[32];
  } set;
  memset(&set, 0, sizeof(set));
  set.record = iree_hal_profile_counter_set_record_default();
  set.record.record_length = (uint32_t)(sizeof(set.record) + set_name.size);
  set.record.counter_set_id = IREE_HAL_AMD_GPU_PROFILE_COUNTER_SET_ID;
  set.record.physical_device_ordinal = 0;
  set.record.counter_count =
      config->engine_counter_count + config->instance_counter_count;
  set.record.sample_value_count =
      iree_hal_amd_gpu_pm4_perf_counter_value_count(config);
  set.record.name_length = (uint32_t)set_name.size;
  memcpy(set.name, set_name.data, set_name.size);
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS, UINT32_MAX,
      iree_make_const_byte_span(&set, set.record.record_length)));
  for (uint32_t i = 0; i < set.record.counter_count; ++i) {
    const bool engine = i < config->engine_counter_count;
    const uint32_t ordinal = engine ? i : i - config->engine_counter_count;
    const iree_hal_amd_gpu_pm4_perf_counter_event_t* event =
        engine ? config->engine_counters[ordinal]
               : config->instance_counters[ordinal];
    struct {
      iree_hal_profile_counter_record_t record;
      char text[208];
    } counter;
    memset(&counter, 0, sizeof(counter));
    const iree_host_size_t text_length =
        event->block_name.size + event->name.size + event->description.size;
    IREE_ASSERT(text_length <= sizeof(counter.text));
    counter.record = iree_hal_profile_counter_record_default();
    counter.record.record_length =
        (uint32_t)(sizeof(counter.record) + text_length);
    counter.record.flags = IREE_HAL_PROFILE_COUNTER_FLAG_RAW;
    counter.record.unit = event->cycles ? IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES
                                        : IREE_HAL_PROFILE_COUNTER_UNIT_COUNT;
    counter.record.physical_device_ordinal = 0;
    counter.record.counter_set_id = IREE_HAL_AMD_GPU_PROFILE_COUNTER_SET_ID;
    counter.record.counter_ordinal = i;
    counter.record.sample_value_offset =
        engine ? iree_hal_amd_gpu_pm4_perf_counter_engine_value_offset(config,
                                                                       ordinal)
               : iree_hal_amd_gpu_pm4_perf_counter_instance_value_offset(
                     config, ordinal);
    counter.record.sample_value_count =
        engine ? config->engine_count : config->instance_count;
    counter.record.block_name_length = (uint32_t)event->block_name.size;
    counter.record.name_length = (uint32_t)event->name.size;
    counter.record.description_length = (uint32_t)event->description.size;
    char* text = counter.text;
    memcpy(text, event->block_name.data, event->block_name.size);
    text += event->block_name.size;
    memcpy(text, event->name.data, event->name.size);
    text += event->name.size;
    memcpy(text, event->description.data, event->description.size);
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS, UINT32_MAX,
        iree_make_const_byte_span(&counter, counter.record.record_length)));
  }
  return iree_ok_status();
}

// Emits one queue's completed counter samples.
static iree_status_t iree_hal_amd_gpu_profile_flush_counter_samples(
    iree_hal_amd_gpu_profile_t* profile, uint32_t ordinal,
    iree_hal_amd_gpu_profile_stream_t* stream) {
  iree_hal_profile_event_ring_snapshot_t snapshot;
  iree_slim_mutex_lock(&stream->mutex);
  iree_status_t status =
      iree_hal_profile_event_ring_snapshot(&stream->counter_samples, &snapshot);
  iree_slim_mutex_unlock(&stream->mutex);
  if (!iree_status_is_ok(status) ||
      (!snapshot.record_count && !snapshot.dropped_record_count)) {
    return status;
  }
  iree_hal_profile_chunk_metadata_t metadata = profile->metadata;
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  metadata.queue_ordinal = ordinal;
  metadata.stream_id = ordinal + 1ull;
  metadata.dropped_record_count = snapshot.dropped_record_count;
  if (snapshot.dropped_record_count) {
    metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  }
  status = iree_hal_profile_sink_write(
      profile->sink, &metadata, snapshot.record_span_count,
      snapshot.record_span_count ? snapshot.record_spans : NULL);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&stream->mutex);
    iree_hal_profile_event_ring_commit_snapshot(&stream->counter_samples,
                                                &snapshot);
    iree_slim_mutex_unlock(&stream->mutex);
  }
  return status;
}

// Emits the session's cumulative sampling facts when they changed.
static iree_status_t iree_hal_amd_gpu_profile_flush_counter_capture(
    iree_hal_amd_gpu_profile_t* profile) {
  const iree_hal_amd_gpu_profile_counter_storage_t* storage =
      profile->counters.storage;
  iree_hal_profile_counter_capture_record_t capture =
      iree_hal_profile_counter_capture_record_default();
  // The range is exclusive across this device's queues by construction; the
  // hardware still counts any other enabled queue of the process.
  capture.flags = IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_EXCLUSIVE_WINDOW |
                  IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_PROCESS_SCOPE;
  if (storage->stable_power_state) {
    capture.flags |= IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_STABLE_POWER_STATE;
  }
  if (iree_atomic_load(&profile->counters.limit_reached,
                       iree_memory_order_relaxed)) {
    capture.flags |= IREE_HAL_PROFILE_COUNTER_CAPTURE_FLAG_LIMIT_REACHED;
  }
  capture.counter_set_id = IREE_HAL_AMD_GPU_PROFILE_COUNTER_SET_ID;
  capture.physical_device_ordinal = 0;
  capture.shader_engine_count = storage->config.engine_count;
  capture.instance_count = storage->config.instance_counter_count
                               ? storage->config.instance_count
                               : 0;
  capture.sample_limit = storage->sample_limit;
  capture.sampled_operation_count = (uint64_t)iree_atomic_load(
      &profile->counters.sampled_operations, iree_memory_order_relaxed);
  capture.sampled_dispatch_count = (uint64_t)iree_atomic_load(
      &profile->counters.sampled_dispatches, iree_memory_order_relaxed);
  capture.skipped_operation_count = (uint64_t)iree_atomic_load(
      &profile->counters.skipped_operations, iree_memory_order_relaxed);
  capture.dropped_sample_count = (uint64_t)iree_atomic_load(
      &profile->counters.dropped_samples, iree_memory_order_relaxed);
  capture.invalid_sample_count = (uint64_t)iree_atomic_load(
      &profile->counters.invalid_samples, iree_memory_order_relaxed);
  capture.foreign_state_count = (uint64_t)iree_atomic_load(
      &profile->counters.foreign_states, iree_memory_order_relaxed);
  if (profile->counters.capture_emitted &&
      !memcmp(&capture, &profile->counters.capture, sizeof(capture))) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_profile_write(
      profile, IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_CAPTURES, UINT32_MAX,
      iree_make_const_byte_span(&capture, sizeof(capture))));
  profile->counters.capture = capture;
  profile->counters.capture_emitted = true;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_profile_flush(
    iree_hal_amd_gpu_profile_t* profile) {
  if (!profile) {
    return iree_ok_status();
  }
  // Sampled before the streams: completion publishes a trace after its
  // dispatch events, so events of a trace seen here are in this flush.
  const bool trace_complete =
      iree_atomic_load(&profile->trace.state, iree_memory_order_acquire) ==
      IREE_HAL_AMD_GPU_PROFILE_TRACE_COMPLETE;
  // Explicit lifecycle calls are externally serialized. Registration may run
  // during output, but every captured stream remains valid through session end.
  iree_hal_amd_gpu_profile_stream_t* streams[IREE_ARRAYSIZE(profile->streams)];
  iree_slim_mutex_lock(&profile->streams_mutex);
  const uint32_t queue_count = profile->queue_count;
  memcpy(streams, profile->streams, queue_count * sizeof(*streams));
  iree_slim_mutex_unlock(&profile->streams_mutex);

  // Metadata precedes every referring event. Commit successful queue records
  // separately so a failed device record can retry without repeating them.
  iree_status_t status = iree_ok_status();
  uint32_t stream_count = 0;
  for (uint32_t i = 0; i < queue_count && iree_status_is_ok(status); ++i) {
    if (!streams[i]) {
      continue;
    }
    ++stream_count;
    if (streams[i]->emitted) {
      continue;
    }
    iree_hal_profile_queue_record_t record =
        iree_hal_profile_queue_record_default();
    record.physical_device_ordinal = 0;
    record.queue_ordinal = i;
    record.stream_id = i + 1ull;
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES, i,
        iree_make_const_byte_span(&record, sizeof(record)));
    streams[i]->emitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status) &&
      stream_count != profile->device.queue_count) {
    iree_hal_profile_device_record_t device = profile->device;
    device.queue_count = stream_count;
    status = iree_hal_amd_gpu_profile_write(
        profile, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES, UINT32_MAX,
        iree_make_const_byte_span(&device, sizeof(device)));
    if (iree_status_is_ok(status)) {
      profile->device = device;
    }
  }
  // Counter metadata precedes every sample that refers to it.
  if (iree_status_is_ok(status) && profile->counters.storage &&
      !profile->counters.metadata_emitted) {
    status = iree_hal_amd_gpu_profile_flush_counter_metadata(profile);
    profile->counters.metadata_emitted = iree_status_is_ok(status);
  }
  for (uint32_t i = 0; i < queue_count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_gpu_profile_stream_t* stream = streams[i];
    if (!stream) {
      continue;
    }
    status = iree_hal_amd_gpu_profile_flush_dispatches(profile, i, stream);
    if (iree_status_is_ok(status) && profile->counters.storage) {
      status =
          iree_hal_amd_gpu_profile_flush_counter_samples(profile, i, stream);
    }
    if (!iree_status_is_ok(status)) {
      continue;
    }
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
  if (iree_status_is_ok(status) && profile->counters.storage) {
    status = iree_hal_amd_gpu_profile_flush_counter_capture(profile);
  }
  if (trace_complete && iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_profile_flush_trace(profile);
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
  event->submission_id = iree_atomic_fetch_add(
      &profile->streams[queue_ordinal]->next_submission_id, 1,
      iree_memory_order_relaxed);
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
      profile->streams[event->queue_ordinal];
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
                      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
                      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES |
                      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES));
}

bool iree_hal_amd_gpu_profile_traces_executables(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile &&
         (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

void iree_hal_amd_gpu_profile_set_trace_storage(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_trace_storage_t* storage) {
  if (iree_hal_amd_gpu_profile_traces_executables(profile)) {
    profile->trace.storage = storage;
  }
}

bool iree_hal_amd_gpu_profile_timestamps_dispatches(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile &&
         (profile->families & (IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                               IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES));
}

bool iree_hal_amd_gpu_profile_counts_dispatches(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile &&
         (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES);
}

iree_status_t iree_hal_amd_gpu_profile_set_counter_storage(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_counter_storage_t* storage) {
  if (!iree_hal_amd_gpu_profile_counts_dispatches(profile)) {
    return iree_ok_status();
  }
  profile->counters.record_size =
      sizeof(iree_hal_profile_counter_sample_record_t) +
      iree_hal_amd_gpu_pm4_perf_counter_value_count(&storage->config) *
          sizeof(uint64_t);
  profile->counters.storage = storage;
  iree_atomic_store(
      &profile->counters.remaining,
      (int64_t)iree_min(storage->sample_limit, (uint64_t)INT64_MAX),
      iree_memory_order_relaxed);
  iree_atomic_store(&profile->counters.next_sample_id, 1,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < profile->queue_count && iree_status_is_ok(status);
       ++i) {
    if (profile->streams[i]) {
      status = iree_hal_amd_gpu_profile_stream_allocate_counters(
          profile, profile->streams[i]);
    }
  }
  if (!iree_status_is_ok(status)) {
    // No queue has borrowed the session; it simply samples nothing.
    profile->counters.storage = NULL;
  }
  return status;
}

uint32_t iree_hal_amd_gpu_profile_counter_sample_byte_length(
    const iree_hal_amd_gpu_profile_t* profile) {
  return profile && profile->counters.storage
             ? 2 * iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(
                       &profile->counters.storage->config)
             : 0;
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

// A direct dispatch was selected at acceptance. A recording is traced whole
// when any of its dispatches passes the session's predicates.
static bool iree_hal_amd_gpu_profile_trace_matches(
    const iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_operation_t* operation) {
  if (!operation->completion.count) {
    return false;
  }
  if (!operation->commands || !profile->filter.flags) {
    return true;
  }
  const bool named =
      profile->filter.flags &
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  for (uint32_t i = 0; i < operation->commands->count; ++i) {
    const iree_hal_amd_gpu_profile_dispatch_t* dispatch =
        &operation->commands->dispatches[i];
    if (iree_hal_amd_gpu_profile_captures_dispatch(
            profile, operation->event.queue_ordinal,
            dispatch->event.command_buffer_id, dispatch->event.command_index,
            named ? dispatch->executable
                        ->function_names[dispatch->event.function_ordinal]
                  : iree_string_view_empty())) {
      return true;
    }
  }
  return false;
}

const iree_hal_amd_gpu_pm4_thread_trace_config_t*
iree_hal_amd_gpu_profile_trace_claim(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  if (!profile || !operation) {
    return NULL;
  }
  const iree_hal_amd_gpu_profile_trace_storage_t* storage =
      profile->trace.storage;
  if (!storage ||
      (uint32_t)iree_atomic_load(&profile->trace.started,
                                 iree_memory_order_relaxed) >=
          storage->trace_limit ||
      !iree_hal_amd_gpu_profile_trace_matches(profile, operation)) {
    return NULL;
  }
  int32_t expected = IREE_HAL_AMD_GPU_PROFILE_TRACE_FREE;
  if (!iree_atomic_compare_exchange_strong(
          &profile->trace.state, &expected,
          IREE_HAL_AMD_GPU_PROFILE_TRACE_CLAIMED, iree_memory_order_acq_rel,
          iree_memory_order_relaxed)) {
    // Publication can be attempted several times before it is accepted.
    if (!operation->trace_refused) {
      operation->trace_refused = true;
      iree_atomic_fetch_add(&profile->trace.skipped, 1,
                            iree_memory_order_relaxed);
    }
    return NULL;
  }
  // The bracket is exclusive from here; the bound is exact under that owner.
  if ((uint32_t)iree_atomic_fetch_add(&profile->trace.started, 1,
                                      iree_memory_order_relaxed) >=
      storage->trace_limit) {
    iree_hal_amd_gpu_profile_trace_cancel(profile);
    return NULL;
  }
  // An info word the stop sequence never replaced must not read as a result.
  const iree_host_size_t info_word_count =
      (iree_host_size_t)storage->config.engine_count *
      IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_WORD_COUNT;
  for (iree_host_size_t i = 0; i < info_word_count; ++i) {
    ((volatile uint32_t*)storage->info_words)[i] =
        (uint32_t)IREE_HAL_AMD_GPU_PM4_THREAD_TRACE_INFO_UNWRITTEN;
  }
  return &storage->config;
}

void iree_hal_amd_gpu_profile_trace_cancel(
    iree_hal_amd_gpu_profile_t* profile) {
  iree_atomic_fetch_sub(&profile->trace.started, 1, iree_memory_order_relaxed);
  iree_atomic_store(&profile->trace.state, IREE_HAL_AMD_GPU_PROFILE_TRACE_FREE,
                    iree_memory_order_release);
}

// Counts one matching operation that ran unsampled, once per operation.
static void iree_hal_amd_gpu_profile_counter_refuse(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  if (!operation->counter_refused) {
    operation->counter_refused = true;
    iree_atomic_fetch_add(&profile->counters.skipped_operations, 1,
                          iree_memory_order_relaxed);
  }
}

// Leaves the sampled range; the last operation of the owning queue frees it.
static void iree_hal_amd_gpu_profile_counter_window_release(
    iree_hal_amd_gpu_profile_t* profile) {
  int64_t window =
      iree_atomic_load(&profile->counters.window, iree_memory_order_relaxed);
  int64_t next = 0;
  do {
    next = (window & UINT32_MAX) > 1 ? window - 1 : 0;
  } while (!iree_atomic_compare_exchange_weak(
      &profile->counters.window, &window, next, iree_memory_order_acq_rel,
      iree_memory_order_relaxed));
}

const iree_hal_amd_gpu_profile_counter_storage_t*
iree_hal_amd_gpu_profile_counter_claim(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation,
    const uint64_t* recorded_signature) {
  if (!profile || !operation) {
    return NULL;
  }
  const iree_hal_amd_gpu_profile_counter_storage_t* storage =
      profile->counters.storage;
  if (!storage || !iree_hal_amd_gpu_profile_trace_matches(profile, operation)) {
    return NULL;
  }
  if (iree_atomic_load(&profile->counters.remaining,
                       iree_memory_order_relaxed) <= 0) {
    iree_atomic_store(&profile->counters.limit_reached, 1,
                      iree_memory_order_relaxed);
    return NULL;
  }
  // A recording carries the sample sequences it was prepared with. One made
  // without them, or for another selection, cannot be sampled per dispatch.
  if (recorded_signature && *recorded_signature != storage->signature) {
    iree_hal_amd_gpu_profile_counter_refuse(profile, operation);
    return NULL;
  }
  const int64_t owner = ((int64_t)operation->event.queue_ordinal + 1) << 32;
  int64_t window =
      iree_atomic_load(&profile->counters.window, iree_memory_order_relaxed);
  do {
    if (window && (window & ~(int64_t)UINT32_MAX) != owner) {
      iree_hal_amd_gpu_profile_counter_refuse(profile, operation);
      return NULL;
    }
  } while (!iree_atomic_compare_exchange_weak(
      &profile->counters.window, &window, window ? window + 1 : owner + 1,
      iree_memory_order_acq_rel, iree_memory_order_relaxed));
  // The bound is exact: each claim takes what it will emit at most.
  int64_t remaining =
      iree_atomic_load(&profile->counters.remaining, iree_memory_order_relaxed);
  int64_t grant = 0;
  do {
    grant = iree_min(remaining, (int64_t)operation->completion.count);
  } while (grant > 0 &&
           !iree_atomic_compare_exchange_weak(
               &profile->counters.remaining, &remaining, remaining - grant,
               iree_memory_order_acq_rel, iree_memory_order_relaxed));
  if (grant <= 0) {
    iree_atomic_store(&profile->counters.limit_reached, 1,
                      iree_memory_order_relaxed);
    iree_hal_amd_gpu_profile_counter_window_release(profile);
    return NULL;
  }
  // An earlier attempt of this operation was refused and counted as skipped.
  if (operation->counter_refused) {
    operation->counter_refused = false;
    iree_atomic_fetch_sub(&profile->counters.skipped_operations, 1,
                          iree_memory_order_relaxed);
  }
  operation->counter_grant = (uint32_t)grant;
  return storage;
}

void iree_hal_amd_gpu_profile_counter_cancel(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  iree_atomic_fetch_add(&profile->counters.remaining, operation->counter_grant,
                        iree_memory_order_relaxed);
  operation->counter_grant = 0;
  iree_hal_amd_gpu_profile_counter_window_release(profile);
}

void iree_hal_amd_gpu_profile_counter_skip(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_operation_t* operation) {
  iree_hal_amd_gpu_profile_counter_refuse(profile, operation);
}

// Turns one dispatch's two raw samples into a sample record. The caller holds
// the stream mutex and has appended the dispatch's own event. Returns false
// when no record was kept.
static bool iree_hal_amd_gpu_profile_counter_sample(
    iree_hal_amd_gpu_profile_t* profile,
    iree_hal_amd_gpu_profile_stream_t* stream,
    const iree_hal_amd_gpu_profile_operation_t* operation,
    const iree_hal_profile_dispatch_event_t* event,
    const iree_hal_amd_gpu_profile_sample_t* sample) {
  const iree_hal_amd_gpu_pm4_perf_counter_config_t* config =
      &profile->counters.storage->config;
  const uint32_t raw_length =
      iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(config);
  // Sample identifiers are unique across the session's streams, so the
  // ring's own per-stream identifier is not used.
  uint64_t position = 0, ring_id = 0;
  if (!iree_hal_profile_event_ring_try_append(&stream->counter_samples,
                                              &position, &ring_id)) {
    iree_atomic_fetch_add(&profile->counters.dropped_samples, 1,
                          iree_memory_order_relaxed);
    return false;
  }
  uint8_t* slot =
      iree_hal_profile_event_ring_record_at(&stream->counter_samples, position);
  uint64_t* values =
      (uint64_t*)(slot + sizeof(iree_hal_profile_counter_sample_record_t));
  if (!iree_hal_amd_gpu_pm4_perf_counter_sample_delta(
          config, (const uint8_t*)(sample + 1),
          (const uint8_t*)(sample + 1) + raw_length, values)) {
    // The slot is the newest record and the mutex is held: give it back.
    --stream->counter_samples.write_position;
    iree_atomic_fetch_add(&profile->counters.invalid_samples, 1,
                          iree_memory_order_relaxed);
    return false;
  }
  iree_hal_profile_counter_sample_record_t record =
      iree_hal_profile_counter_sample_record_default();
  record.record_length = (uint32_t)profile->counters.record_size;
  record.scope = IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH;
  record.flags = IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE;
  if (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS) {
    record.flags |= IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT;
    record.dispatch_event_id = event->event_id;
  }
  if (operation->commands) {
    record.flags |= IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION;
    record.command_buffer_id = event->command_buffer_id;
    record.command_index = event->command_index;
  }
  record.sample_id = (uint64_t)iree_atomic_fetch_add(
      &profile->counters.next_sample_id, 1, iree_memory_order_relaxed);
  record.counter_set_id = IREE_HAL_AMD_GPU_PROFILE_COUNTER_SET_ID;
  record.submission_id = operation->event.submission_id;
  record.executable_id = event->executable_id;
  record.function_ordinal = event->function_ordinal;
  record.stream_id = operation->event.stream_id;
  record.start_tick = sample->begin;
  record.end_tick = sample->end;
  record.physical_device_ordinal = 0;
  record.queue_ordinal = operation->event.queue_ordinal;
  record.sample_value_count =
      iree_hal_amd_gpu_pm4_perf_counter_value_count(config);
  memcpy(slot, &record, sizeof(record));
  iree_atomic_fetch_add(&profile->counters.sampled_dispatches, 1,
                        iree_memory_order_relaxed);
  return true;
}

// Hands a stopped trace to the next flush. The buffers and info words stay
// untouched until that flush frees the bracket, so nothing is copied here.
static void iree_hal_amd_gpu_profile_trace_complete(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_operation_t* operation,
    const iree_hal_amd_gpu_profile_completion_t* completion,
    bool has_dispatch_event, uint64_t dispatch_event_id) {
  iree_hal_profile_executable_trace_record_t record =
      iree_hal_profile_executable_trace_record_default();
  record.format = IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT;
  record.trace_id = ++profile->trace.last_trace_id;
  record.submission_id = operation->event.submission_id;
  record.stream_id = operation->event.stream_id;
  record.physical_device_ordinal = 0;
  record.queue_ordinal = operation->event.queue_ordinal;
  iree_hal_amd_gpu_profile_metadata_t metadata = {0};
  if (operation->commands) {
    // The bracket encloses the whole recording: no single dispatch or command
    // owns the trace, so neither correlation flag is set.
    record.command_buffer_id = operation->event.command_buffer_id;
    metadata.commands = operation->commands;
    iree_hal_amd_gpu_profile_commands_retain(operation->commands);
  } else {
    record.executable_id = operation->dispatch.event.executable_id;
    record.function_ordinal = operation->dispatch.event.function_ordinal;
    if (has_dispatch_event) {
      record.flags |= IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_DISPATCH_EVENT;
      record.dispatch_event_id = dispatch_event_id;
    }
    metadata.executable = operation->dispatch.executable;
    iree_hal_amd_gpu_profile_executable_retain(metadata.executable);
  }
  profile->trace.pending.record = record;
  profile->trace.pending.metadata = metadata;
  profile->trace.pending.dispatch_count = completion->count;
  iree_atomic_store(&profile->trace.state,
                    IREE_HAL_AMD_GPU_PROFILE_TRACE_COMPLETE,
                    iree_memory_order_release);
}

void iree_hal_amd_gpu_profile_operation_complete(
    iree_hal_amd_gpu_profile_t* profile,
    const iree_hal_amd_gpu_profile_operation_t* operation,
    const iree_hal_amd_gpu_profile_completion_t* completion) {
  if (!completion->count) {
    return;
  }
  // Identity of a direct dispatch's event, when the ring kept one.
  bool has_dispatch_event = false;
  uint64_t dispatch_event_id = 0;
  // Raw counter samples follow each timestamp pair of a counted operation.
  // The publisher brackets a whole operation, so only a whole completion can
  // be read; a split one is accounted as invalid.
  const iree_hal_amd_gpu_profile_counter_storage_t* counters =
      operation->counted ? profile->counters.storage : NULL;
  const bool counted =
      counters && completion->samples && completion->first == 0 &&
      completion->count == operation->completion.count &&
      completion->sample_stride ==
          sizeof(iree_hal_amd_gpu_profile_sample_t) +
              2 * iree_hal_amd_gpu_pm4_perf_counter_sample_byte_length(
                      &counters->config);
  uint32_t counter_sample_count = 0;
  iree_hal_amd_gpu_profile_stream_t* stream =
      profile->streams[operation->event.queue_ordinal];
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
    if (!operation->commands &&
        (profile->families & IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS)) {
      has_dispatch_event = true;
      dispatch_event_id = event_id;
    }
    if (completion->samples) {
      const iree_hal_amd_gpu_profile_sample_t* sample =
          (const void*)((const uint8_t*)completion->samples +
                        (iree_host_size_t)i * completion->sample_stride);
      event->start_tick = sample->begin;
      event->end_tick = sample->end;
      if (counted && counter_sample_count < operation->counter_grant &&
          iree_hal_amd_gpu_profile_counter_sample(profile, stream, operation,
                                                  event, sample)) {
        ++counter_sample_count;
      }
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
  if (operation->counted) {
    if (counted) {
      iree_atomic_fetch_add(&profile->counters.sampled_operations, 1,
                            iree_memory_order_relaxed);
      // Read after the operation retired: its start sequence has copied them.
      uint32_t state_words[IREE_HAL_AMD_GPU_PM4_PERF_COUNTER_STATE_WORD_COUNT];
      memcpy(state_words, counters->state_words, sizeof(state_words));
      if (!iree_hal_amd_gpu_pm4_perf_counter_state_is_default(&counters->config,
                                                              state_words)) {
        iree_atomic_fetch_add(&profile->counters.foreign_states, 1,
                              iree_memory_order_relaxed);
      }
    } else {
      iree_atomic_fetch_add(&profile->counters.invalid_samples,
                            operation->counter_grant,
                            iree_memory_order_relaxed);
    }
    // Dispatches the filter or a lost event left unsampled stay available.
    iree_atomic_fetch_add(&profile->counters.remaining,
                          operation->counter_grant - counter_sample_count,
                          iree_memory_order_relaxed);
    iree_hal_amd_gpu_profile_counter_window_release(profile);
  }
  // After the events: a flush that sees the completed trace also sees them.
  if (operation->traced) {
    iree_hal_amd_gpu_profile_trace_complete(
        profile, operation, completion, has_dispatch_event, dispatch_event_id);
  }
}
