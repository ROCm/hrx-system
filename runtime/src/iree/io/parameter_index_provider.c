// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include "iree/hal/utils/file_cache.h"

// Limit concurrent operations to avoid blowing the stack. This is arbitrary and
// if we wanted to support more we could switch to using heap allocations or
// a growable stack scratchpad.
#define IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY 8

typedef struct iree_io_parameter_index_provider_t {
  iree_io_parameter_provider_t base;
  iree_allocator_t host_allocator;
  iree_host_size_t max_concurrent_operations;
  iree_string_view_t scope;
  iree_io_parameter_index_t* index;
  iree_hal_file_cache_t* file_cache;
} iree_io_parameter_index_provider_t;

static const iree_io_parameter_provider_vtable_t
    iree_io_parameter_index_provider_vtable;

static iree_io_parameter_index_provider_t*
iree_io_parameter_index_provider_cast(
    iree_io_parameter_provider_t* IREE_RESTRICT base_provider) {
  return (iree_io_parameter_index_provider_t*)base_provider;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_provider_create(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_host_size_t max_concurrent_operations, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, scope.data, scope.size);

  max_concurrent_operations =
      iree_max(1, iree_min(max_concurrent_operations,
                           IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY));

  iree_io_parameter_index_provider_t* provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_with_trailing(host_allocator, sizeof(*provider),
                                              scope.size, (void**)&provider));
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &iree_io_parameter_index_provider_vtable;
  provider->host_allocator = host_allocator;
  provider->max_concurrent_operations = max_concurrent_operations;

  provider->scope = iree_make_string_view(
      (const char*)provider + sizeof(*provider), scope.size);
  if (scope.size > 0) {
    memcpy((void*)provider->scope.data, scope.data, scope.size);
  }

  provider->index = index;
  iree_io_parameter_index_retain(index);

  iree_status_t status =
      iree_hal_file_cache_create(host_allocator, &provider->file_cache);

  if (iree_status_is_ok(status)) {
    *out_provider = (iree_io_parameter_provider_t*)provider;
  } else {
    iree_io_parameter_provider_release((iree_io_parameter_provider_t*)provider);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_io_parameter_index_provider_destroy(
    iree_io_parameter_provider_t* IREE_RESTRICT base_provider) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_cache_release(provider->file_cache);
  iree_io_parameter_index_release(provider->index);

  iree_allocator_free(host_allocator, provider);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_io_parameter_index_provider_notify(
    iree_io_parameter_provider_t* base_provider,
    iree_io_parameter_provider_signal_t signal) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (signal) {
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND:
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY:
      iree_hal_file_cache_trim(provider->file_cache);
      break;
    default:
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static bool iree_io_parameter_index_provider_query_support(
    iree_io_parameter_provider_t* base_provider, iree_string_view_t scope) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  return iree_string_view_equal(scope, provider->scope);
}

// Resolves a parameter with |key| for use on the given |device|.
// Returns the entry containing the parameter metadata and a retained
// HAL file that stores it (must be released by the caller).
// If the parameter is synthetic and not backed by a file then the returned
// file will be NULL.
static iree_status_t iree_io_parameter_index_provider_resolve(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_string_view_t scope,
    iree_string_view_t key, iree_hal_memory_access_t access,
    const iree_io_parameter_index_entry_t** out_entry,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_entry);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_entry = NULL;
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup the parameter in the index.
  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_lookup(provider->index, key, &entry));

  // Get (or import) the HAL file backing the entry.
  // NOTE: file is retained!
  iree_hal_file_t* file = NULL;
  if (entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_file_cache_lookup(provider->file_cache, device, queue_affinity,
                                   access, entry->storage.file.handle,
                                   IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &file));
  }

  *out_entry = entry;
  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Validates that the range specified by [offset, offset+length) is in bounds.
static iree_status_t iree_io_validate_parameter_range(
    iree_hal_memory_access_t required_access,
    const iree_io_parameter_index_entry_t* entry, uint64_t offset,
    uint64_t length) {
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
  switch (entry->type) {
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
      // Synthetic entries are read-only.
      allowed_access = IREE_HAL_MEMORY_ACCESS_READ;
      break;
    case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
      // Access to the entry depends on the access of the file backing it.
      if (iree_all_bits_set(
              iree_io_file_handle_access(entry->storage.file.handle),
              IREE_IO_FILE_ACCESS_READ)) {
        allowed_access |= IREE_HAL_MEMORY_ACCESS_READ;
      }
      if (iree_all_bits_set(
              iree_io_file_handle_access(entry->storage.file.handle),
              IREE_IO_FILE_ACCESS_WRITE)) {
        allowed_access |=
            IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD;
      }
      break;
    default:
      // Unknown entries are inaccessible.
      allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
      break;
  }
  if (!iree_all_bits_set(allowed_access, required_access)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_memory_access_str =
        iree_hal_memory_access_format(allowed_access, &temp0);
    iree_string_view_t required_memory_access_str =
        iree_hal_memory_access_format(required_access, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "parameter `%.*s` storage does not support the requested access "
        "type; parameter allows %.*s, operation requires %.*s",
        (int)entry->key.size, entry->key.data,
        (int)allowed_memory_access_str.size, allowed_memory_access_str.data,
        (int)required_memory_access_str.size, required_memory_access_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }

  if (offset + length > entry->length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "parameter `%.*s` range out of bounds (offset=%" PRIu64
        ", length=%" PRIu64 ", size=%" PRIu64 ")",
        (int)entry->key.size, entry->key.data, offset, length, entry->length);
  }

  return iree_ok_status();
}

// Stateful batch management of multiple parameter operations.
//
// The batch distributes operations over multiple timelines based on how much
// I/O is done on each. It's possible for pathologically bad latency if there
// are large and small operations interleaved as all large operations may end up
// serialized on one timeline and all small ones on the other.
// We distribute by tracking the total bytes outstanding on each timeline and
// always placing the next operation on the one with the fewest. This assumes
// that all I/O happens at roughly the same speed but if parameters come from
// different files on different devices that may not be the case. It's better
// than doing nothing, though.
//
// Though we mostly focus on file I/O we also support DMA operations such as
// splats (synthetic parameters) and copies (device->device transfers) by way
// of a command buffer we build and submit when needed. Today there is just a
// single command buffer we submit with zero barriers which should be equivalent
// to spreading it out over multiple timelines.
//
// NOTE: we expect count == 0 to have been handled by callers to avoid the
// overhead of the batch setup and submission but it's valid to have a zero
// count.
typedef struct iree_io_parameter_op_batch_t {
  // Parameter provider sourcing the parameter metadata.
  iree_io_parameter_index_provider_t* provider;  // unretained
  // Device hosting the batch operation.
  iree_hal_device_t* device;  // unretained
  // Queue affinity indicating where batch operations can run.
  iree_hal_queue_affinity_t queue_affinity;

  // Semaphores that must be waited on prior to any operations begin.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores that must be signaled after all operations complete.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Number of concurrent timelines available for processing the batch.
  // Expects 0 < concurrency <= IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY.
  // Not all timelines may be used and timeline_live_count should be checked to
  // see which are active.
  iree_host_size_t concurrency;
  // Number of timelines which have had operations scheduled against them.
  // This allows us to filter out timelines that are idle upon completion
  // and avoid unneeded waits. 0 if no timeline was used (count = 0, etc).
  iree_host_size_t timeline_live_count;
  // Sum of byte read/writes outstanding per timeline as a proxy for the amount
  // of work performed in a particular timeline.
  uint64_t
      timeline_bytes_outstanding[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  // Semaphore per timeline.
  iree_hal_semaphore_t*
      timeline_semaphores[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  // Current payload per timeline; when the semaphore reaches this value the
  // timeline has quiesced.
  uint64_t timeline_values[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];

  // On-demand allocated command buffer used for transfer operations (usually
  // splats, but could be copies to/from device buffers).
  iree_hal_command_buffer_t* transfer_command_buffer;
  // Sum of byte read/writes in the transfer command buffer as a proxy for
  // the amount of work performed within the command buffer. We expect transfer
  // operations to be cheaper than file I/O operations but are not trying to be
  // precise here.
  uint64_t transfer_bytes_outstanding;
} iree_io_parameter_op_batch_t;

// Begins a parameter operation batch against the given |provider|.
// Operations will be scheduled on |device| with |queue_affinity|. All batch
// operations will wait until |wait_semaphore_list| has been reached and after
// all batch operations complete |signal_semaphore_list| will be signaled.
// Upon return callers are required to call iree_io_parameter_op_batch_end
// regardless of whether the caller encounters an error.
static void iree_io_parameter_op_batch_begin(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_io_parameter_op_batch_t* IREE_RESTRICT out_batch) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_batch);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_batch, 0, sizeof(*out_batch));

  out_batch->provider = provider;
  out_batch->device = device;
  out_batch->queue_affinity = queue_affinity;

  out_batch->wait_semaphore_list = wait_semaphore_list;
  out_batch->signal_semaphore_list = signal_semaphore_list;

  // We could limit the concurrency from the max based on the batch size but
  // since the compiler batches everything and most models go over the default
  // max concurrency this is fine for now.
  out_batch->concurrency =
      iree_max(1, iree_min(provider->max_concurrent_operations,
                           IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY));

  IREE_TRACE_ZONE_END(z0);
}

// Resolves the parameter entry from |enumerator| at index |i|.
// |access| indicates the required access permissions to the parameter storage.
// Returns the entry, the span indicating source/target ranges, and optionally
// a file (NULL if a splat). |out_file| is retained and must be released by the
// caller if set.
static iree_status_t iree_io_parameter_index_provider_resolve_entry(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_string_view_t scope,
    iree_io_parameter_enumerator_t enumerator, iree_host_size_t i,
    iree_hal_memory_access_t access,
    const iree_io_parameter_index_entry_t** IREE_RESTRICT out_entry,
    iree_io_parameter_span_t* IREE_RESTRICT out_span,
    iree_hal_file_t** IREE_RESTRICT out_file) {
  IREE_ASSERT_ARGUMENT(out_entry);
  IREE_ASSERT_ARGUMENT(out_span);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_entry = NULL;
  memset(out_span, 0, sizeof(*out_span));
  *out_file = NULL;

  // Fetch the next parameter to copy and its buffer range.
  iree_string_view_t key = iree_string_view_empty();
  iree_io_parameter_span_t span = {0};
  IREE_RETURN_IF_ERROR(enumerator.fn(enumerator.user_data, i, &key, &span));

  // Lookup the parameter metadata and get its backing file.
  const iree_io_parameter_index_entry_t* entry = NULL;
  iree_hal_file_t* file = NULL;  // retained, NULL if splat
  IREE_RETURN_IF_ERROR(iree_io_parameter_index_provider_resolve(
      provider, device, queue_affinity, scope, key, access, &entry, &file));

  // Validate the parameter range is in-bounds.
  iree_status_t status = iree_io_validate_parameter_range(
      access, entry, span.parameter_offset, span.length);

  if (iree_status_is_ok(status)) {
    *out_entry = entry;
    *out_span = span;
    *out_file = file;
  } else {
    iree_hal_file_release(file);
  }
  return status;
}

typedef struct iree_io_parameter_file_transfer_t {
  // Retained file backing the pending transfer.
  iree_hal_file_t* file;
  // Byte offset within |file|.
  uint64_t file_offset;
  // Buffer backing the pending transfer.
  iree_hal_buffer_t* buffer;
  // Byte offset within |buffer|.
  iree_device_size_t buffer_offset;
  // Byte length of the pending transfer.
  iree_device_size_t length;
} iree_io_parameter_file_transfer_t;

static void iree_io_parameter_file_transfer_deinitialize(
    iree_io_parameter_file_transfer_t* transfer) {
  iree_hal_file_release(transfer->file);
  memset(transfer, 0, sizeof(*transfer));
}

static bool iree_io_parameter_file_transfer_try_extend(
    iree_io_parameter_file_transfer_t* transfer, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  if (!transfer->file || transfer->file != file || transfer->buffer != buffer) {
    return false;
  }

  if (transfer->length > UINT64_MAX - transfer->file_offset) {
    return false;
  }
  const uint64_t file_end_offset = transfer->file_offset + transfer->length;
  iree_device_size_t buffer_end_offset = 0;
  if (!iree_device_size_checked_add(transfer->buffer_offset, transfer->length,
                                    &buffer_end_offset)) {
    return false;
  }
  if (file_offset != file_end_offset || buffer_offset != buffer_end_offset) {
    return false;
  }

  iree_device_size_t extended_length = 0;
  if (!iree_device_size_checked_add(transfer->length, length,
                                    &extended_length)) {
    return false;
  }
  transfer->length = extended_length;
  return true;
}

static void iree_io_parameter_file_transfer_assign(
    iree_io_parameter_file_transfer_t* transfer, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  transfer->file = file;
  transfer->file_offset = file_offset;
  transfer->buffer = buffer;
  transfer->buffer_offset = buffer_offset;
  transfer->length = length;
}

typedef struct {
  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;
  uint64_t scratch_values[2];  // wait/signal payload values
} iree_io_parameter_op_step_t;

// Selects a timeline with the fewest bytes outstanding and accounts for the new
// |op_byte_length| bytes on that timeline. Returns semaphore lists the caller
// must wait on before performing their operation and signal after their
// operation completes.
static iree_status_t iree_io_parameter_op_batch_advance_timeline(
    iree_io_parameter_op_batch_t* batch, uint64_t op_byte_length,
    iree_io_parameter_op_step_t* IREE_RESTRICT out_step) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(out_step);
  memset(out_step, 0, sizeof(*out_step));
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, op_byte_length);

  // Find the timeline with the fewest outstanding bytes.
  // Linear scan as the number of timelines is expected to be small.
  uint64_t smallest_value = batch->timeline_bytes_outstanding[0];
  iree_host_size_t smallest_index = 0;
  for (iree_host_size_t i = 1; i < batch->concurrency; ++i) {
    if (batch->timeline_bytes_outstanding[i] < smallest_value) {
      smallest_value = batch->timeline_bytes_outstanding[i];
      smallest_index = i;
    }
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)smallest_index);
  const iree_host_size_t timeline_index = smallest_index;

  // Acquire the timeline semaphore used for this operation.
  // We create the semaphores on-demand so that in cases where we don't perform
  // any operations (loads that perform synchronous imports) or only a small
  // amount (one transfer command buffer or just a handful of operations) we
  // don't create so much garbage. The intent is that HAL devices pool
  // semaphores but not all do - if we made that an expected requirement we
  // could simplify this.
  iree_hal_semaphore_t* timeline_semaphore =
      batch->timeline_semaphores[timeline_index];
  const bool is_first_timeline_use = timeline_semaphore == NULL;
  if (!timeline_semaphore) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_create(
                batch->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                batch->timeline_values[timeline_index],
                IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                &batch->timeline_semaphores[timeline_index]));
    timeline_semaphore = batch->timeline_semaphores[timeline_index];
  }
  const uint64_t previous_timeline_value =
      batch->timeline_values[timeline_index];
  const uint64_t next_timeline_value = ++batch->timeline_values[timeline_index];

  // Account for the bytes processed by the operation, which is a good enough
  // metric for distribution (assuming all operations take about the same amount
  // of memory or I/O bandwidth).
  batch->timeline_bytes_outstanding[timeline_index] += op_byte_length;

  // Select the wait semaphore list; the first wave of operations all wait on
  // the original wait semaphore list provided by the initiator.
  if (is_first_timeline_use) {
    // First use of this timeline; wait on incoming list and begin the timeline.
    IREE_ASSERT_EQ(timeline_index, batch->timeline_live_count);
    ++batch->timeline_live_count;
    out_step->wait_semaphore_list = batch->wait_semaphore_list;
  } else {
    // Continuation of the selected timeline.
    out_step->scratch_values[0] = previous_timeline_value;
    out_step->wait_semaphore_list.count = 1;
    out_step->wait_semaphore_list.semaphores =
        &batch->timeline_semaphores[timeline_index];
    out_step->wait_semaphore_list.payload_values = &out_step->scratch_values[0];
  }

  // Signal the continuation of the timeline.
  out_step->scratch_values[1] = next_timeline_value;
  out_step->signal_semaphore_list.count = 1;
  out_step->signal_semaphore_list.semaphores =
      &batch->timeline_semaphores[timeline_index];
  out_step->signal_semaphore_list.payload_values = &out_step->scratch_values[1];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Enqueues a queue-ordered allocation.
// A timeline is selected based on utilization and the following operation is
// guaranteed to select the same timeline to ensure the allocation and
// operation are serialized and the wait has a higher chance of being elided.
static iree_status_t iree_io_parameter_op_batch_enqueue_alloca(
    iree_io_parameter_op_batch_t* batch, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // By passing 0 for the operation size we ensure that subsequent operations
  // select the same timeline because the timeline with the fewest outstanding
  // bytes will be returned after this. This intuitively seems good as we keep
  // the allocation and operation using it serialized within a single timeline
  // and allow devices that can elide back-to-back barriers to do so, but we may
  // find that we also want to distribute allocations.
  iree_io_parameter_op_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_io_parameter_op_batch_advance_timeline(
                                        batch, /*op_byte_length=*/0, &step));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_alloca(
              batch->device, batch->queue_affinity, step.wait_semaphore_list,
              step.signal_semaphore_list, pool, params, allocation_size,
              IREE_HAL_ALLOCA_FLAG_NONE, out_buffer));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Enqueues a splat operation in the batch into the |buffer| range.
// Splats get routed to a transfer command buffer that we'll submit at the end
// of the batch. This avoids the need for us to check all of the operations
// ahead of time at the cost of potentially acquiring more semaphores than we
// need in cases where everything is a splat. Splats are pretty much only useful
// for testing/development, though, so it's ok to not be super efficient here.
static iree_status_t iree_io_parameter_op_batch_enqueue_splat(
    iree_io_parameter_op_batch_t* batch, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create the transfer command buffer on first use.
  if (!batch->transfer_command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_create(
                batch->device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                IREE_HAL_COMMAND_CATEGORY_TRANSFER, batch->queue_affinity, 0,
                &batch->transfer_command_buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_begin(batch->transfer_command_buffer));
  }

  // Add the splat fill to the command buffer.
  // Parameter ranges cannot overlap so there's no barrier required.
  batch->transfer_bytes_outstanding += length;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_fill_buffer(
              batch->transfer_command_buffer,
              iree_hal_make_buffer_ref(buffer, buffer_offset, length), pattern,
              pattern_length, IREE_HAL_FILL_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Enqueues a file read operation in the batch.
static iree_status_t iree_io_parameter_op_batch_enqueue_file_read(
    iree_io_parameter_op_batch_t* batch, iree_hal_file_t* source_file,
    uint64_t source_file_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_buffer_offset, iree_device_size_t length,
    iree_hal_read_flags_t flags) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_op_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_op_batch_advance_timeline(batch, length, &step));

  iree_status_t status = iree_hal_device_queue_read(
      batch->device, batch->queue_affinity, step.wait_semaphore_list,
      step.signal_semaphore_list, source_file, source_file_offset,
      target_buffer, target_buffer_offset, length, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_op_batch_flush_file_read(
    iree_io_parameter_op_batch_t* batch,
    iree_io_parameter_file_transfer_t* transfer) {
  if (!transfer->file) return iree_ok_status();
  iree_status_t status = iree_io_parameter_op_batch_enqueue_file_read(
      batch, transfer->file, transfer->file_offset, transfer->buffer,
      transfer->buffer_offset, transfer->length, 0);
  iree_io_parameter_file_transfer_deinitialize(transfer);
  return status;
}

// Enqueues a file write operation in the batch.
static iree_status_t iree_io_parameter_op_batch_enqueue_file_write(
    iree_io_parameter_op_batch_t* batch, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_buffer_offset, iree_hal_file_t* target_file,
    uint64_t target_file_offset, iree_device_size_t length,
    iree_hal_write_flags_t flags) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_op_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_op_batch_advance_timeline(batch, length, &step));

  iree_status_t status = iree_hal_device_queue_write(
      batch->device, batch->queue_affinity, step.wait_semaphore_list,
      step.signal_semaphore_list, source_buffer, source_buffer_offset,
      target_file, target_file_offset, length, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_op_batch_flush_file_write(
    iree_io_parameter_op_batch_t* batch,
    iree_io_parameter_file_transfer_t* transfer) {
  if (!transfer->file) return iree_ok_status();
  iree_status_t status = iree_io_parameter_op_batch_enqueue_file_write(
      batch, transfer->buffer, transfer->buffer_offset, transfer->file,
      transfer->file_offset, transfer->length, 0);
  iree_io_parameter_file_transfer_deinitialize(transfer);
  return status;
}

// Flushes any outstanding work in the |batch| and signals the user timeline.
// Must only be called once at the end of the batch.
static iree_status_t iree_io_parameter_op_batch_flush(
    iree_io_parameter_op_batch_t* batch) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If any transfers were performed we'll need to submit the command buffer we
  // built during recording. Order doesn't matter so we can issue it alongside
  // all of the other work by just appending it to an arbitrary timeline. We try
  // to still balance things by selecting a timeline with the fewest operation
  // bytes outstanding even if the cost of a byte differs between file I/O and
  // pure DMA operations.
  iree_status_t status = iree_ok_status();
  if (batch->transfer_command_buffer) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_transfer,
                                "iree_io_parameter_op_batch_flush_transfer");
    status = iree_hal_command_buffer_end(batch->transfer_command_buffer);
    iree_io_parameter_op_step_t step;
    if (iree_status_is_ok(status)) {
      status = iree_io_parameter_op_batch_advance_timeline(
          batch, batch->transfer_bytes_outstanding, &step);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_execute(
          batch->device, batch->queue_affinity, step.wait_semaphore_list,
          step.signal_semaphore_list, batch->transfer_command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
    }
    IREE_TRACE_ZONE_END(z_transfer);
  }

  // Join all concurrent timelines and continue the user-provided timeline.
  if (iree_status_is_ok(status)) {
    // If no queue operations were performed (all load imports, 0 entries, etc)
    // we need to issue a barrier to link the wait->signal semaphore lists.
    if (batch->timeline_live_count == 0) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "pass-through wait-signal");
      status = iree_hal_device_queue_barrier(
          batch->device, batch->queue_affinity, batch->wait_semaphore_list,
          batch->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
    } else {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "timeline set wait chain");
      // Note that we allocate timelines on-demand up to timeline_live_count so
      // we can just pass the [0, timeline_live_count) range here.
      iree_hal_semaphore_list_t join_semaphore_list = {
          .count = batch->timeline_live_count,
          .semaphores = batch->timeline_semaphores,
          .payload_values = batch->timeline_values,
      };
      status = iree_hal_device_queue_barrier(
          batch->device, batch->queue_affinity, join_semaphore_list,
          batch->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
    }
  }

  // Report the total number of bytes transferred by the batch.
  IREE_TRACE({
    uint64_t total_bytes = batch->transfer_bytes_outstanding;
    for (iree_host_size_t i = 0; i < batch->concurrency; ++i) {
      total_bytes += batch->timeline_bytes_outstanding[i];
    }
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, total_bytes);
  });

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Ends a parameter operation batch.
// The provided |status| must be set to any failure that may have occurred
// between when the batch began and this method was called. The status will be
// passed on to the initiating caller by way of the signal semaphores being
// immediately failed. Returns the status provided to allow for propagating
// failures.
static iree_status_t iree_io_parameter_op_batch_end(
    iree_io_parameter_op_batch_t* batch, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  // If the batch recording succeeded we flush now to finish any pending
  // operations and signal the semaphores.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_op_batch_flush(batch);
  }

  // If the batch recording failed (or our flush did) we need to propagate that
  // to the downstream user semaphores.
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(batch->signal_semaphore_list,
                                 iree_status_clone(status));
  }

  // Resources are safe to release even if there are pending device operations
  // as the device guarantees the resources remain live.
  for (iree_host_size_t i = 0; i < batch->concurrency; ++i) {
    iree_hal_semaphore_release(batch->timeline_semaphores[i]);
  }
  iree_hal_command_buffer_release(batch->transfer_command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_io_parameter_transfer_batch_group_t {
  // Number of operations scheduled for this group.
  iree_host_size_t operation_count;
  // Semaphores that must be reached before this group can start.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled after this group completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Readiness semaphore for groups with multi-semaphore incoming waits.
  iree_hal_semaphore_t* start_semaphore;
  // Payload value signaled on |start_semaphore|.
  uint64_t start_value;
  // Timelines touched by this group.
  uint8_t timeline_mask;
  // Latest payload value used by this group on each timeline.
  uint64_t timeline_values[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
} iree_io_parameter_transfer_batch_group_t;

typedef struct iree_io_parameter_transfer_batch_t {
  // Parameter provider sourcing the parameter metadata.
  iree_io_parameter_index_provider_t* provider;
  // Device hosting the batch operation.
  iree_hal_device_t* device;
  // Queue affinity indicating where batch operations can run.
  iree_hal_queue_affinity_t queue_affinity;
  // Number of independently synchronized transfer groups.
  iree_host_size_t group_count;
  // Per-group readiness and completion tracking.
  iree_io_parameter_transfer_batch_group_t* groups;
  // Number of groups that already have completion barriers submitted.
  iree_host_size_t completed_group_count;
  // Number of concurrent timelines available for processing the batch.
  iree_host_size_t concurrency;
  // Sum of byte read/writes outstanding per timeline as a proxy for work.
  uint64_t
      timeline_bytes_outstanding[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  // Semaphore per timeline.
  iree_hal_semaphore_t*
      timeline_semaphores[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  // Current payload per timeline.
  uint64_t timeline_values[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
} iree_io_parameter_transfer_batch_t;

static iree_status_t iree_io_parameter_transfer_batch_begin(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t group_count,
    iree_io_parameter_transfer_batch_t* IREE_RESTRICT out_batch) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_batch);
  memset(out_batch, 0, sizeof(*out_batch));
  out_batch->provider = provider;
  out_batch->device = device;
  out_batch->queue_affinity = queue_affinity;
  out_batch->group_count = group_count;
  out_batch->concurrency =
      iree_max(1, iree_min(provider->max_concurrent_operations,
                           IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      provider->host_allocator, group_count, sizeof(out_batch->groups[0]),
      (void**)&out_batch->groups));
  memset(out_batch->groups, 0, group_count * sizeof(out_batch->groups[0]));
  return iree_ok_status();
}

static void iree_io_parameter_transfer_batch_initialize_group(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_host_size_t operation_count,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(batch);
  iree_io_parameter_transfer_batch_group_t* group = &batch->groups[group_index];
  group->operation_count = operation_count;
  group->wait_semaphore_list = wait_semaphore_list;
  group->signal_semaphore_list = signal_semaphore_list;
}

static iree_status_t iree_io_parameter_transfer_batch_begin_group(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index) {
  IREE_ASSERT_ARGUMENT(batch);
  iree_io_parameter_transfer_batch_group_t* group = &batch->groups[group_index];
  if (group->operation_count == 0 || group->wait_semaphore_list.count <= 1) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      batch->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &group->start_semaphore));
  group->start_value = 1;
  iree_hal_semaphore_list_t start_signal_list = {
      .count = 1,
      .semaphores = &group->start_semaphore,
      .payload_values = &group->start_value,
  };
  return iree_hal_device_queue_barrier(
      batch->device, batch->queue_affinity, group->wait_semaphore_list,
      start_signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

typedef struct iree_io_parameter_transfer_batch_step_t {
  // Wait list for the operation.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Signal list for the operation.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Scratch wait semaphore storage.
  iree_hal_semaphore_t* wait_semaphores[2];
  // Scratch wait payload value storage.
  uint64_t wait_values[2];
  // Scratch signal payload value storage.
  uint64_t signal_value;
} iree_io_parameter_transfer_batch_step_t;

static iree_status_t iree_io_parameter_transfer_batch_advance_timeline(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    uint64_t op_byte_length,
    iree_io_parameter_transfer_batch_step_t* IREE_RESTRICT out_step) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(out_step);
  memset(out_step, 0, sizeof(*out_step));
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, op_byte_length);

  uint64_t smallest_value = batch->timeline_bytes_outstanding[0];
  iree_host_size_t timeline_index = 0;
  for (iree_host_size_t i = 1; i < batch->concurrency; ++i) {
    if (batch->timeline_bytes_outstanding[i] < smallest_value) {
      smallest_value = batch->timeline_bytes_outstanding[i];
      timeline_index = i;
    }
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)timeline_index);

  iree_hal_semaphore_t* timeline_semaphore =
      batch->timeline_semaphores[timeline_index];
  if (!timeline_semaphore) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_create(
                batch->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                batch->timeline_values[timeline_index],
                IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                &batch->timeline_semaphores[timeline_index]));
    timeline_semaphore = batch->timeline_semaphores[timeline_index];
  }

  const uint64_t previous_timeline_value =
      batch->timeline_values[timeline_index];
  const uint64_t next_timeline_value = ++batch->timeline_values[timeline_index];
  batch->timeline_bytes_outstanding[timeline_index] += op_byte_length;

  iree_io_parameter_transfer_batch_group_t* group = &batch->groups[group_index];
  const uint8_t timeline_bit = (uint8_t)(1u << timeline_index);
  const bool group_touched_timeline =
      iree_all_bits_set(group->timeline_mask, timeline_bit);

  iree_host_size_t wait_count = 0;
  if (!group_touched_timeline) {
    if (group->start_semaphore) {
      out_step->wait_semaphores[wait_count] = group->start_semaphore;
      out_step->wait_values[wait_count++] = group->start_value;
    } else if (group->wait_semaphore_list.count == 1) {
      out_step->wait_semaphores[wait_count] =
          group->wait_semaphore_list.semaphores[0];
      out_step->wait_values[wait_count++] =
          group->wait_semaphore_list.payload_values[0];
    }
  }
  if (previous_timeline_value != 0) {
    out_step->wait_semaphores[wait_count] = timeline_semaphore;
    out_step->wait_values[wait_count++] = previous_timeline_value;
  }
  out_step->wait_semaphore_list.count = wait_count;
  out_step->wait_semaphore_list.semaphores = out_step->wait_semaphores;
  out_step->wait_semaphore_list.payload_values = out_step->wait_values;

  out_step->signal_value = next_timeline_value;
  out_step->signal_semaphore_list.count = 1;
  out_step->signal_semaphore_list.semaphores =
      &batch->timeline_semaphores[timeline_index];
  out_step->signal_semaphore_list.payload_values = &out_step->signal_value;

  group->timeline_mask |= timeline_bit;
  group->timeline_values[timeline_index] = next_timeline_value;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_io_parameter_transfer_batch_enqueue_splat(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_transfer_batch_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_transfer_batch_advance_timeline(batch, group_index,
                                                            length, &step));
  iree_status_t status = iree_hal_device_queue_fill(
      batch->device, batch->queue_affinity, step.wait_semaphore_list,
      step.signal_semaphore_list, buffer, buffer_offset, length, pattern,
      pattern_length, IREE_HAL_FILL_FLAG_NONE);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_transfer_batch_enqueue_file_read(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_hal_file_t* source_file, uint64_t source_file_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_buffer_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_transfer_batch_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_transfer_batch_advance_timeline(batch, group_index,
                                                            length, &step));
  iree_status_t status = iree_hal_device_queue_read(
      batch->device, batch->queue_affinity, step.wait_semaphore_list,
      step.signal_semaphore_list, source_file, source_file_offset,
      target_buffer, target_buffer_offset, length, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_transfer_batch_flush_file_read(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_io_parameter_file_transfer_t* transfer) {
  if (!transfer->file) return iree_ok_status();
  iree_status_t status = iree_io_parameter_transfer_batch_enqueue_file_read(
      batch, group_index, transfer->file, transfer->file_offset,
      transfer->buffer, transfer->buffer_offset, transfer->length, 0);
  iree_io_parameter_file_transfer_deinitialize(transfer);
  return status;
}

static iree_status_t iree_io_parameter_transfer_batch_enqueue_file_write(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_buffer_offset,
    iree_hal_file_t* target_file, uint64_t target_file_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_transfer_batch_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_transfer_batch_advance_timeline(batch, group_index,
                                                            length, &step));
  iree_status_t status = iree_hal_device_queue_write(
      batch->device, batch->queue_affinity, step.wait_semaphore_list,
      step.signal_semaphore_list, source_buffer, source_buffer_offset,
      target_file, target_file_offset, length, flags);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_transfer_batch_flush_file_write(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index,
    iree_io_parameter_file_transfer_t* transfer) {
  if (!transfer->file) return iree_ok_status();
  iree_status_t status = iree_io_parameter_transfer_batch_enqueue_file_write(
      batch, group_index, transfer->buffer, transfer->buffer_offset,
      transfer->file, transfer->file_offset, transfer->length, 0);
  iree_io_parameter_file_transfer_deinitialize(transfer);
  return status;
}

static iree_status_t iree_io_parameter_transfer_batch_complete_group(
    iree_io_parameter_transfer_batch_t* batch, iree_host_size_t group_index) {
  IREE_ASSERT_ARGUMENT(batch);
  const iree_io_parameter_transfer_batch_group_t* group =
      &batch->groups[group_index];
  if (group->timeline_mask == 0) {
    return iree_hal_device_queue_barrier(
        batch->device, batch->queue_affinity, group->wait_semaphore_list,
        group->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
  }

  iree_hal_semaphore_t*
      wait_semaphores[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  uint64_t wait_values[IREE_IO_PARAMETER_OP_BATCH_MAX_CONCURRENCY];
  iree_host_size_t wait_count = 0;
  for (iree_host_size_t i = 0; i < batch->concurrency; ++i) {
    const uint8_t timeline_bit = (uint8_t)(1u << i);
    if (!iree_all_bits_set(group->timeline_mask, timeline_bit)) continue;
    wait_semaphores[wait_count] = batch->timeline_semaphores[i];
    wait_values[wait_count++] = group->timeline_values[i];
  }
  iree_hal_semaphore_list_t wait_semaphore_list = {
      .count = wait_count,
      .semaphores = wait_semaphores,
      .payload_values = wait_values,
  };
  return iree_hal_device_queue_barrier(
      batch->device, batch->queue_affinity, wait_semaphore_list,
      group->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_io_parameter_transfer_batch_end(
    iree_io_parameter_transfer_batch_t* batch, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  while (iree_status_is_ok(status) &&
         batch->completed_group_count < batch->group_count) {
    status = iree_io_parameter_transfer_batch_complete_group(
        batch, batch->completed_group_count);
    if (iree_status_is_ok(status)) {
      ++batch->completed_group_count;
    }
  }

  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = batch->completed_group_count;
         i < batch->group_count; ++i) {
      iree_hal_semaphore_list_fail(batch->groups[i].signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }

  for (iree_host_size_t i = 0; i < batch->concurrency; ++i) {
    iree_hal_semaphore_release(batch->timeline_semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < batch->group_count; ++i) {
    iree_hal_semaphore_release(batch->groups[i].start_semaphore);
  }
  iree_allocator_free(batch->provider->host_allocator, batch->groups);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_io_file_handle_buffer_release(void* user_data,
                                               iree_hal_buffer_t* buffer) {
  iree_io_file_handle_release((iree_io_file_handle_t*)user_data);
}

static iree_status_t iree_io_parameter_index_provider_load(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  // Initialize the batch state.
  iree_io_parameter_op_batch_t batch;
  iree_io_parameter_op_batch_begin(provider, device, queue_affinity,
                                   wait_semaphore_list, signal_semaphore_list,
                                   &batch);

  // Process each entry by enqueuing the appropriate operation.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_entry,
                                "iree_io_parameter_index_provider_load_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* source_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue_affinity, source_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_READ, &source_entry, &span, &source_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, source_entry->key.data,
                                  source_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // TODO(benvanik): refactor iree_io_parameter_index_provider_resolve so that
    // it doesn't resolve the HAL file. Today if we hit the perfect case where
    // all loads are able to be imported directly then we don't end up using the
    // file we get back and instead have just wasted time/resources managing it.
    // On CPU it's relatively cheap (a few mallocs) but on GPU it may require
    // extremely expensive driver handling. Startup paths with parameters aren't
    // usually critical, though, so it's (probably) fine today as-is.

    // Try first to reuse the file backing store directly as a buffer. This only
    // works with specific file types and with specific target usage. The most
    // common cases for this are when using parameters as staging sources (so
    // host memory is ok) or on unified memory systems (where host memory is
    // device memory) and the file was originally mapped. We could extend the
    // conditions in which we use this with some better file handle helpers that
    // allow us to map files that we already have open via other mechanisms
    // (FILE, fd, etc).
    iree_hal_buffer_t* target_buffer = NULL;
    if (iree_status_is_ok(status) &&
        source_entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE &&
        iree_io_file_handle_type(source_entry->storage.file.handle) ==
            IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
      iree_byte_span_t host_allocation =
          iree_io_file_handle_primitive(source_entry->storage.file.handle)
              .value.host_allocation;
      iree_hal_external_buffer_t external_buffer = {
          .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
          .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
          .size = host_allocation.data_length,
          .handle =
              {
                  .host_allocation =
                      {
                          .ptr = host_allocation.data +
                                 source_entry->storage.file.offset,
                      },
              },
      };
      iree_hal_buffer_release_callback_t release_callback = {
          .fn = iree_io_file_handle_buffer_release,
          .user_data = source_entry->storage.file.handle,
      };
      iree_io_file_handle_retain(source_entry->storage.file.handle);
      iree_status_t import_status = iree_hal_allocator_import_buffer(
          iree_hal_device_allocator(device), target_params, &external_buffer,
          release_callback, &target_buffer);
      if (iree_status_is_ok(import_status)) {
        // Import succeeded - issue a barrier to preserve the async timeline.
        IREE_TRACE_ZONE_APPEND_TEXT(z_entry, "import succeeded");
      } else {
        // Failed to import - that's ok as we'll just do the full allocate +
        // read.
        IREE_TRACE_ZONE_APPEND_TEXT(z_entry, "import failed");
        import_status = iree_status_ignore(import_status);
        iree_io_file_handle_release(source_entry->storage.file.handle);
      }
    }

    // When the import path above fails we fall back to alloca + fill/read.
    if (iree_status_is_ok(status) && !target_buffer) {
      // Enqueue an allocation of the target buffer on a timeline.
      // The next operation we enqueue will go on the same timeline.
      status = iree_io_parameter_op_batch_enqueue_alloca(
          &batch, /*pool=*/NULL, target_params, span.length, &target_buffer);

      // Enqueue the operation on the same timeline as the allocation.
      if (iree_status_is_ok(status)) {
        switch (source_entry->type) {
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
            IREE_ASSERT(!source_file);
            status = iree_io_parameter_op_batch_enqueue_splat(
                &batch, target_buffer, span.buffer_offset, span.length,
                source_entry->storage.splat.pattern,
                source_entry->storage.splat.pattern_length);
            break;
          }
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
            IREE_ASSERT(source_file);
            status = iree_io_parameter_op_batch_enqueue_file_read(
                &batch, source_file,
                source_entry->storage.file.offset + span.parameter_offset,
                target_buffer, span.buffer_offset, span.length, 0);
            break;
          }
          default: {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "load not supported with parameters of type %d",
                (int)source_entry->type);
            break;
          }
        }
      }
    }

    iree_hal_file_release(source_file);

    // Emit the target buffer so the caller can handle it. The callee must
    // retain it if they want to keep it live. We're allowed to emit out of
    // order but are currently always 1:1 with enumeration (which may be useful
    // in the future if we decide to make enumeration non-indexing).
    if (iree_status_is_ok(status)) {
      status = emitter.fn(emitter.user_data, i, target_buffer);
    }
    iree_hal_buffer_release(target_buffer);

    IREE_TRACE_ZONE_END(z_entry);
    if (!iree_status_is_ok(status)) break;
  }

  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_gather(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  // Initialize the batch state.
  iree_io_parameter_op_batch_t batch;
  iree_io_parameter_op_batch_begin(provider, device, queue_affinity,
                                   wait_semaphore_list, signal_semaphore_list,
                                   &batch);

  // Process each entry by enqueuing the appropriate operation.
  iree_status_t status = iree_ok_status();
  iree_io_parameter_file_transfer_t pending_file_read = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_entry, "iree_io_parameter_index_provider_gather_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* source_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue_affinity, source_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_READ, &source_entry, &span, &source_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, source_entry->key.data,
                                  source_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // Enqueue the transfer/file operation.
    if (iree_status_is_ok(status)) {
      switch (source_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
          IREE_ASSERT(!source_file);
          status = iree_io_parameter_op_batch_flush_file_read(
              &batch, &pending_file_read);
          if (iree_status_is_ok(status)) {
            status = iree_io_parameter_op_batch_enqueue_splat(
                &batch, target_buffer, span.buffer_offset, span.length,
                source_entry->storage.splat.pattern,
                source_entry->storage.splat.pattern_length);
          }
          break;
        }
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
          IREE_ASSERT(source_file);
          const uint64_t source_file_offset =
              source_entry->storage.file.offset + span.parameter_offset;
          if (iree_io_parameter_file_transfer_try_extend(
                  &pending_file_read, source_file, source_file_offset,
                  target_buffer, span.buffer_offset, span.length)) {
            iree_hal_file_release(source_file);
            source_file = NULL;
          } else {
            status = iree_io_parameter_op_batch_flush_file_read(
                &batch, &pending_file_read);
            if (iree_status_is_ok(status)) {
              iree_io_parameter_file_transfer_assign(
                  &pending_file_read, source_file, source_file_offset,
                  target_buffer, span.buffer_offset, span.length);
              source_file = NULL;
            }
          }
          break;
        }
        default: {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "gather not supported with parameters of type %d",
              (int)source_entry->type);
          break;
        }
      }
    }

    iree_hal_file_release(source_file);

    IREE_TRACE_ZONE_END(z_entry);
    if (!iree_status_is_ok(status)) break;
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_parameter_op_batch_flush_file_read(&batch, &pending_file_read);
  } else {
    iree_io_parameter_file_transfer_deinitialize(&pending_file_read);
  }

  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_gather_batch(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t gather_count,
    const iree_io_parameter_gather_t* gathers) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, gather_count);

  iree_io_parameter_transfer_batch_t batch;
  iree_status_t status = iree_io_parameter_transfer_batch_begin(
      provider, device, queue_affinity, gather_count, &batch);
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < gather_count; ++i) {
      iree_hal_semaphore_list_fail(gathers[i].signal_semaphore_list,
                                   iree_status_clone(status));
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t group_index = 0; group_index < gather_count;
       ++group_index) {
    const iree_io_parameter_gather_t* gather = &gathers[group_index];
    iree_io_parameter_transfer_batch_initialize_group(
        &batch, group_index, gather->count, gather->wait_semaphore_list,
        gather->signal_semaphore_list);
  }

  for (iree_host_size_t group_index = 0;
       group_index < gather_count && iree_status_is_ok(status); ++group_index) {
    const iree_io_parameter_gather_t* gather = &gathers[group_index];
    if (gather->count != 0 && !gather->target_buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "gather batch group %" PRIhsz
                                " requires a target buffer",
                                group_index);
      break;
    }
    status = iree_io_parameter_transfer_batch_begin_group(&batch, group_index);
  }

  for (iree_host_size_t group_index = 0;
       group_index < gather_count && iree_status_is_ok(status); ++group_index) {
    const iree_io_parameter_gather_t* gather = &gathers[group_index];
    iree_io_parameter_file_transfer_t pending_file_read = {0};
    for (iree_host_size_t i = 0; i < gather->count; ++i) {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z_entry, "iree_io_parameter_index_provider_gather_batch_entry");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

      const iree_io_parameter_index_entry_t* source_entry = NULL;
      iree_io_parameter_span_t span;
      iree_hal_file_t* source_file = NULL;
      status = iree_io_parameter_index_provider_resolve_entry(
          provider, device, queue_affinity, gather->source_scope,
          gather->enumerator, i, IREE_HAL_MEMORY_ACCESS_READ, &source_entry,
          &span, &source_file);
      if (iree_status_is_ok(status)) {
        IREE_TRACE_ZONE_APPEND_TEXT(z_entry, source_entry->key.data,
                                    source_entry->key.size);
        IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
      }

      if (iree_status_is_ok(status)) {
        switch (source_entry->type) {
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
            IREE_ASSERT(!source_file);
            status = iree_io_parameter_transfer_batch_flush_file_read(
                &batch, group_index, &pending_file_read);
            if (iree_status_is_ok(status)) {
              status = iree_io_parameter_transfer_batch_enqueue_splat(
                  &batch, group_index, gather->target_buffer,
                  span.buffer_offset, span.length,
                  source_entry->storage.splat.pattern,
                  source_entry->storage.splat.pattern_length);
            }
            break;
          }
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
            IREE_ASSERT(source_file);
            const uint64_t source_file_offset =
                source_entry->storage.file.offset + span.parameter_offset;
            if (iree_io_parameter_file_transfer_try_extend(
                    &pending_file_read, source_file, source_file_offset,
                    gather->target_buffer, span.buffer_offset, span.length)) {
              iree_hal_file_release(source_file);
              source_file = NULL;
            } else {
              status = iree_io_parameter_transfer_batch_flush_file_read(
                  &batch, group_index, &pending_file_read);
              if (iree_status_is_ok(status)) {
                iree_io_parameter_file_transfer_assign(
                    &pending_file_read, source_file, source_file_offset,
                    gather->target_buffer, span.buffer_offset, span.length);
                source_file = NULL;
              }
            }
            break;
          }
          default: {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "gather not supported with parameters of type %d",
                (int)source_entry->type);
            break;
          }
        }
      }

      iree_hal_file_release(source_file);
      IREE_TRACE_ZONE_END(z_entry);
      if (!iree_status_is_ok(status)) break;
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_parameter_transfer_batch_flush_file_read(
          &batch, group_index, &pending_file_read);
    } else {
      iree_io_parameter_file_transfer_deinitialize(&pending_file_read);
    }
  }

  status = iree_io_parameter_transfer_batch_end(&batch, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_scatter(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  // Initialize the batch state.
  iree_io_parameter_op_batch_t batch;
  iree_io_parameter_op_batch_begin(provider, device, queue_affinity,
                                   wait_semaphore_list, signal_semaphore_list,
                                   &batch);

  // Process each entry by enqueuing the appropriate operation.
  iree_status_t status = iree_ok_status();
  iree_io_parameter_file_transfer_t pending_file_write = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_entry, "iree_io_parameter_index_provider_scatter_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* target_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* target_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue_affinity, target_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_WRITE, &target_entry, &span, &target_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, target_entry->key.data,
                                  target_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // Enqueue the transfer/file operation.
    if (iree_status_is_ok(status)) {
      switch (target_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
          IREE_ASSERT(target_file);
          const uint64_t target_file_offset =
              target_entry->storage.file.offset + span.parameter_offset;
          if (iree_io_parameter_file_transfer_try_extend(
                  &pending_file_write, target_file, target_file_offset,
                  source_buffer, span.buffer_offset, span.length)) {
            iree_hal_file_release(target_file);
            target_file = NULL;
          } else {
            status = iree_io_parameter_op_batch_flush_file_write(
                &batch, &pending_file_write);
            if (iree_status_is_ok(status)) {
              iree_io_parameter_file_transfer_assign(
                  &pending_file_write, target_file, target_file_offset,
                  source_buffer, span.buffer_offset, span.length);
              target_file = NULL;
            }
          }
          break;
        }
        default: {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "scatter not supported with parameters of type %d",
              (int)target_entry->type);
          break;
        }
      }
    }

    iree_hal_file_release(target_file);

    IREE_TRACE_ZONE_END(z_entry);
    if (!iree_status_is_ok(status)) break;
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_op_batch_flush_file_write(&batch,
                                                         &pending_file_write);
  } else {
    iree_io_parameter_file_transfer_deinitialize(&pending_file_write);
  }

  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_scatter_batch(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t scatter_count,
    const iree_io_parameter_scatter_t* scatters) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, scatter_count);

  iree_io_parameter_transfer_batch_t batch;
  iree_status_t status = iree_io_parameter_transfer_batch_begin(
      provider, device, queue_affinity, scatter_count, &batch);
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < scatter_count; ++i) {
      iree_hal_semaphore_list_fail(scatters[i].signal_semaphore_list,
                                   iree_status_clone(status));
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  for (iree_host_size_t group_index = 0; group_index < scatter_count;
       ++group_index) {
    const iree_io_parameter_scatter_t* scatter = &scatters[group_index];
    iree_io_parameter_transfer_batch_initialize_group(
        &batch, group_index, scatter->count, scatter->wait_semaphore_list,
        scatter->signal_semaphore_list);
  }

  for (iree_host_size_t group_index = 0;
       group_index < scatter_count && iree_status_is_ok(status);
       ++group_index) {
    const iree_io_parameter_scatter_t* scatter = &scatters[group_index];
    if (scatter->count != 0 && !scatter->source_buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "scatter batch group %" PRIhsz
                                " requires a source buffer",
                                group_index);
      break;
    }
    status = iree_io_parameter_transfer_batch_begin_group(&batch, group_index);
  }

  for (iree_host_size_t group_index = 0;
       group_index < scatter_count && iree_status_is_ok(status);
       ++group_index) {
    const iree_io_parameter_scatter_t* scatter = &scatters[group_index];
    iree_io_parameter_file_transfer_t pending_file_write = {0};
    for (iree_host_size_t i = 0; i < scatter->count; ++i) {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z_entry, "iree_io_parameter_index_provider_scatter_batch_entry");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

      const iree_io_parameter_index_entry_t* target_entry = NULL;
      iree_io_parameter_span_t span;
      iree_hal_file_t* target_file = NULL;
      status = iree_io_parameter_index_provider_resolve_entry(
          provider, device, queue_affinity, scatter->target_scope,
          scatter->enumerator, i, IREE_HAL_MEMORY_ACCESS_WRITE, &target_entry,
          &span, &target_file);
      if (iree_status_is_ok(status)) {
        IREE_TRACE_ZONE_APPEND_TEXT(z_entry, target_entry->key.data,
                                    target_entry->key.size);
        IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
      }

      if (iree_status_is_ok(status)) {
        switch (target_entry->type) {
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
            IREE_ASSERT(target_file);
            const uint64_t target_file_offset =
                target_entry->storage.file.offset + span.parameter_offset;
            if (iree_io_parameter_file_transfer_try_extend(
                    &pending_file_write, target_file, target_file_offset,
                    scatter->source_buffer, span.buffer_offset, span.length)) {
              iree_hal_file_release(target_file);
              target_file = NULL;
            } else {
              status = iree_io_parameter_transfer_batch_flush_file_write(
                  &batch, group_index, &pending_file_write);
              if (iree_status_is_ok(status)) {
                iree_io_parameter_file_transfer_assign(
                    &pending_file_write, target_file, target_file_offset,
                    scatter->source_buffer, span.buffer_offset, span.length);
                target_file = NULL;
              }
            }
            break;
          }
          default: {
            status = iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "scatter not supported with parameters of type %d",
                (int)target_entry->type);
            break;
          }
        }
      }

      iree_hal_file_release(target_file);
      IREE_TRACE_ZONE_END(z_entry);
      if (!iree_status_is_ok(status)) break;
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_parameter_transfer_batch_flush_file_write(
          &batch, group_index, &pending_file_write);
    } else {
      iree_io_parameter_file_transfer_deinitialize(&pending_file_write);
    }
  }

  status = iree_io_parameter_transfer_batch_end(&batch, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_io_parameter_provider_vtable_t
    iree_io_parameter_index_provider_vtable = {
        .destroy = iree_io_parameter_index_provider_destroy,
        .notify = iree_io_parameter_index_provider_notify,
        .query_support = iree_io_parameter_index_provider_query_support,
        .load = iree_io_parameter_index_provider_load,
        .gather = iree_io_parameter_index_provider_gather,
        .gather_batch = iree_io_parameter_index_provider_gather_batch,
        .scatter = iree_io_parameter_index_provider_scatter,
        .scatter_batch = iree_io_parameter_index_provider_scatter_batch,
};
