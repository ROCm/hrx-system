// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include "iree/hal/utils/file_cache.h"

// Small batches keep their timeline metadata on the stack. Larger batches
// allocate count-sized storage without imposing another concurrency limit.
#define IREE_IO_PARAMETER_OP_BATCH_INLINE_CONCURRENCY 8

typedef struct iree_io_parameter_index_provider_t {
  // Reference-counted provider interface.
  iree_io_parameter_provider_t base;
  // Allocator owning the provider and transient batch metadata.
  iree_allocator_t host_allocator;
  // Caller-selected timeline limit per independently synchronized group.
  iree_host_size_t max_concurrent_operations;
  // Parameter scope stored in the trailing provider allocation.
  iree_string_view_t scope;
  // Retained index resolving parameter keys to backing storage.
  iree_io_parameter_index_t* index;
  // Retained cache of device-imported backing files.
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

  max_concurrent_operations = iree_max(1, max_concurrent_operations);

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

// Returns the singleton family affinity containing |queue|.
static iree_hal_queue_family_affinity_t iree_io_parameter_queue_family_affinity(
    iree_hal_queue_t* queue) {
  return iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(queue)));
}

// Resolves a parameter with |key| for use on the given |device|.
// Returns the entry containing the parameter metadata and a retained
// HAL file that stores it (must be released by the caller).
// If the parameter is synthetic and not backed by a file then the returned
// file will be NULL.
static iree_status_t iree_io_parameter_index_provider_resolve(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_string_view_t scope, iree_string_view_t key,
    iree_hal_memory_access_t access,
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
    // Keep the cached file on its native I/O path. Load operations select and
    // import the exact parameter range through their caller-provided pool;
    // attaching the entire backing allocation here would duplicate that route
    // and give the cache a different memory contract from the emitted buffer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_file_cache_lookup(
                provider->file_cache, device, /*storage_pool=*/NULL,
                iree_io_parameter_queue_family_affinity(queue), access,
                entry->storage.file.handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
                &file));
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
        allowed_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
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

  if (offset > entry->length || length > entry->length - offset) {
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
// Each span gets an independent timeline until the caller's concurrency limit
// is reached. Bounded batches then reuse the timeline with the fewest assigned
// bytes as a proxy for outstanding work. An uncapped batch never introduces
// dependencies between nonempty spans; load allocation and its payload remain
// ordered. Zero-byte operations can share a timeline.
// A gather/scatter retains its final transfer until enumeration completes; if
// it is the only transfer, it signals the caller directly without a timeline.
//
// File I/O and transfer operations such as splats share the same timelines and
// are submitted directly to the caller-selected queue.
//
// NOTE: we expect count == 0 to have been handled by callers to avoid the
// overhead of the batch setup and submission but it's valid to have a zero
// count.
typedef struct iree_io_parameter_op_batch_t {
  // Parameter provider sourcing the parameter metadata.
  iree_io_parameter_index_provider_t* provider;  // unretained
  // Device hosting the batch operation.
  iree_hal_device_t* device;  // unretained
  // Exact queue hosting the batch operation.
  iree_hal_queue_t* queue;  // unretained

  // Semaphores that must be waited on prior to any operations begin.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores that must be signaled after all operations complete.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Number of timelines provisioned for this batch, bounded by its span count.
  // Zero until metadata allocation succeeds or for an empty batch.
  iree_host_size_t concurrency;
  // Number of timelines which have had operations scheduled against them.
  // This allows us to filter out timelines that are idle upon completion
  // and avoid unneeded waits. 0 if no timeline was used (count = 0, etc).
  iree_host_size_t timeline_live_count;
  // Sum of byte read/writes outstanding per timeline as a proxy for the amount
  // of work performed in a particular timeline.
  uint64_t* timeline_bytes_outstanding;
  // Semaphore per timeline.
  iree_hal_semaphore_t** timeline_semaphores;
  // Current payload per timeline; when the semaphore reaches this value the
  // timeline has quiesced.
  uint64_t* timeline_values;
  // Optional owned allocation containing the three timeline arrays.
  void* timeline_storage;
  // Inline metadata for small batches; no allocation is needed in that case.
  struct {
    // Sum of bytes assigned to each inline timeline.
    uint64_t bytes[IREE_IO_PARAMETER_OP_BATCH_INLINE_CONCURRENCY];
    // Retained inline timeline semaphores, created on demand.
    iree_hal_semaphore_t*
        semaphores[IREE_IO_PARAMETER_OP_BATCH_INLINE_CONCURRENCY];
    // Last submission value on each inline timeline.
    uint64_t values[IREE_IO_PARAMETER_OP_BATCH_INLINE_CONCURRENCY];
  } inline_timelines;
} iree_io_parameter_op_batch_t;

// Begins a parameter operation batch against the given |provider|.
// Operations will be scheduled on the exact |queue| owned by |device|. All
// batch operations will wait until |wait_semaphore_list| has been reached and
// after all batch operations complete |signal_semaphore_list| will be signaled.
// Upon return callers are required to call iree_io_parameter_op_batch_end
// regardless of whether the caller encounters an error.
static iree_status_t iree_io_parameter_op_batch_begin(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t count,
    iree_io_parameter_op_batch_t* IREE_RESTRICT out_batch) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_batch);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_batch, 0, sizeof(*out_batch));

  out_batch->provider = provider;
  out_batch->device = device;
  out_batch->queue = queue;

  out_batch->wait_semaphore_list = wait_semaphore_list;
  out_batch->signal_semaphore_list = signal_semaphore_list;

  const iree_host_size_t concurrency =
      iree_min(provider->max_concurrent_operations, count);
  if (concurrency <= IREE_IO_PARAMETER_OP_BATCH_INLINE_CONCURRENCY) {
    out_batch->timeline_bytes_outstanding = out_batch->inline_timelines.bytes;
    out_batch->timeline_semaphores = out_batch->inline_timelines.semaphores;
    out_batch->timeline_values = out_batch->inline_timelines.values;
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(
                provider->host_allocator, concurrency,
                2 * sizeof(uint64_t) + sizeof(iree_hal_semaphore_t*),
                &out_batch->timeline_storage));
    // Put uint64_t arrays first so pointer and value arrays remain naturally
    // aligned on both 32-bit and 64-bit hosts.
    out_batch->timeline_bytes_outstanding = out_batch->timeline_storage;
    out_batch->timeline_values =
        out_batch->timeline_bytes_outstanding + concurrency;
    out_batch->timeline_semaphores =
        (iree_hal_semaphore_t**)(out_batch->timeline_values + concurrency);
  }
  out_batch->concurrency = concurrency;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Resolves the parameter entry from |enumerator| at index |i|.
// |access| indicates the required access permissions to the parameter storage.
// Returns the entry, the span indicating source/target ranges, and optionally
// a file (NULL if a splat). |out_file| is retained and must be released by the
// caller if set.
static iree_status_t iree_io_parameter_index_provider_resolve_entry(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_string_view_t scope,
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
      provider, device, queue, scope, key, access, &entry, &file));

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

typedef enum iree_io_parameter_transfer_type_e {
  IREE_IO_PARAMETER_TRANSFER_SPLAT = 0,
  IREE_IO_PARAMETER_TRANSFER_READ,
  IREE_IO_PARAMETER_TRANSFER_WRITE,
} iree_io_parameter_transfer_type_t;

typedef struct iree_io_parameter_transfer_t {
  // Operation selecting the payload below.
  iree_io_parameter_transfer_type_t type;
  // Buffer backing the pending transfer.
  iree_hal_buffer_t* buffer;
  // Byte offset within |buffer|.
  iree_device_size_t buffer_offset;
  // Byte length of the pending transfer; zero means no pending operation.
  iree_device_size_t length;
  union {
    // File payload shared by reads and writes.
    struct {
      // Retained file backing the pending transfer.
      iree_hal_file_t* handle;
      // Byte offset within |handle|.
      uint64_t offset;
    } file;
    // Splat payload borrowed from the provider's retained parameter index.
    struct {
      // Pattern bytes captured by the queue before submission returns.
      const void* pattern;
      // Number of bytes in |pattern|.
      iree_host_size_t pattern_length;
    } splat;
  };
} iree_io_parameter_transfer_t;

static void iree_io_parameter_transfer_deinitialize(
    iree_io_parameter_transfer_t* transfer) {
  if (transfer->type != IREE_IO_PARAMETER_TRANSFER_SPLAT) {
    iree_hal_file_release(transfer->file.handle);
  }
  memset(transfer, 0, sizeof(*transfer));
}

static bool iree_io_parameter_transfer_try_extend_file(
    iree_io_parameter_transfer_t* transfer, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  if (transfer->type == IREE_IO_PARAMETER_TRANSFER_SPLAT ||
      transfer->file.handle != file || transfer->buffer != buffer) {
    return false;
  }

  if (transfer->length > UINT64_MAX - transfer->file.offset) {
    return false;
  }
  const uint64_t file_end_offset = transfer->file.offset + transfer->length;
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

static void iree_io_parameter_transfer_assign_file(
    iree_io_parameter_transfer_t* transfer,
    iree_io_parameter_transfer_type_t type, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  *transfer = (iree_io_parameter_transfer_t){
      .type = type,
      .buffer = buffer,
      .buffer_offset = buffer_offset,
      .length = length,
      .file = {.handle = file, .offset = file_offset},
  };
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

  // An allocation contributes no bytes, keeping its following payload on
  // the same timeline. Otherwise use a fresh timeline before balancing used
  // ones. Uncapped batches therefore assign all spans in linear total work.
  iree_host_size_t timeline_index = batch->timeline_live_count;
  if (timeline_index &&
      !batch->timeline_bytes_outstanding[timeline_index - 1]) {
    --timeline_index;
  } else if (timeline_index == batch->concurrency) {
    timeline_index = 0;
    for (iree_host_size_t i = 1; i < batch->concurrency; ++i) {
      if (batch->timeline_bytes_outstanding[i] <
          batch->timeline_bytes_outstanding[timeline_index]) {
        timeline_index = i;
      }
    }
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)timeline_index);

  // Each active lane owns an independent failure and dependency domain.
  // Empty batches, synchronous imports, and direct single transfers need none.
  iree_hal_semaphore_t* timeline_semaphore =
      batch->timeline_semaphores[timeline_index];
  const bool is_first_timeline_use = timeline_semaphore == NULL;
  if (!timeline_semaphore) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_create(
                batch->device,
                iree_io_parameter_queue_family_affinity(batch->queue),
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
    iree_device_size_t min_alignment, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(out_buffer);
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

  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = allocation_size,
      .min_alignment = min_alignment,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_queue_alloca(batch->queue, step.wait_semaphore_list,
                                step.signal_semaphore_list, pool,
                                /*request_count=*/1, &request, out_buffer));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Submits one resolved transfer with the caller-selected dependency endpoints.
// The queue captures the payload before returning; ownership stays with caller.
static iree_status_t iree_io_parameter_transfer_submit(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_io_parameter_transfer_t* transfer) {
  switch (transfer->type) {
    case IREE_IO_PARAMETER_TRANSFER_SPLAT:
      return iree_hal_queue_fill(queue, waits, signals, transfer->buffer,
                                 transfer->buffer_offset, transfer->length,
                                 transfer->splat.pattern,
                                 transfer->splat.pattern_length,
                                 /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE);
    case IREE_IO_PARAMETER_TRANSFER_READ:
      return iree_hal_queue_read(queue, waits, signals, transfer->file.handle,
                                 transfer->file.offset, transfer->buffer,
                                 transfer->buffer_offset, transfer->length,
                                 /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE);
    case IREE_IO_PARAMETER_TRANSFER_WRITE:
      return iree_hal_queue_write(
          queue, waits, signals, transfer->buffer, transfer->buffer_offset,
          transfer->file.handle, transfer->file.offset, transfer->length,
          /*barriers=*/NULL, IREE_HAL_WRITE_FLAG_NONE);
    default:
      IREE_CHECK_UNREACHABLE("unknown parameter transfer type");
  }
}

static iree_status_t iree_io_parameter_op_batch_enqueue_transfer(
    iree_io_parameter_op_batch_t* batch,
    const iree_io_parameter_transfer_t* transfer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_op_step_t step;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_op_batch_advance_timeline(batch, transfer->length,
                                                      &step));
  iree_status_t status =
      iree_io_parameter_transfer_submit(batch->queue, step.wait_semaphore_list,
                                        step.signal_semaphore_list, transfer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_op_batch_flush_transfer(
    iree_io_parameter_op_batch_t* batch,
    iree_io_parameter_transfer_t* transfer) {
  if (!transfer->length) {
    return iree_ok_status();
  }
  iree_status_t status =
      iree_io_parameter_op_batch_enqueue_transfer(batch, transfer);
  iree_io_parameter_transfer_deinitialize(transfer);
  return status;
}

// Flushes any outstanding work in the |batch| and signals the user timeline.
// Must only be called once at the end of the batch.
static iree_status_t iree_io_parameter_op_batch_flush(
    iree_io_parameter_op_batch_t* batch,
    iree_io_parameter_transfer_t* pending_transfer) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (batch->timeline_live_count && pending_transfer) {
    status = iree_io_parameter_op_batch_flush_transfer(batch, pending_transfer);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // A single transfer owns the group's complete readiness contract. Forward
  // the caller's endpoints directly instead of creating a private join lane.
  if (!batch->timeline_live_count && pending_transfer &&
      pending_transfer->length) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "direct transfer completion");
    status = iree_io_parameter_transfer_submit(
        batch->queue, batch->wait_semaphore_list, batch->signal_semaphore_list,
        pending_transfer);
  } else if (batch->timeline_live_count == 0) {
    // Empty batches and synchronous imports still preserve input dependencies.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "pass-through wait-signal");
    status = iree_hal_queue_transfer(
        batch->queue, batch->wait_semaphore_list, batch->signal_semaphore_list,
        /*operation_count=*/0, NULL, /*barriers=*/NULL);
  } else {
    // Join all concurrent timelines and continue the user-provided timeline.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "timeline set wait chain");
    // Note that we allocate timelines on-demand up to timeline_live_count so
    // we can just pass the [0, timeline_live_count) range here.
    iree_hal_semaphore_list_t join_semaphore_list = {
        .count = batch->timeline_live_count,
        .semaphores = batch->timeline_semaphores,
        .payload_values = batch->timeline_values,
    };
    status = iree_hal_queue_transfer(
        batch->queue, join_semaphore_list, batch->signal_semaphore_list,
        /*operation_count=*/0, NULL, /*barriers=*/NULL);
  }

  // Report the total number of bytes transferred by the batch.
  IREE_TRACE({
    uint64_t total_bytes = pending_transfer ? pending_transfer->length : 0;
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
    iree_io_parameter_op_batch_t* batch,
    iree_io_parameter_transfer_t* pending_transfer, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(batch);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  // If the batch recording succeeded we flush now to finish any pending
  // operations and signal the semaphores.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_op_batch_flush(batch, pending_transfer);
  }

  // If the batch recording failed (or our flush did) we need to propagate that
  // to the downstream user semaphores.
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(batch->signal_semaphore_list,
                                 iree_status_clone(status));
  }

  // Resources are safe to release even if there are pending device operations
  // as the device guarantees the resources remain live.
  if (pending_transfer) {
    iree_io_parameter_transfer_deinitialize(pending_transfer);
  }
  for (iree_host_size_t i = 0; i < batch->concurrency; ++i) {
    iree_hal_semaphore_release(batch->timeline_semaphores[i]);
  }
  iree_allocator_free(batch->provider->host_allocator, batch->timeline_storage);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_io_file_handle_buffer_release(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload) {
  (void)storage_class;
  (void)payload;
  iree_io_file_handle_release((iree_io_file_handle_t*)user_data);
}

static iree_status_t iree_io_parameter_index_provider_load(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_hal_pool_t* pool,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_device_size_t target_min_alignment,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  // Initialize the batch state.
  iree_io_parameter_op_batch_t batch;
  iree_status_t status = iree_io_parameter_op_batch_begin(
      provider, device, queue, wait_semaphore_list, signal_semaphore_list,
      count, &batch);

  // Process each entry by enqueuing the appropriate operation.
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z_entry,
                                "iree_io_parameter_index_provider_load_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* source_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue, source_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_READ, &source_entry, &span, &source_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, source_entry->key.data,
                                  source_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // Each emitted buffer contains its requested span at |buffer_offset|.
    // Validate the full output geometry before trying either import or
    // allocation so both paths expose the same logical extent.
    iree_device_size_t target_allocation_size = 0;
    if (iree_status_is_ok(status) &&
        IREE_UNLIKELY(!iree_device_size_checked_add(
            span.buffer_offset, span.length, &target_allocation_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "loaded parameter buffer range exceeds the device address space");
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
    if (iree_status_is_ok(status) && target_min_alignment <= 1 &&
        span.buffer_offset == 0 &&
        source_entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE &&
        iree_io_file_handle_type(source_entry->storage.file.handle) ==
            IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
      const iree_hal_external_storage_class_t storage_class =
          iree_hal_make_external_storage_class(
              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
      iree_hal_pool_import_requirements_t import_requirements;
      iree_byte_span_t host_allocation =
          iree_io_file_handle_primitive(source_entry->storage.file.handle)
              .value.host_allocation;
      if (iree_hal_pool_query_import_requirements(pool, storage_class,
                                                  &import_requirements) &&
          host_allocation.data_length >= import_requirements.min_byte_length &&
          (!import_requirements.max_byte_length ||
           host_allocation.data_length <=
               import_requirements.max_byte_length) &&
          iree_host_ptr_has_alignment(
              host_allocation.data,
              (iree_host_size_t)import_requirements.base_alignment) &&
          host_allocation.data_length %
                  import_requirements.byte_length_granularity ==
              0) {
        iree_hal_external_storage_t external_buffer = {
            .storage_class = storage_class,
            .payload.host_pointer = host_allocation.data,
            .byte_length = host_allocation.data_length,
            .host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                                IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
            .maintenance_range =
                iree_hal_make_buffer_range(0, host_allocation.data_length),
            .release = iree_io_file_handle_buffer_release,
            .release_user_data = source_entry->storage.file.handle,
        };
        iree_io_file_handle_retain(source_entry->storage.file.handle);
        status = iree_hal_pool_import_buffer(
            pool, &external_buffer,
            iree_hal_make_buffer_range(
                source_entry->storage.file.offset + span.parameter_offset,
                target_allocation_size),
            &target_buffer);
        if (iree_status_is_ok(status)) {
          IREE_TRACE_ZONE_APPEND_TEXT(z_entry, "import succeeded");
        } else {
          iree_hal_external_storage_release(&external_buffer);
        }
      }
    }

    // When no admitted zero-copy route applies, allocate and populate through
    // the queue. A selected import route that fails is propagated: hiding a
    // malformed handle or native attachment failure behind a copy would make
    // the pool contract nondeterministic.
    if (iree_status_is_ok(status) && !target_buffer) {
      // Enqueue an allocation of the target buffer on a timeline.
      // The next operation we enqueue will go on the same timeline.
      status = iree_io_parameter_op_batch_enqueue_alloca(
          &batch, pool, target_min_alignment, target_allocation_size,
          &target_buffer);

      // Enqueue the operation on the same timeline as the allocation.
      if (iree_status_is_ok(status)) {
        switch (source_entry->type) {
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
            IREE_ASSERT(!source_file);
            const iree_io_parameter_transfer_t transfer = {
                .type = IREE_IO_PARAMETER_TRANSFER_SPLAT,
                .buffer = target_buffer,
                .buffer_offset = span.buffer_offset,
                .length = span.length,
                .splat =
                    {
                        .pattern = source_entry->storage.splat.pattern,
                        .pattern_length =
                            source_entry->storage.splat.pattern_length,
                    },
            };
            status =
                iree_io_parameter_op_batch_enqueue_transfer(&batch, &transfer);
            break;
          }
          case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
            IREE_ASSERT(source_file);
            iree_io_parameter_transfer_t transfer;
            iree_io_parameter_transfer_assign_file(
                &transfer, IREE_IO_PARAMETER_TRANSFER_READ, source_file,
                source_entry->storage.file.offset + span.parameter_offset,
                target_buffer, span.buffer_offset, span.length);
            status =
                iree_io_parameter_op_batch_enqueue_transfer(&batch, &transfer);
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
  }

  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, NULL, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_gather(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue,
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
  iree_status_t status = iree_io_parameter_op_batch_begin(
      provider, device, queue, wait_semaphore_list, signal_semaphore_list,
      count, &batch);

  // Process each entry by enqueuing the appropriate operation.
  iree_io_parameter_transfer_t pending_transfer = {0};
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_entry, "iree_io_parameter_index_provider_gather_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* source_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue, source_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_READ, &source_entry, &span, &source_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, source_entry->key.data,
                                  source_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // Empty spans still resolve and validate above, but have no transfer or
    // per-entry dependency. The batch flush preserves the caller's timeline.
    if (iree_status_is_ok(status) && span.length) {
      switch (source_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
          IREE_ASSERT(!source_file);
          status = iree_io_parameter_op_batch_flush_transfer(&batch,
                                                             &pending_transfer);
          if (iree_status_is_ok(status)) {
            pending_transfer = (iree_io_parameter_transfer_t){
                .type = IREE_IO_PARAMETER_TRANSFER_SPLAT,
                .buffer = target_buffer,
                .buffer_offset = span.buffer_offset,
                .length = span.length,
                .splat =
                    {
                        .pattern = source_entry->storage.splat.pattern,
                        .pattern_length =
                            source_entry->storage.splat.pattern_length,
                    },
            };
          }
          break;
        }
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
          IREE_ASSERT(source_file);
          const uint64_t source_file_offset =
              source_entry->storage.file.offset + span.parameter_offset;
          if (iree_io_parameter_transfer_try_extend_file(
                  &pending_transfer, source_file, source_file_offset,
                  target_buffer, span.buffer_offset, span.length)) {
            iree_hal_file_release(source_file);
            source_file = NULL;
          } else {
            status = iree_io_parameter_op_batch_flush_transfer(
                &batch, &pending_transfer);
            if (iree_status_is_ok(status)) {
              iree_io_parameter_transfer_assign_file(
                  &pending_transfer, IREE_IO_PARAMETER_TRANSFER_READ,
                  source_file, source_file_offset, target_buffer,
                  span.buffer_offset, span.length);
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
  }
  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, &pending_transfer, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_gather_batch(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_host_size_t gather_count,
    const iree_io_parameter_gather_t* gathers) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, gather_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < gather_count && iree_status_is_ok(status);
       ++i) {
    if (gathers[i].count && !gathers[i].target_buffer) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "gather batch group %" PRIhsz " requires a target buffer", i);
    }
  }

  // Each group owns its timelines: unrelated incoming waits must not become
  // prerequisites of another group through a shared scheduling lane.
  iree_host_size_t submitted_count = 0;
  for (; submitted_count < gather_count && iree_status_is_ok(status);
       ++submitted_count) {
    const iree_io_parameter_gather_t* group = &gathers[submitted_count];
    status = iree_io_parameter_index_provider_gather(
        provider, device, queue, group->wait_semaphore_list,
        group->signal_semaphore_list, group->source_scope, group->target_buffer,
        group->count, group->enumerator);
  }
  // The failing group has already failed its signals; accepted earlier
  // groups keep their independent completion and HAL-owned captures.
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = submitted_count; i < gather_count; ++i) {
      iree_hal_semaphore_list_fail(gathers[i].signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_scatter(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue,
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
  iree_status_t status = iree_io_parameter_op_batch_begin(
      provider, device, queue, wait_semaphore_list, signal_semaphore_list,
      count, &batch);

  // Process each entry by enqueuing the appropriate operation.
  iree_io_parameter_transfer_t pending_transfer = {0};
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_entry, "iree_io_parameter_index_provider_scatter_entry");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

    // Fetch the next parameter to process.
    const iree_io_parameter_index_entry_t* target_entry = NULL;
    iree_io_parameter_span_t span;
    iree_hal_file_t* target_file = NULL;  // retained, NULL if splat
    status = iree_io_parameter_index_provider_resolve_entry(
        provider, device, queue, target_scope, enumerator, i,
        IREE_HAL_MEMORY_ACCESS_WRITE, &target_entry, &span, &target_file);
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z_entry, target_entry->key.data,
                                  target_entry->key.size);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, span.length);
    }

    // Empty spans neither advance a timeline nor split a pending transfer.
    // The batch flush still joins the caller's incoming and outgoing signals.
    if (iree_status_is_ok(status) && span.length) {
      switch (target_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
          IREE_ASSERT(target_file);
          const uint64_t target_file_offset =
              target_entry->storage.file.offset + span.parameter_offset;
          if (iree_io_parameter_transfer_try_extend_file(
                  &pending_transfer, target_file, target_file_offset,
                  source_buffer, span.buffer_offset, span.length)) {
            iree_hal_file_release(target_file);
            target_file = NULL;
          } else {
            status = iree_io_parameter_op_batch_flush_transfer(
                &batch, &pending_transfer);
            if (iree_status_is_ok(status)) {
              iree_io_parameter_transfer_assign_file(
                  &pending_transfer, IREE_IO_PARAMETER_TRANSFER_WRITE,
                  target_file, target_file_offset, source_buffer,
                  span.buffer_offset, span.length);
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
  }
  // Flush any outstanding batch operations and end the batch.
  status = iree_io_parameter_op_batch_end(&batch, &pending_transfer, status);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_scatter_batch(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_host_size_t scatter_count,
    const iree_io_parameter_scatter_t* scatters) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, scatter_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < scatter_count && iree_status_is_ok(status);
       ++i) {
    if (scatters[i].count && !scatters[i].source_buffer) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "scatter batch group %" PRIhsz " requires a source buffer", i);
    }
  }

  // Each group owns its timelines: unrelated incoming waits must not become
  // prerequisites of another group through a shared scheduling lane.
  iree_host_size_t submitted_count = 0;
  for (; submitted_count < scatter_count && iree_status_is_ok(status);
       ++submitted_count) {
    const iree_io_parameter_scatter_t* group = &scatters[submitted_count];
    status = iree_io_parameter_index_provider_scatter(
        provider, device, queue, group->wait_semaphore_list,
        group->signal_semaphore_list, group->source_buffer, group->target_scope,
        group->count, group->enumerator);
  }
  // The failing group has already failed its signals; accepted earlier
  // groups keep their independent completion and HAL-owned captures.
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = submitted_count; i < scatter_count; ++i) {
      iree_hal_semaphore_list_fail(scatters[i].signal_semaphore_list,
                                   iree_status_clone(status));
    }
  }
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
