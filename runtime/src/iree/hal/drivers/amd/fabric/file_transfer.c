// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/file_transfer.h"

#include "iree/async/file.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/fabric/virtual_buffer.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/utils/memory_file.h"

struct iree_hal_amd_fabric_file_host_range_t {
  // Owned view borrowing the application's installed physical source.
  amdf_host_mapping_t* mapping;
  // Host bytes corresponding exactly to this portion of the file operation.
  iree_byte_span_t contents;
  // Source-relative position in the whole-memory native host view.
  uint64_t mapping_offset;
  // Transfer-relative first byte, independent of file and buffer offsets.
  uint64_t offset;
  // Exact pair requires host invalidation before accessing device-written data.
  bool invalidate;
  // Exact pair requires host publication after native file reads.
  bool flush;
};

static void iree_hal_amd_fabric_file_transfer_release_host(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  for (iree_host_size_t i = 0; i < transfer->host.count; ++i) {
    amdf_host_mapping_t* mapping = transfer->host.ranges[i].mapping;
    if (!mapping) {
      continue;
    }
    const amdf_status_t status =
        transfer->host.api->host_mapping_destroy(mapping);
    if (status != AMDF_STATUS_OK) {
      iree_hal_amd_report_failure(transfer->host.event_sink, IREE_SV("amd"),
                                  status, "host_mapping_destroy");
    }
  }
  iree_allocator_free(transfer->host_allocator, transfer->host.ranges);
  memset(&transfer->host, 0, sizeof(transfer->host));
}

// Native pairs are validated once at translation. A transfer can consume
// queue-side global/range effects and CPU-side cache operations, but cannot
// substitute queue operations for a recipe requiring program instructions.
static bool iree_hal_amd_fabric_file_transfer_queue_effect(
    const iree_hal_memory_transition_recipe_info_t* recipe, bool release,
    iree_hal_memory_effects_t* effects) {
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    return true;
  }
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      recipe->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE) {
    return false;
  }
  effects->bits |=
      recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE
          ? (release ? IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM
                     : IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM)
          : (release ? IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM
                     : IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM);
  return true;
}

static bool iree_hal_amd_fabric_file_transfer_host_effect(
    const iree_hal_memory_transition_recipe_info_t* recipe, bool* required) {
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    return true;
  }
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      (recipe->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT &&
       recipe->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API)) {
    return false;
  }
  *required = true;
  return true;
}

static iree_status_t iree_hal_amd_fabric_file_transfer_query_pair(
    const amdf_api_t* api, const amdf_memory_site_t* producer,
    const amdf_memory_site_t* consumer, iree_hal_memory_pair_info_t* out_pair,
    bool* available) {
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t status =
      api->memory_query_pair_info(producer, consumer, &pair);
  if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    *available = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "memory_query_pair_info"));
  return iree_hal_amd_fabric_memory_translate_pair(&pair, out_pair);
}

static iree_status_t iree_hal_amd_fabric_file_transfer_map_source(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    const amdf_memory_mapping_update_t* source, uint32_t native_family_ordinal,
    iree_hal_memory_access_t queue_access,
    iree_hal_amd_fabric_file_host_range_t* range,
    iree_hal_memory_effects_t* before, iree_hal_memory_effects_t* after,
    bool* available) {
  const amdf_api_t* api = transfer->host.api;
  amdf_memory_info_t info = {.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
                             .structure_size = sizeof(info)};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_query_info(source->memory, &info), "memory_query_info"));
  if (!(info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
    *available = false;
    return iree_ok_status();
  }
  // Map the native extent to preserve profile offset/length alignment; the
  // file only touches the exact source subrange. No CPU reservation alias or
  // HOST_MAPPING_SOURCE capability is needed, including for visible VRAM.
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = info.byte_length,
      .flags = (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
                    ? AMDF_MEMORY_MAP_FLAG_WRITE
                    : AMDF_MEMORY_MAP_FLAG_READ) |
               ((queue_access & IREE_HAL_MEMORY_ACCESS_WRITE)
                    ? AMDF_MEMORY_MAP_FLAG_READ
                    : 0),
  };
  const amdf_status_t status =
      api->memory_map(source->memory, &map_info, &range->mapping);
  if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    *available = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(status, "memory_map"));
  amdf_host_mapping_info_t host_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(host_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->host_mapping_query_info(range->mapping, &host_info),
      "host_mapping_query_info"));
  range->contents = iree_make_byte_span(
      (uint8_t*)host_info.pointer + source->source_byte_offset,
      (iree_host_size_t)source->byte_length);
  range->mapping_offset = source->source_byte_offset;
  const amdf_memory_site_t device = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(device),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device = {.memory = source->memory,
                       .access_ordinal = source->memory_access_ordinal,
                       .queue_family_ordinal = native_family_ordinal},
  };
  const amdf_memory_site_t host = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(host),
      .kind = AMDF_MEMORY_SITE_KIND_HOST,
      .value.host = {.mapping = range->mapping},
  };
  iree_hal_memory_pair_info_t pair;
  if (queue_access & IREE_HAL_MEMORY_ACCESS_WRITE) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_transfer_query_pair(
        api, &device, &host, &pair, available));
    if (!*available) {
      return iree_ok_status();
    }
    *available = iree_hal_amd_fabric_file_transfer_queue_effect(&pair.release,
                                                                true, before) &&
                 iree_hal_amd_fabric_file_transfer_host_effect(
                     &pair.acquire, &range->invalidate);
  }
  if (*available && transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_transfer_query_pair(
        api, &host, &device, &pair, available));
    if (!*available) {
      return iree_ok_status();
    }
    *available = iree_hal_amd_fabric_file_transfer_host_effect(&pair.release,
                                                               &range->flush) &&
                 iree_hal_amd_fabric_file_transfer_queue_effect(&pair.acquire,
                                                                false, after);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_virtual(
    iree_hal_amd_fabric_file_transfer_t* transfer, const amdf_api_t* api,
    const iree_hal_queue_family_t* family, uint32_t native_family_ordinal,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_memory_effects_t* out_before, iree_hal_memory_effects_t* out_after,
    bool* out_available) {
  *out_before = (iree_hal_memory_effects_t){0};
  *out_after = (iree_hal_memory_effects_t){0};
  *out_available = false;
  if (!transfer->async.file ||
      (transfer->async.file->flags & IREE_ASYNC_FILE_FLAG_DIRECT)) {
    return iree_ok_status();
  }
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(transfer->buffer);
  if (!view.root || view.root->source_type != IREE_HAL_MEMORY_BACKEND_AMDF) {
    return iree_ok_status();
  }
  transfer->host.api = api;
  transfer->host.event_sink = event_sink;
  const iree_hal_memory_access_t queue_access = iree_hal_buffer_usage_access(
      iree_hal_buffer_family_usage(transfer->buffer, family));
  bool available = true;
  iree_hal_memory_effects_t before = {0}, after = {0};
  iree_host_size_t capacity = 0;
  iree_status_t status = iree_ok_status();
  for (uint64_t offset = 0;
       offset < transfer->length && iree_status_is_ok(status);) {
    amdf_memory_mapping_update_t source;
    status = iree_hal_amd_fabric_virtual_buffer_query_source(
        view, family->memory.queue_scope_id, transfer->buffer_offset + offset,
        transfer->length - offset, &source);
    const amdf_memory_access_t required =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? AMDF_MEMORY_ACCESS_WRITE
            : AMDF_MEMORY_ACCESS_READ;
    if (iree_status_is_ok(status) && !(source.access & required)) {
      status =
          iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                           "file access exceeds mapped source permissions");
    }
    if (iree_status_is_ok(status) && available &&
        transfer->host.count == capacity) {
      status = iree_allocator_grow_array(
          transfer->host_allocator, iree_max(transfer->host.count + 1, 4),
          sizeof(*transfer->host.ranges), &capacity,
          (void**)&transfer->host.ranges);
    }
    if (iree_status_is_ok(status) && available) {
      iree_hal_amd_fabric_file_host_range_t* range =
          &transfer->host.ranges[transfer->host.count++];
      *range = (iree_hal_amd_fabric_file_host_range_t){.offset = offset};
      status = iree_hal_amd_fabric_file_transfer_map_source(
          transfer, &source, native_family_ordinal, queue_access, range,
          &before, &after, &available);
    }
    offset += source.byte_length;
  }
  if (iree_status_is_ok(status) && available) {
    *out_before = before;
    *out_after = after;
    *out_available = true;
  } else {
    iree_hal_amd_fabric_file_transfer_release_host(transfer);
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_initialize(
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_allocator_t host_allocator,
    iree_hal_amd_fabric_file_transfer_t* out_transfer) {
  memset(out_transfer, 0, sizeof(*out_transfer));
  iree_async_file_t* async_file = iree_hal_file_async_handle(file);
  iree_byte_span_t file_contents = iree_byte_span_empty();
  if (!async_file) {
    if (!iree_hal_memory_file_isa(file)) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "AMD queue file transfers require a memory file or an async file "
          "handle");
    }
    IREE_RETURN_IF_ERROR(iree_hal_memory_file_contents(file, &file_contents));
  }
  out_transfer->host_allocator = host_allocator;
  out_transfer->file = file;
  out_transfer->buffer = buffer;
  out_transfer->file_access = file_access;
  out_transfer->file_offset = file_offset;
  out_transfer->buffer_offset = buffer_offset;
  out_transfer->length = length;
  out_transfer->file_contents = file_contents;
  out_transfer->async.file = async_file;
  iree_hal_file_retain(file);
  iree_hal_buffer_retain(buffer);
  return iree_ok_status();
}

void iree_hal_amd_fabric_file_transfer_deinitialize(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  iree_hal_amd_fabric_file_transfer_release_host(transfer);
  iree_allocator_free_aligned(transfer->host_allocator,
                              transfer->async.staging.data);
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
}

iree_status_t iree_hal_amd_fabric_file_transfer_flush(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  if (!transfer->host.count) {
    return iree_hal_amd_fabric_buffer_flush_host_range(
        transfer->buffer, transfer->buffer_offset, transfer->length);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < transfer->host.count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_fabric_file_host_range_t* range =
        &transfer->host.ranges[i];
    if (range->flush) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          transfer->host.api->host_mapping_cache_control(
              range->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
              range->mapping_offset, range->contents.data_length),
          "host_mapping_cache_control(FLUSH)");
    }
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_finish(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  const iree_host_size_t offset =
      (iree_host_size_t)transfer->async.chunk.offset;
  const iree_host_size_t length =
      transfer->async.file ? transfer->async.chunk.contents.data_length
                           : transfer->buffer_contents.data_length;
  iree_status_t status = iree_ok_status();
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    if (transfer->async.staging.data) {
      memcpy(transfer->buffer_contents.data + offset,
             transfer->async.staging.data, length);
    }
    status = iree_hal_amd_fabric_buffer_flush_host_range(
        transfer->buffer, transfer->buffer_offset + offset, length);
  }
  if (transfer->async.bytes_transferred == transfer->length) {
    iree_allocator_free_aligned(transfer->host_allocator,
                                transfer->async.staging.data);
    transfer->async.staging = iree_byte_span_empty();
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (transfer->host.count) {
    const uint64_t position = transfer->async.bytes_transferred;
    const iree_hal_amd_fabric_file_host_range_t* range =
        &transfer->host.ranges[transfer->host.index];
    if (position == range->offset + range->contents.data_length) {
      range = &transfer->host.ranges[++transfer->host.index];
    }
    if (position == range->offset && range->invalidate) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
                  transfer->host.api->host_mapping_cache_control(
                      range->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                      range->mapping_offset, range->contents.data_length),
                  "host_mapping_cache_control(INVALIDATE)"));
    }
    const iree_host_size_t offset =
        (iree_host_size_t)(position - range->offset);
    const iree_status_t status =
        iree_hal_amd_fabric_file_transfer_prepare_chunk(
            transfer,
            iree_make_byte_span(range->contents.data + offset,
                                iree_min(range->contents.data_length - offset,
                                         UINT32_C(1) << 30)));
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (!transfer->buffer_contents.data) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_buffer_native_host_span(
                transfer->buffer, transfer->buffer_offset, transfer->length,
                &transfer->buffer_contents));
    if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amd_fabric_buffer_invalidate_host_range(
                  transfer->buffer, transfer->buffer_offset, transfer->length));
    }
  }
  iree_status_t status = iree_ok_status();
  if (!transfer->async.file) {
    uint8_t* file_data = transfer->file_contents.data + transfer->file_offset;
    if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
      memmove(transfer->buffer_contents.data, file_data,
              transfer->buffer_contents.data_length);
    } else {
      memmove(file_data, transfer->buffer_contents.data,
              transfer->buffer_contents.data_length);
    }
    status = iree_hal_amd_fabric_file_transfer_finish(transfer);
  } else {
    const iree_async_file_requirements_t* file_requirements =
        iree_hal_file_requirements(transfer->file);
    const iree_async_file_transfer_requirements_t* requirements =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? &file_requirements->read
            : &file_requirements->write;
    if (iree_any_bit_set(transfer->async.file->flags,
                         IREE_ASYNC_FILE_FLAG_DIRECT) &&
        !transfer->async.staging.data) {
      // Device mappings may be CPU-accessible without supporting the page
      // pins required by native uncached I/O. Keep that distinction private to
      // this transfer instead of changing the buffer's allocation contract.
      const iree_host_size_t capacity =
          iree_min(transfer->buffer_contents.data_length, 1024 * 1024);
      const iree_host_size_t alignment =
          iree_max(requirements->buffer_alignment, sizeof(void*));
      status = iree_allocator_malloc_aligned(
          transfer->host_allocator, capacity, alignment, 0,
          (void**)&transfer->async.staging.data);
      if (iree_status_is_ok(status)) {
        transfer->async.staging.data_length = capacity;
      }
    }
    const iree_host_size_t offset =
        (iree_host_size_t)transfer->async.bytes_transferred;
    uint8_t* data = transfer->async.staging.data
                        ? transfer->async.staging.data
                        : transfer->buffer_contents.data + offset;
    const iree_host_size_t capacity =
        transfer->async.staging.data
            ? transfer->async.staging.data_length
            : iree_min(transfer->buffer_contents.data_length - offset,
                       UINT32_C(1) << 30);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_file_transfer_prepare_chunk(
          transfer, iree_make_byte_span(data, capacity));
    }
    if (iree_status_is_ok(status)) {
      if (transfer->async.staging.data &&
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
        memcpy(data, transfer->buffer_contents.data + offset,
               transfer->async.chunk.contents.data_length);
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_chunk(
    iree_hal_amd_fabric_file_transfer_t* transfer, iree_byte_span_t contents) {
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(transfer->file);
  const uint64_t offset = transfer->async.bytes_transferred;
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_prepare_transfer(
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
          ? &requirements->read
          : &requirements->write,
      transfer->file_offset + offset, contents.data, transfer->length - offset,
      contents.data_length, &length));
  transfer->async.chunk.offset = offset;
  transfer->async.chunk.contents = iree_make_byte_span(contents.data, length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_fabric_file_transfer_issue(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_async_completion_fn_t completion) {
  const uint64_t offset = transfer->async.bytes_transferred;
  const iree_host_size_t chunk_offset =
      (iree_host_size_t)(offset - transfer->async.chunk.offset);
  const iree_async_span_t span = iree_async_span_from_ptr(
      transfer->async.chunk.contents.data + chunk_offset,
      transfer->async.chunk.contents.data_length - chunk_offset);
  iree_async_operation_t* operation = NULL;
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    iree_async_file_read_operation_t* read = &transfer->async.read;
    iree_async_operation_zero(&read->base, sizeof(*read));
    iree_async_operation_initialize(
        &read->base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, transfer);
    read->file = transfer->async.file;
    read->offset = transfer->file_offset + offset;
    read->buffer = span;
    operation = &read->base;
  } else {
    iree_async_file_write_operation_t* write = &transfer->async.write;
    iree_async_operation_zero(&write->base, sizeof(*write));
    iree_async_operation_initialize(
        &write->base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, transfer);
    write->file = transfer->async.file;
    write->offset = transfer->file_offset + offset;
    write->buffer = span;
    operation = &write->base;
  }
  return iree_async_proactor_submit_one(transfer->async.file->proactor,
                                        operation);
}

static void iree_hal_amd_fabric_file_transfer_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amd_fabric_file_transfer_t* transfer = user_data;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t byte_count =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? transfer->async.read.bytes_read
            : transfer->async.write.bytes_written;
    transfer->async.bytes_transferred += byte_count;
    if (byte_count == 0) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file %s stopped after %" PRIu64 " of %" PRIu64 " bytes",
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ ? "read"
                                                               : "write",
          transfer->async.bytes_transferred, (uint64_t)transfer->length);
    } else if (transfer->async.bytes_transferred <
               transfer->async.chunk.offset +
                   transfer->async.chunk.contents.data_length) {
      const uint64_t offset = transfer->async.bytes_transferred;
      const iree_host_size_t chunk_offset =
          (iree_host_size_t)(offset - transfer->async.chunk.offset);
      const iree_host_size_t remaining =
          transfer->async.chunk.contents.data_length - chunk_offset;
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(transfer->file);
      iree_host_size_t length = 0;
      status = iree_hal_file_prepare_transfer(
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? &requirements->read
              : &requirements->write,
          transfer->file_offset + offset,
          transfer->async.chunk.contents.data + chunk_offset, remaining,
          remaining, &length);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_file_transfer_issue(
            transfer, iree_hal_amd_fabric_file_transfer_complete);
      }
      if (iree_status_is_ok(status)) {
        return;
      }
    }
  }
  transfer->async.callback.fn(transfer->async.callback.user_data, status);
}

iree_status_t iree_hal_amd_fabric_file_transfer_submit(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_hal_amd_fabric_file_transfer_callback_t callback) {
  transfer->async.callback = callback;
  return iree_hal_amd_fabric_file_transfer_issue(
      transfer, iree_hal_amd_fabric_file_transfer_complete);
}
