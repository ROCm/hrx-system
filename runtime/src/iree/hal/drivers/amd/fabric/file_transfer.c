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

static void iree_hal_amd_fabric_file_transfer_release_sources(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  if (transfer->sources.ranges != &transfer->sources.inline_range) {
    iree_allocator_free(transfer->host_allocator, transfer->sources.ranges);
  }
  memset(&transfer->sources, 0, sizeof(transfer->sources));
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

static iree_status_t iree_hal_amd_fabric_file_transfer_prepare_source(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    const iree_hal_amd_fabric_io_binding_table_t* io_bindings,
    uint32_t native_family_ordinal, iree_hal_memory_access_t queue_access,
    iree_hal_amd_fabric_file_range_t* range,
    iree_hal_memory_effects_t* out_before,
    iree_hal_memory_effects_t* out_after) {
  const amdf_api_t* api = transfer->sources.api;
  const iree_hal_amd_fabric_memory_t* storage = range->storage;
  iree_async_region_t* region =
      io_bindings ? io_bindings->regions[range->access_ordinal] : NULL;
  if (region) {
    if (region->proactor != transfer->async.file->proactor) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "registered buffer and file use different I/O proactors");
    }
  } else {
    if (!(storage->host_features & AMDF_HOST_MAPPING_FEATURE_FILE_IO)) {
      // CPU access alone does not establish native file access or page pins.
      // Preserve the backing for an engine copy through file-capable staging.
      return iree_ok_status();
    }
    const amdf_memory_map_flags_t required_access =
        (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
             ? AMDF_MEMORY_MAP_FLAG_WRITE
             : AMDF_MEMORY_MAP_FLAG_READ) |
        ((queue_access & IREE_HAL_MEMORY_ACCESS_WRITE)
             ? AMDF_MEMORY_MAP_FLAG_READ
             : 0);
    if (!iree_all_bits_set(storage->host_access, required_access)) {
      return iree_ok_status();
    }
  }
  const amdf_memory_site_t device = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(device),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device = {.memory = storage->handle,
                       .access_ordinal = range->access_ordinal,
                       .queue_family_ordinal = native_family_ordinal},
  };
  const amdf_memory_site_t host = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(host),
      .kind = AMDF_MEMORY_SITE_KIND_FILE_IO,
      .value.host = {.mapping = storage->mapping},
  };
  bool available = true;
  bool invalidate = false, flush = false;
  iree_hal_memory_effects_t before = {0}, after = {0};
  iree_hal_memory_pair_info_t pair;
  if (queue_access & IREE_HAL_MEMORY_ACCESS_WRITE) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_transfer_query_pair(
        api, &device, &host, &pair, &available));
    if (available) {
      available = iree_hal_amd_fabric_file_transfer_queue_effect(
                      &pair.release, true, &before) &&
                  iree_hal_amd_fabric_file_transfer_host_effect(&pair.acquire,
                                                                &invalidate);
    }
  }
  if (available && transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_transfer_query_pair(
        api, &host, &device, &pair, &available));
    if (available) {
      available = iree_hal_amd_fabric_file_transfer_host_effect(&pair.release,
                                                                &flush) &&
                  iree_hal_amd_fabric_file_transfer_queue_effect(&pair.acquire,
                                                                 false, &after);
    }
  }
  if (available) {
    // Only a complete host route contributes visibility effects. A native copy
    // range retains its source identity without partially applied host recipes.
    out_before->bits |= before.bits;
    out_after->bits |= after.bits;
    range->region = region;
    range->contents.data =
        (region ? (uint8_t*)region->base_ptr : storage->contents.data) +
        range->source_offset;
    range->effects.bits =
        (invalidate ? IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE : 0) |
        (flush ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH : 0);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_ranges(
    iree_hal_amd_fabric_file_transfer_t* transfer, const amdf_api_t* api,
    const iree_hal_queue_family_t* family, uint32_t native_family_ordinal,
    iree_hal_memory_effects_t* out_before,
    iree_hal_memory_effects_t* out_after) {
  *out_before = (iree_hal_memory_effects_t){0};
  *out_after = (iree_hal_memory_effects_t){0};
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(transfer->buffer);
  const iree_hal_amd_fabric_io_binding_table_t* physical_bindings = NULL;
  const iree_hal_amd_fabric_memory_t* physical_storage = NULL;
  amdf_memory_mapping_update_t physical_source = {0};
  if (view.root) {
    if (view.root->source_type != IREE_HAL_MEMORY_BACKEND_AMDF) {
      return iree_ok_status();
    }
  } else {
    if (!transfer->async.file) {
      return iree_ok_status();
    }
    const iree_hal_buffer_memory_view_t memory =
        iree_hal_buffer_memory_view(transfer->buffer);
    if (!memory.contract) {
      return iree_ok_status();
    }
    const iree_hal_buffer_binding_layout_t* layout =
        &memory.contract->binding_layout;
    const iree_hal_amd_fabric_memory_binding_facts_t* facts =
        (const void*)(memory.bindings + layout->binding_count);
    physical_storage = facts->memory;
    if (memory.contract->scopes[family->memory.queue_scope_id].interfaces &
        (1u << IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO)) {
      const iree_hal_buffer_native_binding_t binding =
          iree_hal_buffer_memory_native_binding(
              &memory,
              (iree_hal_buffer_native_binding_slot_t){
                  .index =
                      memory.contract->scopes[family->memory.queue_scope_id]
                          .bindings[IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO],
                  .type = IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
              });
      physical_bindings = binding.registered_io.table;
    }
    physical_source = (amdf_memory_mapping_update_t){
        .memory = physical_storage->handle,
        .memory_access_ordinal =
            facts->scope_access_ordinals[family->memory.queue_scope_id],
        .source_byte_offset = memory.offset + transfer->buffer_offset,
        .byte_length = transfer->length,
    };
  }
  transfer->sources.api = api;
  transfer->sources.ranges = &transfer->sources.inline_range;
  const iree_hal_memory_access_t queue_access = iree_hal_buffer_usage_access(
      iree_hal_buffer_family_usage(transfer->buffer, family));
  iree_hal_memory_effects_t before = {0}, after = {0};
  iree_host_size_t capacity = 1;
  iree_status_t status = iree_ok_status();
  // Qualify the complete range before any I/O or copy touches a prefix. Memory
  // files need the same permission check but use the engine's direct transfer.
  for (uint64_t offset = 0;
       offset < transfer->length && iree_status_is_ok(status);) {
    amdf_memory_mapping_update_t source = physical_source;
    const iree_hal_amd_fabric_memory_t* storage = physical_storage;
    const iree_hal_amd_fabric_io_binding_table_t* io_bindings =
        physical_bindings;
    if (view.root) {
      iree_hal_amd_fabric_virtual_mapping_t mapping;
      status = iree_hal_amd_fabric_virtual_buffer_query_source(
          view, family->memory.queue_scope_id, transfer->buffer_offset + offset,
          transfer->length - offset, &mapping);
      if (iree_status_is_ok(status)) {
        source = mapping.native;
        storage = mapping.source->storage;
        io_bindings = mapping.source->io_bindings;
      }
    }
    const amdf_memory_access_t required =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? AMDF_MEMORY_ACCESS_WRITE
            : AMDF_MEMORY_ACCESS_READ;
    if (iree_status_is_ok(status) && view.root && !(source.access & required)) {
      status =
          iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                           "file access exceeds mapped source permissions");
    }
    if (iree_status_is_ok(status) && transfer->async.file &&
        transfer->sources.count == capacity) {
      if (transfer->sources.ranges == &transfer->sources.inline_range) {
        iree_hal_amd_fabric_file_range_t* ranges = NULL;
        status = iree_allocator_grow_array(transfer->host_allocator, 4,
                                           sizeof(*ranges), &capacity,
                                           (void**)&ranges);
        if (iree_status_is_ok(status)) {
          ranges[0] = transfer->sources.inline_range;
          transfer->sources.ranges = ranges;
        }
      } else {
        status = iree_allocator_grow_array(
            transfer->host_allocator, transfer->sources.count + 1,
            sizeof(*transfer->sources.ranges), &capacity,
            (void**)&transfer->sources.ranges);
      }
    }
    if (iree_status_is_ok(status) && transfer->async.file) {
      iree_hal_amd_fabric_file_range_t* range =
          &transfer->sources.ranges[transfer->sources.count++];
      *range = (iree_hal_amd_fabric_file_range_t){
          .storage = storage,
          .contents = iree_make_byte_span(NULL, source.byte_length),
          .source_offset = source.source_byte_offset,
          .offset = offset,
          .access_ordinal = source.memory_access_ordinal,
      };
      status = iree_hal_amd_fabric_file_transfer_prepare_source(
          transfer, io_bindings, native_family_ordinal, queue_access, range,
          &before, &after);
      transfer->sources.host_count += range->contents.data != NULL;
    }
    offset += source.byte_length;
  }
  if (iree_status_is_ok(status)) {
    *out_before = before;
    *out_after = after;
  } else {
    iree_hal_amd_fabric_file_transfer_release_sources(transfer);
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
  iree_hal_amd_fabric_file_transfer_release_sources(transfer);
  iree_allocator_free_aligned(transfer->host_allocator,
                              transfer->async.staging.data);
  iree_hal_buffer_release(transfer->buffer);
  iree_hal_file_release(transfer->file);
}

iree_status_t iree_hal_amd_fabric_file_transfer_flush(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  if (!transfer->sources.count) {
    return iree_hal_amd_fabric_buffer_flush_host_range(
        transfer->buffer, transfer->buffer_offset, transfer->length);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < transfer->sources.count && iree_status_is_ok(status); ++i) {
    const iree_hal_amd_fabric_file_range_t* range =
        &transfer->sources.ranges[i];
    if (range->effects.bits & IREE_HAL_MEMORY_EFFECT_HOST_FLUSH) {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "amdf.host_mapping_flush");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, range->contents.data_length);
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          transfer->sources.api->host_mapping_cache_control(
              range->storage->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH,
              range->source_offset, range->contents.data_length),
          "host_mapping_cache_control(FLUSH)");
      IREE_TRACE_ZONE_END(z0);
    }
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_finish(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  if (transfer->sources.count) {
    return iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer) ==
                   transfer->length
               ? iree_hal_amd_fabric_file_transfer_flush(transfer)
               : iree_ok_status();
  }
  const iree_host_size_t offset =
      (iree_host_size_t)transfer->async.request.offset;
  const iree_host_size_t length = transfer->async.file
                                      ? transfer->async.request.span.length
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
  if (iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer) ==
      transfer->length) {
    iree_allocator_free_aligned(transfer->host_allocator,
                                transfer->async.staging.data);
    transfer->async.staging = iree_byte_span_empty();
  }
  return status;
}

const iree_hal_amd_fabric_file_range_t*
iree_hal_amd_fabric_file_transfer_current_range(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  if (!transfer->sources.count) {
    return NULL;
  }
  const iree_hal_amd_fabric_file_range_t* range =
      &transfer->sources.ranges[transfer->sources.index];
  if (iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer) ==
      range->offset + range->contents.data_length) {
    range = &transfer->sources.ranges[++transfer->sources.index];
  }
  return range;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_range(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    const iree_hal_amd_fabric_file_range_t* range) {
  const uint64_t position =
      iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer);
  if (position == range->offset &&
      (range->effects.bits & IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "amdf.host_mapping_invalidate");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, range->contents.data_length);
    const amdf_status_t status =
        transfer->sources.api->host_mapping_cache_control(
            range->storage->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
            range->source_offset, range->contents.data_length);
    IREE_TRACE_ZONE_END(z0);
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "host_mapping_cache_control(INVALIDATE)"));
  }
  const iree_host_size_t offset = (iree_host_size_t)(position - range->offset);
  return iree_hal_amd_fabric_file_transfer_prepare_chunk(
      transfer,
      range->region
          ? iree_async_span_make(range->region, range->source_offset + offset,
                                 range->contents.data_length - offset)
          : iree_async_span_from_ptr(
                range->contents.data + offset,
                iree_min(range->contents.data_length - offset, UINT32_C(1)
                                                                   << 30)));
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare(
    iree_hal_amd_fabric_file_transfer_t* transfer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (transfer->sources.count) {
    if (transfer->sources.host_count == transfer->sources.count) {
      iree_status_t status = iree_hal_amd_fabric_file_transfer_prepare_range(
          transfer, iree_hal_amd_fabric_file_transfer_current_range(transfer));
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    // Host-only engines require one complete public host binding when native
    // file views do not cover the operation. Resolve it before touching a
    // prefix; a device-only virtual buffer cannot silently enter the host copy
    // path.
    iree_hal_amd_fabric_file_transfer_release_sources(transfer);
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
        (iree_host_size_t)iree_hal_amd_fabric_file_transfer_bytes_transferred(
            transfer);
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
          transfer, iree_async_span_from_ptr(data, capacity));
    }
    if (iree_status_is_ok(status)) {
      if (transfer->async.staging.data &&
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_WRITE) {
        memcpy(data, transfer->buffer_contents.data + offset,
               transfer->async.request.span.length);
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amd_fabric_file_transfer_prepare_chunk(
    iree_hal_amd_fabric_file_transfer_t* transfer, iree_async_span_t span) {
  const uint64_t offset =
      iree_hal_amd_fabric_file_transfer_bytes_transferred(transfer);
  return iree_hal_amd_fabric_file_request_prepare(transfer, offset, span,
                                                  &transfer->async.request);
}

iree_status_t iree_hal_amd_fabric_file_request_prepare(
    const iree_hal_amd_fabric_file_transfer_t* transfer, uint64_t offset,
    iree_async_span_t span, iree_hal_amd_fabric_file_request_t* out_request) {
  // A fixed registration can span several native slots. Keep each request in
  // one slot even when the first byte lies in the middle of that slot.
  if (span.region && span.region->buffer_size) {
    span.length =
        iree_min(span.length, span.region->buffer_size -
                                  span.offset % span.region->buffer_size);
  }
  const iree_async_file_requirements_t* file_requirements =
      iree_hal_file_requirements(transfer->file);
  const iree_async_file_transfer_requirements_t* requirements =
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
          ? &file_requirements->read
          : &file_requirements->write;
  iree_host_size_t admitted_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_prepare_transfer(
      requirements, transfer->file_offset + offset, iree_async_span_ptr(span),
      transfer->length - offset, span.length, &admitted_length));
  span.length = admitted_length;
  out_request->offset = offset;
  out_request->span = span;
  out_request->bytes_transferred = 0;
  out_request->read.base.user_data = (void*)transfer;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_fabric_file_request_issue(
    iree_hal_amd_fabric_file_request_t* request,
    iree_async_completion_fn_t completion) {
  // Both native variants begin with the same operation header. Capture its
  // borrowed owner before resetting the native request for this issuance.
  const iree_hal_amd_fabric_file_transfer_t* transfer =
      request->read.base.user_data;
  const iree_host_size_t offset = request->bytes_transferred;
  iree_async_span_t span = request->span;
  span.offset += offset;
  span.length -= offset;
  iree_async_operation_t* operation = NULL;
  if (transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    iree_async_file_read_operation_t* read = &request->read;
    iree_async_operation_zero(&read->base, sizeof(*read));
    iree_async_operation_initialize(
        &read->base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, (void*)transfer);
    read->file = transfer->async.file;
    read->offset = transfer->file_offset + request->offset + offset;
    read->buffer = span;
    operation = &read->base;
  } else {
    iree_async_file_write_operation_t* write = &request->write;
    iree_async_operation_zero(&write->base, sizeof(*write));
    iree_async_operation_initialize(
        &write->base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
        IREE_ASYNC_OPERATION_FLAG_NONE, completion, (void*)transfer);
    write->file = transfer->async.file;
    write->offset = transfer->file_offset + request->offset + offset;
    write->buffer = span;
    operation = &write->base;
  }
  return iree_async_proactor_submit_one(transfer->async.file->proactor,
                                        operation);
}

static void iree_hal_amd_fabric_file_request_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)flags;
  const iree_hal_amd_fabric_file_transfer_t* transfer = user_data;
  iree_hal_amd_fabric_file_request_t* request = iree_containerof(
      operation, iree_hal_amd_fabric_file_request_t, read.base);
  if (iree_status_is_ok(status)) {
    const iree_host_size_t byte_count =
        transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
            ? request->read.bytes_read
            : request->write.bytes_written;
    request->bytes_transferred += byte_count;
    if (byte_count == 0) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file %s stopped after %" PRIu64 " of %" PRIu64 " bytes",
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ ? "read"
                                                               : "write",
          (uint64_t)request->bytes_transferred, (uint64_t)request->span.length);
    } else if (request->bytes_transferred < request->span.length) {
      const iree_host_size_t offset = request->bytes_transferred;
      const iree_host_size_t remaining = request->span.length - offset;
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(transfer->file);
      iree_host_size_t length = 0;
      status = iree_hal_file_prepare_transfer(
          transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
              ? &requirements->read
              : &requirements->write,
          transfer->file_offset + request->offset + offset,
          (uint8_t*)iree_async_span_ptr(request->span) + offset, remaining,
          remaining, &length);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_file_request_issue(
            request, iree_hal_amd_fabric_file_request_complete);
      }
      if (iree_status_is_ok(status)) {
        return;
      }
    }
  }
  request->callback.fn(request->callback.user_data, status);
}

iree_status_t iree_hal_amd_fabric_file_request_submit(
    iree_hal_amd_fabric_file_request_t* request,
    iree_hal_amd_fabric_file_transfer_callback_t callback) {
  request->callback = callback;
  return iree_hal_amd_fabric_file_request_issue(
      request, iree_hal_amd_fabric_file_request_complete);
}
