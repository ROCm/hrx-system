// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/file_transfer.h"

#include "iree/async/region.h"
#include "iree/hal/driver/amd/fabric/memory_transition.h"
#include "iree/hal/driver/amd/status.h"

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
    const amdf_memory_site_t* consumer, bool* inout_available,
    iree_hal_memory_pair_info_t* out_pair) {
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t status =
      api->memory_query_pair_info(producer, consumer, &pair);
  if (amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    *inout_available = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "memory_query_pair_info"));
  return iree_hal_amd_fabric_memory_translate_pair(&pair, out_pair);
}

static iree_status_t iree_hal_amd_fabric_file_transfer_qualify_source(
    iree_hal_memory_access_t file_access,
    const iree_hal_amd_fabric_io_binding_table_t* io_bindings,
    uint32_t native_family_ordinal, iree_hal_memory_access_t queue_access,
    iree_hal_amd_fabric_file_source_t* range,
    iree_hal_memory_effects_t* out_before,
    iree_hal_memory_effects_t* out_after) {
  const amdf_api_t* api = range->api;
  const iree_hal_amd_fabric_memory_t* storage = range->storage;
  iree_async_region_t* region =
      io_bindings ? io_bindings->regions[range->access_ordinal] : NULL;
  if (!region) {
    if (!(storage->host_features & AMDF_HOST_MAPPING_FEATURE_FILE_IO)) {
      // Unpinnable views retain the staging route even when native buffered
      // access is supported. Host file spans use the FILE_IO publication
      // contract; buffered capability alone does not select a direct route.
      return iree_ok_status();
    }
    const amdf_memory_map_flags_t required_access =
        (file_access == IREE_HAL_MEMORY_ACCESS_READ
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
        api, &device, &host, &available, &pair));
    if (available) {
      available = iree_hal_amd_fabric_file_transfer_queue_effect(
                      &pair.release, true, &before) &&
                  iree_hal_amd_fabric_file_transfer_host_effect(&pair.acquire,
                                                                &invalidate);
    }
  }
  if (available && file_access == IREE_HAL_MEMORY_ACCESS_READ) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_transfer_query_pair(
        api, &host, &device, &available, &pair));
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

iree_status_t iree_hal_amd_fabric_file_source_initialize(
    const amdf_api_t* api, const iree_hal_queue_family_t* family,
    uint32_t native_family_ordinal, iree_hal_memory_access_t file_access,
    iree_hal_buffer_binding_t binding,
    iree_hal_amd_fabric_file_source_t* out_source,
    iree_hal_memory_effects_t* out_before,
    iree_hal_memory_effects_t* out_after) {
  *out_source = (iree_hal_amd_fabric_file_source_t){0};
  *out_before = (iree_hal_memory_effects_t){0};
  *out_after = (iree_hal_memory_effects_t){0};
  // Direct I/O consumes the physical buffer selected by its owner. Virtual
  // addresses use the engine's copy route or an admitted host binding; native
  // translations and mapping permissions remain the caller's responsibility.
  if (iree_hal_buffer_virtual_view(binding.buffer).root) {
    return iree_ok_status();
  }
  const iree_hal_buffer_memory_view_t memory =
      iree_hal_buffer_memory_view(binding.buffer);
  if (!memory.contract) {
    return iree_ok_status();
  }
  const iree_hal_buffer_binding_layout_t* layout =
      &memory.contract->binding_layout;
  const iree_hal_amd_fabric_memory_binding_facts_t* facts =
      (const void*)(memory.bindings + layout->binding_count);
  const iree_hal_amd_fabric_io_binding_table_t* io_bindings = NULL;
  if (memory.contract->scopes[family->memory.queue_scope_id].interfaces &
      (1u << IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO)) {
    const iree_hal_buffer_native_binding_t native_binding =
        iree_hal_buffer_memory_native_binding(
            &memory,
            (iree_hal_buffer_native_binding_slot_t){
                .index = memory.contract->scopes[family->memory.queue_scope_id]
                             .bindings[IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO],
                .type = IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO,
            });
    io_bindings = native_binding.registered_io.table;
  }
  *out_source = (iree_hal_amd_fabric_file_source_t){
      .api = api,
      .storage = facts->memory,
      .contents = iree_make_byte_span(NULL, binding.length),
      .source_offset = memory.offset + binding.offset,
      .access_ordinal =
          facts->scope_facts[family->memory.queue_scope_id].access_ordinal,
  };
  const iree_hal_memory_access_t queue_access = iree_hal_buffer_usage_access(
      iree_hal_buffer_family_usage(binding.buffer, family));
  iree_status_t status = iree_hal_amd_fabric_file_transfer_qualify_source(
      file_access, io_bindings, native_family_ordinal, queue_access, out_source,
      out_before, out_after);
  if (!iree_status_is_ok(status)) {
    *out_source = (iree_hal_amd_fabric_file_source_t){0};
  }
  return status;
}
