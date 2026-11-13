// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "device_group.h"
#include "hrx_internal.h"
#include "mem_pool.h"

IREE_STATIC_ASSERT_ENUM_EQ(HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
                           IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
                           "virtual buffer address constraints must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_MAPPING_OPERATION_REPLACE,
                           IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE,
                           "mapping operations must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_MAPPING_OPERATION_BIND,
                           IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
                           "mapping operations must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_MAPPING_OPERATION_UNBIND,
                           IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
                           "mapping operations must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_MAPPING_OPERATION_PROTECT,
                           IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT,
                           "mapping operations must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_MAPPING_SOURCE_FLAG_WHOLE_ALLOCATION,
                           IREE_HAL_POOL_MAPPING_FLAG_WHOLE_ALLOCATION,
                           "mapping source restrictions must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_QUEUE_MAP_FLAG_BORROW_BUFFERS,
                           IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS,
                           "mapping submission flags must match HAL");

// Translation scratch is local to construction. No source pool is created or
// retained here; callers own the exact contracts being admitted.
typedef struct hrx_virtual_buffer_inputs_t {
  // Common-case family translation storage.
  iree_hal_pool_family_access_t inline_families[8];
  // Common-case borrowed native sources.
  iree_hal_pool_t* inline_pools[8];
  // Combined overflow allocation, or NULL for inline storage.
  void* storage;
  // Native access scope borrowing the translated families.
  iree_hal_pool_scope_t scope;
  // Native source list borrowing translated pool handles.
  iree_hal_virtual_buffer_sources_t sources;
} hrx_virtual_buffer_inputs_t;

static iree_status_t hrx_virtual_buffer_prepare_inputs(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    hrx_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_inputs_t* out_inputs) {
  memset(out_inputs, 0, sizeof(*out_inputs));
  if (!group || (scope.family_count && !scope.families) ||
      !sources.source_count || !sources.sources) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid virtual buffer group, scope, or sources");
  }
  iree_hal_pool_family_access_t* families = out_inputs->inline_families;
  iree_hal_pool_t** pools = out_inputs->inline_pools;
  if (scope.family_count > IREE_ARRAYSIZE(out_inputs->inline_families) ||
      sources.source_count > IREE_ARRAYSIZE(out_inputs->inline_pools)) {
    iree_host_size_t family_bytes = 0;
    iree_host_size_t source_bytes = 0;
    iree_host_size_t storage_bytes = 0;
    if (!iree_host_size_checked_mul(scope.family_count, sizeof(*families),
                                    &family_bytes) ||
        !iree_host_size_checked_mul(sources.source_count, sizeof(*pools),
                                    &source_bytes) ||
        !iree_host_size_checked_add(family_bytes, source_bytes,
                                    &storage_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "virtual buffer construction inputs overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        group->runtime->host_allocator, storage_bytes, &out_inputs->storage));
    families = out_inputs->storage;
    pools = (iree_hal_pool_t**)((uint8_t*)out_inputs->storage + family_bytes);
  }
  iree_status_t status =
      hrx_mem_pool_prepare_scope(group, scope, families, &out_inputs->scope);
  for (size_t i = 0; i < sources.source_count && iree_status_is_ok(status);
       ++i) {
    hrx_mem_pool_t pool = sources.sources[i];
    if (!pool || pool->group != group) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "source pool %zu is not in the sealed group", i);
    } else {
      pools[i] = pool->hal_pool;
    }
  }
  out_inputs->sources =
      (iree_hal_virtual_buffer_sources_t){sources.source_count, pools};
  return status;
}

static void hrx_virtual_buffer_release_inputs(
    hrx_device_group_t group, hrx_virtual_buffer_inputs_t* inputs) {
  if (inputs->storage) {
    iree_allocator_free(group->runtime->host_allocator, inputs->storage);
  }
}

hrx_status_t hrx_virtual_buffer_query_capabilities(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    hrx_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_capabilities_t* out_capabilities) {
  if (!out_capabilities) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "reservation capabilities output is NULL");
  }
  *out_capabilities = (hrx_virtual_buffer_capabilities_t){0};
  hrx_virtual_buffer_inputs_t inputs;
  iree_status_t status =
      hrx_virtual_buffer_prepare_inputs(group, scope, sources, &inputs);
  iree_hal_virtual_buffer_capabilities_t capabilities = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_virtual_buffer_query_capabilities(
        group->hal_group, inputs.scope, inputs.sources,
        group->runtime->host_allocator, &capabilities);
  }
  if (iree_status_is_ok(status) &&
      capabilities.reservation_granularity <= SIZE_MAX &&
      capabilities.mapping_granularity <= SIZE_MAX) {
    *out_capabilities = (hrx_virtual_buffer_capabilities_t){
        .max_byte_length =
            (size_t)iree_min(capabilities.max_byte_length, SIZE_MAX),
        .reservation_granularity = (size_t)capabilities.reservation_granularity,
        .mapping_granularity = (size_t)capabilities.mapping_granularity,
        .max_alignment =
            (size_t)iree_min(capabilities.max_alignment, (SIZE_MAX >> 1) + 1),
        .operations = capabilities.operations,
        .supported_flags = capabilities.supported_flags,
    };
  }
  hrx_virtual_buffer_release_inputs(group, &inputs);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_virtual_buffer_create(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    hrx_virtual_buffer_sources_t sources,
    const hrx_virtual_buffer_options_t* options, hrx_buffer_t* out_buffer) {
  if (!out_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "reservation output is NULL");
  }
  *out_buffer = NULL;
  if (!options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "reservation options are NULL");
  }
  hrx_virtual_buffer_inputs_t inputs;
  iree_status_t status =
      hrx_virtual_buffer_prepare_inputs(group, scope, sources, &inputs);
  iree_hal_buffer_t* buffer = NULL;
  const iree_hal_virtual_buffer_options_t hal_options = {
      .byte_length = options->byte_length,
      .min_alignment = options->min_alignment,
      .mapping_granularity = options->mapping_granularity,
      .flags = options->flags,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_virtual_buffer_create(
        group->hal_group, inputs.scope, inputs.sources, &hal_options,
        group->runtime->host_allocator, &buffer);
  }
  if (iree_status_is_ok(status)) {
    status = hrx_buffer_create_from_hal(buffer, group,
                                        HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                                        options->byte_length, NULL, out_buffer);
  }
  hrx_virtual_buffer_release_inputs(group, &inputs);
  iree_hal_buffer_release(buffer);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_virtual_buffer_query_info(
    hrx_buffer_t buffer, hrx_virtual_buffer_info_t* out_info) {
  if (!buffer || !out_info) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "virtual buffer or info output is NULL");
  }
  *out_info = (hrx_virtual_buffer_info_t){0};
  iree_hal_virtual_buffer_info_t info;
  iree_status_t status =
      iree_hal_virtual_buffer_query_info(buffer->hal_buffer, &info);
  if (iree_status_is_ok(status)) {
    *out_info = (hrx_virtual_buffer_info_t){
        .mapping_byte_offset = (size_t)info.mapping_byte_offset,
        .mapping_granularity = (size_t)info.mapping_granularity,
        .operations = info.operations,
        .projection_count = info.projection_count,
    };
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_virtual_buffer_resolve_projection(hrx_buffer_t buffer,
                                                   hrx_memory_scope_t scope,
                                                   uint32_t* out_projection) {
  if (!buffer || !out_projection) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "projection buffer or output is NULL");
  }
  const iree_hal_memory_scope_t hal_scope = {scope.domain, scope.id};
  return hrx_status_from_iree(iree_hal_virtual_buffer_resolve_projection(
      buffer->hal_buffer, hal_scope, out_projection));
}

hrx_status_t hrx_virtual_buffer_detach(hrx_buffer_t buffer,
                                       hrx_memory_scope_t scope) {
  if (!buffer || !buffer->hal_buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "virtual buffer is NULL or has no HAL storage");
  }
  const iree_hal_memory_contract_t* contract =
      buffer->hal_buffer->memory.contract;
  if (!scope.domain && !scope.id && contract) {
    scope.domain = contract->domain;
  }
  const iree_hal_memory_scope_t hal_scope = {scope.domain, scope.id};
  return hrx_status_from_iree(
      iree_hal_virtual_buffer_detach(buffer->hal_buffer, hal_scope));
}
