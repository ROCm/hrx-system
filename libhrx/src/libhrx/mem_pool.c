// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "mem_pool.h"

#include "buffer.h"
#include "device_group.h"
#include "hrx_internal.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"

IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
                           IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
                           IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH_IMAGE_READ,
                           IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH_IMAGE_WRITE,
                           IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH_IMAGE,
                           IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_BUFFER_USAGE_DISPATCH,
                           IREE_HAL_BUFFER_USAGE_DISPATCH,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MAPPING_MODE_SCOPED,
                           IREE_HAL_MAPPING_MODE_SCOPED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MAPPING_MODE_PERSISTENT,
                           IREE_HAL_MAPPING_MODE_PERSISTENT,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_HOST_MEMORY_PROPERTY_READ_CACHED,
                           IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_HOST_MEMORY_PROPERTY_READ_UNCACHED,
                           IREE_HAL_HOST_MEMORY_PROPERTY_READ_UNCACHED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_HOST_MEMORY_PROPERTY_WRITE_BACK,
                           IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_HOST_MEMORY_PROPERTY_WRITE_COMBINED,
                           IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_COMBINED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_WAIT,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_STORE,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_STORE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_RMW_ADD,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_RMW_AND,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_RMW_OR,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ATOMIC_OPERATION_FLAG_RMW_XOR,
                           IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST,
                           IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_ACCESS_REQUIRE_UNCACHED,
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_TRIM_FLAG_NONE,
                           IREE_HAL_POOL_TRIM_FLAG_NONE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_TRIM_FLAG_ALL, IREE_HAL_POOL_TRIM_FLAG_ALL,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_TRIM_FLAG_EXCESS,
                           IREE_HAL_POOL_TRIM_FLAG_EXCESS,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_PLACEMENT_AUTOMATIC,
                           IREE_HAL_POOL_PLACEMENT_AUTOMATIC,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_PLACEMENT_PREFERRED,
                           IREE_HAL_POOL_PLACEMENT_PREFERRED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_POOL_PLACEMENT_REQUIRED,
                           IREE_HAL_POOL_PLACEMENT_REQUIRED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_NONE,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_OPAQUE_FD,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_VULKAN_OPAQUE_WIN32,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_VULKAN_OPAQUE_WIN32,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE,
                           IREE_HAL_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_SLAB_POOL_OWNED_ENABLED,
                           IREE_HAL_SLAB_POOL_OWNED_ENABLED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_SLAB_POOL_IMPORT_ONLY,
                           IREE_HAL_SLAB_POOL_IMPORT_ONLY,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ASAN_POOL_MODE_DISABLED,
                           IREE_HAL_ASAN_POOL_MODE_DISABLED,
                           "memory vocabulary must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_ASAN_POOL_MODE_SHADOW,
                           IREE_HAL_ASAN_POOL_MODE_SHADOW,
                           "memory vocabulary must match HAL");

static iree_hal_atomic_operation_capabilities_t hrx_atomic_to_hal(
    hrx_atomic_operation_capabilities_t value) {
  return (iree_hal_atomic_operation_capabilities_t){
      value.device_scope_32, value.device_scope_64, value.system_scope_32,
      value.system_scope_64};
}

static hrx_atomic_operation_capabilities_t hrx_atomic_from_hal(
    iree_hal_atomic_operation_capabilities_t value) {
  return (hrx_atomic_operation_capabilities_t){
      value.device_scope_32, value.device_scope_64, value.system_scope_32,
      value.system_scope_64};
}

static iree_hal_pool_host_access_t hrx_pool_host_access_to_hal(
    hrx_pool_host_access_t value) {
  return (iree_hal_pool_host_access_t){
      .access = value.access,
      .modes = value.modes,
      .properties = value.properties,
      .atomics = hrx_atomic_to_hal(value.atomics),
  };
}

static iree_hal_asan_pool_options_t hrx_asan_pool_options_to_hal(
    hrx_asan_pool_options_t value) {
  return (iree_hal_asan_pool_options_t){
      .mode = (iree_hal_asan_pool_mode_t)value.mode,
      .shadow_granule_size = value.shadow_granule_size,
      .redzone_size = value.redzone_size,
      .backing_alignment = value.backing_alignment,
      .quarantine_size = value.quarantine_size,
  };
}

// Takes the native constructor's reference on success or failure.
static iree_status_t hrx_mem_pool_wrap(iree_hal_pool_t* hal_pool,
                                       hrx_device_group_t group,
                                       hrx_buffer_t backing_buffer,
                                       hrx_mem_pool_t* out_pool) {
  hrx_mem_pool_t pool = NULL;
  iree_status_t status = iree_allocator_malloc(group->runtime->host_allocator,
                                               sizeof(*pool), (void**)&pool);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&pool->ref_count);
    pool->hal_pool = hal_pool;
    pool->group = group;
    hrx_device_group_retain(group);
    pool->backing_buffer = backing_buffer;
    hrx_buffer_retain(backing_buffer);
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(hal_pool, &capabilities);
    pool->memory_type = capabilities.memory_type;
    *out_pool = pool;
  } else {
    iree_hal_pool_release(hal_pool);
  }
  return status;
}

void hrx_mem_pool_retain(hrx_mem_pool_t pool) {
  if (pool) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void hrx_mem_pool_release(hrx_mem_pool_t pool) {
  if (!pool) {
    return;
  }
  if (iree_atomic_ref_count_dec(&pool->ref_count) != 1) {
    return;
  }
  hrx_device_group_t group = pool->group;
  iree_hal_pool_release(pool->hal_pool);
  hrx_buffer_release(pool->backing_buffer);
  iree_allocator_free(group->runtime->host_allocator, pool);
  hrx_device_group_release(group);
}

hrx_device_group_t hrx_mem_pool_get_device_group(hrx_mem_pool_t pool) {
  return pool ? pool->group : NULL;
}

static iree_hal_external_storage_class_t hrx_storage_class_to_hal(
    hrx_external_storage_class_t value) {
  return (iree_hal_external_storage_class_t){
      .type = (iree_hal_external_storage_type_t)value.type,
      .provenance = {value.provenance[0], value.provenance[1]},
  };
}

// Public counted-array validation and size accounting precede native route
// qualification. Translation scratch exists only during source construction.
static iree_status_t hrx_pool_count_exports(hrx_pool_export_classes_t exports,
                                            size_t* inout_count) {
  if ((exports.class_count && !exports.classes) ||
      !iree_host_size_checked_add(*inout_count, exports.class_count,
                                  inout_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid external storage class array");
  }
  return iree_ok_status();
}

static iree_hal_pool_export_classes_t hrx_pool_copy_exports(
    hrx_pool_export_classes_t exports,
    iree_hal_external_storage_class_t** cursor) {
  iree_hal_pool_export_classes_t result = {exports.class_count, *cursor};
  for (size_t i = 0; i < exports.class_count; ++i) {
    *(*cursor)++ = hrx_storage_class_to_hal(exports.classes[i]);
  }
  return result;
}

iree_status_t hrx_mem_pool_prepare_scope(
    hrx_device_group_t group, hrx_pool_scope_t scope,
    iree_hal_pool_family_access_t* out_families,
    iree_hal_pool_scope_t* out_scope) {
  *out_scope = (iree_hal_pool_scope_t){0};
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < scope.family_count && iree_status_is_ok(status); ++i) {
    const hrx_pool_family_access_t* access = &scope.families[i];
    const iree_hal_queue_family_t* family =
        access->device && access->device->group == group
            ? iree_hal_device_queue_family(access->device->hal_device,
                                           access->family_ordinal)
            : NULL;
    if (!family) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "pool family %zu is not in the sealed group", i);
    } else {
      out_families[i] = (iree_hal_pool_family_access_t){
          .family = family,
          .usage = access->usage,
          .interfaces = access->interfaces,
          .requirements = access->requirements,
          .atomics = {.queue = hrx_atomic_to_hal(access->atomics.queue),
                      .queue_zero_compute =
                          hrx_atomic_to_hal(access->atomics.queue_zero_compute),
                      .program = hrx_atomic_to_hal(access->atomics.program)},
      };
    }
  }
  if (iree_status_is_ok(status)) {
    *out_scope = (iree_hal_pool_scope_t){
        .family_count = scope.family_count,
        .families = out_families,
        .host = hrx_pool_host_access_to_hal(scope.host),
    };
  }
  return status;
}

hrx_status_t hrx_slab_pool_create(hrx_device_group_t group,
                                  hrx_pool_scope_t scope,
                                  const hrx_slab_pool_options_t* options,
                                  hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!group || (scope.family_count && !scope.families)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid group or pool scope");
  }
  const hrx_slab_pool_options_t defaults = {0};
  if (!options) {
    options = &defaults;
  }
  if (options->imports.class_count && !options->imports.classes) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid import class array");
  }
  size_t export_count = 0;
  iree_status_t status =
      hrx_pool_count_exports(options->owned.exports, &export_count);
  for (size_t i = 0;
       i < options->imports.class_count && iree_status_is_ok(status); ++i) {
    status = hrx_pool_count_exports(options->imports.classes[i].exports,
                                    &export_count);
  }
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  iree_allocator_t allocator = group->runtime->host_allocator;
  iree_hal_pool_family_access_t inline_families[8];
  iree_hal_pool_family_access_t* families = inline_families;
  iree_hal_pool_import_class_t* imports = NULL;
  iree_hal_external_storage_class_t* exports = NULL;
  if (scope.family_count > IREE_ARRAYSIZE(inline_families)) {
    status = iree_allocator_malloc_array(allocator, scope.family_count,
                                         sizeof(*families), (void**)&families);
  }
  if (iree_status_is_ok(status) && options->imports.class_count) {
    status =
        iree_allocator_malloc_array(allocator, options->imports.class_count,
                                    sizeof(*imports), (void**)&imports);
  }
  if (iree_status_is_ok(status) && export_count) {
    status = iree_allocator_malloc_array(allocator, export_count,
                                         sizeof(*exports), (void**)&exports);
  }
  iree_hal_pool_scope_t hal_scope;
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_prepare_scope(group, scope, families, &hal_scope);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_external_storage_class_t* cursor = exports;
    iree_hal_slab_pool_options_t hal_options = {
        .placement = {(iree_hal_pool_placement_mode_t)options->placement.mode,
                      options->placement.node},
        .owned = {.mode = (iree_hal_slab_pool_owned_mode_t)options->owned.mode,
                  .exports =
                      hrx_pool_copy_exports(options->owned.exports, &cursor)},
        .imports = {options->imports.class_count, imports},
        .preferences = {.host = hrx_pool_host_access_to_hal(
                            options->preferences.host)},
        .trace_name = iree_make_cstring_view(options->trace_name),
    };
    for (size_t i = 0; i < options->imports.class_count; ++i) {
      imports[i] = (iree_hal_pool_import_class_t){
          .storage_class = hrx_storage_class_to_hal(
              options->imports.classes[i].storage_class),
          .exports = hrx_pool_copy_exports(options->imports.classes[i].exports,
                                           &cursor),
      };
    }
    iree_hal_pool_t* hal_pool = NULL;
    status = iree_hal_slab_pool_create(group->hal_group, hal_scope,
                                       &hal_options, allocator, &hal_pool);
    if (iree_status_is_ok(status)) {
      status = hrx_mem_pool_wrap(hal_pool, group, NULL, out_pool);
    }
  }
  iree_allocator_free(allocator, exports);
  iree_allocator_free(allocator, imports);
  if (families != inline_families) {
    iree_allocator_free(allocator, families);
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_slab_cache_create(hrx_mem_pool_t backing,
                                   const hrx_slab_cache_options_t* options,
                                   hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!backing || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "backing or options is NULL");
  }
  const iree_hal_slab_cache_options_t hal_options = {
      .slab = {options->slab.allocation_size, options->slab.min_alignment},
      .minimum_cached_size = options->minimum_cached_size,
      .max_count = options->max_count,
      .trace_name = iree_make_cstring_view(options->trace_name),
  };
  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_slab_cache_create(
      backing->hal_pool, &hal_options, backing->group->runtime->host_allocator,
      &hal_pool);
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_wrap(hal_pool, backing->group,
                               backing->backing_buffer, out_pool);
  }
  return hrx_status_from_iree(status);
}

static iree_hal_tlsf_pool_options_t hrx_tlsf_pool_options_to_hal(
    const hrx_tlsf_pool_options_t* options) {
  return (iree_hal_tlsf_pool_options_t){
      .tlsf_options = {.range_length = options->slab_size,
                       .alignment =
                           options->alignment ? options->alignment : 16,
                       .initial_block_capacity =
                           options->initial_block_capacity,
                       .frontier_capacity = options->frontier_capacity},
      .max_suballocation_size = options->max_suballocation_size,
      .asan = hrx_asan_pool_options_to_hal(options->asan),
      .budget_limit = options->budget_limit,
      .trace_name = iree_make_cstring_view(options->trace_name),
  };
}

static iree_hal_fixed_block_pool_options_t hrx_fixed_block_pool_options_to_hal(
    const hrx_fixed_block_pool_options_t* options) {
  return (iree_hal_fixed_block_pool_options_t){
      .block_size = options->block_size,
      .blocks_per_slab = options->blocks_per_slab,
      .frontier_capacity = options->frontier_capacity,
      .alignment = options->alignment,
      .asan = hrx_asan_pool_options_to_hal(options->asan),
      .budget_limit = options->budget_limit,
      .trace_name = iree_make_cstring_view(options->trace_name),
  };
}

hrx_status_t hrx_tlsf_pool_query_backing_request(
    hrx_mem_pool_t backing, const hrx_tlsf_pool_options_t* options,
    hrx_pool_reservation_request_t* out_request) {
  if (!out_request) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_request = (hrx_pool_reservation_request_t){0};
  if (!backing || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "backing or options is NULL");
  }
  const iree_hal_tlsf_pool_options_t hal_options =
      hrx_tlsf_pool_options_to_hal(options);
  iree_hal_pool_reservation_request_t request;
  iree_status_t status = iree_hal_tlsf_pool_query_backing_request(
      backing->hal_pool, &hal_options, &request);
  if (iree_status_is_ok(status)) {
    *out_request = (hrx_pool_reservation_request_t){request.allocation_size,
                                                    request.min_alignment};
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_tlsf_pool_create(hrx_mem_pool_t backing,
                                  const hrx_tlsf_pool_options_t* options,
                                  hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!backing || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "backing or options is NULL");
  }
  const iree_hal_tlsf_pool_options_t hal_options =
      hrx_tlsf_pool_options_to_hal(options);
  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_tlsf_pool_create(
      backing->hal_pool, &hal_options, backing->group->runtime->host_allocator,
      &hal_pool);
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_wrap(hal_pool, backing->group,
                               backing->backing_buffer, out_pool);
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_tlsf_pool_create_from_buffer(
    hrx_buffer_t buffer, uint64_t offset, uint64_t length,
    const hrx_tlsf_pool_options_t* options, hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!buffer || !buffer->hal_buffer || !buffer->group || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid backing buffer or options");
  }
  const iree_hal_tlsf_pool_options_t hal_options =
      hrx_tlsf_pool_options_to_hal(options);
  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_tlsf_pool_create_from_buffer(
      buffer->hal_buffer, offset, length, &hal_options,
      buffer->group->runtime->host_allocator, &hal_pool);
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_wrap(hal_pool, buffer->group, buffer, out_pool);
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_fixed_block_pool_query_backing_request(
    hrx_mem_pool_t backing, const hrx_fixed_block_pool_options_t* options,
    hrx_pool_reservation_request_t* out_request) {
  if (!out_request) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_request = (hrx_pool_reservation_request_t){0};
  if (!backing || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "backing or options is NULL");
  }
  const iree_hal_fixed_block_pool_options_t hal_options =
      hrx_fixed_block_pool_options_to_hal(options);
  iree_hal_pool_reservation_request_t request;
  iree_status_t status = iree_hal_fixed_block_pool_query_backing_request(
      backing->hal_pool, &hal_options, &request);
  if (iree_status_is_ok(status)) {
    *out_request = (hrx_pool_reservation_request_t){request.allocation_size,
                                                    request.min_alignment};
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_fixed_block_pool_create(
    hrx_mem_pool_t backing, const hrx_fixed_block_pool_options_t* options,
    hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!backing || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "backing or options is NULL");
  }
  const iree_hal_fixed_block_pool_options_t hal_options =
      hrx_fixed_block_pool_options_to_hal(options);
  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_fixed_block_pool_create(
      backing->hal_pool, &hal_options, backing->group->runtime->host_allocator,
      &hal_pool);
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_wrap(hal_pool, backing->group,
                               backing->backing_buffer, out_pool);
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_fixed_block_pool_create_from_buffer(
    hrx_buffer_t buffer, uint64_t offset, uint64_t length,
    const hrx_fixed_block_pool_options_t* options, hrx_mem_pool_t* out_pool) {
  if (!out_pool) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_pool = NULL;
  if (!buffer || !buffer->hal_buffer || !buffer->group || !options) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid backing buffer or options");
  }
  const iree_hal_fixed_block_pool_options_t hal_options =
      hrx_fixed_block_pool_options_to_hal(options);
  iree_hal_pool_t* hal_pool = NULL;
  iree_status_t status = iree_hal_fixed_block_pool_create_from_buffer(
      buffer->hal_buffer, offset, length, &hal_options,
      buffer->group->runtime->host_allocator, &hal_pool);
  if (iree_status_is_ok(status)) {
    status = hrx_mem_pool_wrap(hal_pool, buffer->group, buffer, out_pool);
  }
  return hrx_status_from_iree(status);
}

void hrx_mem_pool_trim(hrx_mem_pool_t pool, uint64_t min_bytes_to_keep,
                       hrx_pool_trim_flags_t flags) {
  if (pool) {
    iree_hal_pool_trim(pool->hal_pool, flags, min_bytes_to_keep);
  }
}

void hrx_mem_pool_query_capabilities(
    hrx_mem_pool_t pool, hrx_pool_capabilities_t* out_capabilities) {
  iree_hal_pool_capabilities_t value;
  iree_hal_pool_query_capabilities(pool->hal_pool, &value);
  *out_capabilities = (hrx_pool_capabilities_t){
      .placement = {(hrx_pool_placement_mode_t)value.placement.mode,
                    value.placement.node},
      .memory_type = value.memory_type,
      .allowed_access = value.allowed_access,
      .supported_usage = value.supported_usage,
      .min_allocation_size = value.min_allocation_size,
      .max_allocation_size = value.max_allocation_size,
      .max_allocation_alignment = value.max_allocation_alignment,
      .maintenance_alignment = value.maintenance_alignment,
  };
}

hrx_pool_host_access_t hrx_mem_pool_query_host_access(hrx_mem_pool_t pool) {
  iree_hal_pool_host_access_t value =
      iree_hal_pool_query_host_access(pool->hal_pool);
  return (hrx_pool_host_access_t){
      .access = value.access,
      .modes = value.modes,
      .properties = value.properties,
      .atomics = hrx_atomic_from_hal(value.atomics),
  };
}

hrx_atomic_operation_capabilities_t hrx_mem_pool_query_atomic_operations(
    hrx_mem_pool_t pool, hrx_memory_scope_t scope) {
  return hrx_atomic_from_hal(iree_hal_pool_query_atomic_operations(
      pool->hal_pool, (iree_hal_memory_scope_t){scope.domain, scope.id}));
}

hrx_atomic_operation_capabilities_t
hrx_mem_pool_query_zero_compute_atomic_operations(hrx_mem_pool_t pool,
                                                  hrx_memory_scope_t scope) {
  return hrx_atomic_from_hal(iree_hal_pool_query_zero_compute_atomic_operations(
      pool->hal_pool, (iree_hal_memory_scope_t){scope.domain, scope.id}));
}

hrx_atomic_operation_capabilities_t hrx_buffer_query_atomic_operations(
    hrx_buffer_t buffer, hrx_memory_scope_t scope) {
  return hrx_atomic_from_hal(iree_hal_buffer_query_atomic_operations(
      buffer->hal_buffer, (iree_hal_memory_scope_t){scope.domain, scope.id}));
}

hrx_atomic_operation_capabilities_t
hrx_buffer_query_zero_compute_atomic_operations(hrx_buffer_t buffer,
                                                hrx_memory_scope_t scope) {
  return hrx_atomic_from_hal(
      iree_hal_buffer_query_zero_compute_atomic_operations(
          buffer->hal_buffer,
          (iree_hal_memory_scope_t){scope.domain, scope.id}));
}

void hrx_mem_pool_query_stats(hrx_mem_pool_t pool,
                              hrx_pool_stats_t* out_stats) {
  iree_hal_pool_stats_t value;
  iree_hal_pool_query_stats(pool->hal_pool, &value);
  *out_stats = (hrx_pool_stats_t){
      .bytes_reserved = value.bytes_reserved,
      .bytes_free = value.bytes_free,
      .bytes_quarantined = value.bytes_quarantined,
      .bytes_committed = value.bytes_committed,
      .bytes_abandoned = value.bytes_abandoned,
      .budget_limit = value.budget_limit,
      .reservation_count = value.reservation_count,
      .slab_count = value.slab_count,
      .reserve_count = value.reserve_count,
      .release_count = value.release_count,
      .reuse_count = value.reuse_count,
      .reuse_miss_count = value.reuse_miss_count,
      .fresh_count = value.fresh_count,
      .exhausted_count = value.exhausted_count,
      .over_budget_count = value.over_budget_count,
      .wait_count = value.wait_count,
      .quarantine_eviction_count = value.quarantine_eviction_count,
  };
}

hrx_memory_mapping_requirements_t hrx_mem_pool_query_mapping_requirements(
    hrx_mem_pool_t pool) {
  if (!pool) {
    return (hrx_memory_mapping_requirements_t){0};
  }
  const iree_hal_pool_mapping_requirements_t requirements =
      iree_hal_pool_query_mapping_requirements(pool->hal_pool);
  return (hrx_memory_mapping_requirements_t){
      .minimum_granularity = (size_t)requirements.minimum_granularity,
      .recommended_granularity = (size_t)requirements.recommended_granularity,
      .flags = requirements.flags,
  };
}
