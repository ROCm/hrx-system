// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"

IREE_STATIC_ASSERT_ENUM_EQ(HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
                           IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
                           "virtual buffer address constraints must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_HOST, IREE_HAL_MEMORY_SITE_HOST,
                           "memory sites must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_QUEUE,
                           "memory sites must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_PROGRAM,
                           IREE_HAL_MEMORY_SITE_PROGRAM,
                           "memory sites must match HAL");
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

// Construction and capability queries are cold paths. Submission consumes the
// resulting immutable scope tokens and native owners directly.
static iree_status_t hrx_virtual_buffer_prepare_scope(
    hrx_virtual_buffer_scope_t scope, iree_hal_pool_scope_t* out_scope) {
  *out_scope = (iree_hal_pool_scope_t){0};
  if (!scope.device_count || !scope.devices || !scope.devices[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation scope requires at least one device");
  }
  iree_hal_device_group_t* group = scope.devices[0]->hal_device_group;
  iree_host_size_t family_count = 0;
  for (size_t i = 0; i < scope.device_count; ++i) {
    hrx_device_t device = scope.devices[i];
    if (!device || device->hal_device_group != group) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation devices must share one sealed group");
    }
    const iree_hal_device_queue_spec_t* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device->hal_device));
    if (!iree_host_size_checked_add(family_count, queues->family_count,
                                    &family_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "reservation family count overflows");
    }
  }
  iree_hal_pool_family_access_t* families = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(iree_allocator_system(), family_count,
                                  sizeof(*families), (void**)&families));
  iree_host_size_t family_index = 0;
  for (size_t i = 0; i < scope.device_count; ++i) {
    hrx_device_t device = scope.devices[i];
    const iree_hal_device_queue_spec_t* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device->hal_device));
    const iree_hal_buffer_interface_t binding_type =
        device->type == HRX_ACCELERATOR_CPU
            ? IREE_HAL_BUFFER_INTERFACE_HOST
            : IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    for (iree_host_size_t j = 0; j < queues->family_count; ++j) {
      families[family_index++] = (iree_hal_pool_family_access_t){
          .family = iree_hal_device_queue_family(device->hal_device, j),
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
          .interfaces = UINT64_C(1) << binding_type,
      };
    }
  }
  *out_scope = (iree_hal_pool_scope_t){
      .family_count = family_count,
      .families = families,
      .host =
          {
              .access = scope.host_access,
              .modes = scope.host_access ? IREE_HAL_MAPPING_MODE_SCOPED |
                                               IREE_HAL_MAPPING_MODE_PERSISTENT
                                         : 0,
          },
  };
  return iree_ok_status();
}

hrx_status_t hrx_device_query_memory_scope(hrx_device_t device,
                                           hrx_queue_affinity_t affinity,
                                           hrx_memory_site_t site,
                                           hrx_memory_scope_t* out_scope) {
  if (!device || !out_scope || site < HRX_MEMORY_SITE_HOST ||
      site > HRX_MEMORY_SITE_PROGRAM) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid memory scope device, site or output");
  }
  *out_scope = (hrx_memory_scope_t){0};
  iree_hal_memory_site_t hal_site = {.kind = (iree_hal_memory_site_kind_t)site};
  if (site != HRX_MEMORY_SITE_HOST) {
    iree_hal_queue_t* queue = NULL;
    const iree_hal_queue_family_role_flags_t roles =
        site == HRX_MEMORY_SITE_PROGRAM
            ? IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH
            : IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
    iree_status_t status = hrx_hal_device_select_queue(
        device->hal_device, affinity, roles, NULL, &queue);
    if (!iree_status_is_ok(status)) {
      return hrx_status_from_iree(status);
    }
    hal_site.family = iree_hal_queue_family(queue);
  }
  iree_hal_memory_scope_t hal_scope;
  iree_status_t status = iree_hal_device_group_resolve_memory_scope(
      device->hal_device_group, hal_site, &hal_scope);
  if (iree_status_is_ok(status)) {
    *out_scope = (hrx_memory_scope_t){hal_scope.domain, hal_scope.id};
  }
  return hrx_status_from_iree(status);
}

// Acquire metadata from the same owners as later MAPPING_SOURCE allocations.
// Virtual host access does not imply coherent host-owned physical backing.
static iree_status_t hrx_virtual_buffer_prepare_sources(
    iree_hal_device_group_t* group, hrx_virtual_buffer_sources_t sources,
    iree_hal_pool_t*** out_pools) {
  *out_pools = NULL;
  if (!sources.source_count || !sources.sources) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation requires accepted backing classes");
  }
  iree_hal_pool_t** pools = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(iree_allocator_system(), sources.source_count,
                                  sizeof(*pools), (void**)&pools));
  iree_status_t status = iree_ok_status();
  size_t acquired_count = 0;
  for (size_t i = 0; i < sources.source_count && iree_status_is_ok(status);
       ++i) {
    const hrx_virtual_buffer_source_t* source = &sources.sources[i];
    if (!source->device || source->device->hal_device_group != group) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation source must belong to the same sealed group");
    } else {
      status = hrx_mem_pool_backing_retain_allocation_source(
          source->device, source->memory_type, HRX_BUFFER_USAGE_MAPPING_SOURCE,
          &pools[acquired_count]);
      if (iree_status_is_ok(status)) {
        ++acquired_count;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_pools = pools;
  } else {
    for (size_t i = 0; i < acquired_count; ++i) {
      iree_hal_pool_release(pools[i]);
    }
    iree_allocator_free(iree_allocator_system(), pools);
  }
  return status;
}

static void hrx_virtual_buffer_release_sources(size_t count,
                                               iree_hal_pool_t** pools) {
  if (!pools) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    iree_hal_pool_release(pools[i]);
  }
  iree_allocator_free(iree_allocator_system(), pools);
}

hrx_status_t hrx_virtual_buffer_query_capabilities(
    hrx_virtual_buffer_scope_t scope, hrx_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_capabilities_t* out_capabilities) {
  if (!out_capabilities) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "reservation capabilities output is NULL");
  }
  *out_capabilities = (hrx_virtual_buffer_capabilities_t){0};
  iree_hal_pool_scope_t hal_scope;
  iree_status_t status = hrx_virtual_buffer_prepare_scope(scope, &hal_scope);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  iree_hal_pool_t** source_pools = NULL;
  status = hrx_virtual_buffer_prepare_sources(
      scope.devices[0]->hal_device_group, sources, &source_pools);
  iree_hal_virtual_buffer_capabilities_t capabilities = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_virtual_buffer_query_capabilities(
        scope.devices[0]->hal_device_group, hal_scope,
        (iree_hal_virtual_buffer_sources_t){sources.source_count, source_pools},
        iree_allocator_system(), &capabilities);
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
  hrx_virtual_buffer_release_sources(sources.source_count, source_pools);
  iree_allocator_free(iree_allocator_system(), (void*)hal_scope.families);
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_virtual_buffer_create(
    hrx_virtual_buffer_scope_t scope, hrx_virtual_buffer_sources_t sources,
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
  iree_hal_pool_scope_t hal_scope;
  iree_status_t status = hrx_virtual_buffer_prepare_scope(scope, &hal_scope);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  iree_hal_pool_t** source_pools = NULL;
  status = hrx_virtual_buffer_prepare_sources(
      scope.devices[0]->hal_device_group, sources, &source_pools);
  iree_hal_buffer_t* buffer = NULL;
  const iree_hal_virtual_buffer_options_t hal_options = {
      .byte_length = options->byte_length,
      .min_alignment = options->min_alignment,
      .mapping_granularity = options->mapping_granularity,
      .flags = options->flags,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_virtual_buffer_create(
        scope.devices[0]->hal_device_group, hal_scope,
        (iree_hal_virtual_buffer_sources_t){sources.source_count, source_pools},
        &hal_options, iree_allocator_system(), &buffer);
  }
  if (iree_status_is_ok(status)) {
    hrx_device_t device = scope.devices[0];
    // Construction required this interface for every family. The canonical
    // program coordinate and prepared slot are immutable producer facts.
    const iree_hal_queue_family_t* family =
        iree_hal_queue_family(device->dispatch_queue);
    const iree_hal_buffer_interface_t binding_type =
        device->type == HRX_ACCELERATOR_CPU
            ? IREE_HAL_BUFFER_INTERFACE_HOST
            : IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    const iree_hal_buffer_native_binding_slot_t slot = {
        .index =
            buffer->memory.contract->scopes[family->memory.queue_scope_id + 1]
                .bindings[binding_type],
        .type = binding_type,
    };
    const iree_hal_buffer_native_binding_t binding =
        iree_hal_buffer_native_binding(buffer, slot);
    void* pointer = device->type == HRX_ACCELERATOR_CPU
                        ? binding.host_pointer
                        : (void*)(uintptr_t)binding.device_address;
    status = hrx_buffer_create_from_hal(
        buffer, device, HRX_MEMORY_TYPE_DEVICE_VISIBLE, options->byte_length,
        pointer, out_buffer);
  }
  hrx_virtual_buffer_release_sources(sources.source_count, source_pools);
  iree_allocator_free(iree_allocator_system(), (void*)hal_scope.families);
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

hrx_status_t hrx_allocator_query_mapping_requirements(
    hrx_allocator_t allocator, hrx_memory_type_t memory_type,
    hrx_memory_mapping_requirements_t* out_requirements) {
  if (!allocator || !out_requirements) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "mapping allocator or requirements output is NULL");
  }
  *out_requirements = (hrx_memory_mapping_requirements_t){0};
  iree_hal_pool_t* pool = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      allocator->device, memory_type, HRX_BUFFER_USAGE_MAPPING_SOURCE, &pool);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_mapping_requirements_t requirements =
        iree_hal_pool_query_mapping_requirements(pool);
    *out_requirements = (hrx_memory_mapping_requirements_t){
        .minimum_granularity = (size_t)requirements.minimum_granularity,
        .recommended_granularity = (size_t)requirements.recommended_granularity,
        .flags = requirements.flags,
    };
  }
  iree_hal_pool_release(pool);
  return hrx_status_from_iree(status);
}
