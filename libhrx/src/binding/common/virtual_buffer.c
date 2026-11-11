// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/virtual_buffer.h"

#include "iree/hal/api.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"
#include "libhrx/src/libhrx/mem_pool_backing.h"

// Construction and capability queries are cold paths. Submission consumes the
// resulting immutable scope tokens and native owners directly.
static iree_status_t iree_hal_streaming_virtual_buffer_prepare_scope(
    iree_hal_streaming_virtual_buffer_scope_t scope,
    iree_hal_pool_scope_t* out_scope) {
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
      const iree_hal_queue_family_t* family =
          iree_hal_device_queue_family(device->hal_device, j);
      families[family_index++] = (iree_hal_pool_family_access_t){
          .family = family,
          .usage = hrx_mem_pool_family_usage(family),
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

// Acquire metadata from the same owners as later MAPPING_SOURCE allocations.
// Virtual host access does not imply coherent host-owned physical backing.
static iree_status_t iree_hal_streaming_virtual_buffer_prepare_sources(
    iree_hal_device_group_t* group,
    iree_hal_streaming_virtual_buffer_sources_t sources,
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
    const iree_hal_streaming_virtual_buffer_source_t* source =
        &sources.sources[i];
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

static void iree_hal_streaming_virtual_buffer_release_sources(
    size_t count, iree_hal_pool_t** pools) {
  if (!pools) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    iree_hal_pool_release(pools[i]);
  }
  iree_allocator_free(iree_allocator_system(), pools);
}

iree_status_t iree_hal_streaming_virtual_buffer_query_capabilities(
    iree_hal_streaming_virtual_buffer_scope_t scope,
    iree_hal_streaming_virtual_buffer_sources_t sources,
    hrx_virtual_buffer_capabilities_t* out_capabilities) {
  if (!out_capabilities) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation capabilities output is NULL");
  }
  *out_capabilities = (hrx_virtual_buffer_capabilities_t){0};
  iree_hal_pool_scope_t hal_scope;
  iree_status_t status =
      iree_hal_streaming_virtual_buffer_prepare_scope(scope, &hal_scope);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_hal_pool_t** source_pools = NULL;
  status = iree_hal_streaming_virtual_buffer_prepare_sources(
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
  iree_hal_streaming_virtual_buffer_release_sources(sources.source_count,
                                                    source_pools);
  iree_allocator_free(iree_allocator_system(), (void*)hal_scope.families);
  return status;
}

iree_status_t iree_hal_streaming_virtual_buffer_create(
    iree_hal_streaming_virtual_buffer_scope_t scope,
    iree_hal_streaming_virtual_buffer_sources_t sources,
    const hrx_virtual_buffer_options_t* options, hrx_buffer_t* out_buffer) {
  if (!out_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation output is NULL");
  }
  *out_buffer = NULL;
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation options are NULL");
  }
  iree_hal_pool_scope_t hal_scope;
  iree_status_t status =
      iree_hal_streaming_virtual_buffer_prepare_scope(scope, &hal_scope);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_hal_pool_t** source_pools = NULL;
  status = iree_hal_streaming_virtual_buffer_prepare_sources(
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
    status = hrx_buffer_create_from_hal(buffer, scope.devices[0]->group,
                                        HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                                        options->byte_length, NULL, out_buffer);
  }
  iree_hal_streaming_virtual_buffer_release_sources(sources.source_count,
                                                    source_pools);
  iree_allocator_free(iree_allocator_system(), (void*)hal_scope.families);
  iree_hal_buffer_release(buffer);
  return status;
}

iree_status_t iree_hal_streaming_virtual_buffer_query_source_requirements(
    hrx_device_t device, hrx_memory_type_t memory_type,
    hrx_memory_mapping_requirements_t* out_requirements) {
  if (!device || !out_requirements) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mapping device or requirements output is NULL");
  }
  *out_requirements = (hrx_memory_mapping_requirements_t){0};
  iree_hal_pool_t* pool = NULL;
  iree_status_t status = hrx_mem_pool_backing_retain_allocation_source(
      device, memory_type, HRX_BUFFER_USAGE_MAPPING_SOURCE, &pool);
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
  return status;
}
