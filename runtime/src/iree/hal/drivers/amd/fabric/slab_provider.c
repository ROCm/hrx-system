// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/slab_provider.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/fabric/asan.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/tracing.h"

typedef struct iree_hal_amd_fabric_slab_provider_t {
  // Base slab-provider interface header.
  iree_hal_slab_provider_t base;
  // Retained HAL device owning placement, scope, and progress services.
  iree_hal_device_t* device;
  // Native allocation domain shared by all engines on this NUMA node.
  iree_hal_amd_fabric_allocation_queue_t* allocation_queue;
  // Borrowed native core API, dominated by retained device owners.
  const amdf_api_t* api;
  // Borrowed diagnostic destination for native cleanup failures.
  iree_hal_device_event_sink_t event_sink;
  // Qualified construction contract with provider-owned array storage.
  iree_hal_amd_fabric_slab_provider_options_t options;
  // Immutable HAL capabilities of all slabs.
  iree_hal_slab_provider_properties_t properties;
  // Physical backing ledger shared across providers in this domain.
  iree_hal_memory_trace_t trace;
  // Whether slabs need one shared prepared source descriptor.
  bool has_mapping_source;
  // Native access ordinal of the optional ASAN namespace's device.
  uint32_t asan_access_ordinal;
  // Cumulative successful slab acquisitions.
  iree_atomic_int64_t total_acquired;
  // Cumulative slab release attempts.
  iree_atomic_int64_t total_released;
} iree_hal_amd_fabric_slab_provider_t;

typedef struct iree_hal_amd_fabric_slab_handle_t {
  // Native allocation and optional persistent host mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Prepared addresses in the contract layout order.
  iree_hal_buffer_native_binding_t* bindings;
  // Prepared backing identity; absent for ordinary address-only slabs.
  iree_hal_amd_fabric_mapping_source_t* mapping_source;
  // Explicit external attachment whose teardown completes before release.
  bool imported;
} iree_hal_amd_fabric_slab_handle_t;

// Present immediately after the handle only for sanitized providers.
typedef struct iree_hal_amd_fabric_slab_asan_t {
  // Retained provider after ownership transfers to the quarantine FIFO.
  iree_hal_amd_fabric_slab_provider_t* provider;
  // Owned compact application interval, or NULL for native/imported addresses.
  iree_hal_amd_fabric_asan_application_range_t* application_range;
  // Device execution address used by this namespace's instrumented shaders.
  uint64_t address;
  // Imported attachments retain their caller-managed lifetime contract.
  bool imported;
  // Whether a pool has published allocation/release shadow for this slab.
  iree_atomic_int32_t has_advice;
  // Intrusive FIFO link requiring no allocation at slab release.
  iree_hal_amd_fabric_asan_quarantine_entry_t quarantine;
} iree_hal_amd_fabric_slab_asan_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_amd_fabric_slab_provider_vtable;

static iree_hal_amd_fabric_slab_provider_t*
iree_hal_amd_fabric_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_amd_fabric_slab_provider_t*)base_provider;
}

static const iree_hal_amd_fabric_slab_provider_t*
iree_hal_amd_fabric_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_amd_fabric_slab_provider_t*)base_provider;
}

static iree_hal_pool_export_classes_t iree_hal_amd_fabric_copy_exports(
    iree_hal_pool_export_classes_t source,
    iree_hal_external_storage_class_t** cursor) {
  iree_hal_pool_export_classes_t result = {.class_count = source.class_count};
  if (source.class_count) {
    result.classes = *cursor;
    memcpy(*cursor, source.classes, source.class_count * sizeof(**cursor));
    *cursor += source.class_count;
  }
  return result;
}

iree_status_t iree_hal_amd_fabric_slab_provider_create(
    iree_hal_device_t* device, const amdf_api_t* api,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_allocation_queue_t* allocation_queue,
    const iree_hal_amd_fabric_slab_provider_options_t* options,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t accesses_offset = 0;
  iree_host_size_t access_devices_offset = 0;
  iree_host_size_t import_accesses_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t imports_offset = 0;
  iree_host_size_t exports_offset = 0;
  iree_host_size_t export_count = options->memory.exports.class_count;
  for (iree_host_size_t i = 0; i < options->import_count; ++i) {
    if (!iree_host_size_checked_add(
            export_count, options->imports[i].policy.exports.class_count,
            &export_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMD export class count overflows host size");
    }
  }
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_fabric_slab_provider_t), &total_size,
      IREE_STRUCT_FIELD(options->memory.access_count, iree_hal_device_t*,
                        &access_devices_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          options->memory.access_count, amdf_memory_device_access_t,
          iree_alignof(amdf_memory_device_access_t), &accesses_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          options->import_accesses ? options->memory.access_count : 0,
          amdf_memory_device_access_t,
          iree_alignof(amdf_memory_device_access_t), &import_accesses_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          options->binding_count, iree_hal_amd_fabric_binding_request_t,
          iree_alignof(iree_hal_amd_fabric_binding_request_t),
          &bindings_offset),
      IREE_STRUCT_FIELD(options->import_count,
                        iree_hal_amd_fabric_import_route_t, &imports_offset),
      IREE_STRUCT_FIELD(export_count, iree_hal_external_storage_class_t,
                        &exports_offset)));
  iree_hal_amd_fabric_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&provider));
  memset(provider, 0, total_size);
  iree_hal_slab_provider_initialize(&iree_hal_amd_fabric_slab_provider_vtable,
                                    host_allocator, &provider->base);
  provider->device = device;
  iree_hal_device_retain(device);
  provider->allocation_queue = allocation_queue;
  iree_hal_amd_fabric_allocation_queue_retain(allocation_queue);
  provider->api = api;
  provider->event_sink = event_sink;
  provider->options = *options;
  iree_hal_device_t** access_devices =
      (iree_hal_device_t**)((uint8_t*)provider + access_devices_offset);
  for (uint32_t i = 0; i < options->memory.access_count; ++i) {
    access_devices[i] = options->access_devices[i];
    iree_hal_device_retain(access_devices[i]);
  }
  provider->options.access_devices = access_devices;
  amdf_memory_device_access_t* accesses =
      (amdf_memory_device_access_t*)((uint8_t*)provider + accesses_offset);
  memcpy(accesses, options->memory.accesses,
         options->memory.access_count * sizeof(*accesses));
  provider->options.memory.accesses = accesses;
  if (options->import_accesses) {
    amdf_memory_device_access_t* import_accesses =
        (void*)((uint8_t*)provider + import_accesses_offset);
    memcpy(import_accesses, options->import_accesses,
           options->memory.access_count * sizeof(*import_accesses));
    provider->options.import_accesses = import_accesses;
  }
  if (options->asan) {
    for (uint32_t i = 0; i < options->memory.access_count; ++i) {
      if (accesses[i].device == options->asan->device) {
        provider->asan_access_ordinal = i;
        break;
      }
    }
  }
  iree_hal_amd_fabric_binding_request_t* bindings =
      (iree_hal_amd_fabric_binding_request_t*)((uint8_t*)provider +
                                               bindings_offset);
  memcpy(bindings, options->bindings,
         options->binding_count * sizeof(*bindings));
  provider->options.bindings = bindings;
  for (uint32_t i = 0; i < options->binding_count; ++i) {
    provider->has_mapping_source |=
        bindings[i].type == IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  iree_hal_external_storage_class_t* exports =
      (iree_hal_external_storage_class_t*)((uint8_t*)provider + exports_offset);
  provider->options.memory.exports =
      iree_hal_amd_fabric_copy_exports(options->memory.exports, &exports);
  iree_hal_amd_fabric_import_route_t* imports =
      (iree_hal_amd_fabric_import_route_t*)((uint8_t*)provider +
                                            imports_offset);
  for (iree_host_size_t i = 0; i < options->import_count; ++i) {
    imports[i] = options->imports[i];
    imports[i].policy.exports = iree_hal_amd_fabric_copy_exports(
        options->imports[i].policy.exports, &exports);
  }
  provider->options.imports = imports;
  const bool owned = iree_all_bits_set(options->memory.profile.roles,
                                       AMDF_MEMORY_PROFILE_ROLE_CREATE);
  provider->properties = (iree_hal_slab_provider_properties_t){
      .memory_type = options->memory_type,
      .supported_usage = options->supported_usage,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .allocation_alignment =
          owned ? options->memory.profile.allocation.minimum_alignment : 1,
      .max_allocation_alignment =
          owned ? options->memory.profile.allocation.maximum_alignment : 1,
      .maintenance_alignment = options->maintenance_alignment,
  };
  iree_status_t status = iree_hal_memory_trace_initialize_domain(
      options->trace_domain ? options->trace_domain : "libamdf",
      options->trace_domain ? options->trace_ordinal : UINT32_MAX,
      &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }
  return status;
}

static void iree_hal_amd_fabric_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  for (uint32_t i = 0; i < provider->options.memory.access_count; ++i) {
    iree_hal_device_release(provider->options.access_devices[i]);
  }
  iree_hal_device_t* device = provider->device;
  iree_hal_amd_fabric_allocation_queue_t* queue = provider->allocation_queue;
  iree_allocator_free(provider->base.host_allocator, provider);
  iree_hal_amd_fabric_allocation_queue_release(queue);
  iree_hal_device_release(device);
}

static iree_status_t iree_hal_amd_fabric_slab_handle_allocate(
    iree_hal_amd_fabric_slab_provider_t* provider,
    iree_hal_amd_fabric_slab_handle_t** out_handle) {
  const uint32_t binding_count = provider->options.binding_count;
  iree_host_size_t total_size = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t mapping_source_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_fabric_slab_handle_t) +
          (provider->options.asan ? sizeof(iree_hal_amd_fabric_slab_asan_t)
                                  : 0),
      &total_size,
      IREE_STRUCT_FIELD_ALIGNED(binding_count, iree_hal_buffer_native_binding_t,
                                iree_alignof(iree_hal_buffer_native_binding_t),
                                &bindings_offset),
      IREE_STRUCT_FIELD(provider->has_mapping_source ? 1 : 0,
                        iree_hal_amd_fabric_mapping_source_t,
                        &mapping_source_offset)));
  iree_hal_amd_fabric_slab_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(provider->base.host_allocator,
                                             total_size, (void**)&handle));
  memset(handle, 0, total_size);
  handle->bindings =
      (iree_hal_buffer_native_binding_t*)((uint8_t*)handle + bindings_offset);
  if (provider->has_mapping_source) {
    handle->mapping_source =
        (iree_hal_amd_fabric_mapping_source_t*)((uint8_t*)handle +
                                                mapping_source_offset);
  }
  *out_handle = handle;
  return iree_ok_status();
}

static bool iree_hal_amd_fabric_slab_handle_free(
    iree_hal_amd_fabric_slab_provider_t* provider,
    iree_hal_amd_fabric_slab_handle_t* handle) {
  if (provider->options.asan) {
    iree_hal_amd_fabric_slab_asan_t* asan = (void*)(handle + 1);
    if (!iree_hal_amd_fabric_asan_unmap_application(provider->options.asan,
                                                    asan->application_range)) {
      return false;
    }
  }
  const bool owns_memory = handle->memory.handle && !handle->imported;
  iree_hal_amd_fabric_memory_deinitialize(provider->api, provider->event_sink,
                                        &handle->memory);
  if (owns_memory) {
    iree_hal_memory_trace_free(&provider->trace, handle);
  }
  iree_allocator_free(provider->base.host_allocator, handle);
  return true;
}

static iree_status_t iree_hal_amd_fabric_slab_prepare_asan(
    iree_hal_amd_fabric_slab_provider_t* provider,
    iree_hal_amd_fabric_slab_handle_t* handle, bool imported,
    uint64_t imported_length) {
  if (!provider->options.asan) {
    return iree_ok_status();
  }
  iree_hal_amd_fabric_asan_t* state = provider->options.asan;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      (amdf_status_t)iree_atomic_load(&state->failure,
                                      iree_memory_order_acquire),
      "ASAN shadow publication"));
  iree_hal_amd_fabric_slab_asan_t* asan = (void*)(handle + 1);
  asan->imported = imported;
  if (!imported && provider->options.asan_application_mapping) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_map_application(
        state, handle->memory.handle, provider->asan_access_ordinal,
        handle->memory.info.byte_length, handle->memory.info.alignment,
        &asan->application_range));
    asan->address = asan->application_range->address;
  } else {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        provider->api->memory_query_address(
            handle->memory.handle, provider->asan_access_ordinal,
            AMDF_MEMORY_ADDRESS_GPU, &asan->address),
        "ASAN application address"));
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_asan_map_range(
      state, asan->address, handle->memory.info.byte_length));
  return imported ? iree_hal_amd_fabric_asan_publish_imported_range(
                        state, asan->address, imported_length)
                  : iree_ok_status();
}

static iree_status_t iree_hal_amd_fabric_slab_handle_bind(
    iree_hal_amd_fabric_slab_provider_t* provider,
    iree_hal_amd_fabric_slab_handle_t* handle) {
  iree_status_t status = iree_ok_status();
  if (provider->has_mapping_source) {
    *handle->mapping_source = (iree_hal_amd_fabric_mapping_source_t){
        .memory = handle->memory.handle,
        .access_count = provider->options.memory.access_count,
        .accesses = provider->options.memory.accesses,
        .logical_byte_offset = handle->memory.info.source_byte_offset,
    };
    if (provider->options.import_accesses) {
      const iree_hal_amd_fabric_slab_asan_t* asan = (const void*)(handle + 1);
      if (asan->imported) {
        handle->mapping_source->accesses = provider->options.import_accesses;
      }
    }
  }
  for (uint32_t i = 0;
       i < provider->options.binding_count && iree_status_is_ok(status); ++i) {
    if (i == provider->options.host_binding_index) {
      handle->bindings[i].host_pointer = handle->memory.contents.data;
    } else {
      const iree_hal_amd_fabric_binding_request_t* binding =
          &provider->options.bindings[i];
      if (binding->type == IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE) {
        handle->bindings[i].mapping_source.descriptor = handle->mapping_source;
        handle->bindings[i].mapping_source.granularity =
            binding->mapping_granularity;
        handle->bindings[i].mapping_source.offset =
            handle->mapping_source->logical_byte_offset;
        continue;
      }
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          provider->api->memory_query_address(
              handle->memory.handle, binding->access_ordinal,
              binding->address_kind, &handle->bindings[i].device_address),
          "memory_query_address");
      if (iree_status_is_ok(status) && provider->options.asan &&
          binding->access_ordinal == provider->asan_access_ordinal &&
          binding->address_kind == AMDF_MEMORY_ADDRESS_GPU) {
        const iree_hal_amd_fabric_slab_asan_t* asan = (const void*)(handle + 1);
        handle->bindings[i].device_address = asan->address;
      }
    }
  }
  return status;
}

// Publishes achieved allocation facts without widening the requested slab view.
static void iree_hal_amd_fabric_slab_initialize(
    iree_hal_amd_fabric_slab_handle_t* handle, iree_device_size_t length,
    iree_hal_slab_t* out_slab) {
  const iree_hal_amd_fabric_memory_t* memory = &handle->memory;
  *out_slab = (iree_hal_slab_t){
      .base_ptr = memory->contents.data,
      .length = length,
      .provider_handle = (uint64_t)(uintptr_t)handle,
      .native_allocation =
          {
              .byte_length = memory->info.native_allocation_byte_length,
              .byte_offset = memory->info.source_byte_offset,
              .granularity = memory->info.native_allocation_granularity,
          },
      .maintenance_range = memory->maintenance_range,
      .host_cacheability = memory->host_cacheability,
      .physical_backing_id = {{memory->info.physical_backing_id.words[0],
                               memory->info.physical_backing_id.words[1]}},
      .allocation_alignment = memory->info.alignment,
  };
}

static iree_status_t iree_hal_amd_fabric_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  memset(out_slab, 0, sizeof(*out_slab));
  if (!min_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMD slab allocations must be non-empty");
  }
  if (!iree_all_bits_set(provider->options.memory.profile.roles,
                         AMDF_MEMORY_PROFILE_ROLE_CREATE)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMD provider admits imported storage only");
  }
  iree_hal_amd_fabric_slab_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_slab_handle_allocate(provider, &handle));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      provider->api, &provider->options.memory, min_length, min_alignment,
      provider->event_sink, &handle->memory);
  if (iree_status_is_ok(status)) {
    iree_hal_memory_trace_alloc(
        &provider->trace, handle,
        handle->memory.info.native_allocation_byte_length);
    status = iree_hal_amd_fabric_slab_prepare_asan(provider, handle, false, 0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_slab_handle_bind(provider, handle);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_fabric_slab_initialize(handle, min_length, out_slab);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    if (!iree_hal_amd_fabric_slab_handle_free(provider, handle)) {
      // Failed native removal leaves backing borrowing the provider's facts.
      iree_hal_slab_provider_retain(&provider->base);
    }
  }
  return status;
}

static const iree_hal_amd_fabric_import_route_t*
iree_hal_amd_fabric_slab_provider_find_import(
    const iree_hal_amd_fabric_slab_provider_t* provider,
    iree_hal_external_storage_class_t storage_class) {
  for (iree_host_size_t i = 0; i < provider->options.import_count; ++i) {
    const iree_hal_amd_fabric_import_route_t* route =
        &provider->options.imports[i];
    if (iree_hal_external_storage_class_equal(route->policy.storage_class,
                                              storage_class)) {
      return route;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_amd_fabric_slab_provider_import_slab(
    iree_hal_slab_provider_t* base_provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  const iree_hal_amd_fabric_import_route_t* route =
      iree_hal_amd_fabric_slab_provider_find_import(provider,
                                                    storage->storage_class);
  if (!route) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "external storage class was not admitted");
  }
  iree_hal_external_storage_t registration;
  iree_device_size_t storage_offset = 0;
  if (storage->storage_class.type ==
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    storage_offset = storage->maintenance_range.byte_offset;
    registration = iree_hal_slab_provider_registration_view(storage);
    storage = &registration;
  }
  iree_hal_amd_fabric_slab_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_slab_handle_allocate(provider, &handle));
  handle->imported = true;
  iree_hal_amd_fabric_memory_options_t options = provider->options.memory;
  if (provider->options.import_accesses) {
    options.accesses = provider->options.import_accesses;
  }
  options.profile = route->profile;
  options.required_flags = route->required_flags;
  options.exports = route->policy.exports;
  iree_status_t status =
      storage->storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER
          ? iree_hal_amd_fabric_memory_register(provider->api, &options,
                                                storage, provider->event_sink,
                                                &handle->memory)
          : iree_hal_amd_fabric_memory_import(provider->api, &options, storage,
                                              provider->event_sink,
                                              &handle->memory);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_slab_prepare_asan(provider, handle, true,
                                                   storage->byte_length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_slab_handle_bind(provider, handle);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_fabric_slab_initialize(handle, handle->memory.info.byte_length,
                                        &out_import->slab);
    out_import->storage_offset = storage_offset;
    // Registration can establish a page cover without deriving a physical
    // identity. Preserve an identity already qualified by the storage source.
    if (!out_import->slab.physical_backing_id.words[0] &&
        !out_import->slab.physical_backing_id.words[1]) {
      out_import->slab.physical_backing_id = storage->physical_backing_id;
    }
    out_import->properties = provider->properties;
    out_import->properties.allocation_alignment = handle->memory.info.alignment;
    out_import->properties.max_allocation_alignment =
        handle->memory.info.alignment;
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    if (!iree_hal_amd_fabric_slab_handle_free(provider, handle)) {
      // Failed native removal leaves backing borrowing the provider's facts.
      iree_hal_slab_provider_retain(&provider->base);
    }
  }
  return status;
}

static bool iree_hal_amd_fabric_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_amd_fabric_import_route_t* route =
      iree_hal_amd_fabric_slab_provider_find_import(
          iree_hal_amd_fabric_slab_provider_const_cast(base_provider),
          storage_class);
  if (!route) {
    return false;
  }
  const bool registered =
      storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER;
  const amdf_memory_construction_capabilities_t* capabilities =
      registered ? &route->profile.registration : &route->profile.import;
  *out_requirements = (iree_hal_pool_import_requirements_t){
      .min_byte_length = capabilities->byte_length_granularity,
      .max_byte_length = capabilities->maximum_byte_length,
      .base_alignment =
          registered ? capabilities->registered_host_pointer_alignment : 1,
      .byte_length_granularity = capabilities->byte_length_granularity,
  };
  return true;
}

static void iree_hal_amd_fabric_slab_quarantine_release(void* user_data) {
  iree_hal_amd_fabric_slab_handle_t* handle = user_data;
  iree_hal_amd_fabric_slab_asan_t* asan = (void*)(handle + 1);
  iree_hal_amd_fabric_slab_provider_t* provider = asan->provider;
  if (iree_hal_amd_fabric_slab_handle_free(provider, handle)) {
    iree_hal_slab_provider_release(&provider->base);
  }
}

static void iree_hal_amd_fabric_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  iree_hal_amd_fabric_slab_handle_t* handle =
      (iree_hal_amd_fabric_slab_handle_t*)(uintptr_t)slab->provider_handle;
  if (!handle) {
    return;
  }
  iree_atomic_fetch_add(&provider->total_released, 1,
                        iree_memory_order_relaxed);
  if (provider->options.asan) {
    iree_hal_amd_fabric_slab_asan_t* asan = (void*)(handle + 1);
    if (!asan->imported &&
        iree_atomic_load(&asan->has_advice, iree_memory_order_relaxed)) {
      asan->provider = provider;
      iree_hal_slab_provider_retain(base_provider);
      iree_hal_amd_fabric_asan_quarantine(
          provider->options.asan, &asan->quarantine,
          handle->memory.info.byte_length,
          iree_hal_amd_fabric_slab_quarantine_release, handle);
      return;
    }
  }
  if (!iree_hal_amd_fabric_slab_handle_free(provider, handle)) {
    iree_hal_slab_provider_retain(base_provider);
  }
}

static iree_status_t iree_hal_amd_fabric_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  iree_hal_amd_fabric_slab_handle_t* handle =
      (iree_hal_amd_fabric_slab_handle_t*)(uintptr_t)slab->provider_handle;
  const iree_hal_buffer_placement_t placement = {
      .device = provider->device,
      .queue_family_affinity = provider->properties.queue_family_affinity,
  };
  return iree_hal_amd_fabric_buffer_wrap(
      placement, provider->api,
      (iree_hal_amd_fabric_buffer_storage_t){
          .memory = &handle->memory,
          .bindings = handle->bindings,
          .host_binding_index = provider->options.host_binding_index,
          .offset = slab_offset,
      },
      allocation_size, provider->properties.memory_type, allowed_access,
      allowed_usage, release_callback, provider->base.host_allocator,
      out_buffer);
}

static iree_status_t iree_hal_amd_fabric_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_const_cast(base_provider);
  if (!provider->options.asan) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMD provider has no ASAN shadow namespace");
  }
  if (options->shadow_granule_size !=
      provider->options.asan->pool_options.shadow_granule_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ASAN pool and device shadow granules differ");
  }
  return iree_ok_status();
}

static void iree_hal_amd_fabric_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base_provider);
  iree_hal_amd_fabric_slab_handle_t* handle =
      (void*)(uintptr_t)slab->provider_handle;
  iree_hal_amd_fabric_slab_asan_t* asan = (void*)(handle + 1);
  iree_atomic_store(&asan->has_advice, 1, iree_memory_order_relaxed);
  switch (advice_flags) {
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED:
      iree_hal_amd_fabric_asan_advise_range(
          provider->options.asan, asan->address + backing_offset,
          layout->backing_length, layout->user_offset, layout->user_length);
      break;
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED:
      iree_hal_amd_fabric_asan_advise_range(provider->options.asan,
                                            asan->address + backing_offset,
                                            layout->backing_length, 0, 0);
      break;
    default:
      IREE_CHECK_UNREACHABLE("invalid ASAN range advice flags");
  }
}

static void iree_hal_amd_fabric_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  (void)base_provider;
  (void)slab;
  (void)offset;
  (void)length;
}

static void iree_hal_amd_fabric_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_amd_fabric_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_amd_fabric_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_const_cast(base_provider);
  *out_properties = provider->properties;
}

static void iree_hal_amd_fabric_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_acquire_request_t* request) {
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base);
  iree_hal_amd_fabric_allocation_queue_enqueue(provider->allocation_queue,
                                               &request->operation);
}

static void iree_hal_amd_fabric_slab_provider_release_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_release_request_t* request) {
  // Explicit imports can borrow caller-owned payloads without any release
  // callback. Their last buffer release is therefore the native detach
  // boundary. Only owned native frees remain on the allocation queue.
  iree_host_size_t owned_count = 0;
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    const iree_hal_amd_fabric_slab_handle_t* handle =
        (const iree_hal_amd_fabric_slab_handle_t*)(uintptr_t)request->slabs[i]
            .provider_handle;
    if (handle && handle->imported) {
      iree_hal_amd_fabric_slab_provider_release_slab(base, &request->slabs[i]);
    } else {
      if (owned_count != i) {
        request->slabs[owned_count] = request->slabs[i];
      }
      ++owned_count;
    }
  }
  request->count = owned_count;
  if (!owned_count) {
    request->completion.fn(request->completion.user_data);
    return;
  }
  iree_hal_amd_fabric_slab_provider_t* provider =
      iree_hal_amd_fabric_slab_provider_cast(base);
  iree_hal_amd_fabric_allocation_queue_enqueue(provider->allocation_queue,
                                               &request->operation);
}

void iree_hal_amd_fabric_slab_provider_execute(
    iree_hal_slab_operation_t* operation) {
  switch (operation->type) {
    case IREE_HAL_SLAB_OPERATION_ACQUIRE:
      iree_hal_slab_provider_acquire_slabs_inline(
          operation->provider, (iree_hal_slab_acquire_request_t*)operation,
          iree_hal_amd_fabric_slab_provider_acquire_slab,
          iree_hal_amd_fabric_slab_provider_release_slab);
      break;
    case IREE_HAL_SLAB_OPERATION_RELEASE:
      iree_hal_slab_provider_release_slabs_inline(
          operation->provider, (iree_hal_slab_release_request_t*)operation,
          iree_hal_amd_fabric_slab_provider_release_slab);
      break;
  }
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_amd_fabric_slab_provider_vtable = {
        .destroy = iree_hal_amd_fabric_slab_provider_destroy,
        .acquire_slabs = iree_hal_amd_fabric_slab_provider_acquire_slabs,
        .import_slab = iree_hal_amd_fabric_slab_provider_import_slab,
        .release_slabs = iree_hal_amd_fabric_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_amd_fabric_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_amd_fabric_slab_provider_validate_asan_options,
        .advise_asan_range =
            iree_hal_amd_fabric_slab_provider_advise_asan_range,
        .prefault = iree_hal_amd_fabric_slab_provider_prefault,
        .trim = iree_hal_amd_fabric_slab_provider_trim,
        .query_stats = iree_hal_amd_fabric_slab_provider_query_stats,
        .query_properties = iree_hal_amd_fabric_slab_provider_query_properties,
        .query_import_requirements =
            iree_hal_amd_fabric_slab_provider_query_import_requirements,
};
