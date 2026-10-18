// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/fabric/memory_backend.h"
#include "iree/hal/drivers/amd/status.h"

// A native consumer and its canonical shared page-table projection. Distinct
// consumers may alias one projection but still require exact source accesses.
typedef struct iree_hal_amd_fabric_reservation_access_t {
  // Consumer selected by the admitted families.
  amdf_memory_reservation_access_t native;
  // Cached live limits, including native permissions that cannot be narrowed.
  amdf_memory_reservation_capabilities_t capabilities;
  // Stable projection facts queried once after native construction.
  amdf_memory_reservation_access_info_t info;
  // Dense HAL projection identity, shared by native aliases.
  uint32_t projection;
} iree_hal_amd_fabric_reservation_access_t;

typedef struct iree_hal_amd_fabric_virtual_buffer_t {
  // Ordinary buffer resource owning the native address interval.
  iree_hal_virtual_buffer_t base;
  // Borrowed group-lifetime core API.
  const amdf_api_t* api;
  // Owned native reservation, released at final buffer destruction.
  amdf_memory_reservation_t* native;
  // Owned cache-maintenance view borrowing the stable host projection.
  amdf_host_mapping_t* host_mapping;
  // Native host consumer ordinal, or INVALID when no CPU access is admitted.
  uint32_t host_access;
  // Borrowed diagnostic sink owned by the creating device.
  iree_hal_device_event_sink_t event_sink;
  // Number of exact native consumers.
  uint32_t access_count;
  // Consumer facts in native access-ordinal order.
  iree_hal_amd_fabric_reservation_access_t* accesses;
  // Native access ordinal per exact HAL site, INVALID for excluded sites.
  uint32_t* scope_accesses;
  // Dense shared page-table projection per exact HAL site.
  uint32_t* projections;
  // One stable execution address per native consumer.
  iree_hal_buffer_native_binding_t* bindings;
  // Virtual geometry and borrowed progress services for child pools.
  iree_hal_buffer_backing_facts_t backing;
} iree_hal_amd_fabric_virtual_buffer_t;

static void iree_hal_amd_fabric_virtual_buffer_destroy(
    iree_hal_buffer_t* base) {
  iree_hal_amd_fabric_virtual_buffer_t* buffer =
      (iree_hal_amd_fabric_virtual_buffer_t*)base;
  if (buffer->host_mapping) {
    const amdf_status_t status =
        buffer->api->host_mapping_destroy(buffer->host_mapping);
    if (status != AMDF_STATUS_OK) {
      iree_hal_amd_report_failure(buffer->event_sink, IREE_SV("amd"), status,
                                  "host_mapping_destroy");
    }
  }
  const amdf_status_t status =
      buffer->api->memory_reservation_destroy(buffer->native);
  if (status != AMDF_STATUS_OK) {
    iree_hal_amd_report_failure(buffer->event_sink, IREE_SV("amd"), status,
                                "memory_reservation_destroy");
  }
  iree_hal_memory_contract_release(base->memory.contract);
  iree_allocator_free(buffer->base.host_allocator, buffer);
}

static iree_status_t iree_hal_amd_fabric_virtual_buffer_detach(
    iree_hal_buffer_t* base, uint32_t scope_id) {
  iree_hal_amd_fabric_virtual_buffer_t* buffer =
      (iree_hal_amd_fabric_virtual_buffer_t*)base;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      buffer->api->memory_reservation_detach(buffer->native,
                                             buffer->scope_accesses[scope_id]),
      "memory_reservation_detach");
}

static iree_status_t iree_hal_amd_fabric_virtual_buffer_update(
    iree_hal_buffer_t* base, uint32_t projection, iree_host_size_t update_count,
    const iree_hal_memory_mapping_update_t* updates) {
  iree_hal_amd_fabric_virtual_buffer_t* reservation =
      (iree_hal_amd_fabric_virtual_buffer_t*)base;
  if (update_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mapping vector exceeds native count capacity");
  }
  (void)projection;
  amdf_memory_mapping_update_t inline_updates[16];
  amdf_memory_mapping_update_t* native_updates = inline_updates;
  if (update_count > IREE_ARRAYSIZE(inline_updates)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        reservation->base.host_allocator, update_count, sizeof(*native_updates),
        (void**)&native_updates));
  }
  const uint32_t ordinal =
      reservation->accesses[reservation->scope_accesses[updates[0].scope_id]]
          .info.mapping_access_ordinal;
  const iree_hal_amd_fabric_reservation_access_t* access =
      &reservation->accesses[ordinal];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < update_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_fabric_mapping_source_t* source =
        updates[i].access ? updates[i].source.mapping_source.descriptor : NULL;
    uint32_t source_ordinal = 0;
    if (source && access->native.device) {
      for (; source_ordinal < source->access_count; ++source_ordinal) {
        if (source->accesses[source_ordinal].device == access->native.device) {
          break;
        }
      }
      if (source_ordinal == source->access_count) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "mapping source lacks a consumer sharing target translations");
        break;
      }
    }
    amdf_memory_access_t permissions = 0;
    if (source) {
      permissions = access->capabilities.required_access;
      if (iree_any_bit_set(updates[i].access, IREE_HAL_MEMORY_ACCESS_READ)) {
        permissions |= AMDF_MEMORY_ACCESS_READ;
      }
      if (iree_any_bit_set(updates[i].access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        permissions |= AMDF_MEMORY_ACCESS_WRITE;
      }
    }
    native_updates[i] = (amdf_memory_mapping_update_t){
        .target_byte_offset = updates[i].target_range.byte_offset,
        .byte_length = updates[i].target_range.byte_length,
        .memory = source ? source->memory : NULL,
        .memory_access_ordinal = source_ordinal,
        .access = permissions,
        .source_byte_offset = source ? updates[i].source.mapping_source.offset -
                                           source->logical_byte_offset
                                     : 0,
    };
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        reservation->api->memory_reservation_update(
            reservation->native, ordinal, (uint32_t)update_count,
            native_updates),
        "memory_reservation_update");
  }
  if (native_updates != inline_updates) {
    iree_allocator_free(reservation->base.host_allocator, native_updates);
  }
  return status;
}

static iree_status_t iree_hal_amd_fabric_virtual_buffer_invalidate_range(
    iree_hal_buffer_t* base, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_hal_amd_fabric_virtual_buffer_t* buffer =
      (iree_hal_amd_fabric_virtual_buffer_t*)base;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      buffer->api->host_mapping_cache_control(
          buffer->host_mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, offset,
          length),
      "host_mapping_cache_control(INVALIDATE)");
}

static iree_status_t iree_hal_amd_fabric_virtual_buffer_flush_range(
    iree_hal_buffer_t* base, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_hal_amd_fabric_virtual_buffer_t* buffer =
      (iree_hal_amd_fabric_virtual_buffer_t*)base;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      buffer->api->host_mapping_cache_control(buffer->host_mapping,
                                              AMDF_HOST_CACHE_OPERATION_FLUSH,
                                              offset, length),
      "host_mapping_cache_control(FLUSH)");
}

static const iree_hal_buffer_vtable_t
    iree_hal_amd_fabric_virtual_buffer_vtable = {
        .destroy = iree_hal_amd_fabric_virtual_buffer_destroy,
        .export_range = iree_hal_virtual_buffer_export_range,
        .map_range = iree_hal_virtual_buffer_map_range,
        .unmap_range = iree_hal_virtual_buffer_unmap_range,
        .invalidate_range = iree_hal_amd_fabric_virtual_buffer_invalidate_range,
        .flush_range = iree_hal_amd_fabric_virtual_buffer_flush_range,
        .query_virtual = iree_hal_virtual_buffer_query_root,
        .update_mappings = iree_hal_amd_fabric_virtual_buffer_update,
        .detach_mappings = iree_hal_amd_fabric_virtual_buffer_detach,
};

// Temporary qualification product; contains no native address or payload.
typedef struct iree_hal_amd_fabric_reservation_plan_t {
  // Prepared metadata adopted by creation or freed after a capability query.
  iree_hal_amd_fabric_virtual_buffer_t* reservation;
  // Owned immutable execution contract for the requested scope.
  iree_hal_memory_contract_t* contract;
  // Complete-scope intersection of native address geometry.
  iree_hal_virtual_buffer_capabilities_t capabilities;
  // Native consumers borrowed from the prepared metadata allocation.
  amdf_memory_reservation_access_t* native_accesses;
} iree_hal_amd_fabric_reservation_plan_t;

static iree_status_t iree_hal_amd_fabric_virtual_buffer_prepare(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_reservation_plan_t* out_plan) {
  *out_plan = (iree_hal_amd_fabric_reservation_plan_t){0};
  const iree_hal_amd_fabric_memory_backend_t* owner =
      (const iree_hal_amd_fabric_memory_backend_t*)base;
  if (!owner->native.instance ||
      scope.family_count >= IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE - 1 ||
      (scope.host.access &&
       scope.host.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN &&
       scope.host.cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK)) {
    return iree_ok_status();
  }
  const uint32_t access_capacity = (uint32_t)scope.family_count + 1;
  const uint32_t scope_count = iree_hal_device_group_memory_scope_count(group);
  iree_host_size_t total_size = 0;
  iree_host_size_t accesses_offset = 0, scopes_offset = 0;
  iree_host_size_t projections_offset = 0, bindings_offset = 0;
  iree_host_size_t types_offset = 0, native_accesses_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_fabric_virtual_buffer_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          access_capacity, iree_hal_amd_fabric_reservation_access_t,
          iree_alignof(iree_hal_amd_fabric_reservation_access_t),
          &accesses_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope_count, uint32_t, iree_alignof(uint32_t),
                                &scopes_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope_count, uint32_t, iree_alignof(uint32_t),
                                &projections_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          access_capacity, iree_hal_buffer_native_binding_t,
          iree_alignof(iree_hal_buffer_native_binding_t), &bindings_offset),
      IREE_STRUCT_FIELD_ALIGNED(access_capacity, uint16_t,
                                iree_alignof(uint16_t), &types_offset),
      IREE_STRUCT_FIELD_ALIGNED(access_capacity,
                                amdf_memory_reservation_access_t,
                                iree_alignof(amdf_memory_reservation_access_t),
                                &native_accesses_offset)));
  iree_hal_amd_fabric_virtual_buffer_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&reservation));
  reservation->host_access = IREE_HAL_MEMORY_SCOPE_INVALID;
  reservation->api = owner->native.api;
  reservation->event_sink = owner->native.event_sink;
  reservation->accesses =
      (iree_hal_amd_fabric_reservation_access_t*)((uint8_t*)reservation +
                                                  accesses_offset);
  reservation->scope_accesses =
      (uint32_t*)((uint8_t*)reservation + scopes_offset);
  reservation->projections =
      (uint32_t*)((uint8_t*)reservation + projections_offset);
  reservation->bindings =
      (iree_hal_buffer_native_binding_t*)((uint8_t*)reservation +
                                          bindings_offset);
  uint16_t* types = (uint16_t*)((uint8_t*)reservation + types_offset);
  amdf_memory_reservation_access_t* native_accesses =
      (amdf_memory_reservation_access_t*)((uint8_t*)reservation +
                                          native_accesses_offset);
  for (uint32_t i = 0; i < scope_count; ++i) {
    reservation->scope_accesses[i] = IREE_HAL_MEMORY_SCOPE_INVALID;
    reservation->projections[i] = IREE_HAL_MEMORY_SCOPE_INVALID;
  }
  bool supported = true;
  bool has_native_device = false;
  iree_hal_virtual_buffer_capabilities_t capabilities = {
      .max_byte_length = IREE_DEVICE_SIZE_MAX,
      .reservation_granularity = 1,
      .mapping_granularity = 1,
      .max_alignment = IREE_DEVICE_SIZE_MAX,
      .operations = IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL,
  };
  iree_hal_buffer_usage_t usage = 0;
  iree_hal_memory_access_t host_permissions = scope.host.access;
  amdf_memory_access_t shared_permissions = 0;
  iree_status_t status = iree_ok_status();
  // Public host access and Task execution share one native host consumer.
  // Family order does not select ownership or introduce duplicate host views.
  for (iree_host_size_t i = 0;
       i <= scope.family_count && supported && iree_status_is_ok(status); ++i) {
    const iree_hal_pool_family_access_t* requested =
        i < scope.family_count ? &scope.families[i] : NULL;
    if (!requested && !host_permissions) {
      continue;
    }
    amdf_device_t* device = NULL;
    uint32_t address_domain = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
    amdf_memory_access_t minimum_access = 0;
    uint16_t type = IREE_HAL_BUFFER_INTERFACE_HOST;
    iree_hal_memory_access_t permissions = host_permissions;
    if (requested) {
      const iree_hal_memory_backend_t* participant =
          iree_hal_device_memory_backend(
              iree_hal_queue_family_device(requested->family));
      const iree_hal_atomic_operation_capabilities_t no_atomics = {0};
      if (!participant || requested->requirements ||
          !iree_hal_atomic_operation_capabilities_support(
              no_atomics, requested->atomic_operations)) {
        supported = false;
        break;
      }
      permissions = iree_hal_buffer_usage_access(requested->usage);
      usage |= requested->usage;
      if (participant->type == IREE_HAL_MEMORY_BACKEND_HOST) {
        supported = iree_all_bits_set(
                        iree_hal_queue_family_spec(requested->family)->flags,
                        IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_HOST_MEMORY_BARRIERS) &&
                    !(requested->interfaces &
                      ~(UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST));
      } else if (participant->create_virtual_buffer ==
                 iree_hal_amd_fabric_virtual_buffer_create) {
        const iree_hal_amd_fabric_memory_backend_t* backend =
            (const iree_hal_amd_fabric_memory_backend_t*)participant;
        const iree_hal_amd_fabric_memory_family_t* family =
            &backend->native
                 .families[iree_hal_queue_family_ordinal(requested->family)];
        // Native reservations publish one execution address per consumer.
        // Qualification remains native: unsupported NPU address domains are
        // rejected by the live reservation capability query below.
        bool execution_address = false;
        for (uint32_t b = 0; b < family->binding_count; ++b) {
          const iree_hal_amd_fabric_memory_binding_t binding =
              family->bindings[b];
          if ((binding.type == IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS &&
               binding.address_kind == AMDF_MEMORY_ADDRESS_GPU) ||
              (binding.type == IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA &&
               binding.address_kind == AMDF_MEMORY_ADDRESS_XDNA_DMA)) {
            type = binding.type;
            execution_address = true;
          }
        }
        supported = backend->native.api == owner->native.api &&
                    backend->native.instance == owner->native.instance &&
                    execution_address &&
                    !(requested->usage & ~family->supported_usage) &&
                    !(requested->interfaces & ~(UINT64_C(1) << type));
        host_permissions |=
            iree_hal_buffer_usage_access(requested->usage & family->host_usage);
        device = backend->native.device;
        address_domain = backend->native.address_domain_ordinal;
        minimum_access = family->minimum_access;
        has_native_device = true;
      } else {
        supported = false;
      }
      if (!supported) {
        break;
      }
    }
    uint32_t a = 0;
    for (; a < reservation->access_count; ++a) {
      if (reservation->accesses[a].native.device == device &&
          reservation->accesses[a].native.address_domain_ordinal ==
              address_domain) {
        break;
      }
    }
    iree_hal_amd_fabric_reservation_access_t* access =
        &reservation->accesses[a];
    if (a == reservation->access_count) {
      access->native.device = device;
      access->native.address_domain_ordinal = address_domain;
      access->capabilities = (amdf_memory_reservation_capabilities_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
          .structure_size = sizeof(access->capabilities),
      };
      const amdf_status_t native_status =
          device
              ? owner->native.api->device_query_memory_reservation_capabilities(
                    device, address_domain, &access->capabilities)
              : owner->native.api
                    ->instance_query_host_memory_reservation_capabilities(
                        owner->native.instance, &access->capabilities);
      if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        supported = false;
        break;
      }
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          native_status, "query_memory_reservation_capabilities");
      if (!iree_status_is_ok(status)) {
        break;
      }
      ++reservation->access_count;
      types[a] = type;
      if (!device) {
        reservation->host_access = a;
      }
      capabilities.mapping_granularity =
          iree_max(capabilities.mapping_granularity,
                   access->capabilities.mapping_granularity);
      capabilities.reservation_granularity =
          iree_max(capabilities.reservation_granularity,
                   access->capabilities.reservation_granularity);
      capabilities.max_byte_length =
          iree_min(capabilities.max_byte_length,
                   access->capabilities.maximum_byte_length);
      capabilities.max_alignment = iree_min(
          capabilities.max_alignment, access->capabilities.maximum_alignment);
    }
    access->native.access |=
        minimum_access | access->capabilities.required_access;
    if (iree_any_bit_set(permissions, IREE_HAL_MEMORY_ACCESS_READ)) {
      access->native.access |= AMDF_MEMORY_ACCESS_READ;
    }
    if (iree_any_bit_set(permissions, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      access->native.access |= AMDF_MEMORY_ACCESS_WRITE;
    }
    shared_permissions |= access->native.access &
                          (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
    if (requested) {
      const uint32_t id = requested->family->memory.queue_scope_id;
      reservation->scope_accesses[id] = a;
      reservation->scope_accesses[id + 1] = a;
    } else if (scope.host.access) {
      reservation->scope_accesses[1] = a;
    }
  }
  capabilities.reservation_granularity = iree_max(
      capabilities.reservation_granularity, capabilities.mapping_granularity);
  // CPU-only scopes retain the host backend's prepared backing format.
  supported &=
      has_native_device &&
      capabilities.reservation_granularity <= capabilities.max_byte_length &&
      capabilities.reservation_granularity <= capabilities.max_alignment;
  for (uint32_t a = 0;
       a < reservation->access_count && supported && iree_status_is_ok(status);
       ++a) {
    iree_hal_amd_fabric_reservation_access_t* access =
        &reservation->accesses[a];
    // Shared host/SVA translations require equal native permission envelopes.
    // Use a common read/write upper bound; HAL sites retain their exact logical
    // usage, and each mapping still carries its explicitly requested access.
    access->native.access |=
        shared_permissions & access->capabilities.supported_access;
    supported &=
        (access->native.access & ~access->capabilities.supported_access) == 0;
    native_accesses[a] = access->native;
  }
  iree_hal_memory_contract_t* contract = NULL;
  if (supported && iree_status_is_ok(status)) {
    const iree_hal_buffer_binding_layout_t layout = {
        .byte_length =
            reservation->access_count * sizeof(*reservation->bindings),
        .binding_count = (uint16_t)reservation->access_count,
        .host_binding_index =
            reservation->host_access == IREE_HAL_MEMORY_SCOPE_INVALID
                ? IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE
                : (uint16_t)reservation->host_access,
        .types = types,
    };
    status = iree_hal_memory_contract_create(
        iree_hal_device_group_memory_domain(group), scope_count, &layout,
        host_allocator, &contract);
  }
  if (contract && iree_status_is_ok(status)) {
    contract->host = scope.host;
    if (scope.host.access) {
      contract->host.cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
      if (iree_any_bit_set(scope.host.modes, IREE_HAL_MAPPING_MODE_SCOPED)) {
        usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
      }
      if (iree_any_bit_set(scope.host.modes,
                           IREE_HAL_MAPPING_MODE_PERSISTENT)) {
        usage |= IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
      }
    }
    contract->allowed_usage = usage;
    contract->allowed_access =
        iree_hal_buffer_usage_access(usage) | scope.host.access;
    *out_plan = (iree_hal_amd_fabric_reservation_plan_t){
        .reservation = reservation,
        .contract = contract,
        .capabilities = capabilities,
        .native_accesses = native_accesses,
    };
  } else {
    iree_hal_memory_contract_release(contract);
    iree_allocator_free(host_allocator, reservation);
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* backend, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  iree_hal_amd_fabric_reservation_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_virtual_buffer_prepare(
      backend, group, scope, host_allocator, &plan));
  *out_capabilities = plan.capabilities;
  iree_hal_memory_contract_release(plan.contract);
  iree_allocator_free(host_allocator, plan.reservation);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_virtual_buffer_create(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  const iree_hal_amd_fabric_memory_backend_t* owner =
      (const iree_hal_amd_fabric_memory_backend_t*)base;
  iree_hal_amd_fabric_reservation_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_virtual_buffer_prepare(
      base, group, scope, host_allocator, &plan));
  if (!plan.reservation) {
    return iree_ok_status();
  }
  iree_hal_amd_fabric_virtual_buffer_t* reservation = plan.reservation;
  iree_hal_memory_contract_t* contract = plan.contract;
  const iree_device_size_t granularity = iree_max(
      options->mapping_granularity, plan.capabilities.mapping_granularity);
  const iree_device_size_t alignment = iree_max(
      options->min_alignment, plan.capabilities.reservation_granularity);
  iree_status_t status = iree_ok_status();
  if (options->byte_length > plan.capabilities.max_byte_length ||
      options->byte_length % plan.capabilities.reservation_granularity ||
      options->byte_length % granularity ||
      alignment > plan.capabilities.max_alignment) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "reservation exceeds native address geometry");
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t a = 0; a < reservation->access_count; ++a) {
      plan.native_accesses[a].mapping_granularity = granularity;
    }
    const amdf_memory_reservation_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .byte_length = options->byte_length,
        .minimum_alignment = alignment,
        .access_count = reservation->access_count,
        .accesses = plan.native_accesses,
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        owner->native.api->memory_reservation_create(
            owner->native.instance, &create_info, &reservation->native),
        "memory_reservation_create");
  }

  uint32_t projection_count = 0;
  for (uint32_t a = 0; reservation->native && a < reservation->access_count &&
                       iree_status_is_ok(status);
       ++a) {
    iree_hal_amd_fabric_reservation_access_t* access =
        &reservation->accesses[a];
    access->info = (amdf_memory_reservation_access_info_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
        .structure_size = sizeof(access->info),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        reservation->api->memory_reservation_query_access_info(
            reservation->native, a, &access->info),
        "memory_reservation_query_access_info");
    if (iree_status_is_ok(status)) {
      access->projection =
          access->info.mapping_access_ordinal == a
              ? projection_count++
              : reservation->accesses[access->info.mapping_access_ordinal]
                    .projection;
      reservation->accesses[access->info.mapping_access_ordinal]
          .capabilities.required_access |= access->capabilities.required_access;
      if (a == reservation->host_access) {
        reservation->bindings[a].host_pointer =
            (void*)(uintptr_t)access->info.address;
      } else {
        reservation->bindings[a].device_address = access->info.address;
      }
    }
  }
  amdf_host_mapping_info_t host_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(host_info),
  };
  if (iree_status_is_ok(status) &&
      reservation->host_access != IREE_HAL_MEMORY_SCOPE_INVALID) {
    const amdf_memory_access_t access =
        reservation->accesses[reservation->host_access].native.access;
    const amdf_memory_map_info_t map_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map_info),
        .byte_length = options->byte_length,
        .flags =
            ((access & AMDF_MEMORY_ACCESS_READ) ? AMDF_MEMORY_MAP_FLAG_READ
                                                : 0) |
            ((access & AMDF_MEMORY_ACCESS_WRITE) ? AMDF_MEMORY_MAP_FLAG_WRITE
                                                 : 0),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        reservation->api->memory_reservation_map(
            reservation->native, reservation->host_access, &map_info,
            &reservation->host_mapping),
        "memory_reservation_map");
    if (iree_status_is_ok(status)) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          reservation->api->host_mapping_query_info(reservation->host_mapping,
                                                    &host_info),
          "host_mapping_query_info");
    }
  }
  if (reservation->native && iree_status_is_ok(status)) {
    for (uint32_t id = 1; id < contract->scope_count; ++id) {
      const uint32_t a = reservation->scope_accesses[id];
      if (a == IREE_HAL_MEMORY_SCOPE_INVALID) {
        continue;
      }
      const uint16_t type = contract->binding_layout.types[a];
      contract->scopes[id].interfaces = 1u << type;
      contract->scopes[id].bindings[type] = (uint16_t)a;
      reservation->projections[id] = reservation->accesses[a].projection;
    }
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const uint32_t id = scope.families[i].family->memory.queue_scope_id;
      contract->scopes[id].usage = scope.families[i].usage;
      contract->scopes[id + 1].usage = scope.families[i].usage;
    }
    reservation->backing = (iree_hal_buffer_backing_facts_t){
        .notification = owner->notification,
        .tracker =
            iree_hal_device_topology_info(owner->device)->frontier.tracker,
        .maintenance_range = {.byte_length = options->byte_length},
        .allocation_alignment = alignment,
        .maintenance_alignment = iree_max(1u, host_info.cache_line_size),
    };
    iree_hal_virtual_buffer_initialize(
        contract, reservation->bindings, &reservation->backing,
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
            (reservation->host_mapping ? IREE_HAL_MEMORY_TYPE_HOST_VISIBLE : 0),
        options->byte_length, granularity, plan.capabilities.operations,
        IREE_HAL_MEMORY_BACKEND_AMDF,
        &iree_hal_amd_fabric_virtual_buffer_vtable, host_allocator,
        &reservation->base);
    reservation->base.projection_count = projection_count;
    reservation->base.projections = reservation->projections;
    *out_buffer = &reservation->base.buffer;
  } else {
    if (reservation->host_mapping) {
      status = iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(
              reservation->api->host_mapping_destroy(reservation->host_mapping),
              "host_mapping_destroy"));
    }
    if (reservation->native) {
      status = iree_status_join(
          status,
          IREE_HAL_AMD_STATUS_FROM_AMDF(
              reservation->api->memory_reservation_destroy(reservation->native),
              "memory_reservation_destroy"));
    }
    iree_allocator_free(host_allocator, reservation);
  }
  iree_hal_memory_contract_release(contract);
  return status;
}
