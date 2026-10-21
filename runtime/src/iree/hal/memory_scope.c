// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory_scope.h"

#include "iree/base/internal/math.h"
#include "iree/hal/device_group.h"
#include "iree/hal/pool.h"

IREE_API_EXPORT iree_status_t
iree_hal_pool_host_access_validate(iree_hal_pool_host_access_t host) {
  if (iree_any_bit_set(host.access, ~IREE_HAL_MEMORY_ACCESS_ALL) ||
      iree_any_bit_set(host.modes, ~(IREE_HAL_MAPPING_MODE_SCOPED |
                                     IREE_HAL_MAPPING_MODE_PERSISTENT)) ||
      (uint32_t)host.cacheability > IREE_HAL_HOST_CACHEABILITY_UNCACHED ||
      !iree_hal_atomic_operation_capabilities_are_valid(host.atomics)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid host access, mapping mode, cacheability, "
                            "or atomic operations");
  }
  if ((!host.access != !host.modes) ||
      (!host.access &&
       (host.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN ||
        !iree_hal_atomic_operation_capabilities_is_empty(host.atomics)))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host permissions and mapping modes must be specified together");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_pool_scope_validate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope) {
  if (!group || (!scope.family_count && !scope.host.access)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "memory construction requires a group and an access scope");
  }
  if (scope.family_count && !scope.families) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool family array is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_pool_host_access_validate(scope.host));
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    const iree_hal_memory_site_t site = {
        .kind = IREE_HAL_MEMORY_SITE_QUEUE,
        .family = access->family,
    };
    iree_hal_memory_scope_t resolved;
    IREE_RETURN_IF_ERROR(
        iree_hal_device_group_resolve_memory_scope(group, site, &resolved));
    if (iree_any_bit_set(access->usage, ~(IREE_HAL_BUFFER_USAGE_TRANSFER |
                                          IREE_HAL_BUFFER_USAGE_DISPATCH)) ||
        (access->interfaces >> IREE_HAL_BUFFER_INTERFACE_COUNT) ||
        iree_any_bit_set(access->requirements,
                         ~(IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST |
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED)) ||
        !iree_hal_atomic_operation_capabilities_are_valid(
            access->atomics.queue) ||
        !iree_hal_atomic_operation_capabilities_are_valid(
            access->atomics.program)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid family access flags");
    }
    if (iree_any_bit_set(access->requirements,
                         IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST) &&
        !scope.host.access) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "host coherence requires a public host mapping");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (scope.families[j].family == access->family) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "pool scope repeats a queue family");
      }
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_group_resolve_memory_scope(
    const iree_hal_device_group_t* group, iree_hal_memory_site_t site,
    iree_hal_memory_scope_t* out_scope) {
  *out_scope = (iree_hal_memory_scope_t){
      .id = IREE_HAL_MEMORY_SCOPE_INVALID,
  };
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory scope requires a sealed device group");
  }
  if (site.kind == IREE_HAL_MEMORY_SITE_HOST) {
    *out_scope = (iree_hal_memory_scope_t){
        .domain = iree_hal_device_group_memory_domain(group),
        .id = 1,
    };
    return iree_ok_status();
  }
  if (site.kind != IREE_HAL_MEMORY_SITE_QUEUE &&
      site.kind != IREE_HAL_MEMORY_SITE_PROGRAM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid memory site kind %u", site.kind);
  }
  if (!site.family) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue/program memory site requires a family");
  }
  iree_hal_device_t* device = iree_hal_queue_family_device(site.family);
  bool is_member = false;
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group);
       ++i) {
    is_member |= iree_hal_device_group_device_at(group, i) == device;
  }
  if (!is_member ||
      site.family->ordinal >=
          iree_hal_device_spec_queues(iree_hal_device_spec(device))
              ->family_count ||
      iree_hal_device_queue_family(device, site.family->ordinal) !=
          site.family) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "memory site family is not a canonical group member");
  }
  *out_scope = (iree_hal_memory_scope_t){
      .domain = site.family->memory.domain,
      .id = site.family->memory.queue_scope_id +
            (site.kind == IREE_HAL_MEMORY_SITE_PROGRAM ? 1 : 0),
  };
  return iree_ok_status();
}

// Each half converts to unsigned 32 bits before widening. Microsoft enum
// representation can otherwise sign-extend high-bit effect flags into the
// other half of the cell.
static uint64_t iree_hal_memory_transition_pack_effects(uint32_t release,
                                                        uint32_t acquire) {
  return (uint64_t)release | ((uint64_t)acquire << 32);
}

IREE_API_EXPORT iree_status_t iree_hal_memory_contract_create(
    const void* domain, uint32_t scope_count,
    const iree_hal_buffer_binding_layout_t* binding_layout,
    iree_allocator_t host_allocator,
    iree_hal_memory_contract_t** out_contract) {
  *out_contract = NULL;
  // A flattened prepared key uses 32 bits. This bound also keeps rounding the
  // row stride and multiplying the cell count representable on every host.
  if (scope_count < 2 || scope_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "memory scope count does not fit the table layout");
  }
  const uint32_t row_shift =
      32u - iree_math_count_leading_zeros_u32(scope_count - 1);
  const uint32_t cell_count = scope_count << row_shift;
  iree_host_size_t total_size = 0;
  iree_host_size_t scopes_offset = 0;
  iree_host_size_t types_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_memory_contract_t), &total_size,
      IREE_STRUCT_FIELD_FAM(cell_count, uint64_t),
      IREE_STRUCT_FIELD_ALIGNED(scope_count, iree_hal_memory_scope_access_t, 32,
                                &scopes_offset),
      IREE_STRUCT_FIELD(binding_layout->binding_count, uint16_t,
                        &types_offset)));
  iree_hal_memory_contract_t* contract = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, 32, scopes_offset, (void**)&contract));
  iree_atomic_ref_count_init(&contract->ref_count);
  contract->host_allocator = host_allocator;
  contract->domain = domain;
  contract->scope_count = scope_count;
  contract->transition_row_shift = row_shift;
  contract->scopes =
      (iree_hal_memory_scope_access_t*)((uint8_t*)contract + scopes_offset);
  uint16_t* binding_types = (uint16_t*)((uint8_t*)contract + types_offset);
  if (binding_layout->binding_count) {
    memcpy(binding_types, binding_layout->types,
           binding_layout->binding_count * sizeof(*binding_types));
  }
  contract->binding_layout = *binding_layout;
  contract->binding_layout.types = binding_types;
  const uint64_t unsupported = iree_hal_memory_transition_pack_effects(
      IREE_HAL_MEMORY_EFFECT_UNSUPPORTED, IREE_HAL_MEMORY_EFFECT_UNSUPPORTED);
  for (uint32_t i = 0; i < cell_count; ++i) {
    contract->transitions[i] = unsupported;
  }
  for (uint32_t i = 0; i < scope_count; ++i) {
    memset(contract->scopes[i].bindings, 0xFF,
           sizeof(contract->scopes[i].bindings));
  }
  *out_contract = contract;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_memory_contract_retain(
    iree_hal_memory_contract_t* contract) {
  if (contract) {
    iree_atomic_ref_count_inc(&contract->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_memory_contract_release(
    iree_hal_memory_contract_t* contract) {
  if (contract && iree_atomic_ref_count_dec(&contract->ref_count) == 1) {
    iree_allocator_free(contract->host_allocator, contract->transition_details);
    iree_allocator_free_aligned(contract->host_allocator, contract);
  }
}

static_assert(IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL <= UINT8_MAX,
              "prepared atomic operation masks must fit in one byte");

IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_memory_site_filter_atomic_operations(
    iree_hal_memory_site_t site,
    iree_hal_atomic_operation_capabilities_t backing_operations) {
  if (site.kind == IREE_HAL_MEMORY_SITE_HOST) {
    return backing_operations;
  }
  const iree_hal_queue_family_spec_t* family =
      iree_hal_queue_family_spec(site.family);
  if (site.kind == IREE_HAL_MEMORY_SITE_QUEUE) {
    return iree_hal_atomic_operation_capabilities_intersect(
        backing_operations, family->atomic_capabilities.operations);
  }
  return iree_any_bit_set(family->role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)
             ? backing_operations
             : (iree_hal_atomic_operation_capabilities_t){0};
}

IREE_API_EXPORT void iree_hal_memory_scope_access_set_atomic_operations(
    iree_hal_memory_scope_access_t* access,
    iree_hal_atomic_operation_capabilities_t operations) {
  access->atomic_operations =
      operations.device_scope_32 | (operations.device_scope_64 << 8) |
      (operations.system_scope_32 << 16) | (operations.system_scope_64 << 24);
}

static iree_hal_atomic_operation_capabilities_t
iree_hal_memory_scope_unpack_atomic_operations(uint32_t packed) {
  return (iree_hal_atomic_operation_capabilities_t){
      .device_scope_32 = packed & 0xFFu,
      .device_scope_64 = (packed >> 8) & 0xFFu,
      .system_scope_32 = (packed >> 16) & 0xFFu,
      .system_scope_64 = packed >> 24,
  };
}

IREE_API_EXPORT void iree_hal_memory_contract_finalize_atomic_operations(
    iree_hal_memory_contract_t* contract) {
  uint32_t common = UINT32_MAX;
  bool any = false;
  for (uint32_t id = 1; id < contract->scope_count; ++id) {
    if (id == 1 ? contract->host.access != 0
                : contract->scopes[id].interfaces != 0) {
      common &= contract->scopes[id].atomic_operations;
      any = true;
    }
  }
  contract->scopes[IREE_HAL_MEMORY_SCOPE_ANY].atomic_operations =
      any ? common : 0;
  contract->host.atomics = iree_hal_memory_scope_unpack_atomic_operations(
      contract->host.access ? contract->scopes[1].atomic_operations : 0);
}

static iree_hal_atomic_operation_capabilities_t
iree_hal_memory_contract_query_atomic_operations(
    const iree_hal_memory_contract_t* contract, iree_hal_memory_scope_t scope) {
  if (!contract || scope.domain != contract->domain ||
      scope.id >= contract->scope_count) {
    return (iree_hal_atomic_operation_capabilities_t){0};
  }
  return iree_hal_memory_scope_unpack_atomic_operations(
      contract->scopes[scope.id].atomic_operations);
}

IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_pool_query_atomic_operations(const iree_hal_pool_t* pool,
                                      iree_hal_memory_scope_t scope) {
  return iree_hal_memory_contract_query_atomic_operations(pool->memory_contract,
                                                          scope);
}

IREE_API_EXPORT iree_hal_atomic_operation_capabilities_t
iree_hal_buffer_query_atomic_operations(const iree_hal_buffer_t* buffer,
                                        iree_hal_memory_scope_t scope) {
  return iree_hal_memory_contract_query_atomic_operations(
      buffer->memory.contract, scope);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_resolve_binding(
    const iree_hal_pool_t* pool, iree_hal_memory_scope_t scope,
    iree_hal_buffer_interface_t type,
    iree_hal_buffer_native_binding_slot_t* out_slot) {
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
  };
  const iree_hal_memory_contract_t* contract = pool->memory_contract;
  if (!contract || scope.domain != contract->domain ||
      scope.id == IREE_HAL_MEMORY_SCOPE_ANY ||
      scope.id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding scope is outside the pool memory domain");
  }
  if ((uint32_t)type >= IREE_ARRAYSIZE(contract->scopes[0].bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid native binding interface %u", type);
  }
  uint16_t index = contract->scopes[scope.id].bindings[type];
  if (index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "native interface is not prepared for this scope");
  }
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = index,
      .type = (uint16_t)type,
  };
  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_pool_host_access_t
iree_hal_pool_query_host_access(const iree_hal_pool_t* pool) {
  return pool->memory_contract ? pool->memory_contract->host
                               : (iree_hal_pool_host_access_t){0};
}

IREE_API_EXPORT iree_hal_pool_mapping_requirements_t
iree_hal_pool_query_mapping_requirements(const iree_hal_pool_t* pool) {
  return pool->memory_contract
             ? pool->memory_contract->mapping_source.requirements
             : (iree_hal_pool_mapping_requirements_t){0};
}

static const iree_hal_memory_scope_access_t*
iree_hal_memory_contract_family_access(
    const iree_hal_memory_contract_t* contract,
    const iree_hal_queue_family_t* family) {
  if (family->memory.domain != contract->domain ||
      family->memory.queue_scope_id >= contract->scope_count) {
    return NULL;
  }
  const iree_hal_memory_scope_access_t* access =
      &contract->scopes[family->memory.queue_scope_id];
  return access->interfaces ? access : NULL;
}

static bool iree_hal_buffer_placement_allows_family(
    const iree_hal_buffer_t* buffer, const iree_hal_queue_family_t* family) {
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (iree_hal_buffer_placement_is_undefined(placement)) {
    return true;
  }
  if (placement.device != iree_hal_queue_family_device(family)) {
    return false;
  }
  return iree_hal_queue_family_affinity_is_any(
             placement.queue_family_affinity) ||
         iree_any_bit_set(placement.queue_family_affinity,
                          iree_hal_make_queue_family_affinity(
                              iree_hal_queue_family_ordinal(family)));
}

IREE_API_EXPORT iree_hal_buffer_usage_t iree_hal_buffer_family_usage(
    const iree_hal_buffer_t* buffer, const iree_hal_queue_family_t* family) {
  if (!buffer->memory.contract) {
    return iree_hal_buffer_placement_allows_family(buffer, family)
               ? buffer->allowed_usage
               : IREE_HAL_BUFFER_USAGE_NONE;
  }
  const iree_hal_memory_scope_access_t* access =
      iree_hal_memory_contract_family_access(buffer->memory.contract, family);
  return access ? access->usage : IREE_HAL_BUFFER_USAGE_NONE;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_validate_family_usage(
    const iree_hal_buffer_t* buffer, const iree_hal_queue_family_t* family,
    iree_hal_buffer_usage_t required_usage) {
  if (!buffer->memory.contract) {
    if (!iree_hal_buffer_placement_allows_family(buffer, family)) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "queue family is outside the buffer allocation placement");
    }
    return iree_hal_buffer_validate_usage(buffer->allowed_usage,
                                          required_usage);
  }
  const iree_hal_memory_scope_access_t* access =
      iree_hal_memory_contract_family_access(buffer->memory.contract, family);
  if (!access) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue family is outside the buffer memory scope");
  }
  return iree_hal_buffer_validate_usage(access->usage, required_usage);
}

IREE_API_EXPORT iree_status_t iree_hal_pool_validate_family_usage(
    const iree_hal_pool_t* pool, const iree_hal_queue_family_t* family,
    iree_hal_buffer_usage_t required_usage) {
  if (!pool->memory_contract) {
    return iree_ok_status();
  }
  const iree_hal_memory_scope_access_t* access =
      iree_hal_memory_contract_family_access(pool->memory_contract, family);
  if (!access) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "queue family is outside the pool memory scope");
  }
  return iree_hal_buffer_validate_usage(access->usage, required_usage);
}

IREE_API_EXPORT iree_hal_memory_transition_table_t
iree_hal_pool_transition_table(const iree_hal_pool_t* pool) {
  return (iree_hal_memory_transition_table_t){pool->memory_contract};
}

typedef struct iree_hal_memory_pair_record_t {
  // Complete native qualification for an exact pair.
  iree_hal_memory_pair_info_t info;
  // Release and acquire recipes borrowing their corresponding info fields.
  iree_hal_memory_transition_recipe_t recipes[2];
} iree_hal_memory_pair_record_t;

// One allocation holds the cold cell indices and interned exact-pair records.
// Index zero is the all-unknown record, including for wildcard/padding cells.
struct iree_hal_memory_transition_details_t {
  // Interned exact-pair descriptions and their resource recipes.
  iree_hal_memory_pair_record_t* records;
  // Release and acquire recipes per local site, with composed inline actions.
  iree_hal_memory_transition_recipe_t* wildcards;
  // Exact-pair record index per flattened transition cell.
  uint32_t indices[];
};

static bool iree_hal_memory_scope_can_read(
    const iree_hal_memory_contract_t* contract, uint32_t id) {
  if (id == 1) {
    return iree_any_bit_set(contract->host.access, IREE_HAL_MEMORY_ACCESS_READ);
  }
  return contract->scopes[id].interfaces &&
         iree_any_bit_set(
             contract->scopes[id].usage,
             IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                 IREE_HAL_BUFFER_USAGE_STORAGE_READ |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_READ);
}

static bool iree_hal_memory_scope_can_write(
    const iree_hal_memory_contract_t* contract, uint32_t id) {
  if (id == 1) {
    return iree_any_bit_set(contract->host.access,
                            IREE_HAL_MEMORY_ACCESS_WRITE);
  }
  return contract->scopes[id].interfaces &&
         iree_any_bit_set(contract->scopes[id].usage,
                          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                              IREE_HAL_BUFFER_USAGE_STORAGE_WRITE |
                              IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE_WRITE);
}

static bool iree_hal_memory_transition_has_roles(
    const iree_hal_memory_contract_t* contract,
    iree_hal_memory_scope_t producer, iree_hal_memory_scope_t consumer) {
  return contract && producer.domain == contract->domain &&
         consumer.domain == contract->domain &&
         producer.id < contract->scope_count &&
         consumer.id < contract->scope_count && (producer.id || consumer.id) &&
         (!producer.id ||
          iree_hal_memory_scope_can_write(contract, producer.id)) &&
         (!consumer.id ||
          iree_hal_memory_scope_can_read(contract, consumer.id));
}

IREE_API_EXPORT iree_status_t iree_hal_memory_transition_prepare_pair(
    iree_hal_memory_transition_table_t table, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer,
    iree_hal_memory_transition_action_t action,
    iree_hal_memory_transition_pair_t* out_pair) {
  *out_pair = (iree_hal_memory_transition_pair_t){0};
  if ((action != IREE_HAL_MEMORY_TRANSITION_RELEASE &&
       action != IREE_HAL_MEMORY_TRANSITION_ACQUIRE) ||
      (!producer.id && action == IREE_HAL_MEMORY_TRANSITION_RELEASE) ||
      (!consumer.id && action == IREE_HAL_MEMORY_TRANSITION_ACQUIRE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "transition requires an exact local action site");
  }
  if (!iree_hal_memory_transition_has_roles(table.contract, producer,
                                            consumer)) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "transition scopes require admitted writes and reads in one domain");
  }
  out_pair->cell_index =
      (producer.id << table.contract->transition_row_shift) + consumer.id;
  return iree_ok_status();
}

static iree_hal_memory_transition_t iree_hal_memory_contract_query_transition(
    const iree_hal_memory_contract_t* contract,
    iree_hal_memory_scope_t producer, iree_hal_memory_scope_t consumer) {
  if (!iree_hal_memory_transition_has_roles(contract, producer, consumer)) {
    return (iree_hal_memory_transition_t){
        .release = {IREE_HAL_MEMORY_EFFECT_UNSUPPORTED},
        .acquire = {IREE_HAL_MEMORY_EFFECT_UNSUPPORTED},
    };
  }
  return iree_hal_memory_transition_query(
      (iree_hal_memory_transition_table_t){contract},
      (iree_hal_memory_transition_pair_t){
          (producer.id << contract->transition_row_shift) + consumer.id});
}

IREE_API_EXPORT iree_hal_memory_transition_t iree_hal_pool_query_transition(
    const iree_hal_pool_t* pool, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer) {
  return iree_hal_memory_contract_query_transition(pool->memory_contract,
                                                   producer, consumer);
}

IREE_API_EXPORT iree_hal_memory_transition_t iree_hal_buffer_query_transition(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t producer,
    iree_hal_memory_scope_t consumer) {
  return iree_hal_memory_contract_query_transition(buffer->memory.contract,
                                                   producer, consumer);
}

IREE_API_EXPORT iree_hal_memory_pair_info_t
iree_hal_memory_transition_query_info(iree_hal_memory_transition_table_t table,
                                      iree_hal_memory_transition_pair_t pair) {
  const iree_hal_memory_transition_details_t* details =
      table.contract->transition_details;
  return details ? details->records[details->indices[pair.cell_index]].info
                 : (iree_hal_memory_pair_info_t){0};
}

IREE_API_EXPORT const iree_hal_memory_transition_recipe_t*
iree_hal_memory_transition_recipe(iree_hal_memory_transition_table_t table,
                                  iree_hal_memory_transition_pair_t pair,
                                  iree_hal_memory_transition_action_t action) {
  const iree_hal_memory_transition_details_t* details =
      table.contract->transition_details;
  if (!details) {
    return NULL;
  }
  const uint32_t shift = table.contract->transition_row_shift;
  const uint32_t producer = pair.cell_index >> shift;
  const uint32_t consumer = pair.cell_index & ((1u << shift) - 1);
  const iree_hal_memory_transition_recipe_t* recipe = NULL;
  if (consumer == 0 && action == IREE_HAL_MEMORY_TRANSITION_RELEASE) {
    recipe = &details->wildcards[producer * 2];
  } else if (producer == 0 && action == IREE_HAL_MEMORY_TRANSITION_ACQUIRE) {
    recipe = &details->wildcards[consumer * 2 + 1];
  } else {
    recipe =
        &details->records[details->indices[pair.cell_index]].recipes[action];
  }
  return recipe->operation_count ? recipe : NULL;
}

//===----------------------------------------------------------------------===//
// Cold qualification capture
//===----------------------------------------------------------------------===//

static uint32_t iree_hal_memory_transition_encode_effects(
    const iree_hal_memory_transition_recipe_info_t* info) {
  if (info->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    return IREE_HAL_MEMORY_EFFECT_UNSUPPORTED;
  } else if (info->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    return 0;
  } else if (info->kind != IREE_HAL_MEMORY_TRANSITION_KIND_RANGE &&
             info->kind != IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL) {
    return IREE_HAL_MEMORY_EFFECT_UNSUPPORTED;
  }
  uint32_t bits = 0;
  switch (info->operation) {
    case IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM:
      bits = info->kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE
                 ? IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM
                 : IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
      break;
    case IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM:
      bits = info->kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE
                 ? IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM
                 : IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM;
      break;
    case IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH:
      bits = IREE_HAL_MEMORY_EFFECT_HOST_FLUSH;
      break;
    case IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE:
      bits = IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE;
      break;
    default:
      return IREE_HAL_MEMORY_EFFECT_UNSUPPORTED;
  }
  if (info->executor == IREE_HAL_MEMORY_TRANSITION_EXECUTOR_PROGRAM) {
    bits |= IREE_HAL_MEMORY_EFFECT_PROGRAM_EXECUTOR;
  }
  return bits;
}

IREE_API_EXPORT iree_status_t iree_hal_memory_transition_recipe_validate(
    const iree_hal_memory_transition_recipe_t* recipe) {
  if (IREE_UNLIKELY(!recipe || !recipe->operation_count ||
                    !recipe->operations)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory transition recipe is empty");
  }
  uint32_t effects = 0;
  for (uint32_t i = 0; i < recipe->operation_count; ++i) {
    const iree_hal_memory_transition_recipe_info_t* info =
        &recipe->operations[i];
    const bool is_global = info->kind == IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL;
    if (IREE_UNLIKELY(!is_global &&
                      (info->kind != IREE_HAL_MEMORY_TRANSITION_KIND_RANGE ||
                       !info->range_granularity))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "memory transition recipe has no range or global action");
    }
    if (IREE_UNLIKELY(
            is_global &&
            (info->range_granularity ||
             (info->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API &&
              info->executor !=
                  IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT) ||
             info->host.instruction != IREE_HAL_HOST_CACHE_INSTRUCTION_NONE))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "global resource recipe requires a host API or fence-only action");
    }
    switch (info->executor) {
      case IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE:
      case IREE_HAL_MEMORY_TRANSITION_EXECUTOR_PROGRAM:
        if (IREE_UNLIKELY(
                (info->operation !=
                     IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM &&
                 info->operation !=
                     IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM) ||
                info->host.instruction !=
                    IREE_HAL_HOST_CACHE_INSTRUCTION_NONE ||
                info->host.fence_before != IREE_HAL_HOST_CACHE_FENCE_NONE ||
                info->host.fence_after != IREE_HAL_HOST_CACHE_FENCE_NONE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "queue/program recipe contains an invalid cache operation");
        }
        break;
      case IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT:
        if (IREE_UNLIKELY(!is_global && !iree_device_size_is_power_of_two(
                                            info->range_granularity))) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "direct host recipe requires a power-of-two range granularity");
        }
        if (IREE_UNLIKELY(
                info->operation !=
                    IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH &&
                info->operation !=
                    IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "host recipe requires flush or invalidate operation");
        }
        switch (info->host.instruction) {
          case IREE_HAL_HOST_CACHE_INSTRUCTION_NONE:
          case IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH:
          case IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSHOPT:
          case IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLWB:
            break;
          default:
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "direct host recipe has an invalid cache instruction");
        }
        switch (info->host.fence_before) {
          case IREE_HAL_HOST_CACHE_FENCE_NONE:
          case IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE:
          case IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE:
            break;
          default:
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "direct host recipe has an invalid leading fence");
        }
        switch (info->host.fence_after) {
          case IREE_HAL_HOST_CACHE_FENCE_NONE:
          case IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE:
          case IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE:
            break;
          default:
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "direct host recipe has an invalid trailing fence");
        }
        break;
      case IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API:
        if (IREE_UNLIKELY(
                info->operation !=
                    IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH &&
                info->operation !=
                    IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "host recipe requires flush or invalidate operation");
        }
        if (IREE_UNLIKELY(
                info->host.instruction !=
                    IREE_HAL_HOST_CACHE_INSTRUCTION_NONE ||
                info->host.fence_before != IREE_HAL_HOST_CACHE_FENCE_NONE ||
                info->host.fence_after != IREE_HAL_HOST_CACHE_FENCE_NONE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "host API recipe contains direct cache instructions");
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "memory transition recipe has no executor");
    }
    const uint32_t operation_effects =
        iree_hal_memory_transition_encode_effects(info);
    if (IREE_UNLIKELY(operation_effects & IREE_HAL_MEMORY_EFFECT_UNSUPPORTED)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory transition recipe is not representable");
    }
    effects |= operation_effects & IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK;
  }
  if (IREE_UNLIKELY(!effects || recipe->effects.bits != effects)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "memory transition recipe effects do not match its operations");
  }
  return iree_ok_status();
}

static uint64_t iree_hal_memory_pair_encode(
    const iree_hal_memory_pair_info_t* info) {
  if (!iree_any_bit_set(info->flags,
                        IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE)) {
    return iree_hal_memory_transition_pack_effects(
        IREE_HAL_MEMORY_EFFECT_UNSUPPORTED, IREE_HAL_MEMORY_EFFECT_UNSUPPORTED);
  }
  return iree_hal_memory_transition_pack_effects(
      iree_hal_memory_transition_encode_effects(&info->release),
      iree_hal_memory_transition_encode_effects(&info->acquire));
}

static bool iree_hal_memory_recipe_info_equal(
    const iree_hal_memory_transition_recipe_info_t* lhs,
    const iree_hal_memory_transition_recipe_info_t* rhs) {
  return lhs->kind == rhs->kind && lhs->executor == rhs->executor &&
         lhs->operation == rhs->operation &&
         lhs->range_granularity == rhs->range_granularity &&
         lhs->host.instruction == rhs->host.instruction &&
         lhs->host.fence_before == rhs->host.fence_before &&
         lhs->host.fence_after == rhs->host.fence_after;
}

static bool iree_hal_memory_pair_info_equal(
    const iree_hal_memory_pair_info_t* lhs,
    const iree_hal_memory_pair_info_t* rhs) {
  return lhs->flags == rhs->flags &&
         iree_hal_memory_recipe_info_equal(&lhs->release, &rhs->release) &&
         iree_hal_memory_recipe_info_equal(&lhs->acquire, &rhs->acquire) &&
         lhs->atomic_reach.scope_32 == rhs->atomic_reach.scope_32 &&
         lhs->atomic_reach.scope_64 == rhs->atomic_reach.scope_64 &&
         lhs->estimated_fixed_cost_nanoseconds ==
             rhs->estimated_fixed_cost_nanoseconds;
}

// Only the local side contributes to a wildcard. An inapplicable read-only
// producer or write-only consumer is omitted; an applicable UNKNOWN poisons
// the join. This preserves missing PROGRAM qualification instead of silently
// treating a queue-only native query as proof for executable program accesses.
static void iree_hal_memory_transition_join_wildcards(
    const iree_hal_memory_contract_t* contract, uint64_t* cells) {
  const uint32_t shift = contract->transition_row_shift;
  for (uint32_t local = 1; local < contract->scope_count; ++local) {
    if (iree_hal_memory_scope_can_write(contract, local)) {
      uint32_t release = 0;
      for (uint32_t remote = 1; remote < contract->scope_count; ++remote) {
        if (iree_hal_memory_scope_can_read(contract, remote)) {
          release |= (uint32_t)cells[(local << shift) + remote];
        }
      }
      cells[local << shift] = iree_hal_memory_transition_pack_effects(
          release, IREE_HAL_MEMORY_EFFECT_UNSUPPORTED);
    }
    if (iree_hal_memory_scope_can_read(contract, local)) {
      uint32_t acquire = 0;
      for (uint32_t remote = 1; remote < contract->scope_count; ++remote) {
        if (iree_hal_memory_scope_can_write(contract, remote)) {
          acquire |= (uint32_t)(cells[(remote << shift) + local] >> 32);
        }
      }
      cells[local] = iree_hal_memory_transition_pack_effects(
          IREE_HAL_MEMORY_EFFECT_UNSUPPORTED, acquire);
    }
  }
}

// Captures wildcard resource lists once.
static iree_host_size_t iree_hal_memory_transition_compose_wildcards(
    const iree_hal_memory_contract_t* contract, uint64_t* cells,
    const uint32_t* indices, const iree_hal_memory_pair_info_t* infos,
    iree_hal_memory_transition_recipe_t* recipes,
    iree_hal_memory_transition_recipe_info_t* operations) {
  iree_host_size_t operation_count = 0;
  const uint32_t shift = contract->transition_row_shift;
  for (uint32_t local = 1; local < contract->scope_count; ++local) {
    for (uint32_t action = 0; action < 2; ++action) {
      const uint32_t cell_index = action ? local : local << shift;
      const uint32_t bits = (uint32_t)(cells[cell_index] >> (action * 32));
      if (!iree_hal_memory_effects_requires_resources(
              (iree_hal_memory_effects_t){bits}) ||
          (bits & IREE_HAL_MEMORY_EFFECT_UNSUPPORTED)) {
        continue;
      }
      iree_hal_memory_transition_recipe_info_t* local_operations =
          &operations[operation_count];
      iree_host_size_t local_count = 0;
      uint32_t local_effects = 0;
      for (uint32_t remote = 1; remote < contract->scope_count; ++remote) {
        const uint32_t exact_index =
            action ? (remote << shift) + local : (local << shift) + remote;
        const uint32_t exact_bits =
            (uint32_t)(cells[exact_index] >> (action * 32));
        if (!iree_hal_memory_effects_requires_resources(
                (iree_hal_memory_effects_t){exact_bits})) {
          continue;
        }
        const iree_hal_memory_pair_info_t* pair = &infos[indices[exact_index]];
        const iree_hal_memory_transition_recipe_info_t* info =
            action ? &pair->acquire : &pair->release;
        iree_host_size_t i = 0;
        while (i < local_count &&
               !iree_hal_memory_recipe_info_equal(&local_operations[i], info)) {
          ++i;
        }
        if (i == local_count) {
          local_operations[local_count++] = *info;
        }
        local_effects |= exact_bits & IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK;
      }
      recipes[local * 2 + action] = (iree_hal_memory_transition_recipe_t){
          .effects = {local_effects},
          .operation_count = local_count,
          .operations = local_operations,
      };
      operation_count += local_count;
    }
  }
  return operation_count;
}

IREE_API_EXPORT iree_status_t iree_hal_memory_contract_initialize_transitions(
    iree_hal_memory_contract_t* contract, iree_hal_memory_pair_query_fn_t query,
    void* user_data) {
  const uint32_t cell_count = contract->scope_count
                              << contract->transition_row_shift;
  iree_host_size_t total_size = 0;
  iree_host_size_t indices_offset = 0;
  iree_host_size_t cells_offset = 0;
  iree_host_size_t infos_offset = 0;
  iree_host_size_t wildcards_offset = 0;
  iree_host_size_t operations_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size, IREE_STRUCT_FIELD(cell_count, uint32_t, &indices_offset),
      IREE_STRUCT_FIELD_ALIGNED(cell_count, uint64_t, iree_alignof(uint64_t),
                                &cells_offset),
      IREE_STRUCT_FIELD_ALIGNED(cell_count, iree_hal_memory_pair_info_t,
                                iree_alignof(iree_hal_memory_pair_info_t),
                                &infos_offset),
      IREE_STRUCT_ARRAY_FIELD(contract->scope_count, 2,
                              iree_hal_memory_transition_recipe_t,
                              &wildcards_offset),
      IREE_STRUCT_ARRAY_FIELD(cell_count, 2,
                              iree_hal_memory_transition_recipe_info_t,
                              &operations_offset)));
  uint8_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(contract->host_allocator,
                                             total_size, (void**)&scratch));
  uint32_t* indices = (uint32_t*)(scratch + indices_offset);
  uint64_t* cells = (uint64_t*)(scratch + cells_offset);
  iree_hal_memory_pair_info_t* infos =
      (iree_hal_memory_pair_info_t*)(scratch + infos_offset);
  iree_hal_memory_transition_recipe_t* wildcards =
      (iree_hal_memory_transition_recipe_t*)(scratch + wildcards_offset);
  iree_hal_memory_transition_recipe_info_t* operations =
      (iree_hal_memory_transition_recipe_info_t*)(scratch + operations_offset);
  memcpy(cells, contract->transitions, cell_count * sizeof(*cells));

  uint32_t info_count = 1;  // Record zero is the unqualified description.
  iree_status_t status = iree_ok_status();
  for (uint32_t producer = 1;
       producer < contract->scope_count && iree_status_is_ok(status);
       ++producer) {
    if (!iree_hal_memory_scope_can_write(contract, producer)) {
      continue;
    }
    for (uint32_t consumer = 1;
         consumer < contract->scope_count && iree_status_is_ok(status);
         ++consumer) {
      if (!iree_hal_memory_scope_can_read(contract, consumer)) {
        continue;
      }
      iree_hal_memory_pair_info_t info = {0};
      status = query(user_data, producer, consumer, &info);
      if (iree_status_is_ok(status)) {
        uint32_t index = 0;
        while (index < info_count &&
               !iree_hal_memory_pair_info_equal(&infos[index], &info)) {
          ++index;
        }
        if (index == info_count) {
          infos[info_count++] = info;
        }
        const uint32_t cell_index =
            (producer << contract->transition_row_shift) + consumer;
        indices[cell_index] = index;
        cells[cell_index] = iree_hal_memory_pair_encode(&info);
      }
    }
  }

  iree_host_size_t operation_count = 0;
  if (iree_status_is_ok(status)) {
    iree_hal_memory_transition_join_wildcards(contract, cells);
    operation_count = iree_hal_memory_transition_compose_wildcards(
        contract, cells, indices, infos, wildcards, operations);
  }
  iree_hal_memory_transition_details_t* details = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(*details), &total_size,
        IREE_STRUCT_FIELD_FAM(cell_count, uint32_t),
        IREE_STRUCT_FIELD_ALIGNED(info_count, iree_hal_memory_pair_record_t,
                                  iree_alignof(iree_hal_memory_pair_record_t),
                                  &infos_offset),
        IREE_STRUCT_ARRAY_FIELD(contract->scope_count, 2,
                                iree_hal_memory_transition_recipe_t,
                                &wildcards_offset),
        IREE_STRUCT_FIELD(operation_count,
                          iree_hal_memory_transition_recipe_info_t,
                          &operations_offset));
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(contract->host_allocator, total_size,
                                   (void**)&details);
  }
  if (iree_status_is_ok(status)) {
    details->records =
        (iree_hal_memory_pair_record_t*)((uint8_t*)details + infos_offset);
    details->wildcards =
        (iree_hal_memory_transition_recipe_t*)((uint8_t*)details +
                                               wildcards_offset);
    iree_hal_memory_transition_recipe_info_t* captured_operations =
        (iree_hal_memory_transition_recipe_info_t*)((uint8_t*)details +
                                                    operations_offset);
    memcpy(details->indices, indices, cell_count * sizeof(*indices));
    memcpy(captured_operations, operations,
           operation_count * sizeof(*operations));
    for (uint32_t i = 0; i < info_count; ++i) {
      iree_hal_memory_pair_record_t* record = &details->records[i];
      record->info = infos[i];
      const uint64_t cell = iree_hal_memory_pair_encode(&record->info);
      for (uint32_t action = 0; action < 2; ++action) {
        const uint32_t bits = (uint32_t)(cell >> (action * 32));
        if (iree_hal_memory_effects_requires_resources(
                (iree_hal_memory_effects_t){bits}) &&
            !(bits & IREE_HAL_MEMORY_EFFECT_UNSUPPORTED)) {
          record->recipes[action] = (iree_hal_memory_transition_recipe_t){
              .effects = {bits & IREE_HAL_MEMORY_EFFECT_RESOURCE_MASK},
              .operation_count = 1,
              .operations =
                  action ? &record->info.acquire : &record->info.release,
          };
        }
      }
    }
    for (uint32_t i = 0; i < contract->scope_count * 2; ++i) {
      if (!wildcards[i].operation_count) {
        continue;
      }
      details->wildcards[i] = (iree_hal_memory_transition_recipe_t){
          .effects = wildcards[i].effects,
          .operation_count = wildcards[i].operation_count,
          .operations =
              captured_operations + (wildcards[i].operations - operations),
      };
    }
    contract->transition_details = details;
    memcpy(contract->transitions, cells, cell_count * sizeof(*cells));
  }
  iree_allocator_free(contract->host_allocator, scratch);
  return status;
}

//===----------------------------------------------------------------------===//
// Explicit host mapping transitions
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_buffer_mapping_transition_validate(
    const iree_hal_buffer_mapping_transition_t* transition,
    uint32_t* out_effect_bits) {
  *out_effect_bits = 0;
  if (!transition->mapping || !transition->mapping->buffer ||
      !transition->recipe) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host transition requires a live mapping and recipe");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_memory_transition_recipe_validate(transition->recipe));
  const iree_hal_buffer_mapping_t* mapping = transition->mapping;
  iree_device_size_t offset = 0;
  iree_device_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      0, mapping->contents.data_length, transition->offset, transition->length,
      &offset, &length));
  if (length && !mapping->contents.data) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "host transition mapping is not committed");
  }
  if (offset > UINTPTR_MAX - (uintptr_t)mapping->contents.data ||
      length > UINTPTR_MAX - ((uintptr_t)mapping->contents.data + offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "host transition address range overflows");
  }
  for (iree_host_size_t i = 0; i < transition->recipe->operation_count; ++i) {
    const iree_hal_memory_transition_recipe_info_t* info =
        &transition->recipe->operations[i];
    if (info->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API &&
        info->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "transition recipe does not execute on the host");
    }
    iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_NONE;
    if (info->operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH) {
      access = IREE_HAL_MEMORY_ACCESS_WRITE;
    } else if (info->operation ==
               IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE) {
      access = IREE_HAL_MEMORY_ACCESS_READ;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "transition recipe is not host cache maintenance");
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_validate_access(mapping->impl.allowed_access, access));
    *out_effect_bits |= iree_hal_memory_transition_encode_effects(info);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_mapping_memory_barrier(
    iree_hal_memory_effects_t effects, iree_host_size_t mapping_count,
    const iree_hal_buffer_mapping_transition_t* mappings) {
  if (!iree_hal_memory_effects_is_supported(effects)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "host transition is not qualified for this backing");
  }
  const uint32_t supported_bits = IREE_HAL_MEMORY_EFFECT_HOST_FLUSH |
                                  IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE;
  if (effects.bits & ~supported_bits) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host barrier cannot execute queue/program effects");
  }
  if (mapping_count && !mappings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host barrier requires its mapping array");
  }
  uint32_t covered_bits = 0;
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    uint32_t bits = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_mapping_transition_validate(&mappings[i], &bits));
    covered_bits |= bits;
  }
  if (covered_bits != effects.bits) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host barrier recipes do not cover its effects");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, mapping_count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < mapping_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_mapping_transition_t* transition = &mappings[i];
    const iree_device_size_t length =
        transition->length == IREE_HAL_WHOLE_BUFFER
            ? transition->mapping->contents.data_length - transition->offset
            : transition->length;
    for (iree_host_size_t j = 0;
         j < transition->recipe->operation_count && iree_status_is_ok(status);
         ++j) {
      const iree_hal_memory_transition_recipe_info_t* info =
          &transition->recipe->operations[j];
      if (info->executor == IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API) {
        if (info->operation ==
            IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH) {
          status = iree_hal_buffer_mapping_flush_range(
              transition->mapping, transition->offset, length);
        } else {
          status = iree_hal_buffer_mapping_invalidate_range(
              transition->mapping, transition->offset, length);
        }
      } else {
        if (length) {
          iree_hal_memory_transition_execute_host_direct(
              info, transition->mapping->contents.data + transition->offset,
              length);
        }
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
