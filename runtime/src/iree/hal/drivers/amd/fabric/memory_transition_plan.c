// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory_transition_plan.h"

#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/status.h"

static iree_host_size_t iree_hal_amd_fabric_memory_owned_route_count(
    const iree_hal_amd_fabric_memory_transition_query_t* query) {
  return iree_all_bits_set(query->provider_options->memory.profile.roles,
                           AMDF_MEMORY_PROFILE_ROLE_CREATE)
             ? 1
             : 0;
}

// Completes one correlated acquisition route without changing its consumers.
static const amdf_memory_profile_t*
iree_hal_amd_fabric_memory_select_transition_route(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_host_size_t index) {
  const iree_hal_amd_fabric_slab_provider_options_t* options =
      query->provider_options;
  const iree_host_size_t owned_count =
      iree_hal_amd_fabric_memory_owned_route_count(query);
  const iree_hal_amd_fabric_import_route_t* import =
      index < owned_count ? NULL : &options->imports[index - owned_count];
  const amdf_memory_profile_t* profile =
      import ? &import->profile : &options->memory.profile;
  query->native.memory_profile_ordinal = profile->ordinal;
  query->native.required_flags =
      import ? import->required_flags : options->memory.required_flags;
  query->native.accesses = import && options->import_accesses
                               ? options->import_accesses
                               : options->memory.accesses;
  query->native.registered_host_properties =
      profile->registration.registered_host_properties;
  query->native.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_NONE;
  query->native.external_memory_provenance =
      (amdf_external_memory_provenance_t){{0, 0}};
  if (import && import->policy.storage_class.type !=
                    IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    const iree_hal_external_storage_class_t storage_class =
        import->policy.storage_class;
    query->native.external_memory_type =
        iree_hal_amd_fabric_memory_external_type(storage_class.type);
    query->native.external_memory_provenance =
        (amdf_external_memory_provenance_t){
            {storage_class.provenance[0], storage_class.provenance[1]}};
  }
  return profile;
}

iree_status_t iree_hal_amd_fabric_memory_capture_copy_protocols(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_contract_t* contract) {
  contract->backend_type = IREE_HAL_MEMORY_BACKEND_AMDF;
  bool has_copy = false;
  for (uint32_t id = 1; id < contract->scope_count; ++id) {
    has_copy |=
        query->sites[id].kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE &&
        query->sites[id].copy_family_ordinal != UINT32_MAX;
  }
  if (!has_copy) {
    return iree_ok_status();
  }
  iree_hal_amd_fabric_memory_copy_protocol_t* protocols = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      contract->host_allocator, contract->scope_count, sizeof(*protocols),
      (void**)&protocols));
  contract->backend_data = protocols;
  const iree_hal_memory_effects_t unsupported = {
      IREE_HAL_MEMORY_EFFECT_UNSUPPORTED};
  const iree_host_size_t route_count =
      iree_hal_amd_fabric_memory_owned_route_count(query) +
      query->provider_options->import_count;
  iree_status_t status = iree_ok_status();
  for (uint32_t id = 0; id < contract->scope_count && iree_status_is_ok(status);
       ++id) {
    protocols[id] = (iree_hal_amd_fabric_memory_copy_protocol_t){
        .begin = {unsupported, unsupported}, .end = {unsupported, unsupported}};
    const iree_hal_amd_fabric_memory_transition_site_t* site =
        &query->sites[id];
    if (site->kind != IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE ||
        site->copy_family_ordinal == UINT32_MAX || !route_count) {
      continue;
    }
    amdf_memory_profile_site_t copy = site->native;
    copy.value.device.queue_family_ordinal = site->copy_family_ordinal;
    for (uint32_t edge = 0; edge < 2 && iree_status_is_ok(status); ++edge) {
      iree_hal_memory_transition_t* effects =
          edge ? &protocols[id].end : &protocols[id].begin;
      *effects = (iree_hal_memory_transition_t){0};
      query->native.producer = edge ? copy : site->native;
      query->native.consumer = edge ? site->native : copy;
      for (iree_host_size_t route = 0;
           route < route_count && iree_status_is_ok(status); ++route) {
        iree_hal_amd_fabric_memory_select_transition_route(query, route);
        amdf_memory_pair_info_t native = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            .structure_size = sizeof(native)};
        const amdf_status_t result = query->api->memory_scope_query_pair_info(
            query->provider_options->memory.scope, &query->native, &native);
        if (result == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          *effects = (iree_hal_memory_transition_t){unsupported, unsupported};
          break;
        }
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(result,
                                               "memory_scope_query_pair_info");
        iree_hal_memory_pair_info_t pair = {0};
        if (iree_status_is_ok(status)) {
          status = iree_hal_amd_fabric_memory_translate_pair(&native, &pair);
        }
        if (iree_status_is_ok(status)) {
          const iree_hal_memory_transition_t transition =
              iree_hal_amd_fabric_memory_queue_transition(&pair);
          effects->release = iree_hal_memory_effects_combine(
              effects->release, transition.release);
          effects->acquire = iree_hal_memory_effects_combine(
              effects->acquire, transition.acquire);
        }
      }
    }
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_memory_capture_queue_atomics(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_scope_id_t site_id,
    iree_hal_atomic_operation_capabilities_t* out_operations) {
  *out_operations = (iree_hal_atomic_operation_capabilities_t){0};
  const iree_hal_amd_fabric_memory_transition_site_t* site =
      &query->sites[site_id];
  query->native.producer = query->native.consumer = site->native;
  const iree_host_size_t owned_count =
      iree_hal_amd_fabric_memory_owned_route_count(query);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < owned_count + query->provider_options->import_count &&
       iree_status_is_ok(status);
       ++i) {
    const amdf_memory_profile_t* profile =
        iree_hal_amd_fabric_memory_select_transition_route(query, i);
    const amdf_memory_access_capabilities_t* access =
        (i < owned_count ? query->owned_capabilities
                         : query->profile_capabilities +
                               (iree_host_size_t)profile->ordinal *
                                   query->native.access_count) +
        site->native.value.device.access_ordinal;
    if (!access->atomic_operations_32 && !access->atomic_operations_64) {
      *out_operations = (iree_hal_atomic_operation_capabilities_t){0};
      break;
    }
    amdf_memory_pair_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(native),
    };
    const amdf_status_t native_status =
        query->api->memory_scope_query_pair_info(
            query->provider_options->memory.scope, &query->native, &native);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      *out_operations = (iree_hal_atomic_operation_capabilities_t){0};
      break;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_pair_info");
    if (iree_status_is_ok(status)) {
      const iree_hal_atomic_operation_capabilities_t operations =
          iree_hal_amd_fabric_memory_translate_atomics(
              access->atomic_operations_32, access->atomic_operations_64,
              native.atomic_reach);
      *out_operations = !i ? operations
                           : iree_hal_atomic_operation_capabilities_intersect(
                                 *out_operations, operations);
    }
  }
  return status;
}

static void iree_hal_amd_fabric_memory_accumulate_pair(
    const iree_hal_amd_fabric_memory_transition_site_t* producer,
    const iree_hal_amd_fabric_memory_transition_site_t* consumer,
    const iree_hal_memory_pair_info_t* pair,
    iree_hal_amd_fabric_memory_transition_query_t* query) {
  if (!iree_all_bits_set(pair->flags,
                         IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE) ||
      pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    query->complete = false;
    return;
  }
  if (pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE) {
    query->maintenance_alignment =
        iree_max(query->maintenance_alignment, pair->release.range_granularity);
  }
  if (pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE) {
    query->maintenance_alignment =
        iree_max(query->maintenance_alignment, pair->acquire.range_granularity);
  }
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    query->host_coherent &=
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
  if (consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    query->host_coherent &=
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
  if (query->reservation_alias) {
    return;
  }
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
      consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE) {
    query->complete &=
        !consumer->requires_host_coherent ||
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  } else if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE &&
             consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    query->complete &=
        !producer->requires_host_coherent ||
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  } else if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
             consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
             (producer->requires_host_coherent ||
              consumer->requires_host_coherent)) {
    query->complete &=
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE &&
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
}

typedef struct iree_hal_amd_fabric_memory_atomic_executor_t {
  // Native issuer of the actual memory transactions.
  amdf_memory_profile_site_t native;
  // Disjoint semantic operation subset using this issuer.
  iree_hal_atomic_operation_capabilities_t operations;
} iree_hal_amd_fabric_memory_atomic_executor_t;

static iree_hal_atomic_operation_capabilities_t
iree_hal_amd_fabric_memory_atomic_difference(
    iree_hal_atomic_operation_capabilities_t left,
    iree_hal_atomic_operation_capabilities_t right) {
  return (iree_hal_atomic_operation_capabilities_t){
      .device_scope_32 = left.device_scope_32 & ~right.device_scope_32,
      .device_scope_64 = left.device_scope_64 & ~right.device_scope_64,
      .system_scope_32 = left.system_scope_32 & ~right.system_scope_32,
      .system_scope_64 = left.system_scope_64 & ~right.system_scope_64,
  };
}

static uint32_t iree_hal_amd_fabric_memory_atomic_executors(
    const iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_scope_id_t id,
    iree_hal_amd_fabric_memory_atomic_executor_t executors[2]) {
  const iree_hal_amd_fabric_memory_transition_site_t* site = &query->sites[id];
  executors[0] = (iree_hal_amd_fabric_memory_atomic_executor_t){
      .native = site->native,
      .operations = site->atomics,
  };
  if (site->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
      query->reservation_alias) {
    executors[0].native.flags |= AMDF_MEMORY_PROFILE_SITE_FLAG_HOST_RESERVATION;
  }
  if (site->kind != IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE ||
      !site->program_atomics) {
    return 1;
  }
  const amdf_memory_profile_site_t program = {
      .kind = AMDF_MEMORY_SITE_KIND_PROGRAM,
      .value.program.access_ordinal = site->native.value.device.access_ordinal,
  };
  if (query->reservation_alias) {
    // Virtual queue controls have a fixed shader recipe across all mappings.
    executors[0].native = program;
    executors[0].operations = iree_hal_atomic_operation_capabilities_intersect(
        site->atomics, query->sites[id + 1].atomics);
    return 1;
  }
  executors[0].operations = site->queue_atomics;
  executors[1] = (iree_hal_amd_fabric_memory_atomic_executor_t){
      .native = program,
      .operations = iree_hal_amd_fabric_memory_atomic_difference(
          site->atomics, site->queue_atomics),
  };
  return 2;
}

static iree_hal_atomic_operation_capabilities_t
iree_hal_amd_fabric_memory_translate_pair_atomics(
    amdf_memory_site_kind_t kind, amdf_memory_site_kind_t peer_kind,
    amdf_memory_atomic_capabilities_t operations) {
  if (kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    // CP controls always use their qualified native coherence domain. They
    // can satisfy a narrower HAL request, but a PROGRAM peer still emits the
    // requested shader scope. Native SYSTEM/SYSTEM is not SYSTEM/DEVICE.
    iree_hal_atomic_operation_capabilities_t result =
        iree_hal_amd_fabric_memory_translate_atomics(
            operations.system_scope_32, operations.system_scope_64,
            (amdf_atomic_reach_t){AMDF_ATOMIC_SCOPE_SYSTEM,
                                  AMDF_ATOMIC_SCOPE_SYSTEM});
    const iree_hal_atomic_operation_capabilities_t device =
        iree_hal_amd_fabric_memory_translate_atomics(
            operations.device_scope_32 | operations.fabric_scope_32 |
                (peer_kind == AMDF_MEMORY_SITE_KIND_PROGRAM
                     ? 0
                     : operations.system_scope_32),
            operations.device_scope_64 | operations.fabric_scope_64 |
                (peer_kind == AMDF_MEMORY_SITE_KIND_PROGRAM
                     ? 0
                     : operations.system_scope_64),
            (amdf_atomic_reach_t){AMDF_ATOMIC_SCOPE_DEVICE,
                                  AMDF_ATOMIC_SCOPE_DEVICE});
    result.device_scope_32 = device.device_scope_32;
    result.device_scope_64 = device.device_scope_64;
    return result;
  }
  if (kind == AMDF_MEMORY_SITE_KIND_HOST &&
      peer_kind != AMDF_MEMORY_SITE_KIND_PROGRAM) {
    operations.device_scope_32 = operations.system_scope_32;
    operations.device_scope_64 = operations.system_scope_64;
  }
  return iree_hal_amd_fabric_memory_translate_program_atomics(&operations);
}

// Keeps operations of this executor only when the exact native pair qualifies
// them. Other executors' disjoint operations are unaffected. Applying this to
// both sides for every executor combination preserves a Cartesian guarantee;
// merely unioning the native rectangles would invent cross-executor edges.
static void iree_hal_amd_fabric_memory_restrict_pair_atomics(
    iree_hal_atomic_operation_capabilities_t executor,
    iree_hal_atomic_operation_capabilities_t peer,
    iree_hal_atomic_operation_capabilities_t qualified,
    iree_hal_atomic_operation_capabilities_t* operations) {
  executor.device_scope_32 =
      peer.device_scope_32 ? executor.device_scope_32 : 0;
  executor.device_scope_64 =
      peer.device_scope_64 ? executor.device_scope_64 : 0;
  executor.system_scope_32 =
      peer.system_scope_32 ? executor.system_scope_32 : 0;
  executor.system_scope_64 =
      peer.system_scope_64 ? executor.system_scope_64 : 0;
  *operations = iree_hal_atomic_operation_capabilities_union(
      iree_hal_amd_fabric_memory_atomic_difference(*operations, executor),
      iree_hal_atomic_operation_capabilities_intersect(*operations, qualified));
}

static iree_status_t iree_hal_amd_fabric_memory_capture_pair_atomics(
    iree_hal_amd_fabric_memory_transition_query_t* query,
    iree_hal_memory_scope_id_t producer_id,
    iree_hal_memory_scope_id_t consumer_id,
    const amdf_memory_pair_atomics_t* base_atomics,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_amd_fabric_memory_atomic_executor_t producers[2], consumers[2];
  const uint32_t producer_count = iree_hal_amd_fabric_memory_atomic_executors(
      query, producer_id, producers);
  const uint32_t consumer_count = iree_hal_amd_fabric_memory_atomic_executors(
      query, consumer_id, consumers);
  for (uint32_t i = 0; i < producer_count; ++i) {
    out_info->producer_atomics = iree_hal_atomic_operation_capabilities_union(
        out_info->producer_atomics, producers[i].operations);
  }
  for (uint32_t i = 0; i < consumer_count; ++i) {
    out_info->consumer_atomics = iree_hal_atomic_operation_capabilities_union(
        out_info->consumer_atomics, consumers[i].operations);
  }
  // CPU aliases share the qualified host atomic domain regardless of which
  // GPU supplied the physical pages. Native GPU/CPU RMW is a different edge.
  if (producers[0].native.kind == AMDF_MEMORY_SITE_KIND_HOST &&
      consumers[0].native.kind == AMDF_MEMORY_SITE_KIND_HOST) {
    return iree_ok_status();
  }
  amdf_memory_profile_pair_query_t native_query = query->native;
  iree_status_t status = iree_ok_status();
  for (uint32_t p = 0; p < producer_count && iree_status_is_ok(status); ++p) {
    if (iree_hal_atomic_operation_capabilities_is_empty(
            producers[p].operations)) {
      continue;
    }
    for (uint32_t c = 0; c < consumer_count && iree_status_is_ok(status); ++c) {
      if (iree_hal_atomic_operation_capabilities_is_empty(
              consumers[c].operations)) {
        continue;
      }
      native_query.producer = producers[p].native;
      native_query.consumer = consumers[c].native;
      amdf_memory_pair_atomics_t atomics = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_ATOMICS,
          .structure_size = sizeof(atomics),
      };
      if (native_query.producer.kind == query->native.producer.kind &&
          native_query.consumer.kind == query->native.consumer.kind) {
        atomics = *base_atomics;
      } else {
        amdf_memory_pair_info_t native = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
            .structure_size = sizeof(native),
            .next = &atomics,
        };
        const amdf_status_t native_status =
            query->api->memory_scope_query_pair_info(
                query->provider_options->memory.scope, &native_query, &native);
        if (native_status !=
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          status = IREE_HAL_AMD_STATUS_FROM_AMDF(
              native_status, "memory_scope_query_pair_info");
        }
      }
      if (iree_status_is_ok(status)) {
        iree_hal_amd_fabric_memory_restrict_pair_atomics(
            producers[p].operations, consumers[c].operations,
            iree_hal_amd_fabric_memory_translate_pair_atomics(
                native_query.producer.kind, native_query.consumer.kind,
                atomics.producer),
            &out_info->producer_atomics);
        iree_hal_amd_fabric_memory_restrict_pair_atomics(
            consumers[c].operations, producers[p].operations,
            iree_hal_amd_fabric_memory_translate_pair_atomics(
                native_query.consumer.kind, native_query.producer.kind,
                atomics.consumer),
            &out_info->consumer_atomics);
      }
    }
  }
  return status;
}

iree_status_t iree_hal_amd_fabric_memory_query_transition(
    void* user_data, iree_hal_memory_scope_id_t producer_id,
    iree_hal_memory_scope_id_t consumer_id,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_amd_fabric_memory_transition_query_t* query = user_data;
  const iree_hal_amd_fabric_memory_transition_site_t* producer =
      &query->sites[producer_id];
  const iree_hal_amd_fabric_memory_transition_site_t* consumer =
      &query->sites[consumer_id];
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN ||
      consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN) {
    return iree_ok_status();
  }
  query->native.producer = producer->native;
  query->native.consumer = consumer->native;
  if (query->reservation_alias) {
    if (query->native.producer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      query->native.producer.flags |=
          AMDF_MEMORY_PROFILE_SITE_FLAG_HOST_RESERVATION;
    }
    if (query->native.consumer.kind == AMDF_MEMORY_SITE_KIND_HOST) {
      query->native.consumer.flags |=
          AMDF_MEMORY_PROFILE_SITE_FLAG_HOST_RESERVATION;
    }
  }
  const iree_host_size_t route_count =
      iree_hal_amd_fabric_memory_owned_route_count(query) +
      query->provider_options->import_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < route_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amd_fabric_memory_select_transition_route(query, i);
    amdf_memory_pair_atomics_t atomics = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_ATOMICS,
        .structure_size = sizeof(atomics),
    };
    amdf_memory_pair_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(native),
        .next = &atomics,
    };
    const amdf_status_t native_status =
        query->api->memory_scope_query_pair_info(
            query->provider_options->memory.scope, &query->native, &native);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      query->complete = false;
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_pair_info");
    iree_hal_memory_pair_info_t info = {0};
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_memory_translate_pair(&native, &info);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_memory_capture_pair_atomics(
          query, producer_id, consumer_id, &atomics, &info);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_amd_fabric_memory_accumulate_pair(producer, consumer, &info,
                                                 query);
      if (!i) {
        *out_info = info;
      } else {
        out_info->flags &= info.flags;
        out_info->atomic_reach.scope_32 = iree_min(
            out_info->atomic_reach.scope_32, info.atomic_reach.scope_32);
        out_info->atomic_reach.scope_64 = iree_min(
            out_info->atomic_reach.scope_64, info.atomic_reach.scope_64);
        out_info->producer_atomics =
            iree_hal_atomic_operation_capabilities_intersect(
                out_info->producer_atomics, info.producer_atomics);
        out_info->consumer_atomics =
            iree_hal_atomic_operation_capabilities_intersect(
                out_info->consumer_atomics, info.consumer_atomics);
        out_info->estimated_fixed_cost_nanoseconds =
            iree_max(out_info->estimated_fixed_cost_nanoseconds,
                     info.estimated_fixed_cost_nanoseconds);
        query->complete &= iree_hal_memory_transition_recipe_join(
            &out_info->release, &info.release);
        query->complete &= iree_hal_memory_transition_recipe_join(
            &out_info->acquire, &info.acquire);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    if (out_info->producer_atomics.system_scope_32 &&
        out_info->consumer_atomics.system_scope_32) {
      out_info->atomic_reach.scope_32 = IREE_HAL_ATOMIC_REACH_SYSTEM;
    } else if (out_info->producer_atomics.device_scope_32 &&
               out_info->consumer_atomics.device_scope_32) {
      out_info->atomic_reach.scope_32 = iree_max(
          out_info->atomic_reach.scope_32, IREE_HAL_ATOMIC_REACH_DEVICE);
    }
    if (out_info->producer_atomics.system_scope_64 &&
        out_info->consumer_atomics.system_scope_64) {
      out_info->atomic_reach.scope_64 = IREE_HAL_ATOMIC_REACH_SYSTEM;
    } else if (out_info->producer_atomics.device_scope_64 &&
               out_info->consumer_atomics.device_scope_64) {
      out_info->atomic_reach.scope_64 = iree_max(
          out_info->atomic_reach.scope_64, IREE_HAL_ATOMIC_REACH_DEVICE);
    }
  }
  return status;
}
