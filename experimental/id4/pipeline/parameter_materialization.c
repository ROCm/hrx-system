// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_materialization.h"

#include <string.h>

#include "experimental/id4/pipeline/plan.h"
#include "iree/base/internal/atomics.h"

typedef uint32_t id4_pipeline_parameter_materialization_state_t;

enum id4_pipeline_parameter_materialization_state_e {
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED = 1u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_PUBLISHED = 2u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRING = 3u,
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRED = 4u,
};

typedef uint32_t id4_pipeline_parameter_materialization_list_flags_t;

enum id4_pipeline_parameter_materialization_list_flag_bits_e {
  ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY = 1u << 0,
};

typedef struct id4_pipeline_parameter_materialization_edge_t {
  // Number of retained timeline points.
  iree_host_size_t count;
  // Timeline semaphores retained in |storage|.
  iree_hal_semaphore_t** semaphores;
  // Payload values paired with |semaphores| and stored in |storage|.
  uint64_t* payload_values;
  // Packed semaphore and payload storage owned by this edge.
  void* storage;
} id4_pipeline_parameter_materialization_edge_t;

struct id4_pipeline_parameter_materialization_t {
  // Reference count for shared materialization ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for object and readiness storage.
  iree_allocator_t host_allocator;
  // Current materialization lifecycle state.
  id4_pipeline_parameter_materialization_state_t state;
  // Plan retaining the parameter-domain layout and device group.
  const id4_pipeline_plan_t* plan;
  // Immutable slabs with the target domain replaced.
  id4_pipeline_parameter_slab_set_t* derived_parameter_slabs;
  // Plan-local slab index replaced by the materialization.
  iree_host_size_t target_slab_index;
  // Device owning the replacement allocation; retained by |plan|.
  iree_hal_device_t* device;
  // Queue affinity used for materialization operations.
  iree_hal_queue_affinity_t queue_affinity;
  // Optional allocation pool retained for the replacement lifetime.
  iree_hal_pool_t* allocation_pool;
  // Queue-allocated replacement buffer owned by this object.
  iree_hal_buffer_t* target_buffer;
  // Acquisition edge retained while the replacement may be populated.
  id4_pipeline_parameter_materialization_edge_t acquisition_edge;
  // Publication edge retained while the replacement may be bound.
  id4_pipeline_parameter_materialization_edge_t publication_edge;
  // Retirement edge retained until the replacement allocation is reusable.
  id4_pipeline_parameter_materialization_edge_t retirement_edge;
  // True while this object owns the target allocation preserve.
  bool owns_target_allocation;
};

static iree_status_t id4_pipeline_parameter_materialization_validate_list(
    iree_hal_semaphore_list_t list, iree_string_view_t list_name,
    id4_pipeline_parameter_materialization_list_flags_t flags) {
  if (iree_any_bit_set(
          flags,
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY) &&
      list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore list must not be empty",
                            (int)list_name.size, list_name.data);
  }
  if (list.count == 0) return iree_ok_status();
  if (!list.semaphores || !list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore list is incomplete",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (!list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_materialization_emit(
    const id4_pipeline_parameter_materialization_t* materialization,
    iree_string_view_t key, iree_string_view_t message,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_diagnostic_event_t event;
  memset(&event, 0, sizeof(event));
  event.kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE;
  event.stage_name = id4_pipeline_plan_stage_name(materialization->plan);
  event.key = key;
  event.message = message;
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_hal_semaphore_list_t
id4_pipeline_parameter_materialization_edge_list(
    const id4_pipeline_parameter_materialization_edge_t* edge) {
  return (iree_hal_semaphore_list_t){
      // Number of retained timeline points.
      .count = edge->count,
      // Timeline semaphores retained by the edge.
      .semaphores = edge->semaphores,
      // Payload values paired with the timeline semaphores.
      .payload_values = edge->payload_values,
  };
}

static void id4_pipeline_parameter_materialization_edge_deinitialize(
    id4_pipeline_parameter_materialization_edge_t* edge,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < edge->count; ++i) {
    iree_hal_semaphore_release(edge->semaphores[i]);
  }
  iree_allocator_free(host_allocator, edge->storage);
  memset(edge, 0, sizeof(*edge));
}

static iree_status_t id4_pipeline_parameter_materialization_edge_initialize(
    iree_hal_semaphore_list_t source, iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_edge_t* out_edge) {
  memset(out_edge, 0, sizeof(*out_edge));
  iree_host_size_t semaphores_offset = 0;
  iree_host_size_t payload_values_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD(source.count, iree_hal_semaphore_t*,
                        &semaphores_offset),
      IREE_STRUCT_FIELD(source.count, uint64_t, &payload_values_offset)));

  void* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, &storage));
  iree_hal_semaphore_t** semaphores =
      (iree_hal_semaphore_t**)((uint8_t*)storage + semaphores_offset);
  uint64_t* payload_values =
      (uint64_t*)((uint8_t*)storage + payload_values_offset);
  for (iree_host_size_t i = 0; i < source.count; ++i) {
    semaphores[i] = source.semaphores[i];
    payload_values[i] = source.payload_values[i];
    iree_hal_semaphore_retain(semaphores[i]);
  }
  out_edge->count = source.count;
  out_edge->semaphores = semaphores;
  out_edge->payload_values = payload_values;
  out_edge->storage = storage;
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_materialization_edge_query_reached(
    const id4_pipeline_parameter_materialization_edge_t* edge,
    bool* out_reached) {
  *out_reached = false;
  for (iree_host_size_t i = 0; i < edge->count; ++i) {
    uint64_t current_payload_value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_query(edge->semaphores[i], &current_payload_value));
    if (current_payload_value < edge->payload_values[i])
      return iree_ok_status();
  }
  *out_reached = true;
  return iree_ok_status();
}

static void id4_pipeline_parameter_materialization_destroy(
    id4_pipeline_parameter_materialization_t* materialization) {
  if (materialization->owns_target_allocation ||
      materialization->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRED) {
    iree_abort();
  }
  iree_allocator_t host_allocator = materialization->host_allocator;
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->retirement_edge, host_allocator);
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->publication_edge, host_allocator);
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->acquisition_edge, host_allocator);
  iree_hal_buffer_release(materialization->target_buffer);
  id4_pipeline_parameter_slab_set_release(
      materialization->derived_parameter_slabs);
  iree_hal_pool_release(materialization->allocation_pool);
  id4_pipeline_plan_release((id4_pipeline_plan_t*)materialization->plan);
  iree_allocator_free(host_allocator, materialization);
}

static iree_status_t id4_pipeline_parameter_materialization_validate_options(
    const id4_pipeline_parameter_materialization_acquire_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter materialization options structure is too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter materialization extension structures are not supported");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization plan is required");
  }
  if (!options->base_parameter_slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter materialization base slab set is required");
  }
  if (iree_all_bits_set(options->alloca_flags,
                        IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter materialization requires determinate allocation lifetime");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      options->wait_semaphore_list, IREE_SV("materialization acquire wait"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      options->signal_semaphore_list, IREE_SV("materialization acquire signal"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("parameter materialization"));
}

static iree_status_t id4_pipeline_parameter_materialization_cleanup_failed(
    id4_pipeline_parameter_materialization_t* materialization) {
  iree_status_t status = iree_ok_status();
  if (materialization->owns_target_allocation) {
    // Acquisition succeeded but construction did not escape. Wait only on this
    // error path, then let final buffer release synchronously reclaim storage.
    status = iree_hal_semaphore_list_wait(
        id4_pipeline_parameter_materialization_edge_list(
            &materialization->acquisition_edge),
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    materialization->owns_target_allocation = false;
  }
  materialization->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRED;
  return status;
}

iree_status_t id4_pipeline_parameter_materialization_acquire(
    const id4_pipeline_parameter_materialization_acquire_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_t** out_materialization) {
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_materialization_validate_options(options));
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_validate_parameter_slabs(
      options->plan, options->base_parameter_slabs));

  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(options->plan);
  if (options->target_slab_index >= slab_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "materialization slab index %" PRIhsz
                            " is outside plan slab count %" PRIhsz,
                            options->target_slab_index, slab_count);
  }
  const id4_pipeline_parameter_slab_plan_t* slab_plan =
      id4_pipeline_plan_parameter_slab_at(options->plan,
                                          options->target_slab_index);
  const id4_pipeline_parameter_request_table_t* request_table =
      id4_pipeline_plan_parameter_request_table_at(options->plan,
                                                   options->target_slab_index);
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(options->plan, slab_plan->placement_id);
  if (!placement) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "materialization slab references missing placement");
  }
  iree_hal_device_t* device = iree_hal_device_group_device_at(
      id4_pipeline_plan_device_group(options->plan), placement->device_index);
  if (!device) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "materialization placement references missing device");
  }

  id4_pipeline_parameter_materialization_t* materialization = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*materialization), (void**)&materialization));
  memset(materialization, 0, sizeof(*materialization));
  iree_atomic_ref_count_init(&materialization->ref_count);
  materialization->host_allocator = host_allocator;
  materialization->state =
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED;
  materialization->plan = options->plan;
  id4_pipeline_plan_retain((id4_pipeline_plan_t*)materialization->plan);
  materialization->target_slab_index = options->target_slab_index;
  materialization->device = device;
  materialization->queue_affinity = placement->queue_affinity;
  materialization->allocation_pool = options->allocation_pool;
  iree_hal_pool_retain(materialization->allocation_pool);
  iree_status_t status = id4_pipeline_parameter_materialization_emit(
      materialization, IREE_SV("parameter.materialization.acquire"),
      IREE_SV("acquiring parameter-domain storage"), options->diagnostics_sink);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_edge_initialize(
        options->signal_semaphore_list, host_allocator,
        &materialization->acquisition_edge);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        device, placement->queue_affinity, options->wait_semaphore_list,
        options->signal_semaphore_list, options->allocation_pool,
        slab_plan->target_params, slab_plan->byte_length, options->alloca_flags,
        &materialization->target_buffer);
    materialization->owns_target_allocation = iree_status_is_ok(status);
  }

  id4_pipeline_parameter_slab_set_t* replacement_slabs = NULL;
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_slab_load_t load = {
        // The replacement source set contains one local slab.
        .slab_index = 0,
        // Exact target-domain allocation metadata.
        .slab = slab_plan,
        // Exact target-domain parameter request metadata.
        .request_table = request_table,
        // Device index selected by the plan placement.
        .device_index = placement->device_index,
        // Device selected by the plan placement.
        .device = device,
        // Queue affinity selected by the plan placement.
        .queue_affinity = placement->queue_affinity,
    };
    iree_hal_buffer_t* buffer = materialization->target_buffer;
    status = id4_pipeline_parameter_slab_set_wrap_resident(
        /*load_count=*/1, &load, &buffer, host_allocator, &replacement_slabs);
  }
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_slab_replacement_t replacement = {
        // Plan-local target domain being replaced.
        .target_slab_index = options->target_slab_index,
        // Semantic domain must match the target plan.
        .expected_domain = slab_plan->domain,
        // One-slab set retaining the queue allocation.
        .source_slab_set = replacement_slabs,
        // The replacement set stores its only slab at index zero.
        .source_slab_index = 0,
    };
    status = id4_pipeline_parameter_slab_set_derive(
        options->base_parameter_slabs, /*replacement_count=*/1, &replacement,
        host_allocator, &materialization->derived_parameter_slabs);
  }
  id4_pipeline_parameter_slab_set_release(replacement_slabs);

  if (iree_status_is_ok(status)) {
    *out_materialization = materialization;
  } else {
    status = iree_status_join(
        status,
        id4_pipeline_parameter_materialization_cleanup_failed(materialization));
    iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                                 iree_status_clone(status));
    id4_pipeline_parameter_materialization_destroy(materialization);
  }
  return status;
}

void id4_pipeline_parameter_materialization_retain(
    id4_pipeline_parameter_materialization_t* materialization) {
  if (!materialization) return;
  iree_atomic_ref_count_inc(&materialization->ref_count);
}

void id4_pipeline_parameter_materialization_release(
    id4_pipeline_parameter_materialization_t* materialization) {
  if (materialization &&
      iree_atomic_ref_count_dec(&materialization->ref_count) == 1) {
    id4_pipeline_parameter_materialization_destroy(materialization);
  }
}

iree_status_t id4_pipeline_parameter_materialization_query_target(
    const id4_pipeline_parameter_materialization_t* materialization,
    id4_pipeline_parameter_materialization_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  memset(out_target, 0, sizeof(*out_target));
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization target is no longer mutable");
  }
  out_target->slab_index = materialization->target_slab_index;
  out_target->target_buffer = materialization->target_buffer;
  out_target->readiness_semaphore_list =
      id4_pipeline_parameter_materialization_edge_list(
          &materialization->acquisition_edge);
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_materialization_abort(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "only an unpublished parameter materialization can be aborted");
  }
  if (iree_atomic_ref_count_load(&materialization->ref_count) != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization cannot abort while another owner retains "
        "it");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      wait_semaphore_list, IREE_SV("materialization abort wait"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter materialization abort")));

  iree_status_t status = id4_pipeline_parameter_materialization_emit(
      materialization, IREE_SV("parameter.materialization.abort"),
      IREE_SV("aborting unpublished parameter-domain contents"),
      diagnostics_sink);
  iree_status_t wait_status = iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  status = iree_status_join(status, wait_status);

  // A reached or failed wait is terminal for all operations represented by the
  // edge. Final buffer release may now reclaim the queue allocation directly.
  materialization->owns_target_allocation = false;
  materialization->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRED;
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->acquisition_edge, materialization->host_allocator);
  return status;
}

iree_status_t id4_pipeline_parameter_materialization_publish(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization must be acquired before publication");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      wait_semaphore_list, IREE_SV("materialization publication wait"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      signal_semaphore_list, IREE_SV("materialization publication signal"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter materialization publication")));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_emit(
      materialization, IREE_SV("parameter.materialization.publish"),
      IREE_SV("publishing parameter-domain contents"), diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_edge_initialize(
      signal_semaphore_list, materialization->host_allocator,
      &materialization->publication_edge));

  iree_status_t status = iree_hal_device_queue_barrier(
      materialization->device, materialization->queue_affinity,
      wait_semaphore_list, signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    materialization->state =
        ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_PUBLISHED;
  } else {
    id4_pipeline_parameter_materialization_edge_deinitialize(
        &materialization->publication_edge, materialization->host_allocator);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

iree_status_t id4_pipeline_parameter_materialization_query_binding(
    const id4_pipeline_parameter_materialization_t* materialization,
    id4_pipeline_parameter_materialization_binding_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  memset(out_binding, 0, sizeof(*out_binding));
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_PUBLISHED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization must be published before binding");
  }
  out_binding->target_slab_index = materialization->target_slab_index;
  out_binding->parameter_slabs = materialization->derived_parameter_slabs;
  out_binding->readiness_semaphore_list =
      id4_pipeline_parameter_materialization_edge_list(
          &materialization->publication_edge);
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_materialization_retire(
    id4_pipeline_parameter_materialization_t* materialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_dealloca_flags_t dealloca_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_ACQUIRED &&
      materialization->state !=
          ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_PUBLISHED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parameter materialization is not active");
  }
  if (iree_atomic_ref_count_load(&materialization->ref_count) != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization cannot retire while bundles retain it");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      wait_semaphore_list, IREE_SV("materialization retirement wait"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_validate_list(
      signal_semaphore_list, IREE_SV("materialization retirement signal"),
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_LIST_FLAG_REQUIRE_NONEMPTY));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("parameter materialization retirement")));
  if (!iree_hal_buffer_allocation_is_terminal(materialization->target_buffer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization allocation has another preserve owner");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_materialization_edge_initialize(
      signal_semaphore_list, materialization->host_allocator,
      &materialization->retirement_edge));
  iree_status_t status = id4_pipeline_parameter_materialization_emit(
      materialization, IREE_SV("parameter.materialization.retire"),
      IREE_SV("retiring parameter-domain storage"), diagnostics_sink);

  bool allocation_discarded = false;
  if (iree_status_is_ok(status)) {
    allocation_discarded = true;
    if (!iree_hal_buffer_allocation_discard(materialization->target_buffer)) {
      materialization->owns_target_allocation = false;
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "parameter materialization lost terminal allocation ownership");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        materialization->device, materialization->queue_affinity,
        wait_semaphore_list, signal_semaphore_list,
        materialization->target_buffer, dealloca_flags);
  }
  if (iree_status_is_ok(status)) {
    materialization->owns_target_allocation = false;
    materialization->state =
        ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRING;
  } else {
    if (allocation_discarded && materialization->owns_target_allocation) {
      iree_hal_buffer_allocation_preserve(materialization->target_buffer);
    }
    id4_pipeline_parameter_materialization_edge_deinitialize(
        &materialization->retirement_edge, materialization->host_allocator);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

iree_status_t id4_pipeline_parameter_materialization_complete_retirement(
    id4_pipeline_parameter_materialization_t* materialization) {
  if (!materialization) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter materialization is required");
  }
  if (materialization->state !=
      ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization has no in-flight retirement");
  }
  bool is_reached = false;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_materialization_edge_query_reached(
          &materialization->retirement_edge, &is_reached));
  if (!is_reached) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter materialization retirement edge has not been reached");
  }
  materialization->state = ID4_PIPELINE_PARAMETER_MATERIALIZATION_STATE_RETIRED;
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->retirement_edge, materialization->host_allocator);
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->publication_edge, materialization->host_allocator);
  id4_pipeline_parameter_materialization_edge_deinitialize(
      &materialization->acquisition_edge, materialization->host_allocator);
  return iree_ok_status();
}
