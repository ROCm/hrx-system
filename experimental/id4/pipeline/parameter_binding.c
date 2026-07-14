// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_binding.h"

#include <string.h>

#include "experimental/id4/pipeline/parameter_materialization.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/base/internal/atomics.h"

struct id4_pipeline_parameter_binding_t {
  // Reference count for shared binding ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for the packed binding allocation.
  iree_allocator_t host_allocator;
  // Exact plan shared by every materialized domain.
  const id4_pipeline_plan_t* plan;
  // Complete resident parameter slab set assembled in plan order.
  id4_pipeline_parameter_slab_set_t* parameter_slabs;
  // Number of materializations retained in |materializations|.
  iree_host_size_t materialization_count;
  // Published domain owners retained by this binding.
  id4_pipeline_parameter_materialization_t** materializations;
  // Number of retained publication timeline points.
  iree_host_size_t readiness_count;
  // Publication semaphores retained by this binding.
  iree_hal_semaphore_t** readiness_semaphores;
  // Payload values paired with |readiness_semaphores|.
  uint64_t* readiness_payload_values;
};

static iree_status_t id4_pipeline_parameter_binding_validate_options(
    const id4_pipeline_parameter_binding_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter binding options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter binding options structure is too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter binding extension structures are not supported");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter binding plan is required");
  }
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(options->plan);
  if (slab_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter binding plan must contain at least one parameter slab");
  }
  if (options->materialization_count != slab_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter binding requires %" PRIhsz
                            " materialized domains but received %" PRIhsz,
                            slab_count, options->materialization_count);
  }
  if (!options->materializations) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter binding materialization array is required");
  }
  return iree_ok_status();
}

static void id4_pipeline_parameter_binding_destroy(
    id4_pipeline_parameter_binding_t* binding) {
  if (!binding) return;
  iree_allocator_t host_allocator = binding->host_allocator;
  for (iree_host_size_t i = 0; i < binding->readiness_count; ++i) {
    iree_hal_semaphore_release(binding->readiness_semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < binding->materialization_count; ++i) {
    id4_pipeline_parameter_materialization_release(
        binding->materializations[i]);
  }
  id4_pipeline_parameter_slab_set_release(binding->parameter_slabs);
  id4_pipeline_plan_release((id4_pipeline_plan_t*)binding->plan);
  iree_allocator_free(host_allocator, binding);
}

iree_status_t id4_pipeline_parameter_binding_create(
    const id4_pipeline_parameter_binding_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_binding_t** out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  *out_binding = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_binding_validate_options(options));

  const iree_host_size_t slab_count = options->materialization_count;
  id4_pipeline_parameter_slab_load_t* loads = NULL;
  iree_hal_buffer_t** buffers = NULL;
  id4_pipeline_parameter_materialization_binding_t* published_domains = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, slab_count, sizeof(loads[0]), (void**)&loads));
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, slab_count, sizeof(buffers[0]), (void**)&buffers);
  if (iree_status_is_ok(status)) {
    memset(buffers, 0, slab_count * sizeof(buffers[0]));
    status = iree_allocator_malloc_array(host_allocator, slab_count,
                                         sizeof(published_domains[0]),
                                         (void**)&published_domains);
  }

  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    status =
        id4_pipeline_plan_parameter_slab_load_at(options->plan, i, &loads[i]);
  }

  iree_host_size_t readiness_count = 0;
  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    id4_pipeline_parameter_materialization_t* materialization =
        options->materializations[i];
    if (!materialization) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "parameter materialization %" PRIhsz " is NULL", i);
      break;
    }
    status = id4_pipeline_parameter_materialization_query_binding(
        materialization, &published_domains[i]);
    if (!iree_status_is_ok(status)) break;
    const id4_pipeline_parameter_materialization_binding_t* published =
        &published_domains[i];
    if (published->plan != options->plan) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameter materialization %" PRIhsz
                                " was acquired from a different plan",
                                i);
      break;
    }
    if (published->slab_index >= slab_count) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter materialization %" PRIhsz
                                " has slab index %" PRIhsz,
                                i, published->slab_index);
      break;
    }
    if (buffers[published->slab_index]) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameter slab %" PRIhsz
                                " was materialized more than once",
                                published->slab_index);
      break;
    }
    status = id4_pipeline_parameter_slab_validate_resident_buffer(
        &loads[published->slab_index], published->buffer);
    if (!iree_status_is_ok(status)) break;
    buffers[published->slab_index] = published->buffer;
    if (!iree_host_size_checked_add(readiness_count,
                                    published->readiness_semaphore_list.count,
                                    &readiness_count)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "parameter binding readiness semaphore count overflows");
    }
  }

  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_slab_set_wrap_resident(
        slab_count, loads, buffers, host_allocator, &parameter_slabs);
  }

  id4_pipeline_parameter_binding_t* binding = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t materializations_offset = 0;
    iree_host_size_t readiness_semaphores_offset = 0;
    iree_host_size_t readiness_payload_values_offset = 0;
    iree_host_size_t total_size = 0;
    status = IREE_STRUCT_LAYOUT(
        sizeof(*binding), &total_size,
        IREE_STRUCT_FIELD(slab_count, id4_pipeline_parameter_materialization_t*,
                          &materializations_offset),
        IREE_STRUCT_FIELD(readiness_count, iree_hal_semaphore_t*,
                          &readiness_semaphores_offset),
        IREE_STRUCT_FIELD(readiness_count, uint64_t,
                          &readiness_payload_values_offset));
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(host_allocator, total_size, (void**)&binding);
    }
    if (iree_status_is_ok(status)) {
      memset(binding, 0, total_size);
      iree_atomic_ref_count_init(&binding->ref_count);
      binding->host_allocator = host_allocator;
      binding->plan = options->plan;
      id4_pipeline_plan_retain((id4_pipeline_plan_t*)binding->plan);
      binding->parameter_slabs = parameter_slabs;
      parameter_slabs = NULL;
      binding->materialization_count = slab_count;
      binding->materializations =
          (id4_pipeline_parameter_materialization_t**)((uint8_t*)binding +
                                                       materializations_offset);
      binding->readiness_count = readiness_count;
      binding->readiness_semaphores =
          (iree_hal_semaphore_t**)((uint8_t*)binding +
                                   readiness_semaphores_offset);
      binding->readiness_payload_values =
          (uint64_t*)((uint8_t*)binding + readiness_payload_values_offset);

      iree_host_size_t readiness_index = 0;
      for (iree_host_size_t i = 0; i < slab_count; ++i) {
        binding->materializations[i] = options->materializations[i];
        id4_pipeline_parameter_materialization_retain(
            binding->materializations[i]);
        const iree_hal_semaphore_list_t readiness =
            published_domains[i].readiness_semaphore_list;
        for (iree_host_size_t j = 0; j < readiness.count; ++j) {
          binding->readiness_semaphores[readiness_index] =
              readiness.semaphores[j];
          binding->readiness_payload_values[readiness_index] =
              readiness.payload_values[j];
          iree_hal_semaphore_retain(
              binding->readiness_semaphores[readiness_index]);
          ++readiness_index;
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_binding = binding;
  } else {
    id4_pipeline_parameter_binding_destroy(binding);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  iree_allocator_free(host_allocator, published_domains);
  iree_allocator_free(host_allocator, buffers);
  iree_allocator_free(host_allocator, loads);
  return status;
}

void id4_pipeline_parameter_binding_retain(
    id4_pipeline_parameter_binding_t* binding) {
  if (!binding) return;
  iree_atomic_ref_count_inc(&binding->ref_count);
}

void id4_pipeline_parameter_binding_release(
    id4_pipeline_parameter_binding_t* binding) {
  if (binding && iree_atomic_ref_count_dec(&binding->ref_count) == 1) {
    id4_pipeline_parameter_binding_destroy(binding);
  }
}

id4_pipeline_parameter_slab_set_t* id4_pipeline_parameter_binding_slabs(
    const id4_pipeline_parameter_binding_t* binding) {
  return binding ? binding->parameter_slabs : NULL;
}

iree_hal_semaphore_list_t
id4_pipeline_parameter_binding_readiness_semaphore_list(
    const id4_pipeline_parameter_binding_t* binding) {
  if (!binding || binding->readiness_count == 0) {
    return iree_hal_semaphore_list_empty();
  }
  return (iree_hal_semaphore_list_t){
      // Number of publication timeline points.
      .count = binding->readiness_count,
      // Publication semaphores retained by the binding.
      .semaphores = binding->readiness_semaphores,
      // Payload values paired with the publication semaphores.
      .payload_values = binding->readiness_payload_values,
  };
}
