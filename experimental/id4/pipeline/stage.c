// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/stage.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct id4_pipeline_bundle_t {
  // Reference count for shared bundle ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for bundle storage.
  iree_allocator_t host_allocator;
  // Retained plan used to prepare this bundle.
  id4_pipeline_plan_t* plan;
  // Optional loaded parameter slabs retained by this bundle.
  id4_pipeline_parameter_slab_set_t* parameter_slabs;
  // Number of retained readiness semaphores.
  iree_host_size_t readiness_count;
  // Retained semaphores that must reach readiness values before use.
  iree_hal_semaphore_t** readiness_semaphores;
  // Payload values paired with retained readiness semaphores.
  uint64_t* readiness_payload_values;
};

static iree_status_t id4_pipeline_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_validate_load_options(
    const id4_pipeline_stage_load_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "load options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("load")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "load extension structures are not supported");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_plan_options(
    const id4_pipeline_stage_plan_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("plan")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "plan extension structures are not supported");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_prepare_options(
    const id4_pipeline_stage_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "prepare extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("prepare wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("prepare signal")));
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_options(
    const id4_pipeline_stage_issue_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "issue extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("issue wait")));
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("issue signal")));
  return iree_ok_status();
}

static void id4_pipeline_bundle_release_readiness(
    id4_pipeline_bundle_t* bundle) {
  for (iree_host_size_t i = 0; i < bundle->readiness_count; ++i) {
    iree_hal_semaphore_release(bundle->readiness_semaphores[i]);
  }
  iree_allocator_free(bundle->host_allocator, bundle->readiness_semaphores);
  iree_allocator_free(bundle->host_allocator, bundle->readiness_payload_values);
  bundle->readiness_count = 0;
  bundle->readiness_semaphores = NULL;
  bundle->readiness_payload_values = NULL;
}

static iree_status_t id4_pipeline_bundle_copy_readiness(
    id4_pipeline_bundle_t* bundle,
    iree_hal_semaphore_list_t readiness_semaphore_list) {
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      readiness_semaphore_list, IREE_SV("bundle readiness")));
  if (readiness_semaphore_list.count == 0) return iree_ok_status();

  iree_hal_semaphore_t** semaphores = NULL;
  uint64_t* payload_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      bundle->host_allocator, readiness_semaphore_list.count,
      sizeof(semaphores[0]), (void**)&semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        bundle->host_allocator, readiness_semaphore_list.count,
        sizeof(payload_values[0]), (void**)&payload_values);
  }
  iree_host_size_t retained_count = 0;
  for (iree_host_size_t i = 0;
       i < readiness_semaphore_list.count && iree_status_is_ok(status); ++i) {
    semaphores[i] = readiness_semaphore_list.semaphores[i];
    iree_hal_semaphore_retain(semaphores[i]);
    payload_values[i] = readiness_semaphore_list.payload_values[i];
    ++retained_count;
  }
  if (iree_status_is_ok(status)) {
    bundle->readiness_count = readiness_semaphore_list.count;
    bundle->readiness_semaphores = semaphores;
    bundle->readiness_payload_values = payload_values;
  } else {
    for (iree_host_size_t i = 0; i < retained_count; ++i) {
      iree_hal_semaphore_release(semaphores[i]);
    }
    iree_allocator_free(bundle->host_allocator, semaphores);
    iree_allocator_free(bundle->host_allocator, payload_values);
  }
  return status;
}

iree_status_t id4_pipeline_stage_initialize(
    const id4_pipeline_stage_vtable_t* vtable,
    const id4_pipeline_stage_services_t* services,
    id4_pipeline_stage_t* out_stage) {
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(services);
  IREE_ASSERT_ARGUMENT(out_stage);
  if (!vtable->destroy || !vtable->load || !vtable->plan || !vtable->prepare ||
      !vtable->issue) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage vtable is incomplete");
  }
  if (!services->device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage device group is required");
  }
  const iree_host_size_t device_count =
      iree_hal_device_group_device_count(services->device_group);
  if (device_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage device group must not be empty");
  }

  memset(out_stage, 0, sizeof(*out_stage));
  iree_atomic_ref_count_init(&out_stage->ref_count);
  out_stage->vtable = vtable;
  out_stage->services = *services;
  iree_hal_device_group_retain(out_stage->services.device_group);
  if (out_stage->services.executable_cache) {
    iree_hal_executable_cache_retain(out_stage->services.executable_cache);
  }
  return iree_ok_status();
}

void id4_pipeline_stage_deinitialize(id4_pipeline_stage_t* stage) {
  if (!stage) return;
  iree_hal_executable_cache_release(stage->services.executable_cache);
  iree_hal_device_group_release(stage->services.device_group);
  memset(&stage->services, 0, sizeof(stage->services));
  stage->vtable = NULL;
}

void id4_pipeline_stage_retain(id4_pipeline_stage_t* stage) {
  if (!stage) return;
  iree_atomic_ref_count_inc(&stage->ref_count);
}

void id4_pipeline_stage_release(id4_pipeline_stage_t* stage) {
  if (stage && iree_atomic_ref_count_dec(&stage->ref_count) == 1) {
    stage->vtable->destroy(stage);
  }
}

const id4_pipeline_stage_services_t* id4_pipeline_stage_services(
    const id4_pipeline_stage_t* stage) {
  return stage ? &stage->services : NULL;
}

iree_status_t id4_pipeline_stage_load(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_load_options_t* options) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_load_options(options));
  return stage->vtable->load(stage, options);
}

iree_status_t id4_pipeline_stage_plan(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_plan_options(options));
  return stage->vtable->plan(stage, options, out_plan);
}

iree_status_t id4_pipeline_stage_prepare(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_prepare_options(options));
  return stage->vtable->prepare(stage, plan, options, out_bundle);
}

iree_status_t id4_pipeline_stage_issue(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options) {
  IREE_ASSERT_ARGUMENT(stage);
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_issue_options(options));
  return stage->vtable->issue(stage, bundle, options);
}

static void id4_pipeline_bundle_destroy(id4_pipeline_bundle_t* bundle) {
  iree_allocator_t host_allocator = bundle->host_allocator;
  id4_pipeline_bundle_release_readiness(bundle);
  id4_pipeline_parameter_slab_set_release(bundle->parameter_slabs);
  id4_pipeline_plan_release(bundle->plan);
  iree_allocator_free(host_allocator, bundle);
}

iree_status_t id4_pipeline_bundle_create(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_slab_set_t* parameter_slabs,
    iree_hal_semaphore_list_t readiness_semaphore_list,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;

  id4_pipeline_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*bundle), (void**)&bundle));
  memset(bundle, 0, sizeof(*bundle));
  iree_atomic_ref_count_init(&bundle->ref_count);
  bundle->host_allocator = host_allocator;
  bundle->plan = (id4_pipeline_plan_t*)plan;
  id4_pipeline_plan_retain(bundle->plan);
  bundle->parameter_slabs = parameter_slabs;
  id4_pipeline_parameter_slab_set_retain(bundle->parameter_slabs);
  iree_status_t status =
      id4_pipeline_bundle_copy_readiness(bundle, readiness_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    id4_pipeline_bundle_destroy(bundle);
  }
  return status;
}

void id4_pipeline_bundle_retain(id4_pipeline_bundle_t* bundle) {
  if (!bundle) return;
  iree_atomic_ref_count_inc(&bundle->ref_count);
}

void id4_pipeline_bundle_release(id4_pipeline_bundle_t* bundle) {
  if (bundle && iree_atomic_ref_count_dec(&bundle->ref_count) == 1) {
    id4_pipeline_bundle_destroy(bundle);
  }
}

const id4_pipeline_plan_t* id4_pipeline_bundle_plan(
    const id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->plan : NULL;
}

id4_pipeline_parameter_slab_set_t* id4_pipeline_bundle_parameter_slabs(
    const id4_pipeline_bundle_t* bundle) {
  return bundle ? bundle->parameter_slabs : NULL;
}

iree_hal_semaphore_list_t id4_pipeline_bundle_readiness_semaphore_list(
    const id4_pipeline_bundle_t* bundle) {
  if (!bundle || bundle->readiness_count == 0) {
    return iree_hal_semaphore_list_empty();
  }
  return (iree_hal_semaphore_list_t){
      // Number of readiness semaphores retained by the bundle.
      .count = bundle->readiness_count,
      // Readiness semaphores retained by the bundle.
      .semaphores = bundle->readiness_semaphores,
      // Payload values paired with the readiness semaphores.
      .payload_values = bundle->readiness_payload_values,
  };
}
