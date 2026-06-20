// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/stage.h"

#include <stddef.h>
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
};

static iree_status_t id4_pipeline_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size == 0 || actual_size >= expected_size) {
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
  if (!options) return iree_ok_status();
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
  if (!options) return iree_ok_status();
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("plan")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "plan extension structures are not supported");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_prepare_options(
    const id4_pipeline_stage_prepare_options_t* options) {
  if (!options) return iree_ok_status();
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "prepare extension structures are not supported");
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_validate_issue_options(
    const id4_pipeline_stage_issue_options_t* options) {
  if (!options) return iree_ok_status();
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "issue extension structures are not supported");
  }
  return iree_ok_status();
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

iree_status_t id4_pipeline_bundle_create(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_parameter_slab_set_t* parameter_slabs,
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
  *out_bundle = bundle;
  return iree_ok_status();
}

static void id4_pipeline_bundle_destroy(id4_pipeline_bundle_t* bundle) {
  iree_allocator_t host_allocator = bundle->host_allocator;
  id4_pipeline_parameter_slab_set_release(bundle->parameter_slabs);
  id4_pipeline_plan_release(bundle->plan);
  iree_allocator_free(host_allocator, bundle);
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
