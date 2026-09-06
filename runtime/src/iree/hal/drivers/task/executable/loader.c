// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/executable/loader.h"

void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_loader_t* out_base_loader) {
  iree_atomic_ref_count_init(&out_base_loader->ref_count);
  out_base_loader->vtable = vtable;
}

void iree_hal_executable_loader_retain(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader)) {
    iree_atomic_ref_count_inc(&executable_loader->ref_count);
  }
}

void iree_hal_executable_loader_release(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader) &&
      iree_atomic_ref_count_dec(&executable_loader->ref_count) == 1) {
    executable_loader->vtable->destroy(executable_loader);
  }
}

bool iree_hal_executable_loader_query_target_support(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_target_t* target) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(target);
  return executable_loader->vtable->query_target_support(executable_loader,
                                                         target);
}

void iree_hal_executable_loader_query_spec(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_device_executable_spec_t* out_executable_spec) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(out_executable_spec);
  executable_loader->vtable->query_spec(executable_loader, out_executable_spec);
}

bool iree_hal_query_any_executable_loader_target_support(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    const iree_hal_executable_target_t* target) {
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(target);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    if (iree_hal_executable_loader_query_target_support(loaders[i], target)) {
      return true;
    }
  }
  return false;
}

bool iree_hal_executable_loader_claims_executable(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  return executable_loader->vtable->claims_executable(executable_loader, target,
                                                      load_params);
}

iree_status_t iree_hal_executable_loader_load(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(load_params->executable_data.data &&
                       load_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(out_executable);
  return executable_loader->vtable->load(executable_loader, queue_family,
                                         target, load_params, worker_capacity,
                                         out_executable);
}

iree_status_t iree_hal_executable_loader_select_and_load(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);

  iree_hal_executable_loader_t* selected_loader = NULL;
  iree_host_size_t selected_loader_ordinal = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_t* loader = loaders[i];
    if (!iree_hal_executable_loader_query_target_support(loader, target) ||
        !iree_hal_executable_loader_claims_executable(loader, target,
                                                      load_params)) {
      continue;
    }
    if (IREE_UNLIKELY(selected_loader)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "executable loaders %" PRIhsz " and %" PRIhsz
                              " both claim target `%.*s:%.*s` artifact bytes",
                              selected_loader_ordinal, i,
                              (int)target->family.size, target->family.data,
                              (int)target->target_key.size,
                              target->target_key.data);
    }
    selected_loader = loader;
    selected_loader_ordinal = i;
  }
  if (IREE_UNLIKELY(!selected_loader)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no executable loader claims target `%.*s:%.*s` artifact bytes",
        (int)target->family.size, target->family.data,
        (int)target->target_key.size, target->target_key.data);
  }
  return iree_hal_executable_loader_load(selected_loader, queue_family, target,
                                         load_params, worker_capacity,
                                         out_executable);
}
