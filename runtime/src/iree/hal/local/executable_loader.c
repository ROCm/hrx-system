// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_loader.h"

iree_status_t iree_hal_executable_import_provider_try_resolve(
    const iree_hal_executable_import_provider_t import_provider,
    iree_host_size_t count, const char* const* symbol_names, void** out_fn_ptrs,
    void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution) {
  if (!count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(symbol_names);
  IREE_ASSERT_ARGUMENT(out_fn_ptrs);
  IREE_ASSERT_ARGUMENT(out_fn_contexts);
  if (out_resolution) *out_resolution = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  // It's fine for there to be no registered provider if all symbols are
  // optional. This is a special case for NULL import providers.
  if (import_provider.resolve == NULL) {
    bool any_required = false;
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (!iree_hal_executable_import_is_optional(symbol_names[i])) {
        any_required = true;
        break;
      }
    }
    if (any_required) {
      IREE_RETURN_AND_END_ZONE(
          z0, iree_make_status(IREE_STATUS_UNAVAILABLE,
                               "no import provider registered for resolving "
                               "required executable imports"));
    } else {
      // No required imports so a NULL provider is fine.
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  iree_status_t status =
      import_provider.resolve(import_provider.self, count, symbol_names,
                              out_fn_ptrs, out_fn_contexts, out_resolution);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_import_provider_t import_provider,
    iree_hal_executable_loader_t* out_base_loader) {
  iree_atomic_ref_count_init(&out_base_loader->ref_count);
  out_base_loader->vtable = vtable;
  out_base_loader->import_provider = import_provider;
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
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(load_params->executable_data.data &&
                       load_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(out_executable);
  return executable_loader->vtable->load(executable_loader, target, load_params,
                                         worker_capacity, out_executable);
}

iree_status_t iree_hal_executable_loader_select_and_load(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

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
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "local executable loaders %" PRIhsz " and %" PRIhsz
          " both claim target `%.*s:%.*s` artifact bytes",
          selected_loader_ordinal, i, (int)target->family.size,
          target->family.data, (int)target->target_key.size,
          target->target_key.data);
    }
    selected_loader = loader;
    selected_loader_ordinal = i;
  }
  if (IREE_UNLIKELY(!selected_loader)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no local executable loader claims target `%.*s:%.*s` artifact bytes",
        (int)target->family.size, target->family.data,
        (int)target->target_key.size, target->target_key.data);
  }
  return iree_hal_executable_loader_load(selected_loader, target, load_params,
                                         worker_capacity, out_executable);
}
