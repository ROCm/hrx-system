// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/execution_backend.h"

void loom_run_execution_backend_registry_initialize_from_entries(
    const loom_run_execution_backend_t* const* backends,
    iree_host_size_t backend_count,
    loom_run_execution_backend_registry_t* out_registry) {
  *out_registry = (loom_run_execution_backend_registry_t){
      .backends = backends,
      .backend_count = backend_count,
  };
}

const loom_run_execution_backend_t* loom_run_execution_backend_registry_lookup(
    const loom_run_execution_backend_registry_t* registry,
    iree_string_view_t name) {
  if (registry == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < registry->backend_count; ++i) {
    const loom_run_execution_backend_t* backend = registry->backends[i];
    if (backend && iree_string_view_equal(backend->name, name)) {
      return backend;
    }
  }
  return NULL;
}

iree_status_t loom_run_execution_backend_registry_format_names(
    const loom_run_execution_backend_registry_t* registry,
    iree_string_builder_t* output) {
  iree_host_size_t appended_count = 0;
  for (iree_host_size_t i = 0; i < registry->backend_count; ++i) {
    const loom_run_execution_backend_t* backend = registry->backends[i];
    if (backend == NULL) {
      continue;
    }
    if (appended_count > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(output, backend->name));
    ++appended_count;
  }
  return iree_ok_status();
}
