// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/binding.h"

iree_status_t loom_serve_realtime_binding_resolve(
    loom_serve_registry_t* registry, iree_string_view_t name,
    loom_serve_model_kind_t kind, loom_serve_realtime_binding_t* out_binding) {
  for (iree_host_size_t i = 0; i < loom_serve_registry_capacity(registry);
       ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(registry, i);
    if (!iree_string_view_equal(model->service.name, name)) {
      continue;
    }
    if (model->service.kind != kind) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "realtime deployment '%.*s' must have kind %s",
                              (int)name.size, name.data,
                              loom_serve_model_kind_name(kind));
    }
    *out_binding = (loom_serve_realtime_binding_t){i, model->incarnation, name};
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "realtime deployment '%.*s' is not registered",
                          (int)name.size, name.data);
}

bool loom_serve_realtime_binding_present(
    loom_serve_registry_t* registry,
    const loom_serve_realtime_binding_t* binding) {
  const loom_serve_registered_model_t* model =
      loom_serve_registry_at(registry, binding->index);
  return model->incarnation == binding->incarnation &&
         iree_string_view_equal(model->service.name, binding->name);
}
