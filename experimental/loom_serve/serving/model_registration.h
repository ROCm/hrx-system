// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_MODEL_REGISTRATION_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_MODEL_REGISTRATION_H_

#include "experimental/loom_serve/serving/registry.h"
#include "loom/serve/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_model_registration_options_t {
  // Borrowed application group and host, surviving every registration
  // operation.
  loom_serve_device_group_t* device_group;
  // Nonnegative periodic observation interval; zero disables periodic samples.
  iree_duration_t heartbeat_interval;
} loom_serve_model_registration_options_t;

// The options remain borrowed until the registry is successfully destroyed.
// Source-defined text, image and finite-vector adapters share the same task
// pool and registration ownership contract. Each declaration supplies its
// resolved device index within this group; no default placement occurs here.
loom_serve_registration_factory_t loom_serve_model_registration_factory(
    const loom_serve_model_registration_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_MODEL_REGISTRATION_H_
