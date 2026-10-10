// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_FINITE_REGISTRATION_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_FINITE_REGISTRATION_H_

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/serving/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_finite_registration_options_t {
  // Borrowed shared device and host, surviving every registration operation.
  loom_serve_device_t* device;
  // Nonnegative periodic observation interval; zero disables periodic samples.
  iree_duration_t heartbeat_interval;
} loom_serve_finite_registration_options_t;

// The options remain borrowed until the registry is successfully destroyed.
// Both finite-vector protocols share this exact preparation/retirement path.
loom_serve_registration_factory_t loom_serve_finite_registration_factory(
    const loom_serve_finite_registration_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_FINITE_REGISTRATION_H_
