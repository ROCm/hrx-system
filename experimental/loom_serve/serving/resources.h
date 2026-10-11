// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_RESOURCES_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_RESOURCES_H_

#include "experimental/loom_serve/http/listener.h"
#include "experimental/loom_serve/runtime/device.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed application resources published through management observation.
// The view and every referenced owner remain live through the router's join.
// Membership is immutable during routing; models can independently come and go.
typedef struct loom_serve_resources_t {
  // Application's single cooperating group, or NULL for externally owned work.
  loom_serve_device_group_t* device_group;
  // Number of bound listeners published by this application.
  iree_host_size_t listener_count;
  // Borrowed array; listener storage survives stop until routing has joined.
  loom_serve_http_listener_t* const* listeners;
} loom_serve_resources_t;

// Writes a JSON object with bound listeners, immutable device identities and
// serving-managed memory observations. Called on the serialized serving owner;
// never waits for devices, polls hardware or changes residency. Retained pool
// accounting is null for fixed backing; an elastic limit is null when
// unbounded. Workspace values are independent atomic observations, not a
// coherent total. These values exclude fixed buffers, loader/driver storage and
// hidden padding; they cannot be interpreted as device-wide usage or available
// allocation bytes.
iree_status_t loom_serve_resources_write_json(
    const loom_serve_resources_t* resources, loom_output_stream_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_RESOURCES_H_
