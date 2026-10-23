// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_RESOURCES_H_
#define LOOM_SERVE_SERVER_RESOURCES_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed application resources available to management operations.
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

// Registers the router's nonblocking scheduling wake with native capacity work
// and final output retirement, including cold workspace requests.
// An empty observer detaches and joins callback access before router teardown.
void loom_serve_resources_set_observer(const loom_serve_resources_t* resources,
                                       loom_serve_wake_callback_t observer);

// Advances accepted workspace growth/trim, publishes completed native capacity
// work, and retires parameter admission interests before services advance.
// Continue advancing during error draining until accepted work retires.
// A terminal platform failure stops admission and initiates ordinary draining;
// it never destroys shared owners.
iree_status_t loom_serve_resources_advance(
    const loom_serve_resources_t* resources);

// Whether cold workspace, native capacity work, or accepted outputs borrow
// application resources. Router shutdown joins this independently of readiness
// or semaphore failure.
bool loom_serve_resources_is_pending(const loom_serve_resources_t* resources);

// Writes a JSON object with bound listeners, immutable device identities and
// serving-managed memory observations. Called on the serialized serving owner;
// never waits for devices, polls hardware or changes residency. Retained pool
// accounting is null for fixed backing; an elastic limit is null when
// unbounded. With elastic backing, workspace is a subset of retained physical
// usage, not an additional amount. Workspace values are independent atomic
// observations, not a coherent total. Acquisition/release counters accumulate
// across models and retired pool generations for the device owner's lifetime.
// wait_count counts reservations needing a reuse prerequisite, not host waits
// or elapsed time. These values exclude fixed buffers, loader/driver storage
// and hidden padding; they cannot be interpreted as device-wide usage or
// available allocation bytes.
iree_status_t loom_serve_resources_write_json(
    const loom_serve_resources_t* resources, loom_output_stream_t* output);

// Parses exactly {"device": <index>, "target_bytes": <unsigned integer>} and
// resolves the device index against immutable membership. Performs no mutation.
// Outputs are valid only on success and through this resource view's lifetime.
iree_status_t loom_serve_resources_parse_trim(
    const loom_serve_resources_t* resources, iree_string_view_t value,
    iree_host_size_t* out_device_index, uint64_t* out_target_bytes);

// Executes a validated trim request on the serialized serving owner and writes
// {"target_bytes": ..., "device": <resource object>} with current accounting.
// Schedules release of eligible idle workspace and unpinned parameters without
// waiting for live work, evicting mutable state, or changing the configured
// budget. The target may remain unreachable. Output is copied and retains no
// model borrow. A platform failure may leave partially released backing: the
// caller must stop admission and drain accepted work, not retry inference
// against that backing.
iree_status_t loom_serve_resources_trim(const loom_serve_resources_t* resources,
                                        iree_host_size_t device_index,
                                        uint64_t target_bytes,
                                        loom_output_stream_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_RESOURCES_H_
