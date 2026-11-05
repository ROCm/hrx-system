// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_
#define LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_

#include "loom/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_pool_configuration_t {
  // Machine-local key and resolved consumer membership.
  loom_runtime_pool_binding_t binding;
  // Storage, admission, and reclamation policy for this owner.
  loom_runtime_pool_options_t options;
} loom_serve_pool_configuration_t;

// Machine-local device and pool declarations, independent of model programs.
// Native drivers are resolved by the application after parsing succeeds.
typedef struct loom_serve_device_configuration_t {
  // Allocator owning decoded strings.
  iree_allocator_t host_allocator;
  // Single allocation holding all decoded strings.
  char* strings;
  // Number of populated device entries.
  iree_host_size_t count;
  // Native creation inputs; driver remains NULL until application resolution.
  loom_runtime_device_options_t options[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  // Number of independent pool owners, unrelated to device count.
  iree_host_size_t pool_count;
  // Owned pool declarations; each membership bit indexes options above.
  loom_serve_pool_configuration_t* pools;
  // Owned idle policies in pool declaration order.
  loom_runtime_pool_idle_policy_t* idle;
} loom_serve_device_configuration_t;

// Parses independent nonempty "devices" and "pools" maps. Device entries
// contain a HAL "uri". Pool entries name a nonempty "devices" array and support
// pool_backing, weight_slab_bytes, state_slab_bytes, memory_bytes,
// workspace_idle_ms, parameter_idle_ms, and placement. Placement is an optional
// object containing exactly one selector: {"device":"key"} names a device's
// allocation domain, or {"host_numa":N} names an OS NUMA node. Device keys
// resolve independently of consumer membership; the sealed group later resolves
// the physical node and qualifies required placement for every backing source.
// Optional root "wait_mode" is "active" (default) or "blocking", applied to
// device-owned completion waits for every device. Requests remain asynchronous.
// Pool defaults are automatic placement, elastic backing,
// 1 GiB weight slabs, 64 MiB state slabs, no byte limit, and disabled expiry.
// Byte counts accept integer JSON numbers or strings parsed by
// iree_string_view_parse_device_size, such as "64gib". Idle durations remain
// integer JSON numbers in milliseconds.
// Limits are per record, not a combined physical-memory-domain budget.
// Rejects duplicate/unknown keys, invalid types and unsupported combinations
// before creating a driver or device. Copies strings and zeros output on error.
iree_status_t loom_serve_device_configuration_initialize(
    iree_string_view_t json, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration);

// Constructs one device named "device" and one pool named "pool", with the
// same defaults as JSON. Copies the URI; does not resolve or open hardware.
iree_status_t loom_serve_device_configuration_initialize_uri(
    iree_string_view_t uri, iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration);

// Overrides the completion policy for every device with "active" or "blocking".
// Rejects other values without changing the configuration. Applies only to
// subsequent device creation; this does not change an existing device group.
iree_status_t loom_serve_device_configuration_set_wait_mode(
    loom_serve_device_configuration_t* configuration,
    iree_string_view_t wait_mode);

void loom_serve_device_configuration_deinitialize(
    loom_serve_device_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_DEVICE_CONFIGURATION_H_
