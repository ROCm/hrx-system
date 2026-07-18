// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_IREE_SERVE_DEVICE_TRANSPORT_H_
#define IREE_TOOLS_IREE_SERVE_DEVICE_TRANSPORT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;

typedef struct iree_serve_device_bind_t {
  // Transport scheme without the URI delimiter.
  iree_string_view_t transport_name;
  // Transport-specific listener address following the URI delimiter.
  iree_string_view_t bind_address;
} iree_serve_device_bind_t;

// Visibility of an actual listener address returned by the transport.
typedef enum iree_serve_device_bind_visibility_e {
  // The listener only accepts connections from the same host.
  IREE_SERVE_DEVICE_BIND_VISIBILITY_LOCAL_ONLY = 0,
  // The listener names one potentially remote-reachable interface.
  IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK = 1,
  // The listener accepts connections on all matching network interfaces.
  IREE_SERVE_DEVICE_BIND_VISIBILITY_NETWORK_WILDCARD = 2,
} iree_serve_device_bind_visibility_t;

// Parses a --bind URI into a transport name and listener address.
iree_status_t iree_serve_device_parse_bind_uri(
    iree_string_view_t bind_uri, iree_serve_device_bind_t* out_bind);

// Classifies the actual listener address returned after the server starts.
// Unknown transports and addresses are conservatively considered network
// visible. TCP loopback addresses and shared-memory endpoints are local-only.
iree_serve_device_bind_visibility_t iree_serve_device_classify_bind_visibility(
    iree_string_view_t transport_name, iree_string_view_t bound_address);

// Creates a transport factory matching |transport_name|.
iree_status_t iree_serve_device_create_transport_factory(
    iree_string_view_t transport_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_IREE_SERVE_DEVICE_TRANSPORT_H_
