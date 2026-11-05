// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_DEVICE_STATE_H_
#define AMDF_SRC_DEVICE_STATE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/memory_profile.h"

struct amdf_memory_reservation_profile_t;

typedef struct amdf_device_vtable_t {
  // Returns cached placement facts for an already-validated family ordinal.
  // All native qualification completes before device publication.
  amdf_queue_family_placement_info_t (*query_queue_family_placement)(
      const amdf_device_t* device, uint32_t queue_family_ordinal);
  // Reads live-domain reservation limits without native work or allocation.
  amdf_status_t (*query_memory_reservation_profile)(
      amdf_device_t* device, uint32_t address_domain_ordinal,
      struct amdf_memory_reservation_profile_t* out_profile);
  // Copies one immutable memory profile into caller-private result storage.
  amdf_status_t (*query_memory_profile)(
      amdf_device_t* device, uint32_t memory_profile_ordinal,
      amdf_memory_native_profile_t* out_profile);
  // Prepares native state in the already-allocated memory owner. Every return
  // leaves partial state there for common rollback; only success fills facts.
  amdf_status_t (*memory_prepare)(
      amdf_memory_t* memory, const amdf_memory_native_group_t* group,
      const amdf_memory_native_create_info_t* create_info,
      amdf_memory_info_t* out_info);
  // Acquires an independent native backing reference without consuming input.
  // Every return leaves partial state in the selected slot for common rollback;
  // only success publishes native backing facts and complete access facts.
  amdf_status_t (*memory_prepare_import)(
      amdf_memory_t* memory, uint32_t access_ordinal,
      const amdf_memory_native_profile_t* profile,
      const amdf_memory_native_import_info_t* import_info,
      const amdf_external_memory_t* external_memory,
      amdf_memory_info_t* out_info);
  // Releases the exact native state owned by a device implementation.
  amdf_status_t (*destroy_native)(amdf_device_t* device);
} amdf_device_vtable_t;

struct amdf_device_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the device is published.
  const amdf_device_vtable_t* vtable;
  // Endpoint borrowed for the lifetime of this device.
  amdf_endpoint_t* endpoint;
  // Immutable physical execution identity copied from the endpoint.
  amdf_endpoint_id_t endpoint_id;
  // Provider instance defining the scope of provider-local identities.
  amdf_instance_t* provider_instance;
  // Exact engine family implementing this device.
  amdf_engine_kind_t engine_kind;
  // Number of queues, queue mappings, events, and contexts borrowing this
  // device. Memory dependencies are caller preconditions and are not counted.
  amdf_child_tracker_t children;
};

#endif  // AMDF_SRC_DEVICE_STATE_H_
