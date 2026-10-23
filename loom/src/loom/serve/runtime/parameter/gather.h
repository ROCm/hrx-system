// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_GATHER_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_GATHER_H_

#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Snapshots count device-produced span records after waits, then gathers the
// requested bytes from one checkpoint key. Each 24-byte record is three
// little-endian u64 values: parameter byte offset, target byte offset, length.
// FILE and SPLAT interpretation belongs to the provider. Target ranges must
// not overlap. The caller keeps descriptor bytes stable until successful
// readiness and orders target consumers after the returned fence. The call
// never waits for descriptors or I/O, and never joins unrelated model work.
// A zero count returns the incoming dependency set without descriptor/payload
// access, a snapshot allocation, or queue submission.
//
// Copies key and retains device, provider, descriptors, and target
// independently of the caller. The snapshot is retained by the actual queued
// download as well as its deferred consumer: failed readiness cannot retire an
// outstanding host memory borrow. Accepted reads retain file and target
// references beyond callback completion. Rejection leaves out_ready null, but
// already accepted operations still own their resources until they retire.
// Discarding the returned fence does not cancel work or release those
// resources. On rejection or failed readiness, actual buffer retirement gates
// reuse of their backing storage.
//
// This staged path requires queue download and deferred host-call support.
// Provider admission controls read concurrency independently of staging.
iree_status_t loom_serve_parameter_gather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_string_view_t key, iree_hal_buffer_t* descriptors,
    iree_device_size_t descriptor_offset, iree_host_size_t count,
    iree_hal_buffer_t* target, iree_allocator_t host_allocator,
    iree_hal_fence_t** out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_GATHER_H_
