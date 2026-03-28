// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Stub implementations for APIs declared but not yet wired up.
// These exist to unblock the streaming rebase. Each will be replaced
// with a real implementation as the underlying pipeline connects.

#include "pyre_internal.h"

pyre_status_t pyre_queue_dispatch(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_executable_t executable, uint32_t export_ordinal,
    const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_queue_dispatch not yet implemented");
}

pyre_status_t pyre_stream_dispatch(
    pyre_stream_t stream, pyre_executable_t executable,
    uint32_t export_ordinal, const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_stream_dispatch not yet implemented");
}

pyre_status_t pyre_queue_host_call(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_host_call_fn_t callback, void* user_data) {
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_queue_host_call not yet implemented");
}

pyre_status_t pyre_stream_execution_barrier(pyre_stream_t stream) {
  return pyre_make_status(PYRE_STATUS_UNIMPLEMENTED,
                          "pyre_stream_execution_barrier not yet implemented");
}
