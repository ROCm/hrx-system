// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_
#define AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_

#include <pthread.h>

#include "libamdf/src/gpu/umd/instance.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the borrowed descriptor after successful instance preparation.
int amdf_gpu_kfd_instance_descriptor(const amdf_gpu_umd_instance_t* instance);

// Returns the connection's cold VM-mutation lock, borrowed through destruction.
// All devices and peer mappings on this connection share this lock. Native
// mapping completion extends past KFD's own process mutex into PASID flushes.
pthread_mutex_t* amdf_gpu_kfd_instance_memory_mutex(
    amdf_gpu_umd_instance_t* instance);

// Acquires or reuses this context's exact render-file VM for one GPU. The
// caller holds the platform instance's native-state lock and this connection's
// memory lock. Any return may retain native progress on instance; only success
// publishes the borrowed descriptor and refines topology's memory facts from
// that same connection. Native memory qualification precedes VM bootstrap;
// failure leaves topology unchanged. KFD has no per-GPU inverse of ACQUIRE_VM,
// so the binding belongs to the instance even when no execution device
// currently borrows it.
amdf_status_t amdf_gpu_kfd_instance_prepare_vm(
    amdf_gpu_umd_instance_t* instance, amdf_platform_endpoint_t* endpoint,
    amdf_gpu_kfd_topology_t* topology, size_t page_size,
    int* out_render_descriptor);

// One caller-owned signal ID and its borrowed context-wide native resources.
typedef struct amdf_gpu_kfd_event_info_t {
  // Mailbox address in the exact requesting GPU's VM.
  uint64_t mailbox_address;
  // Signal event ID transferred to the caller on success.
  uint32_t signal_id;
  // Native mailbox payload returned by CREATE_EVENT.
  uint32_t trigger_data;
  // Context-owned manual-reset memory-failure event, borrowed until teardown.
  uint32_t memory_exception_id;
  // Context-owned manual-reset hardware-failure event, borrowed until teardown.
  uint32_t hardware_exception_id;
} amdf_gpu_kfd_event_info_t;

// Prepares shared signal-page access and allocates one native signal event.
// The caller holds the connection memory lock and has an existing VM binding.
// Partial bootstrap remains on instance; failure preserves out_info. Success
// transfers only the signal ID, not the page or shared terminal event sources.
amdf_status_t amdf_gpu_kfd_instance_create_event(
    amdf_gpu_umd_instance_t* instance, const amdf_gpu_kfd_topology_t* topology,
    size_t page_size, amdf_gpu_kfd_event_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_KFD_INSTANCE_H_
