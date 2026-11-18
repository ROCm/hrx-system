// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_TRANSFER_H_

#include "iree/hal/driver/amd/gpu/completion.h"
#include "iree/hal/driver/amd/gpu/context.h"
#include "iree/hal/driver/amd/gpu/profile.h"
#include "iree/hal/driver/amd/gpu/queue/capture_pool.h"
#include "iree/hal/driver/amd/gpu/queue/file_staging.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"
#include "iree/hal/utils/file_transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native execution borrowed by the shared file pipeline. The application keeps
// the queue's device alive. Packet encoding and publication remain with the
// executor; this component owns I/O progress and captured payload lifetime.
typedef struct iree_hal_amd_gpu_file_executor_t {
  // Borrowed exact queue whose native transfer engine moves buffer bytes.
  iree_hal_queue_t* queue;
  // Borrowed native API context, live through the queue's accepted operations.
  const iree_hal_amd_gpu_context_t* context;
  // Queue-owned reusable CPU metadata storage.
  iree_hal_amd_gpu_capture_pool_t* captures;
  // Marks accepted host work in the queue's existing shutdown accounting.
  void (*begin)(iree_hal_queue_t* queue);
  // Returns that borrow after the final capture access and wakes shutdown.
  void (*end)(iree_hal_queue_t* queue);

  // Family-owned file staging, or NULL without file-capable native backing.
  iree_hal_amd_gpu_file_staging_t* staging;
  // Queue-owned host dependency associations, finalized by their last borrower.
  iree_async_semaphore_wait_context_t* waits;
  // Optional device-owned session, stable while operations are in flight.
  iree_hal_amd_gpu_profile_t* profile;
  // Exact stream index within the device-owned session.
  uint32_t queue_ordinal;
  // Exact native family for physical-source host/device visibility queries.
  uint32_t native_family_ordinal;
  // Native address contract consumed by the queue's transfer implementation.
  iree_hal_memory_site_kind_t transfer_site;
  // Size of one immutable native barrier value copied into each operation.
  // The executor defines its representation; the file pipeline never reads it.
  iree_host_size_t barrier_size;
  // Combines qualified queue-side visibility effects into an owned native
  // barrier value. Host actions execute at their I/O boundary. This consumes
  // trusted global/ranged queue effects and performs no allocation or query.
  void (*add_effects)(iree_hal_queue_t* queue,
                      iree_hal_memory_effects_t effects, void* barrier);
  // Captures a bounded transfer window or, for zero count, a barrier. The
  // caller has already accepted the file operation, resolved dependencies, and
  // prepared immutable cache recipes. Public queue closure does not reject
  // these continuations; native queue failure still does. Success owns captured
  // arguments until callback reports retirement; failure retains no borrowed
  // payload and invokes no callback. An early failure notification does not end
  // borrowed payload lifetime. NULL before/after omits that boundary. Other
  // values point to barrier_size prepared bytes that remain live through this
  // call.
  iree_status_t (*submit)(
      iree_hal_queue_t* queue, iree_host_size_t count,
      const iree_hal_amd_gpu_transfer_operation_t* operations,
      const void* before, const void* after,
      iree_hal_amd_gpu_retirement_callback_t callback);
} iree_hal_amd_gpu_file_executor_t;

// Captures a finite file group. Indirect bytes are acquired only after waits;
// bounded windows share staging, registration membership, and native copies.
// Acceptance retains operands through every accepted access and final cache
// boundary. Synchronous rejection retains nothing and signals nothing.
// Captured buffer uses end before public signaling, permitting immediate
// source-pool destruction once completion is observed. Before/after each point
// to executor->barrier_size immutable prepared bytes, copied before returning;
// their representation and application belong to the queue executor.
iree_status_t iree_hal_amd_gpu_file_transfer_submit_batch(
    const iree_hal_amd_gpu_file_executor_t* executor,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_memory_access_t file_access, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers, const void* before,
    const void* after);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_FILE_TRANSFER_H_
