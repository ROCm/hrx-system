// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_OPERATION_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_OPERATION_H_

#include "iree/hal/driver/amd/gpu/profile.h"
#include "iree/hal/driver/amd/gpu/queue/allocation.h"
#include "iree/hal/driver/amd/gpu/queue/capture_pool.h"
#include "iree/hal/driver/amd/gpu/queue/host_wait.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/command_buffer.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/native_queue.h"
#include "iree/hal/driver/amd/gpu/queue/semaphore.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_sdma_queue_t iree_hal_amd_gpu_sdma_queue_t;

// One reusable host wait per operation, reserved during public capture. Waiting
// on the next unresolved edge cannot fail a new allocation after acceptance.
typedef struct iree_hal_amd_gpu_sdma_pending_t {
  // Cancellation membership joins every callback borrower.
  iree_hal_amd_gpu_host_wait_entry_t entry;
  // Capture retaining the original dependency and unpublished native suffix.
  struct iree_hal_amd_gpu_sdma_operation_t* operation;
  // Retained completion or submission-availability view used by the tracker.
  iree_hal_semaphore_t* semaphore;
  // Exact requested value in the borrowed view.
  uint64_t value;
  // Original public dependency, borrowed through operation.
  iree_hal_semaphore_t* source;
  // One reusable tracker allocation in the capture's trailing bytes.
  iree_allocator_inline_storage_t tracker;
} iree_hal_amd_gpu_sdma_pending_t;

typedef struct iree_hal_amd_gpu_sdma_wait_t {
  // Retained public timeline, independent of its native producer's lifetime.
  iree_hal_semaphore_t* semaphore;
  // Exact requested timeline value.
  uint64_t value;
  // Eligible native reader, retained through the last native use.
  iree_hal_amd_gpu_signal_t* signal;
} iree_hal_amd_gpu_sdma_wait_t;

typedef enum iree_hal_amd_gpu_sdma_operation_kind_e {
  IREE_HAL_AMD_GPU_SDMA_OPERATION_BARRIER,
  IREE_HAL_AMD_GPU_SDMA_OPERATION_TRANSFER,
  IREE_HAL_AMD_GPU_SDMA_OPERATION_EXECUTE,
  IREE_HAL_AMD_GPU_SDMA_OPERATION_TIMESTAMP,
  IREE_HAL_AMD_GPU_SDMA_OPERATION_ALLOCATION,
} iree_hal_amd_gpu_sdma_operation_kind_t;

// Native phases stream through bounded spans. Acceptance commits the candidate
// cursor; a capacity retry never repeats capture, upload, or allocation work.
typedef enum iree_hal_amd_gpu_sdma_phase_e {
  IREE_HAL_AMD_GPU_SDMA_PHASE_WAITS,
  IREE_HAL_AMD_GPU_SDMA_PHASE_BEFORE,
  IREE_HAL_AMD_GPU_SDMA_PHASE_INITIALIZE,
  IREE_HAL_AMD_GPU_SDMA_PHASE_BODY,
  IREE_HAL_AMD_GPU_SDMA_PHASE_AFTER,
  IREE_HAL_AMD_GPU_SDMA_PHASE_SIGNAL,
  IREE_HAL_AMD_GPU_SDMA_PHASE_END,
} iree_hal_amd_gpu_sdma_phase_t;

typedef struct iree_hal_amd_gpu_sdma_cursor_t {
  // Next native phase of the logical operation.
  iree_hal_amd_gpu_sdma_phase_t phase;
  // First input whose native wait has not yet been emitted.
  iree_host_size_t wait;
  // Bytes of cold private-data initialization already emitted.
  uint64_t initialize_offset;
  // Committed recording position; unused for direct transfers.
  iree_hal_amd_gpu_sdma_program_cursor_t program;
  // Committed direct-transfer position; unused for recordings.
  iree_hal_amd_gpu_transfer_cursor_t transfer;
} iree_hal_amd_gpu_sdma_cursor_t;

// One public transaction. Publication owns its unpublished suffix; each
// accepted span independently retains the capture. Shared services own payload
// storage, while this object owns only descriptors, bindings, and completion
// lifetimes.
typedef struct iree_hal_amd_gpu_sdma_operation_t {
  // Publication and each native span retain independent references.
  iree_atomic_ref_count_t ref_count;
  // Reusable native submission entry, detached at each acceptance.
  iree_hal_amd_gpu_sdma_submission_t submission;
  // Exact execution owner, borrowed through operation and callback accounting.
  iree_hal_amd_gpu_sdma_queue_t* queue;
  // Allocation extent returned to the queue's capture pool at final release.
  iree_host_size_t allocation_size;
  // Active captured payload representation.
  iree_hal_amd_gpu_sdma_operation_kind_t kind;
  // Captured dependencies; admission may consume a failed prefix in place.
  iree_hal_amd_gpu_sdma_wait_t* waits;
  // Number of retained input edges.
  iree_host_size_t wait_count;
  // Inline host-wait storage, present only when input dependencies exist.
  iree_hal_amd_gpu_sdma_pending_t* pending;
  // Retained public outputs and copied values.
  iree_hal_semaphore_list_t signals;
  // Native producer for eligible outputs, or NULL for host-only completion.
  iree_hal_amd_gpu_signal_t* signal;
  // Queue-local list link used only while waiting for native backing growth.
  struct iree_hal_amd_gpu_sdma_operation_t* next;
  // Submission transferred the capture to queue completion accounting.
  bool submitted;
  // Input admission is complete and native readers are pinned.
  bool admitted;
  // Payload addresses, backing, and any private placement are ready.
  bool prepared;
  // At least one native span was accepted, independent of logical completion.
  bool accepted;
  // The terminal span published the program's causal reuse coordinate.
  bool program_published;
  // Initial native visibility and execution requirements.
  iree_hal_amd_gpu_sdma_barrier_t before;
  // Terminal visibility and execution requirements.
  iree_hal_amd_gpu_sdma_barrier_t after;
  // First failure; retained through independent callback and native owners.
  iree_atomic_int32_t failure_code;
  // First failure already reported to an external retirement observer.
  iree_atomic_int32_t notified_code;
  // Last safely retired span's canonical coordinate, or empty if unissued.
  iree_async_single_frontier_t frontier;
  // Optional internal file/host-call ownership callback.
  iree_hal_amd_gpu_retirement_callback_t callback;
  // Optional profile metadata, present directly after this header.
  bool profiled;
  // Publisher-owned accepted position, never read by a retirement callback.
  iree_hal_amd_gpu_sdma_cursor_t cursor;
  // Fixed native slot prepared for this entry's next acceptance.
  uint32_t slot;
  union {
    // Shared direct transfer descriptor and retained payload references.
    iree_hal_amd_gpu_transfer_t transfer;
    // Recorder, late bindings, and independently leased private data.
    struct {
      // Captured recording invocation, including its retained command buffer.
      iree_hal_amd_gpu_sdma_command_invocation_t invocation;
      // Borrowed sealed data owner, or NULL for data-free recordings.
      iree_hal_amd_gpu_program_t* storage;
      // Reserved/private placement retained through every native piece.
      iree_hal_amd_gpu_program_instance_t* instance;
      // Initialization source selected once under dependency admission.
      uint64_t source_address;
      // Initial bytes to copy; zero for an already-initialized placement.
      uint32_t initialize_length;
    } execute;
    // Qualified timestamp destination, retained through completion.
    iree_hal_amd_gpu_buffer_reference_t timestamp;
    // Shared pool transaction owner, including native acquisition lifetime.
    iree_hal_amd_gpu_allocation_t* allocation;
  };
} iree_hal_amd_gpu_sdma_operation_t;

// Captures already-validated public arrays and reserves inline payload storage.
// Native addresses remain unresolved until dependencies admit execution. A
// successful capture owns no queue accounting until its submit call.
iree_status_t iree_hal_amd_gpu_sdma_operation_create(
    iree_hal_amd_gpu_sdma_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t payload_size,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_amd_gpu_sdma_operation_t** out_operation, void** out_payload);

// Independent references cover publication and each accepted native span. The
// final submitted release completes outputs after payload retirement; a capture
// rejected before submission simply relinquishes its retained resources.
void iree_hal_amd_gpu_sdma_operation_retain(
    iree_hal_amd_gpu_sdma_operation_t* operation);
void iree_hal_amd_gpu_sdma_operation_release(
    iree_hal_amd_gpu_sdma_operation_t* operation);
void iree_hal_amd_gpu_sdma_operation_release_payload(
    iree_hal_amd_gpu_sdma_operation_t* operation);

static inline iree_hal_amd_gpu_profile_operation_t*
iree_hal_amd_gpu_sdma_operation_profile(
    iree_hal_amd_gpu_sdma_operation_t* operation) {
  return operation->profiled
             ? (iree_hal_amd_gpu_profile_operation_t*)(operation + 1)
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_OPERATION_H_
