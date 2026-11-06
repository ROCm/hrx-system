// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_NATIVE_QUEUE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_NATIVE_QUEUE_H_

#include "iree/hal/driver/amd/gpu/completion.h"
#include "iree/hal/driver/amd/gpu/publication.h"
#include "iree/hal/driver/amd/gpu/queue/publisher.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_sdma_native_queue_t
    iree_hal_amd_gpu_sdma_native_queue_t;

// A native failure reports logical completion independently of storage safety.
// Unretired operands remain owned by the caller even after the error is
// reported.
typedef enum iree_hal_amd_gpu_sdma_retirement_e {
  IREE_HAL_AMD_GPU_SDMA_UNRETIRED = 0,
  IREE_HAL_AMD_GPU_SDMA_RETIRED,
} iree_hal_amd_gpu_sdma_retirement_t;

// Fixed per-queue retirement window, independent of public operation sizes.
// Native KERNEL admission may supply a smaller limit within this window.
#define IREE_HAL_AMD_GPU_SDMA_RETIREMENT_CAPACITY 64u

// Copied into one native retirement slot before acceptance. This ownership is
// independent of the submission record, which may prepare a subsequent span.
typedef struct iree_hal_amd_gpu_sdma_completion_t {
  // Optional readiness after native completion. False retains this slot while
  // dependency payloads retire or a worker-owned download copy completes. The
  // hook may offer one bounded copy through out_copy; NULL permits observation
  // only. The observer supplies no active-wait hint while this hook is blocked.
  bool (*ready)(void* user_data, iree_hal_amd_gpu_completion_copy_t* out_copy);
  // Reports logical failure independently of native storage safety. One
  // UNRETIRED failure may precede the terminal RETIRED notification. A rejected
  // span receives only RETIRED. No native call or unbounded payload work runs
  // here; callbacks hand continuations to the existing publication owner.
  void (*notify)(void* user_data, amdf_status_t status,
                 iree_hal_amd_gpu_sdma_retirement_t retirement);
  // Borrowed per-span owner, live through its final notification.
  void* user_data;
} iree_hal_amd_gpu_sdma_completion_t;

// Cold protocol admission from captured native family facts.
amdf_queue_publication_modes_t
iree_hal_amd_gpu_sdma_native_queue_publication_modes(
    const amdf_queue_family_info_t* family);

// One ready native span with resolved addresses, dependencies, and cache work.
// The caller owns the record and immutable commands until acceptance; the
// copied completion retains referenced operands through retirement. This bound
// applies to native publication storage, not a public transfer or recording:
// larger executions retain their unpublished suffix and submit bounded spans
// as capacity returns.
typedef struct iree_hal_amd_gpu_sdma_submission_t {
  // Intrusive publication ownership, reusable after acceptance detaches it.
  iree_hal_amd_gpu_publication_entry_t entry;
  // Optional preparation under this queue's exclusive publisher. position is
  // the zero-based native retirement slot sequence. True seals words and
  // completion for admission, including capacity retries. False transfers the
  // record to its callback-selected wait/continuation without publishing; the
  // callback may free it. Preparation does not block or issue native calls.
  bool (*prepare)(void* user_data, uint64_t position);
  // Borrowed complete native packets, or NULL for an empty boundary span.
  const uint32_t* words;
  // Initialized command extent, at most MAX_COMMAND_WORDS.
  uint32_t word_count;
  // Optional publication of prepared producer metadata. Runs once after
  // irrevocable native acceptance, before completion can return its operands.
  // USER has completed every fallible release step; KERNEL has accepted its
  // submission. The callback may reuse the record; native retirement owns the
  // copied completion independently. This is not payload visibility or
  // execution completion. The callback performs no native calls, allocation,
  // or user work. A rejected span never invokes it; an accepted span can still
  // fail before retirement.
  void (*accepted)(void* user_data);
  // Copied retirement action for this exact span. Rejection before preparation
  // uses the initial action; preparation may replace it with a per-slot owner.
  iree_hal_amd_gpu_sdma_completion_t completion;
  // Borrowed preparation/acceptance owner, live through acceptance/rejection.
  void* user_data;
} iree_hal_amd_gpu_sdma_submission_t;

// Cold constructor for a device's explicitly selected transfer queue. All
// command and completion storage is provisioned here. The device installs
// the returned publication/completion sources before submitting commands.
// No HAL device reference, worker, or payload staging allocation is acquired.
iree_status_t iree_hal_amd_gpu_sdma_native_queue_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_queue_publication_modes_t publication_mode,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_hal_amd_gpu_publication_t* publication, uint32_t publication_ordinal,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sdma_native_queue_t** out_queue);

// Publishes one admitted command span without allocation. An idle caller
// publishes one entry immediately; native capacity parks the prepared span on
// the existing publisher. Acceptance and completion callbacks may precede
// return; the caller joins its submitting borrow before reusing the record.
// Native retirement and event packets are appended after the caller's command
// span.
void iree_hal_amd_gpu_sdma_native_queue_submit(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_submission_t* submission);

// Enqueues a continuation without consuming it on this caller. Allocation and
// host-wait callbacks use this path so they never execute native publication.
// The existing device publication worker owns the resulting bounded turn.
void iree_hal_amd_gpu_sdma_native_queue_defer(
    iree_hal_amd_gpu_sdma_native_queue_t* queue,
    iree_hal_amd_gpu_sdma_submission_t* submission);

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_sdma_native_queue_publication_source(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_sdma_native_queue_completion_source(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);

// Cold query after the device has joined its command producers.
bool iree_hal_amd_gpu_sdma_native_queue_is_idle(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);

// Joins CPU publication and failure-notification borrowers after admission has
// closed. Unretired native commands retain their operands and are not certified
// safe by this predicate. NULL is idle.
bool iree_hal_amd_gpu_sdma_native_queue_is_host_idle(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);

// Stops native access while retaining command, mapping, and operand ownership.
// The parent excludes completion observers first. False means a CPU publisher
// is still active and leaves out_status unchanged; true supplies the native
// stop result. A failed native stop remains sticky and preserves backing.
bool iree_hal_amd_gpu_sdma_native_queue_try_inactivate(
    iree_hal_amd_gpu_sdma_native_queue_t* queue, amdf_status_t* out_status);

// Returns accepted operands with the original failure after all affected
// native queues have stopped. The parent exclusively owns completion. False
// means a concurrent CPU publisher or failure notifier must join first.
bool iree_hal_amd_gpu_sdma_native_queue_retire_failed(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);

// Joins the final notification tail and releases native storage. Called after
// both sources and every submitting caller have joined. Failure preserves the
// owner and context; it never certifies reclamation after a failed native join.
amdf_status_t iree_hal_amd_gpu_sdma_native_queue_free(
    iree_hal_amd_gpu_sdma_native_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_NATIVE_QUEUE_H_
