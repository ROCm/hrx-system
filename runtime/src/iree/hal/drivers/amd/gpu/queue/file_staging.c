// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/file_staging.h"

#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/tracing.h"

typedef struct iree_hal_amd_gpu_file_staging_block_t
    iree_hal_amd_gpu_file_staging_block_t;

typedef struct iree_hal_amd_gpu_file_staging_list_t {
  // Oldest range with unassigned bytes.
  iree_hal_amd_gpu_file_staging_request_t* head;
  // Continuing ranges append here for fair slot assignment.
  iree_hal_amd_gpu_file_staging_request_t* tail;
} iree_hal_amd_gpu_file_staging_list_t;

typedef enum iree_hal_amd_gpu_file_staging_phase_e {
  IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY = 0,
  IREE_HAL_AMD_GPU_FILE_STAGING_CONSTRUCTING,
  IREE_HAL_AMD_GPU_FILE_STAGING_READY,
  IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING,
  IREE_HAL_AMD_GPU_FILE_STAGING_RELEASING,
} iree_hal_amd_gpu_file_staging_phase_t;

typedef enum iree_hal_amd_gpu_file_staging_view_phase_e {
  IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_EMPTY = 0,
  IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_PENDING,
  IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_READY,
  IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED,
  IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING,
} iree_hal_amd_gpu_file_staging_view_phase_t;

typedef struct iree_hal_amd_gpu_file_staging_view_t {
  // Exact I/O poll owner and stable cache membership.
  iree_hal_amd_gpu_file_staging_bucket_t* bucket;
  // Physical owner, stable until every view detaches and native free completes.
  iree_hal_amd_gpu_file_staging_block_t* block;
  // Intrusive cold attachment queue; no allocation is needed to publish a view.
  struct iree_hal_amd_gpu_file_staging_view_t* next;
  // Entry owned by block.memory.registrations, or NULL for ordinary host I/O.
  iree_async_buffer_registration_entry_t* registration;
  // Attachment result and eligibility, protected by the staging mutex.
  iree_hal_amd_gpu_file_staging_view_phase_t phase;
  // First attachment failure, consumed by waiting requests or an explicit
  // retry.
  iree_status_t status;
  // Successful native attachment requires a detach before backing release.
  bool attached;
} iree_hal_amd_gpu_file_staging_view_t;

typedef struct iree_hal_amd_gpu_file_staging_slot_t {
  // Owning physical block, stable through accepted I/O and GPU retirement.
  iree_hal_amd_gpu_file_staging_block_t* block;
  // Next available slot; protected by the staging mutex.
  struct iree_hal_amd_gpu_file_staging_slot_t* next;
  // Borrowed range retaining its device through every accepted slot.
  iree_hal_amd_gpu_file_staging_request_t* request;
  // First byte within the block's logical host/GPU extent.
  uint64_t offset;
  union {
    // Placed native file request, reused only after its preceding use retires.
    iree_hal_amd_fabric_file_request_t io;
    struct {
      // Presecured terminal result delivered on the file's actual poll owner.
      iree_async_nop_operation_t publication;
      // Native queue failure converted by the receiving proactor.
      amdf_status_t status;
      // Whether accepted native consumers have relinquished this slot.
      iree_hal_amd_gpu_sdma_retirement_t retirement;
    } failure;
  };
  // Placed GPU copy, reused after native retirement.
  iree_hal_amd_gpu_sdma_copy_t copy;
  // Joins the submitting call and potentially immediate completion.
  iree_atomic_int32_t pending;
  // Status transferred by completion to the final join.
  iree_status_t status;
  // Next step after both borrowers return.
  void (*resume)(struct iree_hal_amd_gpu_file_staging_slot_t*, iree_status_t);
} iree_hal_amd_gpu_file_staging_slot_t;

struct iree_hal_amd_gpu_file_staging_block_t {
  // Device-owned engine containing this stable owner record.
  iree_hal_amd_gpu_file_staging_t* staging;
  // Index in the fixed two-block owner and per-proactor view tables.
  uint32_t ordinal;
  // Native lifetime phase, protected by the staging mutex.
  iree_hal_amd_gpu_file_staging_phase_t phase;
  // Physical credit, reserved before acquisition and returned after native
  // free.
  uint64_t reserved_bytes;
  // Native backing and every detached/attached registration over its host view.
  iree_hal_amd_fabric_memory_t memory;
  // Immutable native request borrowing the device's captured placement profile.
  iree_hal_amd_fabric_memory_request_t allocation;
  // Presecured asynchronous native acquisition descriptor.
  iree_hal_amd_gpu_memory_acquire_t acquire;
  // Presecured asynchronous native release descriptor.
  iree_hal_amd_gpu_memory_release_t release;
  // Cold allocation/free results return to the device's existing I/O proactor.
  iree_async_nop_operation_t publication;
  // Native construction result consumed by publication.
  iree_status_t status;
  // Canonical detached preparation shared by independent proactor views.
  iree_async_buffer_registration_entry_t* preparation;
  // Stable GPU address corresponding to logical host byte zero.
  uint64_t address;
  // Capacity of one lane in bytes, independent of native allocation length.
  uint32_t capacity;
  // Number of slot records and independently pipelined lanes in this block.
  uint32_t lane_count;
  // Slots with accepted I/O/GPU use, including a submitting caller's join.
  uint32_t active_count;
  // Queued or outstanding view attachment/detachment preventing release.
  uint32_t pending_views;
  // Demand-sized array retained through native free and result publication.
  iree_hal_amd_gpu_file_staging_slot_t* slots;
  // Available slots, selected without allocation under the metadata mutex.
  iree_hal_amd_gpu_file_staging_slot_t* available;
  // Joins detach callbacks with the caller enumerating prepared views.
  iree_atomic_int32_t pending_detaches;
  // An unproven detach retains this entire native ownership graph.
  bool detach_failed;
  // Release initiated during ordinary use retains the device through
  // completion.
  bool release_owns_device;
  // Physical tracing began after successful native acquisition.
  bool traced;
  // An idle trim also returns acquisition already accepted by the native owner.
  bool retire_on_publish;
  struct {
    // Acquire before SDMA consumes file-read output.
    iree_hal_memory_effects_t read;
    // Release before file writes consume SDMA output.
    iree_hal_memory_effects_t write;
  } effects;
};

struct iree_hal_amd_gpu_file_staging_bucket_t {
  // Distinct exact proactors; membership changes only on first use.
  iree_hal_amd_gpu_file_staging_bucket_t* next;
  // Borrowed device-owned engine.
  iree_hal_amd_gpu_file_staging_t* staging;
  // Retained file proactor; foreign views detach before the last range
  // completes.
  iree_async_proactor_t* proactor;
  // Metadata-only views of the same two physical blocks.
  iree_hal_amd_gpu_file_staging_view_t views[2];
  // Ranges awaiting a suitable physical block or this proactor's attachment.
  iree_hal_amd_gpu_file_staging_list_t waiting;
  // Final ranges awaiting foreign-view detach before relinquishing polling.
  iree_hal_amd_gpu_file_staging_list_t completed;
  // Accepted whole file operations, including gaps between physical ranges.
  uint32_t transfer_count;
};

struct iree_hal_amd_gpu_file_staging_t {
  // Borrowed parent; active requests and asynchronous preparation retain uses.
  iree_hal_device_t* device;
  // Borrowed native lifetime and typed allocation owner.
  iree_hal_amd_gpu_context_t* context;
  // Borrowed SDMA engine, independent of executable queue protocol.
  iree_hal_amd_gpu_sdma_queue_t* queue;
  // Borrowed device poll owner, live through all preparations and final
  // release.
  iree_async_proactor_t* proactor;
  // Allocator for per-proactor metadata and demand-sized slot records.
  iree_allocator_t host_allocator;
  // Protects metadata only; native calls, allocations, I/O, and callbacks run
  // out.
  iree_slim_mutex_t mutex;
  // Cold teardown joins actual detach/free without a tracing wait zone.
  iree_notification_t notification;
  // Stable per-proactor views; no bucket owns physical payload.
  iree_hal_amd_gpu_file_staging_bucket_t* buckets;
  // At most two acquired, constructing, or retiring native allocations.
  iree_hal_amd_gpu_file_staging_block_t blocks[2];
  // FIFOs for block eligibility masks 01, 10, and 11, respectively.
  iree_hal_amd_gpu_file_staging_list_t requests[3];
  // Cold registration work, explicitly published when a view becomes necessary.
  iree_hal_amd_gpu_file_staging_view_t* pending_views;
  // Total native-byte budget, including required native prefix and granularity.
  uint64_t byte_budget;
  // Reserved physical bytes across construction, use, detachment, and native
  // free.
  uint64_t reserved_bytes;
  // Accepted whole file operations, including gaps between physical ranges.
  uint32_t transfer_count;
  // Physical ranges with queued or accepted work; only these justify growth.
  uint32_t request_count;
  // Maximum pipelined lanes requested by this device's file policy.
  uint32_t lane_count;
  // Maximum bytes per lane, qualified against the native single-allocation
  // limit.
  uint32_t lane_capacity;
  // Largest requested lane capacity since the last quiescent trim.
  uint32_t desired_capacity;
  // Next ready FIFO considered for round-robin assignment.
  uint32_t next_queue;
  // One caller owns progress through synchronous callback reentry.
  bool pumping;
  // Cold geometry/ownership changed; only these events reclassify queued
  // ranges.
  bool dirty;
  // Explicit demand permits growth; allocation failure clears it until new
  // work.
  bool growth_requested;
  // Return backing once requests quiesce; live backing remains reusable.
  bool trim_requested;
  // Cold teardown joins existing release and does not start new work.
  bool closing;
  // Failed native detach prevents destruction of the ownership graph.
  bool cleanup_blocked;
  // Terminal copy-engine failure, protected by the metadata mutex.
  iree_status_t failure;
  // Exact native transfer family used for visibility qualification.
  uint32_t family_ordinal;
  // Immutable native access for all physical staging.
  amdf_memory_device_access_t access;
  // Captured native placement, geometry, and canonical preparation proactor.
  iree_hal_amd_fabric_memory_options_t memory;
  // System physical-memory stream shared with native slab providers.
  iree_hal_memory_trace_t trace;
};

static void iree_hal_amd_gpu_file_staging_progress(
    iree_hal_amd_gpu_file_staging_t* staging);

static void iree_hal_amd_gpu_file_staging_list_append(
    iree_hal_amd_gpu_file_staging_list_t* list,
    iree_hal_amd_gpu_file_staging_request_t* request) {
  request->previous = list->tail;
  request->next = NULL;
  if (list->tail) {
    list->tail->next = request;
  } else {
    list->head = request;
  }
  list->tail = request;
}

static void iree_hal_amd_gpu_file_staging_list_remove(
    iree_hal_amd_gpu_file_staging_list_t* list,
    iree_hal_amd_gpu_file_staging_request_t* request) {
  if (request->previous) {
    request->previous->next = request->next;
  } else {
    list->head = request->next;
  }
  if (request->next) {
    request->next->previous = request->previous;
  } else {
    list->tail = request->previous;
  }
  request->previous = request->next = NULL;
}

static iree_hal_amd_gpu_file_staging_list_t*
iree_hal_amd_gpu_file_staging_request_list(
    iree_hal_amd_gpu_file_staging_request_t* request) {
  return request->block_mask
             ? &request->staging->requests[request->block_mask - 1]
             : &request->bucket->waiting;
}

static bool iree_hal_amd_gpu_file_staging_transfer_released(
    iree_hal_amd_gpu_file_staging_request_t* request) {
  --request->bucket->transfer_count;
  if (--request->staging->transfer_count == 0 &&
      request->staging->trim_requested) {
    request->staging->dirty = true;
  }
  if (!request->bucket->transfer_count &&
      request->bucket->proactor != request->staging->proactor &&
      ((request->bucket->views[0].attached &&
        request->bucket->views[0].phase !=
            IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED) ||
       (request->bucket->views[1].attached &&
        request->bucket->views[1].phase !=
            IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED) ||
       request->bucket->views[0].phase ==
           IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_PENDING ||
       request->bucket->views[1].phase ==
           IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_PENDING ||
       request->bucket->views[0].phase ==
           IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING ||
       request->bucket->views[1].phase ==
           IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING)) {
    iree_hal_amd_gpu_file_staging_list_append(&request->bucket->completed,
                                              request);
    request->staging->dirty = true;
    return false;
  }
  return true;
}

static void iree_hal_amd_gpu_file_staging_request_complete(
    iree_hal_amd_gpu_file_staging_request_t* request) {
  const iree_hal_amd_fabric_file_transfer_callback_t callback =
      request->completion;
  iree_status_t status = request->status;
  request->status = iree_ok_status();
  callback.fn(callback.user_data, status);
}

static void iree_hal_amd_gpu_file_staging_fail_requests(
    iree_hal_amd_gpu_file_staging_list_t* requests, iree_status_t status,
    iree_hal_amd_gpu_file_staging_list_t* completed) {
  while (requests->head) {
    iree_hal_amd_gpu_file_staging_request_t* request = requests->head;
    iree_hal_amd_gpu_file_staging_list_remove(requests, request);
    request->status = iree_status_clone(status);
    request->cursor = request->end;
    if (!request->in_flight) {
      --request->staging->request_count;
      iree_hal_amd_gpu_file_staging_list_append(completed, request);
    }
  }
}

static void iree_hal_amd_gpu_file_staging_complete_list(
    iree_hal_amd_gpu_file_staging_list_t completed) {
  while (completed.head) {
    iree_hal_amd_gpu_file_staging_request_t* request = completed.head;
    iree_hal_amd_gpu_file_staging_list_remove(&completed, request);
    iree_hal_amd_gpu_file_staging_request_complete(request);
  }
}

static void iree_hal_amd_gpu_file_staging_slot_finish(
    iree_hal_amd_gpu_file_staging_slot_t* slot, iree_status_t status) {
  iree_hal_amd_gpu_file_staging_block_t* block = slot->block;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_hal_amd_gpu_file_staging_request_t* request = slot->request;
  iree_slim_mutex_lock(&staging->mutex);
  if (!iree_status_is_ok(status)) {
    if (iree_status_is_ok(request->status)) {
      request->status = status;
      if (request->cursor < request->end) {
        iree_hal_amd_gpu_file_staging_list_remove(
            iree_hal_amd_gpu_file_staging_request_list(request), request);
        request->cursor = request->end;
      }
    } else {
      iree_status_free(status);
    }
  }
  slot->request = NULL;
  slot->next = block->available;
  block->available = slot;
  if (--block->active_count == 0 &&
      block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING) {
    staging->dirty = true;
  }
  const bool complete =
      --request->in_flight == 0 && request->cursor == request->end;
  if (complete) {
    --staging->request_count;
  }
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
  if (complete) {
    iree_hal_amd_gpu_file_staging_request_complete(request);
  }
}

static void iree_hal_amd_gpu_file_staging_slot_join(
    iree_hal_amd_gpu_file_staging_slot_t* slot) {
  if (iree_atomic_fetch_sub(&slot->pending, 1, iree_memory_order_acq_rel) !=
      1) {
    return;
  }
  iree_status_t status = slot->status;
  slot->status = iree_ok_status();
  slot->resume(slot, status);
}

static void iree_hal_amd_gpu_file_staging_step_completed(void* user_data,
                                                         iree_status_t status) {
  iree_hal_amd_gpu_file_staging_slot_t* slot = user_data;
  slot->status = status;
  iree_hal_amd_gpu_file_staging_slot_join(slot);
}

static void iree_hal_amd_gpu_file_staging_copy_failed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_amd_gpu_file_staging_slot_t* slot = user_data;
  iree_hal_amd_gpu_file_staging_block_t* block = slot->block;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  const iree_hal_amd_gpu_sdma_retirement_t retirement =
      slot->failure.retirement;
  status = iree_status_join(
      IREE_HAL_AMD_STATUS_FROM_AMDF(slot->failure.status, "SDMA file copy"),
      status);
  iree_slim_mutex_lock(&staging->mutex);
  const bool first_failure = iree_status_is_ok(staging->failure);
  if (first_failure) {
    staging->failure = iree_status_clone(status);
    staging->dirty = true;
  }
  iree_slim_mutex_unlock(&staging->mutex);
  if (first_failure) {
    iree_hal_amd_gpu_context_report_status(staging->context, staging->device,
                                           status);
  }
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_hal_semaphore_list_fail(*slot->request->signals,
                               iree_status_clone(status));
  if (retirement == IREE_HAL_AMD_GPU_SDMA_RETIRED) {
    iree_hal_amd_gpu_file_staging_step_completed(slot, status);
  } else {
    // A failed observer/queue is not a retirement proof. Keep the callback's
    // slot borrow, request capture, backing, and device alive together.
    iree_status_free(status);
  }
}

static void iree_hal_amd_gpu_file_staging_copy_notified(
    void* user_data, amdf_status_t status,
    iree_hal_amd_gpu_sdma_retirement_t retirement) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amd_gpu_file_staging_slot_t* slot = user_data;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, slot->copy.length);
  if (amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_step_completed(slot, iree_ok_status());
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  slot->failure.status = status;
  slot->failure.retirement = retirement;
  slot->failure.publication.base = (iree_async_operation_t){
      .type = IREE_ASYNC_OPERATION_TYPE_NOP,
      .completion_fn = iree_hal_amd_gpu_file_staging_copy_failed,
      .user_data = slot};
  IREE_CHECK_OK(iree_async_proactor_submit_one(
      slot->request->bucket->proactor, &slot->failure.publication.base));
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amd_gpu_file_staging_issue_io(
    iree_hal_amd_gpu_file_staging_slot_t* slot,
    void (*resume)(iree_hal_amd_gpu_file_staging_slot_t*, iree_status_t)) {
  slot->resume = resume;
  iree_atomic_store(&slot->pending, 2, iree_memory_order_relaxed);
  iree_status_t status = iree_hal_amd_fabric_file_request_submit(
      &slot->io, (iree_hal_amd_fabric_file_transfer_callback_t){
                     iree_hal_amd_gpu_file_staging_step_completed, slot});
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_step_completed(slot, status);
  }
  iree_hal_amd_gpu_file_staging_slot_join(slot);
}

static void iree_hal_amd_gpu_file_staging_downloaded(
    iree_hal_amd_gpu_file_staging_slot_t* slot, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_slot_finish(slot, status);
    return;
  }
  iree_hal_amd_gpu_file_staging_issue_io(
      slot, iree_hal_amd_gpu_file_staging_slot_finish);
}

static void iree_hal_amd_gpu_file_staging_copy(
    iree_hal_amd_gpu_file_staging_slot_t* slot, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_slot_finish(slot, status);
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_hal_amd_gpu_file_staging_request_t* request = slot->request;
  const bool read =
      request->transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ;
  const uint64_t staging_address = slot->block->address + slot->offset;
  const uint64_t device_address = request->device_address + slot->io.offset;
  slot->copy = (iree_hal_amd_gpu_sdma_copy_t){
      .source = read ? staging_address : device_address,
      .target = read ? device_address : staging_address,
      .length = (uint32_t)slot->io.span.length,
      .effects = request->effects,
      .notify = iree_hal_amd_gpu_file_staging_copy_notified,
      .user_data = slot};
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, slot->copy.length);
  slot->copy.effects.bits |=
      read ? slot->block->effects.read.bits : slot->block->effects.write.bits;
  slot->resume = read ? iree_hal_amd_gpu_file_staging_slot_finish
                      : iree_hal_amd_gpu_file_staging_downloaded;
  iree_atomic_store(&slot->pending, 2, iree_memory_order_relaxed);
  iree_hal_amd_gpu_sdma_queue_submit(request->staging->queue, &slot->copy);
  iree_hal_amd_gpu_file_staging_slot_join(slot);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amd_gpu_file_staging_query_effects(
    iree_hal_amd_gpu_file_staging_block_t* block, bool read) {
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  const amdf_memory_site_t file = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(file),
      .kind = (block->memory.host_features & AMDF_HOST_MAPPING_FEATURE_FILE_IO)
                  ? AMDF_MEMORY_SITE_KIND_FILE_IO
                  : AMDF_MEMORY_SITE_KIND_HOST,
      .value.host = {.mapping = block->memory.mapping}};
  const amdf_memory_site_t device = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(device),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device = {.memory = block->memory.handle,
                       .queue_family_ordinal = staging->family_ordinal}};
  amdf_memory_pair_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(native)};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      staging->context->api->memory_query_pair_info(
          read ? &file : &device, read ? &device : &file, &native),
      "memory_query_pair_info(file staging)"));
  iree_hal_memory_pair_info_t pair;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_memory_translate_pair(&native, &pair));
  const iree_hal_memory_transition_recipe_info_t* host =
      read ? &pair.release : &pair.acquire;
  const iree_hal_memory_transition_recipe_info_t* queue =
      read ? &pair.acquire : &pair.release;
  if (host->kind != IREE_HAL_MEMORY_TRANSITION_KIND_NONE ||
      (queue->kind != IREE_HAL_MEMORY_TRANSITION_KIND_NONE &&
       (queue->kind != IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL ||
        queue->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE ||
        !(staging->context->families[staging->family_ordinal].format_features &
          AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR)))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "native file/SDMA staging visibility is unsupported");
  }
  iree_hal_memory_effects_t* effects =
      read ? &block->effects.read : &block->effects.write;
  if (queue->kind != IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    effects->bits = read ? IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM
                         : IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_file_staging_queue_effect(
    const iree_hal_memory_transition_recipe_info_t* recipe, bool release,
    iree_hal_memory_effects_t* effects) {
  if (recipe->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    return iree_ok_status();
  }
  if (recipe->kind != IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL ||
      recipe->executor != IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file copy requires unsupported queue visibility");
  }
  effects->bits |= release ? IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM
                           : IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_file_staging_prepare(
    iree_hal_amd_gpu_file_staging_t* staging,
    const iree_hal_queue_family_t* family, uint32_t native_family_ordinal,
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_hal_memory_effects_t* before, iree_hal_memory_effects_t* after) {
  if (!staging || !transfer->sources.count) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file copy requires native SDMA backing access");
  }
  if ((transfer->async.file->flags & IREE_ASYNC_FILE_FLAG_DIRECT) &&
      !(staging->memory.required_flags & AMDF_MEMORY_FLAG_HOST_FILE_IO)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "DIRECT file copies require file-capable staging");
  }
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(transfer->file);
  const uint64_t alignment =
      transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
          ? requirements->read.buffer_alignment
          : requirements->write.buffer_alignment;
  if (alignment > staging->lane_capacity) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "file alignment exceeds staging lane capacity");
  }
  const bool release =
      iree_hal_buffer_usage_access(
          iree_hal_buffer_family_usage(transfer->buffer, family)) &
      IREE_HAL_MEMORY_ACCESS_WRITE;
  const bool acquire = transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < transfer->sources.count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_fabric_file_range_t* range = &transfer->sources.ranges[i];
    if (range->contents.data) {
      continue;
    }
    const amdf_memory_site_t device = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
        .structure_size = sizeof(device),
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value.device = {.memory = range->storage->handle,
                         .access_ordinal = range->access_ordinal,
                         .queue_family_ordinal = native_family_ordinal}};
    amdf_memory_site_t copy = device;
    copy.value.device.queue_family_ordinal = staging->family_ordinal;
    for (uint32_t edge = 0; edge < 2 && iree_status_is_ok(status); ++edge) {
      if (edge == 0 ? !release : !acquire) {
        continue;
      }
      amdf_memory_pair_info_t native = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(native)};
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          staging->context->api->memory_query_pair_info(
              edge ? &copy : &device, edge ? &device : &copy, &native),
          "memory_query_pair_info(file copy)");
      iree_hal_memory_pair_info_t pair;
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_memory_translate_pair(&native, &pair);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_file_staging_queue_effect(
            &pair.release, true, edge ? &range->effects : before);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_file_staging_queue_effect(
            &pair.acquire, false, edge ? after : &range->effects);
      }
    }
    if (iree_status_is_ok(status) && range->effects.bits &&
        !(staging->context->families[staging->family_ordinal].format_features &
          AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR)) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "SDMA family lacks required cache operations");
    }
  }
  return status;
}

// Native extents include prefix/granularity, not merely the usable lane bytes.
static uint64_t iree_hal_amd_gpu_file_staging_native_length(
    const iree_hal_amd_gpu_file_staging_t* staging, uint64_t logical_length) {
  const amdf_memory_construction_capabilities_t* allocation =
      &staging->memory.profile.allocation;
  logical_length = (logical_length + allocation->byte_length_granularity - 1) /
                   allocation->byte_length_granularity *
                   allocation->byte_length_granularity;
  return (logical_length + allocation->native_byte_length_prefix +
          allocation->native_byte_length_granularity - 1) /
         allocation->native_byte_length_granularity *
         allocation->native_byte_length_granularity;
}

static uint32_t iree_hal_amd_gpu_file_staging_eligible(
    const iree_hal_amd_gpu_file_staging_request_t* request) {
  uint32_t ready = 0, draining = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    const iree_hal_amd_gpu_file_staging_block_t* block =
        &request->staging->blocks[i];
    if (request->bucket->views[i].phase !=
            IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_READY ||
        block->capacity < request->chunk_capacity) {
      continue;
    }
    if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_READY) {
      ready |= 1u << i;
    } else if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING) {
      draining |= 1u << i;
    }
  }
  return ready ? ready : draining;
}

// Only cold view/geometry changes reclassify queues. Hot slot assignment uses
// the retained mask and three FIFO heads directly.
static void iree_hal_amd_gpu_file_staging_refresh(
    iree_hal_amd_gpu_file_staging_t* staging,
    iree_hal_amd_gpu_file_staging_list_t* completed) {
  if (!iree_status_is_ok(staging->failure)) {
    for (uint32_t i = 0; i < 3; ++i) {
      iree_hal_amd_gpu_file_staging_fail_requests(&staging->requests[i],
                                                  staging->failure, completed);
    }
    for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
         bucket; bucket = bucket->next) {
      iree_hal_amd_gpu_file_staging_fail_requests(&bucket->waiting,
                                                  staging->failure, completed);
    }
    return;
  }
  for (uint32_t i = 0; i < 3; ++i) {
    for (iree_hal_amd_gpu_file_staging_request_t* request =
             staging->requests[i].head;
         request;) {
      iree_hal_amd_gpu_file_staging_request_t* next = request->next;
      const uint32_t mask = iree_hal_amd_gpu_file_staging_eligible(request);
      if (mask != request->block_mask) {
        iree_hal_amd_gpu_file_staging_list_remove(&staging->requests[i],
                                                  request);
        request->block_mask = mask;
        iree_hal_amd_gpu_file_staging_list_append(
            iree_hal_amd_gpu_file_staging_request_list(request), request);
      }
      request = next;
    }
  }
  for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
       bucket; bucket = bucket->next) {
    for (iree_hal_amd_gpu_file_staging_request_t* request =
             bucket->waiting.head;
         request;) {
      iree_hal_amd_gpu_file_staging_request_t* next = request->next;
      const uint32_t mask = iree_hal_amd_gpu_file_staging_eligible(request);
      if (mask) {
        iree_hal_amd_gpu_file_staging_list_remove(&bucket->waiting, request);
        request->block_mask = mask;
        iree_hal_amd_gpu_file_staging_list_append(&staging->requests[mask - 1],
                                                  request);
      }
      request = next;
    }
  }
}

static void iree_hal_amd_gpu_file_staging_ensure_views(
    iree_hal_amd_gpu_file_staging_bucket_t* bucket, bool retry_failed) {
  iree_hal_amd_gpu_file_staging_t* staging = bucket->staging;
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_amd_gpu_file_staging_block_t* block = &staging->blocks[i];
    iree_hal_amd_gpu_file_staging_view_t* view = &bucket->views[i];
    if (block->phase != IREE_HAL_AMD_GPU_FILE_STAGING_READY ||
        (view->phase != IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_EMPTY &&
         !(retry_failed && !view->attached &&
           view->phase == IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED))) {
      continue;
    }
    iree_status_free(view->status);
    view->status = iree_ok_status();
    view->block = block;
    view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_PENDING;
    view->next = staging->pending_views;
    staging->pending_views = view;
    ++block->pending_views;
  }
}

static void iree_hal_amd_gpu_file_staging_view_attached(void* user_data,
                                                        iree_status_t status) {
  iree_hal_amd_gpu_file_staging_view_t* view = user_data;
  iree_hal_amd_gpu_file_staging_t* staging = view->bucket->staging;
  iree_hal_device_t* device = staging->device;
  iree_hal_amd_gpu_file_staging_list_t completed = {0};
  iree_slim_mutex_lock(&staging->mutex);
  --view->block->pending_views;
  view->status = status;
  view->attached = iree_status_is_ok(status) && view->registration != NULL;
  view->phase = iree_status_is_ok(status)
                    ? IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_READY
                    : IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_fail_requests(&view->bucket->waiting, status,
                                                &completed);
  }
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_hal_amd_gpu_file_staging_complete_list(completed);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_file_staging_attach_view(
    iree_hal_amd_gpu_file_staging_view_t* view) {
  iree_hal_amd_gpu_file_staging_block_t* block = view->block;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_hal_device_retain(staging->device);
  iree_status_t status = iree_ok_status();
  if (!view->registration && block->preparation) {
    if (view->bucket->proactor == staging->proactor) {
      view->registration = block->preparation;
    } else {
      iree_async_buffer_registration_state_t registrations = {0};
      status = iree_async_proactor_clone_buffer_registration(
          view->bucket->proactor, &registrations, block->preparation,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
              IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
          &view->registration);
      if (iree_status_is_ok(status)) {
        iree_slim_mutex_lock(&staging->mutex);
        view->registration->next = block->memory.registrations.head;
        block->memory.registrations.head = view->registration;
        iree_slim_mutex_unlock(&staging->mutex);
      }
    }
  }
  if (iree_status_is_ok(status) && view->registration) {
    iree_async_proactor_attach_buffer(
        view->bucket->proactor, view->registration,
        (iree_async_buffer_registration_callback_t){
            iree_hal_amd_gpu_file_staging_view_attached, view});
  } else {
    iree_hal_amd_gpu_file_staging_view_attached(view, status);
  }
}

// A foreign poll owner is borrowed through file completion, not through device
// lifetime. Keep its prepared metadata but relinquish attached native slots
// before the final completed range permits the caller to stop polling.
static void iree_hal_amd_gpu_file_staging_settle_bucket(
    iree_hal_amd_gpu_file_staging_bucket_t* bucket,
    iree_hal_amd_gpu_file_staging_list_t* completed) {
  if (!bucket->completed.head) {
    return;
  }
  bool pending = false;
  if (!bucket->transfer_count) {
    for (uint32_t i = 0; i < 2; ++i) {
      iree_hal_amd_gpu_file_staging_view_t* view = &bucket->views[i];
      if (view->phase == IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_PENDING ||
          view->phase == IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING) {
        pending = true;
      } else if (view->attached &&
                 view->phase != IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED) {
        view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING;
        view->next = bucket->staging->pending_views;
        bucket->staging->pending_views = view;
        ++view->block->pending_views;
        pending = true;
      }
    }
  }
  if (!pending) {
    while (bucket->completed.head) {
      iree_hal_amd_gpu_file_staging_request_t* request = bucket->completed.head;
      iree_hal_amd_gpu_file_staging_list_remove(&bucket->completed, request);
      request->status = iree_status_join(
          request->status, iree_status_clone(bucket->staging->failure));
      iree_hal_amd_gpu_file_staging_list_append(completed, request);
    }
  }
}

static void iree_hal_amd_gpu_file_staging_view_detached(void* user_data,
                                                        iree_status_t status) {
  iree_hal_amd_gpu_file_staging_view_t* view = user_data;
  iree_hal_amd_gpu_file_staging_t* staging = view->bucket->staging;
  iree_hal_device_t* device = staging->device;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(staging->context, device, status);
    iree_slim_mutex_lock(&staging->mutex);
    staging->failure =
        iree_status_join(staging->failure, iree_status_clone(status));
    staging->cleanup_blocked = true;
    staging->dirty = true;
    view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED;
    view->status = status;
    iree_slim_mutex_unlock(&staging->mutex);
    // File/GPU uses have retired. Unknown registration ownership retains the
    // view, physical backing, and device use independently of completed ranges.
    iree_hal_amd_gpu_file_staging_progress(staging);
    return;
  }
  iree_slim_mutex_lock(&staging->mutex);
  --view->block->pending_views;
  view->attached = false;
  view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_EMPTY;
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_file_staging_detach_view(
    iree_hal_amd_gpu_file_staging_view_t* view) {
  iree_hal_device_retain(view->bucket->staging->device);
  iree_async_proactor_detach_buffer(
      view->bucket->proactor, view->registration,
      (iree_async_buffer_registration_callback_t){
          iree_hal_amd_gpu_file_staging_view_detached, view});
}

static void iree_hal_amd_gpu_file_staging_released(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_amd_gpu_file_staging_block_t* block = user_data;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_hal_device_t* device = staging->device;
  iree_hal_amd_gpu_file_staging_slot_t* slots = block->slots;
  const bool release_device = block->release_owns_device;
  iree_slim_mutex_lock(&staging->mutex);
  if (!iree_status_is_ok(status)) {
    staging->failure = iree_status_join(staging->failure, status);
  }
  for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
       bucket; bucket = bucket->next) {
    iree_hal_amd_gpu_file_staging_view_t* view = &bucket->views[block->ordinal];
    iree_status_free(view->status);
    *view = (iree_hal_amd_gpu_file_staging_view_t){.bucket = bucket};
  }
  staging->reserved_bytes -= block->reserved_bytes;
  block->reserved_bytes = 0;
  block->slots = block->available = NULL;
  block->phase = IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY;
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_allocator_free(staging->host_allocator, slots);
  iree_hal_amd_gpu_file_staging_progress(staging);
  if (release_device) {
    iree_hal_device_release(device);
  }
}

static void iree_hal_amd_gpu_file_staging_native_released(void* user_data) {
  iree_hal_amd_gpu_file_staging_block_t* block = user_data;
  if (block->traced) {
    iree_hal_memory_trace_free(&block->staging->trace, block);
  }
  block->publication.base = (iree_async_operation_t){
      .type = IREE_ASYNC_OPERATION_TYPE_NOP,
      .completion_fn = iree_hal_amd_gpu_file_staging_released,
      .user_data = block};
  IREE_CHECK_OK(iree_async_proactor_submit_one(block->staging->proactor,
                                               &block->publication.base));
}

static void iree_hal_amd_gpu_file_staging_detach_join(
    iree_hal_amd_gpu_file_staging_block_t* block) {
  if (iree_atomic_fetch_sub(&block->pending_detaches, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  if (block->detach_failed) {
    return;
  }
  iree_hal_amd_gpu_context_release_memory(
      block->staging->context, 1, &block->memory,
      (iree_hal_slab_release_callback_t){
          iree_hal_amd_gpu_file_staging_native_released, block},
      &block->release);
}

static void iree_hal_amd_gpu_file_staging_detached(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_gpu_file_staging_view_t* view = user_data;
  iree_hal_amd_gpu_file_staging_block_t* block = view->block;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report_status(staging->context, staging->device,
                                           status);
  }
  iree_slim_mutex_lock(&staging->mutex);
  if (iree_status_is_ok(status)) {
    view->attached = false;
    view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_EMPTY;
  } else {
    view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_FAILED;
    staging->failure = iree_status_join(staging->failure, status);
    block->detach_failed = true;
    staging->cleanup_blocked = true;
    iree_notification_post(&staging->notification, IREE_ALL_WAITERS);
  }
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  // File completion joins its registration detach, not the subsequent native
  // free. Publish this view's relinquishment while block release still owns it.
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_hal_amd_gpu_file_staging_detach_join(block);
}

static void iree_hal_amd_gpu_file_staging_block_release(
    iree_hal_amd_gpu_file_staging_block_t* block) {
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_atomic_store(&block->pending_detaches, 1, iree_memory_order_relaxed);
  iree_slim_mutex_lock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
  // Block retirement and final foreign file completion share the same view
  // transition. Mark ownership before callbacks can request another detach.
  for (iree_hal_amd_gpu_file_staging_bucket_t* current = bucket; current;
       current = current->next) {
    iree_hal_amd_gpu_file_staging_view_t* view =
        &current->views[block->ordinal];
    if (view->attached) {
      view->phase = IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING;
    }
  }
  iree_slim_mutex_unlock(&staging->mutex);
  for (; bucket; bucket = bucket->next) {
    iree_hal_amd_gpu_file_staging_view_t* view = &bucket->views[block->ordinal];
    if (!view->attached) {
      continue;
    }
    iree_atomic_fetch_add(&block->pending_detaches, 1,
                          iree_memory_order_relaxed);
    iree_async_proactor_detach_buffer(
        bucket->proactor, view->registration,
        (iree_async_buffer_registration_callback_t){
            iree_hal_amd_gpu_file_staging_detached, view});
  }
  iree_hal_amd_gpu_file_staging_detach_join(block);
}

static void iree_hal_amd_gpu_file_staging_construction_failed(
    iree_hal_amd_gpu_file_staging_block_t* block, iree_status_t status) {
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_hal_amd_gpu_file_staging_list_t completed = {0};
  iree_hal_amd_gpu_file_staging_slot_t* slots = NULL;
  iree_slim_mutex_lock(&staging->mutex);
  staging->growth_requested = false;
  for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
       bucket; bucket = bucket->next) {
    iree_hal_amd_gpu_file_staging_fail_requests(&bucket->waiting, status,
                                                &completed);
  }
  if (block->memory.handle) {
    block->phase = IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING;
  } else {
    staging->reserved_bytes -= block->reserved_bytes;
    block->reserved_bytes = 0;
    slots = block->slots;
    block->slots = block->available = NULL;
    block->phase = IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY;
  }
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_status_free(status);
  iree_allocator_free(staging->host_allocator, slots);
  iree_hal_amd_gpu_file_staging_complete_list(completed);
}

static void iree_hal_amd_gpu_file_staging_published(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_hal_amd_gpu_file_staging_block_t* block = user_data;
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_hal_device_t* device = staging->device;
  status = iree_status_join(block->status, status);
  block->status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    block->traced = true;
    iree_hal_memory_trace_alloc(
        &staging->trace, block,
        block->memory.info.native_allocation_byte_length);
    block->preparation = block->memory.registrations.head;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        staging->context->api->memory_query_address(
            block->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &block->address),
        "memory_query_address(file staging)");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_file_staging_query_effects(block, true);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_file_staging_query_effects(block, false);
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&staging->mutex);
    block->phase =
        block->retire_on_publish || !iree_status_is_ok(staging->failure)
            ? IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING
            : IREE_HAL_AMD_GPU_FILE_STAGING_READY;
    staging->dirty = true;
    iree_slim_mutex_unlock(&staging->mutex);
  } else {
    iree_hal_amd_gpu_file_staging_construction_failed(block, status);
  }
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_hal_device_release(device);
}

static void iree_hal_amd_gpu_file_staging_acquired(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_gpu_file_staging_block_t* block = user_data;
  block->status = status;
  block->publication.base = (iree_async_operation_t){
      .type = IREE_ASYNC_OPERATION_TYPE_NOP,
      .completion_fn = iree_hal_amd_gpu_file_staging_published,
      .user_data = block};
  IREE_CHECK_OK(iree_async_proactor_submit_one(block->staging->proactor,
                                               &block->publication.base));
}

static void iree_hal_amd_gpu_file_staging_block_allocate(
    iree_hal_amd_gpu_file_staging_block_t* block) {
  iree_hal_amd_gpu_file_staging_t* staging = block->staging;
  iree_status_t status = iree_allocator_malloc(
      staging->host_allocator, block->lane_count * sizeof(*block->slots),
      (void**)&block->slots);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_file_staging_construction_failed(block, status);
    return;
  }
  for (uint32_t i = 0; i < block->lane_count; ++i) {
    block->slots[i].block = block;
    block->slots[i].offset = (uint64_t)i * block->capacity;
    block->slots[i].next = block->available;
    block->available = &block->slots[i];
  }
  iree_hal_device_retain(staging->device);
  iree_hal_amd_gpu_context_acquire_memory(
      staging->context, 1, &block->allocation, &block->memory,
      (iree_hal_slab_acquire_callback_t){iree_hal_amd_gpu_file_staging_acquired,
                                         block},
      &block->acquire);
}

// The caller holds the metadata lock. All credits include native rounding and
// remain charged through final release, including allocations being prepared.
static iree_hal_amd_gpu_file_staging_block_t*
iree_hal_amd_gpu_file_staging_plan_growth(
    iree_hal_amd_gpu_file_staging_t* staging) {
  if (!staging->growth_requested || !staging->request_count ||
      !iree_status_is_ok(staging->failure) || staging->closing) {
    return NULL;
  }
  iree_hal_amd_gpu_file_staging_block_t* empty = NULL;
  uint32_t lane_count = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_amd_gpu_file_staging_block_t* block = &staging->blocks[i];
    if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_CONSTRUCTING) {
      return NULL;
    }
    if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY) {
      empty = block;
    }
    if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_READY &&
        block->capacity >= staging->desired_capacity) {
      lane_count += block->lane_count;
    }
  }
  if (lane_count >= staging->lane_count) {
    staging->growth_requested = false;
    return NULL;
  }
  if (!empty) {
    return NULL;
  }
  const amdf_memory_construction_capabilities_t* allocation =
      &staging->memory.profile.allocation;
  const uint64_t available = staging->byte_budget - staging->reserved_bytes;
  uint64_t maximum_length = available /
                            allocation->native_byte_length_granularity *
                            allocation->native_byte_length_granularity;
  if (maximum_length <= allocation->native_byte_length_prefix) {
    return NULL;
  }
  maximum_length =
      iree_min(maximum_length - allocation->native_byte_length_prefix,
               allocation->maximum_byte_length);
  maximum_length = maximum_length / allocation->byte_length_granularity *
                   allocation->byte_length_granularity;
  lane_count = (uint32_t)iree_min(staging->lane_count - lane_count,
                                  maximum_length / staging->desired_capacity);
  if (!lane_count) {
    return NULL;
  }
  const uint32_t ordinal = empty->ordinal;
  *empty = (iree_hal_amd_gpu_file_staging_block_t){
      .staging = staging,
      .ordinal = ordinal,
      .phase = IREE_HAL_AMD_GPU_FILE_STAGING_CONSTRUCTING,
      .capacity = staging->desired_capacity,
      .lane_count = lane_count,
      .reserved_bytes = iree_hal_amd_gpu_file_staging_native_length(
          staging, (uint64_t)staging->desired_capacity * lane_count),
      .allocation = {
          .options = &staging->memory,
          .byte_length = (uint64_t)staging->desired_capacity * lane_count,
          .alignment = iree_min(staging->desired_capacity,
                                allocation->maximum_alignment)}};
  staging->reserved_bytes += empty->reserved_bytes;
  return empty;
}

static void iree_hal_amd_gpu_file_staging_progress(
    iree_hal_amd_gpu_file_staging_t* staging) {
  iree_slim_mutex_lock(&staging->mutex);
  if (staging->pumping) {
    iree_slim_mutex_unlock(&staging->mutex);
    return;
  }
  staging->pumping = true;
  const bool retain_device = !staging->closing;
  iree_hal_device_t* device = staging->device;
  if (retain_device) {
    iree_hal_device_retain(device);
  }
  iree_slim_mutex_unlock(&staging->mutex);
  for (;;) {
    iree_hal_amd_gpu_file_staging_list_t completed = {0};
    iree_slim_mutex_lock(&staging->mutex);
    if (staging->dirty) {
      staging->dirty = false;
      if (staging->trim_requested && !staging->transfer_count) {
        staging->trim_requested = false;
        staging->desired_capacity = 0;
        staging->growth_requested = false;
        for (uint32_t i = 0; i < 2; ++i) {
          if (staging->blocks[i].phase == IREE_HAL_AMD_GPU_FILE_STAGING_READY) {
            staging->blocks[i].phase = IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING;
          } else if (staging->blocks[i].phase ==
                     IREE_HAL_AMD_GPU_FILE_STAGING_CONSTRUCTING) {
            staging->blocks[i].retire_on_publish = true;
          }
        }
      }
      for (uint32_t i = 0; i < 2; ++i) {
        if (staging->blocks[i].phase == IREE_HAL_AMD_GPU_FILE_STAGING_READY &&
            staging->blocks[i ^ 1].phase ==
                IREE_HAL_AMD_GPU_FILE_STAGING_READY &&
            staging->blocks[i].capacity < staging->blocks[i ^ 1].capacity) {
          staging->blocks[i].phase = IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING;
        }
      }
      iree_hal_amd_gpu_file_staging_refresh(staging, &completed);
      for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
           bucket; bucket = bucket->next) {
        iree_hal_amd_gpu_file_staging_settle_bucket(bucket, &completed);
      }
      if (iree_status_is_ok(staging->failure)) {
        for (iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
             bucket; bucket = bucket->next) {
          if (bucket->transfer_count) {
            iree_hal_amd_gpu_file_staging_ensure_views(bucket, false);
          }
        }
      }
    }
    if (completed.head) {
      iree_slim_mutex_unlock(&staging->mutex);
      iree_hal_amd_gpu_file_staging_complete_list(completed);
      continue;
    }
    if (staging->blocks[0].phase == IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY &&
        staging->blocks[1].phase == IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY) {
      iree_hal_amd_gpu_file_staging_bucket_t** link = &staging->buckets;
      while (*link && ((*link)->transfer_count || (*link)->completed.head)) {
        link = &(*link)->next;
      }
      iree_hal_amd_gpu_file_staging_bucket_t* unused = *link;
      if (unused) {
        *link = unused->next;
        iree_slim_mutex_unlock(&staging->mutex);
        iree_async_proactor_release(unused->proactor);
        iree_allocator_free(staging->host_allocator, unused);
        continue;
      }
    }
    iree_hal_amd_gpu_file_staging_block_t* release = NULL;
    for (uint32_t i = 0; i < 2; ++i) {
      iree_hal_amd_gpu_file_staging_block_t* block = &staging->blocks[i];
      if (block->phase == IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING &&
          !block->active_count && !block->pending_views &&
          !staging->requests[(1u << i) - 1].head &&
          !staging->requests[2].head) {
        block->phase = IREE_HAL_AMD_GPU_FILE_STAGING_RELEASING;
        block->release_owns_device = !staging->closing;
        if (block->release_owns_device) {
          iree_hal_device_retain(device);
        }
        release = block;
        break;
      }
    }
    if (release) {
      iree_slim_mutex_unlock(&staging->mutex);
      iree_hal_amd_gpu_file_staging_block_release(release);
      continue;
    }
    iree_hal_amd_gpu_file_staging_view_t* view = staging->pending_views;
    if (view) {
      staging->pending_views = view->next;
      view->next = NULL;
      iree_slim_mutex_unlock(&staging->mutex);
      if (view->phase == IREE_HAL_AMD_GPU_FILE_STAGING_VIEW_DETACHING) {
        iree_hal_amd_gpu_file_staging_detach_view(view);
      } else {
        iree_hal_amd_gpu_file_staging_attach_view(view);
      }
      continue;
    }
    iree_hal_amd_gpu_file_staging_block_t* grow =
        iree_hal_amd_gpu_file_staging_plan_growth(staging);
    if (grow) {
      iree_slim_mutex_unlock(&staging->mutex);
      iree_hal_amd_gpu_file_staging_block_allocate(grow);
      continue;
    }
    uint32_t available = 0;
    for (uint32_t i = 0; i < 2; ++i) {
      if ((staging->blocks[i].phase == IREE_HAL_AMD_GPU_FILE_STAGING_READY ||
           staging->blocks[i].phase ==
               IREE_HAL_AMD_GPU_FILE_STAGING_DRAINING) &&
          staging->blocks[i].available) {
        available |= 1u << i;
      }
    }
    iree_hal_amd_gpu_file_staging_request_t* request = NULL;
    for (uint32_t i = 0; i < 3; ++i) {
      const uint32_t queue = (staging->next_queue + i) % 3;
      if (staging->requests[queue].head && ((queue + 1) & available)) {
        request = staging->requests[queue].head;
        staging->next_queue = (queue + 1) % 3;
        break;
      }
    }
    if (!request) {
      staging->pumping = false;
      if (staging->closing) {
        iree_notification_post(&staging->notification, IREE_ALL_WAITERS);
      }
      iree_slim_mutex_unlock(&staging->mutex);
      if (retain_device) {
        iree_hal_device_release(device);
      }
      return;
    }
    const uint32_t ordinal = (request->block_mask & available & 1) ? 0 : 1;
    iree_hal_amd_gpu_file_staging_block_t* block = &staging->blocks[ordinal];
    iree_hal_amd_gpu_file_staging_slot_t* slot = block->available;
    block->available = slot->next;
    ++block->active_count;
    ++request->in_flight;
    slot->request = request;
    const uint64_t offset = request->cursor;
    const uint32_t length =
        (uint32_t)iree_min(request->end - offset, block->capacity);
    iree_async_buffer_registration_entry_t* registration =
        request->bucket->views[ordinal].registration;
    iree_status_t status = iree_hal_amd_fabric_file_request_prepare(
        request->transfer, offset,
        registration
            ? iree_async_span_make(registration->region, slot->offset, length)
            : iree_async_span_from_ptr(
                  block->memory.contents.data + slot->offset, length),
        &slot->io);
    if (iree_status_is_ok(status)) {
      request->cursor += slot->io.span.length;
      iree_hal_amd_gpu_file_staging_list_remove(
          iree_hal_amd_gpu_file_staging_request_list(request), request);
      if (request->cursor < request->end) {
        iree_hal_amd_gpu_file_staging_list_append(
            iree_hal_amd_gpu_file_staging_request_list(request), request);
      }
    }
    iree_slim_mutex_unlock(&staging->mutex);
    if (!iree_status_is_ok(status)) {
      iree_hal_amd_gpu_file_staging_slot_finish(slot, status);
    } else if (request->transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ) {
      iree_hal_amd_gpu_file_staging_issue_io(
          slot, iree_hal_amd_gpu_file_staging_copy);
    } else {
      iree_hal_amd_gpu_file_staging_copy(slot, iree_ok_status());
    }
  }
}

static iree_status_t iree_hal_amd_gpu_file_staging_bucket(
    iree_hal_amd_gpu_file_staging_t* staging, iree_async_proactor_t* proactor,
    iree_hal_amd_gpu_file_staging_bucket_t** out_bucket) {
  iree_hal_amd_gpu_file_staging_bucket_t* candidate = NULL;
  for (;;) {
    iree_slim_mutex_lock(&staging->mutex);
    iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
    while (bucket && bucket->proactor != proactor) {
      bucket = bucket->next;
    }
    if (!bucket && candidate) {
      candidate->next = staging->buckets;
      staging->buckets = candidate;
      bucket = candidate;
      candidate = NULL;
    }
    if (bucket) {
      ++bucket->transfer_count;
      ++staging->transfer_count;
    }
    iree_slim_mutex_unlock(&staging->mutex);
    if (bucket) {
      if (candidate) {
        iree_async_proactor_release(candidate->proactor);
        iree_allocator_free(staging->host_allocator, candidate);
      }
      *out_bucket = bucket;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        staging->host_allocator, sizeof(*candidate), (void**)&candidate));
    candidate->staging = staging;
    candidate->proactor = proactor;
    for (uint32_t i = 0; i < 2; ++i) {
      candidate->views[i].bucket = candidate;
    }
    iree_async_proactor_retain(proactor);
  }
}

iree_status_t iree_hal_amd_gpu_file_staging_acquire(
    iree_hal_amd_gpu_file_staging_t* staging,
    const iree_hal_amd_fabric_file_transfer_t* transfer,
    const iree_hal_semaphore_list_t* signals,
    iree_hal_amd_gpu_file_staging_request_t* request) {
  *request = (iree_hal_amd_gpu_file_staging_request_t){
      .staging = staging, .transfer = transfer, .signals = signals};
  return iree_hal_amd_gpu_file_staging_bucket(
      staging, transfer->async.file->proactor, &request->bucket);
}

void iree_hal_amd_gpu_file_staging_submit(
    iree_hal_amd_gpu_file_staging_request_t* request, uint64_t offset,
    uint64_t length, uint64_t device_address, iree_hal_memory_effects_t effects,
    iree_hal_amd_fabric_file_transfer_callback_t completion) {
  iree_hal_amd_gpu_file_staging_t* staging = request->staging;
  request->cursor = offset;
  request->end = offset + length;
  request->device_address = device_address;
  request->effects = effects;
  request->completion = completion;
  request->in_flight = 1;
  request->block_mask = 0;
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(request->transfer->file);
  uint64_t capacity = iree_max(
      4096, request->transfer->file_access == IREE_HAL_MEMORY_ACCESS_READ
                ? requirements->read.buffer_alignment
                : requirements->write.buffer_alignment);
  const uint64_t desired = iree_min(length, staging->lane_capacity);
  while (capacity < desired) {
    capacity <<= 1;
  }
  request->chunk_capacity = (uint32_t)capacity;
  iree_slim_mutex_lock(&staging->mutex);
  ++staging->request_count;
  if (request->bucket->completed.head) {
    staging->dirty = true;
  }
  if (!iree_status_is_ok(staging->failure)) {
    request->status = iree_status_clone(staging->failure);
    request->cursor = request->end;
  } else {
    staging->desired_capacity =
        iree_max(staging->desired_capacity, request->chunk_capacity);
    staging->growth_requested = true;
    iree_hal_amd_gpu_file_staging_ensure_views(request->bucket, true);
    request->block_mask = iree_hal_amd_gpu_file_staging_eligible(request);
    iree_hal_amd_gpu_file_staging_list_append(
        iree_hal_amd_gpu_file_staging_request_list(request), request);
  }
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
  iree_slim_mutex_lock(&staging->mutex);
  const bool complete =
      --request->in_flight == 0 && request->cursor == request->end;
  if (complete) {
    --staging->request_count;
  }
  iree_slim_mutex_unlock(&staging->mutex);
  if (complete) {
    iree_hal_amd_gpu_file_staging_request_complete(request);
  }
}

void iree_hal_amd_gpu_file_staging_release(
    iree_hal_amd_gpu_file_staging_request_t* request, iree_status_t status,
    iree_hal_amd_fabric_file_transfer_callback_t completion) {
  iree_hal_amd_gpu_file_staging_t* staging = request->staging;
  request->status = status;
  request->completion = completion;
  iree_slim_mutex_lock(&staging->mutex);
  const bool complete =
      iree_hal_amd_gpu_file_staging_transfer_released(request);
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
  if (complete) {
    iree_hal_amd_gpu_file_staging_request_complete(request);
  }
}

void iree_hal_amd_gpu_file_staging_trim(
    iree_hal_amd_gpu_file_staging_t* staging) {
  if (!staging) {
    return;
  }
  iree_slim_mutex_lock(&staging->mutex);
  staging->trim_requested = true;
  staging->dirty = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_progress(staging);
}

static bool iree_hal_amd_gpu_file_staging_joined(void* user_data) {
  iree_hal_amd_gpu_file_staging_t* staging = user_data;
  iree_slim_mutex_lock(&staging->mutex);
  const bool complete =
      (!staging->pumping &&
       staging->blocks[0].phase == IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY &&
       staging->blocks[1].phase == IREE_HAL_AMD_GPU_FILE_STAGING_EMPTY) ||
      staging->cleanup_blocked;
  iree_slim_mutex_unlock(&staging->mutex);
  return complete;
}

bool iree_hal_amd_gpu_file_staging_free(
    iree_hal_amd_gpu_file_staging_t* staging) {
  if (!staging) {
    return true;
  }
  iree_slim_mutex_lock(&staging->mutex);
  staging->closing = true;
  iree_slim_mutex_unlock(&staging->mutex);
  iree_hal_amd_gpu_file_staging_trim(staging);
  iree_notification_await(&staging->notification,
                          iree_hal_amd_gpu_file_staging_joined, staging,
                          iree_infinite_timeout());
  iree_slim_mutex_lock(&staging->mutex);
  const bool cleanup_blocked = staging->cleanup_blocked;
  iree_slim_mutex_unlock(&staging->mutex);
  if (cleanup_blocked) {
    return false;
  }
  while (staging->buckets) {
    iree_hal_amd_gpu_file_staging_bucket_t* bucket = staging->buckets;
    staging->buckets = bucket->next;
    iree_async_proactor_release(bucket->proactor);
    iree_allocator_free(staging->host_allocator, bucket);
  }
  iree_hal_memory_trace_deinitialize(&staging->trace);
  iree_status_free(staging->failure);
  iree_notification_deinitialize(&staging->notification);
  iree_slim_mutex_deinitialize(&staging->mutex);
  iree_allocator_free(staging->host_allocator, staging);
  return true;
}

iree_status_t iree_hal_amd_gpu_file_staging_allocate(
    iree_hal_device_t* device, iree_hal_amd_gpu_context_t* context,
    uint32_t family_ordinal, iree_hal_amd_gpu_sdma_queue_t* queue,
    iree_async_proactor_t* proactor, uint32_t lane_count,
    uint32_t lane_capacity, uint32_t numa_node, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_file_staging_t** out_staging) {
  *out_staging = NULL;
  iree_hal_amd_gpu_file_staging_t* staging = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*staging),
                                             (void**)&staging));
  staging->device = device;
  staging->context = context;
  staging->queue = queue;
  staging->proactor = proactor;
  for (uint32_t i = 0; i < 2; ++i) {
    staging->blocks[i].staging = staging;
    staging->blocks[i].ordinal = i;
  }
  staging->host_allocator = host_allocator;
  staging->family_ordinal = family_ordinal;
  staging->lane_count = lane_count;
  staging->lane_capacity = lane_capacity;
  iree_slim_mutex_initialize(&staging->mutex);
  iree_notification_initialize(&staging->notification);
  staging->access = (amdf_memory_device_access_t){
      .device = context->device,
      .requirements = {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags =
              AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT,
          .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU}};
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info)};
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(context->api->memory_scope_query_info(
                                        context->fabric->system_scope, &info),
                                    "memory_scope_query_info(file staging)");
  bool found = false, registered = false;
  for (uint32_t i = 0; i < info.memory_profile_count && !registered &&
                       iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access)};
    const amdf_status_t native =
        context->api->memory_scope_query_device_profile(
            context->fabric->system_scope, i, 1, &staging->access, &profile,
            &access);
    if (amdf_status_code(native) == AMDF_STATUS_CODE_UNSUPPORTED) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native, "file staging profile");
    const bool file_capable =
        profile.supported_flags & AMDF_MEMORY_FLAG_HOST_FILE_IO;
    const amdf_memory_flags_t flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE |
        (file_capable ? AMDF_MEMORY_FLAG_HOST_FILE_IO : 0);
    if (iree_status_is_ok(status) && amdf_status_is_ok(profile.availability) &&
        (!found || file_capable) &&
        iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_all_bits_set(profile.supported_flags, flags)) {
      staging->memory = (iree_hal_amd_fabric_memory_options_t){
          .scope = context->fabric->system_scope,
          .profile = profile,
          .access_count = 1,
          .accesses = &staging->access,
          .required_flags = flags,
          .host_access =
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE};
      if (numa_node != UINT32_MAX &&
          (profile.allocation.placement_kinds &
           (UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA))) {
        staging->memory.placement = (amdf_memory_placement_t){
            .kind = AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, .node = numa_node};
      }
      found = true;
      registered = file_capable;
    }
  }
  if (iree_status_is_ok(status) && !found) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "GPU has no coherent host-visible staging profile");
  }
  if (iree_status_is_ok(status)) {
    const amdf_memory_construction_capabilities_t* allocation =
        &staging->memory.profile.allocation;
    const uint64_t maximum_byte_length = allocation->maximum_byte_length /
                                         allocation->byte_length_granularity *
                                         allocation->byte_length_granularity;
    while (staging->lane_capacity > maximum_byte_length) {
      staging->lane_capacity >>= 1;
    }
    if (staging->lane_capacity < 4096) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "native staging cannot provide 4 KiB lanes");
    }
  }
  if (iree_status_is_ok(status)) {
    staging->byte_budget = iree_hal_amd_gpu_file_staging_native_length(
        staging, (uint64_t)staging->lane_capacity * staging->lane_count);
    if (staging->memory.required_flags & AMDF_MEMORY_FLAG_HOST_FILE_IO) {
      staging->memory.io.proactor_count = 1;
      staging->memory.io.proactors = &staging->proactor;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_memory_trace_initialize_domain("libamdf-numa", numa_node,
                                                     &staging->trace);
  }
  if (iree_status_is_ok(status)) {
    *out_staging = staging;
  } else {
    iree_hal_amd_gpu_file_staging_free(staging);
  }
  return status;
}
