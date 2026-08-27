// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/device_internal.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_chain_cache.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_single_cache.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/utils/resource_set.h"

static const uint64_t kAmdxdnaControlCodeOpcode = 3u;

static bool iree_hal_amdxdna_patch_table_is_valid(
    const iree_hal_amdxdna_u32_list_t* patch_table) {
  return patch_table && patch_table->count != 0 &&
         (patch_table->count % 3 == 0);
}

typedef struct iree_hal_amdxdna_direct_command_buffer {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // A resource set to maintain references to all resources used within this
  // one-shot command buffer.
  iree_hal_resource_set_t* resource_set;
  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  iree_hal_amdxdna_device* device;

  // Dispatches lowered through the host-patched path accumulate here until
  // end(). A single child is submitted directly. Multi-child groups flush as
  // ERT_CMD_CHAIN(s) when supported, or as direct child submissions otherwise.
  iree_hal_amdxdna_chain_accum_t chain_accum;
  // Optional batch installed by queue execution. When present and native caps
  // support async submit, native commands are issued without waiting here; the
  // batch owns native completion, cleanup, and HAL semaphore signaling.
  iree_hal_amdxdna_completion_batch_t* completion_batch;
} iree_hal_amdxdna_direct_command_buffer;

typedef struct iree_hal_amdxdna_cached_single_release_t {
  iree_hal_amdxdna_device_single_command_cache_t* cache;
  iree_hal_amdxdna_single_command_cache_entry_t* entry;
} iree_hal_amdxdna_cached_single_release_t;

typedef struct iree_hal_amdxdna_cached_chain_release_t {
  iree_hal_amdxdna_device_chain_command_cache_t* cache;
  iree_hal_amdxdna_chain_command_cache_entry_t* entry;
} iree_hal_amdxdna_cached_chain_release_t;

static bool iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
    const iree_hal_amdxdna_direct_command_buffer* command_buffer) {
  return command_buffer->completion_batch &&
         command_buffer->device->native_caps.submit_completion_is_deferred;
}

static void iree_hal_amdxdna_completion_destroy_native_command(
    void* user_data) {
  iree_hal_amdxdna_native_command_c_destroy(
      (iree_hal_amdxdna_native_command_t*)user_data);
}

static void iree_hal_amdxdna_completion_destroy_native_buffer(void* user_data) {
  iree_hal_amdxdna_native_buffer_c_destroy(
      (iree_hal_amdxdna_native_buffer_t*)user_data);
}

static void iree_hal_amdxdna_completion_release_single_cache_entry(
    void* user_data) {
  iree_hal_amdxdna_cached_single_release_t* release =
      (iree_hal_amdxdna_cached_single_release_t*)user_data;
  if (!release) return;
  iree_hal_amdxdna_single_command_cache_entry_release_in_flight(release->cache,
                                                                release->entry);
  iree_allocator_free(release->cache->host_allocator, release);
}

static void iree_hal_amdxdna_completion_release_chain_cache_entry(
    void* user_data) {
  iree_hal_amdxdna_cached_chain_release_t* release =
      (iree_hal_amdxdna_cached_chain_release_t*)user_data;
  if (!release) return;
  iree_hal_amdxdna_chain_command_cache_entry_release_in_flight(release->cache,
                                                               release->entry);
  iree_allocator_free(release->cache->host_allocator, release);
}

static void iree_hal_amdxdna_completion_release_context_ref(void* user_data) {
  iree_hal_amdxdna_native_context_ref_release(
      (iree_hal_amdxdna_native_context_ref_t*)user_data);
}

void iree_hal_amdxdna_device_invalidate_command_caches_for_queue(
    iree_hal_amdxdna_device* device, iree_hal_amdxdna_native_queue_t* queue) {
  if (!device || !queue) return;
  iree_hal_amdxdna_single_command_cache_invalidate_queue(
      device->single_command_cache, queue);
  iree_hal_amdxdna_chain_command_cache_invalidate_queue(
      device->chain_command_cache, queue);
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* entry) {
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer) ||
      !entry) {
    return iree_ok_status();
  }
  iree_hal_amdxdna_cached_single_release_t* release = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      cache->host_allocator, sizeof(*release), (void**)&release));
  release->cache = cache;
  release->entry = entry;
  iree_status_t status = iree_hal_amdxdna_completion_batch_add_cleanup(
      command_buffer->completion_batch,
      iree_hal_amdxdna_completion_release_single_cache_entry, release);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(cache->host_allocator, release);
  }
  return status;
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_defer_chain_cache_release(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_device_chain_command_cache_t* cache,
    iree_hal_amdxdna_chain_command_cache_entry_t* entry) {
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer) ||
      !entry) {
    return iree_ok_status();
  }
  iree_hal_amdxdna_cached_chain_release_t* release = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      cache->host_allocator, sizeof(*release), (void**)&release));
  release->cache = cache;
  release->entry = entry;
  iree_status_t status = iree_hal_amdxdna_completion_batch_add_cleanup(
      command_buffer->completion_batch,
      iree_hal_amdxdna_completion_release_chain_cache_entry, release);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(cache->host_allocator, release);
  }
  return status;
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_defer_native_command_destroy(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_command_t** inout_command) {
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer) ||
      !*inout_command) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_completion_batch_add_cleanup(
      command_buffer->completion_batch,
      iree_hal_amdxdna_completion_destroy_native_command, *inout_command));
  *inout_command = NULL;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_defer_native_buffer_destroy(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_buffer_t** inout_buffer) {
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer) ||
      !*inout_buffer) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_completion_batch_add_cleanup(
      command_buffer->completion_batch,
      iree_hal_amdxdna_completion_destroy_native_buffer, *inout_buffer));
  *inout_buffer = NULL;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_defer_native_command_and_buffer_destroy(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_command_t** inout_command,
    iree_hal_amdxdna_native_buffer_t** inout_buffer) {
  // Completion cleanups run in reverse registration order. Register the buffer
  // first so the command is destroyed before the control-code BO it references.
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_direct_command_buffer_defer_native_buffer_destroy(
          command_buffer, inout_buffer));
  return iree_hal_amdxdna_direct_command_buffer_defer_native_command_destroy(
      command_buffer, inout_command);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_submit(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer)) {
    return iree_hal_amdxdna_native_queue_c_submit_and_wait(queue, command,
                                                           label);
  }
  iree_hal_amdxdna_native_submission_t* submission = NULL;
  iree_status_t status = iree_hal_amdxdna_native_queue_c_submit(
      queue, command, label, &submission);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_completion_batch_add_submission(
        command_buffer->completion_batch, submission);
    if (iree_status_is_ok(status)) {
      submission = NULL;  // completion batch owns it.
    }
  }
  iree_hal_amdxdna_native_submission_c_destroy(submission);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_submit_all(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t label) {
  if (command_count == 0) return iree_ok_status();
  if (!iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer)) {
    return iree_hal_amdxdna_native_queue_c_submit_all_and_wait(
        queue, commands, command_count, label);
  }
  // This helper is only used for parent-chain batches. Keep those in the native
  // batch path so Windows MCDM can issue all parent chains into distinct
  // completion slots and retire them with one collective wait. Represent the
  // whole native batch as one async completion item instead of splitting it
  // into independent parent submissions.
  iree_hal_amdxdna_native_submission_t* submission = NULL;
  iree_status_t status = iree_hal_amdxdna_native_queue_c_submit_all(
      queue, commands, command_count, label, &submission);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_completion_batch_add_submission(
        command_buffer->completion_batch, submission);
    if (iree_status_is_ok(status)) {
      submission = NULL;  // completion batch owns it.
    }
  }
  iree_hal_amdxdna_native_submission_c_destroy(submission);
  return status;
}

static iree_status_t iree_hal_amdxdna_validate_live_dispatch_bindings(
    iree_hal_buffer_ref_list_t bindings) {
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (IREE_UNLIKELY(!bindings.values || !bindings.values[i].buffer)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch binding %" PRIhsz " is NULL", i);
    }
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(bindings.values[i].buffer);
    if (iree_hal_amdxdna_buffer_is_deallocated(allocated_buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding %" PRIhsz " is a deallocated amdxdna buffer", i);
    }
  }
  return iree_ok_status();
}

extern const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable;

iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  if (binding_capacity > 0) {
    // Indirect command buffers with binding tables are not supported by this
    // direct recording path.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }
  // The amdxdna direct CB has submission state: begin/end are not implemented
  // as resets, and accumulated dispatch groups are finalized by end(). A
  // non-ONE_SHOT CB would carry that state across reuses. Require ONE_SHOT to
  // match the only mode IREE creates through us today (queue_execute passes
  // ONE_SHOT | ALLOW_INLINE_EXECUTION | UNVALIDATED) and to fail loudly if a
  // future caller hands us a reusable CB.
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna command buffers require "
                            "IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT");
  }
  if (iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna command buffers require retained resource lifetimes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_amdxdna_direct_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device = device;
  iree_hal_amdxdna_chain_accum_initialize(&command_buffer->chain_accum);
  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);

  return status;
}

static void iree_hal_amdxdna_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_hal_amdxdna_chain_accum_deinitialize(host_allocator,
                                            &command_buffer->chain_accum);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_amdxdna_direct_command_buffer_set_completion_batch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_amdxdna_completion_batch_t* completion_batch) {
  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  command_buffer->completion_batch = completion_batch;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  // Command buffers are one-shot; create initializes all per-recording state
  // and end() flushes any accumulated chain commands.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barrier_count;
  (void)memory_barriers;

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t* barrier = &buffer_barriers[i];
    const bool flush_host_to_device =
        iree_any_bit_set(barrier->source_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier->target_scope,
                         IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
                             IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                             IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    const bool invalidate_device_to_host =
        iree_any_bit_set(barrier->source_scope,
                         IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                             IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE) &&
        iree_any_bit_set(barrier->target_scope,
                         IREE_HAL_ACCESS_SCOPE_HOST_READ |
                             IREE_HAL_ACCESS_SCOPE_MEMORY_READ);
    if (!flush_host_to_device && !invalidate_device_to_host) {
      continue;
    }
    if (IREE_UNLIKELY(!barrier->buffer_ref.buffer)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "amdxdna direct command buffer cannot sync an indirect buffer "
          "barrier without a resolved buffer");
    }
    if (flush_host_to_device) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_flush_range(
          barrier->buffer_ref.buffer, barrier->buffer_ref.offset,
          barrier->buffer_ref.length));
    }
    if (invalidate_device_to_host) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_buffer_invalidate_range(
          barrier->buffer_ref.buffer, barrier->buffer_ref.offset,
          barrier->buffer_ref.length));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  // The amdxdna direct command buffer executes synchronously against a single
  // in-order queue today, so recording an event signal has no extra device work
  // to enqueue.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)event_count;
  (void)events;
  return iree_hal_amdxdna_direct_command_buffer_execution_barrier(
      base_command_buffer, source_stage_mask, target_stage_mask,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, memory_barrier_count,
      memory_barriers, buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  if (target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)source_buffer;
  (void)source_offset;
  (void)target_ref;
  (void)flags;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna command_buffer_update_buffer requires native blit support; "
      "host-emulated map/sync/memcpy transfers are not available on device "
      "queues");
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)target_ref;
  (void)pattern;
  (void)pattern_length;
  (void)flags;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna command_buffer_fill_buffer requires native blit support; "
      "host-emulated map/sync/memcpy transfers are not available on device "
      "queues");
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  if (source_ref.length == 0 && target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)source_ref;
  (void)target_ref;
  (void)flags;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna command_buffer_copy_buffer requires native blit support; "
      "host-emulated map/sync/memcpy transfers are not available on device "
      "queues");
}

// ===========================================================================
// ERT_CMD_CHAIN support.
//
// Command buffers are lowered by command-stream shape: one recorded child uses
// the direct single-dispatch native command path, while two or more children,
// or artifacts with reconfiguration/multiple control codes, are batched into an
// ERT_CMD_CHAIN submitted with one issue/wait. Each chain slot is submitted as
// ERT_START_NPU (PARTIAL_ELF) with arg[0]=AIE2_EXEC_BUFFER_KERNEL_OP_TXN so the
// firmware runs the same XAie TXN control code as the direct path; the I/O
// addresses that the CU path lets the firmware patch are instead host-patched
// into the control-code BD registers here.
// ===========================================================================
// TXN-interpreter selector: tells the firmware to interpret the instruction
// buffer as an XAie transaction (same value the default ERT_START_CU path
// passes as its opcode arg).
static const uint64_t kAie2ExecBufferKernelOpTxn = 3;

iree_status_t iree_hal_amdxdna_make_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t* txn,
    const iree_hal_amdxdna_u32_list_t* patches, const uint64_t* args,
    iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count,
    iree_const_byte_span_t constants,
    const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patches,
    bool use_native_partial_elf, bool retain_signature,
    iree_hal_amdxdna_chain_cmd_t* out_cmd) {
  size_t bytes = txn->count * sizeof(uint32_t);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_c_alloc_buffer(
      command_buffer->device->native_device, bytes,
      IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &out_cmd->ctrl_code));

  void* mapped_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_buffer_c_map(out_cmd->ctrl_code, &mapped_ptr));
  out_cmd->ctrl_code_mapped_ptr = mapped_ptr;
  uint32_t* dst = (uint32_t*)mapped_ptr;
  memcpy(dst, txn->data, bytes);

  if (constant_patches) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_patch_write32_constants_with_list(
        dst, txn->count, constant_patches, constants));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_patch_write32_constants(dst, txn->count, constants));
  }

  if (!iree_hal_amdxdna_apply_patch_table(dst, txn->count, patches->data,
                                          patches->count, args, arg_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna cmd-chain: invalid host patch table for control code");
  }

  if (retain_signature) {
    iree_status_t status = iree_hal_amdxdna_chain_cmd_set_signature(
        command_buffer->host_allocator, out_cmd, dst, txn->count, arg_buffers,
        args, arg_offsets, arg_lengths, arg_count);
    IREE_RETURN_IF_ERROR(status);

    // The signature setter copies and then frees any prior command-owned
    // arrays; the input pointers may alias those arrays when a deferred
    // descriptor is materialized in place. Use the copied command-owned arrays
    // after this point.
    arg_buffers = out_cmd->binding_buffers;
    args = out_cmd->binding_device_addrs;
    arg_offsets = out_cmd->binding_offsets;
    arg_lengths = out_cmd->binding_lengths;
  }
  if (!command_buffer->device->native_caps
           .native_owns_control_code_publication) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_sync_all(
        out_cmd->ctrl_code,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
  }

  const iree_hal_amdxdna_native_c_command_opcode_t command_opcode =
      use_native_partial_elf
          ? IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF
          : IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_create(
      command_buffer->device->native_device, command_opcode,
      &out_cmd->command));

  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_command_c_set_cu_index(out_cmd->command, cu_idx));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_control_buffer(
      out_cmd->command, out_cmd->ctrl_code, bytes));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_arg_64(
      out_cmd->command, kAie2ExecBufferKernelOpTxn));
  const bool native_uses_dpu_regmap_args =
      !use_native_partial_elf &&
      command_buffer->device->native_caps.default_dispatch_opcode ==
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  if (command_opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF) {
    if (IREE_UNLIKELY(arg_count &&
                      (!arg_buffers || !arg_offsets || !arg_lengths))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna PARTIAL_ELF cmd-chain child is missing BO "
          "bindings for its runtime args");
    }
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          out_cmd->command, /*position=*/i + 1, arg_buffers[i], arg_offsets[i],
          arg_lengths[i]));
    }
  } else if (native_uses_dpu_regmap_args) {
    // Some native drivers expose DPU kernels through an xclbin XML register
    // map. In that path the runtime data VAs are regular ERT args, but native
    // completion still needs the backing BO ranges to refresh host-visible
    // outputs.
    if (IREE_UNLIKELY(arg_count &&
                      (!arg_buffers || !arg_offsets || !arg_lengths))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "amdxdna START_NPU DPU regmap command is missing BO bindings for "
          "its runtime args");
    }
    for (size_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
              out_cmd->command, arg_buffers[i], arg_offsets[i]));
    }
  }
  out_cmd->built = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group,
    const iree_hal_amdxdna_u32_list_t* control_code,
    const iree_hal_amdxdna_u32_list_t* patch_table, const uint64_t* args,
    const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patches,
    iree_hal_amdxdna_native_buffer_t* const* arg_buffers,
    const iree_device_size_t* arg_offsets,
    const iree_device_size_t* arg_lengths, size_t arg_count,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_const_byte_span_t constants, bool use_native_partial_elf,
    bool defer_build, uint64_t executable_identity, uint32_t entry_point,
    uint32_t run_ordinal) {
  iree_hal_amdxdna_chain_cmd_t cmd;
  iree_hal_amdxdna_chain_cmd_initialize(&cmd);
  iree_status_t status = iree_hal_amdxdna_chain_cmd_set_deferred_descriptor(
      command_buffer->host_allocator, &cmd, control_code, patch_table,
      constant_patches, cu_idx, constants, use_native_partial_elf, arg_buffers,
      args, arg_offsets, arg_lengths, arg_count);
  cmd.src_executable_identity = executable_identity;
  cmd.src_entry_point = entry_point;
  cmd.src_run_ordinal = run_ordinal;
  if (iree_status_is_ok(status) && !defer_build) {
    status = iree_hal_amdxdna_make_npu_cmd(
        command_buffer, cu_idx, control_code, patch_table, args, arg_buffers,
        arg_offsets, arg_lengths, arg_count, constants, constant_patches,
        use_native_partial_elf, /*retain_signature=*/true, &cmd);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_chain_group_append_cmd_move(
        command_buffer->host_allocator, group, &cmd);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_chain_cmd_deinitialize(command_buffer->host_allocator,
                                            &cmd);
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_rewrite_cached_start_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_cmd_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh) {
  IREE_ASSERT_ARGUMENT(cached);
  IREE_ASSERT_ARGUMENT(fresh);
  if (IREE_UNLIKELY(!cached->built || !cached->ctrl_code || !cached->command ||
                    !fresh->src_asm_inst || !fresh->src_patches ||
                    fresh->src_use_native_partial_elf)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU chain cache rewrite requires a built non-partial "
        "cached child and a deferred fresh descriptor");
  }

  const iree_hal_amdxdna_u32_list_t* txn = fresh->src_asm_inst;
  const iree_hal_amdxdna_u32_list_t* patches = fresh->src_patches;
  const size_t bytes = txn->count * sizeof(uint32_t);
  if (IREE_UNLIKELY(bytes >
                    iree_hal_amdxdna_native_buffer_c_size(cached->ctrl_code))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU cached control buffer is too small");
  }

  void* mapped_ptr = cached->ctrl_code_mapped_ptr;
  if (!mapped_ptr) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_buffer_c_map(cached->ctrl_code, &mapped_ptr));
    cached->ctrl_code_mapped_ptr = mapped_ptr;
  }
  uint32_t* dst = (uint32_t*)mapped_ptr;
  memcpy(dst, txn->data, bytes);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      dst, txn->data, txn->count, fresh->src_constant_patches,
      iree_make_const_byte_span(fresh->src_constants,
                                fresh->src_constant_count),
      patches->data, patches->count, fresh->binding_device_addrs,
      fresh->binding_count));

  if (IREE_UNLIKELY(cached->ctrl_word_count != txn->count ||
                    cached->binding_count != fresh->binding_count)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna START_NPU cached command signature shape "
                            "changed unexpectedly");
  }
  memcpy(cached->ctrl_words, dst, txn->count * sizeof(*cached->ctrl_words));
  memcpy(cached->binding_buffers, fresh->binding_buffers,
         fresh->binding_count * sizeof(*cached->binding_buffers));
  memcpy(cached->binding_device_addrs, fresh->binding_device_addrs,
         fresh->binding_count * sizeof(*cached->binding_device_addrs));
  memcpy(cached->binding_offsets, fresh->binding_offsets,
         fresh->binding_count * sizeof(*cached->binding_offsets));
  memcpy(cached->binding_lengths, fresh->binding_lengths,
         fresh->binding_count * sizeof(*cached->binding_lengths));

  if (!command_buffer->device->native_caps
           .native_owns_control_code_publication) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_sync_all(
        cached->ctrl_code, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
  }

  const bool native_uses_dpu_regmap_args =
      command_buffer->device->native_caps.default_dispatch_opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  if (native_uses_dpu_regmap_args) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(cached->command));
    iree_status_t update_status = iree_ok_status();
    for (iree_host_size_t i = 0; i < fresh->binding_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          cached->command, /*position=*/i + 1, fresh->binding_buffers[i],
          fresh->binding_offsets[i], fresh->binding_lengths[i]));
      update_status = iree_hal_amdxdna_native_command_c_update_arg_64(
          cached->command, i + 1, fresh->binding_device_addrs[i]);
      if (!iree_status_is_ok(update_status)) break;
    }
    if (!iree_status_is_ok(update_status)) {
      const iree_status_code_t update_status_code =
          iree_status_code(update_status);
      iree_status_ignore(update_status);
      if (update_status_code != IREE_STATUS_UNIMPLEMENTED) {
        return iree_make_status(
            update_status_code,
            "amdxdna cached START_NPU command arg update failed");
      }
      IREE_RETURN_IF_ERROR(
          iree_hal_amdxdna_native_command_c_reset(cached->command));
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_set_cu_index(
          cached->command, fresh->src_cu_idx));
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_control_buffer(
          cached->command, cached->ctrl_code, bytes));
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_arg_64(
          cached->command, kAie2ExecBufferKernelOpTxn));
      for (iree_host_size_t i = 0; i < fresh->binding_count; ++i) {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
                cached->command, fresh->binding_buffers[i],
                fresh->binding_offsets[i]));
      }
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_command_c_mark_code_dirty(cached->command));
  cached->native_bindings_current = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_rewrite_cached_single_start_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_single_command_cache_entry_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh) {
  IREE_ASSERT_ARGUMENT(cached);
  IREE_ASSERT_ARGUMENT(fresh);
  if (IREE_UNLIKELY(!cached->ctrl_code_buffer || !cached->command ||
                    !fresh->src_asm_inst || !fresh->src_patches ||
                    fresh->src_use_native_partial_elf)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU single cache rewrite requires a non-partial "
        "deferred descriptor and cached native command");
  }
  if (IREE_UNLIKELY((command_buffer->device->native_caps.dispatch_models &
                     IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) ==
                    0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU single cache rewrite requires START_NPU native "
        "dispatch");
  }

  const iree_hal_amdxdna_u32_list_t* txn = fresh->src_asm_inst;
  const iree_hal_amdxdna_u32_list_t* patches = fresh->src_patches;
  const size_t bytes = txn->count * sizeof(uint32_t);
  if (IREE_UNLIKELY(bytes > iree_hal_amdxdna_native_buffer_c_size(
                                cached->ctrl_code_buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU cached single control buffer is too small");
  }
  if (IREE_UNLIKELY(cached->ctrl_word_count != txn->count ||
                    cached->binding_count != fresh->binding_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna START_NPU cached single command signature shape changed "
        "unexpectedly");
  }
  void* mapped_ptr = cached->ctrl_code_mapped_ptr;
  if (!mapped_ptr) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_map(
        cached->ctrl_code_buffer, &mapped_ptr));
    cached->ctrl_code_mapped_ptr = mapped_ptr;
  }
  uint32_t* dst = (uint32_t*)mapped_ptr;
  memcpy(dst, txn->data, bytes);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_patch_dynamic_fields_from_template(
      dst, txn->data, txn->count, fresh->src_constant_patches,
      iree_make_const_byte_span(fresh->src_constants,
                                fresh->src_constant_count),
      patches->data, patches->count, fresh->binding_device_addrs,
      fresh->binding_count));

  memcpy(cached->ctrl_words, dst, txn->count * sizeof(*cached->ctrl_words));
  memcpy(cached->binding_buffers, fresh->binding_buffers,
         fresh->binding_count * sizeof(*cached->binding_buffers));
  memcpy(cached->binding_device_addrs, fresh->binding_device_addrs,
         fresh->binding_count * sizeof(*cached->binding_device_addrs));
  memcpy(cached->binding_offsets, fresh->binding_offsets,
         fresh->binding_count * sizeof(*cached->binding_offsets));
  memcpy(cached->binding_lengths, fresh->binding_lengths,
         fresh->binding_count * sizeof(*cached->binding_lengths));

  if (!command_buffer->device->native_caps
           .native_owns_control_code_publication) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_sync_all(
        cached->ctrl_code_buffer,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE));
  }

  const bool native_uses_dpu_regmap_args =
      command_buffer->device->native_caps.default_dispatch_opcode ==
      IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU;
  bool rebuild_command = !native_uses_dpu_regmap_args;
  if (native_uses_dpu_regmap_args) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(cached->command));
    for (iree_host_size_t i = 0; i < fresh->binding_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_bind_buffer(
          cached->command, /*position=*/i + 1, fresh->binding_buffers[i],
          fresh->binding_offsets[i], fresh->binding_lengths[i]));
      iree_status_t update_status =
          iree_hal_amdxdna_native_command_c_update_arg_64(
              cached->command, i + 1, fresh->binding_device_addrs[i]);
      if (!iree_status_is_ok(update_status)) {
        const iree_status_code_t update_status_code =
            iree_status_code(update_status);
        iree_status_ignore(update_status);
        if (update_status_code != IREE_STATUS_UNIMPLEMENTED) {
          return iree_make_status(
              update_status_code,
              "amdxdna cached START_NPU single command arg update failed");
        }
        rebuild_command = true;
        break;
      }
    }
  }
  if (rebuild_command) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_command_c_reset(cached->command));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_set_cu_index(
        cached->command, fresh->src_cu_idx));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_control_buffer(
        cached->command, cached->ctrl_code_buffer, bytes));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_add_arg_64(
        cached->command, kAie2ExecBufferKernelOpTxn));
    if (native_uses_dpu_regmap_args) {
      for (iree_host_size_t i = 0; i < fresh->binding_count; ++i) {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
                cached->command, fresh->binding_buffers[i],
                fresh->binding_offsets[i]));
      }
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_native_command_c_mark_code_dirty(cached->command));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_rewrite_cached_single_partial_elf_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_device_single_command_cache_t* cache,
    iree_hal_amdxdna_single_command_cache_entry_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh) {
  IREE_ASSERT_ARGUMENT(cached);
  IREE_ASSERT_ARGUMENT(fresh);
  if (IREE_UNLIKELY(!cached->ctrl_code_buffer || !cached->command ||
                    !fresh->src_asm_inst || !fresh->src_patches ||
                    !fresh->src_use_native_partial_elf ||
                    cached->ctrl_word_count != fresh->src_asm_inst->count ||
                    cached->binding_count != fresh->binding_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna PARTIAL_ELF single cache rewrite has incompatible shape");
  }

  void* mapped_ptr = cached->ctrl_code_mapped_ptr;
  if (!mapped_ptr) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_c_map(
        cached->ctrl_code_buffer, &mapped_ptr));
    cached->ctrl_code_mapped_ptr = mapped_ptr;
  }
  const size_t control_bytes =
      fresh->src_asm_inst->count * sizeof(uint32_t);
  uint32_t* prepared_words = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(command_buffer->host_allocator,
                                              control_bytes,
                                              (void**)&prepared_words));
  memcpy(prepared_words, fresh->src_asm_inst->data, control_bytes);
  iree_status_t status = iree_hal_amdxdna_patch_dynamic_fields_from_template(
      prepared_words, fresh->src_asm_inst->data, fresh->src_asm_inst->count,
      fresh->src_constant_patches,
      iree_make_const_byte_span(fresh->src_constants,
                                fresh->src_constant_count),
      fresh->src_patches->data, fresh->src_patches->count,
      fresh->binding_device_addrs, fresh->binding_count);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(command_buffer->host_allocator, prepared_words);
    return status;
  }

  const bool control_changed =
      memcmp(cached->ctrl_words, prepared_words, control_bytes) != 0;
  const bool bindings_changed =
      fresh->binding_count != 0 &&
      (memcmp(cached->binding_buffers, fresh->binding_buffers,
              fresh->binding_count * sizeof(*cached->binding_buffers)) != 0 ||
       memcmp(cached->binding_offsets, fresh->binding_offsets,
              fresh->binding_count * sizeof(*cached->binding_offsets)) != 0 ||
       memcmp(cached->binding_lengths, fresh->binding_lengths,
              fresh->binding_count * sizeof(*cached->binding_lengths)) != 0);

  if (bindings_changed) {
    status =
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(cached->command);
    for (iree_host_size_t i = 0;
         i < fresh->binding_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amdxdna_native_command_c_bind_buffer(
          cached->command, i + 1, fresh->binding_buffers[i],
          fresh->binding_offsets[i], fresh->binding_lengths[i]);
    }
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_single_command_cache_entry_discard(cache, cached);
      iree_allocator_free(command_buffer->host_allocator, prepared_words);
      return status;
    }
  }
  if (control_changed) {
    memcpy(mapped_ptr, prepared_words, control_bytes);
    status =
        iree_hal_amdxdna_native_command_c_mark_code_dirty(cached->command);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_single_command_cache_entry_discard(cache, cached);
      iree_allocator_free(command_buffer->host_allocator, prepared_words);
      return status;
    }
    memcpy(cached->ctrl_words, prepared_words, control_bytes);
  }
  if (fresh->binding_count != 0) {
    memcpy(cached->binding_buffers, fresh->binding_buffers,
           fresh->binding_count * sizeof(*cached->binding_buffers));
    memcpy(cached->binding_device_addrs, fresh->binding_device_addrs,
           fresh->binding_count * sizeof(*cached->binding_device_addrs));
    memcpy(cached->binding_offsets, fresh->binding_offsets,
           fresh->binding_count * sizeof(*cached->binding_offsets));
    memcpy(cached->binding_lengths, fresh->binding_lengths,
           fresh->binding_count * sizeof(*cached->binding_lengths));
  }
  iree_allocator_free(command_buffer->host_allocator, prepared_words);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_rewrite_cached_chain_partial_elf_cmd(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_chain_cmd_t* cached,
    const iree_hal_amdxdna_chain_cmd_t* fresh, bool* out_code_changed,
    bool* out_bindings_changed) {
  IREE_ASSERT_ARGUMENT(cached);
  IREE_ASSERT_ARGUMENT(fresh);
  IREE_ASSERT_ARGUMENT(out_code_changed);
  IREE_ASSERT_ARGUMENT(out_bindings_changed);
  *out_code_changed = false;
  *out_bindings_changed = false;
  if (IREE_UNLIKELY(!cached->built || !cached->ctrl_code || !cached->command ||
                    !fresh->src_asm_inst || !fresh->src_patches ||
                    !fresh->src_use_native_partial_elf ||
                    cached->ctrl_word_count != fresh->src_asm_inst->count ||
                    cached->binding_count != fresh->binding_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna PARTIAL_ELF cached chain rewrite has incompatible shape");
  }

  void* mapped_ptr = cached->ctrl_code_mapped_ptr;
  if (!mapped_ptr) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdxdna_native_buffer_c_map(cached->ctrl_code, &mapped_ptr));
    cached->ctrl_code_mapped_ptr = mapped_ptr;
  }
  const bool code_changed =
      (fresh->src_constant_count != 0 &&
       memcmp(cached->src_constants, fresh->src_constants,
              fresh->src_constant_count) != 0) ||
      (fresh->binding_count != 0 &&
       memcmp(cached->binding_device_addrs, fresh->binding_device_addrs,
              fresh->binding_count * sizeof(*cached->binding_device_addrs)) !=
           0);
  const bool bindings_changed =
      fresh->binding_count != 0 &&
      (memcmp(cached->binding_buffers, fresh->binding_buffers,
              fresh->binding_count * sizeof(*cached->binding_buffers)) != 0 ||
       memcmp(cached->binding_offsets, fresh->binding_offsets,
              fresh->binding_count * sizeof(*cached->binding_offsets)) != 0 ||
       memcmp(cached->binding_lengths, fresh->binding_lengths,
              fresh->binding_count * sizeof(*cached->binding_lengths)) != 0);

  uint32_t* prepared_words = NULL;
  size_t control_bytes = 0;
  if (code_changed) {
    const iree_const_byte_span_t constants = iree_make_const_byte_span(
        fresh->src_constants, fresh->src_constant_count);
    control_bytes = fresh->src_asm_inst->count * sizeof(uint32_t);
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, control_bytes,
                                               (void**)&prepared_words));
    memcpy(prepared_words, fresh->src_asm_inst->data, control_bytes);
    iree_status_t status = iree_hal_amdxdna_patch_dynamic_fields_from_template(
        prepared_words, fresh->src_asm_inst->data, fresh->src_asm_inst->count,
        fresh->src_constant_patches, constants, fresh->src_patches->data,
        fresh->src_patches->count, fresh->binding_device_addrs,
        fresh->binding_count);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, prepared_words);
      return status;
    }
  }

  if (bindings_changed) {
    iree_status_t status =
        iree_hal_amdxdna_native_command_c_reset_bound_buffers(cached->command);
    for (iree_host_size_t i = 0;
         i < fresh->binding_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amdxdna_native_command_c_bind_buffer(
          cached->command, i + 1, fresh->binding_buffers[i],
          fresh->binding_offsets[i], fresh->binding_lengths[i]);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, prepared_words);
      return status;
    }
    cached->native_bindings_current = true;
  }
  if (code_changed) {
    memcpy(mapped_ptr, prepared_words, control_bytes);
    memcpy(cached->ctrl_words, prepared_words, control_bytes);
    iree_status_t status =
        iree_hal_amdxdna_native_command_c_mark_code_dirty(cached->command);
    iree_allocator_free(host_allocator, prepared_words);
    IREE_RETURN_IF_ERROR(status);
  }
  if (fresh->src_constant_count != 0) {
    memcpy(cached->src_constants, fresh->src_constants,
           fresh->src_constant_count);
  }
  if (fresh->binding_count != 0) {
    memcpy(cached->binding_buffers, fresh->binding_buffers,
           fresh->binding_count * sizeof(*cached->binding_buffers));
    memcpy(cached->binding_device_addrs, fresh->binding_device_addrs,
           fresh->binding_count * sizeof(*cached->binding_device_addrs));
    memcpy(cached->binding_offsets, fresh->binding_offsets,
           fresh->binding_count * sizeof(*cached->binding_offsets));
    memcpy(cached->binding_lengths, fresh->binding_lengths,
           fresh->binding_count * sizeof(*cached->binding_lengths));
  }
  *out_code_changed = code_changed;
  *out_bindings_changed = bindings_changed;
  return iree_ok_status();
}

// Accumulate one dispatch's reconfig+exec sub-commands into the direct command
// buffer's chain accumulator. Does NOT submit; flush_chains() chooses direct
// single-dispatch submit for one child and ERT_CMD_CHAIN for multi-child groups
// at end(). Dispatches that share a hw queue (e.g. all entry points of one
// control-packet executable, or separate executables resolved to the same
// shared context) accumulate into one group.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_context_ref_t* context_ref,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    const iree_hal_amdxdna_dispatch_plan_t* plan,
    iree_const_byte_span_t constants, bool use_native_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The chained path host-patches I/O addresses using the compiler-emitted
  // patch table (parallel to asm_inst_runlist). Require it: an executable
  // compiled before the patch table existed cannot use cmd-chain.
  iree_status_t status = iree_ok_status();
  uint64_t* binding_addrs = NULL;
  iree_hal_amdxdna_native_buffer_t** binding_buffers = NULL;
  iree_device_size_t* binding_offsets = NULL;
  iree_device_size_t* binding_lengths = NULL;

  if (plan->patch_table_count != plan->control_code_count) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cmd-chain requires a host patch table in the executable "
        "(have %zu patch lists for %zu control codes); recompile with a "
        "patch-table-aware compiler",
        plan->patch_table_count, plan->control_code_count);
  }

  // Binding device addresses (exec args). For control packets the reconfig arg
  // is the per-reconfiguration data buffer (built below).
  if (iree_status_is_ok(status) && bindings.count != 0) {
    status = iree_allocator_malloc_array(command_buffer->host_allocator,
                                         bindings.count, sizeof(*binding_addrs),
                                         (void**)&binding_addrs);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, bindings.count,
          sizeof(*binding_buffers), (void**)&binding_buffers);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, bindings.count,
          sizeof(*binding_offsets), (void**)&binding_offsets);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, bindings.count,
          sizeof(*binding_lengths), (void**)&binding_lengths);
    }
  }
  for (iree_host_size_t j = 0; iree_status_is_ok(status) && j < bindings.count;
       ++j) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    status = iree_hal_amdxdna_native_buffer_c_ensure_allocated(native_buffer);
    if (iree_status_is_ok(status)) {
      // Match the normal ERT_START_CU path: a binding may reference a subspan
      // of its allocated root BO, so host-patched DDR addresses must include
      // both offsets in addition to the BO base address.
      binding_buffers[j] = native_buffer;
      binding_offsets[j] =
          iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
          bindings.values[j].offset;
      binding_lengths[j] = bindings.values[j].length;
      binding_addrs[j] =
          iree_hal_amdxdna_native_buffer_c_device_address(native_buffer) +
          binding_offsets[j];
    }
  }

  // Append to the current group, opening a new one when the native queue
  // changes (a chain runs on a single native context/queue).
  iree_hal_amdxdna_chain_accum_t* accum = &command_buffer->chain_accum;
  iree_hal_amdxdna_chain_group_t* group = NULL;
  if (accum->group_count != 0) {
    group = &accum->groups[accum->group_count - 1];
  }
  if (iree_status_is_ok(status) &&
      (!group || group->queue != queue ||
       group->native_partial_elf != use_native_partial_elf)) {
    status = iree_hal_amdxdna_chain_accum_append_group(
        command_buffer->host_allocator, accum, &group);
    if (iree_status_is_ok(status)) {
      group->context = iree_hal_amdxdna_native_context_ref_retain(context_ref);
      group->queue = queue;
      group->native_partial_elf = use_native_partial_elf;
    }
  }

  // Defer the per-child native build for cacheable no-reconfiguration chains.
  // Those children are recorded as lightweight descriptors here and built or
  // rewritten lazily in flush. This covers both partial-ELF packets and Linux
  // START_NPU packets: START_NPU needs an explicit rewrite of scalar arg64
  // packet words before a cached parent chain can be reused safely.
  const bool defer_build = plan->data_payload_count == 0;

  // Emit exactly one kernel slot for this HAL dispatch. Reconfiguration control
  // packets retain their repeat count for existing multi-PDI artifacts.
  if (iree_status_is_ok(status) && plan->data_payload_count == 0) {
    status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
        command_buffer, group, &plan->control_codes[0], &plan->patch_tables[0],
        binding_addrs,
        plan->constant_patch_table_count > 0 ? &plan->constant_patch_tables[0]
                                             : NULL,
        binding_buffers, binding_offsets, binding_lengths, bindings.count,
        cu_idx, constants, use_native_partial_elf, defer_build,
        plan->executable->cache_identity, plan->entry_point,
        /*run_ordinal=*/0);
  } else if (iree_status_is_ok(status)) {
    for (size_t i = 0;
         iree_status_is_ok(status) && i < plan->data_payload_count; i++) {
      // Control-packet data buffer for this reconfiguration (reconfig arg[0]).
      const iree_hal_amdxdna_u32_list_t* seq = &plan->data_payloads[i];
      size_t seq_bytes = seq->count * sizeof(uint32_t);
      iree_hal_amdxdna_native_buffer_t* reconf_buffer = NULL;
      status = iree_hal_amdxdna_native_device_c_alloc_buffer(
          command_buffer->device->native_device, seq_bytes,
          IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY, &reconf_buffer);
      void* seq_buffer_ptr = NULL;
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_buffer_c_map(reconf_buffer,
                                                      &seq_buffer_ptr);
      }
      if (iree_status_is_ok(status)) {
        memcpy(seq_buffer_ptr, seq->data, seq_bytes);
        status = iree_hal_amdxdna_native_buffer_c_sync_all(
            reconf_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_chain_group_append_reconf_buffer(
            command_buffer->host_allocator, group, reconf_buffer);
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_native_buffer_c_destroy(reconf_buffer);
        break;
      }
      const uint64_t reconf_arg =
          iree_hal_amdxdna_native_buffer_c_device_address(reconf_buffer);
      const iree_device_size_t reconf_offset = 0;
      const iree_device_size_t reconf_length = (iree_device_size_t)seq_bytes;
      for (uint32_t r = 0;
           iree_status_is_ok(status) && r < plan->data_payload_run_count; r++) {
        status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
            command_buffer, group, &plan->control_codes[2 * i],
            &plan->patch_tables[2 * i], &reconf_arg,
            plan->constant_patch_table_count > 2 * i
                ? &plan->constant_patch_tables[2 * i]
                : NULL,
            &reconf_buffer, &reconf_offset, &reconf_length,
            /*arg_count=*/1, cu_idx, constants, use_native_partial_elf,
            defer_build, plan->executable->cache_identity, plan->entry_point,
            /*run_ordinal=*/2 * i);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_direct_command_buffer_emit_chain_cmd(
            command_buffer, group, &plan->control_codes[2 * i + 1],
            &plan->patch_tables[2 * i + 1], binding_addrs,
            plan->constant_patch_table_count > 2 * i + 1
                ? &plan->constant_patch_tables[2 * i + 1]
                : NULL,
            binding_buffers, binding_offsets, binding_lengths, bindings.count,
            cu_idx, constants, use_native_partial_elf, defer_build,
            plan->executable->cache_identity, plan->entry_point,
            /*run_ordinal=*/2 * i + 1);
      }
    }
  }

  // Track I/O bindings for residency + final device->host sync at flush.
  // Multiple HAL dispatches in the same command buffer often use the same
  // binding ranges; keep only exact ranges so a 240-dispatch chain does not
  // perform 240 duplicate host invalidations after the parent completes.
  for (iree_host_size_t j = 0; iree_status_is_ok(status) && j < bindings.count;
       ++j) {
    status = iree_hal_amdxdna_chain_group_append_binding_ref_unique(
        command_buffer->host_allocator, group, bindings.values[j]);
  }

  iree_allocator_free(command_buffer->host_allocator, binding_lengths);
  iree_allocator_free(command_buffer->host_allocator, binding_offsets);
  iree_allocator_free(command_buffer->host_allocator, binding_buffers);
  iree_allocator_free(command_buffer->host_allocator, binding_addrs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Build a native ERT_CMD_CHAIN for a contiguous span [begin, end) of a group's
// sub-commands, binding the group's referenced buffers for residency. The
// caller owns submission so native backends can submit all chunks before
// waiting when their DDI supports that shape.
static iree_status_t iree_hal_amdxdna_prepare_chain(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_amdxdna_chain_group_t* group, size_t begin, size_t end,
    iree_hal_amdxdna_native_command_t** out_chain) {
  *out_chain = NULL;
  size_t n = end - begin;
  iree_hal_amdxdna_native_command_t* chain = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_c_create(
      native_device, IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_COMMAND_CHAIN,
      &chain));
  iree_hal_amdxdna_native_command_t** commands = NULL;
  iree_status_t status = iree_ok_status();
  if (n != 0) {
    status = iree_allocator_malloc_array(host_allocator, n, sizeof(*commands),
                                         (void**)&commands);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }
  for (size_t i = begin; i < end; ++i) {
    commands[i - begin] = group->cmds[i].command;
  }
  status = iree_hal_amdxdna_native_command_c_prepare_chain(chain, commands, n);
  iree_allocator_free(host_allocator, commands);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }

  const bool child_bo_table_parent_chain = group->native_partial_elf;
  if (child_bo_table_parent_chain) {
    *out_chain = chain;
    return iree_ok_status();
  }

  // Native address-list chain paths register every BO the firmware dereferences
  // (control code + control-packet data + I/O bindings) as arg BOs on the
  // submitted chain so the driver keeps them resident; the sub-command slots
  // reference them only by address. Module-style partial-ELF chains
  // intentionally skip this superset above: the parent binds child exec BOs
  // only, while each child command BO carries its own BO table.
  const size_t arg_bo_ceiling =
      iree_hal_amdxdna_native_command_c_arg_binding_capacity();
  size_t arg_total = n + group->reconf_buffer_count + group->binding_ref_count;
  if (arg_total > arg_bo_ceiling) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna cmd-chain: %zu arg BOs exceeds native ceiling %zu (chunk "
        "groups or reduce binding count)",
        arg_total, arg_bo_ceiling);
  }
  size_t arg_pos = 0;
  for (size_t i = begin; i < end; i++) {
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, group->cmds[i].ctrl_code, 0,
        iree_hal_amdxdna_native_buffer_c_size(group->cmds[i].ctrl_code));
    if (!iree_status_is_ok(status)) break;
  }
  for (iree_host_size_t i = 0; i < group->reconf_buffer_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_amdxdna_native_buffer_t* seq_buffer = group->reconf_buffers[i];
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, seq_buffer, 0,
        iree_hal_amdxdna_native_buffer_c_size(seq_buffer));
  }
  for (iree_host_size_t i = 0; i < group->binding_ref_count; ++i) {
    if (!iree_status_is_ok(status)) break;
    const iree_hal_buffer_ref_t* binding_ref = &group->binding_refs[i];
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(binding_ref->buffer));
    status = iree_hal_amdxdna_native_command_c_bind_buffer(
        chain, arg_pos++, native_buffer, 0,
        iree_hal_amdxdna_native_buffer_c_size(native_buffer));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_native_command_c_destroy(chain);
    return status;
  }
  *out_chain = chain;
  return iree_ok_status();
}

static bool iree_hal_amdxdna_chain_group_requires_parent_chain(
    const iree_hal_amdxdna_chain_group_t* group) {
  return group->cmd_count > 1 || group->reconf_buffer_count != 0;
}

static iree_status_t iree_hal_amdxdna_rebuild_cached_parent_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_command_cache_entry_t* chain_cache,
    uint32_t max_slots) {
  iree_hal_amdxdna_chain_command_cache_entry_clear_chains(
      command_buffer->host_allocator, chain_cache);
  for (size_t begin = 0; begin < chain_cache->group.cmd_count;
       begin += max_slots) {
    size_t end = begin + max_slots;
    if (end > chain_cache->group.cmd_count) end = chain_cache->group.cmd_count;
    iree_hal_amdxdna_native_command_t* chain = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_prepare_chain(
        command_buffer->host_allocator, command_buffer->device->native_device,
        &chain_cache->group, begin, end, &chain));
    iree_status_t append_status =
        iree_hal_amdxdna_chain_command_cache_entry_append_chain(
            command_buffer->host_allocator, chain_cache, chain);
    if (!iree_status_is_ok(append_status)) {
      iree_hal_amdxdna_native_command_c_destroy(chain);
      return append_status;
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_submit_uncached_parent_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group, uint32_t max_slots) {
  iree_status_t status = iree_ok_status();
  iree_hal_amdxdna_native_command_t** chains = NULL;
  uint8_t* completion_owns_chain = NULL;
  size_t chain_count = 0;
  size_t chain_capacity =
      (group->cmd_count + (size_t)max_slots - 1) / (size_t)max_slots;
  if (chain_capacity != 0) {
    status = iree_allocator_malloc_array(command_buffer->host_allocator,
                                         chain_capacity, sizeof(*chains),
                                         (void**)&chains);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, chain_capacity,
          sizeof(*completion_owns_chain), (void**)&completion_owns_chain);
    }
  }
  if (iree_status_is_ok(status) && completion_owns_chain) {
    memset(completion_owns_chain, 0,
           chain_capacity * sizeof(*completion_owns_chain));
  }
  for (size_t begin = 0; begin < group->cmd_count && iree_status_is_ok(status);
       begin += max_slots) {
    size_t end = begin + max_slots;
    if (end > group->cmd_count) end = group->cmd_count;
    iree_hal_amdxdna_native_command_t* chain = NULL;
    status = iree_hal_amdxdna_prepare_chain(
        command_buffer->host_allocator, command_buffer->device->native_device,
        group, begin, end, &chain);
    if (iree_status_is_ok(status)) chains[chain_count++] = chain;
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer)) {
    for (size_t i = 0; i < chain_count && iree_status_is_ok(status); ++i) {
      status = iree_hal_amdxdna_completion_batch_add_cleanup(
          command_buffer->completion_batch,
          iree_hal_amdxdna_completion_destroy_native_command, chains[i]);
      if (iree_status_is_ok(status)) completion_owns_chain[i] = 1;
    }
  }
  if (iree_status_is_ok(status) && chain_count != 0) {
    status = iree_hal_amdxdna_direct_command_buffer_submit_all(
        command_buffer, group->queue, chains, chain_count,
        IREE_SV("ERT_CMD_CHAIN"));
  }
  for (size_t i = 0; i < chain_count; ++i) {
    if (!completion_owns_chain || !completion_owns_chain[i]) {
      iree_hal_amdxdna_native_command_c_destroy(chains[i]);
    }
  }
  iree_allocator_free(command_buffer->host_allocator, completion_owns_chain);
  iree_allocator_free(command_buffer->host_allocator, chains);
  return status;
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (group->cmd_count != 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "single-dispatch flush expected one command, got "
                            "%zu",
                            group->cmd_count);
  }
  iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[0];
  if (!cmd->src_asm_inst || !cmd->src_patches) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "single-dispatch flush is missing recorded control-code descriptors");
  }

  if (!cmd->src_use_native_partial_elf) {
    iree_status_t status = iree_ok_status();
    iree_hal_amdxdna_native_command_t* submit_command = NULL;
    iree_hal_amdxdna_device_single_command_cache_t* single_command_cache = NULL;
    iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = NULL;
    bool single_cache_locked = false;
    bool release_single_cache_entry_after_submit = false;
    const bool can_use_start_npu_template_cache =
        (command_buffer->device->native_caps.dispatch_models &
         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;

    if (can_use_start_npu_template_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to allocate amdxdna single command cache");
      }
    }
    if (iree_status_is_ok(status) && single_command_cache) {
      iree_slim_mutex_lock(&single_command_cache->mutex);
      single_cache_locked = true;
      status =
          iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
              single_command_cache, group->queue, cmd->src_cu_idx.index,
              cmd->src_asm_inst, cmd->src_patches, cmd->src_constant_count,
              cmd->src_use_native_partial_elf, cmd->binding_count,
              &single_cache_entry);
    }

    if (iree_status_is_ok(status) && single_cache_entry) {
      status = iree_hal_amdxdna_rewrite_cached_single_start_npu_cmd(
          command_buffer, single_cache_entry, cmd);
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_single_command_cache_entry_discard(
            single_command_cache, single_cache_entry);
        single_cache_entry = NULL;
      }
      if (iree_status_is_ok(status)) {
        iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
            single_cache_entry);
        if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
                command_buffer)) {
          status =
              iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
                  command_buffer, single_command_cache, single_cache_entry);
          if (!iree_status_is_ok(status)) {
            iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
                single_command_cache, single_cache_entry);
          }
        } else {
          release_single_cache_entry_after_submit = true;
        }
      }
      if (single_cache_entry) submit_command = single_cache_entry->command;
    } else if (iree_status_is_ok(status)) {
      if (!cmd->built) {
        status = iree_hal_amdxdna_make_npu_cmd(
            command_buffer, cmd->src_cu_idx, cmd->src_asm_inst,
            cmd->src_patches, cmd->binding_device_addrs, cmd->binding_buffers,
            cmd->binding_offsets, cmd->binding_lengths, cmd->binding_count,
            iree_make_const_byte_span(cmd->src_constants,
                                      cmd->src_constant_count),
            cmd->src_constant_patches, cmd->src_use_native_partial_elf,
            /*retain_signature=*/true, cmd);
      }
      submit_command = cmd->command;
      if (iree_status_is_ok(status) && single_command_cache) {
        single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
            single_command_cache, group->queue, cmd->src_cu_idx.index,
            cmd->ctrl_words, cmd->ctrl_word_count, cmd->binding_buffers,
            cmd->binding_device_addrs, cmd->binding_offsets,
            cmd->binding_lengths, cmd->binding_count, cmd->ctrl_code,
            cmd->command);
        if (single_cache_entry) {
          iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
              single_command_cache, single_cache_entry, cmd->src_asm_inst,
              cmd->src_patches, cmd->src_constant_count,
              cmd->src_use_native_partial_elf, cmd->ctrl_code_mapped_ptr);
          cmd->ctrl_code = NULL;
          cmd->ctrl_code_mapped_ptr = NULL;
          cmd->command = NULL;
          submit_command = single_cache_entry->command;
          iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
              single_cache_entry);
          if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
                  command_buffer)) {
            status =
                iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
                    command_buffer, single_command_cache, single_cache_entry);
            if (!iree_status_is_ok(status)) {
              iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
                  single_command_cache, single_cache_entry);
            }
          } else {
            release_single_cache_entry_after_submit = true;
          }
        }
      }
    }
    if (single_cache_locked) {
      iree_slim_mutex_unlock(&single_command_cache->mutex);
      single_cache_locked = false;
    }

    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_direct_command_buffer_submit(
          command_buffer, group->queue, submit_command, IREE_SV("dispatch"));
    }
    if (release_single_cache_entry_after_submit) {
      iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
          single_command_cache, single_cache_entry);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  uint32_t* prepared_ctrl_words = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;
  iree_hal_amdxdna_native_command_t* submit_command = NULL;
  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache = NULL;
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = NULL;
  bool single_cache_locked = false;
  iree_status_t status = iree_ok_status();

  single_command_cache =
      iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
  if (!single_command_cache) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "failed to allocate amdxdna single command cache");
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&single_command_cache->mutex);
    single_cache_locked = true;
    status =
        iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
            single_command_cache, group->queue, cmd->src_cu_idx.index,
            cmd->src_asm_inst, cmd->src_patches, cmd->src_constant_count,
            cmd->src_use_native_partial_elf, cmd->binding_count,
            &single_cache_entry);
  }
  if (iree_status_is_ok(status) && single_cache_entry) {
    status = iree_hal_amdxdna_rewrite_cached_single_partial_elf_cmd(
        command_buffer, single_command_cache, single_cache_entry, cmd);
    if (iree_status_is_ok(status) &&
        iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
            command_buffer)) {
      iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
          single_cache_entry);
      status =
          iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
              command_buffer, single_command_cache, single_cache_entry);
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
            single_command_cache, single_cache_entry);
      }
    }
    submit_command = single_cache_entry->command;
  }
  if (single_cache_locked) {
    iree_slim_mutex_unlock(&single_command_cache->mutex);
    single_cache_locked = false;
  }
  if (iree_status_is_ok(status) && submit_command) {
    status = iree_hal_amdxdna_direct_command_buffer_submit(
        command_buffer, group->queue, submit_command, IREE_SV("dispatch"));
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  status = iree_allocator_malloc_array(
      command_buffer->host_allocator, cmd->src_asm_inst->count,
      sizeof(*prepared_ctrl_words), (void**)&prepared_ctrl_words);
  if (iree_status_is_ok(status)) {
    memcpy(prepared_ctrl_words, cmd->src_asm_inst->data,
           cmd->src_asm_inst->count * sizeof(*prepared_ctrl_words));
    status = iree_hal_amdxdna_patch_write32_constants(
        prepared_ctrl_words, cmd->src_asm_inst->count,
        iree_make_const_byte_span(cmd->src_constants, cmd->src_constant_count));
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_amdxdna_apply_patch_table(
          prepared_ctrl_words, cmd->src_asm_inst->count, cmd->src_patches->data,
          cmd->src_patches->count, cmd->binding_device_addrs,
          cmd->binding_count)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna PARTIAL_ELF single dispatch has an invalid host patch table");
  }

  if (iree_status_is_ok(status) && !single_command_cache) {
    status =
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "failed to allocate amdxdna single command cache");
  }
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&single_command_cache->mutex);
    single_cache_locked = true;
    status =
        iree_hal_amdxdna_find_single_command_cache_descriptor_template_entry(
            single_command_cache, group->queue, cmd->src_cu_idx.index,
            cmd->src_asm_inst, cmd->src_patches, cmd->src_constant_count,
            cmd->src_use_native_partial_elf, cmd->binding_count,
            &single_cache_entry);
    if (iree_status_is_ok(status) && single_cache_entry) {
      status = iree_hal_amdxdna_update_single_command_cache_entry(
          single_command_cache, single_cache_entry, prepared_ctrl_words,
          cmd->src_asm_inst->count, cmd->binding_buffers,
          cmd->binding_device_addrs, cmd->binding_offsets, cmd->binding_lengths,
          cmd->binding_count);
    }
    if (iree_status_is_ok(status) && !single_cache_entry) {
      status = iree_hal_amdxdna_find_single_command_cache_entry(
          single_command_cache, group->queue, cmd->src_cu_idx.index,
          prepared_ctrl_words, cmd->src_asm_inst->count, cmd->binding_buffers,
          cmd->binding_device_addrs, cmd->binding_offsets, cmd->binding_lengths,
          cmd->binding_count, &single_cache_entry);
    }
  }

  if (iree_status_is_ok(status) && single_cache_entry) {
    if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
            command_buffer)) {
      iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
          single_cache_entry);
      status =
          iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
              command_buffer, single_command_cache, single_cache_entry);
      if (!iree_status_is_ok(status)) {
        iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
            single_command_cache, single_cache_entry);
      }
    }
    submit_command = single_cache_entry->command;
  } else if (iree_status_is_ok(status)) {
    const size_t ctrl_code_size =
        cmd->src_asm_inst->count * sizeof(*prepared_ctrl_words);
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        command_buffer->device->native_device, ctrl_code_size,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &ctrl_code_buffer);
    void* instr_buffer_ptr = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_buffer_c_map(ctrl_code_buffer,
                                                    &instr_buffer_ptr);
    }
    if (iree_status_is_ok(status)) {
      memcpy(instr_buffer_ptr, prepared_ctrl_words, ctrl_code_size);
      if (!command_buffer->device->native_caps
               .native_owns_control_code_publication) {
        status = iree_hal_amdxdna_native_buffer_c_sync_all(
            ctrl_code_buffer,
            IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
      }
    }

    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_create(
          command_buffer->device->native_device,
          IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF,
          &command);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_set_cu_index(command,
                                                              cmd->src_cu_idx);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_arg_64(
          command, kAie2ExecBufferKernelOpTxn);
    }
    for (size_t i = 0; i < cmd->binding_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_amdxdna_native_command_c_bind_buffer(
          command, /*position=*/i + 1, cmd->binding_buffers[i],
          cmd->binding_offsets[i], cmd->binding_lengths[i]);
    }
    if (iree_status_is_ok(status)) {
      single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
          single_command_cache, group->queue, cmd->src_cu_idx.index,
          prepared_ctrl_words, cmd->src_asm_inst->count, cmd->binding_buffers,
          cmd->binding_device_addrs, cmd->binding_offsets, cmd->binding_lengths,
          cmd->binding_count, ctrl_code_buffer, command);
    }
    if (iree_status_is_ok(status) && single_cache_entry) {
      iree_hal_amdxdna_single_command_cache_entry_set_descriptor_template(
          single_command_cache, single_cache_entry, cmd->src_asm_inst,
          cmd->src_patches, cmd->src_constant_count,
          cmd->src_use_native_partial_elf, instr_buffer_ptr);
      if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
              command_buffer)) {
        iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
            single_cache_entry);
        status =
            iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
                command_buffer, single_command_cache, single_cache_entry);
        if (!iree_status_is_ok(status)) {
          iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
              single_command_cache, single_cache_entry);
        }
      }
      ctrl_code_buffer = NULL;
      command = NULL;
      submit_command = single_cache_entry->command;
    }
  }
  if (single_cache_locked) {
    iree_slim_mutex_unlock(&single_command_cache->mutex);
    single_cache_locked = false;
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_direct_command_buffer_submit(
        command_buffer, group->queue, submit_command, IREE_SV("dispatch"));
  }

  if (single_cache_locked) iree_slim_mutex_unlock(&single_command_cache->mutex);
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrl_code_buffer);
  iree_allocator_free(command_buffer->host_allocator, prepared_ctrl_words);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_materialize_accumulated_child(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_cmd_t* cmd) {
  if (cmd->built) return iree_ok_status();
  if (!cmd->src_asm_inst || !cmd->src_patches) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "serial dispatch fallback is missing recorded "
                            "control-code descriptors");
  }
  return iree_hal_amdxdna_make_npu_cmd(
      command_buffer, cmd->src_cu_idx, cmd->src_asm_inst, cmd->src_patches,
      cmd->binding_device_addrs, cmd->binding_buffers, cmd->binding_offsets,
      cmd->binding_lengths, cmd->binding_count,
      iree_make_const_byte_span(cmd->src_constants, cmd->src_constant_count),
      cmd->src_constant_patches, cmd->src_use_native_partial_elf,
      /*retain_signature=*/true, cmd);
}

static iree_status_t
iree_hal_amdxdna_direct_command_buffer_submit_accumulated_children_serially(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_chain_group_t* group) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < group->cmd_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[i];
    status =
        iree_hal_amdxdna_direct_command_buffer_materialize_accumulated_child(
            command_buffer, cmd);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_direct_command_buffer_submit(
          command_buffer, group->queue, cmd->command, IREE_SV("dispatch"));
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Flush all accumulated groups. Single-child groups submit directly; groups
// with multiple children become native ERT chains when supported, or serial
// direct child submissions otherwise. Groups are submitted in recorded order so
// producer/consumer dependencies across groups are honored by the device's
// in-order completion.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_flush_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer) {
  iree_hal_amdxdna_chain_accum_t* accum = &command_buffer->chain_accum;
  if (accum->group_count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  const bool device_supports_command_chain =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_COMMAND_CHAIN) != 0;
  bool has_parent_chain_group = false;
  for (iree_host_size_t i = 0; i < accum->group_count; ++i) {
    if (device_supports_command_chain &&
        iree_hal_amdxdna_chain_group_requires_parent_chain(&accum->groups[i])) {
      has_parent_chain_group = true;
      break;
    }
  }

  // Max slots per chain that fit the fixed-size exec buffer (constant per
  // device; computed once and cached). Atomic load with relaxed ordering: a
  // racing first-time probe is idempotent (same value), and the slot count is
  // independent data so we don't need ordering against any other state. Use
  // acquire on the success path / release on the store so a thread observing
  // the cached value also observes the probe's published writes.
  uint32_t max_slots = UINT32_MAX;
  if (has_parent_chain_group) {
    max_slots = iree_atomic_load(&command_buffer->device->chain_max_slots,
                                 iree_memory_order_acquire);
    if (max_slots == 0) {
      max_slots = command_buffer->device->native_caps.max_command_chain_slots;
      if (max_slots == 0) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_hal_amdxdna_native_device_c_query_chain_max_slots(
                    command_buffer->device->native_device, &max_slots));
      }
      if (max_slots == 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "amdxdna command-chain backend reported zero max slots");
      }
      iree_atomic_store(&command_buffer->device->chain_max_slots, max_slots,
                        iree_memory_order_release);
    }
  }

  // Submit each accumulated group either as one direct native command or as
  // native ERT chains chunked into max_slots-sized pieces.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t group_index = 0;
       group_index < accum->group_count && iree_status_is_ok(status);
       ++group_index) {
    iree_hal_amdxdna_chain_group_t* group = &accum->groups[group_index];
    const bool group_requires_parent_chain =
        iree_hal_amdxdna_chain_group_requires_parent_chain(group);
    const bool submit_as_chain =
        device_supports_command_chain && group_requires_parent_chain;
    if (!submit_as_chain) {
      status =
          group_requires_parent_chain
              ? iree_hal_amdxdna_direct_command_buffer_submit_accumulated_children_serially(
                    command_buffer, group)
              : iree_hal_amdxdna_direct_command_buffer_submit_accumulated_single(
                    command_buffer, group);
    } else {
      iree_hal_amdxdna_chain_command_cache_entry_t* chain_cache = NULL;
      {
        iree_hal_amdxdna_device_chain_command_cache_t* device_chain_cache =
            iree_hal_amdxdna_get_chain_command_cache(command_buffer->device);
        bool fallback_uncached = false;
        if (!device_chain_cache) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "failed to allocate amdxdna chain command cache");
          break;
        }
        iree_slim_mutex_lock(&device_chain_cache->mutex);
        bool exact_cache_hit = false;
        bool device_cache_hit = false;
        bool template_cache_hit = false;
        // Late-build fast path: reuse an already-built cached chain when
        // the descriptor inputs (control-code template + constants +
        // bindings) match exactly, without building this group's children at
        // all.
        for (iree_host_size_t i = 0; i < device_chain_cache->entry_count; ++i) {
          iree_hal_amdxdna_chain_command_cache_entry_t* entry =
              &device_chain_cache->entries[i];
          if (iree_hal_amdxdna_chain_command_cache_descriptor_matches(
                  entry, group, max_slots)) {
            chain_cache = entry;
            exact_cache_hit = true;
            break;
          }
        }
        if (!chain_cache) {
          // Dynamic constants and bindings make exact descriptor matches
          // uncommon, but the executable's packet template repeats. Match the
          // template and rewrite each retained child in place while preserving
          // its exec BO handle; cached parent chains reference those stable
          // handles.
          for (iree_host_size_t i = 0; i < device_chain_cache->entry_count;
               ++i) {
            iree_hal_amdxdna_chain_command_cache_entry_t* entry =
                &device_chain_cache->entries[i];
            if (iree_hal_amdxdna_chain_command_cache_descriptor_template_matches(
                    entry, group, max_slots)) {
              chain_cache = entry;
              template_cache_hit = true;
              break;
            }
          }
          if (chain_cache) {
            chain_cache->last_use = ++device_chain_cache->use_clock;
            bool binding_refs_changed =
                !iree_hal_amdxdna_chain_group_binding_refs_match(
                    &chain_cache->group, group);
            const bool reconf_buffers_changed =
                !iree_hal_amdxdna_chain_group_reconf_buffers_match(
                    &chain_cache->group, group);
            bool chain_code_changed = !group->native_partial_elf;
            for (iree_host_size_t i = 0;
                 i < group->cmd_count && iree_status_is_ok(status); ++i) {
              if (group->native_partial_elf) {
                bool child_code_changed = false;
                bool child_bindings_changed = false;
                status = iree_hal_amdxdna_rewrite_cached_chain_partial_elf_cmd(
                    command_buffer->host_allocator,
                    &chain_cache->group.cmds[i], &group->cmds[i],
                    &child_code_changed, &child_bindings_changed);
                chain_code_changed |= child_code_changed;
                binding_refs_changed |= child_bindings_changed;
              } else {
                status = iree_hal_amdxdna_rewrite_cached_start_npu_cmd(
                    command_buffer, &chain_cache->group.cmds[i],
                    &group->cmds[i]);
              }
            }
            if (iree_status_is_ok(status) && chain_code_changed) {
              for (iree_host_size_t i = 0;
                   i < chain_cache->chain_count && iree_status_is_ok(status);
                   ++i) {
                status =
                    iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
                        chain_cache->chains[i]);
              }
            }
            if (iree_status_is_ok(status) &&
                (binding_refs_changed || reconf_buffers_changed)) {
              // Parent commands carry driver-visible residency state in
              // addition to child command handles. Rebuild them whenever the
              // referenced resources change, including for partial ELF.
              if (reconf_buffers_changed) {
                status = iree_hal_amdxdna_chain_group_take_reconf_buffers(
                    command_buffer->host_allocator, &chain_cache->group, group);
              }
              if (iree_status_is_ok(status) && binding_refs_changed) {
                status = iree_hal_amdxdna_chain_group_set_binding_refs(
                    command_buffer->host_allocator, &chain_cache->group, group);
              }
              if (iree_status_is_ok(status)) {
                status = iree_hal_amdxdna_rebuild_cached_parent_chains(
                    command_buffer, chain_cache, max_slots);
              }
            }
            if (!iree_status_is_ok(status)) {
              iree_hal_amdxdna_chain_command_cache_entry_discard(
                  device_chain_cache, chain_cache);
              chain_cache = NULL;
            }
          }
        }
        // Not an exact hit: realize any late-bound children now so the
        // ctrl_words-based device/shape/miss logic below can match, update,
        // or cache them.
        if (!chain_cache) {
          fallback_uncached =
              !iree_hal_amdxdna_chain_command_cache_trim_for_group(
                  device_chain_cache, group, max_slots);
          for (iree_host_size_t i = 0;
               i < group->cmd_count && iree_status_is_ok(status); ++i) {
            iree_hal_amdxdna_chain_cmd_t* cmd = &group->cmds[i];
            if (cmd->built) continue;
            status = iree_hal_amdxdna_make_npu_cmd(
                command_buffer, cmd->src_cu_idx, cmd->src_asm_inst,
                cmd->src_patches, cmd->binding_device_addrs,
                cmd->binding_buffers, cmd->binding_offsets,
                cmd->binding_lengths, cmd->binding_count,
                iree_make_const_byte_span(cmd->src_constants,
                                          cmd->src_constant_count),
                cmd->src_constant_patches, cmd->src_use_native_partial_elf,
                /*retain_signature=*/!fallback_uncached, cmd);
          }
        }
        if (chain_cache) {
          chain_cache->last_use = ++device_chain_cache->use_clock;
        } else if (iree_status_is_ok(status)) {
          for (iree_host_size_t i = 0; i < device_chain_cache->entry_count;
               ++i) {
            iree_hal_amdxdna_chain_command_cache_entry_t* entry =
                &device_chain_cache->entries[i];
            if (iree_hal_amdxdna_chain_command_cache_device_matches(
                    entry, group, max_slots)) {
              chain_cache = entry;
              break;
            }
          }
          if (chain_cache) {
            device_cache_hit = true;
            chain_cache->last_use = ++device_chain_cache->use_clock;
          }
        }
        // PARTIAL_ELF dispatches commonly retain the same command shape while
        // changing bindings. Reuse a compatible entry before allocating a new
        // one, even when the cache has not reached its capacity yet.
        if (iree_status_is_ok(status) && !chain_cache &&
            group->native_partial_elf) {
          for (iree_host_size_t i = 0; i < device_chain_cache->entry_count;
               ++i) {
            iree_hal_amdxdna_chain_command_cache_entry_t* entry =
                &device_chain_cache->entries[i];
            if (iree_hal_amdxdna_chain_command_cache_shape_matches(entry, group,
                                                                   max_slots)) {
              chain_cache = entry;
              break;
            }
          }
        }
        if (chain_cache && !exact_cache_hit && !device_cache_hit &&
            !template_cache_hit) {
          chain_cache->last_use = ++device_chain_cache->use_clock;
          bool packet_changed = false;
          for (size_t i = 0; i < group->cmd_count && iree_status_is_ok(status);
               ++i) {
            bool cmd_packet_changed = false;
            status = iree_hal_amdxdna_update_cached_chain_cmd(
                &chain_cache->group.cmds[i], &group->cmds[i],
                &cmd_packet_changed, NULL, NULL, NULL);
            if (cmd_packet_changed) {
              packet_changed = true;
            }
          }
          if (packet_changed) {
            for (iree_host_size_t i = 0;
                 i < chain_cache->chain_count && iree_status_is_ok(status);
                 ++i) {
              if (!iree_status_is_ok(status)) break;
              status = iree_hal_amdxdna_native_command_c_mark_chain_code_dirty(
                  chain_cache->chains[i]);
            }
          }
        } else if (iree_status_is_ok(status) && !chain_cache &&
                   !fallback_uncached) {
          chain_cache = iree_hal_amdxdna_chain_command_cache_allocate_entry(
              device_chain_cache, group, max_slots);
          if (!chain_cache) {
            fallback_uncached = true;
          } else {
            chain_cache->group.context =
                iree_hal_amdxdna_native_context_ref_retain(group->context);
            chain_cache->group.queue = group->queue;
            chain_cache->group.native_partial_elf = group->native_partial_elf;
            status = iree_hal_amdxdna_chain_group_take_cmds(
                command_buffer->host_allocator, &chain_cache->group, group);
            if (iree_status_is_ok(status)) {
              status = iree_hal_amdxdna_chain_group_take_reconf_buffers(
                  command_buffer->host_allocator, &chain_cache->group, group);
            }
            for (iree_host_size_t i = 0;
                 i < chain_cache->group.cmd_count && iree_status_is_ok(status);
                 ++i) {
              status = iree_hal_amdxdna_chain_cmd_make_deferred_lists_owned(
                  command_buffer->host_allocator, &chain_cache->group.cmds[i]);
            }
            if (iree_status_is_ok(status)) {
              status = iree_hal_amdxdna_chain_group_set_binding_refs(
                  command_buffer->host_allocator, &chain_cache->group, group);
            }
            chain_cache->max_slots = max_slots;
            chain_cache->last_use = ++device_chain_cache->use_clock;
            if (iree_status_is_ok(status)) {
              status = iree_hal_amdxdna_rebuild_cached_parent_chains(
                  command_buffer, chain_cache, max_slots);
            }
          }
        }
        if (iree_status_is_ok(status) && chain_cache &&
            chain_cache->chain_count != 0) {
          if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
                  command_buffer)) {
            iree_hal_amdxdna_chain_command_cache_entry_acquire_in_flight(
                chain_cache);
            status =
                iree_hal_amdxdna_direct_command_buffer_defer_chain_cache_release(
                    command_buffer, device_chain_cache, chain_cache);
            if (!iree_status_is_ok(status)) {
              iree_hal_amdxdna_chain_command_cache_entry_release_in_flight(
                  device_chain_cache, chain_cache);
            }
          }
        }
        if (iree_status_is_ok(status) && chain_cache &&
            chain_cache->chain_count != 0) {
          status = iree_hal_amdxdna_direct_command_buffer_submit_all(
              command_buffer, group->queue, chain_cache->chains,
              chain_cache->chain_count, IREE_SV("ERT_CMD_CHAIN"));
        }
        iree_slim_mutex_unlock(&device_chain_cache->mutex);
        if (iree_status_is_ok(status) && fallback_uncached) {
          if (iree_status_is_ok(status)) {
            status =
                iree_hal_amdxdna_direct_command_buffer_submit_uncached_parent_chains(
                    command_buffer, group, max_slots);
          }
        }
      }
    }
    if (!iree_status_is_ok(status)) break;
  }
  // Synchronous flushes can drop the accumulator immediately. Async flushes
  // keep command-buffer-owned child commands, control-code BOs, reconf BOs, and
  // binding refs alive until the completion batch has waited native completion.
  if (!iree_hal_amdxdna_completion_batch_has_work(
          command_buffer->completion_batch)) {
    iree_hal_amdxdna_chain_accum_clear(command_buffer->host_allocator, accum);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    const iree_hal_amdxdna_u32_list_t* asm_inst,
    const iree_hal_amdxdna_u32_list_t* patch_table,
    iree_const_byte_span_t constants, bool use_single_partial_elf) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  uint64_t* binding_addrs = NULL;
  iree_hal_amdxdna_native_buffer_t** binding_buffers = NULL;
  iree_device_size_t* binding_offsets = NULL;
  iree_device_size_t* binding_lengths = NULL;
  uint32_t* prepared_ctrl_words = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrl_code_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;
  iree_hal_amdxdna_native_command_t* submit_command = NULL;
  iree_hal_amdxdna_device_single_command_cache_t* single_command_cache = NULL;
  iree_hal_amdxdna_single_command_cache_entry_t* single_cache_entry = NULL;
  bool single_cache_locked = false;

  if (use_single_partial_elf) {
    if (bindings.count != 0) {
      status = iree_allocator_malloc_array(
          command_buffer->host_allocator, bindings.count,
          sizeof(*binding_addrs), (void**)&binding_addrs);
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_buffers), (void**)&binding_buffers);
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_offsets), (void**)&binding_offsets);
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            command_buffer->host_allocator, bindings.count,
            sizeof(*binding_lengths), (void**)&binding_lengths);
      }
    }
    for (iree_host_size_t j = 0;
         iree_status_is_ok(status) && j < bindings.count; ++j) {
      iree_hal_amdxdna_native_buffer_t* native_buffer =
          iree_hal_amdxdna_buffer_handle(
              iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
      status = iree_hal_amdxdna_native_buffer_c_ensure_allocated(native_buffer);
      if (iree_status_is_ok(status)) {
        const iree_device_size_t native_offset =
            iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
            bindings.values[j].offset;
        binding_addrs[j] =
            iree_hal_amdxdna_native_buffer_c_device_address(native_buffer) +
            native_offset;
        binding_buffers[j] = native_buffer;
        binding_offsets[j] = native_offset;
        binding_lengths[j] = bindings.values[j].length;
      }
    }
  }

  // Reuse a prepared single-dispatch native command across queue_execute calls,
  // keyed by the dispatch signature in the device single-command cache.
  if (use_single_partial_elf) {
    status = iree_allocator_malloc_array(
        command_buffer->host_allocator, asm_inst->count,
        sizeof(*prepared_ctrl_words), (void**)&prepared_ctrl_words);
    if (iree_status_is_ok(status)) {
      memcpy(prepared_ctrl_words, asm_inst->data,
             asm_inst->count * sizeof(*prepared_ctrl_words));
      status = iree_hal_amdxdna_patch_write32_constants(
          prepared_ctrl_words, asm_inst->count, constants);
    }
    if (iree_status_is_ok(status) &&
        !iree_hal_amdxdna_apply_patch_table(
            prepared_ctrl_words, asm_inst->count, patch_table->data,
            patch_table->count, binding_addrs, bindings.count)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "amdxdna PARTIAL_ELF single dispatch has an invalid host "
          "patch table");
    }
    if (iree_status_is_ok(status)) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to allocate amdxdna single command cache");
      }
      if (iree_status_is_ok(status)) {
        iree_slim_mutex_lock(&single_command_cache->mutex);
        single_cache_locked = true;
        status = iree_hal_amdxdna_find_single_command_cache_entry(
            single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
            asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
            binding_lengths, bindings.count, &single_cache_entry);
      }
      if (iree_status_is_ok(status) && single_cache_entry) {
        if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
                command_buffer)) {
          iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
              single_cache_entry);
          status =
              iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
                  command_buffer, single_command_cache, single_cache_entry);
          if (!iree_status_is_ok(status)) {
            iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
                single_command_cache, single_cache_entry);
          }
        }
        submit_command = single_cache_entry->command;
      } else if (iree_status_is_ok(status)) {
        iree_slim_mutex_unlock(&single_command_cache->mutex);
        single_cache_locked = false;
      }
    }
  }

  // Allocate a buffer object to hold the control code (`asm_inst`).
  size_t ctrl_code_size = asm_inst->count * sizeof(uint32_t);
  if (iree_status_is_ok(status) && !submit_command) {
    const bool uses_native_instruction_buffer =
        use_single_partial_elf ||
        (command_buffer->device->native_caps.dispatch_models &
         IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        command_buffer->device->native_device, ctrl_code_size,
        uses_native_instruction_buffer
            ? IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION
            : IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE,
        &ctrl_code_buffer);
  }
  void* instr_buffer_ptr = NULL;
  uint32_t* instr_buffer = NULL;
  if (iree_status_is_ok(status) && !submit_command) {
    status = iree_hal_amdxdna_native_buffer_c_map(ctrl_code_buffer,
                                                  &instr_buffer_ptr);
    if (iree_status_is_ok(status)) {
      instr_buffer = (uint32_t*)instr_buffer_ptr;
      if (use_single_partial_elf) {
        memcpy(instr_buffer, prepared_ctrl_words, ctrl_code_size);
      } else {
        memcpy(instr_buffer, asm_inst->data, ctrl_code_size);
        status = iree_hal_amdxdna_patch_write32_constants(
            instr_buffer, asm_inst->count, constants);
      }
    }
    if (iree_status_is_ok(status) &&
        !command_buffer->device->native_caps
             .native_owns_control_code_publication) {
      status = iree_hal_amdxdna_native_buffer_c_sync_all(
          ctrl_code_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
    }
  }

  const iree_hal_amdxdna_native_c_command_opcode_t command_opcode =
      use_single_partial_elf
          ? IREE_HAL_AMDXDNA_NATIVE_C_COMMAND_OPCODE_START_NPU_PARTIAL_ELF
          : command_buffer->device->native_caps.default_dispatch_opcode;
  if (iree_status_is_ok(status) && !submit_command) {
    status = iree_hal_amdxdna_native_command_c_create(
        command_buffer->device->native_device, command_opcode, &command);
    // Add the kernel arguments.
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_set_cu_index(command, cu_idx);
    }
    if (iree_status_is_ok(status) && use_single_partial_elf) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_64(
            command, kAie2ExecBufferKernelOpTxn);
      }
    } else if (iree_status_is_ok(status) &&
               (command_buffer->device->native_caps.dispatch_models &
                IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0) {
      status = iree_hal_amdxdna_native_command_c_add_control_buffer(
          command, ctrl_code_buffer, ctrl_code_size);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_64(
            command, kAie2ExecBufferKernelOpTxn);
      }
    } else {
      status = iree_hal_amdxdna_native_command_c_add_arg_64(
          command, kAmdxdnaControlCodeOpcode);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_buffer_arg(
            command, ctrl_code_buffer);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_native_command_c_add_arg_32(
            command, (uint32_t)asm_inst->count);
      }
    }
  }

  if (iree_status_is_ok(status) && !submit_command) {
    if (use_single_partial_elf) {
      for (iree_host_size_t j = 0;
           iree_status_is_ok(status) && j < bindings.count; ++j) {
        status = iree_hal_amdxdna_native_command_c_bind_buffer(
            command, /*position=*/j + 1, binding_buffers[j], binding_offsets[j],
            binding_lengths[j]);
      }
    } else {
      for (iree_host_size_t j = 0;
           iree_status_is_ok(status) && j < bindings.count; ++j) {
        iree_hal_amdxdna_native_buffer_t* native_buffer =
            iree_hal_amdxdna_buffer_handle(
                iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
        // Propagate per-binding byte_offset (both the buffer's own subview
        // offset within its allocated root, and the binding-level offset) into
        // the device-side address. Without this, two bindings on the same root
        // BO at different offsets collapse to the same physical address,
        // causing the next dispatch to read/write the wrong slot.
        uint64_t buffer_byte_off =
            (uint64_t)iree_hal_buffer_byte_offset(bindings.values[j].buffer);
        uint64_t binding_off = (uint64_t)bindings.values[j].offset;
        iree_device_size_t native_offset = buffer_byte_off + binding_off;
        status = iree_hal_amdxdna_native_command_c_add_buffer_arg_at_offset(
            command, native_buffer, native_offset);
      }
    }
  }

  if (iree_status_is_ok(status) && use_single_partial_elf && !submit_command) {
    if (!single_command_cache) {
      single_command_cache =
          iree_hal_amdxdna_get_single_command_cache(command_buffer->device);
      if (!single_command_cache) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "failed to allocate amdxdna single command cache");
      }
    }
    if (iree_status_is_ok(status) && !single_cache_locked) {
      iree_slim_mutex_lock(&single_command_cache->mutex);
      single_cache_locked = true;
    }
    // Another queue worker may have populated the entry while this thread was
    // building the native command. Recheck under the cache lock before storing.
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_find_single_command_cache_entry(
          single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
          asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
          binding_lengths, bindings.count, &single_cache_entry);
    }
    bool transferred_to_cache = false;
    if (iree_status_is_ok(status) && !single_cache_entry) {
      single_cache_entry = iree_hal_amdxdna_store_single_command_cache_entry(
          single_command_cache, queue, cu_idx.index, prepared_ctrl_words,
          asm_inst->count, binding_buffers, binding_addrs, binding_offsets,
          binding_lengths, bindings.count, ctrl_code_buffer, command);
      transferred_to_cache = single_cache_entry != NULL;
    }
    if (iree_status_is_ok(status) && single_cache_entry) {
      if (iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
              command_buffer)) {
        iree_hal_amdxdna_single_command_cache_entry_acquire_in_flight(
            single_cache_entry);
        status =
            iree_hal_amdxdna_direct_command_buffer_defer_single_cache_release(
                command_buffer, single_command_cache, single_cache_entry);
        if (!iree_status_is_ok(status)) {
          iree_hal_amdxdna_single_command_cache_entry_release_in_flight(
              single_command_cache, single_cache_entry);
        }
      }
      if (transferred_to_cache) {
        ctrl_code_buffer = NULL;
        command = NULL;
      }
      submit_command = single_cache_entry->command;
    }
  }

  if (iree_status_is_ok(status) && !submit_command) {
    submit_command = command;
    status =
        iree_hal_amdxdna_direct_command_buffer_defer_native_command_and_buffer_destroy(
            command_buffer, &command, &ctrl_code_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_direct_command_buffer_submit(
        command_buffer, queue, submit_command, IREE_SV("dispatch"));
  }
  if (single_cache_locked) iree_slim_mutex_unlock(&single_command_cache->mutex);
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrl_code_buffer);
  iree_allocator_free(command_buffer->host_allocator, prepared_ctrl_words);
  iree_allocator_free(command_buffer->host_allocator, binding_lengths);
  iree_allocator_free(command_buffer->host_allocator, binding_offsets);
  iree_allocator_free(command_buffer->host_allocator, binding_buffers);
  iree_allocator_free(command_buffer->host_allocator, binding_addrs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_create_reconfigure_command(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx,
    iree_hal_amdxdna_native_buffer_t* ctrlpkt_inst_buffer,
    size_t ctrlpkt_inst_size, uint32_t ctrlpkt_inst_word_count,
    iree_hal_amdxdna_native_buffer_t* ctrlpkt_seq_buffer,
    iree_hal_amdxdna_native_command_t** out_command) {
  *out_command = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;
  iree_status_t status = iree_hal_amdxdna_native_command_c_create(
      command_buffer->device->native_device,
      command_buffer->device->native_caps.default_dispatch_opcode, &command);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_command_c_set_cu_index(command, cu_idx);
  }
  if (iree_status_is_ok(status) &&
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0) {
    status = iree_hal_amdxdna_native_command_c_add_control_buffer(
        command, ctrlpkt_inst_buffer, ctrlpkt_inst_size);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_arg_64(
          command, kAie2ExecBufferKernelOpTxn);
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_command_c_add_arg_64(
        command, kAmdxdnaControlCodeOpcode);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_buffer_arg(
          command, ctrlpkt_inst_buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_command_c_add_arg_32(
          command, ctrlpkt_inst_word_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_command_c_add_buffer_arg(
        command, ctrlpkt_seq_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command = command;
  } else {
    iree_hal_amdxdna_native_command_c_destroy(command);
  }
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reconfigure(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_c_cu_index_t cu_idx, uint32_t n_reconfigure_runs,
    const iree_hal_amdxdna_u32_list_t* ctrlpkt_inst,
    const iree_hal_amdxdna_u32_list_t* ctrlpkt_seq,
    iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  iree_hal_amdxdna_native_buffer_t* ctrlpkt_inst_buffer = NULL;
  iree_hal_amdxdna_native_buffer_t* ctrlpkt_seq_buffer = NULL;
  iree_hal_amdxdna_native_command_t* command = NULL;

  // Allocate a buffer object to hold the control packet instructions.
  size_t ctrlpkt_inst_size = ctrlpkt_inst->count * sizeof(uint32_t);
  const bool uses_native_instruction_buffer =
      (command_buffer->device->native_caps.dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_START_NPU) != 0;
  status = iree_hal_amdxdna_native_device_c_alloc_buffer(
      command_buffer->device->native_device, ctrlpkt_inst_size,
      uses_native_instruction_buffer
          ? IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION
          : IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE,
      &ctrlpkt_inst_buffer);
  void* ctrlpkt_inst_ptr = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_buffer_c_map(ctrlpkt_inst_buffer,
                                                  &ctrlpkt_inst_ptr);
  }
  if (iree_status_is_ok(status)) {
    uint32_t* ctrlpkt_inst_words = (uint32_t*)ctrlpkt_inst_ptr;
    memcpy(ctrlpkt_inst_words, ctrlpkt_inst->data, ctrlpkt_inst_size);
    status = iree_hal_amdxdna_patch_write32_constants(
        ctrlpkt_inst_words, ctrlpkt_inst->count, constants);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_buffer_c_sync_all(
        ctrlpkt_inst_buffer,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
  }

  // Allocate a buffer object to hold the control packet sequence (content).
  size_t ctrlpkt_seq_size = ctrlpkt_seq->count * sizeof(uint32_t);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        command_buffer->device->native_device, ctrlpkt_seq_size,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY, &ctrlpkt_seq_buffer);
  }
  void* ctrlpkt_seq_ptr = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_native_buffer_c_map(ctrlpkt_seq_buffer,
                                                  &ctrlpkt_seq_ptr);
  }
  if (iree_status_is_ok(status)) {
    memcpy(ctrlpkt_seq_ptr, ctrlpkt_seq->data, ctrlpkt_seq_size);
    status = iree_hal_amdxdna_native_buffer_c_sync_all(
        ctrlpkt_seq_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_create_reconfigure_command(
        command_buffer, cu_idx, ctrlpkt_inst_buffer, ctrlpkt_inst_size,
        (uint32_t)ctrlpkt_inst->count, ctrlpkt_seq_buffer, &command);
  }

  bool completion_owns_inst_buffer = false;
  bool completion_owns_seq_buffer = false;
  const bool async_completion =
      iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer);
  if (iree_status_is_ok(status) && async_completion) {
    status = iree_hal_amdxdna_completion_batch_add_cleanup(
        command_buffer->completion_batch,
        iree_hal_amdxdna_completion_destroy_native_buffer, ctrlpkt_inst_buffer);
    if (iree_status_is_ok(status)) completion_owns_inst_buffer = true;
  }
  if (iree_status_is_ok(status) && async_completion) {
    status = iree_hal_amdxdna_completion_batch_add_cleanup(
        command_buffer->completion_batch,
        iree_hal_amdxdna_completion_destroy_native_buffer, ctrlpkt_seq_buffer);
    if (iree_status_is_ok(status)) completion_owns_seq_buffer = true;
  }

  // Execute the reconfiguration for `n_reconfigure_runs` times.
  for (uint32_t i = 0; iree_status_is_ok(status) && i < n_reconfigure_runs;
       ++i) {
    iree_hal_amdxdna_native_command_t* submit_command = command;
    bool completion_owns_command = false;
    if (async_completion && i != 0) {
      submit_command = NULL;
      status = iree_hal_amdxdna_create_reconfigure_command(
          command_buffer, cu_idx, ctrlpkt_inst_buffer, ctrlpkt_inst_size,
          (uint32_t)ctrlpkt_inst->count, ctrlpkt_seq_buffer, &submit_command);
    }
    if (iree_status_is_ok(status) && async_completion) {
      status = iree_hal_amdxdna_completion_batch_add_cleanup(
          command_buffer->completion_batch,
          iree_hal_amdxdna_completion_destroy_native_command, submit_command);
      if (iree_status_is_ok(status)) completion_owns_command = true;
    }
    if (!iree_status_is_ok(status)) {
      if (i != 0 || submit_command != command) {
        iree_hal_amdxdna_native_command_c_destroy(submit_command);
      }
      break;
    }
    status = iree_hal_amdxdna_direct_command_buffer_submit(
        command_buffer, queue, submit_command,
        IREE_SV("control-packet reconfiguration"));
    if (completion_owns_command && submit_command == command) {
      command = NULL;
    } else if (!completion_owns_command && submit_command != command) {
      iree_hal_amdxdna_native_command_c_destroy(submit_command);
    }
  }

  if (completion_owns_seq_buffer) ctrlpkt_seq_buffer = NULL;
  if (completion_owns_inst_buffer) ctrlpkt_inst_buffer = NULL;
  iree_hal_amdxdna_native_command_c_destroy(command);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrlpkt_seq_buffer);
  iree_hal_amdxdna_native_buffer_c_destroy(ctrlpkt_inst_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdxdna_dispatch_plan_initialize(
    const iree_hal_amdxdna_native_c_device_caps_t* native_caps,
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_amdxdna_dispatch_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(native_caps);
  IREE_ASSERT_ARGUMENT(base_executable);
  IREE_ASSERT_ARGUMENT(out_plan);
  memset(out_plan, 0, sizeof(*out_plan));

  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point function %" PRIu64
                            " out of range; executable only contains %" PRIhsz
                            " entry points",
                            function.value, executable->entry_point_count);
  }
  const uint32_t entry_point = iree_hal_executable_function_index(function);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %u out of range; executable "
                            "only contains %" PRIhsz " entry points",
                            entry_point, executable->entry_point_count);
  }

  iree_hal_amdxdna_kernel_params_t* kernel_params =
      &executable->entry_points[entry_point];
  out_plan->executable = executable;
  out_plan->kernel_params = kernel_params;
  out_plan->entry_point = entry_point;
  out_plan->control_code_count = kernel_params->asm_inst_runlist_count;
  out_plan->control_codes = kernel_params->asm_inst_runlist;
  out_plan->patch_table_count = kernel_params->patch_runlist_count;
  out_plan->patch_tables = kernel_params->patch_runlist;
  out_plan->constant_patch_table_count =
      kernel_params->constant_patch_runlist_count;
  out_plan->constant_patch_tables = kernel_params->constant_patch_runlist;
  out_plan->data_payload_count = kernel_params->reconf_data_runlist_count;
  out_plan->data_payloads = kernel_params->reconf_data_runlist;
  out_plan->data_payload_run_count = kernel_params->n_reconfigure_runs;
  out_plan->pdi_span = iree_make_const_byte_span(kernel_params->pdi.data,
                                                 kernel_params->pdi.count);
  out_plan->xclbin_span = iree_make_const_byte_span(
      kernel_params->xclbin.data, kernel_params->xclbin.count);
  out_plan->kernel_name = kernel_params->kernel_name;
  out_plan->use_native_partial_elf_context =
      (native_caps->dispatch_models &
       IREE_HAL_AMDXDNA_NATIVE_C_DISPATCH_MODEL_PARTIAL_ELF) != 0 &&
      out_plan->data_payload_count == 0 &&
      out_plan->xclbin_span.data_length > 0 &&
      out_plan->patch_table_count != 0 &&
      iree_hal_amdxdna_patch_table_is_valid(&out_plan->patch_tables[0]);
  out_plan->has_host_patch_table =
      out_plan->patch_table_count == out_plan->control_code_count;
  out_plan->multi_control_code_or_pdi = kernel_params->n_pdi_loads > 1 ||
                                        out_plan->control_code_count > 1 ||
                                        out_plan->data_payload_count != 0;
  // Accumulate every host-patch-table dispatch. The accumulator is the common
  // host-patched native START_NPU/PARTIAL_ELF path; end() decides whether a
  // multi-child group can become a parent ERT_CMD_CHAIN or must fall back to
  // direct child submissions based on native caps.
  out_plan->use_chain_accumulation_policy = out_plan->has_host_patch_table;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch_plan(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_amdxdna_dispatch_plan_t* plan,
    iree_const_byte_span_t constants, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  (void)flags;
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(plan->executable);
  IREE_ASSERT_ARGUMENT(plan->kernel_params);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);

  iree_status_t status =
      iree_hal_amdxdna_validate_live_dispatch_bindings(bindings);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                          &plan->executable);
  }

  iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;
  iree_hal_amdxdna_executable* executable = plan->executable;
  iree_hal_amdxdna_kernel_params_t* kernel_params = plan->kernel_params;
  iree_hal_amdxdna_native_c_cu_index_t cu_idx;
  memset(&cu_idx, 0, sizeof(cu_idx));

  // Resolve the hardware context (loaded NPU array + opened CU) this dispatch
  // runs on. There are three executable shapes here, distinguished by whether
  // the entry point carries reconfigure-data payloads and whether it has its
  // own context image (PDI/xclbin). The backend is NOT what selects the path:
  // both Windows MCDM (PARTIAL_ELF + xclbin context image) and Linux KMQ (PDI
  // context image) take branch 1; they differ only in what
  // get_or_create_context builds, which it picks from the device's
  // context_image_models.
  //
  //   1. data_payload_count == 0 -- self-contained dispatch. The entry point
  //      carries its own context image (PDI or xclbin). The context is
  //      keyed by the native context-image inputs (PDI+CU or xclbin) and
  //      borrowed for this dispatch. Dispatch control code is command identity
  //      and must not split hwctx objects. Do not retain the borrowed context
  //      on the executable: FLM and similar direct-XADX users may cache many
  //      executable handles, and executable-owned native contexts would bypass
  //      the device LRU and exhaust the small driver hwctx pool.
  //
  //   2. data_payload_count != 0 with a PDI/xclbin on this entry point --
  //      context-loading control-packet entry point. It (re)loads the array and
  //      publishes the result as executable->context for sibling reuse-context
  //      entry points to run against.
  //
  //   3. data_payload_count != 0 with no context image -- reuse-context entry
  //      point. It has no image of its own and runs on the executable->context
  //      that a sibling loader (branch 2) published; it fails if none has.
  if (iree_status_is_ok(status) && plan->data_payload_count == 0) {
    status = iree_hal_amdxdna_device_get_or_create_context(
        command_buffer->device, plan->pdi_span, plan->xclbin_span,
        plan->kernel_name, &context_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_context_ref_open_cu(
          context_ref, plan->kernel_name, &cu_idx);
    }
  } else if (iree_status_is_ok(status) && (kernel_params->pdi.count != 0 ||
                                           kernel_params->xclbin.count != 0)) {
    status = iree_hal_amdxdna_device_get_or_create_context(
        command_buffer->device, plan->pdi_span, plan->xclbin_span,
        plan->kernel_name, &context_ref);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_context_ref_open_cu(
          context_ref, plan->kernel_name, &cu_idx);
    }
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&executable->context_mutex);
      iree_hal_amdxdna_native_context_ref_release(executable->context);
      executable->context =
          iree_hal_amdxdna_native_context_ref_retain(context_ref);
      executable->context_cu_index = cu_idx;
      executable->context_cu_index_valid = true;
      iree_slim_mutex_unlock(&executable->context_mutex);
    }
  } else if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&executable->context_mutex);
    if (executable->context_cu_index_valid) {
      context_ref =
          iree_hal_amdxdna_native_context_ref_retain(executable->context);
      cu_idx = executable->context_cu_index;
    }
    iree_slim_mutex_unlock(&executable->context_mutex);
  }
  if (iree_status_is_ok(status) && !context_ref) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna: control-packet dispatch with no context image ran before "
        "its PDI/xclbin-carrying entry point loaded the array");
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdxdna_direct_command_buffer_uses_async_completion(
          command_buffer)) {
    iree_hal_amdxdna_native_context_ref_t* completion_context_ref =
        iree_hal_amdxdna_native_context_ref_retain(context_ref);
    status = iree_hal_amdxdna_completion_batch_add_cleanup(
        command_buffer->completion_batch,
        iree_hal_amdxdna_completion_release_context_ref,
        completion_context_ref);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_native_context_ref_release(completion_context_ref);
    }
  }

  iree_hal_amdxdna_native_queue_t* queue = NULL;
  if (iree_status_is_ok(status)) {
    queue = iree_hal_amdxdna_native_context_ref_queue(context_ref);
  }

  if (iree_status_is_ok(status) && plan->use_chain_accumulation_policy) {
    // Accumulate this dispatch's commands; end() chooses the native shape
    // from the recorded stream: one child submits as a direct single
    // dispatch, while multiple children or multi-control-code artifacts
    // become ERT_CMD_CHAINs.
    status = iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
        bindings, command_buffer, context_ref, queue, cu_idx, plan, constants,
        plan->use_native_partial_elf_context);
  } else if (iree_status_is_ok(status) && plan->data_payload_count == 0) {
    // Normal kernel dispatch.
    const iree_hal_amdxdna_u32_list_t* patch_table =
        plan->patch_table_count == 0 ? NULL : &plan->patch_tables[0];
    status = iree_hal_amdxdna_direct_command_buffer_normal_run(
        bindings, command_buffer, queue, cu_idx, &plan->control_codes[0],
        patch_table, constants, plan->use_native_partial_elf_context);
  } else {
    for (size_t i = 0;
         iree_status_is_ok(status) && i < plan->data_payload_count; i++) {
      // Reconfigure the device.
      status = iree_hal_amdxdna_direct_command_buffer_reconfigure(
          command_buffer, queue, cu_idx, plan->data_payload_run_count,
          &plan->control_codes[2 * i], &plan->data_payloads[i], constants);
      if (iree_status_is_ok(status)) {
        // Dispatch the new kernel.
        const size_t run_idx = 2 * i + 1;
        const iree_hal_amdxdna_u32_list_t* patch_table =
            run_idx < plan->patch_table_count ? &plan->patch_tables[run_idx]
                                              : NULL;
        status = iree_hal_amdxdna_direct_command_buffer_normal_run(
            bindings, command_buffer, queue, cu_idx,
            &plan->control_codes[run_idx], patch_table, constants,
            /*use_single_partial_elf=*/false);
      }
    }
  }
  iree_hal_amdxdna_native_context_ref_release(context_ref);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_hal_amdxdna_dispatch_plan_t plan;
  iree_status_t status = iree_hal_amdxdna_dispatch_plan_initialize(
      &command_buffer->device->native_caps, base_executable, function, &plan);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_direct_command_buffer_dispatch_plan(
        base_command_buffer, &plan, constants, bindings, flags);
  }
  return status;
}

// Flush accumulated dispatches once the whole direct command buffer has been
// recorded. No-op when every dispatch used the immediate path.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_status_t status =
      iree_hal_amdxdna_direct_command_buffer_flush_chains(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDXDNA collectives are not implemented");
}

const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable = {
        .destroy = iree_hal_amdxdna_direct_command_buffer_destroy,
        .begin = iree_hal_amdxdna_direct_command_buffer_begin,
        .end = iree_hal_amdxdna_direct_command_buffer_end,
        .begin_debug_group =
            iree_hal_amdxdna_direct_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_amdxdna_direct_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_amdxdna_direct_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdxdna_direct_command_buffer_signal_event,
        .reset_event = iree_hal_amdxdna_direct_command_buffer_reset_event,
        .wait_events = iree_hal_amdxdna_direct_command_buffer_wait_events,
        .advise_buffer = iree_hal_amdxdna_direct_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdxdna_direct_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdxdna_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdxdna_direct_command_buffer_copy_buffer,
        .collective = iree_hal_amdxdna_direct_command_buffer_collective,
        .dispatch = iree_hal_amdxdna_direct_command_buffer_dispatch,
};
