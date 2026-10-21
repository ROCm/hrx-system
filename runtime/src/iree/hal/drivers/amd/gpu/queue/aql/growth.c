// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/growth.h"

#include "iree/hal/drivers/amd/gpu/queue/aql/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"

struct iree_hal_amd_gpu_aql_growth_t {
  // Native completion and an enrolled program reuse hint own separate borrows.
  iree_atomic_ref_count_t ref_count;
  // Native completion joins the same FIFO as captured commands.
  iree_hal_amd_gpu_aql_entry_t entry;
  // The publisher has consumed the native result and disarmed program reuse.
  bool completed;
  // Stable typed record for a program's coalesced availability hint.
  iree_hal_amd_gpu_aql_entry_t reuse_entry;
  // One while reuse_entry is linked or awaiting consumption.
  iree_atomic_int32_t reuse_pending;
  // Resident state owned through final native clone/hint release.
  iree_hal_amd_gpu_aql_command_program_t* command_program;
  // Borrowed exact publication owner, live through the native callback join.
  iree_hal_amd_gpu_aql_growth_context_t* context;
  // Complete reusable capture extent, independent of the waiting cohort size.
  iree_host_size_t allocation_size;
  // Captures parked outside publication until this native acquisition returns.
  iree_hal_amd_gpu_publication_entry_t* first;
  // Constant-time append and requeue boundary for the waiting cohort.
  iree_hal_amd_gpu_publication_entry_t* last;
  // Native acquisition outcome, owned until publication consumes this record.
  iree_status_t status;
  union {
    // Ordinary argument or signal generation acquired under the queue policy.
    struct {
      // Typed acquisition geometry, borrowed through native completion.
      iree_hal_amd_fabric_memory_request_t request;
      // Context lifetime borrow and native acquisition descriptor.
      iree_hal_amd_gpu_memory_acquire_t acquire;
      union {
        // Owned argument generation for GROW_ARGUMENTS.
        iree_hal_amd_gpu_argument_storage_t* arguments;
        // Owned signal generation for GROW_SIGNALS.
        iree_hal_amd_gpu_signal_storage_t* signals;
      };
    } backing;
    // Indexed physical clone reservation and native acquisition ownership.
    iree_hal_amd_gpu_program_growth_t program;
  };
};

static void iree_hal_amd_gpu_aql_growth_join(
    iree_hal_amd_gpu_aql_growth_t* growth) {
  if (iree_atomic_ref_count_dec(&growth->ref_count) == 1) {
    iree_hal_amd_gpu_aql_growth_context_t* context = growth->context;
    iree_hal_amd_gpu_aql_command_program_release(growth->command_program);
    iree_hal_amd_gpu_capture_pool_release(
        context->captures,
        iree_make_byte_span(growth, growth->allocation_size));
    iree_atomic_fetch_sub(&context->pending_count, 1,
                          iree_memory_order_release);
  }
}

static void iree_hal_amd_gpu_aql_growth_append(
    iree_hal_amd_gpu_aql_growth_t* growth,
    iree_hal_amd_gpu_aql_submission_t* submission) {
  iree_hal_amd_gpu_publication_entry_t* entry = &submission->entry.publication;
  iree_atomic_store(&entry->next, 0, iree_memory_order_relaxed);
  if (growth->last) {
    iree_atomic_store(&growth->last->next, (intptr_t)entry,
                      iree_memory_order_relaxed);
  } else {
    growth->first = entry;
  }
  growth->last = entry;
}

// A ready bit alone cannot reserve publication while another caller owns it.
// Typed records use the same FIFO/handoff contract as captured commands.
void iree_hal_amd_gpu_aql_growth_enqueue(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_t* entry) {
  if (iree_hal_amd_gpu_publisher_enqueue(context->publisher,
                                         &entry->publication)) {
    iree_hal_amd_gpu_publication_entry_t* ready =
        iree_hal_amd_gpu_publisher_next(context->publisher);
    if (iree_hal_amd_gpu_publisher_handoff(context->publisher, ready)) {
      iree_hal_amd_gpu_publication_notify(context->publication,
                                          context->queue_ordinal);
    }
  }
}

static void iree_hal_amd_gpu_aql_growth_completed(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_aql_growth_t* growth = user_data;
  growth->status = status;
  // No growth access follows enqueue: another publisher may consume and free
  // it immediately. The context's native callback join protects the owner.
  iree_hal_amd_gpu_aql_growth_enqueue(growth->context, &growth->entry);
}

static void iree_hal_amd_gpu_aql_growth_program_available(void* user_data) {
  iree_hal_amd_gpu_aql_growth_t* growth = user_data;
  if (!iree_atomic_exchange(&growth->reuse_pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_atomic_ref_count_inc(&growth->ref_count);
    iree_hal_amd_gpu_aql_growth_enqueue(growth->context, &growth->reuse_entry);
  }
}

void iree_hal_amd_gpu_aql_growth_signals_available(void* user_data) {
  iree_hal_amd_gpu_aql_growth_context_t* context = user_data;
  if (iree_atomic_load(&context->signal_cache->waiting,
                       iree_memory_order_acquire) &&
      !iree_atomic_exchange(&context->reuse.pending, 1,
                            iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_aql_growth_enqueue(context, &context->reuse.entry);
  }
}

static void iree_hal_amd_gpu_aql_growth_await_signals(
    iree_hal_amd_gpu_aql_growth_context_t* context) {
  iree_atomic_store(&context->signal_cache->waiting, 1,
                    iree_memory_order_release);
  // Either last release observes enrollment, or this recheck observes cells
  // returned between the acquisition miss and enrollment.
  if (iree_hal_amd_gpu_signal_pool_has_available(
          &context->signal_cache->pool)) {
    iree_hal_amd_gpu_aql_growth_signals_available(context);
  }
}

iree_status_t iree_hal_amd_gpu_aql_growth_begin(
    iree_hal_amd_gpu_aql_growth_context_t* context,
    iree_hal_amd_gpu_aql_entry_kind_t kind,
    iree_hal_amd_gpu_aql_submission_t* submission, uint64_t capacity) {
  iree_hal_amd_gpu_aql_growth_t* pending = NULL;
  iree_hal_amd_gpu_program_t* program = NULL;
  switch (kind) {
    case IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS:
      pending = context->arguments;
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS:
      pending = context->signals;
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_PROGRAM:
      program = &submission->execute.program->resident;
      if (program->queues[context->queue_ordinal].growth) {
        pending =
            iree_containerof(program->queues[context->queue_ordinal].growth,
                             iree_hal_amd_gpu_aql_growth_t, program);
      }
      break;
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
  if (pending) {
    iree_hal_amd_gpu_aql_growth_append(pending, submission);
    if (kind == IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS) {
      iree_hal_amd_gpu_aql_growth_await_signals(context);
    } else if (program) {
      iree_hal_amd_gpu_program_grow_await_reuse(
          &pending->program, iree_hal_amd_gpu_aql_growth_program_available,
          pending);
    }
    return iree_ok_status();
  }

  iree_byte_span_t storage;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_capture_pool_acquire(
      context->captures, sizeof(iree_hal_amd_gpu_aql_growth_t), &storage));
  iree_hal_amd_gpu_aql_growth_t* growth = (void*)storage.data;
  *growth = (iree_hal_amd_gpu_aql_growth_t){
      .entry.kind = kind,
      .reuse_entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_REUSE_PROGRAM,
      .context = context,
      .allocation_size = storage.data_length,
  };
  iree_atomic_ref_count_init(&growth->ref_count);
  iree_hal_amd_gpu_aql_growth_append(growth, submission);
  iree_atomic_fetch_add(&context->pending_count, 1, iree_memory_order_relaxed);
  const iree_hal_amd_fabric_memory_callback_t completion = {
      .fn = iree_hal_amd_gpu_aql_growth_completed, .user_data = growth};
  iree_status_t status = iree_ok_status();
  if (program) {
    growth->command_program = submission->execute.program;
    iree_hal_amd_gpu_aql_command_program_retain(growth->command_program);
    status = iree_hal_amd_gpu_program_grow_begin(
        program, context->queue_ordinal, completion, &growth->program);
  } else {
    iree_hal_amd_fabric_memory_t* memory = NULL;
    if (kind == IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS) {
      status = iree_hal_amd_gpu_argument_storage_prepare(
          capacity, context->captures->host_allocator,
          &growth->backing.arguments);
      if (iree_status_is_ok(status)) {
        memory = &growth->backing.arguments->memory;
        growth->backing.request =
            (iree_hal_amd_fabric_memory_request_t){.options = context->memory,
                                                   .byte_length = capacity,
                                                   .alignment = capacity};
        context->arguments = growth;
      }
    } else {
      status = iree_hal_amd_gpu_signal_storage_prepare(
          context->signal_capacity, context->queue_count,
          context->captures->host_allocator, &growth->backing.signals);
      if (iree_status_is_ok(status)) {
        memory = &growth->backing.signals->memory;
        growth->backing.request = (iree_hal_amd_fabric_memory_request_t){
            .options = context->memory,
            .byte_length = (uint64_t)context->signal_capacity *
                           sizeof(iree_hal_amd_gpu_aql_signal_t),
            .alignment = 64};
        context->signals = growth;
      }
    }
    if (iree_status_is_ok(status)) {
      iree_hal_amd_gpu_context_acquire_memory(
          context->context, 1, &growth->backing.request, memory, completion,
          &growth->backing.acquire);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_gpu_aql_growth_join(growth);
  } else if (kind == IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS) {
    iree_hal_amd_gpu_aql_growth_await_signals(context);
  } else if (program) {
    iree_hal_amd_gpu_program_grow_await_reuse(
        &growth->program, iree_hal_amd_gpu_aql_growth_program_available,
        growth);
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_aql_growth_end(
    iree_hal_amd_gpu_aql_entry_t* entry, uint64_t argument_begin,
    iree_hal_amd_gpu_aql_growth_result_t* out_result) {
  iree_hal_amd_gpu_aql_growth_t* growth =
      iree_containerof(entry, iree_hal_amd_gpu_aql_growth_t, entry);
  iree_hal_amd_gpu_aql_growth_context_t* context = growth->context;
  *out_result = (iree_hal_amd_gpu_aql_growth_result_t){.first = growth->first,
                                                       .last = growth->last};
  iree_status_t status = growth->status;
  growth->completed = true;
  switch (entry->kind) {
    case IREE_HAL_AMD_GPU_AQL_ENTRY_ARGUMENTS:
      context->arguments = NULL;
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_argument_storage_initialize(
            context->context, argument_begin, growth->backing.arguments);
      }
      if (iree_status_is_ok(status)) {
        out_result->arguments = growth->backing.arguments;
      } else {
        iree_hal_amd_gpu_argument_storage_free(context->context,
                                               growth->backing.arguments);
      }
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_SIGNALS:
      context->signals = NULL;
      iree_atomic_store(&context->signal_cache->waiting, 0,
                        iree_memory_order_release);
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_gpu_aql_signal_storage_initialize(
            context->context, &context->signal_cache->pool,
            growth->backing.signals);
      }
      if (iree_status_is_ok(status)) {
        out_result->signals = growth->backing.signals;
      } else {
        iree_hal_amd_gpu_signal_storage_free(context->context,
                                             growth->backing.signals);
      }
      break;
    case IREE_HAL_AMD_GPU_AQL_ENTRY_PROGRAM:
      status = iree_hal_amd_gpu_program_grow_end(&growth->program, status);
      break;
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
  growth->first = growth->last = NULL;
  iree_hal_amd_gpu_aql_growth_join(growth);
  return status;
}

void iree_hal_amd_gpu_aql_growth_reuse_signals(
    iree_hal_amd_gpu_aql_growth_context_t* context) {
  iree_atomic_store(&context->reuse.pending, 0, iree_memory_order_release);
  if (iree_hal_amd_gpu_signal_pool_has_available(
          &context->signal_cache->pool)) {
    iree_atomic_store(&context->signal_cache->waiting, 0,
                      iree_memory_order_release);
    if (!context->signals) {
      return;
    }
    iree_hal_amd_gpu_publication_entry_t* first = context->signals->first;
    iree_hal_amd_gpu_publication_entry_t* last = context->signals->last;
    context->signals->first = context->signals->last = NULL;
    if (first) {
      iree_hal_amd_gpu_publisher_enqueue_batch(context->publisher, first, last);
    }
  }
}

void iree_hal_amd_gpu_aql_growth_reuse_program(
    iree_hal_amd_gpu_aql_entry_t* entry) {
  iree_hal_amd_gpu_aql_growth_t* growth =
      iree_containerof(entry, iree_hal_amd_gpu_aql_growth_t, reuse_entry);
  iree_atomic_store(&growth->reuse_pending, 0, iree_memory_order_release);
  if (!growth->completed) {
    // A hint may precede native completion by an arbitrary duration. Empty
    // cohorts need no notification; a later miss enrolls again explicitly.
    iree_hal_amd_gpu_program_grow_cancel_reuse(&growth->program);
    iree_hal_amd_gpu_publication_entry_t* first = growth->first;
    iree_hal_amd_gpu_publication_entry_t* last = growth->last;
    growth->first = growth->last = NULL;
    if (first) {
      iree_hal_amd_gpu_publisher_enqueue_batch(growth->context->publisher,
                                               first, last);
    }
  }
  iree_hal_amd_gpu_aql_growth_join(growth);
}
