// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/frame_sender.h"

#include <string.h>

#include "iree/net/carrier.h"

// Target block size used when growing the send context pool. The block grows
// elastically, so this is an amortization unit rather than a capacity limit.
#define IREE_NET_FRAME_SEND_CONTEXT_BLOCK_SIZE (64 * 1024)

typedef struct iree_net_frame_send_context_block_t {
  // Intrusive list pointer for sender-owned context blocks.
  iree_atomic_slist_intrusive_ptr_t slist_next;
} iree_net_frame_send_context_block_t;

static inline iree_atomic_slist_entry_t*
iree_net_frame_send_context_slist_entry(
    iree_net_frame_send_context_t* context) {
  return context ? (iree_atomic_slist_entry_t*)&context->slist_next : NULL;
}

static inline iree_net_frame_send_context_t*
iree_net_frame_send_context_from_slist_entry(iree_atomic_slist_entry_t* entry) {
  return entry
             ? (iree_net_frame_send_context_t*)((uint8_t*)entry -
                                                offsetof(
                                                    iree_net_frame_send_context_t,
                                                    slist_next))
             : NULL;
}

static inline iree_atomic_slist_entry_t*
iree_net_frame_send_context_block_slist_entry(
    iree_net_frame_send_context_block_t* block) {
  return block ? (iree_atomic_slist_entry_t*)&block->slist_next : NULL;
}

static inline iree_net_frame_send_context_block_t*
iree_net_frame_send_context_block_from_slist_entry(
    iree_atomic_slist_entry_t* entry) {
  return entry
             ? (iree_net_frame_send_context_block_t*)((uint8_t*)entry -
                                                      offsetof(
                                                          iree_net_frame_send_context_block_t,
                                                          slist_next))
             : NULL;
}

static void iree_net_frame_sender_push_context(
    iree_net_frame_sender_t* sender, iree_net_frame_send_context_t* context) {
  memset(context, 0, sizeof(*context));
  iree_atomic_slist_push(&sender->context_free_list,
                         iree_net_frame_send_context_slist_entry(context));
}

static iree_net_frame_send_context_t* iree_net_frame_sender_pop_context(
    iree_net_frame_sender_t* sender) {
  return iree_net_frame_send_context_from_slist_entry(
      iree_atomic_slist_pop(&sender->context_free_list));
}

static iree_status_t iree_net_frame_sender_grow_context_pool(
    iree_net_frame_sender_t* sender,
    iree_net_frame_send_context_t** out_context) {
  *out_context = NULL;

  const iree_host_size_t context_alignment =
      iree_alignof(iree_net_frame_send_context_t);
  const iree_host_size_t context_bytes = sizeof(iree_net_frame_send_context_t);
  const iree_host_size_t block_header_size = iree_host_align(
      sizeof(iree_net_frame_send_context_block_t), context_alignment);
  const iree_host_size_t target_block_size =
      iree_max((iree_host_size_t)IREE_NET_FRAME_SEND_CONTEXT_BLOCK_SIZE,
               block_header_size + context_bytes);
  const iree_host_size_t context_count =
      (target_block_size - block_header_size) / context_bytes;

  iree_host_size_t contexts_offset = 0;
  iree_host_size_t allocation_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_net_frame_send_context_block_t), &allocation_size,
      IREE_STRUCT_FIELD_ALIGNED(context_count, iree_net_frame_send_context_t,
                                context_alignment, &contexts_offset));

  iree_net_frame_send_context_block_t* block = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(sender->context_allocator, allocation_size,
                                   (void**)&block);
  }

  if (iree_status_is_ok(status)) {
    iree_atomic_slist_push(
        &sender->context_block_list,
        iree_net_frame_send_context_block_slist_entry(block));

    uint8_t* context_base = (uint8_t*)block + contexts_offset;
    for (iree_host_size_t i = 0; i < context_count; ++i) {
      iree_net_frame_send_context_t* context =
          (iree_net_frame_send_context_t*)(context_base + i * context_bytes);
      if (i == 0) {
        *out_context = context;
      } else {
        iree_net_frame_sender_push_context(sender, context);
      }
    }
  }

  return status;
}

static void iree_net_frame_sender_release_context(
    iree_net_frame_sender_t* sender, iree_net_frame_send_context_t* context) {
  if (sender->context_pool.release) {
    sender->context_pool.release(sender->context_pool.user_data, context);
  } else {
    iree_net_frame_sender_push_context(sender, context);
  }
}

static iree_status_t iree_net_frame_sender_initialize_impl(
    iree_net_frame_sender_t* sender, iree_net_frame_send_submit_fn_t submit_fn,
    void* submit_fn_user_data, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_net_frame_sender_context_pool_t context_pool,
    bool require_context_pool, iree_allocator_t context_allocator,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(sender);
  IREE_ASSERT_ARGUMENT(submit_fn);
  IREE_ASSERT_ARGUMENT(header_pool);

  iree_status_t status = iree_ok_status();
  if (require_context_pool &&
      (!context_pool.acquire || !context_pool.release)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external frame sender context pool requires acquire and release");
  } else if ((context_pool.acquire == NULL) != (context_pool.release == NULL)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external frame sender context pool requires acquire and release");
  }

  if (iree_status_is_ok(status)) {
    memset(sender, 0, sizeof(*sender));
    sender->submit_fn = submit_fn;
    sender->submit_fn_user_data = submit_fn_user_data;
    sender->max_send_spans = max_send_spans;
    sender->header_pool = header_pool;
    sender->callback = callback;
    sender->has_batch_lease = false;
    sender->batch_used = 0;
    iree_atomic_store(&sender->sends_in_flight, 0, iree_memory_order_relaxed);
    iree_atomic_slist_initialize(&sender->context_free_list);
    iree_atomic_slist_initialize(&sender->context_block_list);
    sender->context_pool = context_pool;
    sender->context_allocator = context_allocator;
    sender->host_allocator = host_allocator;
  }

  return status;
}

iree_status_t iree_net_frame_sender_initialize(
    iree_net_frame_sender_t* sender, iree_net_frame_send_submit_fn_t submit_fn,
    void* submit_fn_user_data, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_allocator_t context_allocator, iree_allocator_t host_allocator) {
  return iree_net_frame_sender_initialize_impl(
      sender, submit_fn, submit_fn_user_data, max_send_spans, header_pool,
      callback, (iree_net_frame_sender_context_pool_t){0},
      /*require_context_pool=*/false, context_allocator, host_allocator);
}

iree_status_t iree_net_frame_sender_initialize_with_context_pool(
    iree_net_frame_sender_t* sender, iree_net_frame_send_submit_fn_t submit_fn,
    void* submit_fn_user_data, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_net_frame_sender_context_pool_t context_pool,
    iree_allocator_t host_allocator) {
  return iree_net_frame_sender_initialize_impl(
      sender, submit_fn, submit_fn_user_data, max_send_spans, header_pool,
      callback, context_pool, /*require_context_pool=*/true, host_allocator,
      host_allocator);
}

void iree_net_frame_sender_deinitialize(iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);

  // Assert no sends in flight - caller must drain completions first.
  int32_t pending =
      iree_atomic_load(&sender->sends_in_flight, iree_memory_order_acquire);
  IREE_ASSERT(pending == 0, "frame_sender deinitialize with %d sends in flight",
              pending);

  // Release any pending batch lease.
  if (sender->has_batch_lease) {
    iree_async_buffer_lease_release(&sender->batch_lease);
    sender->has_batch_lease = false;
  }

  iree_atomic_slist_discard(&sender->context_free_list);
  iree_atomic_slist_deinitialize(&sender->context_free_list);

  iree_atomic_slist_entry_t* block_entry = NULL;
  if (iree_atomic_slist_flush(&sender->context_block_list,
                              IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                              &block_entry, NULL)) {
    while (block_entry) {
      iree_atomic_slist_entry_t* next_block_entry = block_entry->next;
      iree_net_frame_send_context_block_t* block =
          iree_net_frame_send_context_block_from_slist_entry(block_entry);
      iree_allocator_free(sender->context_allocator, block);
      block_entry = next_block_entry;
    }
  }
  iree_atomic_slist_deinitialize(&sender->context_block_list);

  memset(sender, 0, sizeof(*sender));
}

// Allocates and initializes a send context.
static iree_status_t iree_net_frame_sender_allocate_context(
    iree_net_frame_sender_t* sender, uint64_t operation_user_data,
    iree_net_frame_send_context_t** out_context) {
  *out_context = NULL;

  iree_status_t status = iree_ok_status();
  iree_net_frame_send_context_t* context = NULL;
  if (sender->context_pool.acquire) {
    status =
        sender->context_pool.acquire(sender->context_pool.user_data, &context);
  } else {
    context = iree_net_frame_sender_pop_context(sender);
    if (!context) {
      status = iree_net_frame_sender_grow_context_pool(sender, &context);
    }
  }

  if (iree_status_is_ok(status) && !context) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "frame sender context pool returned NULL");
  }
  if (iree_status_is_ok(status)) {
    memset(context, 0, sizeof(*context));
    context->sender = sender;
    context->operation_user_data = operation_user_data;
    context->span_count = 0;
    *out_context = context;
  }
  return status;
}

// Submits a send context via the configured submit callback.
static iree_status_t iree_net_frame_sender_submit(
    iree_net_frame_sender_t* sender, iree_net_frame_send_context_t* context) {
  iree_async_span_list_t data =
      iree_async_span_list_make(context->spans, context->span_count);
  uint64_t send_user_data = (uint64_t)(uintptr_t)context;

  iree_atomic_fetch_add(&sender->sends_in_flight, 1, iree_memory_order_relaxed);

  iree_status_t status =
      sender->submit_fn(sender->submit_fn_user_data, data, send_user_data);
  if (!iree_status_is_ok(status)) {
    // Use release ordering to match completion path - ensures the decrement is
    // visible to other threads checking has_pending/pending_count.
    iree_atomic_fetch_sub(&sender->sends_in_flight, 1,
                          iree_memory_order_release);
  }
  return status;
}

iree_status_t iree_net_frame_sender_send(iree_net_frame_sender_t* sender,
                                         iree_const_byte_span_t header,
                                         iree_async_span_list_t payload,
                                         uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(sender);

  // NOTE: This function is thread-safe and may be called from any thread.
  // It does NOT auto-flush batched frames - if the caller is mixing send()
  // with queue()/flush() on the proactor thread and needs ordering, they
  // must call flush() explicitly before send().

  // Validate span count: 1 (header) + payload.count <= max_send_spans.
  iree_host_size_t total_spans = 1 + payload.count;
  iree_host_size_t max_spans = IREE_NET_FRAME_SENDER_MAX_SPANS;
  if (sender->max_send_spans > 0 && sender->max_send_spans < max_spans) {
    max_spans = sender->max_send_spans;
  }
  if (total_spans > max_spans) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send requires %" PRIhsz
                            " spans but max is %" PRIhsz,
                            total_spans, max_spans);
  }

  // Acquire header buffer from pool.
  iree_async_buffer_lease_t header_lease;
  iree_status_t status =
      iree_async_buffer_pool_acquire(sender->header_pool, &header_lease);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Verify header fits in pool buffer.
  if (header.data_length > header_lease.span.length) {
    iree_async_buffer_lease_release(&header_lease);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "header size %" PRIhsz
                            " exceeds pool buffer size %" PRIhsz,
                            header.data_length, header_lease.span.length);
  }

  // Copy header into pool buffer (guard against NULL for zero-length headers).
  if (header.data_length > 0) {
    memcpy(iree_async_span_ptr(header_lease.span), header.data,
           header.data_length);
  }

  // Allocate send context.
  iree_net_frame_send_context_t* context = NULL;
  status = iree_net_frame_sender_allocate_context(sender, operation_user_data,
                                                  &context);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&header_lease);
    return status;
  }

  // Transfer header lease to context.
  context->buffer_lease = header_lease;

  // Build span list: [header_span, payload_spans...].
  context->spans[0] = iree_async_span_make(
      header_lease.span.region, header_lease.span.offset, header.data_length);
  for (iree_host_size_t i = 0; i < payload.count; ++i) {
    context->spans[1 + i] = payload.values[i];
  }
  context->span_count = total_spans;

  // Submit to carrier.
  status = iree_net_frame_sender_submit(sender, context);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&context->buffer_lease);
    iree_net_frame_sender_release_context(sender, context);
    return status;
  }

  return iree_ok_status();
}

iree_status_t iree_net_frame_sender_send_copy(iree_net_frame_sender_t* sender,
                                              iree_const_byte_span_t header,
                                              iree_async_span_list_t payload,
                                              uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(sender);

  iree_host_size_t frame_length = header.data_length;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < payload.count;
       ++i) {
    iree_async_span_t span = payload.values[i];
    if (!iree_async_span_is_cpu_accessible(span)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "send_copy requires CPU-accessible payload span %" PRIhsz, i);
    } else if (!iree_host_size_checked_add(frame_length, span.length,
                                           &frame_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "copied frame size overflow");
    }
  }

  iree_async_buffer_lease_t frame_lease;
  memset(&frame_lease, 0, sizeof(frame_lease));
  bool has_frame_lease = false;
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_acquire(sender->header_pool, &frame_lease);
    has_frame_lease = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status) && frame_length > frame_lease.span.length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "copied frame size %" PRIhsz
                              " exceeds pool buffer size %" PRIhsz,
                              frame_length, frame_lease.span.length);
  }

  iree_net_frame_send_context_t* context = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_frame_sender_allocate_context(sender, operation_user_data,
                                                    &context);
  }

  if (iree_status_is_ok(status)) {
    uint8_t* target = iree_async_span_ptr(frame_lease.span);
    if (header.data_length > 0) {
      memcpy(target, header.data, header.data_length);
      target += header.data_length;
    }
    for (iree_host_size_t i = 0; i < payload.count; ++i) {
      iree_async_span_t span = payload.values[i];
      if (span.length > 0) {
        memcpy(target, iree_async_span_ptr(span), span.length);
        target += span.length;
      }
    }

    context->buffer_lease = frame_lease;
    has_frame_lease = false;
    context->spans[0] = iree_async_span_make(
        frame_lease.span.region, frame_lease.span.offset, frame_length);
    context->span_count = 1;

    status = iree_net_frame_sender_submit(sender, context);
  }

  if (!iree_status_is_ok(status)) {
    if (context) {
      iree_async_buffer_lease_release(&context->buffer_lease);
      iree_net_frame_sender_release_context(sender, context);
    }
    if (has_frame_lease) {
      iree_async_buffer_lease_release(&frame_lease);
    }
  }
  return status;
}

iree_status_t iree_net_frame_sender_queue(iree_net_frame_sender_t* sender,
                                          iree_const_byte_span_t frame) {
  IREE_ASSERT_ARGUMENT(sender);

  // Acquire batch buffer if we don't have one.
  if (!sender->has_batch_lease) {
    iree_status_t status = iree_async_buffer_pool_acquire(sender->header_pool,
                                                          &sender->batch_lease);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    sender->has_batch_lease = true;
    sender->batch_used = 0;
  }

  // Check if frame fits in remaining space.
  iree_host_size_t remaining =
      sender->batch_lease.span.length - sender->batch_used;
  if (frame.data_length > remaining) {
    // Batch buffer full - caller should flush first.
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // Copy frame to batch buffer (guard against NULL for zero-length frames).
  if (frame.data_length > 0) {
    uint8_t* dest =
        iree_async_span_ptr(sender->batch_lease.span) + sender->batch_used;
    memcpy(dest, frame.data, frame.data_length);
  }
  sender->batch_used += frame.data_length;

  return iree_ok_status();
}

iree_status_t iree_net_frame_sender_flush(iree_net_frame_sender_t* sender,
                                          uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(sender);

  // No-op if nothing queued.
  if (!sender->has_batch_lease || sender->batch_used == 0) {
    return iree_ok_status();
  }

  // Allocate send context.
  iree_net_frame_send_context_t* context = NULL;
  iree_status_t status = iree_net_frame_sender_allocate_context(
      sender, operation_user_data, &context);
  if (!iree_status_is_ok(status)) {
    // Keep batch data for retry.
    return status;
  }

  // Transfer batch lease to context.
  context->buffer_lease = sender->batch_lease;

  // Build single-span list for the used portion of batch buffer.
  context->spans[0] =
      iree_async_span_make(sender->batch_lease.span.region,
                           sender->batch_lease.span.offset, sender->batch_used);
  context->span_count = 1;

  // Submit to carrier.
  status = iree_net_frame_sender_submit(sender, context);
  if (!iree_status_is_ok(status)) {
    // Submission failed - keep batch data for retry.
    // DO NOT release batch_lease, it's still in sender.
    iree_net_frame_sender_release_context(sender, context);
    return status;
  }

  // Success - ownership transferred to context.
  sender->has_batch_lease = false;
  sender->batch_used = 0;

  return iree_ok_status();
}

iree_host_size_t iree_net_frame_sender_queued_bytes(
    const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return sender->batch_used;
}

bool iree_net_frame_sender_has_pending(const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return iree_atomic_load(&((iree_net_frame_sender_t*)sender)->sends_in_flight,
                          iree_memory_order_acquire) > 0;
}

int32_t iree_net_frame_sender_pending_count(
    const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return iree_atomic_load(&((iree_net_frame_sender_t*)sender)->sends_in_flight,
                          iree_memory_order_acquire);
}

void iree_net_frame_sender_handle_completion(
    iree_net_frame_send_context_t* context, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(context);
  iree_net_frame_sender_t* sender = context->sender;

  // Release buffer lease back to pool.
  iree_async_buffer_lease_release(&context->buffer_lease);

  // Decrement in-flight count.
  iree_atomic_fetch_sub(&sender->sends_in_flight, 1, iree_memory_order_release);

  // Fire user callback.
  if (sender->callback.fn) {
    sender->callback.fn(sender->callback.user_data,
                        context->operation_user_data, status);
  }

  // Return context to the pool.
  iree_net_frame_sender_release_context(sender, context);
}

iree_status_t iree_net_frame_sender_carrier_submit(void* user_data,
                                                   iree_async_span_list_t data,
                                                   uint64_t send_user_data) {
  iree_net_carrier_t* carrier = (iree_net_carrier_t*)user_data;
  iree_net_send_params_t params = {
      .data = data,
      .flags = IREE_NET_SEND_FLAG_NONE,
      .user_data = send_user_data,
  };
  return iree_net_carrier_send(carrier, &params);
}

void iree_net_frame_sender_dispatch_carrier_completion(
    void* callback_user_data, uint64_t operation_user_data,
    iree_status_t status, iree_host_size_t bytes_transferred,
    iree_async_buffer_lease_t* recv_lease) {
  (void)callback_user_data;
  (void)bytes_transferred;
  (void)recv_lease;
  // Frame sender always passes a non-NULL context pointer. Completions with
  // operation_user_data=0 are from non-frame_sender carrier operations (NOPs,
  // activation, etc.) and should be ignored.
  if (operation_user_data == 0) {
    iree_status_ignore(status);
    return;
  }
  iree_net_frame_send_context_t* context =
      (iree_net_frame_send_context_t*)(uintptr_t)operation_user_data;
  iree_net_frame_sender_handle_completion(context, status);
}
