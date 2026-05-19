// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ibverbs completion queue integration with the IREE proactor.
//
// This component owns one ibverbs completion channel and one completion queue.
// It translates completion-channel readability into bounded ibv_poll_cq batches
// while preserving the verbs notification race contract: notifications are
// requested before polling so completions arriving during drain are either
// delivered immediately or leave a future channel event.

#ifndef IREE_NET_CARRIER_RDMA_COMPLETION_QUEUE_H_
#define IREE_NET_CARRIER_RDMA_COMPLETION_QUEUE_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier/rdma/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_completion_queue_t
    iree_net_rdma_completion_queue_t;

typedef uint8_t iree_net_rdma_completion_queue_flags_t;
enum iree_net_rdma_completion_queue_flag_bits_e {
  // Defers proactor registration and CQ notification until explicit
  // activation.
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_DEFER_ACTIVATION = 1u << 0,
};

// Completion queue creation options.
typedef struct iree_net_rdma_completion_queue_options_t {
  // Number of work completions requested for the native CQ.
  int completion_capacity;

  // Completion vector passed to ibv_create_cq.
  int completion_vector;

  // Bitfield of iree_net_rdma_completion_queue_flag_bits_e values.
  iree_net_rdma_completion_queue_flags_t flags;
} iree_net_rdma_completion_queue_options_t;

// Returns default completion queue options.
static inline iree_net_rdma_completion_queue_options_t
iree_net_rdma_completion_queue_options_default(void) {
  iree_net_rdma_completion_queue_options_t options = {
      /*.completion_capacity=*/1024,
      /*.completion_vector=*/0,
  };
  return options;
}

// Handles CQ work completions and queue-level errors.
//
// |status| is OK when |completions| is non-NULL. The completion array is
// borrowed and valid only for the duration of the callback. Non-OK |status|
// transfers ownership to the callback and is reported with NULL completions.
typedef void (*iree_net_rdma_completion_queue_callback_fn_t)(
    void* user_data, iree_status_t status, const struct ibv_wc* completions,
    iree_host_size_t completion_count);

typedef struct iree_net_rdma_completion_queue_callback_t {
  // Callback function invoked from the proactor poll thread.
  iree_net_rdma_completion_queue_callback_fn_t fn;

  // Opaque user data passed to fn.
  void* user_data;
} iree_net_rdma_completion_queue_callback_t;

// Creates a completion channel and CQ.
//
// The queue retains |context| and |proactor|. The callback must not release the
// queue or unregister the event source while executing.
//
// By default the queue is activated before this returns and events may start
// flowing on the next proactor poll. Use
// IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_DEFER_ACTIVATION when the owner needs to
// finish installing higher-level receive handlers before completions can be
// delivered.
IREE_API_EXPORT iree_status_t iree_net_rdma_completion_queue_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    iree_net_rdma_completion_queue_options_t options,
    iree_net_rdma_completion_queue_callback_t callback,
    iree_allocator_t host_allocator,
    iree_net_rdma_completion_queue_t** out_queue);

// Registers the queue with its proactor, requests CQ notifications, and drains
// any completions that arrived before activation.
//
// Safe to call multiple times. Once activated, callbacks may fire from this
// call if completions were already pending.
IREE_API_EXPORT iree_status_t iree_net_rdma_completion_queue_activate(
    iree_net_rdma_completion_queue_t* queue);

// Releases the completion queue and unregisters it from the proactor.
//
// Must not be called from the queue callback.
IREE_API_EXPORT void iree_net_rdma_completion_queue_release(
    iree_net_rdma_completion_queue_t* queue);

// Returns the native completion channel.
IREE_API_EXPORT struct ibv_comp_channel*
iree_net_rdma_completion_queue_native_channel(
    const iree_net_rdma_completion_queue_t* queue);

// Returns the native completion queue.
IREE_API_EXPORT struct ibv_cq* iree_net_rdma_completion_queue_native_cq(
    const iree_net_rdma_completion_queue_t* queue);

// Drains currently queued completion-channel events and work completions.
//
// This is normally called automatically by the proactor callback. It is exposed
// for deterministic tests and explicit polling paths.
IREE_API_EXPORT iree_status_t
iree_net_rdma_completion_queue_drain(iree_net_rdma_completion_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_COMPLETION_QUEUE_H_
