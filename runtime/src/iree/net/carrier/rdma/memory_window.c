// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/memory_window.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "iree/net/carrier/rdma/libverbs.h"

struct iree_net_rdma_memory_window_t {
  // RDMA context retaining the protection domain.
  iree_net_rdma_context_t* context;

  // Native memory window allocated from context's protection domain.
  struct ibv_mw* memory_window;

  // Remote handle produced by the most recent successful bind post.
  iree_net_remote_handle_t remote_handle;

  // Host allocator used for this memory window allocation.
  iree_allocator_t host_allocator;
};

static iree_status_t iree_net_rdma_memory_window_status_from_errno_required(
    const char* file, uint32_t line, int error, const char* call) {
  return iree_status_from_errno(file, line, error != 0 ? error : EIO, call);
}

static iree_status_t iree_net_rdma_memory_window_translate_access_flags(
    iree_async_buffer_access_flags_t access_flags, int* out_verbs_flags) {
  *out_verbs_flags = 0;
  int verbs_flags = 0;
  if (iree_any_bit_set(access_flags,
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE)) {
    verbs_flags |= IBV_ACCESS_REMOTE_WRITE;
  }
  if (iree_any_bit_set(access_flags,
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ)) {
    verbs_flags |= IBV_ACCESS_REMOTE_READ;
  }
  if (verbs_flags == 0) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "RDMA memory window requires remote read or write access");
  }
  *out_verbs_flags = verbs_flags;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_memory_window_allocate(
    iree_net_rdma_context_t* context, iree_allocator_t host_allocator,
    iree_net_rdma_memory_window_t** out_memory_window) {
  if (!out_memory_window) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_memory_window must not be NULL");
  }
  *out_memory_window = NULL;

  iree_status_t status = iree_ok_status();
  if (!context) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "context must not be NULL");
  }

  iree_net_rdma_memory_window_t* memory_window = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*memory_window),
                                   (void**)&memory_window);
  }

  if (iree_status_is_ok(status)) {
    memset(memory_window, 0, sizeof(*memory_window));
    memory_window->context = context;
    iree_net_rdma_context_retain(context);
    memory_window->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    if (!iree_net_rdma_context_supports_memory_windows(context)) {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "RDMA device does not support Type 1 memory windows");
    }
  }

  if (iree_status_is_ok(status)) {
    errno = 0;
    memory_window->memory_window = ibv_alloc_mw(
        iree_net_rdma_context_protection_domain(context), IBV_MW_TYPE_1);
    if (!memory_window->memory_window) {
      status = iree_net_rdma_memory_window_status_from_errno_required(
          __FILE__, __LINE__, errno, "ibv_alloc_mw");
    }
  }

  if (iree_status_is_ok(status)) {
    *out_memory_window = memory_window;
  } else {
    iree_net_rdma_memory_window_release(memory_window);
  }
  return status;
}

IREE_API_EXPORT void iree_net_rdma_memory_window_release(
    iree_net_rdma_memory_window_t* memory_window) {
  if (!memory_window) return;

  if (memory_window->memory_window) {
    errno = 0;
    int result = ibv_dealloc_mw(memory_window->memory_window);
    if (result != 0) {
      iree_status_t status =
          iree_net_rdma_memory_window_status_from_errno_required(
              __FILE__, __LINE__, result > 0 ? result : errno,
              "ibv_dealloc_mw");
      iree_status_abort(status);
    }
  }
  iree_net_rdma_context_release(memory_window->context);

  iree_allocator_t host_allocator = memory_window->host_allocator;
  iree_allocator_free(host_allocator, memory_window);
}

IREE_API_EXPORT iree_status_t iree_net_rdma_memory_window_post_bind(
    iree_net_rdma_memory_window_t* memory_window,
    iree_net_rdma_queue_pair_t* queue_pair, struct ibv_mr* memory_region,
    void* base_ptr, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags, uint64_t work_request_id,
    iree_net_remote_handle_t* out_handle) {
  if (!out_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must not be NULL");
  }
  *out_handle = iree_net_remote_handle_null();
  if (!memory_window || !memory_window->memory_window) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_window must be allocated");
  }
  if (!queue_pair) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_pair must not be NULL");
  }
  if (!memory_region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory_region must not be NULL");
  }
  if (!base_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "base_ptr must not be NULL");
  }
  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "length must be non-zero");
  }

  int verbs_flags = 0;
  IREE_RETURN_IF_ERROR(iree_net_rdma_memory_window_translate_access_flags(
      access_flags, &verbs_flags));

  struct ibv_mw_bind bind;
  memset(&bind, 0, sizeof(bind));
  bind.wr_id = work_request_id;
  bind.send_flags = IBV_SEND_SIGNALED;
  bind.bind_info.mr = memory_region;
  bind.bind_info.addr = (uint64_t)(uintptr_t)base_ptr;
  bind.bind_info.length = length;
  bind.bind_info.mw_access_flags = (unsigned int)verbs_flags;

  errno = 0;
  int result = ibv_bind_mw(iree_net_rdma_queue_pair_native_qp(queue_pair),
                           memory_window->memory_window, &bind);
  if (result != 0) {
    return iree_net_rdma_memory_window_status_from_errno_required(
        __FILE__, __LINE__, result > 0 ? result : errno, "ibv_bind_mw");
  }

  memory_window->remote_handle.opaque[0] = memory_window->memory_window->rkey;
  memory_window->remote_handle.opaque[1] = (uint64_t)(uintptr_t)base_ptr;
  *out_handle = memory_window->remote_handle;
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_net_rdma_memory_window_matches_handle(
    const iree_net_rdma_memory_window_t* memory_window,
    iree_net_remote_handle_t handle) {
  if (!memory_window) return false;
  return memory_window->remote_handle.opaque[0] == handle.opaque[0] &&
         memory_window->remote_handle.opaque[1] == handle.opaque[1];
}
