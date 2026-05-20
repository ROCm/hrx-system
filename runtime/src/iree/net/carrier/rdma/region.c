// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/region.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/net/carrier/rdma/libverbs.h"

#define IREE_NET_RDMA_REGION_KNOWN_ACCESS_FLAGS                               \
  (IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE | \
   IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ |                                \
   IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE)

typedef struct iree_net_rdma_region_t {
  // Base async region; must be first for destroy callback casting.
  iree_async_region_t base;

  // RDMA context retained while the memory region is registered.
  iree_net_rdma_context_t* context;

  // Host allocator used for this region allocation.
  iree_allocator_t host_allocator;
} iree_net_rdma_region_t;

static iree_status_t iree_net_rdma_region_translate_access_flags(
    iree_async_buffer_access_flags_t access_flags, int* out_verbs_flags) {
  *out_verbs_flags = 0;
  if (access_flags == IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "RDMA region access flags must be non-zero");
  }
  if ((access_flags & ~IREE_NET_RDMA_REGION_KNOWN_ACCESS_FLAGS) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown RDMA region access flags: 0x%08X",
                            access_flags);
  }

  int verbs_flags = 0;
  if ((access_flags & (IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE |
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE)) != 0) {
    verbs_flags |= IBV_ACCESS_LOCAL_WRITE;
  }
  if ((access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ) != 0) {
    verbs_flags |= IBV_ACCESS_REMOTE_READ;
  }
  if ((access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE) != 0) {
    verbs_flags |= IBV_ACCESS_REMOTE_WRITE;
  }
  if ((access_flags & (IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ |
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE)) != 0) {
    verbs_flags |= IBV_ACCESS_MW_BIND;
  }

  *out_verbs_flags = verbs_flags;
  return iree_ok_status();
}

static void iree_net_rdma_region_destroy(iree_async_region_t* base_region) {
  iree_net_rdma_region_t* region = (iree_net_rdma_region_t*)base_region;

  if (base_region->handles.rdma.mr) {
    iree_status_t status = iree_net_rdma_context_deregister_memory(
        region->context, (struct ibv_mr*)base_region->handles.rdma.mr);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
  }
  iree_async_slab_release(base_region->slab);
  iree_net_rdma_context_release(region->context);

  iree_allocator_t host_allocator = region->host_allocator;
  iree_allocator_free(host_allocator, region);
}

IREE_API_EXPORT iree_status_t iree_net_rdma_region_register_slab(
    iree_net_rdma_context_t* context, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_allocator_t host_allocator, iree_async_region_t** out_region) {
  IREE_ASSERT_ARGUMENT(out_region);
  *out_region = NULL;

  int verbs_flags = 0;
  iree_status_t status =
      iree_net_rdma_region_translate_access_flags(access_flags, &verbs_flags);
  if (iree_status_is_ok(status) && !context) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "context must not be NULL");
  }
  if (iree_status_is_ok(status) && !slab) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "slab must not be NULL");
  }
  if (iree_status_is_ok(status) && !iree_async_slab_base_ptr(slab)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "slab base pointer must not be NULL");
  }
  if (iree_status_is_ok(status) && iree_async_slab_total_size(slab) == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "slab total size must be non-zero");
  }
  if (iree_status_is_ok(status) &&
      iree_async_slab_buffer_count(slab) > UINT32_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "slab buffer count too large for RDMA region: %" PRIhsz,
        iree_async_slab_buffer_count(slab));
  }

  iree_net_rdma_region_t* region = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*region), (void**)&region);
  }

  if (iree_status_is_ok(status)) {
    memset(region, 0, sizeof(*region));
    iree_atomic_ref_count_init(&region->base.ref_count);
    region->base.proactor = NULL;
    region->base.slab = slab;
    iree_async_slab_retain(slab);
    region->base.destroy_fn = iree_net_rdma_region_destroy;
    region->base.recycle = iree_async_buffer_recycle_callback_null();
    region->base.type = IREE_ASYNC_REGION_TYPE_RDMA;
    region->base.access_flags = access_flags;
    region->base.base_ptr = iree_async_slab_base_ptr(slab);
    region->base.length = iree_async_slab_total_size(slab);
    region->base.buffer_size = iree_async_slab_buffer_size(slab);
    region->base.buffer_count = (uint32_t)iree_async_slab_buffer_count(slab);
    region->context = context;
    iree_net_rdma_context_retain(context);
    region->host_allocator = host_allocator;
  }

  struct ibv_mr* memory_region = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_context_register_host_memory(
        context, region->base.base_ptr, region->base.length, verbs_flags,
        &memory_region);
  }

  if (iree_status_is_ok(status)) {
    region->base.handles.rdma.lkey = memory_region->lkey;
    region->base.handles.rdma.rkey = memory_region->rkey;
    region->base.handles.rdma.mr = memory_region;
    *out_region = &region->base;
  } else if (region) {
    iree_async_region_release(&region->base);
  }
  return status;
}
