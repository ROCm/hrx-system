// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/credit_memory.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/net/carrier/rdma/libverbs.h"

#define IREE_NET_RDMA_CREDIT_MEMORY_ALIGNMENT \
  iree_hardware_destructive_interference_size

typedef struct iree_net_rdma_credit_counter_t {
  // Counter written by the peer NIC and loaded by the local sender.
  iree_atomic_uint32_t value;

  // Reserved words isolate the NIC-updated counter from allocator neighbors.
  uint32_t reserved_words[15];
} iree_net_rdma_credit_counter_t;

static_assert(sizeof(iree_net_rdma_credit_counter_t) ==
                  IREE_NET_RDMA_CREDIT_MEMORY_ALIGNMENT,
              "");

struct iree_net_rdma_credit_memory_t {
  // RDMA context retained while the counter is registered.
  iree_net_rdma_context_t* context;

  // Registered memory region for counter->value.
  struct ibv_mr* memory_region;

  // Aligned storage containing the peer-writable counter.
  iree_net_rdma_credit_counter_t* counter;

  // Host allocator used for the memory allocation.
  iree_allocator_t host_allocator;
};

IREE_API_EXPORT iree_status_t iree_net_rdma_credit_memory_create(
    iree_net_rdma_context_t* context, uint32_t initial_credit_limit,
    iree_allocator_t host_allocator,
    iree_net_rdma_credit_memory_t** out_memory) {
  IREE_ASSERT_ARGUMENT(out_memory);
  *out_memory = NULL;

  iree_status_t status = iree_ok_status();
  if (!context) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "context must not be NULL");
  }

  iree_net_rdma_credit_memory_t* memory = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*memory), (void**)&memory);
  }

  if (iree_status_is_ok(status)) {
    memset(memory, 0, sizeof(*memory));
    memory->context = context;
    iree_net_rdma_context_retain(context);
    memory->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_aligned(host_allocator, sizeof(*memory->counter),
                                      IREE_NET_RDMA_CREDIT_MEMORY_ALIGNMENT,
                                      /*offset=*/0, (void**)&memory->counter);
  }

  if (iree_status_is_ok(status)) {
    iree_atomic_store(&memory->counter->value, initial_credit_limit,
                      iree_memory_order_release);
    status = iree_net_rdma_context_register_host_memory(
        context, &memory->counter->value, sizeof(memory->counter->value),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        &memory->memory_region);
  }

  if (iree_status_is_ok(status)) {
    *out_memory = memory;
  } else if (memory) {
    iree_net_rdma_credit_memory_release(memory);
  }
  return status;
}

IREE_API_EXPORT void iree_net_rdma_credit_memory_release(
    iree_net_rdma_credit_memory_t* memory) {
  if (!memory) return;

  if (memory->memory_region) {
    iree_status_t status = iree_net_rdma_context_deregister_memory(
        memory->context, memory->memory_region);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
  }
  if (memory->counter) {
    iree_allocator_free_aligned(memory->host_allocator, memory->counter);
  }
  iree_net_rdma_context_release(memory->context);

  iree_allocator_t host_allocator = memory->host_allocator;
  iree_allocator_free(host_allocator, memory);
}

IREE_API_EXPORT iree_net_rdma_remote_credit_memory_t
iree_net_rdma_credit_memory_remote(
    const iree_net_rdma_credit_memory_t* memory) {
  if (!memory || !memory->memory_region) {
    iree_net_rdma_remote_credit_memory_t empty_memory = {0};
    return empty_memory;
  }
  iree_net_rdma_remote_credit_memory_t remote_memory;
  remote_memory.address = (uint64_t)(uintptr_t)&memory->counter->value;
  remote_memory.rkey = memory->memory_region->rkey;
  remote_memory.length = sizeof(memory->counter->value);
  return remote_memory;
}

IREE_API_EXPORT uint32_t
iree_net_rdma_credit_memory_load(const iree_net_rdma_credit_memory_t* memory) {
  IREE_ASSERT_ARGUMENT(memory);
  return iree_atomic_load(
      &((iree_net_rdma_credit_memory_t*)memory)->counter->value,
      iree_memory_order_acquire);
}

IREE_API_EXPORT void iree_net_rdma_credit_memory_store(
    iree_net_rdma_credit_memory_t* memory, uint32_t credit_limit) {
  IREE_ASSERT_ARGUMENT(memory);
  iree_atomic_store(&memory->counter->value, credit_limit,
                    iree_memory_order_release);
}

IREE_API_EXPORT iree_status_t iree_net_rdma_credit_memory_store_sge(
    iree_net_rdma_credit_memory_t* memory, uint32_t credit_limit,
    struct ibv_sge* out_sge) {
  if (!out_sge) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_sge must not be NULL");
  }
  memset(out_sge, 0, sizeof(*out_sge));

  iree_status_t status = iree_ok_status();
  if (!memory || !memory->counter || !memory->memory_region) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory must be initialized");
  }
  if (iree_status_is_ok(status)) {
    iree_net_rdma_credit_memory_store(memory, credit_limit);
    out_sge->addr = (uint64_t)(uintptr_t)&memory->counter->value;
    out_sge->length = sizeof(memory->counter->value);
    out_sge->lkey = memory->memory_region->lkey;
  }
  return status;
}
