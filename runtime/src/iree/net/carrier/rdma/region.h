// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA registration for async memory regions.
//
// This component adapts slab-backed async memory into iree_async_region_t
// values with RDMA lkey/rkey handles. Carriers can then build buffer pools and
// SGEs from the same region abstraction used by other transports without
// duplicating registration lifetime rules.

#ifndef IREE_NET_CARRIER_RDMA_REGION_H_
#define IREE_NET_CARRIER_RDMA_REGION_H_

#include "iree/async/region.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"
#include "iree/net/carrier/rdma/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers |slab| in |context| and returns an RDMA async region.
//
// The returned region retains both |context| and |slab| and must be released
// with iree_async_region_release(). |access_flags| uses the generic async
// access flags and is translated to the corresponding ibverbs MR access flags.
// The region has no owning proactor; its destroy callback deregisters through
// the retained RDMA context.
IREE_API_EXPORT iree_status_t iree_net_rdma_region_register_slab(
    iree_net_rdma_context_t* context, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_allocator_t host_allocator, iree_async_region_t** out_region);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_REGION_H_
