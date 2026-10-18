// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TESTING_DEFERRED_POOL_H_
#define IREE_HAL_TESTING_DEFERRED_POOL_H_

#include <algorithm>

#include "iree/hal/pool.h"
#include "iree/testing/gtest.h"

namespace iree::hal::testing {

// Delays only the subject's dependency. All reservations, views, histories,
// and bytes still come from the real native pool. A held call borrows exactly
// the input/output lifetime promised by pool acquisition.
struct DeferredBackingPool : iree_hal_pool_t {
  // Owned real parent, released after all forwarded tokens and views return.
  iree_hal_pool_t* backing = nullptr;
  // Whether subsequent acquisitions wait for the test's explicit completion.
  bool defer = false;
  // Parent submission depth, checked for bounded inline continuation stacks.
  size_t depth = 0;
  // Maximum simultaneously active parent submissions.
  size_t maximum_depth = 0;
  // One captured request; the consumer submits its next request on completion.
  struct Pending {
    // Number of borrowed request and output array elements.
    iree_host_size_t count;
    // Borrowed parent request geometry.
    const iree_hal_pool_reservation_request_t* requests;
    // Borrowed requester ordering proof.
    const iree_async_frontier_t* frontier;
    // Borrowed output reservation array.
    iree_hal_pool_reservation_t* reservations;
    // Borrowed output history array.
    iree_hal_pool_acquire_info_t* infos;
    // Exactly one completion accepted by this capture.
    iree_hal_pool_acquire_callback_t completion;
    // Forwarded acquisition flags.
    iree_hal_pool_reserve_flags_t flags;
  } pending = {};

  static iree_status_t Create(iree_hal_pool_t* backing,
                              DeferredBackingPool** out_pool) {
    static const iree_hal_pool_vtable_t vtable = {
        .destroy =
            [](iree_hal_pool_t* base) {
              auto* self = static_cast<DeferredBackingPool*>(base);
              EXPECT_EQ(self->pending.count, 0u);
              iree_hal_pool_deinitialize(base);
              iree_hal_pool_release(self->backing);
              delete self;
            },
        .acquire_reservations =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               const iree_async_frontier_t* frontier,
               iree_hal_pool_reservation_t* reservations,
               iree_hal_pool_acquire_info_t* infos,
               iree_hal_pool_acquire_callback_t completion,
               iree_hal_pool_reserve_flags_t flags) -> iree_status_t {
          auto* self = static_cast<DeferredBackingPool*>(base);
          Pending pending = {
              .count = count,
              .requests = requests,
              .frontier = frontier,
              .reservations = reservations,
              .infos = infos,
              .completion = completion,
              .flags = flags,
          };
          if (self->defer) {
            EXPECT_EQ(self->pending.count, 0u);
            self->pending = pending;
            return iree_ok_status();
          }
          return self->Forward(pending);
        },
        .release_reservations =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_t* reservations,
               const iree_async_frontier_t* frontier) {
              iree_hal_pool_release_reservations(
                  static_cast<DeferredBackingPool*>(base)->backing, count,
                  reservations, frontier);
            },
        .materialize_reservations =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               const iree_hal_pool_reservation_t* reservations,
               iree_hal_pool_materialize_flags_t flags,
               iree_hal_buffer_t** buffers) -> iree_status_t {
          return iree_hal_pool_materialize_reservations(
              static_cast<DeferredBackingPool*>(base)->backing, count, requests,
              reservations, flags, buffers);
        },
        .query_capabilities =
            [](const iree_hal_pool_t* base,
               iree_hal_pool_capabilities_t* capabilities) {
              iree_hal_pool_query_capabilities(
                  static_cast<const DeferredBackingPool*>(base)->backing,
                  capabilities);
            },
        .query_stats =
            [](const iree_hal_pool_t* base, iree_hal_pool_stats_t* stats) {
              iree_hal_pool_query_stats(
                  static_cast<const DeferredBackingPool*>(base)->backing,
                  stats);
            },
        .trim =
            [](iree_hal_pool_t* base, iree_hal_pool_trim_flags_t flags,
               iree_device_size_t min_bytes_to_keep) {
              iree_hal_pool_trim(
                  static_cast<DeferredBackingPool*>(base)->backing, flags,
                  min_bytes_to_keep);
            },
    };
    auto* self = new DeferredBackingPool();
    iree_status_t status = iree_hal_pool_initialize(
        &vtable, backing->memory_contract, backing->notification,
        backing->wait_sources, backing->frontier_tracker,
        iree_allocator_system(), self);
    if (iree_status_is_ok(status)) {
      self->backing = backing;
      iree_hal_pool_retain(backing);
      self->epoch_query = backing->epoch_query;
      *out_pool = self;
    } else {
      delete self;
    }
    return status;
  }

  iree_status_t Forward(Pending call) {
    ++depth;
    maximum_depth = std::max(maximum_depth, depth);
    iree_status_t status = iree_hal_pool_acquire_reservations(
        backing, call.count, call.requests, call.frontier, call.reservations,
        call.infos, call.completion, call.flags);
    --depth;
    return status;
  }

  void CompleteOne() {
    ASSERT_NE(pending.count, 0u);
    Pending call = pending;
    pending = {};
    iree_status_t status = Forward(call);
    if (!iree_status_is_ok(status)) {
      call.completion.fn(call.completion.user_data, status,
                         IREE_HAL_POOL_ACQUIRE_NONE);
    }
  }

  void FailOne() {
    ASSERT_NE(pending.count, 0u);
    Pending call = pending;
    pending = {};
    call.completion.fn(call.completion.user_data,
                       iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "native allocation failed"),
                       IREE_HAL_POOL_ACQUIRE_NONE);
  }
};

}  // namespace iree::hal::testing

#endif  // IREE_HAL_TESTING_DEFERRED_POOL_H_
