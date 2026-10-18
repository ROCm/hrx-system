// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TESTING_DEFERRED_POOL_H_
#define IREE_HAL_TESTING_DEFERRED_POOL_H_

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <vector>

#include "iree/async/frontier.h"
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
  // Publishes a held request between the queue owner and the test thread.
  std::mutex acquire_mutex;
  // Signals that a real consumer has handed off its borrowed request.
  std::condition_variable acquire_condition;
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

  IREE_ASYNC_FIXED_FRONTIER_TYPE(ReturnFrontier, UINT8_MAX);
  struct PendingReturn {
    // Parent tokens kept live after the consumer has logically returned them.
    std::vector<iree_hal_pool_reservation_t> reservations;
    // Exact copied release history, independent of the caller's capture.
    ReturnFrontier frontier = {};
    // Distinguishes a null history from a supplied empty history.
    bool has_frontier = false;
  };
  // Serializes returned tokens with explicit test-controlled retirement.
  std::mutex return_mutex;
  // Holds subsequent returns without blocking the queue performing them.
  bool defer_returns = false;
  // Real parent ownership awaiting explicit forwarding by the test.
  std::vector<PendingReturn> pending_returns;

  static iree_status_t Create(iree_hal_pool_t* backing,
                              DeferredBackingPool** out_pool) {
    static const iree_hal_pool_vtable_t vtable = {
        .destroy =
            [](iree_hal_pool_t* base) {
              auto* self = static_cast<DeferredBackingPool*>(base);
              EXPECT_EQ(self->pending.count, 0u);
              EXPECT_TRUE(self->pending_returns.empty());
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
          {
            std::lock_guard<std::mutex> lock(self->acquire_mutex);
            if (self->defer) {
              EXPECT_EQ(self->pending.count, 0u);
              self->pending = pending;
              self->acquire_condition.notify_all();
              return iree_ok_status();
            }
          }
          return self->Forward(pending);
        },
        .release_reservations =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_t* reservations,
               const iree_async_frontier_t* frontier) {
              static_cast<DeferredBackingPool*>(base)->Return(
                  count, reservations, frontier);
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
    Pending call;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      ASSERT_NE(pending.count, 0u);
      call = pending;
      pending = {};
    }
    iree_status_t status = Forward(call);
    if (!iree_status_is_ok(status)) {
      call.completion.fn(call.completion.user_data, status,
                         IREE_HAL_POOL_ACQUIRE_NONE);
    }
  }

  void FailOne() {
    Pending call;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      ASSERT_NE(pending.count, 0u);
      call = pending;
      pending = {};
    }
    call.completion.fn(call.completion.user_data,
                       iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "native allocation failed"),
                       IREE_HAL_POOL_ACQUIRE_NONE);
  }

  void WaitForAcquisition() {
    std::unique_lock<std::mutex> lock(acquire_mutex);
    acquire_condition.wait(lock, [&] { return pending.count != 0; });
  }

  void ResumeAcquisitions() {
    Pending call;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      defer = false;
      call = pending;
      pending = {};
    }
    if (call.count) {
      iree_status_t status = Forward(call);
      if (!iree_status_is_ok(status)) {
        call.completion.fn(call.completion.user_data, status,
                           IREE_HAL_POOL_ACQUIRE_NONE);
      }
    }
  }

  void DeferReturns() {
    std::lock_guard<std::mutex> lock(return_mutex);
    defer_returns = true;
  }

  size_t PendingReturnCount() {
    std::lock_guard<std::mutex> lock(return_mutex);
    size_t count = 0;
    for (const auto& pending : pending_returns) {
      count += pending.reservations.size();
    }
    return count;
  }

  void Return(iree_host_size_t count,
              const iree_hal_pool_reservation_t* reservations,
              const iree_async_frontier_t* frontier) {
    {
      std::lock_guard<std::mutex> lock(return_mutex);
      if (defer_returns) {
        auto& pending = pending_returns.emplace_back();
        pending.reservations.assign(reservations, reservations + count);
        if (frontier) {
          pending.has_frontier = true;
          memcpy(&pending.frontier, frontier,
                 sizeof(*frontier) +
                     frontier->entry_count * sizeof(frontier->entries[0]));
        }
        return;
      }
    }
    iree_hal_pool_release_reservations(backing, count, reservations, frontier);
  }

  void CompleteReturns() {
    std::vector<PendingReturn> returns;
    {
      std::lock_guard<std::mutex> lock(return_mutex);
      defer_returns = false;
      returns.swap(pending_returns);
    }
    for (const auto& pending : returns) {
      iree_hal_pool_release_reservations(
          backing, pending.reservations.size(), pending.reservations.data(),
          pending.has_frontier
              ? iree_async_fixed_frontier_as_const_frontier(&pending.frontier)
              : nullptr);
    }
  }
};

}  // namespace iree::hal::testing

#endif  // IREE_HAL_TESTING_DEFERRED_POOL_H_
