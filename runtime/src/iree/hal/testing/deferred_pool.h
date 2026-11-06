// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_TESTING_DEFERRED_POOL_H_
#define IREE_HAL_TESTING_DEFERRED_POOL_H_

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

#include "iree/async/frontier.h"
#include "iree/hal/pool.h"
#include "iree/hal/slab_pool.h"
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
  // Whether the dependency exposes its optional physical-capacity fact.
  bool report_physical_capacity = true;
  // Owner-installed observation boundary for completing a held dependency
  // between native statistics and the caller's next publication pass.
  std::function<void()> statistics_observer;
  // Active parent calls, including asynchronous callbacks before submit
  // returns.
  std::atomic<size_t> depth{0};
  // Peak active calls; measures inline stack depth for single-threaded parents.
  std::atomic<size_t> maximum_depth{0};
  // Publishes a held request between the queue owner and the test thread.
  std::mutex acquire_mutex;
  // Signals that a real consumer has handed off its borrowed request.
  std::condition_variable acquire_condition;
  // Captured requests preserve their independently accepted completions.
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
    // Admission moved into this dependency capture before native submission.
    iree_hal_memory_credit_t credit = {};
    // Forwarded acquisition flags.
    iree_hal_pool_reserve_flags_t flags;
  };
  // Requests held by this dependency until explicitly completed by the test.
  std::deque<Pending> pending;

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
  // Publishes the handoff of real reservations to the deferred-return owner.
  std::condition_variable return_condition;
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
              EXPECT_TRUE(self->pending.empty());
              EXPECT_TRUE(self->pending_returns.empty());
              iree_hal_pool_deinitialize(base);
              iree_hal_pool_release(self->backing);
              delete self;
            },
        .acquire_reservations =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               const iree_async_frontier_t* frontier,
               iree_hal_pool_acquire_callback_t completion,
               iree_hal_pool_reserve_flags_t flags,
               iree_hal_pool_reservation_t* reservations,
               iree_hal_pool_acquire_info_t* infos) -> iree_status_t {
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
              self->pending.push_back(pending);
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
              const auto* self = static_cast<const DeferredBackingPool*>(base);
              iree_hal_pool_query_capabilities(self->backing, capabilities);
              if (!self->report_physical_capacity) {
                capabilities->physical_capacity = 0;
              }
            },
        .query_stats =
            [](const iree_hal_pool_t* base, iree_hal_pool_stats_t* stats) {
              const auto* self = static_cast<const DeferredBackingPool*>(base);
              iree_hal_pool_query_stats(self->backing, stats);
              // A copied callback can safely detach itself during observation.
              const auto observer = self->statistics_observer;
              if (observer) {
                observer();
              }
            },
        .trim =
            [](iree_hal_pool_t* base, iree_hal_pool_trim_flags_t flags,
               iree_device_size_t min_bytes_to_keep) {
              iree_hal_pool_trim(
                  static_cast<DeferredBackingPool*>(base)->backing, flags,
                  min_bytes_to_keep);
            },
        .query_allocation_cost =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               uint64_t* out_length) {
              return iree_hal_slab_pool_query_allocation_cost(
                  static_cast<DeferredBackingPool*>(base)->backing, count,
                  requests, out_length);
            },
        .acquire_with_credit =
            [](iree_hal_pool_t* base, iree_host_size_t count,
               const iree_hal_pool_reservation_request_t* requests,
               iree_hal_memory_credit_t* credit,
               iree_hal_pool_acquire_callback_t completion,
               iree_hal_pool_reservation_t* reservations,
               iree_hal_pool_acquire_info_t* infos) {
              auto* self = static_cast<DeferredBackingPool*>(base);
              Pending call = {.count = count,
                              .requests = requests,
                              .reservations = reservations,
                              .infos = infos,
                              .completion = completion,
                              .credit = *credit};
              *credit = {};
              {
                std::lock_guard<std::mutex> lock(self->acquire_mutex);
                if (self->defer) {
                  self->pending.push_back(call);
                  self->acquire_condition.notify_all();
                  return iree_ok_status();
                }
              }
              iree_status_t status = self->Forward(call);
              *credit = call.credit;
              return status;
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

  iree_status_t Forward(Pending& call) {
    const size_t current_depth =
        depth.fetch_add(1, std::memory_order_relaxed) + 1;
    size_t previous_maximum = maximum_depth.load(std::memory_order_relaxed);
    while (previous_maximum < current_depth &&
           !maximum_depth.compare_exchange_weak(previous_maximum, current_depth,
                                                std::memory_order_relaxed)) {
    }
    iree_status_t status =
        call.credit.budget
            ? iree_hal_slab_pool_acquire_with_credit(
                  backing, call.count, call.requests, &call.credit,
                  call.completion, call.reservations, call.infos)
            : iree_hal_pool_acquire_reservations(
                  backing, call.count, call.requests, call.frontier,
                  call.completion, call.flags, call.reservations, call.infos);
    depth.fetch_sub(1, std::memory_order_relaxed);
    return status;
  }

  void CompleteOne() {
    Pending call;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      ASSERT_FALSE(pending.empty());
      call = pending.front();
      pending.pop_front();
    }
    iree_status_t status = Forward(call);
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_credit_return(&call.credit);
      call.completion.fn(call.completion.user_data, status,
                         IREE_HAL_POOL_ACQUIRE_NONE);
    }
  }

  void FailOne() {
    Pending call;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      ASSERT_FALSE(pending.empty());
      call = pending.front();
      pending.pop_front();
    }
    iree_hal_memory_credit_return(&call.credit);
    call.completion.fn(call.completion.user_data,
                       iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                        "native allocation failed"),
                       IREE_HAL_POOL_ACQUIRE_NONE);
  }

  void WaitForAcquisition() {
    std::unique_lock<std::mutex> lock(acquire_mutex);
    acquire_condition.wait(lock, [&] { return !pending.empty(); });
  }

  size_t PendingRequestCount() {
    std::lock_guard<std::mutex> lock(acquire_mutex);
    size_t count = 0;
    for (const auto& call : pending) {
      count += call.count;
    }
    return count;
  }

  iree_hal_pool_reservation_request_t FirstRequest() {
    std::lock_guard<std::mutex> lock(acquire_mutex);
    return pending.front().requests[0];
  }

  Pending TakeFirstAcquisition() {
    std::lock_guard<std::mutex> lock(acquire_mutex);
    Pending call = pending.front();
    pending.pop_front();
    return call;
  }

  void ResumeAcquisitions() {
    std::deque<Pending> calls;
    {
      std::lock_guard<std::mutex> lock(acquire_mutex);
      defer = false;
      calls.swap(pending);
    }
    for (auto& call : calls) {
      iree_status_t status = Forward(call);
      if (!iree_status_is_ok(status)) {
        iree_hal_memory_credit_return(&call.credit);
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

  void WaitForReturn() {
    std::unique_lock<std::mutex> lock(return_mutex);
    return_condition.wait(lock, [&] { return !pending_returns.empty(); });
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
        return_condition.notify_all();
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
