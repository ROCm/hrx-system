// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TESTING_SERVICE_SCHEDULE_H_
#define LOOM_RUNTIME_TESTING_SERVICE_SCHEDULE_H_

#include <vector>

#include "loom/runtime/driver.h"

namespace loom_runtime::testing {

// Owns reusable decision storage for tests running the production default
// policy. Accepted-only tests call the service/driver progress API directly.
class ServiceSchedule {
 public:
  explicit ServiceSchedule(iree_host_size_t capacity = 1)
      : capacity_(capacity) {
    iree_host_size_t length = 0;
    IREE_CHECK_OK(loom_runtime_service_plan_size(capacity, &length));
    storage_.resize(length / sizeof(uint64_t));
    IREE_CHECK_OK(loom_runtime_service_plan_initialize(
        capacity, iree_make_byte_span(storage_.data(), length), &plan_));
  }
  ServiceSchedule(const ServiceSchedule&) = delete;
  ServiceSchedule& operator=(const ServiceSchedule&) = delete;

  // Runs one explicit selection round after accepted work has progressed.
  // Multiple sources use their actual driver roster for shared-capacity facts.
  bool Schedule(loom_runtime_driver_t* driver) {
    loom_runtime_service_schedule_budget_t budget = {1, capacity_};
    loom_runtime_service_schedule_context_t context = {
        .now_ns = iree_time_now(),
    };
    bool progress = false;
    loom_runtime_driver_schedule(driver,
                                 loom_runtime_service_default_policy(&policy_),
                                 plan_, &budget, &context, &progress);
    return progress || context.wake_time_ns <= context.now_ns;
  }

  // A complete turn for an independently owned service. Capacity-sharing
  // tests use their real roster or grant the native source explicitly. The
  // live slot remains borrowed across both phases so replacement is visible.
  iree_status_t Advance(const loom_runtime_service_t* service,
                        loom_runtime_service_flags_t flags,
                        bool* out_progress) {
    IREE_ASSERT_EQ(flags & ~LOOM_RUNTIME_SERVICE_FLAG_STOPPING, 0u);
    loom_runtime_driver_t driver;
    loom_runtime_driver_initialize(1, service, &driver);
    driver.stopping = (flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING) != 0;
    loom_runtime_driver_advance(&driver, out_progress);
    *out_progress |= Schedule(&driver);
    return driver.status;
  }

 private:
  // Per-round attempt capacity, independent of source concurrency bounds.
  iree_host_size_t capacity_;
  // Aligned backing retained across owner turns.
  std::vector<uint64_t> storage_;
  // Borrowed view of storage_.
  loom_runtime_service_plan_t* plan_ = nullptr;
  // Stable traversal position across bounded selection rounds.
  loom_runtime_service_default_policy_t policy_ = {};
};

}  // namespace loom_runtime::testing

#endif  // LOOM_RUNTIME_TESTING_SERVICE_SCHEDULE_H_
