// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TESTING_SERVICE_CONTROL_H_
#define LOOM_RUNTIME_TESTING_SERVICE_CONTROL_H_

#include <vector>

#include "loom/runtime/service_control.h"

namespace loom_runtime::testing {

// Caller-owned storage for one contributor, retained beyond engine retirement.
// Uses the production layout/initialization API; adds no scheduling behavior.
class ServiceControl {
 public:
  ServiceControl() { Reset(); }
  ServiceControl(const ServiceControl&) = delete;
  ServiceControl& operator=(const ServiceControl&) = delete;

  // Called only after the preceding incarnation's readers/writers have joined.
  void Reset() {
    const iree_host_size_t uses[] = {1};
    const loom_runtime_service_offer_capacity_t offers[] = {{1, uses}};
    const loom_runtime_service_control_capacity_t capacity = {1, offers, 1,
                                                              uses};
    iree_host_size_t length = 0;
    IREE_CHECK_OK(loom_runtime_service_control_size(&capacity, &length));
    storage_.resize(length / sizeof(uint64_t));
    IREE_CHECK_OK(loom_runtime_service_control_initialize(
        &capacity, ++incarnation_, iree_make_byte_span(storage_.data(), length),
        &control_));
  }

  loom_runtime_service_control_binding_t binding() const {
    return {.control = control_};
  }

 private:
  // Aligned backing storage retained through each explicit Reset boundary.
  std::vector<uint64_t> storage_;
  // Distinguishes successive occupants of this fixture's slot.
  uint64_t incarnation_ = 0;
  // Borrowed view into storage_.
  loom_runtime_service_control_t* control_ = nullptr;
};

}  // namespace loom_runtime::testing

#endif  // LOOM_RUNTIME_TESTING_SERVICE_CONTROL_H_
