// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/util/benchmark_support.h"

#include <algorithm>

namespace loomc::bench {
namespace {

static iree_status_t MakeResultFailureStatus(const loomc_result_t* result,
                                             const char* operation) {
  if (result == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed without producing a result", operation);
  }

  const loomc_host_size_t diagnostic_count =
      loomc_result_diagnostic_count(result);
  if (diagnostic_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed without diagnostics", operation);
  }

  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  if (diagnostic == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed with an unreadable diagnostic",
                            operation);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION, "%s failed: %.*s: %.*s", operation,
      (int)diagnostic->code.size, diagnostic->code.data,
      (int)diagnostic->message.size, diagnostic->message.data);
}

}  // namespace

iree_allocator_t host_allocator() { return iree_allocator_system(); }

loomc_allocator_t loom_allocator() {
  return loomc_allocator_from_iree(host_allocator());
}

iree_status_t to_iree_status(loomc_status_t status) {
  return iree_status_from_loomc(status);
}

std::string FormatStatus(iree_status_t status) {
  char buffer[4096] = {0};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(buffer), buffer, &length);
  return std::string(buffer, std::min(length, sizeof(buffer) - 1));
}

iree_status_t RequireSucceededResult(const loomc_result_t* result,
                                     const char* operation) {
  if (result != nullptr && loomc_result_succeeded(result)) {
    return iree_ok_status();
  }
  return MakeResultFailureStatus(result, operation);
}

}  // namespace loomc::bench
