// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TEST_UTIL_H_
#define LOOMC_TEST_UTIL_H_

#include <memory>

#include "iree/testing/status_matchers.h"
#include "loomc/iree.h"

namespace loomc::testing {

inline iree::Status ConsumeStatusForTest(loomc_status_t& status) {
  iree_status_t owned_status = iree_status_from_loomc(status);
  status = loomc_status_from_iree(
      iree_status_from_code(iree_status_code(owned_status)));
  return iree::Status(std::move(owned_status));
}

inline iree::Status ConsumeStatusForTest(loomc_status_t&& status) {
  return iree::Status(iree_status_from_loomc(status));
}

template <typename T, void (*Release)(T*)>
struct HandleDeleter {
  void operator()(T* handle) const { Release(handle); }
};

template <typename T, void (*Release)(T*)>
using HandlePtr = std::unique_ptr<T, HandleDeleter<T, Release>>;

}  // namespace loomc::testing

#define LOOMC_EXPECT_STATUS_IS(expected_code, expr)         \
  EXPECT_THAT(::loomc::testing::ConsumeStatusForTest(expr), \
              ::iree::testing::status::StatusIs(            \
                  static_cast<::iree::StatusCode>(expected_code)))

#define LOOMC_ASSERT_STATUS_IS(expected_code, expr)         \
  ASSERT_THAT(::loomc::testing::ConsumeStatusForTest(expr), \
              ::iree::testing::status::StatusIs(            \
                  static_cast<::iree::StatusCode>(expected_code)))

#define LOOMC_EXPECT_OK(expr) LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OK, expr)

#define LOOMC_ASSERT_OK(expr) LOOMC_ASSERT_STATUS_IS(LOOMC_STATUS_OK, expr)

#endif  // LOOMC_TEST_UTIL_H_
