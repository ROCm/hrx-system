// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef struct iree_status_handle_t* iree_status_t;

iree_status_t iree_clang_tidy_status_cpp_source();

namespace iree {
class Status {
 public:
  Status() = default;
  Status(iree_status_t&& status);
  const char* ToString(iree_status_t status) { return status ? "error" : "ok"; }
};

template <typename T>
class StatusOr {
 public:
  StatusOr(iree_status_t&& status);
  StatusOr(T value);
};

namespace internal {
Status ConsumeForTest(iree_status_t& status);
Status ConsumeForTest(iree_status_t&& status);
}  // namespace internal
}  // namespace iree

namespace std {
template <typename T>
T&& move(T& value);
}  // namespace std

namespace {
class String {};
class AssertionResult {
 public:
  operator bool() const;
};

String StatusToStringAndFree(iree_status_t status);
AssertionResult AssertOkPredicate(iree::Status status);

void iree_clang_tidy_status_lifetime_consume_for_test() {
  iree_status_t consume_for_test_status = iree_clang_tidy_status_cpp_source();
  (void)::iree::internal::ConsumeForTest(consume_for_test_status);
}

void iree_clang_tidy_status_lifetime_consume_for_test_move() {
  iree_status_t consume_for_test_move_status =
      iree_clang_tidy_status_cpp_source();
  (void)::iree::internal::ConsumeForTest(
      std::move(consume_for_test_move_status));
}

void iree_clang_tidy_status_lifetime_string_to_free() {
  iree_status_t string_to_free_status = iree_clang_tidy_status_cpp_source();
  (void)StatusToStringAndFree(string_to_free_status);
}

void iree_clang_tidy_status_lifetime_gtest_condition_variable() {
  iree_status_t gtest_condition_status = iree_clang_tidy_status_cpp_source();
  if (AssertionResult assertion = AssertOkPredicate(
          ::iree::internal::ConsumeForTest(gtest_condition_status))) {
  } else {
    return;
  }
}

iree::Status iree_clang_tidy_status_lifetime_return_cpp_status_wrapper() {
  iree_status_t cpp_status_wrapper_status = iree_clang_tidy_status_cpp_source();
  return cpp_status_wrapper_status;
}

iree::Status iree_clang_tidy_status_lifetime_return_cpp_status_wrapper_move() {
  iree_status_t cpp_status_wrapper_move_status =
      iree_clang_tidy_status_cpp_source();
  return std::move(cpp_status_wrapper_move_status);
}

iree::StatusOr<int> iree_clang_tidy_status_lifetime_return_cpp_status_or() {
  iree_status_t cpp_status_or_status = iree_clang_tidy_status_cpp_source();
  return cpp_status_or_status;
}

}  // namespace
