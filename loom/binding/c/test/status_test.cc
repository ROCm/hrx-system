// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/status.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loomc/iree.h"
#include "test/util.h"

namespace {

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(StatusTest, CodesMirrorIree) {
  EXPECT_EQ((int)LOOMC_STATUS_OK, (int)IREE_STATUS_OK);
  EXPECT_EQ((int)LOOMC_STATUS_CANCELLED, (int)IREE_STATUS_CANCELLED);
  EXPECT_EQ((int)LOOMC_STATUS_UNKNOWN, (int)IREE_STATUS_UNKNOWN);
  EXPECT_EQ((int)LOOMC_STATUS_INVALID_ARGUMENT,
            (int)IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ((int)LOOMC_STATUS_DEADLINE_EXCEEDED,
            (int)IREE_STATUS_DEADLINE_EXCEEDED);
  EXPECT_EQ((int)LOOMC_STATUS_NOT_FOUND, (int)IREE_STATUS_NOT_FOUND);
  EXPECT_EQ((int)LOOMC_STATUS_ALREADY_EXISTS, (int)IREE_STATUS_ALREADY_EXISTS);
  EXPECT_EQ((int)LOOMC_STATUS_PERMISSION_DENIED,
            (int)IREE_STATUS_PERMISSION_DENIED);
  EXPECT_EQ((int)LOOMC_STATUS_RESOURCE_EXHAUSTED,
            (int)IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ((int)LOOMC_STATUS_FAILED_PRECONDITION,
            (int)IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ((int)LOOMC_STATUS_ABORTED, (int)IREE_STATUS_ABORTED);
  EXPECT_EQ((int)LOOMC_STATUS_OUT_OF_RANGE, (int)IREE_STATUS_OUT_OF_RANGE);
  EXPECT_EQ((int)LOOMC_STATUS_UNIMPLEMENTED, (int)IREE_STATUS_UNIMPLEMENTED);
  EXPECT_EQ((int)LOOMC_STATUS_INTERNAL, (int)IREE_STATUS_INTERNAL);
  EXPECT_EQ((int)LOOMC_STATUS_UNAVAILABLE, (int)IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ((int)LOOMC_STATUS_DATA_LOSS, (int)IREE_STATUS_DATA_LOSS);
  EXPECT_EQ((int)LOOMC_STATUS_UNAUTHENTICATED,
            (int)IREE_STATUS_UNAUTHENTICATED);
  EXPECT_EQ((int)LOOMC_STATUS_DEFERRED, (int)IREE_STATUS_DEFERRED);
  EXPECT_EQ((int)LOOMC_STATUS_INCOMPATIBLE, (int)IREE_STATUS_INCOMPATIBLE);
  EXPECT_EQ((int)LOOMC_STATUS_CODE_MASK, (int)IREE_STATUS_CODE_MASK);
}

TEST(StatusTest, IreeStatusCastPreservesRepresentation) {
  iree_status_t iree_status =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "very specific failure");
  loomc_status_t status = loomc_status_from_iree(iree_status);

  EXPECT_EQ(loomc_status_code(status), LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(iree_status_from_loomc(status), iree_status);

#if LOOMC_STATUS_FEATURES & LOOMC_STATUS_FEATURE_ANNOTATIONS
  EXPECT_EQ(ToString(loomc_status_message(status)), "very specific failure");
#else
  EXPECT_TRUE(loomc_string_view_is_empty(loomc_status_message(status)));
#endif

  loomc_status_free(status);
}

TEST(StatusTest, FormattingUsesTwoPassPattern) {
  loomc_status_t status =
      loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED, "not wired yet");
  loomc_host_size_t required_length = 0;
  loomc_status_format(status, 0, nullptr, &required_length);
#if LOOMC_STATUS_FEATURES == 0
  EXPECT_EQ(required_length, 13u);
#else
  EXPECT_GT(required_length, 0u);
#endif
  std::string buffer(required_length + 1, '\0');
  loomc_host_size_t written_length = 0;
  EXPECT_TRUE(loomc_status_format(status, buffer.size(), buffer.data(),
                                  &written_length));
  EXPECT_EQ(written_length, required_length);
  loomc_status_free(status);
}

}  // namespace
