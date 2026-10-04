// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/krea2/request.h"

#include <limits>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(Krea2RequestTest, RejectsGeometryBeforeAllocationOrTokenization) {
  for (const loom_serve_krea2_request_options_t options : {
           loom_serve_krea2_request_options_t{0, 384, 512, 0, 1},
           {384, 383, 512, 0, 1},
           {384, 384, 17, 0, 1},
           {8193, 384, 512, 0, 1},
           {16, 16, 512, 0, 1},
           {48, 48, 512, 0, 1},
           {4096, 4096, 512, 0, 1},
           {384, 384, 512, 0, std::numeric_limits<float>::infinity()},
       }) {
    loom_serve_krea2_request_t* request = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_krea2_request_create(nullptr, options, IREE_SV("prompt"),
                                        iree_allocator_null(), &request));
    EXPECT_EQ(request, nullptr);
  }
}

TEST(Krea2RequestTest, FailedAllocationReturnsNoOwnership) {
  loom_serve_krea2_request_t* request = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_krea2_request_create(
                            nullptr, {384, 384, 512, 0, 1}, IREE_SV("prompt"),
                            iree_allocator_null(), &request));
  EXPECT_EQ(request, nullptr);
  loom_serve_krea2_request_destroy(request);
}

}  // namespace
