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

TEST(Krea2RequestTest, MeasuresProductionInputLayouts) {
  iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
  IREE_ASSERT_OK(loom_serve_krea2_request_measure(384, 384, 512, sizes));
  const iree_host_size_t full[] = {73728,  2184,   143360, 143360, 560, 16,
                                   557056, 557056, 32,     4,      128};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(full); ++i) {
    EXPECT_EQ(sizes[i], full[i]) << i;
  }
  IREE_ASSERT_OK(loom_serve_krea2_request_measure(256, 384, 32, sizes));
  const iree_host_size_t small[] = {49152,  264,    20480, 20480, 80, 16,
                                    212992, 212992, 32,    4,     128};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(small); ++i) {
    EXPECT_EQ(sizes[i], small[i]) << i;
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_krea2_request_measure(0, 384, 32, sizes));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(small); ++i) {
    EXPECT_EQ(sizes[i], small[i]) << i;
  }
}

TEST(Krea2RequestTest, RejectsGeometryBeforeAllocationOrPromptAccess) {
  for (const loom_serve_krea2_request_options_t options : {
           loom_serve_krea2_request_options_t{0, 384, 512, 0, 1},
           {384, 383, 512, 0, 1},
           {384, 384, 17, 0, 1},
           {8193, 384, 512, 0, 1},
           {16, 16, 512, 0, 1},
           {48, 48, 512, 0, 1},
           {4096, 4096, 512, 0, 1},
           {384, 384, 512, 0, std::numeric_limits<float>::infinity()},
           {384, 384, 512, 0, std::numeric_limits<float>::quiet_NaN()},
       }) {
    loom_serve_krea2_request_t* request = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_krea2_request_create(nullptr, options, &request,
                                        iree_allocator_null()));
    EXPECT_EQ(request, nullptr);
  }
}

TEST(Krea2RequestTest, RejectsPromptExtentBeforeAllocationOrTokenization) {
  for (const uint32_t text_tokens : {0u, 17u, 65537u}) {
    loom_serve_krea2_prompt_t* prompt = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_krea2_prompt_create(nullptr, text_tokens, IREE_SV("prompt"),
                                       &prompt, iree_allocator_null()));
    EXPECT_EQ(prompt, nullptr);
  }
}

TEST(Krea2RequestTest, FailedPromptAllocationReturnsNoOwnership) {
  loom_serve_krea2_prompt_t* prompt = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_krea2_prompt_create(nullptr, 512, IREE_SV("prompt"), &prompt,
                                     iree_allocator_null()));
  EXPECT_EQ(prompt, nullptr);
  loom_serve_krea2_prompt_destroy(prompt);
  loom_serve_krea2_request_destroy(nullptr);
}

}  // namespace
