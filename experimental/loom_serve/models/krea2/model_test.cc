// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/krea2/model.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(Krea2ModelTest, RejectsGeometryBeforeAllocationOrDeviceAccess) {
  loom_serve_krea2_model_options_t options = {};
  loom_serve_krea2_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_krea2_model_create(&options, &model, iree_allocator_null()));
  EXPECT_EQ(model, nullptr);
}

TEST(Krea2ModelTest, FailedAllocationReturnsNoOwnership) {
  loom_serve_krea2_model_options_t options = {};
  options.height = options.width = 384;
  options.text_tokens = 512;
  loom_serve_krea2_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_krea2_model_create(&options, &model, iree_allocator_null()));
  EXPECT_EQ(model, nullptr);
  IREE_EXPECT_OK(loom_serve_krea2_model_destroy(model));
}

TEST(Krea2ModelTest, MissingTokenizerReleasesPartialOwnership) {
  loom_serve_krea2_model_options_t options = {};
  options.checkpoint_directory = IREE_SV("missing-krea-model-directory");
  options.height = options.width = 384;
  options.text_tokens = 512;
  loom_serve_krea2_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_krea2_model_create(&options, &model, iree_allocator_system()));
  EXPECT_EQ(model, nullptr);
}

}  // namespace
