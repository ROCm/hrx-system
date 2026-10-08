// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/model.h"

#include <string>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/host.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, model_source, "", "Production image bootstrap source.");

namespace {

class Krea2ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(1, &host_, iree_allocator_system()));
    loom_serve_device_options_t options = {};
    options.uri = IREE_SV("amdgpu");
    IREE_ASSERT_OK(loom_serve_device_create(host_, &options, &device_,
                                            iree_allocator_system()));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_destroy(device_));
    loom_serve_host_release(host_);
  }

  // Shared task and I/O services outlive the device owner.
  loom_serve_host_t* host_ = nullptr;
  // Device owner remains reusable after failed model creation.
  loom_serve_device_t* device_ = nullptr;
};

TEST_F(Krea2ModelTest, RejectsInvalidGeometry) {
  loom_serve_image_model_options_t options = {};
  loom_serve_image_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_image_model_create(device_, &options, &model,
                                                      iree_allocator_null()));
  EXPECT_EQ(model, nullptr);
}

TEST_F(Krea2ModelTest, FailedAllocationReturnsNoOwnership) {
  loom_serve_image_model_options_t options = {};
  options.height = options.width = 384;
  options.text_tokens = 512;
  loom_serve_image_model_t* model = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_image_model_create(device_, &options, &model,
                                                      iree_allocator_null()));
  EXPECT_EQ(model, nullptr);
  IREE_EXPECT_OK(loom_serve_image_model_destroy(model));
}

TEST_F(Krea2ModelTest, MissingTokenizerReleasesPartialOwnership) {
  loom_serve_image_model_options_t options = {};
  const std::string catalog = FLAG_model_source;
  const std::string directory = catalog.substr(0, catalog.find_last_of('/'));
  options.source_directory = iree_make_cstring_view(directory.c_str());
  options.checkpoint_directory = IREE_SV("missing-krea-model-directory");
  options.height = options.width = 384;
  // Cover sole small/non-64-aligned maxima and both retained-stage maxima.
  for (const uint32_t maximum : {16u, 80u, 128u, 144u, 192u, 512u}) {
    options.text_tokens = maximum;
    loom_serve_image_model_t* model = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, loom_serve_image_model_create(
                                                     device_, &options, &model,
                                                     iree_allocator_system()));
    EXPECT_EQ(model, nullptr);
  }
}

}  // namespace
