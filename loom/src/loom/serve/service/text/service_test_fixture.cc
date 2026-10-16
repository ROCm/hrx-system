// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"

#include <cstdlib>
#include <string>

#include "loom/serve/testing/service_application.h"

namespace loom_serve::testing::text {

void SourceTextServiceTest::SetUp() {
  const auto application_options =
      loom_serve_test_application_options_default();
  loom_serve_test_application_t* application = nullptr;
  IREE_ASSERT_OK(loom_serve_test_application_acquire(
      &application_options, host_allocator_, &application));
  host_ = loom_serve_test_application_host(application);
  group_ = loom_serve_test_application_device_group(application);
  device_ = loom_serve_test_application_device_at(application, 0);
  const char* source_path = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
  ASSERT_NE(source_path, nullptr);
  const std::string source = source_path;
  directory_ = source.substr(0, source.find_last_of('/'));
  IREE_ASSERT_OK(document_.Load(directory_ + "/model.json"));
  tokenizer_ = directory_ + "/tokenizer.json";
  weights_ = iree::testing::MakeTempFilePath("text_fixture", ".safetensors");
  ASSERT_NO_FATAL_FAILURE(WriteWeights(2));
  if (NeedsStandaloneModel()) {
    auto options = ModelOptions();
    IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                  host_allocator_, &model_));
  }
  auto server_options = loom_serve_http_server_options_default();
  IREE_ASSERT_OK(loom_serve_http_server_allocate(
      loom_serve_host_proactor_pool(host_), &server_options, host_allocator_,
      &server_));
  auto listener_options = loom_serve_http_listener_options_default();
  listener_options.address = IREE_SV("127.0.0.1:0");
  IREE_ASSERT_OK(loom_serve_http_listener_allocate(
      server_, &listener_options, host_allocator_, &listener_));
  IREE_ASSERT_OK(http_.Initialize(server_, listener_));
}

void SourceTextServiceTest::TearDown() {
  if (http_.joinable()) {
    loom_serve_http_server_request_stop(server_);
    IREE_EXPECT_OK(http_.Join());
  }
  IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
  loom_serve_http_server_free(server_);
  IREE_EXPECT_OK(loom_serve_text_model_free(model_));
}

}  // namespace loom_serve::testing::text
