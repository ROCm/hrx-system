// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image_request.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ImageRequestTest, OwnsDecodedPromptAndFullWidthSeed) {
  std::string body =
      R"({"prompt":"café \u732b\n\ud83e\udd16\u0000x","seed":"18446744073709551615","strength":0.25,"model":"test-image","size":"384x256","n":1,"response_format":"b64_json"})";
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      iree_make_string_view(body.data(), body.size()), IREE_SV("test-image"),
      384, 256, &request, iree_allocator_system()));
  body.assign(body.size(), 'x');
  const auto prompt = iree_string_builder_view(&request.prompt);
  const char expected[] = "café 猫\n🤖\0x";
  EXPECT_EQ(std::string(prompt.data, prompt.size),
            std::string(expected, sizeof(expected) - 1));
  EXPECT_EQ(request.seed, UINT64_MAX);
  EXPECT_EQ(request.strength, 0.25f);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, EmptyPromptDefaultsAndJsonc) {
  loom_serve_image_request_t request;
  IREE_ASSERT_OK(loom_serve_image_request_initialize(
      IREE_SV(" \n{/* image */\"prompt\":\"\",} // done\n"),
      IREE_SV("test-image"), 384, 256, &request, iree_allocator_system()));
  EXPECT_EQ(iree_string_builder_size(&request.prompt), 0u);
  EXPECT_EQ(request.seed, 0u);
  EXPECT_EQ(request.strength, 1.0f);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, RejectsMalformedOrUnsupportedRequests) {
  for (const char* body : {
           "{}",
           "[]",
           R"({"prompt":null})",
           R"({"prompt":12})",
           R"({"prompt":"x","prompt":"y"})",
           R"({"prompt":"x","unknown":0})",
           R"({"prompt":"x","seed":-1})",
           R"({"prompt":"x","seed":1.2})",
           R"({"prompt":"x","seed":"42junk"})",
           R"({"prompt":"x","strength":1e100})",
           R"({"prompt":"x","strength":"0"})",
           R"({"prompt":"x","size":"256x384"})",
           R"({"prompt":"x","model":"other"})",
           R"({"prompt":"x","n":2})",
           R"({"prompt":"x","response_format":"url"})",
           R"({"prompt":"x"} {})",
           R"({"prompt":"\ud800"})",
           "{\"prompt\":\"\xff\"}",
       }) {
    SCOPED_TRACE(body);
    loom_serve_image_request_t request;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_image_request_initialize(iree_make_cstring_view(body),
                                            IREE_SV("test-image"), 384, 256,
                                            &request, iree_allocator_system()));
    EXPECT_EQ(request.prompt.buffer, nullptr);
    loom_serve_image_request_deinitialize(&request);
  }
}

TEST(ImageRequestTest, AllocationFailurePublishesNoOwnership) {
  loom_serve_image_request_t request;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_image_request_initialize(IREE_SV("{\"prompt\":\"x\"}"),
                                          IREE_SV("test-image"), 384, 256,
                                          &request, iree_allocator_null()));
  EXPECT_EQ(request.prompt.buffer, nullptr);
  loom_serve_image_request_deinitialize(&request);
}

TEST(ImageRequestTest, SeedOverflowPublishesNoOwnership) {
  loom_serve_image_request_t request;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_image_request_initialize(
          IREE_SV("{\"prompt\":\"x\",\"seed\":18446744073709551616}"),
          IREE_SV("test-image"), 384, 256, &request, iree_allocator_system()));
  EXPECT_EQ(request.prompt.buffer, nullptr);
  loom_serve_image_request_deinitialize(&request);
}

}  // namespace
