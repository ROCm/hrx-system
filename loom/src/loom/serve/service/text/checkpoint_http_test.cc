// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_http.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TextCheckpointHttpTest, ValidatesRoutesBeforeModelAdmission) {
  const iree_string_pair_t session =
      iree_make_cstring_pair("x-loom-session", "conversation");
  loom_serve_http_request_t request = {
      .method = IREE_SV("POST"),
      .target = IREE_SV("/v1/checkpoints/retained"),
      .headers = {1, &session},
  };
  loom_serve_text_checkpoint_request_t control;
  IREE_ASSERT_OK(loom_serve_text_checkpoint_http_parse(&request, &control));
  EXPECT_EQ(control.action, LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN);
  EXPECT_EQ(std::string(control.name.data, control.name.size), "retained");
  EXPECT_EQ(std::string(control.session.data, control.session.size),
            "conversation");
  request.method = IREE_SV("DELETE");
  IREE_ASSERT_OK(loom_serve_text_checkpoint_http_parse(&request, &control));
  EXPECT_EQ(control.action, LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE);
  EXPECT_EQ(control.session.size, 0u);
  request.method = IREE_SV("POST");
  request.target = IREE_SV("/v1/checkpoints/retained/suspend");
  IREE_ASSERT_OK(loom_serve_text_checkpoint_http_parse(&request, &control));
  EXPECT_EQ(control.action, LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND);
  EXPECT_EQ(control.session.size, 0u);
  request.method = IREE_SV("DELETE");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_checkpoint_http_parse(&request, &control));
  request.method = IREE_SV("POST");
  for (const char* path :
       {"/v1/chat/completions", "/v1/checkpoints/", "/v1/checkpoints/bad/name",
        "/v1/checkpoints/retained?suspend"}) {
    request.target = iree_make_cstring_view(path);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_text_checkpoint_http_parse(&request, &control));
  }
  request.target = IREE_SV("/v1/checkpoints/retained");
  request.body = IREE_SV("{}");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_checkpoint_http_parse(&request, &control));
  request.body = {};
  request.headers = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_checkpoint_http_parse(&request, &control));
}

}  // namespace
