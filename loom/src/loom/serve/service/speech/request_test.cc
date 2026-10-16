// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/request.h"

#include <limits>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SpeechRequestTest, RetainsNativeInputWithoutJsonEscaping) {
  std::string input = "A quote: \"日\"\n";
  std::string voice = "alba";
  std::string instructions = "softly";
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize_text(
      iree_make_string_view(input.data(), input.size()),
      iree_make_string_view(voice.data(), voice.size()),
      iree_make_string_view(instructions.data(), instructions.size()),
      UINT64_MAX, 0.25f, iree_allocator_system(), &request));
  input.assign(input.size(), 'x');
  voice.assign(voice.size(), 'x');
  instructions.assign(instructions.size(), 'x');
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.input)),
            "A quote: \"日\"\n");
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.voice)), "alba");
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.instructions)),
            "softly");
  EXPECT_EQ(request.speed, 0.25f);
  EXPECT_EQ(request.seed, UINT64_MAX);
  loom_serve_speech_request_deinitialize(&request);
}

TEST(SpeechRequestTest, RejectsInvalidNativeStringsAndSpeed) {
  const iree_string_view_t invalid_strings[][3] = {
      {IREE_SVL(""), IREE_SVL("alba"), IREE_SVL("")},
      {IREE_SVL("hi"), IREE_SVL(""), IREE_SVL("")},
      {IREE_SVL("hi"), IREE_SVL("a\0b"), IREE_SVL("")},
      {IREE_SVL("\xff"), IREE_SVL("alba"), IREE_SVL("")},
      {IREE_SVL("hi"), IREE_SVL("\xff"), IREE_SVL("")},
      {IREE_SVL("hi"), IREE_SVL("alba"), IREE_SVL("\xff")},
  };
  for (const auto& strings : invalid_strings) {
    loom_serve_speech_request_t request;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_speech_request_initialize_text(
                              strings[0], strings[1], strings[2], 0, 1.0f,
                              iree_allocator_system(), &request));
    loom_serve_speech_request_deinitialize(&request);
  }
  for (float speed : {0.0f, 4.1f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    loom_serve_speech_request_t request;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_speech_request_initialize_text(
                              IREE_SV("hi"), IREE_SV("alba"), {}, 0, speed,
                              iree_allocator_system(), &request));
    loom_serve_speech_request_deinitialize(&request);
  }
}

TEST(SpeechRequestTest, DecodesRetainedStringsAndCustomVoice) {
  std::string json = R"({"model":"sp\u0065ech","input":"Good\n\u65e5",
      "voice":{"id":"al\u0062a"},"instructions":"softly","speed":0.5,
      "seed":"18446744073709551615","response_format":"p\u0063m",
      "stream_format":"audio"})";
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize(
      iree_make_string_view(json.data(), json.size()), IREE_SV("speech"),
      iree_allocator_system(), &request));
  json.assign(json.size(), 'x');
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.input)),
            "Good\n日");
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.voice)), "alba");
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.instructions)),
            "softly");
  EXPECT_EQ(request.speed, 0.5f);
  EXPECT_EQ(request.seed, UINT64_MAX);
  loom_serve_speech_request_deinitialize(&request);
}

TEST(SpeechRequestTest, RejectsAmbiguousAndUnsupportedProtocol) {
  const char* cases[] = {
      R"({"input":"hello","voice":"alba","response_format":"pcm"})",
      R"({"model":"speech","input":"","voice":"alba","response_format":"pcm"})",
      R"({"model":"speech","input":"hi","voice":{"name":"alba"},"response_format":"pcm"})",
      R"({"model":"speech","input":"hi","voice":"a\u0000b","response_format":"pcm"})",
      R"({"model":"speech","input":"hi","voice":"alba","response_format":"pcm","speed":0})",
      R"({"model":"speech","input":"hi","voice":"alba","response_format":"pcm","stream_format":"sse"})",
      R"({"model":"speech","input":"hi","voice":"alba","response_format":"mp3"})",
      R"({"model":"speech","input":"hi","voice":"alba"})",
      R"({"model":"speech","input":"hi","voice":"alba","response_format":"pcm","v\u006fice":"other"})",
      R"({"model":"speech","input":"\ud800","voice":"alba","response_format":"pcm"})",
      R"({"model":"speech","input":"hi","voice":"alba","response_format":"pcm"} {})",
  };
  for (const char* json : cases) {
    SCOPED_TRACE(json);
    loom_serve_speech_request_t request;
    iree_status_t status = loom_serve_speech_request_initialize(
        iree_make_cstring_view(json), IREE_SV("speech"),
        iree_allocator_system(), &request);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
    iree_status_free(status);
    loom_serve_speech_request_deinitialize(&request);
  }
}

TEST(SpeechRequestTest, EscapedFieldNamesAndDefaults) {
  loom_serve_speech_request_t request;
  IREE_ASSERT_OK(loom_serve_speech_request_initialize(
      IREE_SV(
          R"({"mo\u0064el":"speech","input":"hi","voice":"alba","response_format":"pcm"})"),
      IREE_SV("speech"), iree_allocator_system(), &request));
  EXPECT_EQ(request.speed, 1.0f);
  EXPECT_EQ(request.seed, 0u);
  EXPECT_EQ(iree_string_builder_size(&request.instructions), 0u);
  loom_serve_speech_request_deinitialize(&request);
}

}  // namespace
