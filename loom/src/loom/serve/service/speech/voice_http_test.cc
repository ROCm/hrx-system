// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/voice_http.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/transport/http/multipart.h"

namespace {

std::string Field(const std::string& name, const std::string& value,
                  const std::string& headers = "") {
  return "--x\r\nContent-Disposition: form-data; name=\"" + name + "\"\r\n" +
         headers + "\r\n" + value + "\r\n";
}

iree_status_t Upload(const std::string& body,
                     loom_serve_speech_voice_request_t* output) {
  loom_serve_http_multipart_t* multipart = nullptr;
  IREE_RETURN_IF_ERROR(loom_serve_http_multipart_parse(
      IREE_SV("multipart/form-data; boundary=x"),
      iree_make_string_view(body.data(), body.size()), 4096, 8,
      iree_allocator_system(), &multipart));
  const loom_serve_http_request_t request = {.multipart = multipart};
  const auto status = loom_serve_speech_voice_http_parse(
      &request, IREE_SV("speech"), iree_allocator_system(), output);
  loom_serve_http_multipart_free(multipart);
  return status;
}

TEST(SpeechVoiceHttpTest, CopiesOriginalRecordingAndUsesExplicitPcmGeometry) {
  const std::string pcm("\x00\x10\x00\x20", 4);
  std::string body =
      Field("id", "my voice") + Field("model", "speech") +
      Field("file", pcm, "Content-Type: audio/pcm;rate=44100;channels=2\r\n") +
      "--x--\r\n";
  loom_serve_speech_voice_request_t request;
  IREE_ASSERT_OK(Upload(body, &request));
  body.assign(body.size(), 'x');
  EXPECT_EQ(std::string(iree_string_builder_buffer(&request.id)), "my voice");
  EXPECT_EQ(request.audio.encoding, LOOM_SERVE_AUDIO_ENCODING_S16LE);
  EXPECT_EQ(request.audio.sample_rate, 44100u);
  EXPECT_EQ(request.audio.channels, 2u);
  EXPECT_EQ(request.audio.frame_count, 1u);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(request.audio.data.data),
                        request.audio.data.data_length),
            pcm);
  loom_serve_speech_voice_request_deinitialize(&request);
}

TEST(SpeechVoiceHttpTest, RejectsAmbiguousOrIncompleteFields) {
  const auto model = Field("model", "speech");
  const auto id = Field("id", "voice");
  const auto file = Field("file", "RIFF", "Content-Type: audio/wav\r\n");
  for (const auto& fields :
       {id + file, model + file, model + id, model + id + file + id,
        model + id + file + model, model + id + file + file,
        model + id + file + Field("path", "recording.wav"),
        Field("model", "other") + id + file, model + Field("id", "") + file,
        model + Field("id", std::string("x\0y", 3)) + file,
        model + Field("id", std::string("\xff", 1)) + file,
        model + id + Field("file", "")}) {
    SCOPED_TRACE(fields);
    loom_serve_speech_voice_request_t request;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Upload(fields + "--x--\r\n", &request));
    EXPECT_EQ(request.audio.data.data, nullptr);
    loom_serve_speech_voice_request_deinitialize(&request);
  }
}

iree_status_t Remove(const std::string& body, iree_string_builder_t* id) {
  const iree_string_pair_t header = {
      .key = IREE_SV("Content-Type"),
      .value = IREE_SV("application/json"),
  };
  const loom_serve_http_request_t request = {
      .headers = {1, &header},
      .body = iree_make_string_view(body.data(), body.size()),
  };
  return loom_serve_speech_voice_http_remove_parse(&request, IREE_SV("speech"),
                                                   id);
}

TEST(SpeechVoiceHttpTest, RemovalUsesExactDecodedIdAndRejectsAmbiguity) {
  iree_string_builder_t id;
  iree_string_builder_initialize(iree_allocator_system(), &id);
  IREE_ASSERT_OK(Remove(R"({"mo\u0064el":"speech","id":"my \"voice\""})", &id));
  EXPECT_EQ(std::string(iree_string_builder_buffer(&id)), "my \"voice\"");
  for (const auto* body :
       {R"({"model":"speech","id":"voice","id":"other"})",
        R"({"model":"speech","model":"other","id":"voice"})",
        R"({"model":"speech","id":3})", R"({"model":"speech","id":""})",
        R"({"model":"speech","id":"x\u0000y"})",
        R"({"model":"speech","id":"voice","extra":true})",
        R"({"model":"other","id":"voice"})", R"({"model":"speech"})",
        R"({"id":"voice"})", "{} {}", "[]"}) {
    SCOPED_TRACE(body);
    iree_string_builder_reset(&id);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Remove(body, &id));
  }
  iree_string_builder_deinitialize(&id);
}

}  // namespace
